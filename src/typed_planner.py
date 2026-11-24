"""
src/typed_planner.py — Unified type-directed planner over TypedIR subgoals.

Searches strictly via formal 4/5-axis signatures, live k-hop cross-domain bridges,
linear resource lifecycle tracking, recursive structured `loop`/`branch` blocks,
and optional SLM choice scoring — zero raw-English heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import heapq
import math
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    from .bridge_graph import (
        CanonicalPortType,
        LiveBridgeGraph,
        LiveBridgePath,
        _apply_edge,
        join_port_types,
        propagate_polymorphic_signature,
        signatures_unify,
    )
    from .config import settings
    from .schema import TOP_CARRIERS, align_node_params_to_ports, derive_value_class, get_attr
    from .subgoal_retrieval import CandidateMatch
    from .typed_ir import IRNode, TypedIR, infer_loop_var
except (ImportError, ValueError):
    from bridge_graph import (
        CanonicalPortType,
        LiveBridgeGraph,
        LiveBridgePath,
        _apply_edge,
        join_port_types,
        propagate_polymorphic_signature,
        signatures_unify,
    )
    from config import settings
    from schema import TOP_CARRIERS, align_node_params_to_ports, derive_value_class, get_attr
    from subgoal_retrieval import CandidateMatch
    from typed_ir import IRNode, TypedIR, infer_loop_var


@dataclass(frozen=True)
class PlannedStep:
    ir_node_id: Optional[str]
    cell: Any
    cell_id: str
    is_live_bridge: bool
    input_wiring: Dict[str, str]
    output_var: str
    output_sig: CanonicalPortType
    step_cost: float
    kind: str = "call"
    condition: Optional[str] = None
    body_steps: Tuple["PlannedStep", ...] = ()
    else_steps: Tuple["PlannedStep", ...] = ()
    output_vars: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidatePlan:
    steps: Tuple[PlannedStep, ...]
    total_cost: float
    posterior_prob: float = 0.0
    open_resources: Tuple[str, ...] = ()
    exit_released_resources: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanResult:
    ok: bool
    chosen_plan: Optional[CandidatePlan]
    candidates: Tuple[CandidatePlan, ...] = ()
    refusal_kind: Optional[str] = None
    reason: Optional[str] = None
    missing_ports: Tuple[str, ...] = ()
    normalized_entropy: float = 0.0
    margin: float = 1.0


def _required_external_asset_ports(cell: Any) -> List[Tuple[str, Any]]:
    raw_ins = get_attr(cell, "inputs", {}) or {}
    if not isinstance(raw_ins, dict):
        return []
    out: List[Tuple[str, Any]] = []
    for pname, pval in raw_ins.items():
        if not bool(get_attr(pval, "required", True)):
            continue
        if get_attr(pval, "default_value", None) is not None:
            continue
        if derive_value_class(pval) == "external_asset":
            out.append((str(pname), pval))
    return out


def _required_data_ports(cell: Any) -> List[Tuple[str, Any]]:
    raw_ins = get_attr(cell, "inputs", {}) or {}
    if not isinstance(raw_ins, dict):
        return []
    out: List[Tuple[str, Any]] = []
    for pname, pval in raw_ins.items():
        if not bool(get_attr(pval, "required", True)):
            continue
        if get_attr(pval, "default_value", None) is not None:
            continue
        if derive_value_class(pval) == "data":
            out.append((str(pname), pval))
    return out


def _primary_output_port(cell: Any) -> Tuple[Optional[str], Any]:
    raw_outs = get_attr(cell, "outputs", {}) or {}
    if not isinstance(raw_outs, dict) or not raw_outs:
        return None, None
    prim = get_attr(cell, "primary_out", None)
    if prim and prim in raw_outs:
        return str(prim), raw_outs[prim]
    first_k = next(iter(raw_outs))
    return str(first_k), raw_outs[first_k]


def _check_external_assets(
    node: IRNode,
    cell: Any,
    grounded_assets: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    cid = str(get_attr(cell, "cell_id", "") or "").strip()
    req_assets = _required_external_asset_ports(cell)
    if not req_assets:
        return True, []

    grounded_node_assets = {
        pk: pv
        for pk, pv in node.params.items()
        if pv.value_class == "external_asset" and pv.provenance == "prompt" and pv.value is not None
    }
    unused_node_keys = list(grounded_node_assets.keys())
    unused_global_keys = [
        inp for inp in node.inputs
        if inp in grounded_assets and grounded_assets[inp] is not None
    ]

    missing: List[str] = []
    for pname, _ in req_assets:
        direct = node.params.get(pname)
        if direct is not None:
            if direct.value_class == "external_asset" and direct.provenance == "prompt" and direct.value is not None:
                if pname in unused_node_keys:
                    unused_node_keys.remove(pname)
                continue
            missing.append(f"{cid}.{pname}")
            continue
        if len(unused_node_keys) == 1 and len(req_assets) == 1:
            unused_node_keys.pop(0)
            continue
        if unused_global_keys:
            unused_global_keys.pop(0)
            continue
        missing.append(f"{cid}.{pname}")

    return len(missing) == 0, missing


def _check_formal_param_names(node: IRNode, cell: Any) -> Tuple[bool, Optional[str]]:
    """Verify explicitly supplied formal parameter keys exist on candidate `cell`."""
    if not node.params:
        return True, None
    cid = str(get_attr(cell, "cell_id", "") or "").strip()
    raw_ins = get_attr(cell, "inputs", {}) or {}
    req_assets = _required_external_asset_ports(cell)
    for pk, pv in node.params.items():
        if isinstance(raw_ins, dict) and pk in raw_ins:
            continue
        if pv.value_class == "external_asset" and len(req_assets) == 1:
            continue
        return False, f"{node.id}:{cid}:unknown_param({pk})"
    return True, None


def _has_explicit_literal_param(node: IRNode, pname: str) -> bool:
    """Return True if caller supplied an explicit literal/enum value for `pname`."""
    p_entry = node.params.get(pname)
    return bool(
        p_entry is not None
        and p_entry.value is not None
        and p_entry.provenance in ("prompt", "default", "estimated")
        and p_entry.value_class in ("literal", "enum")
    )


def _optional_data_ports(cell: Any) -> List[Tuple[str, Any]]:
    """Return optional data input ports eligible for explicit variable wiring."""
    raw_ins = get_attr(cell, "inputs", {}) or {}
    if not isinstance(raw_ins, dict):
        return []
    out: List[Tuple[str, Any]] = []
    for pname, pval in raw_ins.items():
        if bool(get_attr(pval, "required", True)):
            continue
        if derive_value_class(pval) == "data":
            out.append((str(pname), pval))
    return out


def _apply_in_place_updates(
    cell: Any,
    dom: str,
    raw_ins: Any,
    raw_outs: Any,
    prim_name: Optional[str],
    out_sigs_by_port: Dict[str, CanonicalPortType],
    final_wiring: Dict[str, str],
    nxt_env: Dict[str, CanonicalPortType],
    poset: Any,
) -> None:
    """Propagate post-mutation typestate/layout back to aliased mutable input variables."""
    if str(get_attr(cell, "mutation_type", "") or "").strip().lower() != "in_place":
        return
    if not isinstance(raw_outs, dict) or not raw_outs:
        return

    matched_positional = False
    for ip_name, wired_v in final_wiring.items():
        ip_obj = raw_ins.get(ip_name) if isinstance(raw_ins, dict) else None
        if ip_obj is None:
            continue
        ip_sig = CanonicalPortType.from_port(ip_obj, dom)
        if ip_sig.resource_mode == "release":
            continue
        cur_sig = nxt_env.get(wired_v)
        if cur_sig is None:
            continue

        m_out_sig: Optional[CanonicalPortType] = None
        for cand_name in (ip_name, f"out_{ip_name}", f"{ip_name}_out", f"modified_{ip_name}"):
            if cand_name in out_sigs_by_port:
                m_out_sig = out_sigs_by_port[cand_name]
                break
        if m_out_sig is None and not matched_positional and prim_name and prim_name in out_sigs_by_port:
            m_out_sig = out_sigs_by_port[prim_name]
            matched_positional = True
        if m_out_sig is None:
            continue

        out_c = (m_out_sig.carrier or "any").strip()
        cur_c = (cur_sig.carrier or "any").strip()
        if (
            out_c.lower() not in TOP_CARRIERS
            and cur_c.lower() not in TOP_CARRIERS
            and not poset.is_subtype(out_c, cur_c)
            and not poset.is_subtype(cur_c, out_c)
        ):
            continue

        upd_sig = propagate_polymorphic_signature(cur_sig, m_out_sig, cell)
        eff_carrier = cur_sig.carrier if upd_sig.carrier.lower() in TOP_CARRIERS else upd_sig.carrier
        eff_ts = cur_sig.typestate if (upd_sig.typestate == "any" and cur_sig.typestate != "any") else upd_sig.typestate
        eff_dom = cur_sig.domain if eff_ts == cur_sig.typestate else upd_sig.domain
        nxt_env[wired_v] = replace(
            upd_sig,
            carrier=eff_carrier,
            typestate=eff_ts,
            domain=eff_dom,
            resource_mode=cur_sig.resource_mode,
        )


class TypedIRPlanner:
    """Best-first type-directed planner with live cross-domain bridges and structured control flow."""

    def __init__(
        self,
        bridge_graph: LiveBridgeGraph,
        poset: Any,
        choice_scorer_fn: Optional[Callable[[str, List[str]], List[float]]] = None,
        max_bridge_hops: Optional[int] = None,
        ambiguity_temperature: Optional[float] = None,
        entropy_threshold: Optional[float] = None,
        margin_threshold: Optional[float] = None,
        allow_lossy_bridges: bool = True,
    ) -> None:
        self.bridge_graph = bridge_graph
        self.poset = poset
        self.choice_scorer_fn = choice_scorer_fn
        self.max_bridge_hops = max(0, int(settings.max_bridge_hops if max_bridge_hops is None else max_bridge_hops))
        self.ambiguity_temperature = float(
            settings.ambiguity_temperature if ambiguity_temperature is None else ambiguity_temperature
        )
        self.entropy_threshold = float(
            settings.ambiguity_entropy_threshold if entropy_threshold is None else entropy_threshold
        )
        self.margin_threshold = float(
            settings.ambiguity_margin_threshold if margin_threshold is None else margin_threshold
        )
        self.allow_lossy_bridges = bool(allow_lossy_bridges)

    def _wire_data_inputs(
        self,
        node: IRNode,
        cell: Any,
        source_mode: str,
        env_map: Dict[str, CanonicalPortType],
        released_res: Set[str],
        grounded_assets: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, str], List[Tuple[str, LiveBridgePath]], Optional[str]]:
        dom = str(get_attr(cell, "domain_name", None) or "").strip().lower()
        cid = str(get_attr(cell, "cell_id", "") or "").strip()
        explicit_non_asset_refs = [inp for inp in node.inputs if inp not in grounded_assets]
        for ref in explicit_non_asset_refs:
            if ref in released_res:
                return False, {}, [], f"{node.id}:{ref}:use_after_release"

        if source_mode == "formal_syntax" and explicit_non_asset_refs:
            missing_refs = [ref for ref in explicit_non_asset_refs if ref not in env_map]
            if missing_refs:
                return False, {}, [], f"{node.id}:{cid}:undefined_input({','.join(missing_refs)})"

        avail_explicit = [ref for ref in explicit_non_asset_refs if ref in env_map and ref not in released_res]
        req_data = [
            (pname, pval)
            for pname, pval in _required_data_ports(cell)
            if not _has_explicit_literal_param(node, pname)
        ]
        opt_data = [
            (pname, pval)
            for pname, pval in _optional_data_ports(cell)
            if not _has_explicit_literal_param(node, pname)
        ]
        if not req_data and not avail_explicit:
            return True, {}, [], None
        if not req_data and not opt_data:
            if source_mode == "formal_syntax" and avail_explicit:
                return False, {}, [], f"{node.id}:{cid}:unused_explicit_input({','.join(avail_explicit)})"
            return True, {}, [], None

        if source_mode == "formal_syntax" and explicit_non_asset_refs:
            cand_pool = avail_explicit
        else:
            cand_pool = list(avail_explicit)
            for vname in reversed(list(env_map.keys())):
                if vname not in released_res and vname not in cand_pool:
                    cand_pool.append(vname)

        if req_data and not cand_pool:
            return False, {}, [], f"{node.id}:{get_attr(cell, 'cell_id', '')}:no_upstream_carrier"

        wiring: Dict[str, str] = {}
        bridges: List[Tuple[str, LiveBridgePath]] = []
        used_vars: Set[str] = set()

        for pname, pval in req_data:
            dst_sig = CanonicalPortType.from_port(pval, dom)
            ordered_vars = [v for v in cand_pool if v not in used_vars] + [v for v in cand_pool if v in used_vars]

            matched_var: Optional[str] = None
            for vname in ordered_vars:
                if signatures_unify(env_map[vname], dst_sig, self.poset):
                    matched_var = vname
                    break

            if matched_var is not None:
                wiring[pname] = matched_var
                used_vars.add(matched_var)
                continue

            best_var: Optional[str] = None
            best_bridge: Optional[LiveBridgePath] = None
            for vname in ordered_vars:
                found = self.bridge_graph.find_bridges(
                    env_map[vname],
                    dst_sig,
                    max_hops=self.max_bridge_hops,
                    max_results=1,
                    allow_lossy=self.allow_lossy_bridges,
                )
                if found:
                    cand_bp = found[0]
                    if best_bridge is None or (cand_bp.lossy, cand_bp.total_cost, cand_bp.hop_count) < (
                        best_bridge.lossy, best_bridge.total_cost, best_bridge.hop_count
                    ):
                        best_var = vname
                        best_bridge = cand_bp

            if best_var is not None and best_bridge is not None:
                used_vars.add(best_var)
                wiring[pname] = best_var
                bridges.append((pname, best_bridge))
                continue

            cid = str(get_attr(cell, "cell_id", "") or "").strip()
            return False, {}, [], f"{node.id}:{cid}.{pname}"

        unused_explicit = [v for v in avail_explicit if v not in used_vars]
        if unused_explicit and opt_data:
            for pname, pval in opt_data:
                if pname in wiring or not unused_explicit:
                    continue
                dst_sig = CanonicalPortType.from_port(pval, dom)
                matched_opt: Optional[str] = None
                for vname in unused_explicit:
                    if signatures_unify(env_map[vname], dst_sig, self.poset):
                        matched_opt = vname
                        break
                if matched_opt is not None:
                    wiring[pname] = matched_opt
                    used_vars.add(matched_opt)
                    unused_explicit.remove(matched_opt)
                    continue

                best_var_opt: Optional[str] = None
                best_bridge_opt: Optional[LiveBridgePath] = None
                for vname in unused_explicit:
                    found = self.bridge_graph.find_bridges(
                        env_map[vname],
                        dst_sig,
                        max_hops=self.max_bridge_hops,
                        max_results=1,
                        allow_lossy=self.allow_lossy_bridges,
                    )
                    if found:
                        cand_bp = found[0]
                        if best_bridge_opt is None or (cand_bp.lossy, cand_bp.total_cost, cand_bp.hop_count) < (
                            best_bridge_opt.lossy, best_bridge_opt.total_cost, best_bridge_opt.hop_count
                        ):
                            best_var_opt = vname
                            best_bridge_opt = cand_bp
                if best_var_opt is not None and best_bridge_opt is not None:
                    wiring[pname] = best_var_opt
                    used_vars.add(best_var_opt)
                    unused_explicit.remove(best_var_opt)
                    bridges.append((pname, best_bridge_opt))

        if source_mode == "formal_syntax" and unused_explicit:
            return False, {}, [], f"{node.id}:{cid}:unused_explicit_input({','.join(unused_explicit)})"

        return True, wiring, bridges, None

    def _plan_block(
        self,
        nodes: List[IRNode],
        source_mode: str,
        candidates_by_node: Dict[str, List[CandidateMatch]],
        grounded_assets: Dict[str, Any],
        deliverable: Optional[str],
        initial_env: Dict[str, CanonicalPortType],
        initial_open: Tuple[str, ...],
        initial_released: Tuple[str, ...],
        *,
        allow_deliverable_open: bool,
        require_all_closed: bool = True,
        max_plans: int = 8,
        max_expansions: int = 400,
    ) -> PlanResult:
        n_nodes = len(nodes)
        missing_asset_ports: Set[str] = set()
        step_missing_assets: Dict[int, Set[str]] = {}
        step_advanced: Dict[int, bool] = {}
        unbalanced_loop_notes: List[str] = []
        unbalanced_branch_notes: List[str] = []
        unclosed_resource_notes: List[str] = []
        uncomposable_notes: List[str] = []
        raw_complete_plans: List[CandidatePlan] = []
        choice_cache: Dict[Tuple[str, Tuple[str, ...]], List[float]] = {}

        seq = 0
        pq: List[
            Tuple[
                float,
                int,
                int,
                Dict[str, CanonicalPortType],
                Tuple[str, ...],
                Tuple[str, ...],
                Tuple[PlannedStep, ...],
            ]
        ] = [(0.0, 0, seq, dict(initial_env), tuple(initial_open), tuple(initial_released), ())]

        def _record_subblock_failure(sub_res: PlanResult) -> None:
            if sub_res.missing_ports:
                missing_asset_ports.update(sub_res.missing_ports)
            if sub_res.refusal_kind == "UNBALANCED_LOOP_RESOURCES" and sub_res.reason:
                unbalanced_loop_notes.append(sub_res.reason)
            elif sub_res.refusal_kind == "UNBALANCED_BRANCH_RESOURCES" and sub_res.reason:
                unbalanced_branch_notes.append(sub_res.reason)
            elif sub_res.refusal_kind == "UNCLOSED_RESOURCE" and sub_res.reason:
                unclosed_resource_notes.append(sub_res.reason)
            elif sub_res.reason:
                uncomposable_notes.append(sub_res.reason)

        expansions = 0
        while pq and expansions < max_expansions and len(raw_complete_plans) < max_plans * 3:
            cur_cost, step_idx, _, env_map, open_res_tup, released_res_tup, steps_tup = heapq.heappop(pq)
            expansions += 1

            if step_idx >= n_nodes:
                if require_all_closed and open_res_tup:
                    leaked = [
                        rv for rv in open_res_tup
                        if not (allow_deliverable_open and deliverable and rv == deliverable)
                    ]
                    if leaked:
                        unclosed_resource_notes.append(f"Unclosed linear resource(s): {', '.join(leaked)}")
                        continue
                raw_complete_plans.append(
                    CandidatePlan(
                        steps=steps_tup,
                        total_cost=round(cur_cost, 6),
                        open_resources=open_res_tup,
                        exit_released_resources=released_res_tup,
                    )
                )
                continue

            node = nodes[step_idx]
            released_set = set(released_res_tup)

            if node.kind in ("loop", "branch"):
                for ref in node.inputs:
                    if ref not in grounded_assets and ref in released_set:
                        uncomposable_notes.append(f"{node.id}:{ref}:use_after_release")
                        break
                else:
                    src_iter_sig = env_map.get(node.inputs[0]) if node.inputs else None
                    body_env = dict(env_map)
                    if node.kind == "loop":
                        loop_var = infer_loop_var(node)
                        if loop_var:
                            elem_rank = (
                                src_iter_sig.rank - 1
                                if (src_iter_sig is not None and isinstance(src_iter_sig.rank, int) and src_iter_sig.rank > 0)
                                else None
                            )
                            elem_carrier = "any"
                            if src_iter_sig is not None:
                                if elem_rank == 0 and src_iter_sig.dtype:
                                    elem_carrier = src_iter_sig.dtype
                                elif elem_rank is not None and elem_rank > 0 and src_iter_sig.carrier:
                                    elem_carrier = src_iter_sig.carrier
                            body_env[loop_var] = CanonicalPortType(
                                carrier=elem_carrier,
                                typestate="any",
                                rank=elem_rank,
                                dtype=src_iter_sig.dtype if src_iter_sig is not None else None,
                                domain=src_iter_sig.domain if src_iter_sig is not None else "",
                            )

                    body_res = self._plan_block(
                        node.body,
                        source_mode,
                        candidates_by_node,
                        grounded_assets,
                        deliverable=None,
                        initial_env=body_env,
                        initial_open=open_res_tup,
                        initial_released=released_res_tup,
                        allow_deliverable_open=False,
                        require_all_closed=False,
                        max_plans=max_plans,
                        max_expansions=max_expansions,
                    )
                    if not body_res.ok or body_res.chosen_plan is None:
                        _record_subblock_failure(body_res)
                        continue

                    body_plan = body_res.chosen_plan
                    body_steps_tup = body_plan.steps
                    body_open_set = set(body_open := body_plan.open_resources)
                    body_rel_set = set(body_plan.exit_released_resources)
                    body_tail_sig = (
                        body_steps_tup[-1].output_sig
                        if body_steps_tup
                        else CanonicalPortType(carrier="any")
                    )

                    else_steps_tup: Tuple[PlannedStep, ...] = ()
                    else_cost = 0.0
                    out_var = node.outputs[0] if node.outputs else node.id

                    if node.kind == "loop":
                        outer_open_set = set(open_res_tup)
                        released_outer = (body_rel_set - released_set) & outer_open_set
                        if body_open_set != outer_open_set or released_outer:
                            unbalanced_loop_notes.append(
                                f"{node.id}:loop must preserve linear resources "
                                f"(in={sorted(outer_open_set)}, out={sorted(body_open_set)})"
                            )
                            continue
                        out_sig = body_tail_sig
                        nxt_open_tup = tuple(body_open)
                        nxt_released_tup = tuple(body_plan.exit_released_resources)
                    elif node.else_body:
                        else_res = self._plan_block(
                            node.else_body,
                            source_mode,
                            candidates_by_node,
                            grounded_assets,
                            deliverable=None,
                            initial_env=dict(env_map),
                            initial_open=open_res_tup,
                            initial_released=released_res_tup,
                            allow_deliverable_open=False,
                            require_all_closed=False,
                            max_plans=max_plans,
                            max_expansions=max_expansions,
                        )
                        if not else_res.ok or else_res.chosen_plan is None:
                            _record_subblock_failure(else_res)
                            continue
                        else_plan = else_res.chosen_plan
                        else_steps_tup = else_plan.steps
                        else_cost = else_plan.total_cost
                        else_open_set = set(else_plan.open_resources)
                        if body_open_set != else_open_set:
                            unbalanced_branch_notes.append(
                                f"{node.id}:unbalanced branch resources "
                                f"(then={sorted(body_open_set)}, else={sorted(else_open_set)})"
                            )
                            continue
                        else_tail_sig = (
                            else_steps_tup[-1].output_sig
                            if else_steps_tup
                            else CanonicalPortType(carrier="any")
                        )
                        out_sig = join_port_types(body_tail_sig, else_tail_sig, self.poset)
                        nxt_open_tup = tuple(body_open)
                        nxt_released_tup = tuple(
                            dict.fromkeys(
                                list(body_plan.exit_released_resources)
                                + list(else_plan.exit_released_resources)
                            )
                        )
                    else:
                        outer_open_set = set(open_res_tup)
                        released_outer = (body_rel_set - released_set) & outer_open_set
                        if body_open_set != outer_open_set or released_outer:
                            unbalanced_branch_notes.append(
                                f"{node.id}:single-arm branch must preserve linear resources "
                                f"(in={sorted(outer_open_set)}, out={sorted(body_open_set)})"
                            )
                            continue
                        existing_outer = env_map.get(out_var) or env_map.get(node.id)
                        if existing_outer is not None:
                            out_sig = join_port_types(body_tail_sig, existing_outer, self.poset)
                        elif (
                            body_tail_sig.carrier
                            and body_tail_sig.carrier.lower() not in TOP_CARRIERS
                            and not body_tail_sig.carrier.startswith("Optional[")
                        ):
                            out_sig = replace(body_tail_sig, carrier=f"Optional[{body_tail_sig.carrier}]")
                        else:
                            out_sig = body_tail_sig
                        nxt_open_tup = tuple(body_open)
                        nxt_released_tup = tuple(body_plan.exit_released_resources)

                    nxt_env = dict(env_map)
                    nxt_env[node.id] = out_sig
                    if out_var:
                        nxt_env[out_var] = out_sig

                    block_cost = body_plan.total_cost + else_cost + 0.01
                    block_step = PlannedStep(
                        ir_node_id=node.id,
                        cell=None,
                        cell_id=f"__{node.kind}__:{node.id}",
                        is_live_bridge=False,
                        input_wiring={},
                        output_var=out_var,
                        output_sig=out_sig,
                        step_cost=round(block_cost, 6),
                        kind=node.kind,
                        condition=node.condition,
                        body_steps=body_steps_tup,
                        else_steps=else_steps_tup,
                    )
                    step_advanced[step_idx] = True
                    seq += 1
                    heapq.heappush(
                        pq,
                        (
                            cur_cost + block_cost,
                            step_idx + 1,
                            seq,
                            nxt_env,
                            nxt_open_tup,
                            nxt_released_tup,
                            steps_tup + (block_step,),
                        ),
                    )
                continue

            cands = candidates_by_node.get(node.id, []) or []
            if not cands:
                uncomposable_notes.append(f"{node.id}:no_retrieved_candidates")
                continue

            choice_probs: List[float] = [0.0] * len(cands)
            if len(cands) > 1 and self.choice_scorer_fn is not None:
                cand_ids = tuple(cm.cell_id for cm in cands)
                ckey = (node.id, cand_ids)
                if ckey in choice_cache:
                    choice_probs = choice_cache[ckey]
                else:
                    try:
                        raw_cp = list(self.choice_scorer_fn(node.intent, list(cand_ids)))
                        if len(raw_cp) == len(cands):
                            choice_probs = [float(x) for x in raw_cp]
                    except Exception:
                        pass
                    choice_cache[ckey] = choice_probs

            for cand_idx, cm in enumerate(cands):
                cell = cm.cell
                assets_ok, missing_here = _check_external_assets(node, cell, grounded_assets)
                if not assets_ok:
                    step_missing_assets.setdefault(step_idx, set()).update(missing_here)
                    continue

                if source_mode == "formal_syntax":
                    params_ok, param_fail = _check_formal_param_names(node, cell)
                    if not params_ok:
                        if param_fail:
                            uncomposable_notes.append(param_fail)
                        continue

                wired_ok, base_wiring, needed_bridges, fail_note = self._wire_data_inputs(
                    node,
                    cell,
                    source_mode,
                    env_map,
                    released_set,
                    grounded_assets,
                )
                if not wired_ok:
                    if fail_note:
                        uncomposable_notes.append(fail_note)
                    continue

                nxt_env = dict(env_map)
                nxt_steps: List[PlannedStep] = list(steps_tup)
                final_wiring = dict(base_wiring)
                bridge_cost_sum = 0.0

                for pname, b_path in needed_bridges:
                    cur_var = base_wiring[pname]
                    cur_sig = nxt_env[cur_var]
                    for edge in b_path.edges:
                        b_var = f"_b_{len(nxt_steps) + 1}"
                        nxt_sig = _apply_edge(cur_sig, edge)
                        nxt_steps.append(
                            PlannedStep(
                                ir_node_id=node.id,
                                cell=edge.cell,
                                cell_id=edge.cell_id,
                                is_live_bridge=True,
                                input_wiring={edge.in_port: cur_var},
                                output_var=b_var,
                                output_sig=nxt_sig,
                                step_cost=round(edge.cost, 6),
                            )
                        )
                        nxt_env[b_var] = nxt_sig
                        cur_var = b_var
                        cur_sig = nxt_sig
                    final_wiring[pname] = cur_var
                    bridge_cost_sum += b_path.total_cost

                dom = str(get_attr(cell, "domain_name", None) or "").strip().lower()
                raw_outs = get_attr(cell, "outputs", {}) or {}
                prim_name, out_port_obj = _primary_output_port(cell)

                pp = get_attr(cell, "polymorphic_propagation", None)
                pp_src_port = get_attr(pp, "source_port", None) if pp is not None else None
                poly_src_sig: Optional[CanonicalPortType] = None
                if pp_src_port and str(pp_src_port) in final_wiring:
                    poly_src_sig = nxt_env.get(final_wiring[str(pp_src_port)])
                elif final_wiring:
                    poly_src_sig = nxt_env.get(next(iter(final_wiring.values())))

                out_sigs_by_port: Dict[str, CanonicalPortType] = {}
                if isinstance(raw_outs, dict):
                    for ok, ov in raw_outs.items():
                        ok_str = str(ok)
                        raw_port_sig = CanonicalPortType.from_port(ov, dom)
                        out_sigs_by_port[ok_str] = propagate_polymorphic_signature(poly_src_sig, raw_port_sig, cell)

                step_output_vars: Dict[str, str] = {}
                if isinstance(raw_outs, dict) and raw_outs:
                    if len(node.outputs) > len(raw_outs):
                        uncomposable_notes.append(f"{node.id}:{cm.cell_id}:output_arity_mismatch")
                        continue
                    if len(node.outputs) > 1:
                        assigned_vars: Set[str] = set()
                        for ov_name in node.outputs:
                            if ov_name in out_sigs_by_port and ov_name not in step_output_vars:
                                step_output_vars[ov_name] = ov_name
                                assigned_vars.add(ov_name)
                        ordered_ports = [str(k) for k in raw_outs.keys()]
                        rem_ports = [pk for pk in ordered_ports if pk not in step_output_vars]
                        rem_vars = [ov for ov in node.outputs if ov not in assigned_vars]
                        for pk, ov_name in zip(rem_ports, rem_vars):
                            step_output_vars[pk] = ov_name
                        base_out = node.outputs[0]
                        for pk in out_sigs_by_port:
                            if pk not in step_output_vars:
                                step_output_vars[pk] = f"{base_out}_{pk}"
                    else:
                        base_out = node.outputs[0] if node.outputs else node.id
                        if prim_name:
                            step_output_vars[prim_name] = base_out
                        for pk in out_sigs_by_port:
                            if pk not in step_output_vars:
                                step_output_vars[pk] = f"{base_out}_{pk}"

                    out_var = step_output_vars.get(prim_name, node.outputs[0] if node.outputs else node.id)
                    out_sig = (
                        out_sigs_by_port.get(prim_name)
                        if prim_name and prim_name in out_sigs_by_port
                        else CanonicalPortType(carrier="void", domain=dom)
                    )
                else:
                    out_var = node.outputs[0] if node.outputs else node.id
                    out_sig = CanonicalPortType(carrier="void", domain=dom)

                nxt_open = list(open_res_tup)
                nxt_released = list(released_res_tup)
                raw_ins = get_attr(cell, "inputs", {}) or {}
                if isinstance(raw_ins, dict):
                    for ip_name, wired_v in final_wiring.items():
                        ip_obj = raw_ins.get(ip_name)
                        if ip_obj is not None:
                            ip_sig = CanonicalPortType.from_port(ip_obj, dom)
                            if ip_sig.resource_mode == "release":
                                if wired_v in nxt_open:
                                    nxt_open.remove(wired_v)
                                if wired_v not in nxt_released:
                                    nxt_released.append(wired_v)

                for ok_str, ov_name in step_output_vars.items():
                    ov_sig = out_sigs_by_port[ok_str]
                    nxt_env[ov_name] = ov_sig
                    if ov_sig.resource_mode == "acquire" and ov_name not in nxt_open:
                        nxt_open.append(ov_name)
                if not out_sigs_by_port:
                    nxt_env[out_var] = out_sig

                _apply_in_place_updates(
                    cell,
                    dom,
                    raw_ins,
                    raw_outs,
                    prim_name,
                    out_sigs_by_port,
                    final_wiring,
                    nxt_env,
                    self.poset,
                )

                clamped_score = max(1e-4, min(1.0, float(cm.score)))
                base_cost = max(0.01, -math.log(clamped_score))
                if source_mode == "slm_compiled":
                    avail_explicit = [
                        inp
                        for inp in node.inputs
                        if inp not in grounded_assets and inp in env_map and inp not in released_set
                    ]
                    if avail_explicit and not any(v in final_wiring.values() for v in avail_explicit):
                        base_cost += 0.85 if not final_wiring else 0.65
                if source_mode == "slm_compiled" and node.params and isinstance(raw_ins, dict):
                    prompt_cfg = {
                        pk: pv
                        for pk, pv in node.params.items()
                        if pv.value is not None and pv.provenance == "prompt" and pv.value_class != "external_asset"
                    }
                    if prompt_cfg:
                        aligned_cfg = align_node_params_to_ports(
                            prompt_cfg,
                            raw_ins,
                            allow_single_fallback=False,
                        )
                        unmatched_cnt = len(prompt_cfg) - len(aligned_cfg)
                        if unmatched_cnt > 0:
                            base_cost += 0.65 * unmatched_cnt
                max_cp = max(choice_probs) if choice_probs else 0.0
                if max_cp > 0.0:
                    rel_cp = max(0.0, min(1.0, float(choice_probs[cand_idx]) / max(max_cp, 1e-6)))
                    base_cost += 0.35 * (1.0 - rel_cp)
                total_step_cost = base_cost + bridge_cost_sum

                nxt_steps.append(
                    PlannedStep(
                        ir_node_id=node.id,
                        cell=cell,
                        cell_id=cm.cell_id,
                        is_live_bridge=False,
                        input_wiring=final_wiring,
                        output_var=out_var,
                        output_sig=out_sig,
                        step_cost=round(total_step_cost, 6),
                        output_vars=dict(step_output_vars),
                    )
                )

                step_advanced[step_idx] = True
                seq += 1
                heapq.heappush(
                    pq,
                    (
                        cur_cost + total_step_cost,
                        step_idx + 1,
                        seq,
                        nxt_env,
                        tuple(nxt_open),
                        tuple(nxt_released),
                        tuple(nxt_steps),
                    ),
                )

        if not raw_complete_plans:
            blocking_missing: Set[str] = set(missing_asset_ports)
            for s_idx, ports in step_missing_assets.items():
                if not step_advanced.get(s_idx, False):
                    blocking_missing.update(ports)
            if blocking_missing:
                sorted_missing = tuple(sorted(blocking_missing))
                return PlanResult(
                    ok=False,
                    chosen_plan=None,
                    refusal_kind="MISSING_EXTERNAL_ASSET",
                    reason=f"Missing required prompt-grounded external asset(s): {', '.join(sorted_missing)}",
                    missing_ports=sorted_missing,
                )
            if unbalanced_loop_notes:
                return PlanResult(
                    ok=False,
                    chosen_plan=None,
                    refusal_kind="UNBALANCED_LOOP_RESOURCES",
                    reason="; ".join(dict.fromkeys(unbalanced_loop_notes)),
                )
            if unbalanced_branch_notes:
                return PlanResult(
                    ok=False,
                    chosen_plan=None,
                    refusal_kind="UNBALANCED_BRANCH_RESOURCES",
                    reason="; ".join(dict.fromkeys(unbalanced_branch_notes)),
                )
            if unclosed_resource_notes:
                return PlanResult(
                    ok=False,
                    chosen_plan=None,
                    refusal_kind="UNCLOSED_RESOURCE",
                    reason="; ".join(dict.fromkeys(unclosed_resource_notes)),
                )
            return PlanResult(
                ok=False,
                chosen_plan=None,
                refusal_kind="UNCOMPOSABLE_TYPES",
                reason="; ".join(dict.fromkeys(uncomposable_notes)) or "No type-valid compositional plan found.",
            )

        best_by_ops: Dict[Tuple[str, ...], CandidatePlan] = {}
        for plan in raw_complete_plans:
            op_key = tuple(s.cell_id for s in plan.steps if not s.is_live_bridge)
            prev = best_by_ops.get(op_key)
            if prev is None or (plan.total_cost, len(plan.steps)) < (prev.total_cost, len(prev.steps)):
                best_by_ops[op_key] = plan

        non_equiv = sorted(
            best_by_ops.values(),
            key=lambda p: (
                p.total_cost,
                len(p.steps),
                tuple(s.cell_id for s in p.steps),
            ),
        )[:max_plans]

        c0 = non_equiv[0].total_cost
        temp = max(self.ambiguity_temperature, 1e-6)
        weights = [math.exp(-max(0.0, p.total_cost - c0) / temp) for p in non_equiv]
        z_part = sum(weights) or 1.0
        probs = [w / z_part for w in weights]
        k_plans = len(non_equiv)
        entropy = -sum(pr * math.log(max(pr, 1e-15)) for pr in probs)
        h_norm = (entropy / math.log(k_plans)) if k_plans >= 2 else 0.0
        margin = (probs[0] - probs[1]) if k_plans >= 2 else 1.0

        calibrated: Tuple[CandidatePlan, ...] = tuple(
            CandidatePlan(
                steps=p.steps,
                total_cost=p.total_cost,
                posterior_prob=round(probs[idx], 6),
                open_resources=p.open_resources,
                exit_released_resources=p.exit_released_resources,
            )
            for idx, p in enumerate(non_equiv)
        )

        is_ambiguous = bool(
            k_plans >= 2
            and h_norm >= self.entropy_threshold
            and margin <= self.margin_threshold
        )
        if is_ambiguous:
            return PlanResult(
                ok=False,
                chosen_plan=None,
                candidates=calibrated,
                refusal_kind="AMBIGUITY",
                reason=(
                    f"AMBIGUITY: indecisive posterior across {k_plans} non-equivalent plans "
                    f"(H_norm={h_norm:.3f}, margin={margin:.3f})"
                ),
                normalized_entropy=round(h_norm, 6),
                margin=round(margin, 6),
            )

        return PlanResult(
            ok=True,
            chosen_plan=calibrated[0],
            candidates=calibrated,
            normalized_entropy=round(h_norm, 6),
            margin=round(margin, 6),
        )

    def plan(
        self,
        ir: TypedIR,
        candidates_by_node: Dict[str, List[CandidateMatch]],
        max_plans: int = 8,
        max_expansions: int = 400,
    ) -> PlanResult:
        dag_errs = ir.validate_dag()
        if dag_errs:
            return PlanResult(
                ok=False,
                chosen_plan=None,
                refusal_kind="INVALID_IR",
                reason="; ".join(dag_errs),
            )
        return self._plan_block(
            ir.nodes,
            ir.source_mode,
            candidates_by_node,
            ir.grounded_assets(),
            deliverable=ir.deliverable,
            initial_env={},
            initial_open=(),
            initial_released=(),
            allow_deliverable_open=True,
            max_plans=max_plans,
            max_expansions=max_expansions,
        )


__all__ = [
    "PlannedStep",
    "CandidatePlan",
    "PlanResult",
    "TypedIRPlanner",
]
