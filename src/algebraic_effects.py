"""
src/algebraic_effects.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Solves Issue 1, Issue 4 & Proposal 4.B:
  1. Compositional Property Meet-Semilattice (Heyting Property Algebra) unifying
     atomic string typestates and postcondition predicates.
  2. Universal Algebraic Port Effects (`mutates`, `partition_index`, `key_of`/`key_effect`)
     replacing hardcoded pandas/sklearn/cv2 branches in verification contracts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from .lattice import Cell, PortSignature, AlgebraicSignature, TypeRegistry, is_path_port
except (ImportError, ValueError):
    from lattice import Cell, PortSignature, AlgebraicSignature, TypeRegistry, is_path_port


# ============================================================================
# 1. Compositional Property Meet-Semilattice (Solves Issue 4)
# ============================================================================

@dataclass(frozen=True)
class PropertyBundle:
    """
    Represents a variable's typestate as a monotonic conjunction of atomic state
    tags and key-value property predicates:
      Phi(v) = (active_states, predicates, key_space, partition_tag)
    When a pure or in-place transform `f` is applied to `v`, `Phi(f(v))` inherits
    all non-invalidated predicates of `v` plus the postconditions of `f`.
    """
    carrier_type: str
    active_states: Tuple[str, ...] = field(default_factory=tuple)
    predicates: Tuple[Tuple[str, str, Any], ...] = field(default_factory=tuple)
    keys_present: Tuple[str, ...] = field(default_factory=tuple)
    keys_deleted: Tuple[str, ...] = field(default_factory=tuple)
    partition_source: Optional[str] = None
    partition_index: Optional[int] = None
    lineage_root: Optional[str] = None

    def transition(
        self,
        new_carrier: Optional[str] = None,
        new_state: Optional[str] = None,
        added_predicates: Sequence[Tuple[str, str, Any]] = (),
        invalidated_props: Sequence[str] = (),
        added_keys: Sequence[str] = (),
        removed_keys: Sequence[str] = (),
        partition_source: Optional[str] = None,
        partition_index: Optional[int] = None,
    ) -> "PropertyBundle":
        """Monotonic semilattice join with selective property invalidation."""
        states = set(self.active_states)
        reg = TypeRegistry.get_instance()
        if new_state:
            ns_low = new_state.strip().lower()
            if ns_low and not reg.is_declared_top(ns_low) and not reg.is_declared_top_state(ns_low):
                states.add(ns_low)

        kept_preds = {
            (prop, op, val)
            for (prop, op, val) in self.predicates
            if prop not in invalidated_props and not any(ap[0] == prop for ap in added_predicates)
        }
        kept_preds.update(added_predicates)

        k_pres = (set(self.keys_present) | set(added_keys)) - set(removed_keys)
        k_del = (set(self.keys_deleted) | set(removed_keys)) - set(added_keys)

        return PropertyBundle(
            carrier_type=new_carrier or self.carrier_type,
            active_states=tuple(sorted(states)),
            predicates=tuple(sorted(kept_preds, key=lambda x: (str(x[0]), str(x[1]), repr(x[2])))),
            keys_present=tuple(sorted(k_pres)),
            keys_deleted=tuple(sorted(k_del)),
            partition_source=partition_source if partition_source is not None else self.partition_source,
            partition_index=partition_index if partition_index is not None else self.partition_index,
            lineage_root=self.lineage_root,
        )

    def entails(self, required_state: str, required_preds: Sequence[Tuple[str, str, Any]] = ()) -> bool:
        """
        Checks whether this bundle satisfies a consumer's required state and preconditions:
          Phi(producer) |- Phi(consumer)
        """
        req_s = (required_state or "").strip().lower()
        reg = TypeRegistry.get_instance()
        is_top = (
            not req_s
            or reg.is_declared_top(req_s)
            or (hasattr(reg, "is_top_state") and reg.is_top_state(req_s))
            or reg.is_declared_top_state(req_s)
        )
        if not is_top:
            active_lower = {str(s).lower() for s in self.active_states}
            state_ok = req_s in active_lower or any(
                (hasattr(reg, "is_state_compatible") and reg.is_state_compatible(s, req_s))
                or (hasattr(reg, "is_substate") and reg.is_substate(s, req_s))
                for s in self.active_states
            )
            if not state_ok:
                return False

        pred_map = {(p, op): v for (p, op, v) in self.predicates}
        for req_p, req_op, req_v in required_preds:
            if (req_p, req_op) not in pred_map or pred_map[(req_p, req_op)] != req_v:
                return False
        return True


# ============================================================================
# 2. Universal Algebraic Port Effect Inference (Solves Proposal 4.B & Issue 1)
# ============================================================================

@dataclass
class PortAlgebraicEffect:
    """
    Universal, domain-agnostic port effect annotations:
      - mutates_port    : name of input port mutated or superseded by this output
      - key_of_port     : name of carrier port indexed by this key port
      - key_effect      : "read" | "write" | "delete"
      - partition_index : 0-based partition index for multi-output splitters (e.g. 0=train, 1=test)
    """
    mutates_port: Optional[str] = None
    key_of_port: Optional[str] = None
    key_effect: Optional[str] = None
    partition_index: Optional[int] = None


def _find_carrier_input_port(cell: Any, reg: Optional[TypeRegistry] = None) -> Optional[str]:
    """Identifies the primary data-carrier / receiver input port via declared TypeRegistry roles."""
    inputs = getattr(cell, "inputs", {}) or {}
    if reg is None:
        reg = TypeRegistry.get_instance()
    dataflow_roles = reg.get_dataflow_roles_set()
    receiver_roles = reg.get_roles_for_category("receiver")
    for p_name, p_sig in inputs.items():
        binds = str(getattr(p_sig, "binds", "") or "").strip().lower()
        prole = str(getattr(p_sig, "port_role", "") or "").strip().lower()
        if binds in dataflow_roles or prole in receiver_roles:
            return p_name
    primary = getattr(cell, "primary_input", None)
    if primary is not None and primary.name in inputs:
        p_sig = inputs[primary.name]
        p_binds = str(getattr(p_sig, "binds", "") or "").strip().lower()
        p_role = str(getattr(p_sig, "port_role", "") or getattr(p_sig, "role", "") or "").strip().lower()
        p_kind = str(getattr(p_sig, "param_kind", "") or "").strip().lower()
        p_alg = getattr(p_sig, "signature", p_sig)
        t_l = str(getattr(p_alg, "type_name", "") or "").strip().lower()
        p_state = str(getattr(p_alg, "state", "") or "").strip().lower()
        cfg_roles = reg.get_config_roles_set() | reg.get_roles_for_category("configuration")
        is_scalar_cfg = (
            p_binds in cfg_roles
            or p_role in cfg_roles
            or p_kind in cfg_roles
            or p_binds == "hyperparameter"
            or p_state == "hyperparameter"
            or (
                bool(t_l)
                and (reg.is_subtype(t_l, "numeric") or reg.is_subtype(t_l, "bool"))
                and not reg.is_composite_carrier(t_l)
                and not reg.is_abstract_carrier(t_l)
            )
        )
        if not is_scalar_cfg:
            return primary.name
    return None


def infer_cell_algebraic_effects(cell: Cell) -> Dict[str, PortAlgebraicEffect]:
    """
    Derives algebraic port effects strictly from declared cell and port metadata:
      - Explicit port attributes: `mutates`, `key_of`, `key_effect`, `partition_index`
      - Declared cell attributes: `mutation_type == "in_place"` on identical carrier types
      - Explicit partition contracts in schema
    Zero heuristic string matching, zero ad-hoc hardcoded keyword sets, zero cell_id parsing.
    """
    effects_map: Dict[str, PortAlgebraicEffect] = {}
    reg = TypeRegistry.get_instance()
    inputs = getattr(cell, "inputs", {}) or {}
    outputs = getattr(cell, "outputs", {}) or {}

    # Identify declared carrier input port (if any)
    carrier_in_name = _find_carrier_input_port(cell, reg)

    # 1. Input port key effects
    for p_name, p_sig in inputs.items():
        explicit_mut = getattr(p_sig, "mutates", None)
        explicit_key_of = getattr(p_sig, "key_of", None)
        explicit_key_eff = getattr(p_sig, "key_effect", None)

        if explicit_mut or explicit_key_of or explicit_key_eff:
            effects_map[f"in:{p_name}"] = PortAlgebraicEffect(
                mutates_port=explicit_mut,
                key_of_port=explicit_key_of,
                key_effect=explicit_key_eff,
            )

    # 2. Output port mutation & partition effects
    def _output_group_key(sig_obj: Any) -> Tuple[Any, ...]:
        alg = getattr(sig_obj, "signature", sig_obj)
        rank_v = getattr(sig_obj, "rank", None)
        if rank_v is None and alg is not None:
            rank_v = getattr(alg, "rank", None)
        raw_quals = getattr(sig_obj, "qualifiers", None) or (getattr(alg, "qualifiers", None) if alg is not None else None) or ()
        norm_quals = tuple(sorted(str(q).strip().lower() for q in raw_quals if str(q).strip()))
        return (
            str(getattr(sig_obj, "type_name", "") or "").strip().lower(),
            str(getattr(sig_obj, "port_role", "") or "").strip().lower(),
            rank_v,
            norm_quals,
        )

    out_group_counts: Dict[Tuple[Any, ...], int] = {}
    out_group_cursor: Dict[Tuple[Any, ...], int] = {}
    partition_label_map: Dict[str, int] = {}
    leading_shape_dims: Set[str] = set()
    if len(outputs) >= 2:
        for out_sig in outputs.values():
            g_key = _output_group_key(out_sig)
            out_group_counts[g_key] = out_group_counts.get(g_key, 0) + 1
            sc = getattr(out_sig, "shape_contract", None) or getattr(getattr(out_sig, "signature", None), "shape_contract", None)
            dims = getattr(sc, "shape", None) if sc is not None and not isinstance(sc, dict) else (sc.get("shape") if isinstance(sc, dict) else None)
            if isinstance(dims, (list, tuple)) and dims and isinstance(dims[0], str) and dims[0].strip():
                leading_shape_dims.add(dims[0].strip().lower())

    for idx, (out_name, out_sig) in enumerate(outputs.items()):
        mut_target = getattr(out_sig, "mutates", None)
        part_idx = getattr(out_sig, "partition_index", None)
        g_key = _output_group_key(out_sig)
        # Co-indexed outputs sharing a single leading dimension (e.g. X[N, d], y[N]) form a Cartesian
        # product rather than a disjoint coproduct partition, even when carriers share a base type.
        is_coproduct_split = (
            len(outputs) >= 2
            and bool(carrier_in_name)
            and carrier_in_name in inputs
            and bool(g_key[0])
            and out_group_counts.get(g_key, 0) >= 2
            and len(leading_shape_dims) != 1
        )

        # In-place endomorphism f: T -> T on declared carrier (skip on multi-output splitters unless explicit)
        if (
            mut_target is None
            and carrier_in_name
            and carrier_in_name in inputs
            and (not is_coproduct_split or getattr(cell, "mutation_type", "") == "in_place")
        ):
            in_t = str(getattr(inputs[carrier_in_name], "type_name", "")).lower()
            out_t = str(getattr(out_sig, "type_name", "")).lower()
            if getattr(cell, "mutation_type", "") == "in_place" or (
                in_t and out_t and (in_t == out_t or (reg and reg.is_subtype(out_t, in_t)))
                and getattr(cell, "stage", None) == 2
            ):
                mut_target = carrier_in_name

        # Declared partition metadata, typestate partition property, shape-contract split dim, or sequential fallback
        if part_idx is None and getattr(out_sig, "partition", None) is not None:
            p_decl = getattr(out_sig, "partition")
            if isinstance(p_decl, int):
                part_idx = p_decl
            elif isinstance(p_decl, dict) and "index" in p_decl:
                try:
                    part_idx = int(p_decl["index"])
                except (ValueError, TypeError):
                    pass
            elif isinstance(p_decl, str) and p_decl.strip():
                part_idx = partition_label_map.setdefault(p_decl.strip().lower(), len(partition_label_map))
        if part_idx is None and reg is not None:
            for st_cand in (getattr(out_sig, "typestate", None), getattr(out_sig, "state", None)):
                st_str = str(st_cand or "").strip()
                if not st_str:
                    continue
                st_props = reg.get_state_properties(st_str)
                if st_props and st_props.get("partition") is not None:
                    p_val = st_props["partition"]
                    if isinstance(p_val, int):
                        part_idx = p_val
                    else:
                        p_key = str(p_val).strip().lower()
                        if p_key:
                            part_idx = partition_label_map.setdefault(p_key, len(partition_label_map))
                    if part_idx is not None:
                        break
        if part_idx is None and is_coproduct_split and len(leading_shape_dims) >= 2:
            sc = getattr(out_sig, "shape_contract", None) or getattr(getattr(out_sig, "signature", None), "shape_contract", None)
            dims = getattr(sc, "shape", None) if sc is not None and not isinstance(sc, dict) else (sc.get("shape") if isinstance(sc, dict) else None)
            if isinstance(dims, (list, tuple)) and dims and isinstance(dims[0], str) and dims[0].strip():
                part_idx = partition_label_map.setdefault(dims[0].strip().lower(), len(partition_label_map))
        if part_idx is None and is_coproduct_split:
            part_idx = out_group_cursor.get(g_key, 0)
            out_group_cursor[g_key] = part_idx + 1

        if mut_target is not None or part_idx is not None:
            effects_map[f"out:{out_name}"] = PortAlgebraicEffect(
                mutates_port=mut_target,
                partition_index=part_idx,
            )

    return effects_map


def _is_void_binding(val: Any) -> bool:
    if not isinstance(val, str) or not val.strip():
        return True
    try:
        from .lattice import TypeRegistry
    except (ImportError, ValueError):
        from lattice import TypeRegistry
    head = val.strip().split("(", 1)[0].strip()
    return TypeRegistry.get_instance().is_void_type(head)


def build_universal_verification_contract(
    pipeline_bindings: List[Tuple[Cell, Dict[str, str]]],
    contract_cls: Any,
) -> Any:
    """
    Solves Issue 1 & Proposal 4.B:
    Constructs `VerificationContract` purely from algebraic port effects (`mutates`,
    `partition_index`, and `postconditions`) with ZERO domain-specific if/elif branches.
    """
    contract = contract_cls()
    try:
        from .lattice import TypeRegistry
    except (ImportError, ValueError):
        from lattice import TypeRegistry
    reg = TypeRegistry.get_instance()
    latest_carrier_lineage: Dict[str, str] = {}   # root_var -> latest mutated var
    var_to_root: Dict[str, str] = {}              # var -> root_var
    var_partition: Dict[str, Tuple[str, int]] = {}  # var -> (split_cell_id, partition_index)
    partition_zero_by_role: Dict[str, str] = {}   # role/type -> var with partition_index == 0
    var_props: Dict[str, Dict[str, Any]] = {}          # var -> {prop: expected_val}

    for cell, bindings in pipeline_bindings:
        eff_map = infer_cell_algebraic_effects(cell)
        stage = getattr(cell, "stage", None)

        # 1. Track Linear Carrier Mutation Lineage & Partition Indices before recording postconditions
        carrier_in_name = _find_carrier_input_port(cell, reg)
        in_place_targets: Dict[str, str] = {}

        for out_name, out_sig in (getattr(cell, "outputs", {}) or {}).items():
            out_v = bindings.get(out_name) or bindings.get("output_var")
            if _is_void_binding(out_v):
                continue

            out_eff = eff_map.get(f"out:{out_name}")
            if out_eff and out_eff.mutates_port:
                in_v = bindings.get(out_eff.mutates_port)
                if in_v and isinstance(in_v, str):
                    root = var_to_root.get(in_v, in_v)
                    var_to_root[out_v] = root
                    latest_carrier_lineage[root] = out_v
                    inherited = dict(var_props.get(in_v, {}))
                    var_props[out_v] = {**inherited, **var_props.get(out_v, {})}
                    if getattr(cell, "mutation_type", "") == "in_place":
                        in_place_targets[out_v] = in_v
                else:
                    var_props.setdefault(out_v, {})
            elif out_eff and out_eff.partition_index == 0 and carrier_in_name:
                in_sig_c = (getattr(cell, "inputs", {}) or {}).get(carrier_in_name)
                in_t_c = str(getattr(getattr(in_sig_c, "signature", in_sig_c), "type_name", "") or "").lower()
                out_t_c = str(getattr(getattr(out_sig, "signature", out_sig), "type_name", "") or "").lower()
                same_carrier_family = bool(
                    in_t_c and out_t_c and (out_t_c == in_t_c or (reg and reg.is_subtype(out_t_c, in_t_c)))
                )
                in_v = bindings.get(carrier_in_name)
                if same_carrier_family and in_v and isinstance(in_v, str):
                    root = var_to_root.get(in_v, in_v)
                    var_to_root[out_v] = root
                    latest_carrier_lineage[root] = out_v
                    inherited = dict(var_props.get(in_v, {}))
                    var_props[out_v] = {**inherited, **var_props.get(out_v, {})}
                else:
                    var_to_root.setdefault(out_v, out_v)
                    latest_carrier_lineage.setdefault(out_v, out_v)
                    var_props.setdefault(out_v, {})
            else:
                var_to_root.setdefault(out_v, out_v)
                latest_carrier_lineage.setdefault(out_v, out_v)
                var_props.setdefault(out_v, {})

            if out_eff and out_eff.partition_index is not None:
                var_partition[out_v] = (cell.cell_id, out_eff.partition_index)
                if out_eff.partition_index == 0:
                    role_key = str(getattr(out_sig, "port_role", "") or getattr(out_sig, "type_name", "data")).lower()
                    partition_zero_by_role[role_key] = out_v
            elif out_eff and out_eff.mutates_port:
                mut_in_v = bindings.get(out_eff.mutates_port)
                if isinstance(mut_in_v, str) and mut_in_v in var_partition:
                    var_partition[out_v] = var_partition[mut_in_v]

        # 2. Record declared cell postconditions along per-variable causal lineage
        for post in (getattr(cell, "postconditions", []) or []):
            prop = getattr(post, "property", None) or (post.get("property") if isinstance(post, dict) else None)
            op = getattr(post, "operator", "==") if not isinstance(post, dict) else post.get("operator", "==")
            val = getattr(post, "value", None) if not isinstance(post, dict) else post.get("value")
            expr = getattr(post, "expression", None) if not isinstance(post, dict) else post.get("expression")
            desc = getattr(post, "description", None) if not isinstance(post, dict) else post.get("description")
            t_port = (getattr(post, "target", None) if not isinstance(post, dict) else post.get("target")) or "output_var"
            t_var = bindings.get(t_port) or bindings.get("output_var") or (
                bindings.get(cell.primary_output.name) if getattr(cell, "primary_output", None) else None
            )
            if t_var:
                contract.cell_checks.append({
                    "cell_id": cell.cell_id,
                    "target_var": t_var,
                    "target_port": t_port,
                    "property": prop,
                    "operator": op,
                    "value": val,
                    "expression": expr,
                    "description": desc,
                })
                if prop and str(prop).strip().lower() != "state":
                    # "state" is a compile-time typestate tag verified statically in Layer 2/3,
                    # not a runtime Python attribute on carriers like np.ndarray.
                    var_props.setdefault(t_var, {})[str(prop)] = val
                    if t_var in in_place_targets:
                        var_props.setdefault(in_place_targets[t_var], {})[str(prop)] = val

        # 3. Universal Co-Initial Partition Verification on Multi-Input Consumers
        bound_split_ports: List[Tuple[str, str, Tuple[str, int]]] = []
        for in_name, in_sig in (getattr(cell, "inputs", {}) or {}).items():
            in_v = bindings.get(in_name)
            if isinstance(in_v, str) and in_v in var_partition:
                bound_split_ports.append((in_name, in_v, var_partition[in_v]))

        if len(bound_split_ports) >= 2:
            # Check partition coherence (e.g., X_train paired with y_train, never X_train with y_test)
            distinct_partitions = {p_idx for _, _, (_, p_idx) in bound_split_ports}
            anchor_port, anchor_var, (anchor_split, anchor_p_idx) = bound_split_ports[0]
            port_details = {
                in_name: {"var": in_v, "split_cell": split_cid, "partition_index": p_idx}
                for in_name, in_v, (split_cid, p_idx) in bound_split_ports
            }
            if len(distinct_partitions) > 1:
                for in_name, in_v, (split_cid, p_idx) in bound_split_ports[1:]:
                    if p_idx != anchor_p_idx:
                        contract.terminal_checks.append({
                            "type": "algebraic_partition_coherence",
                            "cell_id": cell.cell_id,
                            "coherent": False,
                            "port": in_name,
                            "anchor_port": anchor_port,
                            "expected_partition": anchor_p_idx,
                            "actual_partition": p_idx,
                            "actual_var": in_v,
                            "ports": port_details,
                            "violation": (
                                f"Cell '{cell.cell_id}' mixes split partitions: "
                                f"'{anchor_port}' ({anchor_var}) has partition {anchor_p_idx} "
                                f"from '{anchor_split}', while '{in_name}' ({in_v}) has partition "
                                f"{p_idx} from '{split_cid}'."
                            ),
                        })
            else:
                contract.terminal_checks.append({
                    "type": "algebraic_partition_coherence",
                    "cell_id": cell.cell_id,
                    "coherent": True,
                    "expected_partition": anchor_p_idx,
                    "actual_partition": anchor_p_idx,
                    "ports": port_details,
                })

        # 4. Universal Linear Egress Verification on Path-Consuming Sinks
        has_path_in = any(is_path_port(p) for p in (getattr(cell, "inputs", {}) or {}).values())
        if has_path_in and stage != 1:
            path_var = None
            data_var = None
            for in_name, in_sig in cell.inputs.items():
                if is_path_port(in_sig):
                    path_var = bindings.get(in_name)
                elif data_var is None:
                    data_var = bindings.get(in_name)

            if data_var and isinstance(data_var, str):
                root = var_to_root.get(data_var, data_var)
                expected_latest = latest_carrier_lineage.get(root, data_var)
                contract.terminal_checks.append({
                    "type": "algebraic_linear_egress",
                    "cell_id": cell.cell_id,
                    "saved_var": data_var,
                    "ingress_var": root,
                    "expected_var": expected_latest,
                    "actual_var": data_var,
                    "output_path": path_var,
                    "expected_properties": dict(var_props.get(data_var, {})),
                })

    # 5. Universal In-Memory Observable Deliverable Verification
    all_output_vars: Set[str] = set()
    for cell, bindings in pipeline_bindings:
        for out_name in (getattr(cell, "outputs", {}) or {}).keys():
            ov = bindings.get(out_name) or bindings.get("output_var")
            if not _is_void_binding(ov):
                all_output_vars.add(ov)
        ov = bindings.get("output_var")
        if not _is_void_binding(ov):
            all_output_vars.add(ov)

    all_downstream_consumed: Set[str] = set()
    for _, b_map in pipeline_bindings:
        for in_k, in_v in b_map.items():
            if isinstance(in_v, str):
                root_v = in_v.split("[")[0].strip()
                if root_v in all_output_vars:
                    all_downstream_consumed.add(root_v)

    n_steps = len(pipeline_bindings)
    for idx, (cell, bindings) in enumerate(pipeline_bindings):
        is_final_step = (idx == n_steps - 1)
        for out_name, out_sig in (getattr(cell, "outputs", {}) or {}).items():
            out_v = bindings.get(out_name) or bindings.get("output_var")
            if _is_void_binding(out_v):
                continue

            is_unconsumed = out_v not in all_downstream_consumed
            has_clause = getattr(cell, "matched_clause_idx", None) is not None
            if is_final_step or (is_unconsumed and has_clause):
                if any(c.get("type") == "observable_deliverable" and c.get("var") == out_v for c in contract.terminal_checks):
                    continue

                cell_posts = getattr(cell, "postconditions", []) or []
                contract.terminal_checks.append({
                    "type": "observable_deliverable",
                    "cell_id": cell.cell_id,
                    "var": out_v,
                    "port": out_name,
                    "type_name": getattr(out_sig, "type_name", None),
                    "structure_kind": getattr(out_sig, "structure_kind", None),
                    "is_final": is_final_step,
                    "clause_idx": getattr(cell, "matched_clause_idx", None),
                    "postconditions": [
                        p for p in cell_posts
                        if (getattr(p, "target", None) if not isinstance(p, dict) else p.get("target")) in (out_name, "output_var", None)
                        and str(getattr(p, "property", None) if not isinstance(p, dict) else p.get("property") or "").strip().lower() != "state"
                    ],
                })

    return contract


__all__ = [
    "PropertyBundle",
    "PortAlgebraicEffect",
    "infer_cell_algebraic_effects",
    "build_universal_verification_contract",
]
