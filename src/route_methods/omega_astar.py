"""
src/route_methods/omega_astar.py - Profile Omega Type-Directed Bidirectional A* Search
over the Free Symmetric Monoidal Category (Stage 3).

Combines:
1. Symmetric Monoidal State Gamma_t = [(out_sig_1, src_1), ..., (out_sig_m, src_m)]
   allowing both sequential morphisms (A -> B -> C) and multi-input DAG joins
   ((A, B) -> C) whenever all required dataflow carriers exist in Gamma_t.
2. Backward Precondition Cone & Reverse Distance Heuristic h_bwd(cell) from target
   goal signatures / terminal clause anchors.
3. Exact Valid Outgoing Frontier E_valid(Gamma_t) computed by Robinson unification
   + pointed monoidal defaults eta_tau.
4. Schema-Constrained SLM Choice Scoring strictly over E_valid(Gamma_t) (zero
   out-of-lattice hallucinations).
"""

from __future__ import annotations

from collections import deque
import copy
import heapq
import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .base import RouteMethod, _clause_has_egress_intent, is_ingress_source, is_terminal_sink
    from ..lattice import Cell, TypeRegistry, is_path_port
    from ..unification import (
        unify,
        Substitution,
        _pointed_default,
        substitute_generics,
        _prop_cond_field,
    )
    from ..tokenizer import CellTokenizer
except (ImportError, ValueError):
    from route_methods.base import RouteMethod, _clause_has_egress_intent, is_ingress_source, is_terminal_sink
    from lattice import Cell, TypeRegistry, is_path_port
    from unification import (
        unify,
        Substitution,
        _pointed_default,
        substitute_generics,
        _prop_cond_field,
    )
    from tokenizer import CellTokenizer


def _get_sk_sets(reg: Optional[Any] = None) -> Tuple[Set[str], Set[str], Set[str]]:
    if reg is None:
        try:
            reg = TypeRegistry.get_instance()
        except Exception:
            reg = None
    if reg is None:
        return set(), set(), set()
    return (
        set(reg.get_verification_semantics("path_compat_structure_kinds") or ()),
        set(reg.get_verification_semantics("scalar_abstract_types") or ()),
        set(reg.get_verification_semantics("compound_structure_kinds") or ()),
    )


def _port_requires_carrier(cell: Any, port: Any, reg: Optional[Any] = None) -> bool:
    """True iff `port` is a required dataflow carrier input (cannot rely solely on eta_tau)."""
    if not getattr(port, "required", True):
        return False
    if reg is None:
        try:
            reg = TypeRegistry.get_instance()
        except Exception:
            reg = None
    role = str(getattr(port, "port_role", "") or getattr(port, "role", "") or "").strip().lower()
    if reg is not None:
        if reg.is_config_role(role) or reg.is_hof_role(role) or role in set(reg.get_verification_semantics("parameter_roles") or ()):
            return False
    sk = str(getattr(port, "structure_kind", "") or "").strip().lower()
    t_name = str(getattr(getattr(port, "signature", port), "type_name", "") or "").strip().lower()
    def_val = getattr(port, "default_value", None)
    if def_val is not None:
        is_str_t = reg.is_subtype(t_name, "str") if (reg is not None and t_name) else (t_name == str.__name__)
        if not (def_val == "" and (role == "data_input" or not is_str_t)):
            return False
    if is_path_port(port):
        return False
    if _pointed_default(port, cell) is not None:
        return False
    builtin_scalars = {cls.__name__.lower() for cls in (int, float, bool, str)}
    if sk == "scalar" and (reg.is_scalar_primitive(t_name) if reg is not None else t_name in builtin_scalars):
        return False
    return True


def _cell_required_data_inputs(cell: Any, reg: Optional[Any] = None) -> List[Tuple[str, Any]]:
    ins = getattr(cell, "inputs", {}) or {}
    if not isinstance(ins, dict):
        return []
    return [(k, p) for k, p in ins.items() if _port_requires_carrier(cell, p, reg=reg)]


def _cell_outputs_ports(cell: Any) -> List[Tuple[Any, Any]]:
    outs = getattr(cell, "outputs", {}) or {}
    if not isinstance(outs, dict):
        return []
    res: List[Tuple[Any, Any]] = []
    for p in outs.values():
        sig = getattr(p, "signature", p)
        if sig is not None:
            res.append((p, sig))
    return res


def _cell_outputs_sigs(cell: Any) -> List[Any]:
    return [sig for _, sig in _cell_outputs_ports(cell)]


def _sig_dedup_key(
    sig: Any,
    cache: Optional[Dict[int, Tuple[Any, ...]]] = None,
) -> Tuple[Any, ...]:
    if cache is not None:
        sid = id(sig)
        hit = cache.get(sid)
        if hit is not None:
            return hit
    tname = str(getattr(sig, "type_name", sig) or "")
    state = str(getattr(sig, "state", "") or "")
    quals = tuple(sorted({str(q) for q in (getattr(sig, "qualifiers", None) or ())}))
    abs_t = str(getattr(sig, "abstract_type", "") or "")
    acc_st = tuple(sorted({str(s) for s in (getattr(sig, "accepted_states", None) or ())}))
    par_st = str(getattr(sig, "parent_state", "") or "")
    res = (tname, state, quals, abs_t, acc_st, par_st)
    if cache is not None:
        cache[id(sig)] = res
    return res


def _cell_declared_props(cell: Any, reg: Optional[Any]) -> Dict[str, Any]:
    """Extracts static output properties declared by `cell` via typestate or postconditions."""
    props: Dict[str, Any] = {}
    outs = getattr(cell, "outputs", {}) or {}
    out_p = getattr(cell, "primary_output", None) or (
        outs.get("output_var") or (next(iter(outs.values())) if outs else None)
    )
    if out_p is not None and reg is not None:
        out_sig = getattr(out_p, "signature", out_p)
        out_state = getattr(out_sig, "state", None) or getattr(out_p, "state", None)
        if out_state:
            try:
                for pk, pv in (reg.get_state_properties(out_state) or {}).items():
                    if pv is not None:
                        props[pk] = pv
            except Exception:
                pass
    for post in (getattr(cell, "postconditions", None) or ()):
        if _prop_cond_field(post, "operator") == "==" and _prop_cond_field(post, "target") in ("output_var", None):
            pk = _prop_cond_field(post, "property")
            if pk:
                props[str(pk)] = _prop_cond_field(post, "value")
    return props


def _cell_required_preconds(cell: Any) -> Dict[str, List[Tuple[str, str, Any]]]:
    """Maps required input port name -> [(property, op, value)] from cell.preconditions."""
    out: Dict[str, List[Tuple[str, str, Any]]] = {}
    for pre in (getattr(cell, "preconditions", None) or ()):
        op = _prop_cond_field(pre, "operator")
        if op not in ("==", "!="):
            continue
        tgt = _prop_cond_field(pre, "target")
        prop = _prop_cond_field(pre, "property")
        val = _prop_cond_field(pre, "value")
        if tgt and prop:
            out.setdefault(str(tgt), []).append((str(prop), op, val))
    return out


def _typevar_to_concrete_blocked(
    out_sig: Any,
    in_sig: Any,
    reg: Optional[Any],
    tv_cache: Optional[Dict[Tuple[str, str], bool]] = None,
) -> bool:
    """Parametricity guard: an uninstantiated existential TypeVar output (e.g. `U`)
    must not unify forward with an unrelated concrete library type (e.g. `cv2.VideoCapture`)
    unless the target is itself a TypeVar or declared Top."""
    if reg is None or out_sig is None or in_sig is None:
        return False
    out_t = str(getattr(out_sig, "type_name", "") or "").strip()
    in_t = str(getattr(in_sig, "type_name", "") or "").strip()
    if not out_t or not in_t:
        return False
    if tv_cache is not None:
        ck = (out_t, in_t)
        hit = tv_cache.get(ck)
        if hit is not None:
            return hit
    res = bool(reg.is_type_variable(out_t) and not (reg.is_type_variable(in_t) or reg.is_declared_top(in_t)))
    if tv_cache is not None:
        tv_cache[(out_t, in_t)] = res
    return res


def _structural_kind_compatible(out_port: Any, in_port: Any) -> bool:
    """Checks fundamental carrier structure_kind / abstract_type compatibility without rejecting liftable rank/state edges."""
    if out_port is None or in_port is None:
        return True
    out_sk = str(getattr(out_port, "structure_kind", "") or "").strip().lower()
    in_sk = str(getattr(in_port, "structure_kind", "") or "").strip().lower()
    if out_sk == "handle" and in_sk and in_sk != "handle":
        return False
    if in_sk == "handle" and out_sk and out_sk != "handle":
        return False
    path_compat_sk, scalar_abs, compound_sk = _get_sk_sets()
    if out_sk == "path_spec" and in_sk and in_sk not in path_compat_sk:
        return False
    if in_sk == "path_spec" and out_sk and out_sk not in path_compat_sk:
        return False
    out_sig = getattr(out_port, "signature", out_port)
    in_sig = getattr(in_port, "signature", in_port)
    out_abs = str(getattr(out_port, "abstract_type", "") or getattr(out_sig, "abstract_type", "") or "").strip().lower()
    in_abs = str(getattr(in_port, "abstract_type", "") or getattr(in_sig, "abstract_type", "") or "").strip().lower()
    is_out_scalar = (out_sk == "scalar") or (out_abs in scalar_abs)
    is_in_scalar = (in_sk == "scalar") or (in_abs in scalar_abs)
    is_out_compound = out_sk in compound_sk
    is_in_compound = in_sk in compound_sk
    if is_out_scalar and is_in_compound:
        return False
    if is_in_scalar and is_out_compound:
        return False
    return True


def _can_fire_under_gamma(
    cell: Any,
    gamma: Sequence_Gamma,
    sigma: Any,
    unify_cache: Optional[Dict[Tuple[Tuple[Any, ...], Tuple[Any, ...]], bool]] = None,
    edge_cache: Optional[Dict[Tuple[int, int, str, str], bool]] = None,
    reg: Optional[Any] = None,
    req_ins: Optional[List[Tuple[str, Any]]] = None,
    sig_key_cache: Optional[Dict[int, Tuple[Any, ...]]] = None,
    tv_cache: Optional[Dict[Tuple[str, str], bool]] = None,
    unify_empty_sub_cache: Optional[Dict[Tuple[Tuple[Any, ...], Tuple[Any, ...]], Optional[Tuple[Tuple[str, Any], ...]]]] = None,
) -> Optional[Any]:
    """
    Returns updated Substitution iff ALL required dataflow inputs of `cell`
    satisfy carrier structure_kind compatibility and unify with carriers in `gamma`, else None.
    """
    if reg is None:
        try:
            reg = TypeRegistry.get_instance()
        except Exception:
            reg = None
    if req_ins is None:
        req_ins = _cell_required_data_inputs(cell)
    cur_sigma = sigma.copy() if hasattr(sigma, "copy") else Substitution()
    if not req_ins:
        return cur_sigma
    if not gamma:
        return None
    empty_sub = not getattr(cur_sigma, "mappings", None)
    cons_id = str(getattr(cell, "cell_id", "") or "")
    for _, in_port in req_ins:
        in_sig = getattr(in_port, "signature", in_port)
        in_key = _sig_dedup_key(in_sig, sig_key_cache) if (unify_cache is not None and empty_sub) else None
        matched = False
        for g_item in gamma:
            if isinstance(g_item, tuple) and len(g_item) == 3:
                out_sig, out_port, prod_cell = g_item
            else:
                out_sig, out_port, prod_cell = g_item, None, None
            if out_port is not None:
                prod_id = str(getattr(prod_cell, "cell_id", "") or "")
                ek = (id(out_port), id(in_port), prod_id, cons_id)
                ev_ok = edge_cache.get(ek) if edge_cache is not None else None
                if ev_ok is None:
                    ev_ok = _structural_kind_compatible(out_port, in_port)
                    if edge_cache is not None:
                        edge_cache[ek] = ev_ok
                if not ev_ok:
                    continue
            if _typevar_to_concrete_blocked(out_sig, in_sig, reg, tv_cache=tv_cache):
                continue
            if empty_sub and in_key is not None:
                out_key = _sig_dedup_key(out_sig, sig_key_cache)
                pair_k = (out_key, in_key)
                if unify_cache is not None:
                    cached = unify_cache.get(pair_k)
                    if cached is False:
                        continue
                if unify_empty_sub_cache is not None and pair_k in unify_empty_sub_cache:
                    m_tup = unify_empty_sub_cache[pair_k]
                    if m_tup is None:
                        continue
                    if not m_tup:
                        matched = True
                        break
                    sub = cur_sigma.copy() if hasattr(cur_sigma, "copy") else Substitution()
                    sub.mappings.update(m_tup)
                    cur_sigma = sub
                    empty_sub = not getattr(cur_sigma, "mappings", None)
                    matched = True
                    break
                sub = unify(out_sig, in_sig, cur_sigma)
                if unify_cache is not None:
                    unify_cache[pair_k] = (sub is not None)
                if unify_empty_sub_cache is not None:
                    unify_empty_sub_cache[pair_k] = (
                        tuple(sub.mappings.items()) if (sub is not None and hasattr(sub, "mappings")) else (
                            () if sub is not None else None
                        )
                    )
            else:
                sub = unify(out_sig, in_sig, cur_sigma)
            if sub is not None:
                cur_sigma = sub
                empty_sub = not getattr(cur_sigma, "mappings", None)
                matched = True
                break
        if not matched:
            return None
    return cur_sigma


Sequence_Gamma = Tuple[Any, ...]


class OmegaTypeDirectedAStarRouteMethod(RouteMethod):
    """
    Stage 3 Search for Profile Omega: Type-Directed Bidirectional A* over
    Symmetric Monoidal Category states (chain, Gamma_t, sigma_t) with SLM
    constrained ranking strictly over E_valid(Gamma_t).
    """

    name: str = "omega"

    def plan(
        self,
        prompt: str,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
        orchestrator: Any = None,
        ctx: Any = None,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        rag: Any = None,
        **kwargs,
    ) -> List[Cell]:
        if orchestrator is not None:
            self.orchestrator = orchestrator
        if not tunnel:
            return []

        pool: List[Cell] = []
        seen_ids: Set[str] = set()
        for c in tunnel:
            cid = getattr(c, "cell_id", None)
            if not cid or cid in seen_ids or getattr(c, "node_type", "") == "constant":
                continue
            pool.append(c)
            seen_ids.add(cid)
        if not pool:
            return []

        try:
            reg = TypeRegistry.get_instance()
        except Exception:
            reg = None

        unify_cache: Dict[Tuple[Tuple[Any, ...], Tuple[Any, ...]], bool] = {}
        unify_empty_sub_cache: Dict[Tuple[Tuple[Any, ...], Tuple[Any, ...]], Optional[Tuple[Tuple[str, Any], ...]]] = {}
        edge_cache: Dict[Tuple[int, int, str, str], bool] = {}
        sig_key_cache: Dict[int, Tuple[Any, ...]] = {}
        tv_cache: Dict[Tuple[str, str], bool] = {}

        def _unify_cached(
            out_sig: Any,
            in_sig: Any,
            out_port: Any = None,
            in_port: Any = None,
            prod_cell: Any = None,
            cons_cell: Any = None,
        ) -> bool:
            if _typevar_to_concrete_blocked(out_sig, in_sig, reg, tv_cache=tv_cache):
                return False
            if out_port is not None and in_port is not None:
                prod_id = str(getattr(prod_cell, "cell_id", "") or "")
                cons_id = str(getattr(cons_cell, "cell_id", "") or "")
                ek = (id(out_port), id(in_port), prod_id, cons_id)
                ev_ok = edge_cache.get(ek)
                if ev_ok is None:
                    ev_ok = _structural_kind_compatible(out_port, in_port)
                    edge_cache[ek] = ev_ok
                if not ev_ok:
                    return False
            pk = (_sig_dedup_key(out_sig, sig_key_cache), _sig_dedup_key(in_sig, sig_key_cache))
            res = unify_cache.get(pk)
            if res is None:
                res = unify(out_sig, in_sig) is not None
                unify_cache[pk] = res
            return res

        prompt_toks = set(CellTokenizer.tokenize_prompt(prompt)) if prompt else set()
        clause_sem_scores = kwargs.get("clause_semantic_scores") or []
        clauses = self.segment_prompt_clauses(prompt or "", ctx=ctx)
        n_clauses = max(len(clauses), len(clause_sem_scores)) if (clauses or clause_sem_scores) else 1

        src_file_literals, dest_file_literals = self.extract_file_literals(prompt or "", ctx=ctx)
        has_egress = bool(dest_file_literals) or _clause_has_egress_intent(prompt or "")
        has_compat_src_in_pool = bool(
            src_file_literals
            and any(
                (is_ingress_source(c, self.orchestrator) or not _cell_required_data_inputs(c))
                and any(is_path_port(p) for p in (getattr(c, "inputs", {}) or {}).values())
                and self.file_compat(c, src_file_literals[0]) >= 0
                for c in pool
            )
        )

        # Build clause coverage sets and lexical priors per candidate cell using
        # both normalized semantic scores and corpus-derived identity evidence.
        cell_lex_score: Dict[str, float] = {}
        for c in pool:
            cid = c.cell_id
            split_id = set(CellTokenizer.tokenize_identifier(cid))
            id_toks = set(getattr(c, "identity_tokens", None) or ()) | split_id
            ov = len(prompt_toks & id_toks)
            split_ov = len(prompt_toks & split_id)
            id_prec = split_ov / max(1, len(split_id))
            tok_rec = ov / max(1, len(prompt_toks))
            all_ov = len(prompt_toks & (set(getattr(c, "token_set", None) or ()) | id_toks))
            cell_lex_score[cid] = (
                0.55 * float(relevance_map.get(cid, 0.0))
                + 0.55 * id_prec
                + 0.40 * tok_rec
                + 0.02 * all_ov
            )

        cell_out_ports_map: Dict[str, List[Tuple[Any, Any]]] = {
            c.cell_id: _cell_outputs_ports(c) for c in pool
        }
        cell_outs_map: Dict[str, List[Any]] = {
            cid: [osig for _, osig in ops] for cid, ops in cell_out_ports_map.items()
        }
        cell_req_ins_map: Dict[str, List[Tuple[str, Any]]] = {
            c.cell_id: _cell_required_data_inputs(c) for c in pool
        }
        is_sink_map: Dict[str, bool] = {
            c.cell_id: bool(is_terminal_sink(c, self.orchestrator)) for c in pool
        }
        is_ingress_map: Dict[str, bool] = {
            c.cell_id: bool(is_ingress_source(c, self.orchestrator)) for c in pool
        }
        cand_needs_path_map: Dict[str, bool] = {
            c.cell_id: any(is_path_port(p) for p in (getattr(c, "inputs", {}) or {}).values())
            for c in pool
        }
        cell_domain_map: Dict[str, str] = {
            c.cell_id: str(getattr(c, "domain_name", "") or getattr(c, "domain", "") or "").strip().lower()
            for c in pool
        }
        cell_is_comb_map: Dict[str, bool] = {
            c.cell_id: (
                str(getattr(c, "node_role", "") or "").strip().lower() == "combinator"
                or any(
                    str(getattr(p, "port_role", "") or "").strip().lower() == "functional_operator"
                    for p in (getattr(c, "inputs", {}) or {}).values()
                )
            )
            for c in pool
        }
        cell_sp_id_map: Dict[str, Set[str]] = {
            c.cell_id: set(CellTokenizer.tokenize_identifier(c.cell_id)) for c in pool
        }
        cell_id_toks_map: Dict[str, Set[str]] = {
            c.cell_id: (set(getattr(c, "identity_tokens", None) or ()) | cell_sp_id_map[c.cell_id])
            for c in pool
        }

        try:
            try:
                from ..token_evidence import get_evidence, anchors_for_clause
            except (ImportError, ValueError):
                from token_evidence import get_evidence, anchors_for_clause
            ev = get_evidence(self.orchestrator)
        except Exception:
            ev = None
            anchors_for_clause = None

        TargetId = Tuple[int, int]
        cell_clause_hits: Dict[str, Set[TargetId]] = {c.cell_id: set() for c in pool}
        clause_cell_strength: Dict[Tuple[str, TargetId], float] = {}
        clause_top_cells: Dict[TargetId, Set[str]] = {}
        for cl_i in range(n_clauses):
            cl_txt = clauses[cl_i] if cl_i < len(clauses) else ""
            cl_dict = clause_sem_scores[cl_i] if cl_i < len(clause_sem_scores) else {}
            sem_vals = sorted((float(v or 0.0) for v in cl_dict.values()), reverse=True)
            max_sem = sem_vals[0] if sem_vals else 0.0
            med_sem = sem_vals[len(sem_vals) // 2] if sem_vals else 0.0
            sem_spread = max(0.0, max_sem - med_sem)
            top5_sem_floor = sem_vals[min(4, len(sem_vals) - 1)] if sem_vals else max_sem
            cf_scores: Dict[str, Tuple[float, float, float]] = {}
            best_cf: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            best_id_prec = 0.0
            cl_pool: List[Cell] = []
            for c in pool:
                if cl_i == 0 and n_clauses > 1 and has_compat_src_in_pool and is_ingress_source(c, self.orchestrator):
                    c_needs_path = any(is_path_port(p) for p in (getattr(c, "inputs", {}) or {}).values())
                    if c_needs_path and self.file_compat(c, src_file_literals[0]) < 0:
                        continue
                if cl_i == n_clauses - 1 and is_terminal_sink(c, self.orchestrator) and not has_egress:
                    c_needs_path = any(is_path_port(p) for p in (getattr(c, "inputs", {}) or {}).values())
                    if c_needs_path:
                        continue
                cl_pool.append(c)
                cf = self.clause_fit(cl_txt, c) if cl_txt else (0.0, 0.0, 0.0)
                cf_scores[c.cell_id] = cf
                if cf > best_cf:
                    best_cf = cf
                if cf[1] > best_id_prec:
                    best_id_prec = cf[1]

            sub_anchors: List[Tuple[Cell, Dict[str, Any]]] = []
            if ev is not None and anchors_for_clause is not None and cl_txt and cl_pool:
                raw_ancs = anchors_for_clause(ev, cl_txt, cl_pool, relevance_map, clause_idx=cl_i)
                if raw_ancs:
                    sub_anchors.append(raw_ancs[0])
                    root_toks = {str(t).lower() for t in (raw_ancs[0][1].get("tokens", ()) or ())}
                    for a_c, a_o in raw_ancs[1:]:
                        # Split compound clauses on operational anchors, macro/combinators,
                        # or distinct-token boundary initializers/sinks (ingress/terminal/producers).
                        a_toks = {str(t).lower() for t in (a_o.get("tokens", ()) or ())}
                        has_distinct_tok = bool(a_toks - root_toks)
                        is_macro_or_comb = (
                            str(getattr(a_c, "node_type", "") or "").strip().lower() == "macro"
                            or bool(getattr(a_c, "slots", None))
                            or str(getattr(a_c, "node_role", "") or "").strip().lower() == "combinator"
                        )
                        is_boundary_op = (
                            is_terminal_sink(a_c, self.orchestrator)
                            or is_ingress_source(a_c, self.orchestrator)
                            or bool(getattr(a_c, "outputs", None))
                        )
                        if cell_req_ins_map.get(a_c.cell_id) or is_macro_or_comb or (has_distinct_tok and is_boundary_op):
                            sub_anchors.append((a_c, a_o))

            if len(sub_anchors) > 1 and ev is not None:
                cl_toks_all = CellTokenizer.tokenize_prompt(cl_txt)
                cl_tok_seq = [t.lower() for t in CellTokenizer.tokenize_identifier(cl_txt)] if hasattr(CellTokenizer, "tokenize_identifier") else []
                cl_words = [w.strip(".,;:\"'()[]{}").lower() for w in cl_txt.split() if w.strip()]
                def _anchor_pos(anc_pair: Tuple[Cell, Dict[str, Any]]) -> Tuple[int, int, float]:
                    anc_c = anc_pair[0]
                    anc_optic = str(getattr(anc_c, "optic_role", "") or "").strip().lower()
                    matched_raw = {str(m).lower() for m in (anc_pair[1].get("tokens", ()) or ())}
                    id_set = set(CellTokenizer.tokenize_identifier(anc_c.cell_id))
                    core_m = (matched_raw & id_set) or matched_raw
                    positions = [
                        wi for wi, w in enumerate(cl_words)
                        if w in core_m or any(mt in w for mt in core_m if len(mt) >= 3)
                    ]
                    if not positions and cl_tok_seq:
                        positions = [ti for ti, t in enumerate(cl_tok_seq) if t in core_m]
                    return (
                        min(positions) if positions else 10**6,
                        1 if anc_optic == "setter" else 0,
                        -float((anc_pair[1].get("rank") or (0.0,))[0]),
                    )
                sub_anchors.sort(key=_anchor_pos)
                for sub_j, (anc_c, anc_o) in enumerate(sub_anchors):
                    t_id: TargetId = (cl_i, sub_j)
                    anc_matched = set(anc_o.get("tokens", ()) or ())
                    anc_rank = anc_o.get("rank", (0.0, 0.0, 0.0))
                    top_cids: Set[str] = set()
                    for c in cl_pool:
                        cid = c.cell_id
                        ov = ev.identity_overlap(cl_toks_all, c)
                        c_matched = set(ov.get("tokens", ()) or ())
                        ov_rec = float((ov.get("rank") or (0.0, 0.0, 0.0))[0] or 0.0)
                        if cid == anc_c.cell_id or (
                            anc_matched
                            and (c_matched & anc_matched)
                            and ov_rec >= 0.70 * anc_rank[0]
                        ):
                            cell_clause_hits[cid].add(t_id)
                            cf_rel = min(1.0, ov_rec / anc_rank[0]) if anc_rank[0] > 0.0 else 1.0
                            sem_sc = float(cl_dict.get(cid, 0.0) or 0.0)
                            sem_rel = (sem_sc / max_sem) if max_sem > 0.0 else 0.0
                            clause_cell_strength[(cid, t_id)] = 0.70 * min(1.0, cf_rel) + 0.30 * sem_rel
                            if cid == anc_c.cell_id or cf_rel >= 0.85:
                                top_cids.add(cid)
                    if top_cids:
                        clause_top_cells[t_id] = top_cids
            else:
                t_id = (cl_i, 0)
                top_cids = set()
                for c in cl_pool:
                    cid = c.cell_id
                    cf = cf_scores.get(cid, (0.0, 0.0, 0.0))
                    sem_sc = float(cl_dict.get(cid, 0.0) or 0.0)
                    lex_hit = bool(
                        best_cf[0] > 0.0
                        and (
                            (cf[0] >= 0.85 * best_cf[0] and cf[2] >= 0.45 * best_cf[2])
                            or (cf[0] >= 0.70 * best_cf[0] and (cf[1] >= 0.50 * best_cf[1] or cf[2] >= 0.65 * best_cf[2]))
                            or (cf[0] >= 0.30 * best_cf[0] and best_id_prec > 0.0 and cf[1] >= 0.85 * best_id_prec)
                        )
                    )
                    if best_cf[0] >= 0.70:
                        if cf[0] >= 0.30 * best_cf[0]:
                            sem_hit = bool(
                                max_sem > 0.0
                                and sem_sc >= 0.85 * max_sem
                                and (sem_spread >= 0.10 or sem_sc >= med_sem + 0.65 * sem_spread)
                            )
                        else:
                            # Guard against weak 1-token generic overlaps or compressed cosines when a strong lexical anchor exists
                            sem_hit = bool(
                                cf[0] == 0.0
                                and max_sem > 0.0
                                and sem_spread >= 0.12
                                and sem_sc >= 0.50
                                and sem_sc >= (max_sem - 0.15 * sem_spread)
                            )
                    else:
                        if sem_spread >= 0.08:
                            sem_hit = bool(
                                max_sem > 0.0
                                and sem_sc >= max(0.35, 0.70 * max_sem, med_sem + 0.50 * sem_spread)
                            )
                        else:
                            sem_hit = bool(
                                max_sem > 0.0
                                and sem_sc >= 0.35
                                and sem_sc >= max(top5_sem_floor, med_sem + 0.70 * sem_spread)
                            )
                    if lex_hit or sem_hit:
                        cell_clause_hits[cid].add(t_id)
                        cf_rel = (cf[0] / best_cf[0]) if best_cf[0] > 0.0 else 0.0
                        sem_rel = (sem_sc / max_sem) if max_sem > 0.0 else 0.0
                        strength = 0.70 * cf_rel + 0.30 * sem_rel
                        clause_cell_strength[(cid, t_id)] = strength
                        if cf_rel >= 0.80 or (best_cf[0] == 0.0 and sem_rel >= 0.85):
                            top_cids.add(cid)
                if top_cids:
                    clause_top_cells[t_id] = top_cids
                    self._trace("anchor_candidate", target=t_id, candidates=sorted(top_cids))

        coverable_clauses: Set[TargetId] = set()
        for _cl_hits in cell_clause_hits.values():
            coverable_clauses.update(_cl_hits)

        # Corpus-derived per-clause focus tokens (residual variable/field identifiers
        # outside lattice operation vocabulary) and setter presence.
        known_op_toks: Set[str] = set()
        for c in pool:
            known_op_toks.update(CellTokenizer.tokenize_identifier(c.cell_id))
            for it in (getattr(c, "identity_tokens", None) or ()):
                known_op_toks.add(str(it).lower())
        for fl in (src_file_literals + dest_file_literals):
            for ft in re.findall(r"[A-Za-z0-9_]+", str(fl)):
                known_op_toks.add(ft.lower())

        clause_focus_toks: Dict[int, Tuple[str, ...]] = {}
        clause_has_setter: Dict[int, bool] = {}
        for cl_i in range(n_clauses):
            cl_txt = clauses[cl_i] if cl_i < len(clauses) else ""
            raw_w = re.findall(r"[A-Za-z0-9_]+", cl_txt)
            clause_focus_toks[cl_i] = tuple(sorted({
                w.lower() for w in raw_w if w.lower() not in known_op_toks
            }))
            clause_has_setter[cl_i] = any(
                str(getattr(pool_by_id_tmp, "optic_role", "") or "").strip().lower() == "setter"
                for pool_by_id_tmp in pool
                if any(t[0] == cl_i for t in cell_clause_hits.get(pool_by_id_tmp.cell_id, ()))
            )

        # Identify target goal cells (goal_sig unifying or verified sinks / final-clause anchors)
        goal_ids: Set[str] = set()
        for c in pool:
            if goal_sig is not None:
                for out_sig in _cell_outputs_sigs(c):
                    if unify(out_sig, goal_sig) is not None:
                        goal_ids.add(c.cell_id)
                        break
            elif is_sink_map[c.cell_id] and float(relevance_map.get(c.cell_id, 0.0)) > 0.1:
                c_needs_path = cand_needs_path_map[c.cell_id]
                if c_needs_path and not dest_file_literals and not has_egress:
                    continue
                fc_dest = self.file_compat(c, dest_file_literals[0]) if (dest_file_literals and c_needs_path) else 0
                if dest_file_literals and c_needs_path and fc_dest < 0:
                    continue
                c_split_id = cell_sp_id_map[c.cell_id]
                dom_toks = set(CellTokenizer.tokenize_identifier(cell_domain_map[c.cell_id])) if cell_domain_map[c.cell_id] else set()
                distinctive_toks = (c_split_id - dom_toks) or c_split_id
                sink_id_toks = (cell_id_toks_map[c.cell_id] - dom_toks) or distinctive_toks
                if prompt_toks & (sink_id_toks if (has_egress and not dest_file_literals) else distinctive_toks) or (
                    dest_file_literals and fc_dest >= 1 and (prompt_toks & (cell_id_toks_map[c.cell_id] - dom_toks))
                ):
                    goal_ids.add(c.cell_id)

        if not goal_ids:
            # Final-clause anchors and top-relevance non-file candidates act as soft goals
            if clause_sem_scores:
                last_cl = clause_sem_scores[-1] or {}
                top_last = sorted(
                    [c for c in pool if not (is_terminal_sink(c, self.orchestrator) and not has_egress)],
                    key=lambda c: float(last_cl.get(c.cell_id, 0.0) or 0.0) + 0.5 * cell_lex_score.get(c.cell_id, 0.0),
                    reverse=True,
                )
                for c in top_last[:3]:
                    goal_ids.add(c.cell_id)
            if not goal_ids:
                top_by_rel = sorted(pool, key=lambda c: cell_lex_score.get(c.cell_id, 0.0), reverse=True)
                for c in top_by_rel[: max(2, min(5, len(top_by_rel)))]:
                    goal_ids.add(c.cell_id)

        # Build backward adjacency inside tunnel via cached unification for h_bwd heuristic
        pool_by_id = {c.cell_id: c for c in pool}
        cell_props_map: Dict[str, Dict[str, Any]] = {
            c.cell_id: _cell_declared_props(c, reg) for c in pool
        }
        cell_preconds_map: Dict[str, Dict[str, List[Tuple[str, str, Any]]]] = {
            c.cell_id: _cell_required_preconds(c) for c in pool
        }
        rev_edges: Dict[str, List[str]] = {c.cell_id: [] for c in pool}
        orch_ref = orchestrator or getattr(self, "orchestrator", None)
        eff_edge_mode = str(getattr(orch_ref, "edge_mode", "") or "").strip().lower()
        static_rev_adj = getattr(orch_ref, "_reverse_adjacency", None) if orch_ref is not None else None
        if eff_edge_mode == "static" and isinstance(static_rev_adj, dict):
            for cons in pool:
                if not cell_req_ins_map[cons.cell_id]:
                    continue
                for prod_id in static_rev_adj.get(cons.cell_id, ()):
                    if prod_id != cons.cell_id and prod_id in pool_by_id:
                        rev_edges[cons.cell_id].append(prod_id)
        else:
            for prod in pool:
                p_outs = cell_out_ports_map[prod.cell_id]
                if not p_outs:
                    continue
                for cons in pool:
                    if cons.cell_id == prod.cell_id:
                        continue
                    req_ins = cell_req_ins_map[cons.cell_id]
                    if not req_ins:
                        continue
                    if any(
                        _unify_cached(
                            osig,
                            getattr(ip, "signature", ip),
                            out_port=oport,
                            in_port=ip,
                            prod_cell=prod,
                            cons_cell=cons,
                        )
                        for oport, osig in p_outs
                        for _, ip in req_ins
                    ):
                        rev_edges[cons.cell_id].append(prod.cell_id)

        bwd_dist: Dict[str, int] = {}
        dq = deque()
        for gid in goal_ids:
            bwd_dist[gid] = 0
            dq.append(gid)
        while dq:
            cur = dq.popleft()
            d_cur = bwd_dist[cur]
            for prev_id in rev_edges.get(cur, ()):
                if prev_id not in bwd_dist:
                    bwd_dist[prev_id] = d_cur + 1
                    dq.append(prev_id)

        # Constrained SLM choice scorer lookup
        try:
            try:
                from ..inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            mm = ModelManager.get_instance()
        except Exception:
            mm = None

        use_choice_scorer = bool(
            mm is not None and getattr(mm, "has_constrained_choice_scorer", lambda: False)()
        )
        scorer_calls = 0
        max_scorer_calls = 4

        def _heuristic(
            last_cid: Optional[str],
            covered_clauses: Tuple[TargetId, ...],
            unlock_credit: float = 0.0,
            gamma_state: Optional[Tuple[Any, ...]] = None,
        ) -> float:
            target_cl = max(len(coverable_clauses), n_clauses) if coverable_clauses else n_clauses
            rem_clauses = max(0.0, float(target_cl - len(covered_clauses)) - unlock_credit) if target_cl > 1 else 0.0
            near_done = len(covered_clauses) >= max(0, target_cl - 1)
            if near_done and last_cid and last_cid in goal_ids:
                h_goal = 0.0
            else:
                bwd_vals: List[int] = []
                if last_cid and last_cid in bwd_dist:
                    if not (not near_done and last_cid in goal_ids):
                        bwd_vals.append(bwd_dist[last_cid])
                if gamma_state:
                    for gs in gamma_state:
                        if isinstance(gs, tuple) and len(gs) == 3 and gs[2] is not None:
                            g_cid = gs[2].cell_id
                            if g_cid in bwd_dist and not (not near_done and g_cid in goal_ids):
                                bwd_vals.append(bwd_dist[g_cid])
                h_goal = (0.35 * float(min(bwd_vals))) if bwd_vals else (0.35 if near_done else 1.5)
            # Encourage covering all coverable sub-targets cleanly
            return rem_clauses * 0.85 + h_goal

        init_sigs: List[Any] = []
        ext_gamma = kwargs.get("initial_gamma")
        if ext_gamma:
            for gs in ext_gamma:
                if gs is not None:
                    init_sigs.append(gs)
        if start_sig is not None:
            if isinstance(start_sig, (tuple, list)):
                init_sigs.extend([s for s in start_sig if s is not None])
            else:
                init_sigs.append(start_sig)
        init_gamma: Tuple[Any, ...] = tuple(
            gs if (isinstance(gs, tuple) and len(gs) == 3) else (gs, None, None)
            for gs in init_sigs
        )
        init_sigma = Substitution()

        # Precompute setter value signatures so we know which transforms modify an open lens
        setter_val_sigs: List[Any] = []
        for c in pool:
            if str(getattr(c, "optic_role", "") or "").strip().lower() == "setter":
                kp = str(getattr(c, "key_port", "") or "").strip()
                for pk, p in (getattr(c, "inputs", {}) or {}).items():
                    if pk != kp and not is_path_port(p) and _port_requires_carrier(c, p):
                        psig = getattr(p, "signature", p)
                        if psig is not None:
                            setter_val_sigs.append(psig)

        # Precompute pool-local required-input producer adjacency for k-hop backward precondition cone
        pool_producers_for: Dict[str, List[Cell]] = {c.cell_id: [] for c in pool}
        for tc in pool:
            tc_req = cell_req_ins_map[tc.cell_id]
            if not tc_req:
                continue
            for pc in pool:
                if pc.cell_id == tc.cell_id or is_sink_map[pc.cell_id]:
                    continue
                pc_outs = cell_out_ports_map[pc.cell_id]
                if pc_outs and any(
                    _unify_cached(
                        osig,
                        getattr(ip, "signature", ip),
                        out_port=oport,
                        in_port=ip,
                        prod_cell=pc,
                        cons_cell=tc,
                    )
                    for oport, osig in pc_outs
                    for _, ip in tc_req
                ):
                    pool_producers_for[tc.cell_id].append(pc)

        # Priority queue items:
        # (f_score, neg_evidence, counter, g_cost, chain_ids, gamma, sigma,
        #  covered_clauses, active_props, open_lens, active_focus, focus_modified,
        #  step_clauses, inv_count, max_cl_seen)
        pq: List[Tuple[
            float, float, int, float, Tuple[str, ...], Tuple[Any, ...], Any,
            Tuple[TargetId, ...], Tuple[Tuple[str, Any], ...], bool,
            Tuple[str, ...], bool, Tuple[Optional[int], ...], int, int
        ]] = []
        counter = 0
        heapq.heappush(
            pq,
            (
                _heuristic(None, ()),
                0.0,
                counter,
                0.0,
                (),
                init_gamma,
                init_sigma,
                (),
                (),
                False,
                (),
                False,
                (),
                0,
                -1,
            ),
        )

        best_full_complete: Optional[Tuple[float, Tuple[str, ...], Tuple[Optional[int], ...]]] = None
        best_complete: Optional[Tuple[float, Tuple[str, ...], Tuple[Optional[int], ...]]] = None
        best_partial: Optional[Tuple[float, Tuple[str, ...], Tuple[Optional[int], ...]]] = None
        visited_states: Set[Tuple[Tuple[str, ...], Tuple[TargetId, ...], Tuple[str, ...]]] = set()

        max_expansions = int(kwargs.get("max_expansions", max(900, n_clauses * 240)))
        max_chain_len = int(kwargs.get("max_chain_len", 18))
        branch_factor = int(kwargs.get("branch_factor", 8))
        expansions = 0
        valid_frontier_sizes: List[int] = []

        while pq and expansions < max_expansions:
            (
                f_sc, neg_ev, _, g_cost, chain_ids, gamma, sigma,
                cov_cl, active_props, open_lens, active_focus, focus_modified,
                step_clauses, inv_count, max_cl_seen,
            ) = heapq.heappop(pq)
            expansions += 1

            state_key = (chain_ids, cov_cl, active_focus)
            if chain_ids and state_key in visited_states:
                continue
            visited_states.add(state_key)

            # Evaluate candidate completeness / quality using marginal coverage + causal optic fidelity
            if chain_ids:
                chain_cells = [pool_by_id[cid] for cid in chain_ids]
                last_c = chain_cells[-1]
                reaches_goal = (last_c.cell_id in goal_ids)
                if goal_sig is not None:
                    reaches_goal = any(
                        _unify_cached(osig, goal_sig) for osig in cell_outs_map[last_c.cell_id]
                    )
                seen_toks: Set[str] = set()
                step_deltas: List[int] = []
                uniq_id_prec: Dict[str, float] = {}
                non_optic_steps = 0
                domain_switches = 0
                prev_dom = ""
                acc_optics = set(TypeRegistry.get_instance().get_verification_semantics("accessor_optic_roles") or ())
                for c_item in chain_cells:
                    c_or = str(getattr(c_item, "optic_role", "") or "").strip().lower()
                    if c_or not in acc_optics:
                        non_optic_steps += 1
                    c_dom = cell_domain_map[c_item.cell_id]
                    if prev_dom and c_dom and c_dom != prev_dom:
                        domain_switches += 1
                    if c_dom:
                        prev_dom = c_dom
                    sp_id = cell_sp_id_map[c_item.cell_id]
                    i_toks = cell_id_toks_map[c_item.cell_id]
                    matched = prompt_toks & i_toks
                    step_deltas.append(len(matched - seen_toks))
                    seen_toks |= matched
                    if c_item.cell_id not in uniq_id_prec:
                        uniq_id_prec[c_item.cell_id] = len(prompt_toks & sp_id) / max(1, len(sp_id))

                last_useful_idx = -1
                for idx_s, d_tok in enumerate(step_deltas):
                    if d_tok > 0:
                        last_useful_idx = idx_s
                n_steps = len(chain_cells)
                trailing_wasted = (n_steps - 1 - last_useful_idx) if last_useful_idx >= 0 else n_steps

                tok_frac = len(seen_toks) / max(1, len(prompt_toks))
                mean_gain = (-neg_ev) / max(1, non_optic_steps)
                mean_id_prec = sum(uniq_id_prec.values()) / max(1, len(uniq_id_prec))
                n_cov_target = len(coverable_clauses) if len(coverable_clauses) > 1 else n_clauses
                clause_frac = (len(cov_cl) / max(1, n_cov_target)) if n_cov_target > 1 else 0.0
                goal_bonus = (
                    0.65
                    if (reaches_goal and (step_deltas[-1] > 0 or goal_sig is not None or n_clauses > 1))
                    else 0.0
                )
                lens_closure_term = -0.35 if (open_lens and focus_modified) else 0.15
                quality = (
                    1.45 * tok_frac
                    + 0.40 * mean_gain
                    + 0.50 * mean_id_prec
                    + 2.85 * clause_frac
                    + goal_bonus
                    + lens_closure_term
                    - 0.45 * inv_count
                    - 0.04 * max(0, non_optic_steps - 1)
                    - 0.25 * domain_switches
                    - 0.60 * trailing_wasted
                )

                if best_partial is None or quality > best_partial[0]:
                    best_partial = (quality, chain_ids, step_clauses)
                if n_cov_target <= 1 or len(cov_cl) >= n_cov_target:
                    full_score = quality + (0.6 if reaches_goal else 0.0)
                    if best_full_complete is None or full_score > best_full_complete[0]:
                        best_full_complete = (full_score, chain_ids, step_clauses)
                if reaches_goal and (n_clauses <= 1 or len(cov_cl) >= max(1, n_clauses // 2)):
                    if best_complete is None or quality > best_complete[0]:
                        best_complete = (quality, chain_ids, step_clauses)

            if len(chain_ids) >= max_chain_len:
                continue

            # Exact Stage 3 Valid Outgoing Morphisms E_valid(Gamma_t)
            used_set = set(chain_ids)
            cov_cl_set = set(cov_cl)
            last_cid = chain_ids[-1] if chain_ids else None
            last_dom = cell_domain_map[last_cid] if last_cid else ""
            uncovered_cl = coverable_clauses - cov_cl_set
            next_target = min(uncovered_cl) if uncovered_cl else None
            active_prop_dict = dict(active_props)
            num_sources_used = sum(
                1 for cid in chain_ids
                if is_ingress_map[cid] or (not cell_req_ins_map[cid] and cand_needs_path_map[cid])
            )

            # Evaluate firing feasibility once per pool cell for this state
            can_fire_map: Dict[str, Optional[Any]] = {
                c.cell_id: _can_fire_under_gamma(
                    c,
                    gamma,
                    sigma,
                    unify_cache=unify_cache,
                    edge_cache=edge_cache,
                    reg=reg,
                    req_ins=cell_req_ins_map[c.cell_id],
                    sig_key_cache=sig_key_cache,
                    tv_cache=tv_cache,
                    unify_empty_sub_cache=unify_empty_sub_cache,
                )
                for c in pool
            }

            # Precompute unused high-value cells currently blocked, and expand up to 2 hops backward
            # along missing required inputs so multi-step initializers (KEY_GEN -> INIT -> ENCRYPT) unlock.
            blocked_targets = [
                tc for tc in pool
                if tc.cell_id not in used_set
                and cell_req_ins_map[tc.cell_id]
                and can_fire_map[tc.cell_id] is None
                and (cell_lex_score.get(tc.cell_id, 0.0) >= 0.45 or bool(cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set))
            ]
            blocked_missing_ins: Dict[str, List[Any]] = {}
            unlock_depth: Dict[str, int] = {tc.cell_id: 0 for tc in blocked_targets}
            unlock_clause_hits: Dict[str, Set[TargetId]] = {
                tc.cell_id: (cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set)
                for tc in blocked_targets
            }

            def _missing_req_ins_for(t_cell: Cell) -> List[Any]:
                cid_t = t_cell.cell_id
                cached_m = blocked_missing_ins.get(cid_t)
                if cached_m is not None:
                    return cached_m
                miss = [
                    ip for _, ip in cell_req_ins_map[cid_t]
                    if not any(
                        _unify_cached(
                            gs[0] if isinstance(gs, tuple) else gs,
                            getattr(ip, "signature", ip),
                            out_port=gs[1] if isinstance(gs, tuple) else None,
                            in_port=ip,
                            prod_cell=gs[2] if isinstance(gs, tuple) else None,
                            cons_cell=t_cell,
                        )
                        for gs in gamma
                    )
                ]
                blocked_missing_ins[cid_t] = miss
                return miss

            frontier_bt = list(blocked_targets)
            for depth_hop in (1, 2):
                next_frontier_bt: List[Cell] = []
                for tc in frontier_bt:
                    missing_ins = _missing_req_ins_for(tc)
                    if not missing_ins:
                        continue
                    tc_prop_hits = unlock_clause_hits.get(tc.cell_id, set())
                    for pc in pool_producers_for.get(tc.cell_id, ()):
                        if pc.cell_id in used_set or can_fire_map.get(pc.cell_id) is not None:
                            continue
                        if any(
                            _unify_cached(
                                osig,
                                getattr(ip, "signature", ip),
                                out_port=oport,
                                in_port=ip,
                                prod_cell=pc,
                                cons_cell=tc,
                            )
                            for oport, osig in cell_out_ports_map[pc.cell_id]
                            for ip in missing_ins
                        ):
                            unlock_clause_hits[pc.cell_id] = (
                                unlock_clause_hits.get(pc.cell_id, set())
                                | tc_prop_hits
                                | (cell_clause_hits.get(pc.cell_id, set()) - cov_cl_set)
                            )
                            if pc.cell_id not in unlock_depth:
                                unlock_depth[pc.cell_id] = depth_hop
                                _missing_req_ins_for(pc)
                                next_frontier_bt.append(pc)
                frontier_bt = next_frontier_bt
                if not frontier_bt:
                    break
            blocked_chain_targets = list(blocked_targets) + [
                pool_by_id[tid] for tid, dp in unlock_depth.items()
                if dp > 0 and tid in pool_by_id
            ]

            # Identify any pending property preconditions on uncovered/goal/near-goal targets violated by active_prop_dict
            needed_prop_repairs: Dict[str, Any] = {}
            for tc in pool:
                if tc.cell_id in used_set:
                    continue
                if (
                    not (cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set)
                    and tc.cell_id not in goal_ids
                    and bwd_dist.get(tc.cell_id, 99) > 2
                ):
                    continue
                for pre_list in cell_preconds_map[tc.cell_id].values():
                    for p_name, p_op, p_want in pre_list:
                        if p_name in active_prop_dict:
                            cur_v = active_prop_dict[p_name]
                            ok = (cur_v == p_want) if p_op == "==" else (cur_v != p_want)
                            if not ok and p_op == "==":
                                needed_prop_repairs[p_name] = p_want

            # Check if next_target has an anchor already fireable or 1-hop unlockable
            next_uncl_fireable = False
            next_uncl_unlockable = False
            next_cl_focus: Tuple[str, ...] = (
                clause_focus_toks.get(next_target[0], ())
                if next_target is not None
                else ()
            )
            focus_switch_needed = bool(
                next_target is not None
                and active_focus
                and next_cl_focus
                and set(next_cl_focus).isdisjoint(active_focus)
                and not clause_has_setter.get(next_target[0], False)
            )
            next_target_anchors: List[Cell] = []
            if next_target is not None:
                for tc in pool:
                    if tc.cell_id in used_set:
                        continue
                    tc_new = cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set
                    if next_target in tc_new:
                        next_target_anchors.append(tc)
                        if can_fire_map[tc.cell_id] is not None:
                            next_uncl_fireable = True
                        else:
                            next_uncl_unlockable = True

            # Check whether an optic getter in E_valid can re-project the carrier for focus_switch_needed
            getter_unlocks_next_focus = False
            if focus_switch_needed and next_target_anchors:
                for gc in pool:
                    if str(getattr(gc, "optic_role", "") or "").strip().lower() != "getter":
                        continue
                    if can_fire_map.get(gc.cell_id) is None or gc.cell_id == last_cid:
                        continue
                    g_outs = cell_out_ports_map[gc.cell_id]
                    if any(
                        _unify_cached(
                            osig,
                            getattr(ip, "signature", ip),
                            out_port=oport,
                            in_port=ip,
                            prod_cell=gc,
                            cons_cell=tc,
                        )
                        for oport, osig in g_outs
                        for tc in next_target_anchors
                        for _, ip in cell_req_ins_map[tc.cell_id]
                    ):
                        getter_unlocks_next_focus = True
                        break

            e_valid: List[Tuple[Cell, Any, float, float, float, Optional[int]]] = []
            rep_optics = set(TypeRegistry.get_instance().get_verification_semantics("repeatable_optic_roles") or ())
            for cand in pool:
                optic_r = str(getattr(cand, "optic_role", "") or "").strip().lower()
                is_repeatable_optic = optic_r in rep_optics
                cand_new_cl_reuse = cell_clause_hits.get(cand.cell_id, set()) - cov_cl_set
                if cand.cell_id in used_set:
                    if (
                        (not is_repeatable_optic and not cand_new_cl_reuse)
                        or cand.cell_id == last_cid
                        or chain_ids.count(cand.cell_id) >= max(3, n_clauses)
                    ):
                        continue
                # Suppress file-writing sinks when prompt has no egress intent
                if (
                    is_sink_map[cand.cell_id]
                    and not has_egress
                    and cand_needs_path_map[cand.cell_id]
                ):
                    continue

                next_sigma = can_fire_map[cand.cell_id]
                if next_sigma is None:
                    continue

                cand_hits = cell_clause_hits.get(cand.cell_id, set())
                new_hits = cand_hits - cov_cl_set
                _is_comb = cell_is_comb_map[cand.cell_id]
                is_closing_open_setter = bool(
                    open_lens
                    and optic_r == "setter"
                    and (not new_hits or (next_target is not None and min(new_hits) > next_target))
                )
                eff_new_hits = set() if is_closing_open_setter else new_hits
                if _is_comb and not eff_new_hits:
                    continue
                prior_sc = cell_lex_score.get(cand.cell_id, 0.0)
                cand_dom = cell_domain_map[cand.cell_id]
                if last_dom and cand_dom == last_dom and (eff_new_hits or cand.cell_id in goal_ids):
                    prior_sc += 0.25
                unlock_credit = 0.0
                order_cost_penalty = 0.0
                step_cl_idx: Optional[int] = min(t[0] for t in eff_new_hits) if eff_new_hits else (
                    max(0, max_cl_seen) if is_closing_open_setter else (next_target[0] if next_target is not None else None)
                )

                next_uncl_hits = [t for t in eff_new_hits if next_target is not None and t == next_target]
                if next_uncl_hits:
                    cl_str = max(clause_cell_strength.get((cand.cell_id, t), 0.5) for t in next_uncl_hits)
                    prior_sc += 0.75 + 0.90 * cl_str
                    if getter_unlocks_next_focus and optic_r != "getter":
                        prior_sc -= 1.05
                        order_cost_penalty += 0.85
                elif eff_new_hits:
                    best_nh = max(clause_cell_strength.get((cand.cell_id, nh), 0.3) for nh in eff_new_hits)
                    prior_sc += 0.20 * len(eff_new_hits) + 0.35 * best_nh
                    if (
                        next_target is not None
                        and min(eff_new_hits) > next_target
                        and (next_uncl_fireable or next_uncl_unlockable)
                    ):
                        prior_sc -= 0.95
                        order_cost_penalty += 0.90

                if cand.cell_id in bwd_dist:
                    prior_sc += 0.20 / (1.0 + bwd_dist[cand.cell_id])
                cand_needs_path = cand_needs_path_map[cand.cell_id]
                is_zero_in_reader = is_ingress_map[cand.cell_id] or (
                    not cell_req_ins_map[cand.cell_id] and cand_needs_path
                )
                if not chain_ids and is_zero_in_reader:
                    if has_compat_src_in_pool:
                        if cand_needs_path and self.file_compat(cand, src_file_literals[0]) < 0:
                            continue
                    prior_sc += 0.45
                elif chain_ids and not cell_req_ins_map[cand.cell_id]:
                    # Zero-input sources/generators mid-chain are valid when either:
                    # (a) an unmatched external source file remains in the prompt, OR
                    # (b) the non-path constructor/generator produces a carrier needed by a blocked/uncovered downstream cell.
                    has_more_files = bool(
                        is_zero_in_reader
                        and cand_needs_path
                        and len(src_file_literals) > num_sources_used
                        and (not src_file_literals or self.file_compat(cand, src_file_literals[min(num_sources_used, len(src_file_literals) - 1)]) >= 0)
                    )
                    unblocks_downstream = False
                    if not cand_needs_path and not has_more_files:
                        if _is_comb:
                            unblocks_downstream = bool(eff_new_hits)
                        else:
                            cand_outs = cell_out_ports_map.get(cand.cell_id, ())
                            unblocks_downstream = (bool(eff_new_hits) and not is_ingress_map[cand.cell_id]) or any(
                                (
                                    unlock_depth.get(tc.cell_id, 0) > 0
                                    or (
                                        len(cell_req_ins_map.get(tc.cell_id, ())) >= 2
                                        and any(
                                            _unify_cached(
                                                gs[0] if (isinstance(gs, tuple) and len(gs) == 3) else gs,
                                                getattr(other_ip, "signature", other_ip),
                                                out_port=(gs[1] if (isinstance(gs, tuple) and len(gs) == 3) else None),
                                                in_port=other_ip,
                                                prod_cell=(gs[2] if (isinstance(gs, tuple) and len(gs) == 3) else None),
                                                cons_cell=tc,
                                            )
                                            for gs in gamma
                                            for ok_name, other_ip in cell_req_ins_map.get(tc.cell_id, ())
                                            if ok_name != tk_name
                                        )
                                    )
                                )
                                and _unify_cached(
                                    osig,
                                    getattr(ip, "signature", ip),
                                    out_port=oport,
                                    in_port=ip,
                                    prod_cell=cand,
                                    cons_cell=tc,
                                )
                                for tc in blocked_chain_targets
                                for oport, osig in cand_outs
                                for tk_name, ip in cell_req_ins_map.get(tc.cell_id, ())
                            )
                    if not has_more_files and not unblocks_downstream:
                        continue
                    if is_ingress_map[cand.cell_id] and not cand_needs_path and not eff_new_hits:
                        # Disallow zero-hit synthetic generators mid-chain when disconnected from active gamma
                        cand_outs_chk = cell_out_ports_map.get(cand.cell_id, ())
                        connects_gamma = False
                        for tc in blocked_chain_targets:
                            tc_req = cell_req_ins_map[tc.cell_id]
                            if not any(
                                _unify_cached(
                                    osig,
                                    getattr(ip, "signature", ip),
                                    out_port=oport,
                                    in_port=ip,
                                    prod_cell=cand,
                                    cons_cell=tc,
                                )
                                for oport, osig in cand_outs_chk
                                for _, ip in tc_req
                            ):
                                continue
                            if any(
                                isinstance(gs, tuple) and len(gs) == 3
                                and _unify_cached(
                                    gs[0],
                                    getattr(ip, "signature", ip),
                                    out_port=gs[1],
                                    in_port=ip,
                                    prod_cell=gs[2],
                                    cons_cell=tc,
                                )
                                for gs in gamma
                                for _, ip in tc_req
                            ):
                                connects_gamma = True
                                break
                        if not connects_gamma:
                            continue

                # Instantiate generic output signatures using next_sigma (only when mappings exist)
                raw_out_pairs = cell_out_ports_map[cand.cell_id]
                if getattr(next_sigma, "mappings", None):
                    cand_out_pairs = [
                        (oport, substitute_generics(osig, next_sigma))
                        for oport, osig in raw_out_pairs
                    ]
                else:
                    cand_out_pairs = raw_out_pairs

                # Causal continuity: reward consuming the immediately preceding cell's live output,
                # and penalize terminal sinks that bypass the latest non-sink transform (dead tail).
                if chain_ids and last_cid:
                    last_gamma_items = [
                        gs for gs in gamma
                        if isinstance(gs, tuple) and len(gs) == 3 and gs[2] is not None and gs[2].cell_id == last_cid
                    ]
                    cand_req_ins = cell_req_ins_map[cand.cell_id]
                    consumes_last = bool(
                        last_gamma_items
                        and cand_req_ins
                        and any(
                            _unify_cached(
                                gs[0],
                                getattr(ip, "signature", ip),
                                out_port=gs[1],
                                in_port=ip,
                                prod_cell=gs[2],
                                cons_cell=cand,
                            )
                            for gs in last_gamma_items
                            for _, ip in cand_req_ins
                        )
                    )
                    if consumes_last:
                        prior_sc += 0.35
                    elif (
                        last_gamma_items
                        and cand_req_ins
                        and is_sink_map[cand.cell_id]
                        and not is_sink_map[pool_by_id[last_cid].cell_id]
                    ):
                        prior_sc -= 0.90
                        order_cost_penalty += 0.75

                # Bridge/optic & multi-hop precondition unlock bonus: reward carriers satisfying
                # missing inputs of blocked targets (or blocked prerequisite initializers).
                best_unlock = 0.0
                unlock_targets = list(blocked_chain_targets)
                if getter_unlocks_next_focus and optic_r == "getter":
                    for tc in next_target_anchors:
                        if tc not in unlock_targets:
                            unlock_targets.append(tc)
                if cand_out_pairs and unlock_targets:
                    for tc in unlock_targets:
                        tc_ins = blocked_missing_ins.get(tc.cell_id)
                        if tc_ins is None:
                            tc_ins = [ip for _, ip in cell_req_ins_map[tc.cell_id]]
                        if tc_ins and any(
                            _unify_cached(
                                osig,
                                getattr(ip, "signature", ip),
                                out_port=oport,
                                in_port=ip,
                                prod_cell=cand,
                                cons_cell=tc,
                            )
                            for oport, osig in cand_out_pairs
                            for ip in tc_ins
                        ):
                            depth_decay = 0.78 ** unlock_depth.get(tc.cell_id, 0)
                            tc_new = unlock_clause_hits.get(
                                tc.cell_id,
                                cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set,
                            )
                            tc_next_hits = [t for t in tc_new if next_target is not None and t == next_target]
                            if tc_next_hits and (not next_uncl_fireable or (getter_unlocks_next_focus and optic_r == "getter")):
                                tc_str = max(clause_cell_strength.get((tc.cell_id, t), 0.5) for t in tc_next_hits)
                                uv = depth_decay * (0.70 + 0.90 * tc_str + 0.25 * cell_lex_score.get(tc.cell_id, 0.0))
                                if optic_r == "getter":
                                    uv += 0.45
                                unlock_credit = max(unlock_credit, 0.65)
                            elif tc_next_hits:
                                tc_str = max(clause_cell_strength.get((tc.cell_id, t), 0.5) for t in tc_next_hits)
                                uv = depth_decay * (0.20 * tc_str + 0.15 * cell_lex_score.get(tc.cell_id, 0.0))
                            else:
                                uv = depth_decay * (0.35 * cell_lex_score.get(tc.cell_id, 0.0) + 0.20 * len(tc_new))
                                if optic_r == "getter":
                                    uv += 0.20
                            if uv > best_unlock:
                                best_unlock = uv
                    prior_sc += min(1.85 if optic_r == "getter" else 1.45, best_unlock)
                    # Only clear order penalty for pure bridge/optic carriers (not future-clause operators)
                    if unlock_credit > 0.0 and order_cost_penalty > 0.0 and not new_hits:
                        order_cost_penalty = 0.0

                # 5-Axis Bridge Profile cost-tiering & lossy guard
                bp = getattr(cand, "bridge_profile", None)
                if isinstance(bp, dict) and bp.get("is_bridge"):
                    if bp.get("lossy") and not eff_new_hits and not is_sink_map[cand.cell_id]:
                        continue
                    tier = float(bp.get("cost_tier", 1.0) or 1.0)
                    if not eff_new_hits:
                        prior_sc -= min(1.5, 0.12 * tier)
                        order_cost_penalty += min(2.0, 0.08 * tier)

                # Property precondition / postcondition repair awareness (e.g. is_complex == False)
                cand_pre = cell_preconds_map[cand.cell_id]
                cand_violates_active = False
                for pre_list in cand_pre.values():
                    for p_name, p_op, p_want in pre_list:
                        if p_name in active_prop_dict:
                            cur_v = active_prop_dict[p_name]
                            ok = (cur_v == p_want) if p_op == "==" else (cur_v != p_want)
                            if not ok:
                                cand_violates_active = True
                                break
                if cand_violates_active:
                    prior_sc -= 1.40
                    order_cost_penalty += 1.00

                cand_props = cell_props_map[cand.cell_id]
                prop_repair_bonus = 0.0
                if needed_prop_repairs and cand_props:
                    for rp_name, rp_want in needed_prop_repairs.items():
                        if cand_props.get(rp_name) == rp_want:
                            prop_repair_bonus += 1.55
                            unlock_credit = max(unlock_credit, 0.65)
                    prior_sc += prop_repair_bonus

                # Optic lens closure: reward setter closing an open modified focus (after repairs)
                if needed_prop_repairs and prop_repair_bonus <= 0.0 and optic_r == "setter":
                    prior_sc -= 0.95
                    order_cost_penalty += 0.75
                elif open_lens and optic_r == "setter":
                    prior_sc += 1.85 if focus_modified else 0.65
                    unlock_credit = max(unlock_credit, 0.70 if focus_modified else 0.35)
                elif open_lens and focus_modified and not prop_repair_bonus and next_target is not None and (
                    not new_hits or min(t[0] for t in new_hits) > max(0, max_cl_seen)
                ):
                    prior_sc -= 1.15
                    order_cost_penalty += 0.85

                # Do not materialize file-path sinks when the prompt declares neither a destination file nor explicit egress hits
                if is_sink_map[cand.cell_id] and cand_needs_path and not dest_file_literals and not (has_egress and eff_new_hits):
                    continue

                # Restrict valid frontier to uncovered prompt-relevant cells, optic/property bridges, or unlockers
                if (
                    not eff_new_hits
                    and best_unlock <= 0.0
                    and prop_repair_bonus <= 0.0
                    and not (open_lens and optic_r == "setter")
                    and not (not uncovered_cl and cand.cell_id in goal_ids)
                ):
                    continue
                e_valid.append((cand, next_sigma, prior_sc, unlock_credit, order_cost_penalty, step_cl_idx))

            if not e_valid:
                continue
            valid_frontier_sizes.append(len(e_valid))
            e_valid.sort(key=lambda x: x[2], reverse=True)
            shortlist = e_valid[: max(branch_factor * 2, 8)]

            # Schema-constrained SLM choice ranking strictly over E_valid(Gamma_t)
            choice_scores: Dict[str, float] = {}
            if (
                use_choice_scorer
                and len(shortlist) > 1
                and scorer_calls < max_scorer_calls
                and (shortlist[0][2] - shortlist[1][2]) < 0.45
            ):
                scorer_calls += 1
                gamma_desc = ", ".join(
                    sorted({
                        str(getattr(s[0] if isinstance(s, tuple) else s, "type_name", s))
                        for s in gamma
                        if s is not None
                    })
                ) or "empty (start)"
                choice_scores = mm.score_choices(
                    prompt,
                    [c.cell_id for c, _, _, _, _, _ in shortlist],
                    context_desc=f"chain={[x for x in chain_ids]}, available_carriers=[{gamma_desc}]",
                ) or {}

            scored_children: List[Tuple[float, float, Cell, Any, float, Optional[int]]] = []
            for cand, next_sigma, base_prior, unlock_cr, ord_pen, st_cl in shortlist:
                slm_sc = float(choice_scores.get(cand.cell_id, 0.0))
                new_cl = cell_clause_hits.get(cand.cell_id, set()) - cov_cl_set
                step_gain = base_prior + 0.65 * slm_sc + 0.30 * len(new_cl)
                step_cost = max(0.10, 0.15 + 0.75 / (1.0 + max(0.0, step_gain))) + ord_pen
                scored_children.append((step_cost, step_gain, cand, next_sigma, unlock_cr, st_cl))

            scored_children.sort(key=lambda x: x[1], reverse=True)
            for step_cost, step_gain, cand, next_sigma, unlock_cr, st_cl in scored_children[:branch_factor]:
                has_mappings = bool(getattr(next_sigma, "mappings", None))
                cand_pp = getattr(cand, "polymorphic_propagation", None)
                src_live_port = None
                if isinstance(cand_pp, dict) and (cand_pp.get("preserves_semantics") or cand_pp.get("preserves_layout")):
                    src_pn = cand_pp.get("source_port") or getattr(cand, "primary_in", None)
                    src_in_p = (getattr(cand, "inputs", None) or {}).get(src_pn) if src_pn else None
                    if src_in_p is None and getattr(cand, "inputs", None):
                        src_in_p = next(iter(cand.inputs.values()), None)
                    if src_in_p is not None:
                        in_sig_pp = getattr(src_in_p, "signature", src_in_p)
                        for gs in reversed(gamma):
                            if isinstance(gs, tuple) and len(gs) == 3 and gs[1] is not None:
                                if _unify_cached(gs[0], in_sig_pp, out_port=gs[1], in_port=src_in_p, prod_cell=gs[2], cons_cell=cand):
                                    src_live_port = gs[1]
                                    break
                inst_pairs: List[Tuple[Any, Any]] = []
                for oport, orig_osig in cell_out_ports_map[cand.cell_id]:
                    inst_sig = substitute_generics(orig_osig, next_sigma) if has_mappings else orig_osig
                    eff_port = oport
                    if has_mappings and inst_sig is not orig_osig:
                        eff_port = oport.model_copy(update={"signature": inst_sig}) if hasattr(oport, "model_copy") else copy.copy(oport)
                        if not hasattr(oport, "model_copy"):
                            eff_port.signature = inst_sig
                    if src_live_port is not None and isinstance(cand_pp, dict):
                        copied = (eff_port is not oport)
                        if cand_pp.get("preserves_semantics") and getattr(eff_port, "semantics", "any") in ("", "any", None):
                            src_sem = getattr(src_live_port, "semantics", None)
                            if src_sem and src_sem != "any":
                                if not copied:
                                    eff_port = copy.copy(eff_port)
                                    copied = True
                                eff_port.semantics = src_sem
                        if cand_pp.get("preserves_layout") and isinstance(getattr(src_live_port, "layout", None), dict):
                            cur_lay = getattr(eff_port, "layout", None)
                            if not isinstance(cur_lay, dict) or cur_lay.get("rank") is None:
                                if not copied:
                                    eff_port = copy.copy(eff_port)
                                eff_port.layout = dict(src_live_port.layout)
                    inst_pairs.append((eff_port, inst_sig))
                inst_outs = [osig for _, osig in inst_pairs]
                existing_keys = {
                    _sig_dedup_key(gs[0] if (isinstance(gs, tuple) and len(gs) == 3) else gs, sig_key_cache)
                    for gs in gamma
                }
                added_sigs = []
                for eff_port, osig in inst_pairs:
                    ok = _sig_dedup_key(osig, sig_key_cache)
                    if ok not in existing_keys:
                        added_sigs.append((osig, eff_port, cand))
                        existing_keys.add(ok)
                next_gamma = gamma + tuple(added_sigs)
                next_chain = chain_ids + (cand.cell_id,)
                cand_optic_pre = str(getattr(cand, "optic_role", "") or "").strip().lower()
                raw_cand_new_cl = cell_clause_hits.get(cand.cell_id, set()) - cov_cl_set
                closing_open_only = bool(
                    open_lens
                    and cand_optic_pre == "setter"
                    and (not raw_cand_new_cl or (next_target is not None and min(raw_cand_new_cl) > next_target))
                )
                cand_new_cl = set() if closing_open_only else raw_cand_new_cl
                next_cov = cov_cl if closing_open_only else tuple(sorted(cov_cl_set | set(cell_clause_hits.get(cand.cell_id, ()))))

                next_props_d = dict(active_props)
                explicit_posts = {
                    str(_prop_cond_field(p, "property"))
                    for p in (getattr(cand, "postconditions", None) or ())
                    if _prop_cond_field(p, "property")
                }
                for pk, pv in cell_props_map[cand.cell_id].items():
                    if isinstance(pv, (list, set)):
                        hpv: Any = tuple(pv)
                    elif isinstance(pv, dict):
                        hpv = tuple(sorted((str(dk), str(dv)) for dk, dv in pv.items()))
                    else:
                        hpv = pv
                    if pk in explicit_posts or pk not in next_props_d:
                        next_props_d[pk] = hpv
                next_active_props = tuple(sorted(next_props_d.items()))

                cand_optic = str(getattr(cand, "optic_role", "") or "").strip().lower()
                if cand_optic == "getter":
                    next_open_lens = True
                    next_focus_modified = False
                    next_active_focus = (
                        clause_focus_toks.get(st_cl, active_focus)
                        if st_cl is not None and clause_focus_toks.get(st_cl)
                        else active_focus
                    )
                elif cand_optic == "setter":
                    next_open_lens = False
                    next_focus_modified = False
                    next_active_focus = active_focus
                else:
                    outputs_writable = bool(
                        setter_val_sigs
                        and any(_unify_cached(osig, sv) for osig in inst_outs for sv in setter_val_sigs)
                    )
                    if open_lens and outputs_writable:
                        next_open_lens = True
                        next_focus_modified = True
                    elif open_lens and not outputs_writable:
                        # Scalar reduction closes open projection without requiring setter writeback
                        next_open_lens = False
                        next_focus_modified = False
                    else:
                        next_open_lens = open_lens
                        next_focus_modified = focus_modified
                    next_active_focus = active_focus
                    if st_cl is not None and not next_active_focus and clause_focus_toks.get(st_cl):
                        next_active_focus = clause_focus_toks[st_cl]

                # Track causal clause-order inversions across primary clause anchors
                if cand_new_cl:
                    cand_cl_min = min(t[0] for t in cand_new_cl)
                    next_inv = inv_count + (1 if cand_cl_min < max_cl_seen else 0)
                    next_max_cl = max(max_cl_seen, cand_cl_min)
                else:
                    next_inv = inv_count
                    next_max_cl = max_cl_seen

                next_g = g_cost + step_cost
                next_h = _heuristic(cand.cell_id, next_cov, unlock_credit=unlock_cr, gamma_state=next_gamma)
                counter += 1
                heapq.heappush(
                    pq,
                    (
                        next_g + next_h,
                        neg_ev - step_gain,
                        counter,
                        next_g,
                        next_chain,
                        next_gamma,
                        next_sigma,
                        next_cov,
                        next_active_props,
                        next_open_lens,
                        next_active_focus,
                        next_focus_modified,
                        step_clauses + (st_cl,),
                        next_inv,
                        next_max_cl,
                    ),
                )

        chosen_entry: Optional[Tuple[float, Tuple[str, ...], Tuple[Optional[int], ...]]] = None
        if best_full_complete is not None:
            chosen_entry = best_full_complete
        elif goal_sig is not None and best_complete is not None:
            chosen_entry = best_complete
        elif best_complete is not None and best_partial is not None:
            chosen_entry = best_complete if best_complete[0] >= best_partial[0] else best_partial
        elif best_complete is not None:
            chosen_entry = best_complete
        elif best_partial is not None:
            chosen_entry = best_partial

        chosen_ids = chosen_entry[1] if chosen_entry is not None else None
        chosen_clauses = chosen_entry[2] if chosen_entry is not None else ()

        self._trace(
            "omega_astar_search",
            expansions=expansions,
            scorer_calls=scorer_calls,
            avg_valid_frontier=(
                round(sum(valid_frontier_sizes) / len(valid_frontier_sizes), 2)
                if valid_frontier_sizes
                else 0.0
            ),
            chosen=list(chosen_ids) if chosen_ids else [],
        )

        if not chosen_ids:
            return []
        result = []
        for idx_r, cid in enumerate(chosen_ids):
            c_cl = pool_by_id[cid].clone() if hasattr(pool_by_id[cid], "clone") else pool_by_id[cid]
            cl_idx_r = chosen_clauses[idx_r] if idx_r < len(chosen_clauses) else None
            if cl_idx_r is not None:
                c_cl.matched_clause_idx = cl_idx_r
            result.append(c_cl)
        for i_r, c_cl in enumerate(result):
            if getattr(c_cl, "matched_clause_idx", None) is None:
                for nxt_r in result[i_r + 1 :]:
                    if getattr(nxt_r, "matched_clause_idx", None) is not None:
                        c_cl.matched_clause_idx = nxt_r.matched_clause_idx
                        break
        return result


__all__ = ["OmegaTypeDirectedAStarRouteMethod"]
