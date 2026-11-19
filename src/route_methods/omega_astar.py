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

import ast
from collections import deque
import copy
import heapq
import importlib.util
import sys
import time
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


_STDLIB_MODULES = set(getattr(sys, "stdlib_module_names", ()))
_MODULE_SPEC_CACHE: Dict[str, bool] = {}


def _is_module_installed(top_mod: str) -> bool:
    """Checks via stdlib importlib.util.find_spec whether a top-level package exists in the active runtime."""
    if not top_mod or top_mod in _STDLIB_MODULES or top_mod in sys.builtin_module_names or top_mod in sys.modules:
        return True
    cached = _MODULE_SPEC_CACHE.get(top_mod)
    if cached is not None:
        return cached
    try:
        installed = importlib.util.find_spec(top_mod) is not None
    except (ValueError, AttributeError, ImportError):
        installed = False
    _MODULE_SPEC_CACHE[top_mod] = installed
    return installed


def _cell_missing_runtime_dep(cell: Any) -> bool:
    """True when AST/dependency inspection of a cell's declared deps/imports/template reveals an uninstallable top-level package."""
    cached = getattr(cell, "_cached_missing_runtime_dep", None)
    if isinstance(cached, bool):
        return cached
    sources: List[str] = []
    bare_deps: List[str] = []
    for attr in ("imports", "dependencies"):
        for imp in (getattr(cell, attr, None) or ()):
            if not isinstance(imp, str):
                continue
            s = imp.strip()
            if not s:
                continue
            if s.startswith(("import ", "from ")):
                sources.append(s)
            else:
                bare_deps.append(s)
    tmpl = getattr(cell, "code_template", None) or ""
    if isinstance(tmpl, str) and tmpl:
        for raw_ln in tmpl.splitlines():
            ln = raw_ln.strip()
            if ln.startswith(("import ", "from ")):
                sources.append(ln)
    missing = False
    for bdep in bare_deps:
        top = bdep.split(".", 1)[0].strip()
        if top and top.isidentifier() and not _is_module_installed(top):
            missing = True
            break
    if not missing:
        for stmt in sources:
            try:
                tree = ast.parse(stmt, mode="exec")
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = (alias.name or "").split(".", 1)[0]
                        if top and not _is_module_installed(top):
                            missing = True
                            break
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    top = node.module.split(".", 1)[0]
                    if top and not _is_module_installed(top):
                        missing = True
                        break
                if missing:
                    break
            if missing:
                break
    try:
        cell._cached_missing_runtime_dep = missing
    except Exception:
        pass
    return missing


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
    if str(getattr(port, "state", "") or "").strip().lower() == "hyperparameter":
        return False
    if reg is not None:
        if reg.is_config_role(role) or reg.is_hof_role(role) or role in reg.get_roles_for_category("parameter"):
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
    is_bytes_carrier = bool(t_name == "bytes" or (reg is not None and reg.is_subtype(t_name, "bytes")))
    if sk == "scalar" and not is_bytes_carrier and (reg.is_scalar_primitive(t_name) if reg is not None else t_name in builtin_scalars):
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
    out_sig = getattr(out_port, "signature", out_port)
    in_sig = getattr(in_port, "signature", in_port)
    out_abs = str(getattr(out_port, "abstract_type", "") or getattr(out_sig, "abstract_type", "") or "").strip().lower()
    in_abs = str(getattr(in_port, "abstract_type", "") or getattr(in_sig, "abstract_type", "") or "").strip().lower()
    return TypeRegistry.get_instance().are_structure_kinds_compatible(out_sk, in_sk, out_abs, in_abs)


_MISSING_SENTINEL = object()


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
    empty_pair_cache: Optional[Dict[Tuple[int, int, int, int], Any]] = None,
    tv_id_cache: Optional[Dict[Tuple[int, int], bool]] = None,
    kept_sigs: Optional[Dict[int, Any]] = None,
) -> Optional[Any]:
    """
    Returns updated Substitution iff ALL required dataflow inputs of `cell`
    satisfy carrier structure_kind compatibility and unify with carriers in `gamma`, else None.
    """
    if req_ins is None:
        req_ins = _cell_required_data_inputs(cell)
    if not req_ins:
        return sigma.copy() if hasattr(sigma, "copy") else Substitution()
    if not gamma:
        return None
    if reg is None:
        try:
            reg = TypeRegistry.get_instance()
        except Exception:
            reg = None
    cur_sigma = sigma
    copied = False
    empty_sub = not getattr(cur_sigma, "mappings", None)
    disjoint_partners: Optional[Dict[str, Set[str]]] = None
    if len(req_ins) >= 2 and any(getattr(ip, "disjoint_from_port", None) for _, ip in req_ins):
        disjoint_partners = {}
        req_name_set = {pn for pn, _ in req_ins}
        for pn, ip in req_ins:
            d_ref = getattr(ip, "disjoint_from_port", None)
            if d_ref and d_ref in req_name_set and d_ref != pn:
                disjoint_partners.setdefault(pn, set()).add(d_ref)
                disjoint_partners.setdefault(d_ref, set()).add(pn)
        if not disjoint_partners:
            disjoint_partners = None
    bound_g_idx: Dict[str, int] = {}
    for in_name, in_port in req_ins:
        in_sig = getattr(in_port, "signature", in_port)
        id_in_port = id(in_port)
        id_in_sig = id(in_sig)
        in_key = None
        matched = False
        blocked_g_idxs = (
            {bound_g_idx[dp] for dp in disjoint_partners.get(in_name, ()) if dp in bound_g_idx}
            if disjoint_partners is not None
            else ()
        )
        for g_idx, g_item in enumerate(gamma):
            if blocked_g_idxs and g_idx in blocked_g_idxs:
                continue
            if type(g_item) is tuple and len(g_item) == 3:
                out_sig, out_port, _prod_cell = g_item
            else:
                out_sig, out_port = g_item, None
            id_out_port = id(out_port) if out_port is not None else 0
            id_out_sig = id(out_sig)

            if empty_sub and empty_pair_cache is not None:
                ep_k = (id_out_port, id_out_sig, id_in_port, id_in_sig)
                ep_hit = empty_pair_cache.get(ep_k, _MISSING_SENTINEL)
                if ep_hit is not _MISSING_SENTINEL:
                    if ep_hit is None:
                        continue
                    if not ep_hit:
                        matched = True
                        if disjoint_partners is not None:
                            bound_g_idx[in_name] = g_idx
                        break
                    if not copied:
                        cur_sigma = sigma.copy() if hasattr(sigma, "copy") else Substitution()
                        copied = True
                    cur_sigma.mappings.update(ep_hit)
                    empty_sub = not cur_sigma.mappings
                    matched = True
                    if disjoint_partners is not None:
                        bound_g_idx[in_name] = g_idx
                    break

            if out_port is not None:
                ek = (id_out_port, id_in_port)
                ev_ok = edge_cache.get(ek) if edge_cache is not None else None
                if ev_ok is None:
                    ev_ok = _structural_kind_compatible(out_port, in_port)
                    if edge_cache is not None:
                        edge_cache[ek] = ev_ok
                if not ev_ok:
                    if empty_sub and empty_pair_cache is not None:
                        if kept_sigs is not None:
                            kept_sigs[id_out_sig] = out_sig
                            kept_sigs[id_in_sig] = in_sig
                            kept_sigs[id_out_port] = out_port
                            kept_sigs[id_in_port] = in_port
                        empty_pair_cache[(id_out_port, id_out_sig, id_in_port, id_in_sig)] = None
                    continue

            tv_k = (id_out_sig, id_in_sig)
            tv_blocked = tv_id_cache.get(tv_k) if tv_id_cache is not None else None
            if tv_blocked is None:
                tv_blocked = _typevar_to_concrete_blocked(out_sig, in_sig, reg, tv_cache=tv_cache)
                if tv_id_cache is not None:
                    if kept_sigs is not None:
                        kept_sigs[id_out_sig] = out_sig
                        kept_sigs[id_in_sig] = in_sig
                    tv_id_cache[tv_k] = tv_blocked
            if tv_blocked:
                if empty_sub and empty_pair_cache is not None:
                    if kept_sigs is not None:
                        kept_sigs[id_out_sig] = out_sig
                        kept_sigs[id_in_sig] = in_sig
                        if out_port is not None:
                            kept_sigs[id_out_port] = out_port
                        kept_sigs[id_in_port] = in_port
                    empty_pair_cache[(id_out_port, id_out_sig, id_in_port, id_in_sig)] = None
                continue

            if empty_sub and (unify_cache is not None or unify_empty_sub_cache is not None):
                if in_key is None:
                    in_key = _sig_dedup_key(in_sig, sig_key_cache)
                out_key = _sig_dedup_key(out_sig, sig_key_cache)
                pair_k = (out_key, in_key)
                if unify_cache is not None and unify_cache.get(pair_k) is False:
                    if empty_pair_cache is not None:
                        if kept_sigs is not None:
                            kept_sigs[id_out_sig] = out_sig
                            kept_sigs[id_in_sig] = in_sig
                            if out_port is not None:
                                kept_sigs[id_out_port] = out_port
                            kept_sigs[id_in_port] = in_port
                        empty_pair_cache[(id_out_port, id_out_sig, id_in_port, id_in_sig)] = None
                    continue
                if unify_empty_sub_cache is not None and pair_k in unify_empty_sub_cache:
                    m_tup = unify_empty_sub_cache[pair_k]
                    if empty_pair_cache is not None:
                        if kept_sigs is not None:
                            kept_sigs[id_out_sig] = out_sig
                            kept_sigs[id_in_sig] = in_sig
                            if out_port is not None:
                                kept_sigs[id_out_port] = out_port
                            kept_sigs[id_in_port] = in_port
                        empty_pair_cache[(id_out_port, id_out_sig, id_in_port, id_in_sig)] = m_tup
                    if m_tup is None:
                        continue
                    if not m_tup:
                        matched = True
                        if disjoint_partners is not None:
                            bound_g_idx[in_name] = g_idx
                        break
                    if not copied:
                        cur_sigma = sigma.copy() if hasattr(sigma, "copy") else Substitution()
                        copied = True
                    cur_sigma.mappings.update(m_tup)
                    empty_sub = not cur_sigma.mappings
                    matched = True
                    if disjoint_partners is not None:
                        bound_g_idx[in_name] = g_idx
                    break
                if not copied:
                    cur_sigma = sigma.copy() if hasattr(sigma, "copy") else Substitution()
                    copied = True
                sub = unify(out_sig, in_sig, cur_sigma)
                if unify_cache is not None:
                    unify_cache[pair_k] = (sub is not None)
                m_store = (
                    tuple(sub.mappings.items()) if (sub is not None and hasattr(sub, "mappings")) else (
                        () if sub is not None else None
                    )
                )
                if unify_empty_sub_cache is not None:
                    unify_empty_sub_cache[pair_k] = m_store
                if empty_pair_cache is not None:
                    if kept_sigs is not None:
                        kept_sigs[id_out_sig] = out_sig
                        kept_sigs[id_in_sig] = in_sig
                        if out_port is not None:
                            kept_sigs[id_out_port] = out_port
                        kept_sigs[id_in_port] = in_port
                    empty_pair_cache[(id_out_port, id_out_sig, id_in_port, id_in_sig)] = m_store
            else:
                if not copied:
                    cur_sigma = sigma.copy() if hasattr(sigma, "copy") else Substitution()
                    copied = True
                sub = unify(out_sig, in_sig, cur_sigma)
            if sub is not None:
                cur_sigma = sub
                copied = True
                empty_sub = not getattr(cur_sigma, "mappings", None)
                matched = True
                if disjoint_partners is not None:
                    bound_g_idx[in_name] = g_idx
                break
        if not matched:
            return None
    if not copied:
        return sigma.copy() if hasattr(sigma, "copy") else Substitution()
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

        orch_ref = orchestrator or getattr(self, "orchestrator", None)

        # Structural lattice augmentation: top-K RAG tunnels emphasize semantic anchors
        # and often omit structural coercion bridges, container optics, property-repair
        # transformers, and unproduced handle initializers. Pull only structurally typed
        # morphisms reachable from carriers/demands already in `pool`.
        if orch_ref is not None and getattr(orch_ref, "loaded_cells", None):
            _p_domains = {
                str(getattr(c, "domain_name", "") or getattr(c, "domain", "") or "").strip().lower()
                for c in pool
            } - {""}
            _p_outs = [
                getattr(op, "signature", op)
                for c in pool
                for op in (getattr(c, "outputs", {}) or {}).values()
                if getattr(op, "signature", op) is not None
            ]
            _p_ins = [
                (ip, getattr(ip, "signature", ip))
                for c in pool
                for _, ip in _cell_required_data_inputs(c, reg=reg)
                if getattr(ip, "signature", ip) is not None
            ]
            _p_prompt_toks = set(CellTokenizer.tokenize_prompt(prompt)) if prompt else set()

            def _add_struct_cell(sc: Any) -> bool:
                scid = getattr(sc, "cell_id", None)
                if not scid or scid in seen_ids or getattr(sc, "node_type", "") == "constant":
                    return False
                if getattr(sc, "cell_type", "") == "macro" and str(getattr(sc, "node_role", "") or "").lower() != "bridge":
                    return False
                pool.append(sc)
                seen_ids.add(scid)
                _p_outs.extend(
                    getattr(op, "signature", op)
                    for op in (getattr(sc, "outputs", {}) or {}).values()
                    if getattr(op, "signature", op) is not None
                )
                _p_ins.extend(
                    (ip, getattr(ip, "signature", ip))
                    for _, ip in _cell_required_data_inputs(sc, reg=reg)
                    if getattr(ip, "signature", ip) is not None
                )
                return True

            # Rule D: Unproduced handle producers (e.g. initializer/trainer for a handle consumer)
            _unmet_handles = [
                ip for ip, isig in _p_ins
                if getattr(ip, "required", True)
                and getattr(ip, "default_value", None) is None
                and self.is_handle_port(ip)
                and not any(unify(osig, isig) is not None for osig in _p_outs)
            ]
            for h_in in _unmet_handles:
                h_sig = getattr(h_in, "signature", h_in)
                h_nom = str(getattr(h_in, "nominal_type", "") or "").strip().lower()
                h_cands: List[Tuple[Tuple[int, int, int], Any]] = []
                for lc in orch_ref.loaded_cells.values():
                    lcid = getattr(lc, "cell_id", None)
                    if not lcid or lcid in seen_ids or is_terminal_sink(lc, orch_ref):
                        continue
                    lc_outs = list((getattr(lc, "outputs", {}) or {}).values())
                    if not any(
                        getattr(op, "signature", op) is not None
                        and unify(getattr(op, "signature", op), h_sig) is not None
                        for op in lc_outs
                    ):
                        continue
                    nom_hit = 1 if (
                        h_nom and any(str(getattr(op, "nominal_type", "") or "").strip().lower() == h_nom for op in lc_outs)
                    ) else 0
                    lc_id_toks = set(getattr(lc, "identity_tokens", None) or ()) | set(CellTokenizer.tokenize_identifier(lcid))
                    id_ov = len(_p_prompt_toks & lc_id_toks)
                    all_ov = len(_p_prompt_toks & (set(getattr(lc, "token_set", None) or ()) | lc_id_toks))
                    if nom_hit or id_ov > 0:
                        h_cands.append(((nom_hit, id_ov, all_ov), lc))
                h_cands.sort(key=lambda item: item[0], reverse=True)
                for _, h_prod in h_cands[:2]:
                    _add_struct_cell(h_prod)
            _p_src_lits, _p_dest_lits = self.extract_file_literals(prompt or "", ctx=ctx)
            if _p_dest_lits:
                _req_handles = [
                    (ip, isig) for ip, isig in _p_ins
                    if getattr(ip, "required", True)
                    and getattr(ip, "default_value", None) is None
                    and self.is_handle_port(ip)
                ]
                for h_in, h_sig in _req_handles:
                    _has_file_prod = any(
                        any(is_path_port(p) for p in (getattr(pc, "inputs", {}) or {}).values())
                        and any(self.file_compat(pc, df) >= 1 for df in _p_dest_lits)
                        and any(
                            getattr(op, "signature", op) is not None
                            and unify(getattr(op, "signature", op), h_sig) is not None
                            for op in (getattr(pc, "outputs", {}) or {}).values()
                        )
                        for pc in pool
                    )
                    if _has_file_prod:
                        continue
                    _fh_cands: List[Tuple[Tuple[int, int, int], Any]] = []
                    for lc in orch_ref.loaded_cells.values():
                        lcid = getattr(lc, "cell_id", None)
                        if not lcid or lcid in seen_ids or is_terminal_sink(lc, orch_ref):
                            continue
                        lc_outs = list((getattr(lc, "outputs", {}) or {}).values())
                        if not any(
                            getattr(op, "signature", op) is not None
                            and unify(getattr(op, "signature", op), h_sig) is not None
                            for op in lc_outs
                        ):
                            continue
                        if not any(is_path_port(p) for p in (getattr(lc, "inputs", {}) or {}).values()):
                            continue
                        fc = max((self.file_compat(lc, df) for df in _p_dest_lits), default=0)
                        if fc < 1:
                            continue
                        lc_id_toks = set(getattr(lc, "identity_tokens", None) or ()) | set(CellTokenizer.tokenize_identifier(lcid))
                        id_ov = len(_p_prompt_toks & lc_id_toks)
                        all_ov = len(_p_prompt_toks & (set(getattr(lc, "token_set", None) or ()) | lc_id_toks))
                        _fh_cands.append(((fc, id_ov, all_ov), lc))
                    _fh_cands.sort(key=lambda item: item[0], reverse=True)
                    for _, h_prod in _fh_cands[:2]:
                        _add_struct_cell(h_prod)

            # Rule A: Declared carrier/state coercion bridges reachable from pool
            _bridge_pool = list(getattr(orch_ref, "bridge_cells", ()) or ()) + [
                lc for lc in orch_ref.loaded_cells.values()
                if str(getattr(lc, "node_role", "") or "").strip().lower() == "bridge"
            ]
            for bc in _bridge_pool:
                bcid = getattr(bc, "cell_id", None)
                if not bcid or bcid in seen_ids or is_ingress_source(bc, orch_ref) or is_terminal_sink(bc, orch_ref):
                    continue
                b_ins = _cell_required_data_inputs(bc, reg=reg)
                b_outs = [
                    getattr(op, "signature", op)
                    for op in (getattr(bc, "outputs", {}) or {}).values()
                    if getattr(op, "signature", op) is not None
                ]
                if not b_ins or not b_outs:
                    continue
                b_in_sig = getattr(b_ins[0][1], "signature", b_ins[0][1])
                in_reachable = any(unify(osig, b_in_sig) is not None for osig in _p_outs)
                out_useful = any(unify(bo, isig) is not None for bo in b_outs for _, isig in _p_ins)
                if in_reachable and out_useful:
                    _add_struct_cell(bc)

            # Rule B: Container optics matching domains/carriers in pool
            _optic_roles = set(reg.get_roles_for_category("optic")) if reg is not None else {"getter", "setter", "complement"}
            for lc in orch_ref.loaded_cells.values():
                lcid = getattr(lc, "cell_id", None)
                if not lcid or lcid in seen_ids:
                    continue
                lc_or = str(getattr(lc, "optic_role", "") or "").strip().lower()
                lc_nr = str(getattr(lc, "node_role", "") or "").strip().lower()
                if lc_or not in _optic_roles and lc_nr != "optic":
                    continue
                lc_dom = str(getattr(lc, "domain_name", "") or getattr(lc, "domain", "") or "").strip().lower()
                if lc_dom in _p_domains:
                    _add_struct_cell(lc)
                    continue
                lc_req = _cell_required_data_inputs(lc, reg=reg)
                if lc_req and any(unify(osig, getattr(lc_req[0][1], "signature", lc_req[0][1])) is not None for osig in _p_outs):
                    _add_struct_cell(lc)

            # Rule C: Property-repair transformers resolving preconditions in pool
            _demanded_props: Set[Tuple[str, str]] = set()
            for c in pool:
                for pre in (getattr(c, "preconditions", None) or ()):
                    p_n = pre.get("property") if isinstance(pre, dict) else getattr(pre, "property", None)
                    p_op = pre.get("operator") if isinstance(pre, dict) else getattr(pre, "operator", None)
                    p_val = pre.get("value") if isinstance(pre, dict) else getattr(pre, "value", None)
                    if p_op == "==" and p_n is not None:
                        _demanded_props.add((str(p_n), repr(p_val)))
            if _demanded_props:
                for lc in orch_ref.loaded_cells.values():
                    lcid = getattr(lc, "cell_id", None)
                    if not lcid or lcid in seen_ids or is_ingress_source(lc, orch_ref) or is_terminal_sink(lc, orch_ref):
                        continue
                    lc_posts = getattr(lc, "postconditions", None) or ()
                    if not any(
                        (q.get("operator") if isinstance(q, dict) else getattr(q, "operator", None)) == "=="
                        and (
                            str(q.get("property") if isinstance(q, dict) else getattr(q, "property", None)),
                            repr(q.get("value") if isinstance(q, dict) else getattr(q, "value", None)),
                        ) in _demanded_props
                        for q in lc_posts
                    ):
                        continue
                    lc_req = _cell_required_data_inputs(lc, reg=reg)
                    if len(lc_req) == 1:
                        in_s = getattr(lc_req[0][1], "signature", lc_req[0][1])
                        if any(unify(osig, in_s) is not None for osig in _p_outs):
                            _add_struct_cell(lc)

        cur_reg_gen = int(getattr(reg, "_generation", 0)) if reg is not None else 0
        persistent_bag = None
        if orch_ref is not None:
            persistent_bag = getattr(orch_ref, "_omega_persistent_caches", None)
            if (
                not isinstance(persistent_bag, dict)
                or persistent_bag.get("gen") != cur_reg_gen
                or len(persistent_bag.get("unify", ())) > 50000
                or len(persistent_bag.get("cls_unify", ())) > 50000
            ):
                persistent_bag = {
                    "gen": cur_reg_gen,
                    "unify": {},
                    "unify_empty": {},
                    "tv": {},
                    "cls_unify": {},
                    "props": {},
                    "preconds": {},
                }
                orch_ref._omega_persistent_caches = persistent_bag

        unify_cache: Dict[Tuple[Tuple[Any, ...], Tuple[Any, ...]], bool] = (
            persistent_bag["unify"] if persistent_bag is not None else {}
        )
        if persistent_bag is not None and "unify_empty" not in persistent_bag:
            persistent_bag["unify_empty"] = {}
        unify_empty_sub_cache: Dict[Tuple[Tuple[Any, ...], Tuple[Any, ...]], Optional[Tuple[Tuple[str, Any], ...]]] = (
            persistent_bag["unify_empty"] if persistent_bag is not None else {}
        )
        edge_cache: Dict[Tuple[int, int], bool] = {}
        sig_key_cache: Dict[int, Tuple[Any, ...]] = {}
        tv_cache: Dict[Tuple[str, str], bool] = (
            persistent_bag["tv"] if persistent_bag is not None else {}
        )
        port_unify_cache: Dict[Tuple[int, int], bool] = {}
        empty_pair_cache: Dict[Tuple[int, int, int, int], Any] = {}
        tv_id_cache: Dict[Tuple[int, int], bool] = {}
        kept_sigs: Dict[int, Any] = {}

        def _unify_cached(
            out_sig: Any,
            in_sig: Any,
            out_port: Any = None,
            in_port: Any = None,
            prod_cell: Any = None,
            cons_cell: Any = None,
        ) -> bool:
            if out_port is not None and in_port is not None:
                is_raw_port_sigs = (
                    out_sig is getattr(out_port, "signature", out_port)
                    and in_sig is getattr(in_port, "signature", in_port)
                )
                if is_raw_port_sigs:
                    puk = (id(out_port), id(in_port))
                    pu_hit = port_unify_cache.get(puk)
                    if pu_hit is not None:
                        return pu_hit
                    if _typevar_to_concrete_blocked(out_sig, in_sig, reg, tv_cache=tv_cache):
                        port_unify_cache[puk] = False
                        return False
                    ek = puk
                    ev_ok = edge_cache.get(ek)
                    if ev_ok is None:
                        ev_ok = _structural_kind_compatible(out_port, in_port)
                        edge_cache[ek] = ev_ok
                    if not ev_ok:
                        port_unify_cache[puk] = False
                        return False
                    pk = (_sig_dedup_key(out_sig, sig_key_cache), _sig_dedup_key(in_sig, sig_key_cache))
                    res = unify_cache.get(pk)
                    if res is None:
                        res = unify(out_sig, in_sig) is not None
                        unify_cache[pk] = res
                    port_unify_cache[puk] = res
                    return res
                ek = (id(out_port), id(in_port))
                ev_ok = edge_cache.get(ek)
                if ev_ok is None:
                    ev_ok = _structural_kind_compatible(out_port, in_port)
                    edge_cache[ek] = ev_ok
                if not ev_ok:
                    return False
            if _typevar_to_concrete_blocked(out_sig, in_sig, reg, tv_cache=tv_cache):
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
        compat_dest_handle_types: Set[str] = set()
        if dest_file_literals:
            for _c_dh in pool:
                if (
                    is_ingress_source(_c_dh, self.orchestrator)
                    and any(is_path_port(p) for p in (getattr(_c_dh, "inputs", {}) or {}).values())
                    and any(self.file_compat(_c_dh, df) >= 1 for df in dest_file_literals)
                ):
                    for _, _osig_dh in _cell_outputs_ports(_c_dh):
                        _otn_dh = str(getattr(_osig_dh, "type_name", "") or "").strip().lower()
                        if _otn_dh:
                            compat_dest_handle_types.add(_otn_dh)

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
        cell_any_req_map: Dict[str, bool] = {
            c.cell_id: any(getattr(p, "required", False) for p in (getattr(c, "inputs", {}) or {}).values())
            for c in pool
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
        cell_missing_dep_map: Dict[str, bool] = {
            c.cell_id: _cell_missing_runtime_dep(c) for c in pool
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
        cell_clause_cf: Dict[Tuple[str, int], Tuple[float, float, float]] = {}
        cell_clause_port_frac: Dict[Tuple[str, int], float] = {}
        cell_input_tok_groups: Dict[str, List[Tuple[str, ...]]] = {}
        for c in pool:
            groups: List[Tuple[str, ...]] = []
            for pn in (getattr(c, "inputs", {}) or {}).keys():
                ptoks = tuple(
                    t.lower()
                    for t in (CellTokenizer.tokenize_identifier(str(pn)) if hasattr(CellTokenizer, "tokenize_identifier") else [str(pn)])
                    if t
                )
                if not ptoks:
                    ptoks = (str(pn).lower(),)
                groups.append(ptoks)
            cell_input_tok_groups[c.cell_id] = groups
        cell_target_has_lex: Set[Tuple[str, TargetId]] = set()
        clause_top_cells: Dict[TargetId, Set[str]] = {}
        sub_anchor_owner: Dict[TargetId, str] = {}
        for cl_i in range(n_clauses):
            cl_txt = clauses[cl_i] if cl_i < len(clauses) else ""
            cl_word_set = {
                w.strip(".,;:\"'()[]{}").lower()
                for w in cl_txt.split()
                if w.strip(".,;:\"'()[]{}")
            } if cl_txt else set()
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
                cell_clause_cf[(c.cell_id, cl_i)] = cf
                p_groups = cell_input_tok_groups.get(c.cell_id, ())
                if p_groups and cl_word_set:
                    p_hits = sum(1 for pg in p_groups if any(pt in cl_word_set for pt in pg))
                    cell_clause_port_frac[(c.cell_id, cl_i)] = p_hits / float(len(p_groups))
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
                def _anchor_pos(anc_pair: Tuple[Cell, Dict[str, Any]]) -> Tuple[int, int, int, float]:
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
                        1 if is_sink_map.get(anc_c.cell_id, False) else 0,
                        min(positions) if positions else 10**6,
                        1 if anc_optic == "setter" else 0,
                        -float((anc_pair[1].get("rank") or (0.0,))[0]),
                    )
                sub_anchors.sort(key=_anchor_pos)
                all_sub_cids = {sa[0].cell_id for sa in sub_anchors}
                for sub_j, (anc_c, anc_o) in enumerate(sub_anchors):
                    t_id: TargetId = (cl_i, sub_j)
                    sub_anchor_owner[t_id] = anc_c.cell_id
                    anc_matched = set(anc_o.get("tokens", ()) or ())
                    anc_tok_eval = anc_matched if anc_matched else cl_toks_all
                    anc_sub_ov = ev.identity_overlap(anc_tok_eval, anc_c) if anc_matched else anc_o
                    anc_rank = anc_sub_ov.get("rank", (0.0, 0.0, 0.0))
                    anc_stage1 = (getattr(anc_c, "stage", None) == 1)
                    top_cids: Set[str] = set()
                    for c in cl_pool:
                        cid = c.cell_id
                        if cid != anc_c.cell_id and cid in all_sub_cids:
                            continue
                        if cid != anc_c.cell_id and anc_stage1 and getattr(c, "stage", None) != 1:
                            continue
                        if cid != anc_c.cell_id and not cell_any_req_map.get(anc_c.cell_id, False) and cell_any_req_map.get(cid, False):
                            continue
                        ov = ev.identity_overlap(anc_tok_eval, c)
                        c_matched = set(ov.get("tokens", ()) or ())
                        ov_rank = ov.get("rank") or (0.0, 0.0, 0.0)
                        ov_rec = float(ov_rank[0] or 0.0)
                        ov_np = float(ov_rank[1] or 0.0)
                        sem_sc = float(cl_dict.get(cid, 0.0) or 0.0)
                        sem_ok_for_sub = (max_sem > 0.0 and sem_sc >= 0.75 * max_sem)
                        np_ok = (anc_rank[1] <= 0.0) or (ov_np >= 0.70 * anc_rank[1]) or sem_ok_for_sub
                        if cid == anc_c.cell_id or (
                            anc_matched
                            and (c_matched & anc_matched)
                            and ov_rec >= 0.80 * anc_rank[0]
                            and np_ok
                        ):
                            cell_clause_hits[cid].add(t_id)
                            if ov_rec > 0.0:
                                cell_target_has_lex.add((cid, t_id))
                            cf_rel = 1.0 if cid == anc_c.cell_id else (min(1.0, ov_rec / anc_rank[0]) if anc_rank[0] > 0.0 else 1.0)
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
                            or (cf[0] >= 0.30 * best_cf[0] and cf[1] > best_cf[1] and cf[2] >= 0.45 * best_cf[2] and cf[1] >= 0.75 * best_id_prec)
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
                        if cf[0] > 0.0:
                            cell_target_has_lex.add((cid, t_id))
                        cf_rel = (cf[0] / best_cf[0]) if best_cf[0] > 0.0 else 0.0
                        prec_rel = (cf[1] / best_id_prec) if best_id_prec > 0.0 else 0.0
                        sem_rel = (sem_sc / max_sem) if max_sem > 0.0 else 0.0
                        if best_cf[0] >= 0.25:
                            strength = 0.55 * cf_rel + 0.15 * prec_rel + 0.30 * sem_rel
                        else:
                            strength = 0.40 * cf_rel + 0.15 * prec_rel + 0.45 * sem_rel
                        clause_cell_strength[(cid, t_id)] = strength
                        if cf_rel >= 0.80 or (best_cf[0] == 0.0 and sem_rel >= 0.85) or (sem_hit and sem_rel >= 0.90 and best_cf[0] < 0.85):
                            top_cids.add(cid)
                if top_cids:
                    clause_top_cells[t_id] = top_cids
                    self._trace("omega_target_candidates", target=t_id, candidates=sorted(top_cids))

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
            for ft in CellTokenizer.tokenize_identifier(str(fl)):
                known_op_toks.add(ft.lower())

        clause_focus_toks: Dict[int, Tuple[str, ...]] = {}
        clause_has_setter: Dict[int, bool] = {}
        for cl_i in range(n_clauses):
            cl_txt = clauses[cl_i] if cl_i < len(clauses) else ""
            raw_w = CellTokenizer.tokenize_prompt(cl_txt)
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

        clause_has_installed: Dict[TargetId, bool] = {
            cl_t: any(not cell_missing_dep_map.get(tcid, False) for tcid in t_cids)
            for cl_t, t_cids in clause_top_cells.items()
        }
        installed_ingress_in_dom: Set[str] = {
            cell_domain_map[c.cell_id]
            for c in pool
            if (
                cell_domain_map[c.cell_id]
                and is_ingress_map[c.cell_id]
                and not cell_missing_dep_map[c.cell_id]
                and (not src_file_literals or not cand_needs_path_map[c.cell_id] or self.file_compat(c, src_file_literals[0]) >= 0)
            )
        }
        installed_sink_in_dom: Set[str] = {
            cell_domain_map[c.cell_id]
            for c in pool
            if (
                cell_domain_map[c.cell_id]
                and is_sink_map[c.cell_id]
                and not cell_missing_dep_map[c.cell_id]
                and (not dest_file_literals or not cand_needs_path_map[c.cell_id] or self.file_compat(c, dest_file_literals[0]) >= 0)
            )
        }

        # Build backward adjacency inside tunnel via cached unification for h_bwd heuristic
        pool_by_id = {c.cell_id: c for c in pool}
        props_store = persistent_bag["props"] if persistent_bag is not None else {}
        preconds_store = persistent_bag["preconds"] if persistent_bag is not None else {}
        cell_props_map: Dict[str, Dict[str, Any]] = {}
        cell_preconds_map: Dict[str, Dict[str, List[Tuple[str, str, Any]]]] = {}
        for c in pool:
            cid_c = c.cell_id
            p_hit = props_store.get(cid_c)
            if p_hit is None:
                p_hit = _cell_declared_props(c, reg)
                props_store[cid_c] = p_hit
            cell_props_map[cid_c] = p_hit
            pc_hit = preconds_store.get(cid_c)
            if pc_hit is None:
                pc_hit = _cell_required_preconds(c)
                preconds_store[cid_c] = pc_hit
            cell_preconds_map[cid_c] = pc_hit
        rev_edges: Dict[str, List[str]] = {c.cell_id: [] for c in pool}
        eff_edge_mode = str(getattr(orch_ref, "edge_mode", "") or "").strip().lower()
        static_rev_adj = getattr(orch_ref, "_reverse_adjacency", None) if orch_ref is not None else None
        port_can_receive_from: Dict[int, Set[str]] = {}
        if eff_edge_mode == "static" and isinstance(static_rev_adj, dict):
            for cons in pool:
                if not cell_req_ins_map[cons.cell_id]:
                    continue
                for prod_id in static_rev_adj.get(cons.cell_id, ()):
                    if prod_id != cons.cell_id and prod_id in pool_by_id:
                        rev_edges[cons.cell_id].append(prod_id)
        else:
            # Group unique (signature_key, structure_kind, abstract_type) classes so we check ~50x50
            # equivalence classes instead of N x N (~215k) cell port pairs directly.
            def _port_abs_sk(p_obj: Any, sig_obj: Any) -> Tuple[str, str]:
                p_sk = str(getattr(p_obj, "structure_kind", "") or "").strip().lower()
                p_abs = str(getattr(p_obj, "abstract_type", "") or getattr(sig_obj, "abstract_type", "") or "").strip().lower()
                return p_sk, p_abs

            out_cls_prods: Dict[Tuple[Any, str, str], List[str]] = {}
            out_cls_rep: Dict[Tuple[Any, str, str], Tuple[Any, Any]] = {}
            for prod in pool:
                p_outs = cell_out_ports_map[prod.cell_id]
                if not p_outs:
                    continue
                for oport, osig in p_outs:
                    ok = (_sig_dedup_key(osig, sig_key_cache), *_port_abs_sk(oport, osig))
                    if ok not in out_cls_rep:
                        out_cls_rep[ok] = (oport, osig)
                        out_cls_prods[ok] = []
                    if not out_cls_prods[ok] or out_cls_prods[ok][-1] != prod.cell_id:
                        out_cls_prods[ok].append(prod.cell_id)

            in_cls_cons: Dict[Tuple[Any, str, str], List[Tuple[str, int]]] = {}
            in_cls_rep: Dict[Tuple[Any, str, str], Tuple[Any, Any]] = {}
            for cons in pool:
                req_ins = cell_req_ins_map[cons.cell_id]
                if not req_ins:
                    continue
                for _, ip in req_ins:
                    isig = getattr(ip, "signature", ip)
                    ik = (_sig_dedup_key(isig, sig_key_cache), *_port_abs_sk(ip, isig))
                    if ik not in in_cls_rep:
                        in_cls_rep[ik] = (ip, isig)
                        in_cls_cons[ik] = []
                    in_cls_cons[ik].append((cons.cell_id, id(ip)))

            cls_unify_store = persistent_bag["cls_unify"] if persistent_bag is not None else {}
            sk_pair_cache: Dict[Tuple[str, str, str, str], bool] = {}
            rev_edge_sets: Dict[str, Set[str]] = {c.cell_id: set() for c in pool}
            for ok, (oport, osig) in out_cls_rep.items():
                prods_for_ok = out_cls_prods[ok]
                o_sk, o_abs = ok[1], ok[2]
                for ik, (ip, isig) in in_cls_rep.items():
                    sk_k = (o_sk, o_abs, ik[1], ik[2])
                    sk_ok = sk_pair_cache.get(sk_k)
                    if sk_ok is None:
                        sk_ok = _structural_kind_compatible(oport, ip)
                        sk_pair_cache[sk_k] = sk_ok
                    if not sk_ok:
                        continue
                    cls_pair = (ok, ik)
                    compat = cls_unify_store.get(cls_pair)
                    if compat is None:
                        compat = bool(_unify_cached(osig, isig, out_port=oport, in_port=ip))
                        cls_unify_store[cls_pair] = compat
                    if compat:
                        for cons_cid, ip_id in in_cls_cons[ik]:
                            recv_set = port_can_receive_from.get(ip_id)
                            if recv_set is None:
                                recv_set = set()
                                port_can_receive_from[ip_id] = recv_set
                            cons_rev_set = rev_edge_sets[cons_cid]
                            cons_rev_list = rev_edges[cons_cid]
                            for prod_cid in prods_for_ok:
                                if prod_cid != cons_cid:
                                    recv_set.add(prod_cid)
                                    if prod_cid not in cons_rev_set:
                                        cons_rev_set.add(prod_cid)
                                        cons_rev_list.append(prod_cid)

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
        scorer_wall_s = 0.0
        _choice_score_cache: Dict[Tuple[Tuple[str, ...], str], Dict[str, float]] = {}

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
            # Encourage covering all coverable sub-targets cleanly without combinatorial stalls
            return rem_clauses * 2.25 + h_goal

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

        cell_has_handle_out_map: Dict[str, bool] = {
            c.cell_id: any(
                self.is_handle_port(oport) or self.is_handle_port(osig)
                for oport, osig in cell_out_ports_map[c.cell_id]
            )
            for c in pool
        }
        cell_has_composite_out_map: Dict[str, bool] = {
            c.cell_id: any(
                (
                    lambda tl: bool(tl) and reg is not None and bool(reg.is_composite_carrier(tl))
                )(str(getattr(osig, "type_name", "") or "").strip().lower())
                for _, osig in cell_out_ports_map[c.cell_id]
            )
            for c in pool
        }
        cell_optic_role_map: Dict[str, str] = {
            c.cell_id: str(getattr(c, "optic_role", "") or "").strip().lower()
            for c in pool
        }
        rep_optics = reg.get_roles_for_category("optic") if reg is not None else ()
        cid_bit: Dict[str, int] = {c.cell_id: (1 << i) for i, c in enumerate(pool)}
        cell_req_prod_masks: Dict[str, Tuple[int, ...]] = {}
        if port_can_receive_from:
            for c in pool:
                r_ins = cell_req_ins_map[c.cell_id]
                if not r_ins:
                    continue
                masks: List[int] = []
                for _, ip in r_ins:
                    m_ip = 0
                    for prod_cid in port_can_receive_from.get(id(ip), ()):
                        pb = cid_bit.get(prod_cid)
                        if pb is not None:
                            m_ip |= pb
                    masks.append(m_ip)
                cell_req_prod_masks[c.cell_id] = tuple(masks)

        # Precompute pool-local required-input producer adjacency for k-hop backward precondition cone
        if port_can_receive_from:
            pool_producers_for: Dict[str, List[Cell]] = {
                c.cell_id: [
                    pool_by_id[pid]
                    for pid in rev_edges.get(c.cell_id, ())
                    if pid in pool_by_id and not is_sink_map[pid]
                ]
                for c in pool
            }
        else:
            pool_producers_for = {c.cell_id: [] for c in pool}
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
        visited_states: Set[Tuple[Tuple[str, ...], Tuple[TargetId, ...], Optional[str], bool, Tuple[str, ...]]] = set()

        max_expansions = int(kwargs.get("max_expansions", max(900, n_clauses * 240)))
        max_chain_len = int(kwargs.get("max_chain_len", max(18, n_clauses + 12, len(coverable_clauses) + 8)))
        branch_factor = int(kwargs.get("branch_factor", 8))
        _budget_ms = float(kwargs.get("planner_time_budget_ms", 18000.0))
        _search_deadline = time.perf_counter() + max(_budget_ms, 500.0) / 1000.0
        first_full_exp: Optional[int] = None
        expansions = 0
        valid_frontier_sizes: List[int] = []

        while pq and expansions < max_expansions:
            if time.perf_counter() >= _search_deadline:
                break
            if first_full_exp is not None and (expansions - first_full_exp) >= 80:
                break
            (
                f_sc, neg_ev, _, g_cost, chain_ids, gamma, sigma,
                cov_cl, active_props, open_lens, active_focus, focus_modified,
                step_clauses, inv_count, max_cl_seen,
            ) = heapq.heappop(pq)
            expansions += 1

            state_key = (tuple(sorted(chain_ids)), cov_cl, chain_ids[-1] if chain_ids else None, open_lens, active_focus)
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
                acc_optics = {"getter"}
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
                if (n_cov_target <= 1 and reaches_goal) or (n_cov_target > 1 and len(cov_cl) >= n_cov_target):
                    full_score = quality + (0.6 if reaches_goal else 0.0)
                    if best_full_complete is None:
                        first_full_exp = expansions
                        best_full_complete = (full_score, chain_ids, step_clauses)
                    elif full_score > best_full_complete[0]:
                        best_full_complete = (full_score, chain_ids, step_clauses)
                min_steps_for_multi = (
                    min(len(clause_top_cells), max(2, n_clauses // 3))
                    if (n_clauses > 2 and len(clause_top_cells) >= 3)
                    else 1
                )
                if reaches_goal and len(chain_ids) >= min_steps_for_multi and (n_clauses <= 1 or len(cov_cl) >= max(1, n_clauses // 2)):
                    if best_complete is None or quality > best_complete[0]:
                        best_complete = (quality, chain_ids, step_clauses)

            if len(chain_ids) >= max_chain_len:
                continue

            num_path_sinks_used = sum(
                1 for cid in chain_ids
                if is_sink_map[cid] and cand_needs_path_map[cid]
            )
            path_sink_budget = max(1, len(dest_file_literals))
            if (
                chain_ids
                and is_sink_map[chain_ids[-1]]
                and cand_needs_path_map[chain_ids[-1]]
                and num_path_sinks_used >= path_sink_budget
            ):
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
                if cand_needs_path_map[cid] and not is_sink_map[cid]
            )

            # Precompute O(1) integer bitmask of current pool producers in gamma when empty_sub & raw sigs
            use_gamma_mask = False
            gamma_mask = 0
            if cell_req_prod_masks and not getattr(sigma, "mappings", None):
                use_gamma_mask = True
                for g_item in gamma:
                    if not (type(g_item) is tuple and len(g_item) == 3):
                        use_gamma_mask = False
                        break
                    g_osig, g_oport, g_pc = g_item
                    if g_pc is None or g_oport is None or g_osig is not getattr(g_oport, "signature", g_oport):
                        use_gamma_mask = False
                        break
                    gb = cid_bit.get(g_pc.cell_id)
                    if gb is None:
                        use_gamma_mask = False
                        break
                    gamma_mask |= gb

            # Evaluate firing feasibility once per pool cell for this state
            can_fire_map: Dict[str, Optional[Any]] = {}
            for c in pool:
                cid_c = c.cell_id
                if use_gamma_mask:
                    rmasks = cell_req_prod_masks.get(cid_c)
                    if rmasks is not None:
                        blocked_by_mask = False
                        for rm in rmasks:
                            if (gamma_mask & rm) == 0:
                                blocked_by_mask = True
                                break
                        if blocked_by_mask:
                            can_fire_map[cid_c] = None
                            continue
                can_fire_map[cid_c] = _can_fire_under_gamma(
                    c,
                    gamma,
                    sigma,
                    unify_cache=unify_cache,
                    edge_cache=edge_cache,
                    reg=reg,
                    req_ins=cell_req_ins_map[cid_c],
                    sig_key_cache=sig_key_cache,
                    tv_cache=tv_cache,
                    unify_empty_sub_cache=unify_empty_sub_cache,
                    empty_pair_cache=empty_pair_cache,
                    tv_id_cache=tv_id_cache,
                    kept_sigs=kept_sigs,
                )

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
            prereq_bridge_cids: Set[str] = set()
            unlock_clause_hits: Dict[str, Set[TargetId]] = {
                tc.cell_id: (cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set)
                for tc in blocked_targets
            }

            def _missing_req_ins_for(t_cell: Cell) -> List[Any]:
                cid_t = t_cell.cell_id
                cached_m = blocked_missing_ins.get(cid_t)
                if cached_m is not None:
                    return cached_m
                req_list = cell_req_ins_map[cid_t]
                rmasks = cell_req_prod_masks.get(cid_t) if use_gamma_mask else None
                if rmasks is not None:
                    miss = [
                        ip for idx_ip, (_, ip) in enumerate(req_list)
                        if (gamma_mask & rmasks[idx_ip]) == 0
                        or not any(
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
                else:
                    miss = [
                        ip for _, ip in req_list
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
                        if (
                            any(pc.cell_id in port_can_receive_from.get(id(ip), ()) for ip in missing_ins)
                            if port_can_receive_from
                            else any(
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
                            )
                        ):
                            unlock_clause_hits[pc.cell_id] = (
                                unlock_clause_hits.get(pc.cell_id, set())
                                | tc_prop_hits
                                | (cell_clause_hits.get(pc.cell_id, set()) - cov_cl_set)
                            )
                            prereq_bridge_cids.add(pc.cell_id)
                            if pc.cell_id not in unlock_depth or depth_hop > 0 and not (cell_clause_hits.get(pc.cell_id, set()) - cov_cl_set):
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

            repairer_fireable_now = bool(
                needed_prop_repairs
                and any(
                    can_fire_map.get(rc.cell_id) is not None
                    and any(cell_props_map[rc.cell_id].get(rp) == rv for rp, rv in needed_prop_repairs.items())
                    for rc in pool
                    if rc.cell_id not in used_set
                )
            )

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
                    if tc.cell_id in used_set and is_sink_map[tc.cell_id]:
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
                    if (
                        any(
                            gc.cell_id in port_can_receive_from.get(id(ip), ())
                            for tc in next_target_anchors
                            for _, ip in cell_req_ins_map[tc.cell_id]
                        )
                        if port_can_receive_from
                        else any(
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
                        )
                    ):
                        getter_unlocks_next_focus = True
                        break

            # Precompute per-popped-state inverted producer lookup tables (replaces inner loops over blocked_chain_targets)
            state_downstream_unblock_cids: Optional[Set[str]] = None
            state_connects_gamma_cids: Optional[Set[str]] = None
            state_unlock_ng: Optional[Dict[str, Tuple[float, float]]] = None
            state_unlock_g: Optional[Dict[str, Tuple[float, float]]] = None
            if port_can_receive_from:
                state_downstream_unblock_cids = set()
                state_connects_gamma_cids = set()
                gamma_cids = {
                    gs[2].cell_id
                    for gs in gamma
                    if isinstance(gs, tuple) and len(gs) == 3 and gs[2] is not None
                } if gamma else set()
                for tc in blocked_chain_targets:
                    tc_cid = tc.cell_id
                    tc_req = cell_req_ins_map.get(tc_cid, ())
                    if not tc_req:
                        continue
                    if unlock_depth.get(tc_cid, 0) > 0 or tc_cid in prereq_bridge_cids:
                        for _, ip in tc_req:
                            state_downstream_unblock_cids.update(port_can_receive_from.get(id(ip), ()))
                    elif len(tc_req) >= 2 and gamma:
                        # Which required ports of tc are fed by current gamma?
                        port_gamma_hit = [
                            any(
                                (
                                    not (isinstance(gs, tuple) and len(gs) == 3 and gs[2] is not None and gs[2].cell_id in pool_by_id)
                                    or gs[2].cell_id in port_can_receive_from.get(id(ip_j), ())
                                )
                                and _unify_cached(
                                    gs[0] if (isinstance(gs, tuple) and len(gs) == 3) else gs,
                                    getattr(ip_j, "signature", ip_j),
                                    out_port=(gs[1] if (isinstance(gs, tuple) and len(gs) == 3) else None),
                                    in_port=ip_j,
                                    prod_cell=(gs[2] if (isinstance(gs, tuple) and len(gs) == 3) else None),
                                    cons_cell=tc,
                                )
                                for gs in gamma
                            )
                            for _, ip_j in tc_req
                        ]
                        total_hits = sum(1 for h in port_gamma_hit if h)
                        if total_hits >= 1:
                            for idx_i, (tk_name, ip_i) in enumerate(tc_req):
                                if total_hits > (1 if port_gamma_hit[idx_i] else 0) or any(
                                    port_gamma_hit[idx_j] and tc_req[idx_j][0] != tk_name
                                    for idx_j in range(len(tc_req))
                                ):
                                    state_downstream_unblock_cids.update(port_can_receive_from.get(id(ip_i), ()))
                if gamma_cids:
                    frontier_gamma_tc: List[Tuple[str, Any]] = []
                    seen_gamma_tc: Set[str] = set()
                    for tc_b in blocked_chain_targets:
                        if tc_b.cell_id in seen_gamma_tc:
                            continue
                        req_b = cell_req_ins_map.get(tc_b.cell_id, ())
                        if not req_b:
                            continue
                        if any(
                            (recv_b := port_can_receive_from.get(id(ip_b)))
                            and (gamma_cids & recv_b)
                            and any(
                                isinstance(gs, tuple) and len(gs) == 3
                                and gs[2] is not None and gs[2].cell_id in recv_b
                                and _unify_cached(
                                    gs[0],
                                    getattr(ip_b, "signature", ip_b),
                                    out_port=gs[1],
                                    in_port=ip_b,
                                    prod_cell=gs[2],
                                    cons_cell=tc_b,
                                )
                                for gs in gamma
                            )
                            for _, ip_b in req_b
                        ):
                            seen_gamma_tc.add(tc_b.cell_id)
                            frontier_gamma_tc.append((tc_b.cell_id, req_b))
                    # Propagate gamma connectivity across multi-hop missing prerequisites
                    for _ in range(3):
                        if not frontier_gamma_tc:
                            break
                        next_frontier_gtc: List[Tuple[str, Any]] = []
                        for ft_cid, ft_req in frontier_gamma_tc:
                            miss_ips = blocked_missing_ins.get(ft_cid)
                            if miss_ips is None:
                                miss_ips = [ip_u for _, ip_u in ft_req]
                            for ip_u in miss_ips:
                                recv_u = port_can_receive_from.get(id(ip_u), ())
                                state_connects_gamma_cids.update(recv_u)
                                for prod_cid in recv_u:
                                    if prod_cid not in seen_gamma_tc and (
                                        unlock_depth.get(prod_cid, 0) > 0 or prod_cid in prereq_bridge_cids
                                    ):
                                        seen_gamma_tc.add(prod_cid)
                                        next_frontier_gtc.append(
                                            (prod_cid, cell_req_ins_map.get(prod_cid, ()))
                                        )
                        frontier_gamma_tc = next_frontier_gtc

                state_unlock_ng = {}
                state_unlock_g = {}

                def _record_unlock_target(tc_obj: Any, getter_only: bool) -> None:
                    tc_cid = tc_obj.cell_id
                    tc_ins = blocked_missing_ins.get(tc_cid)
                    if tc_ins is None:
                        tc_ins = [ip for _, ip in cell_req_ins_map[tc_cid]]
                    if not tc_ins:
                        return
                    prod_cids: Set[str] = set()
                    for ip in tc_ins:
                        prod_cids.update(port_can_receive_from.get(id(ip), ()))
                    if not prod_cids:
                        return
                    depth_decay = 0.78 ** unlock_depth.get(tc_cid, 0)
                    tc_new = unlock_clause_hits.get(
                        tc_cid,
                        cell_clause_hits.get(tc_cid, set()) - cov_cl_set,
                    )
                    tc_next_hits = [t for t in tc_new if next_target is not None and t == next_target]
                    tc_lex_val = cell_lex_score.get(tc_cid, 0.0)
                    if tc_next_hits:
                        tc_str = max(clause_cell_strength.get((tc_cid, t), 0.5) for t in tc_next_hits)
                    else:
                        tc_str = 0.0
                    # Non-getter branch
                    if not getter_only:
                        if tc_next_hits and not next_uncl_fireable:
                            uv_ng = depth_decay * (0.70 + 0.90 * tc_str + 0.25 * tc_lex_val)
                            cr_ng = 0.65
                        elif tc_next_hits:
                            uv_ng = depth_decay * (0.20 * tc_str + 0.15 * tc_lex_val)
                            cr_ng = 0.0
                        else:
                            uv_ng = depth_decay * (0.35 * tc_lex_val + 0.20 * len(tc_new))
                            cr_ng = 0.0
                        for pid in prod_cids:
                            prev = state_unlock_ng.get(pid)
                            if prev is None:
                                state_unlock_ng[pid] = (uv_ng, cr_ng)
                            else:
                                state_unlock_ng[pid] = (uv_ng if uv_ng > prev[0] else prev[0], cr_ng if cr_ng > prev[1] else prev[1])
                    # Getter branch
                    if tc_next_hits and (not next_uncl_fireable or getter_unlocks_next_focus):
                        uv_g = depth_decay * (0.70 + 0.90 * tc_str + 0.25 * tc_lex_val) + 0.45
                        cr_g = 0.65
                    elif tc_next_hits:
                        uv_g = depth_decay * (0.20 * tc_str + 0.15 * tc_lex_val)
                        cr_g = 0.0
                    else:
                        uv_g = depth_decay * (0.35 * tc_lex_val + 0.20 * len(tc_new)) + 0.20
                        cr_g = 0.0
                    for pid in prod_cids:
                        prev = state_unlock_g.get(pid)
                        if prev is None:
                            state_unlock_g[pid] = (uv_g, cr_g)
                        else:
                            state_unlock_g[pid] = (uv_g if uv_g > prev[0] else prev[0], cr_g if cr_g > prev[1] else prev[1])

                seen_unlock_cids: Set[str] = set()
                for tc in blocked_chain_targets:
                    if tc.cell_id not in seen_unlock_cids:
                        seen_unlock_cids.add(tc.cell_id)
                        _record_unlock_target(tc, getter_only=False)
                if getter_unlocks_next_focus:
                    for tc in next_target_anchors:
                        if tc.cell_id not in seen_unlock_cids:
                            seen_unlock_cids.add(tc.cell_id)
                            _record_unlock_target(tc, getter_only=True)

            e_valid: List[Tuple[Cell, Any, float, float, float, Optional[int], Optional[TargetId]]] = []
            for cand in pool:
                optic_r = cell_optic_role_map.get(cand.cell_id, "")
                is_repeatable_optic = optic_r in rep_optics
                cand_new_cl_reuse = cell_clause_hits.get(cand.cell_id, set()) - cov_cl_set
                if cand.cell_id in used_set:
                    reuse_lex_ok = any((cand.cell_id, t_r) in cell_target_has_lex for t_r in cand_new_cl_reuse)
                    is_pure_constructor = not any(
                        self.is_handle_port(ip) for _, ip in cell_req_ins_map[cand.cell_id]
                    )
                    repeat_cap = (
                        max(3, n_clauses)
                        if (is_repeatable_optic or is_ingress_map[cand.cell_id] or is_pure_constructor)
                        else max(2, n_clauses // 4)
                    )
                    if (
                        is_sink_map[cand.cell_id]
                        or (not is_repeatable_optic and (not cand_new_cl_reuse or not reuse_lex_ok))
                        or (cand.cell_id == last_cid and not cand_new_cl_reuse)
                        or chain_ids.count(cand.cell_id) >= repeat_cap
                    ):
                        continue
                    if (
                        cand.cell_id == last_cid
                        and not is_repeatable_optic
                        and bool(cell_req_ins_map[cand.cell_id])
                        and (
                            not cand_new_cl_reuse
                            or all(
                                any(
                                    tc.cell_id != cand.cell_id
                                    and tc.cell_id not in used_set
                                    and t_r in (cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set)
                                    for tc in pool
                                )
                                for t_r in cand_new_cl_reuse
                            )
                        )
                    ):
                        continue
                    if (
                        n_clauses > 1
                        and is_ingress_map[cand.cell_id]
                        and not is_sink_map[cand.cell_id]
                        and next_target is not None
                        and next_target[0] == n_clauses - 1
                        and any(
                            t_r[0] == n_clauses - 1
                            and any(
                                is_sink_map[tc.cell_id]
                                and tc.cell_id not in used_set
                                and t_r in (cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set)
                                for tc in pool
                            )
                            for t_r in cand_new_cl_reuse
                        )
                    ):
                        continue
                # Suppress file-writing sinks when prompt has no egress intent
                if (
                    is_sink_map[cand.cell_id]
                    and not has_egress
                    and cand_needs_path_map[cand.cell_id]
                ):
                    continue
                if (
                    is_sink_map[cand.cell_id]
                    and cand_needs_path_map[cand.cell_id]
                    and num_path_sinks_used >= path_sink_budget
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
                best_target: Optional[TargetId] = (
                    min(
                        eff_new_hits,
                        key=lambda t: (
                            0 if (next_target is not None and t == next_target) else 1,
                            -clause_cell_strength.get((cand.cell_id, t), 0.0),
                            t[0],
                            t[1],
                        ),
                    )
                    if eff_new_hits
                    else None
                )
                if n_clauses >= 3 and best_target is not None:
                    cl_cf = cell_clause_cf.get((cand.cell_id, best_target[0]), (0.0, 0.0, 0.0))
                    port_frac = cell_clause_port_frac.get((cand.cell_id, best_target[0]), 0.0)
                    prior_sc = (
                        0.40 * cell_lex_score.get(cand.cell_id, 0.0)
                        + 1.80 * cl_cf[0]
                        + 0.90 * cl_cf[1]
                        + 0.55 * port_frac
                        + (0.35 if sub_anchor_owner.get(best_target) == cand.cell_id else 0.0)
                        + (0.35 if (cand.cell_id, best_target) in cell_target_has_lex else -0.35)
                    )
                else:
                    prior_sc = cell_lex_score.get(cand.cell_id, 0.0)
                cand_dom = cell_domain_map[cand.cell_id]
                if last_dom and cand_dom == last_dom and (eff_new_hits or cand.cell_id in goal_ids):
                    prior_sc += 0.25
                unlock_credit = 0.0
                order_cost_penalty = 0.0
                if (
                    is_sink_map[cand.cell_id]
                    and n_clauses > 2
                    and best_target is not None
                ):
                    if (
                        best_target[0] == 0
                        and any(
                            not is_sink_map[tc.cell_id]
                            and best_target in (cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set)
                            for tc in pool
                        )
                    ) or (
                        best_target[0] < n_clauses - 1
                        and any(
                            not is_sink_map[tc.cell_id]
                            and can_fire_map.get(tc.cell_id) is not None
                            and best_target in (cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set)
                            for tc in pool
                        )
                    ):
                        continue
                    if (
                        chain_ids
                        and next_target is not None
                        and best_target[0] > next_target[0]
                        and (next_uncl_fireable or next_uncl_unlockable)
                    ):
                        continue
                    if chain_ids and best_target[0] < max(0, n_clauses - 2):
                        has_later_fireable_non_sink = any(
                            not is_sink_map[tc.cell_id]
                            and not (tc.cell_id in used_set and is_sink_map[tc.cell_id])
                            and can_fire_map[tc.cell_id] is not None
                            and any(t[0] > best_target[0] for t in (cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set))
                            for tc in pool
                        )
                        if has_later_fireable_non_sink and best_target != next_target:
                            continue
                if (
                    last_dom
                    and cand_dom
                    and cand_dom != last_dom
                    and best_target is not None
                    and any(
                        cell_domain_map.get(tc.cell_id, "") == last_dom
                        and best_target in (cell_clause_hits.get(tc.cell_id, set()) - cov_cl_set)
                        for tc in pool
                        if not (tc.cell_id in used_set and is_sink_map[tc.cell_id])
                    )
                ):
                    prior_sc -= 0.85
                    order_cost_penalty += 0.75
                step_cl_idx: Optional[int] = (
                    best_target[0]
                    if best_target is not None
                    else (max(0, max_cl_seen) if is_closing_open_setter else None)
                )

                next_uncl_hits = [t for t in eff_new_hits if next_target is not None and t == next_target]
                if next_uncl_hits:
                    cl_str = max(clause_cell_strength.get((cand.cell_id, t), 0.5) for t in next_uncl_hits)
                    prior_sc += 0.75 + 1.60 * cl_str
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
                        prior_sc -= 2.00
                        order_cost_penalty += 2.00

                if cand.cell_id in bwd_dist:
                    prior_sc += 0.20 / (1.0 + bwd_dist[cand.cell_id])
                cand_needs_path = cand_needs_path_map[cand.cell_id]
                cand_has_handle_out = cell_has_handle_out_map[cand.cell_id]
                cand_has_composite_out = cell_has_composite_out_map[cand.cell_id]
                gamma_has_composite = bool(
                    gamma
                    and reg is not None
                    and any(
                        reg.is_composite_carrier(
                            str(getattr(gs[0] if (isinstance(gs, tuple) and len(gs) == 3) else gs, "type_name", "") or "").strip().lower()
                        )
                        for gs in gamma
                        if gs is not None
                    )
                )
                cand_unlocks_prereq = bool(
                    state_connects_gamma_cids is not None and cand.cell_id in state_connects_gamma_cids
                )
                if (
                    not chain_ids
                    and n_clauses > 1
                    and is_sink_map[cand.cell_id]
                    and any(
                        not is_sink_map[c.cell_id] and can_fire_map[c.cell_id] is not None
                        for c in pool
                    )
                ):
                    continue
                # Once all prompt-declared input files are ingested into active composite carriers in gamma,
                # zero-input Stage-1 synthetic dataset generators must not hijack the pipeline mid-chain.
                if (
                    chain_ids
                    and is_ingress_map[cand.cell_id]
                    and not cand_needs_path
                    and not cell_req_ins_map[cand.cell_id]
                    and not cand_has_handle_out
                    and cand_has_composite_out
                    and gamma_has_composite
                    and src_file_literals
                    and num_sources_used >= len(src_file_literals)
                ):
                    continue
                if (
                    chain_ids
                    and is_ingress_map[cand.cell_id]
                    and not cand_needs_path
                    and not cell_req_ins_map[cand.cell_id]
                    and not cand_has_handle_out
                    and not cand_unlocks_prereq
                    and src_file_literals
                    and num_sources_used >= len(src_file_literals)
                    and not (
                        eff_new_hits
                        and any(
                            not any(
                                not is_ingress_map.get(tcid, False)
                                for tcid in clause_top_cells.get(cl_t, ())
                            )
                            for cl_t in eff_new_hits
                        )
                    )
                ):
                    continue
                is_zero_in_reader = is_ingress_map[cand.cell_id] or (
                    not cell_req_ins_map[cand.cell_id] and cand_needs_path
                )
                if not chain_ids and is_zero_in_reader:
                    if has_compat_src_in_pool:
                        if cand_needs_path and self.file_compat(cand, src_file_literals[0]) < 0:
                            continue
                        if cand_needs_path:
                            prior_sc += 0.90
                        else:
                            prior_sc -= 0.75
                    else:
                        prior_sc += 0.45
                elif chain_ids and not cell_req_ins_map[cand.cell_id]:
                    # Zero-input non-path dataset generators (e.g. make_blobs, load_*) must not hijack
                    # an already-started pipeline mid-chain unless directly requested by an uncovered clause
                    # or acting as a typed prerequisite unlocker connected to active gamma.
                    if is_ingress_map[cand.cell_id] and not cand_needs_path and not new_hits and not cand_unlocks_prereq:
                        continue
                    # Zero-input sources/generators mid-chain are valid when either:
                    # (a) an unmatched external source file remains in the prompt, OR
                    # (b) the non-path constructor/generator produces a carrier needed by a blocked/uncovered downstream cell.
                    has_more_files = bool(
                        is_zero_in_reader
                        and cand_needs_path
                        and len(src_file_literals) > num_sources_used
                        and (not src_file_literals or self.file_compat(cand, src_file_literals[min(num_sources_used, len(src_file_literals) - 1)]) >= 0)
                    )
                    has_compat_dest_handle_file = bool(
                        cand_needs_path
                        and cand_has_handle_out
                        and cand.cell_id not in used_set
                        and dest_file_literals
                        and any(self.file_compat(cand, df) >= 1 for df in dest_file_literals)
                    )
                    unblocks_downstream = False
                    if (not cand_needs_path or has_compat_dest_handle_file) and not has_more_files:
                        if _is_comb:
                            unblocks_downstream = bool(eff_new_hits)
                        else:
                            cand_outs = cell_out_ports_map.get(cand.cell_id, ())
                            is_stage1_source = is_ingress_map[cand.cell_id] and (
                                getattr(cand, "stage", None) == 1
                                or str(getattr(cand, "node_role", "") or "").strip().lower() in ("source", "ingress")
                            )
                            unblocks_downstream = (
                                (
                                    bool(eff_new_hits)
                                    and (
                                        not is_stage1_source
                                        or cand_has_handle_out
                                        or not cand_has_composite_out
                                    )
                                )
                                if not has_compat_dest_handle_file
                                else False
                            ) or (
                                (bool(cand_outs and (cand.cell_id in state_downstream_unblock_cids or cand_unlocks_prereq)))
                                if state_downstream_unblock_cids is not None
                                else any(
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
                            )
                    if not has_more_files and not unblocks_downstream:
                        continue
                    if has_compat_dest_handle_file:
                        prior_sc += 0.90
                    elif (
                        compat_dest_handle_types
                        and is_ingress_map[cand.cell_id]
                        and not cand_needs_path
                        and cand_has_handle_out
                        and any(
                            str(getattr(osig_dh, "type_name", "") or "").strip().lower() in compat_dest_handle_types
                            for _, osig_dh in cell_out_ports_map.get(cand.cell_id, ())
                        )
                    ):
                        prior_sc -= 0.75
                        order_cost_penalty += 0.50
                    if is_ingress_map[cand.cell_id] and not cand_needs_path and not eff_new_hits:
                        # Disallow zero-hit synthetic generators mid-chain when disconnected from active gamma
                        cand_outs_chk = cell_out_ports_map.get(cand.cell_id, ())
                        if state_connects_gamma_cids is not None:
                            connects_gamma = bool(cand_outs_chk and cand.cell_id in state_connects_gamma_cids)
                        else:
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
                        prior_sc += 0.15 if (n_clauses >= 3 and eff_new_hits) else 0.35
                    elif (
                        last_gamma_items
                        and cand_req_ins
                        and is_sink_map[cand.cell_id]
                        and not is_sink_map[pool_by_id[last_cid].cell_id]
                    ):
                        prior_sc -= 0.90
                        order_cost_penalty += 0.75
                if chain_ids and is_sink_map[cand.cell_id]:
                    stage2_gamma_items = [
                        gs for gs in gamma
                        if isinstance(gs, tuple) and len(gs) == 3 and gs[2] is not None
                        and not is_ingress_map.get(gs[2].cell_id, False)
                        and not is_sink_map.get(gs[2].cell_id, False)
                    ]
                    cand_req_s2 = cell_req_ins_map[cand.cell_id]
                    if stage2_gamma_items and cand_req_s2:
                        consumes_stage2 = any(
                            _unify_cached(
                                gs[0],
                                getattr(ip, "signature", ip),
                                out_port=gs[1],
                                in_port=ip,
                                prod_cell=gs[2],
                                cons_cell=cand,
                            )
                            for gs in stage2_gamma_items
                            for _, ip in cand_req_s2
                        )
                        if not consumes_stage2:
                            prior_sc -= 2.25
                            order_cost_penalty += 2.00

                # Bridge/optic & multi-hop precondition unlock bonus: reward carriers satisfying
                # missing inputs of blocked targets (or blocked prerequisite initializers).
                best_unlock = 0.0
                if cand_out_pairs:
                    if state_unlock_ng is not None and state_unlock_g is not None:
                        unlock_entry = (state_unlock_g if optic_r == "getter" else state_unlock_ng).get(cand.cell_id)
                        if unlock_entry is not None:
                            best_unlock, unlock_credit = unlock_entry
                            prior_sc += min(1.85 if optic_r == "getter" else 1.45, best_unlock)
                            if unlock_credit > 0.0 and order_cost_penalty > 0.0 and not new_hits:
                                order_cost_penalty = 0.0
                    else:
                        unlock_targets = list(blocked_chain_targets)
                        if getter_unlocks_next_focus and optic_r == "getter":
                            for tc in next_target_anchors:
                                if tc not in unlock_targets:
                                    unlock_targets.append(tc)
                        if unlock_targets:
                            for tc in unlock_targets:
                                tc_ins = blocked_missing_ins.get(tc.cell_id)
                                if tc_ins is None:
                                    tc_ins = [ip for _, ip in cell_req_ins_map[tc.cell_id]]
                                if tc_ins and (
                                    any(cand.cell_id in port_can_receive_from.get(id(ip), ()) for ip in tc_ins)
                                    if port_can_receive_from
                                    else any(
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
                                    )
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
                elif not eff_new_hits and best_unlock > 0.0 and not optic_r and not is_sink_map[cand.cell_id]:
                    req_carrier_names = {pk for pk, _ in cell_req_ins_map[cand.cell_id]}
                    ungrounded_cfg_cnt = sum(
                        1
                        for pk, ip_cfg in (getattr(cand, "inputs", {}) or {}).items()
                        if pk not in req_carrier_names
                        and getattr(ip_cfg, "required", True)
                        and getattr(ip_cfg, "default_value", None) is None
                        and not is_path_port(ip_cfg)
                        and _pointed_default(ip_cfg, cand) is None
                    )
                    if ungrounded_cfg_cnt > 0:
                        prior_sc -= 1.80 * ungrounded_cfg_cnt
                        order_cost_penalty += 1.80 * ungrounded_cfg_cnt
                    # Non-bridge transforms acting purely as structural unlockers pay more than declared bridges
                    prior_sc -= 0.40
                    order_cost_penalty += 0.25

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
                if is_closing_open_setter and not focus_modified:
                    continue
                if needed_prop_repairs and prop_repair_bonus <= 0.0 and optic_r == "setter":
                    if repairer_fireable_now:
                        continue
                    prior_sc -= 2.10
                    order_cost_penalty += 2.10
                elif open_lens and optic_r == "setter":
                    prior_sc += 1.85 if focus_modified else 0.65
                    unlock_credit = max(unlock_credit, 0.70 if focus_modified else 0.35)
                elif open_lens and focus_modified and not prop_repair_bonus and next_target is not None and (
                    not new_hits or min(t[0] for t in new_hits) > max(0, max_cl_seen)
                ):
                    continue

                # Do not materialize file-path sinks when the prompt declares neither a destination file nor explicit egress hits,
                # or when every declared destination file rejects this sink's supported formats.
                if is_sink_map[cand.cell_id] and cand_needs_path:
                    if not dest_file_literals and not (has_egress and eff_new_hits):
                        continue
                    if dest_file_literals and all(self.file_compat(cand, df) < 0 for df in dest_file_literals):
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
                if cell_missing_dep_map.get(cand.cell_id, False):
                    has_installed_alt = (
                        any(clause_has_installed.get(cl_t, False) for cl_t in cand_hits)
                        or (is_ingress_map[cand.cell_id] and cand_dom in installed_ingress_in_dom)
                        or (is_sink_map[cand.cell_id] and cand_dom in installed_sink_in_dom)
                    )
                    if has_installed_alt:
                        prior_sc -= 3.25
                        order_cost_penalty += 3.25

                e_valid.append((cand, next_sigma, prior_sc, unlock_credit, order_cost_penalty, step_cl_idx, best_target))

            if not e_valid:
                continue
            # Causal clause monotonicity pruning: when the earliest uncovered clause target
            # (`next_target`) is directly fireable without a focus-switch getter or open lens
            # closure, restrict expansion to candidates advancing `next_target` (eliminates k!
            # out-of-order permutations on long multi-clause chains).
            if (
                n_clauses > 2
                and not getter_unlocks_next_focus
                and not open_lens
                and next_target is not None
            ):
                if next_uncl_fireable:
                    in_order_valid = [item for item in e_valid if item[6] == next_target]
                    if in_order_valid:
                        e_valid = in_order_valid
                else:
                    unblock_cids_for_next: Set[str] = set()
                    if next_target_anchors and port_can_receive_from:
                        for tc_n in next_target_anchors:
                            for _, ip_n in cell_req_ins_map.get(tc_n.cell_id, ()):
                                unblock_cids_for_next.update(port_can_receive_from.get(id(ip_n), ()))
                    if not (unblock_cids_for_next and any(item[0].cell_id in unblock_cids_for_next for item in e_valid)):
                        fireable_targets = [item[6] for item in e_valid if item[6] is not None]
                        if fireable_targets:
                            eff_next = min(fireable_targets)
                            in_order_valid = [item for item in e_valid if item[6] == eff_next]
                            if in_order_valid:
                                e_valid = in_order_valid
            valid_frontier_sizes.append(len(e_valid))
            e_valid.sort(key=lambda x: x[2], reverse=True)
            shortlist = e_valid[: max(branch_factor * 2, 8)]

            # Schema-constrained SLM choice ranking strictly over E_valid(Gamma_t)
            choice_scores: Dict[str, float] = {}
            if (
                use_choice_scorer
                and len(shortlist) > 1
                and (shortlist[0][2] - shortlist[1][2]) < 0.45
            ):
                gamma_desc = ", ".join(
                    sorted({
                        str(getattr(s[0] if isinstance(s, tuple) else s, "type_name", s))
                        for s in gamma
                        if s is not None
                    })
                ) or "empty (start)"
                cand_ids_tuple = tuple(c.cell_id for c, _, _, _, _, _, _ in shortlist)
                cache_key = (cand_ids_tuple, gamma_desc)
                if cache_key in _choice_score_cache:
                    choice_scores = _choice_score_cache[cache_key]
                elif (
                    scorer_calls < max_scorer_calls
                    and scorer_wall_s < 4.5
                    and not (n_clauses >= 10 and scorer_calls >= 2)
                ):
                    scorer_calls += 1
                    _t_sc0 = time.perf_counter()
                    choice_scores = mm.score_choices(
                        prompt,
                        list(cand_ids_tuple),
                        context_desc=f"chain={[x for x in chain_ids]}, available_carriers=[{gamma_desc}]",
                    ) or {}
                    _dt_sc = time.perf_counter() - _t_sc0
                    scorer_wall_s += _dt_sc
                    # Compensate search deadline so slow synchronous SLM calls don't starve A* expansions.
                    _search_deadline += min(_dt_sc, 6.0)
                    _choice_score_cache[cache_key] = choice_scores

            scored_children: List[Tuple[float, float, Cell, Any, float, Optional[int], Optional[TargetId]]] = []
            for cand, next_sigma, base_prior, unlock_cr, ord_pen, st_cl, bt_tgt in shortlist:
                slm_sc = float(choice_scores.get(cand.cell_id, 0.0))
                new_cl = cell_clause_hits.get(cand.cell_id, set()) - cov_cl_set
                step_cov_cnt = 1 if (n_clauses >= 3 and bt_tgt is not None) else len(new_cl)
                step_gain = base_prior + 0.65 * slm_sc + 0.30 * step_cov_cnt
                step_cost = max(0.10, 0.15 + 0.75 / (1.0 + max(0.0, step_gain))) + ord_pen
                scored_children.append((step_cost, step_gain, cand, next_sigma, unlock_cr, st_cl, bt_tgt))

            scored_children.sort(key=lambda x: x[1], reverse=True)
            for step_cost, step_gain, cand, next_sigma, unlock_cr, st_cl, bt_tgt in scored_children[:branch_factor]:
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
                cand_new_cl = ({bt_tgt} if bt_tgt is not None else set()) if not closing_open_only else set()
                next_cov = cov_cl if closing_open_only else tuple(sorted(cov_cl_set | cand_new_cl))

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

        chosen_ids = chosen_entry[1] if chosen_entry is not None else None
        chosen_clauses = chosen_entry[2] if chosen_entry is not None else ()

        if (
            chosen_ids
            and not src_file_literals
            and not dest_file_literals
            and all(cand_needs_path_map.get(cid, False) for cid in chosen_ids)
        ):
            _ch_seen: Set[str] = set()
            for _cid in chosen_ids:
                _ch_seen |= (prompt_toks & cell_id_toks_map.get(_cid, set()))
            if len(_ch_seen) / max(1, len(prompt_toks)) < 0.12:
                chosen_ids = None
                chosen_clauses = ()

        # Honest coverage accounting. The search silently shrinks its coverage target to the
        # clauses that have at least one candidate (`coverable_clauses`) and can accept a plan that
        # covers only half of them. Never let that pass unreported: record exactly which prompt
        # clauses the returned plan does not serve, and which of those had no candidate at all.
        if chosen_ids and n_clauses > 0:
            _served = {c for c in chosen_clauses if c is not None}
            _coverable_idx = {t[0] for t in coverable_clauses}
            _uncovered = sorted(set(range(n_clauses)) - _served)
            if _uncovered:
                self._trace(
                    "omega_incomplete_plan",
                    complete=bool(best_full_complete is not None and chosen_entry is best_full_complete),
                    n_clauses=n_clauses,
                    uncovered_clauses=_uncovered,
                    uncovered_without_candidates=[i for i in _uncovered if i not in _coverable_idx],
                    uncovered_with_candidates=[i for i in _uncovered if i in _coverable_idx],
                )

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
            else:
                setattr(c_cl, "_auto_inserted_bridge", True)
            result.append(c_cl)
        for i_r, c_cl in enumerate(result):
            if getattr(c_cl, "matched_clause_idx", None) is None:
                for nxt_r in result[i_r + 1 :]:
                    if getattr(nxt_r, "matched_clause_idx", None) is not None:
                        c_cl.matched_clause_idx = nxt_r.matched_clause_idx
                        break
        return result


__all__ = ["OmegaTypeDirectedAStarRouteMethod"]
