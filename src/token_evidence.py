"""
src/token_evidence.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)

Single, corpus-derived source of lexical evidence. Every consumer that asks
"how strongly does this clause point at this cell?" (router, route methods,
clause tagging, preflight coverage audit) goes through here, so the answer is
identical everywhere and contains NO vocabulary: all weights come from the
loaded lattice's own token index (inverse document frequency over cells).

Principles
  * A token that no loaded cell can ever match carries zero evidence (it is
    filler for THIS lattice: "called", "then", "it", ...). No stoplist needed.
  * A token's weight is its IDF over the cell corpus.
  * A cell is judged by IDENTITY evidence only (operation id + declared
    keywords). Docstring/port vocabulary is weak evidence and never claims a
    clause on its own.
  * Precision matters as much as recall: a cell whose identity is mostly
    NOT mentioned by the clause (get_dummies for "get ...") is a poor claim.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Set, List, Tuple

try:
    from .tokenizer import CellTokenizer
except (ImportError, ValueError):
    from tokenizer import CellTokenizer
try:
    from .config import settings
except (ImportError, ValueError):
    from config import settings


def _is_profile_zero() -> bool:
    try:
        from .inference import ModelManager
    except (ImportError, ValueError):
        try:
            from inference import ModelManager
        except Exception:
            return False
    try:
        return getattr(ModelManager.get_instance(), "current_profile_name", None) == "0"
    except Exception:
        return False


class TokenEvidence:
    def __init__(self, orchestrator: Any, pure_symbolic: Optional[bool] = None):
        self.orch = orchestrator
        self._pure_symbolic = _is_profile_zero() if pure_symbolic is None else bool(pure_symbolic)
        cells = list(getattr(orchestrator, "loaded_cells", {}).values())
        self.n = max(len(cells), 1)
        df: Dict[str, int] = {}
        identity_df: Dict[str, int] = {}
        formal_df: Dict[str, int] = {}
        for c in cells:
            for t in c.token_set:
                df[t] = df.get(t, 0) + 1
            for t in getattr(c, "identity_tokens", ()) or ():
                identity_df[t] = identity_df.get(t, 0) + 1
            for t in getattr(c, "formal_identifiers", ()) or (str(getattr(c, "cell_id", "")).lower(),):
                formal_df[t] = formal_df.get(t, 0) + 1
        self._df = formal_df if self._pure_symbolic else df
        self._identity_df = formal_df if self._pure_symbolic else identity_df
        self._idf_cache: Dict[str, float] = {}
        self._skip: Optional[Set[str]] = None
        self._clause_tok_cache: Dict[str, Set[str]] = {}
        self._cell_ev_cache: Dict[str, Tuple[Any, ...]] = {}
        self._clause_mass_cache: Dict[Tuple[str, ...], float] = {}

    def idf(self, tok: str) -> float:
        """0.0 for tokens no cell can match in neural/hybrid profiles; strict OOV penalty mass in Profile 0."""
        v = self._idf_cache.get(tok)
        if v is None:
            d = self._df.get(tok, 0)
            if d == 0:
                v = math.log(1.0 + (self.n + 1) / 1.0) if self._pure_symbolic else 0.0
            else:
                v = math.log(1.0 + (self.n + 1) / (d + 1.0))
            self._idf_cache[tok] = v
        return v

    def mass(self, toks: Iterable[str]) -> float:
        return sum(self.idf(t) for t in toks)

    def _non_evidence(self) -> Set[str]:
        """Words the TREES declare as carrying no operation identity."""
        if self._skip is None:
            skip: Set[str] = set()
            try:
                from lattice import TypeRegistry
                reg = TypeRegistry.get_instance()
                skip |= set(reg.get_prompt_filler_tokens())
                skip |= set(reg.get_sentence_connectives())
                skip |= set(reg.get_function_words())
                skip |= set(reg.get_generic_prepositions())
            except Exception:
                pass
            self._skip = skip
        return self._skip

    def clause_tokens(self, clause: str) -> Set[str]:
        cached = self._clause_tok_cache.get(clause)
        if cached is not None:
            return set(cached)
        res = self._compute_clause_tokens(clause)
        if len(self._clause_tok_cache) < 8192:
            self._clause_tok_cache[clause] = set(res)
        return res

    def _compute_clause_tokens(self, clause: str) -> Set[str]:
        if self._pure_symbolic:
            # Strict Profile 0: preserve every token (zero-IDF conversational prose is NOT stripped),
            # only preserving compound formal identifiers without keyword/synonym expansion.
            return {t for t in CellTokenizer.tokenize_prompt(clause) if len(t) > 1}
        skip = self._non_evidence()
        toks = {
            t for t in CellTokenizer.tokenize_prompt(clause)
            if len(t) > 2
            and self.idf(t) > 0.0
            and (not self._identity_df or self._identity_df.get(t, 0) > 0)
            and t not in skip
        }
        try:
            from unification import ExecutionContext
            lits = ExecutionContext._extract_universal_literals(clause)
            clause_chars = list(clause)
            for lit in lits:
                pos, kind, val = lit
                if kind in ("quoted_str", "file_asset"):
                    end_idx = min(len(clause_chars), getattr(lit, "end_pos", pos + len(val)))
                    for k in range(pos, end_idx):
                        clause_chars[k] = " "
            unquoted = "".join(clause_chars)
            unquoted_toks = {t.lower() for t in CellTokenizer.tokenize_prompt(unquoted)}

            lit_toks = set()
            for lit in lits:
                _, kind, val = lit
                if kind == "file_asset":
                    for part in str(val).split("."):
                        for t in CellTokenizer.tokenize_identifier(part):
                            t_low = t.lower()
                            if t_low not in unquoted_toks:
                                lit_toks.add(t_low)
                elif kind in ("quoted_str", "numeric"):
                    for t in CellTokenizer.tokenize_identifier(str(val)):
                        t_low = t.lower()
                        if t_low not in unquoted_toks:
                            lit_toks.add(t_low)
            if lit_toks:
                rem = toks - lit_toks
                if rem:
                    return rem
        except Exception:
            pass
        return toks

    def identity_overlap(self, clause_toks: Set[str], cell: Any) -> Dict[str, float]:
        """Evidence a clause gives for a cell.
        recall    = share of the clause's mass explained by the cell identity
        precision = share of the cell identity's mass mentioned by the clause
        """
        c_key = tuple(sorted(clause_toks))
        c_mass = self._clause_mass_cache.get(c_key)
        if c_mass is None:
            c_mass = self.mass(c_key)
            if len(self._clause_mass_cache) < 8192:
                self._clause_mass_cache[c_key] = c_mass

        cid = str(getattr(cell, "cell_id", "") or "")
        cell_ev = self._cell_ev_cache.get(cid) if cid else None
        idf_fn = self.idf

        if self._pure_symbolic:
            if cell_ev is None:
                ident = set(getattr(cell, "formal_identifiers", None) or (cid.lower(),))
                i_mass = sum(idf_fn(t) for t in ident)
                cell_ev = (ident, i_mass)
                if cid:
                    self._cell_ev_cache[cid] = cell_ev
            else:
                ident, i_mass = cell_ev
            hit = clause_toks & ident
            hit_mass = sum(idf_fn(t) for t in hit)
            recall = hit_mass / c_mass if c_mass > 0 else 0.0
            precision = hit_mass / i_mass if i_mass > 0 else 0.0
            denom = precision + recall
            f = (2 * precision * recall / denom) if denom > 0 else 0.0
            name_prec = 1.0 if hit else 0.0
            return {
                "hit": hit_mass,
                "recall": recall,
                "precision": precision,
                "f": f,
                "name_precision": name_prec,
                "rank": (recall, name_prec, f),
                "tokens": sorted(hit),
            }

        if cell_ev is None:
            ident = set(getattr(cell, "identity_tokens", None) or ())
            i_mass = sum(idf_fn(t) for t in ident)
            name_toks = {t for t in CellTokenizer.tokenize_identifier(cid) if idf_fn(t) > 0.0}
            n_mass = sum(idf_fn(t) for t in name_toks)
            cell_ev = (ident, i_mass, name_toks, n_mass)
            if cid:
                self._cell_ev_cache[cid] = cell_ev
        else:
            ident, i_mass, name_toks, n_mass = cell_ev

        hit = clause_toks & ident
        hit_mass = sum(idf_fn(t) for t in hit) if hit else 0.0
        recall = hit_mass / c_mass if c_mass > 0 else 0.0
        precision = hit_mass / i_mass if i_mass > 0 else 0.0
        denom = precision + recall
        f = (2 * precision * recall / denom) if denom > 0 else 0.0
        # How completely the clause NAMES the operation (cell id tokens only).
        # Distinguishes fft from ifftshift when both share the same keywords:
        # the clause says "fft", the id of ifftshift also says "shift".
        if n_mass > 0.0:
            name_hit = clause_toks & name_toks
            name_precision = (sum(idf_fn(t) for t in name_hit) / n_mass) if name_hit else 0.0
        else:
            name_precision = 0.0

        return {"hit": hit_mass, "recall": recall, "precision": precision, "f": f,
                "name_precision": name_precision,
                "rank": (recall, name_precision, f),
                "tokens": sorted(hit)}


import threading

_CACHE: Dict[Tuple[int, bool], "TokenEvidence"] = {}
_LOCK = threading.RLock()


def last_orchestrator() -> Any:
    """Deprecated: orchestrators are now explicitly threaded."""
    return None


def get_evidence(orchestrator: Any) -> Optional[TokenEvidence]:
    """Per-orchestrator, per-profile-mode singleton, rebuilt if the loaded cell count changes."""
    if orchestrator is None or not getattr(orchestrator, "loaded_cells", None):
        return None
    is_p0 = _is_profile_zero()
    key = (id(orchestrator), is_p0)
    with _LOCK:
        ev = _CACHE.get(key)
        if ev is None or ev.n != max(len(orchestrator.loaded_cells), 1) or ev._pure_symbolic != is_p0:
            ev = TokenEvidence(orchestrator, pure_symbolic=is_p0)
            _CACHE[key] = ev
        return ev


def anchors_for_clause(ev: "TokenEvidence", clause: str, pool: Iterable[Any],
                       relevance: Optional[Dict[str, float]] = None, on_event: Any = None,
                       clause_idx: Optional[int] = None) -> list:
    """
    The one anchor-selection rule shared by every route method.

    Rank pool cells by identity evidence (recall, name-precision, F), then retrieval relevance.
    The best cell is the clause's primary anchor. A clause is one operation (the segmenter splits
    on conjunctions/sequencing), so another cell joins only if it explains residual clause tokens
    at least as well as the primary did. Returns [(cell, overlap_dict), ...]; empty when no pool
    cell carries identity evidence (callers decide what to do and should say so).
    """
    relevance = relevance or {}
    toks = ev.clause_tokens(clause)
    scored = []
    for c in pool:
        o = ev.identity_overlap(toks, c)
        if o["hit"] > 0.0:
            scored.append((o["rank"], relevance.get(c.cell_id, 0.0), c, o))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    accepted: list = []
    covered: Set[str] = set()
    primary_f = None
    for rank, rel, c, o in scored:
        if not accepted:
            primary_f = o["f"]
            covered |= set(o["tokens"])
            ok, why = True, "primary (best rank: recall, name-precision, F)"
        else:
            residual = ev.identity_overlap(toks - covered, c)
            min_recall = settings.secondary_anchor_min_recall
            min_f = settings.secondary_anchor_min_f
            f_ratio = settings.secondary_anchor_f_ratio
            min_name_prec = getattr(settings, "secondary_anchor_min_name_precision", 0.20)
            ok = residual["hit"] > 0.0 and residual["recall"] >= min_recall and residual["name_precision"] >= min_name_prec and (
                residual["f"] >= min_f
                or (primary_f is not None and residual["f"] >= primary_f * f_ratio)
            )
            # Single-token residual verbs (e.g. "apply") require strong dominance or high name specificity
            # so they don't falsely split single-operation clauses.
            if ok and len(residual["tokens"]) < 2:
                ok = bool(
                    (primary_f is not None and residual["f"] >= primary_f * 0.70)
                    or (residual["name_precision"] >= 0.25 and residual["recall"] >= 0.50)
                )
            if ok and accepted:
                # Category-theoretic reachability: a secondary anchor in the same clause
                # must compose with or share a carrier with an accepted anchor in the clause,
                # preventing spurious linguistic homonyms from pulling in disconnected domains.
                orch = ev.orch
                prim_cell = accepted[0][0]
                try:
                    from unification import is_edge_valid
                except (ImportError, ValueError):
                    from .unification import is_edge_valid

                f12 = any(is_edge_valid(out_p, in_p, prim_cell, c) for out_p in prim_cell.outputs.values() for in_p in c.inputs.values())
                f21 = any(is_edge_valid(out_p, in_p, c, prim_cell) for out_p in c.outputs.values() for in_p in prim_cell.inputs.values())
                t1 = {getattr(getattr(p, "signature", p), "type_name", "") for p in prim_cell.inputs.values()}
                t2 = {getattr(getattr(p, "signature", p), "type_name", "") for p in c.inputs.values()}
                share_in = bool(t1 & t2)

                is_ing = (getattr(c, "stage", None) == 1 or getattr(c, "node_role", "") == "ingress") and (clause_idx is not None and clause_idx > 0)

                if not (f12 or f21 or share_in) or is_ing:
                    ok = False
                    why = "secondary: no categorical composition with primary anchor %s" % prim_cell.cell_id
                else:
                    why = "secondary: residual hit=%.3f, recall=%.3f, name_prec=%.3f, F=%.3f accepted" % (
                        residual["hit"], residual["recall"], residual["name_precision"], residual["f"]
                    )
            else:
                why = "secondary: residual hit=%.3f, recall=%.3f, F=%.3f %s" % (
                    residual["hit"], residual["recall"], residual["f"], "accepted" if ok else "rejected"
                )
            if ok:
                covered |= set(residual["tokens"])
        if on_event is not None:
            on_event("anchor_candidate", clause_idx=clause_idx, cell=c.cell_id,
                     precision=round(o["precision"], 3), recall=round(o["recall"], 3),
                     name_precision=round(o["name_precision"], 3), f=round(o["f"], 3),
                     relevance=round(rel, 3), tokens=o["tokens"], accepted=ok, reason=why)
        if ok:
            accepted.append((c, o))
    return accepted


def _clauses(prompt: str) -> List[str]:
    try:
        from route_methods.base import RouteMethod
    except (ImportError, ValueError):
        try:
            from .route_methods.base import RouteMethod
        except (ImportError, ValueError):
            return [prompt]
    return RouteMethod.segment_prompt_clauses(prompt)


def _resolve_clause_semantic_scores(
    orchestrator: Any,
    prompt: str,
    clause_semantic_scores: Optional[List[Dict[str, float]]] = None,
) -> Optional[List[Dict[str, float]]]:
    if clause_semantic_scores is not None:
        return clause_semantic_scores
    if orchestrator is None:
        return None
    cached = getattr(orchestrator, "_last_clause_semantic_scores", None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == (prompt or "").strip():
        return cached[1]
    return None


def _prop_val(entry: Any, key: str, default: Any = None) -> Any:
    return entry.get(key, default) if isinstance(entry, dict) else getattr(entry, key, default)


def _is_prerequisite_for_downstream_server(idx: int, cells: List[Any], claims: List[Set[int]]) -> bool:
    """True when `cells[idx]` repairs a property conflict or produces a non-generic stateful/handle
    output required by a downstream serving cell `cells[j]` (`j > idx`)."""
    cell = cells[idx]
    downstream_servers = [cells[j] for j in range(idx + 1, len(cells)) if claims[j]]
    if not downstream_servers:
        return False

    # 1. Property-repair connector (e.g., NUMPY_ABS resolving is_complex=False after FFT)
    c_posts = [
        (_prop_val(p, "property"), _prop_val(p, "operator"), _prop_val(p, "value"))
        for p in (getattr(cell, "postconditions", None) or ())
        if _prop_val(p, "property") is not None and _prop_val(p, "operator") == "=="
    ]
    if c_posts:
        prior_posts = set()
        for prev in cells[:idx]:
            for p in (getattr(prev, "postconditions", None) or ()):
                pr = _prop_val(p, "property")
                if pr is not None and _prop_val(p, "operator") == "==":
                    prior_posts.add((pr, _prop_val(p, "value")))
        for ds in downstream_servers:
            for dp in (getattr(ds, "preconditions", None) or ()):
                d_prop, d_op, d_val = _prop_val(dp, "property"), _prop_val(dp, "operator"), _prop_val(dp, "value")
                if d_op == "==" and d_prop is not None:
                    repaired = any(cp == d_prop and cv == d_val for cp, _, cv in c_posts)
                    conflicted = any(pp == d_prop and pv != d_val for pp, pv in prior_posts)
                    if repaired and conflicted:
                        return True

    # 2. Stateful / model handle lifecycle producer derived from TypeRegistry & typestates
    role = str(getattr(cell, "node_role", "") or "").lower()
    _te_reg = None
    try:
        try:
            from .lattice import TypeRegistry
        except (ImportError, ValueError):
            from lattice import TypeRegistry
        _te_reg = TypeRegistry.get_instance()
    except Exception:
        _te_reg = None

    _lifecycle_roles: Set[str] = set()
    _stateful_roots: Set[str] = set()
    _handle_struct_kinds: Set[str] = {"handle"}
    if _te_reg is not None:
        _lifecycle_roles = _te_reg.get_lifecycle_roles_set()
        _stateful_roots |= (
            _te_reg.get_types_with_property("is_stateful_instance", True)
            | _te_reg.get_types_by_kind("model")
            | _te_reg.get_types_by_kind("visual")
            | _te_reg.get_roles_for_category("strict")
        )

    is_lifecycle_role = bool(role and role in _lifecycle_roles)
    c_outs = list((getattr(cell, "outputs", None) or {}).values())
    prior_out_sigs = set()
    for prev in cells[:idx]:
        for o in (getattr(prev, "outputs", None) or {}).values():
            osig = getattr(o, "signature", o)
            ot = str(getattr(osig, "type_name", "") or "").strip().lower()
            ost = str(getattr(osig, "state", "") or "").strip().lower()
            if ot:
                prior_out_sigs.add((ot, ost))

    for o in c_outs:
        osig = getattr(o, "signature", o)
        ot = str(getattr(osig, "type_name", "") or "").strip().lower()
        ost = str(getattr(osig, "state", "") or "").strip().lower()
        osk = str(getattr(osig, "structure_kind", "") or "").strip().lower()
        oat = str(getattr(osig, "abstract_type", "") or "").strip().lower()
        opr = str(getattr(o, "port_role", "") or getattr(osig, "port_role", "") or "").strip().lower()
        if not ot:
            continue
        is_stateful_or_handle = (
            is_lifecycle_role
            or (osk and osk in _handle_struct_kinds)
            or (oat and oat in _handle_struct_kinds)
            or (
                _te_reg is not None
                and (
                    _te_reg.is_subtype(ot, "handle")
                    or _te_reg.is_subtype(opr, "handle")
                    or any(_te_reg.is_subtype(ot, sr) for sr in _stateful_roots)
                    or (bool(ost) and not _te_reg.is_declared_top_state(ost) and bool(role and role in (_te_reg.get_ingress_roles_set() | _te_reg.get_bridge_roles_set() | _lifecycle_roles)))
                )
            )
        )
        if not is_stateful_or_handle:
            continue
        if (ot, ost) in prior_out_sigs:
            continue
        for ds in downstream_servers:
            for inp in (getattr(ds, "inputs", None) or {}).values():
                if not getattr(inp, "required", True):
                    continue
                isig = getattr(inp, "signature", inp)
                it = str(getattr(isig, "type_name", "") or "").strip().lower()
                ist = str(getattr(isig, "state", "") or "").strip().lower()
                if it == ot and (not ist or ist == ost):
                    return True
    return False


def audit_clause_coverage(
    cells: List[Any],
    prompt: str,
    orchestrator: Any = None,
    clause_semantic_scores: Optional[List[Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Prompt-clause <-> pipeline-cell coverage audit. Corpus-derived (token_evidence), vocabulary-free.
    Used by PreflightLinter (hard checks) and debug panels.
    """
    out: Dict[str, Any] = {"available": False, "clauses": [], "cells": [], "issues": []}
    ev = get_evidence(orchestrator)
    if ev is None or not prompt or not cells:
        return out
    clauses = _clauses(prompt)
    ctoks = [ev.clause_tokens(c) for c in clauses]
    table = [[ev.identity_overlap(t, c) if t else None for t in ctoks] for c in cells]
    clause_sem = None if ev._pure_symbolic else _resolve_clause_semantic_scores(orchestrator, prompt, clause_semantic_scores)
    if clause_sem is not None and len(clause_sem) != len(clauses):
        clause_sem = None
    out["available"] = True

    # 1. evidence winner of every clause among the path's cells
    winners: Dict[int, int] = {}
    sem_winners: Dict[int, int] = {}
    for j in range(len(clauses)):
        best_i, best_rank = None, None
        for i in range(len(cells)):
            o = table[i][j]
            if o and o["hit"] > 0.0 and (best_rank is None or o["rank"] > best_rank):
                best_i, best_rank = i, o["rank"]
        if best_i is not None:
            winners[j] = best_i
        elif clause_sem is not None:
            sem_map = clause_sem[j] or {}
            max_sem_j = max(sem_map.values(), default=0.0)
            if max_sem_j >= 0.35:
                best_si, best_sval = None, 0.0
                for i, c in enumerate(cells):
                    sv = float(sem_map.get(c.cell_id, 0.0) or 0.0)
                    if sv >= 0.35 and sv >= 0.70 * max_sem_j and sv > best_sval:
                        best_si, best_sval = i, sv
                if best_si is not None:
                    sem_winners[j] = best_si

    # 2. which clauses does each cell CLAIM? An explicit assignment (planner / tagger) is authoritative;
    #    otherwise a cell claims the clauses it wins plus those where it explains tokens the winner left over.
    bridge_roles = set()
    _te_reg_b = None
    try:
        from lattice import TypeRegistry
        _te_reg_b = TypeRegistry.get_instance()
        bridge_roles = _te_reg_b.get_bridge_roles_set()
    except Exception:
        try:
            from .lattice import TypeRegistry
            _te_reg_b = TypeRegistry.get_instance()
            bridge_roles = _te_reg_b.get_bridge_roles_set()
        except Exception:
            pass
    try:
        from .schema import is_bridge_cell as _schema_is_bridge_cell
    except (ImportError, ValueError):
        from schema import is_bridge_cell as _schema_is_bridge_cell
    claims: List[set] = []
    cell_rows: List[Dict[str, Any]] = []
    for i, cell in enumerate(cells):
        touched = [j for j in range(len(clauses)) if table[i][j] and table[i][j]["hit"] > 0.0]
        won = [j for j in touched if winners.get(j) == i]
        sem_won = [j for j, wi in sem_winners.items() if wi == i]
        residual_serves = []
        for j in touched:
            if j in won:
                continue
            w = table[winners[j]][j]
            if ev.identity_overlap(ctoks[j] - set(w["tokens"]), cell)["hit"] > 0.0:
                residual_serves.append(j)
        tagged = getattr(cell, "matched_clause_idx", None)
        tagged_valid = (
            {tagged}
            if (tagged is not None and 0 <= tagged < len(table[i]) and table[i][tagged]["hit"] > 0.0)
            else set()
        )
        mine = tagged_valid | set(won) | set(sem_won) | set(residual_serves)
        claims.append(mine)
        role = str(getattr(cell, "node_role", "") or "").lower()
        is_structural_connector = (
            _schema_is_bridge_cell(cell, _te_reg_b)
            or bool(getattr(cell, "_auto_inserted_bridge", False))
            or bool(getattr(cell, "_is_prerequisite_producer", False))
            or getattr(cell, "stage", None) in (1, 3)
        )
        if mine:
            status = "serves"
        elif (
            (not touched and is_structural_connector)
            or role in bridge_roles
            or getattr(cell, "_auto_inserted_bridge", False)
            or getattr(cell, "_is_prerequisite_producer", False)
        ):
            status = "connector"
        else:
            status = "UNREQUESTED"
        cell_rows.append({"cell": cell.cell_id, "wins": sorted(set(won) | set(sem_won) | ({tagged} if tagged is not None else set())),
                          "residual_serves": residual_serves, "touches": touched, "status": status})

    # Promote structural property-repair / stateful lifecycle prerequisites that directly enable a downstream server
    for i, row in enumerate(cell_rows):
        if row["status"] == "UNREQUESTED" and _is_prerequisite_for_downstream_server(i, cells, claims):
            row["status"] = "connector"
        if row["status"] == "UNREQUESTED":
            out["issues"].append({"kind": "unrequested_cell", "cell": row["cell"], "outranked_on": row["touches"],
                                  "winners": {j: cells[winners[j]].cell_id for j in row["touches"] if j in winners}})
    out["cells"] = cell_rows

    # 3. a clause is SERVED only by the cells assigned to it, jointly explaining as much of it as the
    #    best single lattice cell could (incidental hits by other cells no longer count: an optimiser
    #    exploits a whole-path union immediately).
    for j, toks in enumerate(ctoks):
        claimed = [i for i in range(len(cells)) if j in claims[i] and table[i][j]]
        lattice_best = lattice_best_fit = 0.0
        if toks:
            memo = ev.__dict__.setdefault("_lattice_best_memo", {})
            key = tuple(sorted(toks))
            if key not in memo:
                lb = lbf = 0.0
                for lc in getattr(ev.orch, "loaded_cells", {}).values():
                    lo = ev.identity_overlap(toks, lc)
                    # fit = how much of the clause the cell explains x how completely the clause NAMES the cell
                    lb, lbf = max(lb, lo["recall"]), max(lbf, lo["recall"] * lo["name_precision"])
                memo[key] = (lb, lbf)
            lattice_best, lattice_best_fit = memo[key]
        c_mass = ev.mass(toks)
        path_tokens: set = set()
        for i in claimed:
            path_tokens |= set(table[i][j]["tokens"])
        path_best = (ev.mass(path_tokens) / c_mass) if c_mass > 0 else 0.0
        quality = (max(table[i][j]["recall"] * table[i][j]["name_precision"] for i in claimed) / lattice_best_fit) if (claimed and lattice_best_fit > 0) else 0.0
        if not toks or lattice_best <= 0.0:
            status = "no-evidence" if not toks else "unservable"
        else:
            status = "served" if path_best >= lattice_best - 1e-9 else "UNSERVED"
        if status != "served" and clause_sem is not None and j < len(clause_sem):
            sem_map = clause_sem[j] or {}
            max_sem_j = max(sem_map.values(), default=0.0)
            claimed_any = [i for i in range(len(cells)) if j in claims[i]]
            claimed_sem = max((float(sem_map.get(cells[i].cell_id, 0.0) or 0.0) for i in claimed_any), default=0.0)
            sem_floor = 0.70 if (path_best <= 0.0 and lattice_best >= 0.50) else 0.35
            sem_rel = 0.88 if (path_best <= 0.0 and lattice_best >= 0.50) else 0.70
            if claimed_sem >= sem_floor and max_sem_j > 0.0 and claimed_sem >= sem_rel * max_sem_j:
                status = "served"
                path_best = max(path_best, claimed_sem)
                quality = max(quality, claimed_sem / max_sem_j)
        best_i = winners.get(j, sem_winners.get(j))
        out["clauses"].append({
            "idx": j, "text": clauses[j], "tokens": sorted(toks),
            "winner": cells[best_i].cell_id if best_i is not None else None,
            "claimed_by": [cells[i].cell_id for i in claimed],
            "path_recall": round(path_best, 3), "lattice_best_recall": round(lattice_best, 3),
            "quality": round(min(quality, 1.0), 3), "status": status})
        if status == "UNSERVED":
            out["issues"].append({"kind": "clause_unserved", "clause_idx": j, "clause": clauses[j],
                                  "path_recall": round(path_best, 3), "lattice_best_recall": round(lattice_best, 3),
                                  "best_on_path": cells[best_i].cell_id if best_i is not None else None})
    total_cl = len(out["clauses"])
    served_cl = sum(1 for c in out["clauses"] if c.get("status") in ("served", "no-evidence"))
    out["coverage"] = (served_cl / total_cl) if total_cl > 0 else 1.0
    return out


audit = audit_clause_coverage

