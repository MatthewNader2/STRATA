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


class TokenEvidence:
    def __init__(self, orchestrator: Any):
        self.orch = orchestrator
        cells = list(getattr(orchestrator, "loaded_cells", {}).values())
        self.n = max(len(cells), 1)
        df: Dict[str, int] = {}
        identity_df: Dict[str, int] = {}
        for c in cells:
            for t in c.token_set:
                df[t] = df.get(t, 0) + 1
            for t in getattr(c, "identity_tokens", ()) or ():
                identity_df[t] = identity_df.get(t, 0) + 1
        self._df = df
        self._identity_df = identity_df
        self._idf_cache: Dict[str, float] = {}
        self._skip: Optional[Set[str]] = None

    def idf(self, tok: str) -> float:
        """0.0 for tokens no cell can match; log-IDF otherwise."""
        v = self._idf_cache.get(tok)
        if v is None:
            d = self._df.get(tok, 0)
            v = 0.0 if d == 0 else math.log(1.0 + (self.n + 1) / (d + 1.0))
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
                skip |= set(reg.get_verification_semantics("generic_prepositions") or ())
            except Exception:
                pass
            self._skip = skip
        return self._skip

    def clause_tokens(self, clause: str) -> Set[str]:
        skip = self._non_evidence()
        toks = {
            t for t in CellTokenizer.tokenize_prompt(clause)
            if len(t) > 2 and self.idf(t) > 0.0 and self._identity_df.get(t, 0) > 0 and t not in skip
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
        ident = set(getattr(cell, "identity_tokens", None) or set())
        hit = clause_toks & ident
        hit_mass = self.mass(hit)
        c_mass = self.mass(clause_toks)
        i_mass = self.mass(ident)
        recall = hit_mass / c_mass if c_mass > 0 else 0.0
        precision = hit_mass / i_mass if i_mass > 0 else 0.0
        denom = precision + recall
        f = (2 * precision * recall / denom) if denom > 0 else 0.0
        # How completely the clause NAMES the operation (cell id tokens only).
        # Distinguishes fft from ifftshift when both share the same keywords:
        # the clause says "fft", the id of ifftshift also says "shift".
        name_toks = {t for t in CellTokenizer.tokenize_identifier(str(getattr(cell, "cell_id", ""))) if self.idf(t) > 0.0}
        n_mass = self.mass(name_toks)
        name_precision = (self.mass(clause_toks & name_toks) / n_mass) if n_mass > 0 else 0.0

        return {"hit": hit_mass, "recall": recall, "precision": precision, "f": f,
                "name_precision": name_precision,
                "rank": (recall, name_precision, f),
                "tokens": sorted(hit)}


import threading

_CACHE: Dict[int, "TokenEvidence"] = {}
_LOCK = threading.RLock()


def last_orchestrator() -> Any:
    """Deprecated: orchestrators are now explicitly threaded."""
    return None


def get_evidence(orchestrator: Any) -> Optional[TokenEvidence]:
    """Per-orchestrator singleton, rebuilt if the loaded cell count changes."""
    if orchestrator is None or not getattr(orchestrator, "loaded_cells", None):
        return None
    key = id(orchestrator)
    with _LOCK:
        ev = _CACHE.get(key)
        if ev is None or ev.n != max(len(orchestrator.loaded_cells), 1):
            ev = TokenEvidence(orchestrator)
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


def audit_clause_coverage(cells: List[Any], prompt: str, orchestrator: Any = None) -> Dict[str, Any]:
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
    out["available"] = True

    # 1. evidence winner of every clause among the path's cells
    winners: Dict[int, int] = {}
    for j in range(len(clauses)):
        best_i, best_rank = None, None
        for i in range(len(cells)):
            o = table[i][j]
            if o and o["hit"] > 0.0 and (best_rank is None or o["rank"] > best_rank):
                best_i, best_rank = i, o["rank"]
        if best_i is not None:
            winners[j] = best_i

    # 2. which clauses does each cell CLAIM? An explicit assignment (planner / tagger) is authoritative;
    #    otherwise a cell claims the clauses it wins plus those where it explains tokens the winner left over.
    bridge_roles = set()
    try:
        from lattice import TypeRegistry
        bridge_roles = {str(r).lower() for r in TypeRegistry.get_instance().get_verification_semantics("bridge_roles")}
    except Exception:
        try:
            from .lattice import TypeRegistry
            bridge_roles = {str(r).lower() for r in TypeRegistry.get_instance().get_verification_semantics("bridge_roles")}
        except Exception:
            pass
    claims: List[set] = []
    cell_rows: List[Dict[str, Any]] = []
    for i, cell in enumerate(cells):
        touched = [j for j in range(len(clauses)) if table[i][j] and table[i][j]["hit"] > 0.0]
        won = [j for j in touched if winners.get(j) == i]
        residual_serves = []
        for j in touched:
            if j in won:
                continue
            w = table[winners[j]][j]
            if ev.identity_overlap(ctoks[j] - set(w["tokens"]), cell)["hit"] > 0.0:
                residual_serves.append(j)
        tagged = getattr(cell, "matched_clause_idx", None)
        mine = {tagged} if tagged is not None else (set(won) | set(residual_serves))
        claims.append(mine)
        role = str(getattr(cell, "node_role", "") or "").lower()
        if mine:
            status = "serves"
        elif not touched or role in bridge_roles:
            status = "connector"
        else:
            status = "UNREQUESTED"
        cell_rows.append({"cell": cell.cell_id, "wins": sorted(set(won) | ({tagged} if tagged is not None else set())),
                          "residual_serves": residual_serves, "touches": touched, "status": status})
        if status == "UNREQUESTED":
            out["issues"].append({"kind": "unrequested_cell", "cell": cell.cell_id, "outranked_on": touched,
                                  "winners": {j: cells[winners[j]].cell_id for j in touched if j in winners}})
    out["cells"] = cell_rows

    # 3. a clause is SERVED only by the cells assigned to it, jointly explaining as much of it as the
    #    best single lattice cell could (incidental hits by other cells no longer count: an optimiser
    #    exploits a whole-path union immediately).
    for j, toks in enumerate(ctoks):
        claimed = [i for i in range(len(cells)) if j in claims[i] and table[i][j]]
        lattice_best = lattice_best_fit = 0.0
        if toks:
            memo = ev.__dict__.setdefault("_lattice_best_memo", {})
            key = frozenset(toks)
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
        best_i = winners.get(j)
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

