"""
src/coalgebraic_unfolder.py - Recursive Coalgebraic Subgoal Unfolding Engine.

Models goal decomposition as a coalgebraic anamorphism (unfold):
    psi_C : Goal -> (G_1, ..., G_k, combinator?)

Given a computational goal g and an active typed lattice C:
1. Compute the Epistemic Grounding Coverage Score kappa(g, C) in [0, 1] using
   domain-agnostic token coverage, structural clause dispersion, and optional
   dense similarity over the declared cells of C.
2. When kappa(g, C) >= theta_closed (or depth >= max_depth), g is treated as
   an atomic/closed subgoal directly grounded in C.
3. When kappa(g, C) < theta_closed and depth < max_depth, extract a dynamic,
   domain-agnostic Lattice Schema Summary (carrier types T, typestate
   transitions sigma_in -> sigma_out, dynamically discovered macro combinators,
   and relevant operations) and invoke schema-constrained SLM unfolding
   alpha_C(g) -> (G_1, ..., G_k, combinator), recursing on every child subgoal
   with cycle detection and bounded depth (default d_max = 3).
4. Ground leaf goals and combinator slots purely into verified lattice cells
   and macro combinator `bound_slots` (never unconstrained raw code).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

try:
    from .tokenizer import CellTokenizer
except (ImportError, ValueError):
    from tokenizer import CellTokenizer


def _normalize_goal_key(text: str) -> str:
    """Normalizes a goal string deterministically for cycle detection."""
    toks = sorted(CellTokenizer.tokenize_prompt(text))
    if toks:
        return " ".join(toks)
    return " ".join(str(text or "").strip().lower().split())


def _count_clauses(text: str) -> int:
    """Domain-agnostic syntactic clause count using punctuation and tokenizer segments."""
    if not text or not text.strip():
        return 0
    try:
        try:
            from .router import _segment_prompt_clauses
        except (ImportError, ValueError):
            from router import _segment_prompt_clauses
        clauses = [c.strip() for c in _segment_prompt_clauses(text) if c and c.strip()]
        if clauses:
            return len(clauses)
    except Exception:
        pass
    parts = [p.strip() for p in re.split(r"[.;\n]+", text) if p.strip()]
    return max(len(parts), 1)


def is_macro_combinator_cell(cell: Any) -> bool:
    """
    Dynamically identifies whether a lattice Cell is a higher-order macro/combinator
    without hardcoding domain names or cell IDs.
    """
    if cell is None:
        return False
    node_type = str(getattr(cell, "node_type", "") or "").lower()
    node_role = str(getattr(cell, "node_role", "") or "").lower()
    slots = getattr(cell, "slots", None)
    inputs = getattr(cell, "inputs", {}) or {}
    if node_type == "macro":
        return True
    if isinstance(slots, dict) and any(str(k) not in inputs for k in slots.keys()):
        return True
    if node_role == "combinator" and isinstance(slots, dict) and bool(slots):
        return True
    return False


def get_macro_slot_names(cell: Any) -> List[str]:
    """Returns declared sub-pipeline slot names on a macro/combinator cell."""
    slots = getattr(cell, "slots", None)
    inputs = getattr(cell, "inputs", {}) or {}
    if isinstance(slots, dict):
        sub_slots = [str(k) for k in slots.keys() if str(k) not in inputs]
        return sub_slots if sub_slots else [str(k) for k in slots.keys()]
    if isinstance(slots, (list, tuple)):
        return [str(k) for k in slots if str(k) not in inputs]
    return []


def discover_combinators(
    orchestrator: Any,
    scope_cells: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Dynamically discovers all macro/combinator cells available in the lattice
    (checking `scope_cells` first, then `orchestrator.loaded_cells`).
    """
    combinators: Dict[str, Any] = {}
    sources: List[Sequence[Any]] = []
    if scope_cells:
        sources.append(scope_cells)
    loaded = getattr(orchestrator, "loaded_cells", None) if orchestrator is not None else None
    if isinstance(loaded, dict):
        sources.append(list(loaded.values()))
    for pool in sources:
        for cell in pool:
            cid = getattr(cell, "cell_id", None)
            if cid and is_macro_combinator_cell(cell) and get_macro_slot_names(cell):
                if cid not in combinators:
                    combinators[cid] = cell
    return combinators


def _port_type_and_state(port: Any) -> Tuple[str, str, str]:
    """Extracts (type_name, state, structure_kind) from either PortSignature or PortSchema."""
    if port is None:
        return "", "", ""
    sig = getattr(port, "signature", None)
    type_name = str(
        getattr(sig, "type_name", None)
        or getattr(port, "type_name", None)
        or getattr(port, "carrier_type", None)
        or ""
    ).strip()
    state = str(
        getattr(sig, "state", None)
        or getattr(port, "state", None)
        or ""
    ).strip()
    skind = str(getattr(port, "structure_kind", None) or "").strip()
    return type_name, state, skind


class EpistemicGroundingScorer:
    """
    Computes the Epistemic Grounding Coverage Score kappa(g, C) in [0, 1].

    Measures how well a single cell (or tight local morphism) in lattice C
    closes goal g directly:
      - High kappa (>= theta_closed): g is atomic/closed with strong identity
        token overlap on a single lattice cell and single syntactic clause.
      - Low kappa (< theta_closed): g is open-domain, abstract, or composite
        (spanning multiple distinct operations or abstract intent), requiring
        recursive coalgebraic decomposition.
    """

    def __init__(self, theta_closed: float = 0.55):
        self.theta_closed = float(theta_closed)

    @staticmethod
    def _stem_match(tok_a: str, tok_b: str) -> bool:
        """Purely structural prefix/inflection equivalence without fixed wordlists."""
        if tok_a == tok_b:
            return True
        if len(tok_a) < 3 or len(tok_b) < 3:
            return False
        shorter, longer = (tok_a, tok_b) if len(tok_a) <= len(tok_b) else (tok_b, tok_a)
        if longer.startswith(shorter) and (len(longer) - len(shorter)) <= 3:
            return True
        return False

    def _fuzzy_intersection_count(self, query_toks: Set[str], target_toks: Set[str]) -> float:
        if not query_toks or not target_toks:
            return 0.0
        exact = query_toks & target_toks
        score = float(len(exact))
        rem_q = query_toks - exact
        if not rem_q:
            return score
        for qt in rem_q:
            if any(self._stem_match(qt, tt) for tt in target_toks):
                score += 0.75
        return score

    def score(
        self,
        goal: str,
        orchestrator: Any,
        tunnel: Optional[Sequence[Any]] = None,
        relevance_map: Optional[Dict[str, float]] = None,
    ) -> float:
        if not goal or not goal.strip() or orchestrator is None:
            return 0.0

        loaded = getattr(orchestrator, "loaded_cells", {}) or {}
        scope_cells = list(tunnel) if tunnel else list(loaded.values())
        if not scope_cells:
            return 0.0

        goal_toks = set(CellTokenizer.tokenize_prompt(goal))
        if not goal_toks:
            return 0.0

        n_clauses = _count_clauses(goal)
        rel_map = relevance_map or {}

        best_cell_score = 0.0
        best_id_cov = 0.0
        cell_cover_sets: List[Set[str]] = []

        for cell in scope_cells:
            if getattr(cell, "node_type", "") == "constant":
                continue
            cid = getattr(cell, "cell_id", "") or ""
            id_toks = set(getattr(cell, "identity_tokens", None) or ()) | set(CellTokenizer.tokenize_identifier(cid))
            for kw in getattr(cell, "keywords", ()) or ():
                id_toks.update(CellTokenizer.tokenize_identifier(str(kw)))
            all_toks = set(getattr(cell, "token_set", None) or ()) | id_toks

            id_hits = self._fuzzy_intersection_count(goal_toks, id_toks)
            all_hits = self._fuzzy_intersection_count(goal_toks, all_toks)

            id_cov = min(1.0, id_hits / max(len(goal_toks), 1))
            all_cov = min(1.0, all_hits / max(len(goal_toks), 1))
            id_prec = min(1.0, id_hits / max(len(id_toks), 1)) if id_toks else 0.0

            lex_match = 0.50 * id_cov + 0.40 * all_cov + 0.10 * id_prec
            raw_rel = rel_map.get(cid)
            if raw_rel is not None and rel_map:
                dense_rel = max(0.0, min(1.0, float(raw_rel)))
                single_score = 0.72 * lex_match + 0.28 * dense_rel
            else:
                single_score = lex_match

            if single_score > best_cell_score:
                best_cell_score = single_score
                best_id_cov = max(id_cov, all_cov * 0.9)

            exact_id = goal_toks & id_toks
            if exact_id:
                cell_cover_sets.append(exact_id)

        # Measure multi-action / multi-cell spread only when no single cell
        # already accounts for the bulk of the goal tokens.
        disjoint_groups = 0
        if best_id_cov < 0.70:
            seen_union: Set[str] = set()
            for cs in sorted(cell_cover_sets, key=len, reverse=True):
                if cs - seen_union:
                    disjoint_groups += 1
                    seen_union.update(cs)

        clause_penalty = 1.0 / (1.0 + 0.45 * max(0, n_clauses - 1))
        spread_penalty = 1.0 if best_id_cov >= 0.70 else 1.0 / (1.0 + 0.22 * max(0, disjoint_groups - 1))
        length_penalty = 1.0 if len(goal_toks) <= 6 else 1.0 / (1.0 + 0.08 * (len(goal_toks) - 6))

        kappa = best_cell_score * clause_penalty * spread_penalty * length_penalty
        return max(0.0, min(1.0, float(kappa)))


def extract_lattice_summary(
    orchestrator: Any,
    goal: str = "",
    tunnel: Optional[Sequence[Any]] = None,
    relevance_map: Optional[Dict[str, float]] = None,
    max_ops: int = 36,
) -> Dict[str, Any]:
    """
    Dynamically extracts a domain-agnostic summary of the active lattice C:
      - `domains`: active domain names
      - `carriers`: distinct carrier (type_name, structure_kind) tuples T
      - `transitions`: state transitions (state_in -> state_out) & type flows
      - `combinators`: dynamically discovered control-flow/macro combinators & slots
      - `operations`: balanced sample of relevant cells from active domains
    """
    loaded = getattr(orchestrator, "loaded_cells", {}) or {} if orchestrator is not None else {}
    scope_cells = list(tunnel) if tunnel else list(loaded.values())
    all_cells = list(loaded.values()) if loaded else scope_cells

    goal_toks = set(CellTokenizer.tokenize_prompt(goal)) if goal else set()
    rel_map = relevance_map or {}

    def _op_score(cell: Any) -> float:
        cid = getattr(cell, "cell_id", "") or ""
        id_toks = set(getattr(cell, "identity_tokens", None) or ()) | set(CellTokenizer.tokenize_identifier(cid))
        all_toks = set(getattr(cell, "token_set", None) or ()) | id_toks
        lex = (len(goal_toks & id_toks) * 2.0 + len(goal_toks & all_toks) * 0.5) if goal_toks else 0.0
        return float(rel_map.get(cid, 0.0)) + lex

    # Include top operations overall plus per-domain balance across loaded/tunnel cells
    ranked_scope = sorted(scope_cells, key=_op_score, reverse=True)
    selected_ops: List[Any] = []
    seen_cids: Set[str] = set()

    # Reserve at least top cells per domain present in scope/loaded so multi-domain
    # interactions (e.g. target domain + control_flow) are always visible in summary.
    by_domain: Dict[str, List[Any]] = {}
    for c in ranked_scope:
        dom = str(getattr(c, "domain_name", "") or "generic")
        by_domain.setdefault(dom, []).append(c)

    if by_domain:
        per_dom_cap = max(4, max_ops // max(len(by_domain), 1))
        for dom, d_cells in sorted(by_domain.items()):
            for c in d_cells[:per_dom_cap]:
                cid = getattr(c, "cell_id", "")
                if cid and cid not in seen_cids:
                    selected_ops.append(c)
                    seen_cids.add(cid)

    for c in ranked_scope:
        if len(selected_ops) >= max_ops:
            break
        cid = getattr(c, "cell_id", "")
        if cid and cid not in seen_cids:
            selected_ops.append(c)
            seen_cids.add(cid)

    domains: Set[str] = set()
    carriers: Set[str] = set()
    transitions: Set[str] = set()
    ops_summary: List[Dict[str, Any]] = []

    for c in selected_ops:
        cid = getattr(c, "cell_id", "") or ""
        dom = str(getattr(c, "domain_name", "") or "").strip()
        if dom:
            domains.add(dom)

        inputs = getattr(c, "inputs", {}) or {}
        outputs = getattr(c, "outputs", {}) or {}
        in_types: List[str] = []
        out_types: List[str] = []
        in_states: List[str] = []
        out_states: List[str] = []

        from lattice import TypeRegistry
        _reg_cu = TypeRegistry.get_instance()
        _top_st_cu = set(_reg_cu.get_verification_semantics("top_states") or ()) | set(_reg_cu.get_verification_semantics("universal_top_types") or ())
        if isinstance(inputs, dict):
            for p_name, port in inputs.items():
                t_name, st, sk = _port_type_and_state(port)
                if t_name:
                    carriers.add(f"{t_name}" if not sk else f"{t_name} ({sk})")
                    in_types.append(f"{p_name}:{t_name}")
                if st and st.lower() not in _top_st_cu and not _reg_cu.is_declared_top_state(st):
                    in_states.append(st)

        if isinstance(outputs, dict):
            for p_name, port in outputs.items():
                t_name, st, sk = _port_type_and_state(port)
                if t_name:
                    carriers.add(f"{t_name}" if not sk else f"{t_name} ({sk})")
                    out_types.append(f"{p_name}:{t_name}")
                if st and st.lower() not in _top_st_cu and not _reg_cu.is_declared_top_state(st):
                    out_states.append(st)

        for is_st in (in_states or ["init"]):
            for out_st in out_states:
                if is_st != out_st:
                    transitions.add(f"{is_st} -> {out_st}")

        doc_raw = str(getattr(c, "docstring", None) or getattr(c, "doc", None) or "").strip()
        doc_short = doc_raw.split(".")[0][:120] if doc_raw else ""
        ops_summary.append({
            "cell_id": cid,
            "domain": dom,
            "inputs": in_types[:5],
            "outputs": out_types[:4],
            "doc": doc_short,
        })

    # Dynamically discover ALL macro/combinator cells across the active lattice
    comb_map = discover_combinators(orchestrator, scope_cells=all_cells)
    combinator_summaries: List[Dict[str, Any]] = []
    for cid, comb in sorted(comb_map.items()):
        dom = str(getattr(comb, "domain_name", "") or "").strip()
        if dom:
            domains.add(dom)
        doc_raw = str(getattr(comb, "docstring", None) or getattr(comb, "doc", None) or "").strip()
        combinator_summaries.append({
            "cell_id": cid,
            "topology_type": str(getattr(comb, "topology_type", "macro") or "macro"),
            "slots": get_macro_slot_names(comb),
            "doc": doc_raw.split(".")[0][:120] if doc_raw else "",
        })

    return {
        "domains": sorted(domains),
        "carriers": sorted(carriers)[:25],
        "transitions": sorted(transitions)[:20],
        "combinators": combinator_summaries,
        "operations": ops_summary,
    }


def format_lattice_summary(summary: Dict[str, Any]) -> str:
    """Formats a dynamic lattice schema summary concisely for constrained SLM unfolding."""
    lines: List[str] = []
    domains = summary.get("domains") or []
    if domains:
        lines.append(f"Active domains in scope: {', '.join(domains)}")
    carriers = summary.get("carriers") or []
    if carriers:
        lines.append(f"Available carrier types T: {', '.join(carriers)}")
    transitions = summary.get("transitions") or []
    if transitions:
        lines.append(f"Available typestate transitions: {', '.join(transitions)}")
    combs = summary.get("combinators") or []
    if combs:
        lines.append("Available structural macro combinators (node_type='macro'):")
        for cb in combs:
            lines.append(
                f"  - {cb['cell_id']} [topology={cb['topology_type']}, slots={cb['slots']}]: {cb['doc']}"
            )
    ops = summary.get("operations") or []
    if ops:
        lines.append("Relevant lattice operations:")
        for op in ops:
            in_s = ", ".join(op.get("inputs") or []) or "()"
            out_s = ", ".join(op.get("outputs") or []) or "()"
            lines.append(f"  - {op['cell_id']} ({op['domain']}): ({in_s}) -> ({out_s}) | {op['doc']}")
    return "\n".join(lines)


@dataclass
class CoalgebraNode:
    """Node in the recursive coalgebraic unfolding tree."""
    goal: str
    kind: str = "atomic"  # "atomic" | "seq" | "combinator"
    kappa: float = 0.0
    depth: int = 0
    combinator_cell_id: Optional[str] = None
    children: List["CoalgebraNode"] = field(default_factory=list)
    slot_children: Dict[str, List["CoalgebraNode"]] = field(default_factory=dict)
    grounded_cells: List[Any] = field(default_factory=list)

    def leaf_goals(self) -> List[str]:
        if self.kind == "atomic":
            return [self.goal]
        if self.kind == "seq":
            out: List[str] = []
            for ch in self.children:
                out.extend(ch.leaf_goals())
            return out
        if self.kind == "combinator":
            out = []
            for s_nodes in self.slot_children.values():
                for ch in s_nodes:
                    out.extend(ch.leaf_goals())
            return out or [self.goal]
        return [self.goal]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "goal": self.goal,
            "kind": self.kind,
            "kappa": round(float(self.kappa), 4),
            "depth": self.depth,
        }
        if self.combinator_cell_id:
            d["combinator_cell_id"] = self.combinator_cell_id
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.slot_children:
            d["slot_children"] = {
                k: [c.to_dict() for c in v] for k, v in self.slot_children.items()
            }
        if self.grounded_cells:
            d["grounded_cells"] = [getattr(c, "cell_id", str(c)) for c in self.grounded_cells]
        return d


@dataclass
class UnfoldingResult:
    """Output bundle from CoalgebraicUnfolder.unfold()."""
    root: CoalgebraNode
    flat_leaf_goals: List[str]
    max_depth_reached: int
    used_combinators: List[str] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)


_ANAMORPHISM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "combinator": {
            "type": ["string", "null"],
            "description": "Exact cell_id of a declared macro combinator from the summary if this goal wraps slots, else null."
        },
        "slots": {
            "type": "object",
            "description": "When combinator is set, maps declared slot names to ordered lists of child subgoals."
        },
        "steps": {
            "type": "array",
            "description": "Ordered sequential subgoals decomposing the goal. Each item may be a string subgoal or a structured combinator step object.",
            "items": {
                "type": ["string", "object"],
            },
        },
    },
    "required": ["steps"],
}


class CoalgebraicUnfolder:
    """
    Implements the Recursive Coalgebraic Subgoal Unfolding Engine:
        alpha_C : Goal -> F_C(Goal)

    Recursively decomposes abstract/composite goals (kappa < theta_closed) up to
    max_depth (default 3) using only declared lattice schemas, carrier types, and
    macro combinators discovered dynamically from C.
    """

    def __init__(
        self,
        theta_closed: float = 0.55,
        max_depth: int = 3,
        max_subgoals_per_step: int = 8,
    ):
        self.theta_closed = float(theta_closed)
        self.max_depth = max(1, int(max_depth))
        self.max_subgoals_per_step = max(2, int(max_subgoals_per_step))
        self.scorer = EpistemicGroundingScorer(theta_closed=self.theta_closed)

    @staticmethod
    def _parse_anamorphism_json(raw: str) -> Optional[Dict[str, Any]]:
        if not raw or not isinstance(raw, str):
            return None
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"steps": parsed}
        except Exception:
            pass
        # Structural extraction of first JSON object or array
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = cleaned.find(open_ch)
            end = cleaned.rfind(close_ch)
            if start != -1 and end > start:
                try:
                    parsed = json.loads(cleaned[start : end + 1])
                    if isinstance(parsed, dict):
                        return parsed
                    if isinstance(parsed, list):
                        return {"steps": parsed}
                except Exception:
                    pass
        return None

    def _query_anamorphism(
        self,
        goal: str,
        orchestrator: Any,
        model_mgr: Any,
        summary_text: str,
        valid_combinators: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if model_mgr is None or not getattr(model_mgr, "profile", None):
            return None
        if not model_mgr.can_synthesize():
            return None

        system_prompt = (
            "You are the Coalgebraic Subgoal Unfolding Engine in STRATA.\n"
            "Given a computational goal and a formal summary of the loaded typed lattice C "
            "(available domains, carrier types T, typestate transitions, macro combinators, and operations),\n"
            "unfold the goal into an ordered composition of finer subgoals.\n"
            "Rules:\n"
            "1. Only use concepts, carriers, and operations supported by the provided lattice C.\n"
            "2. If control flow (loop, condition, error handling, context scope) is required, select "
            "an exact macro combinator cell_id listed in 'Available structural macro combinators' and "
            "populate its declared slots with child subgoals.\n"
            "3. Emit concise, action-oriented subgoals aligned to the lattice operations.\n"
            "4. Never write raw source code or invent non-existent combinator cell IDs.\n"
            "5. Respond strictly in JSON matching schema:\n"
            '   {"steps": [<string or {"goal": str, "combinator": str, "slots": {slot_name: [str, ...]}}>]}\n'
            '   OR for a single top-level combinator: {"combinator": "<cell_id>", "slots": {"<slot>": [<subgoals>]}, "steps": []}'
        )
        user_prompt = (
            f"Active Lattice Summary:\n{summary_text}\n\n"
            f"Target Goal to Unfold:\n{goal}\n\n"
            "Return valid JSON only."
        )
        try:
            raw = model_mgr.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=512,
                schema=_ANAMORPHISM_SCHEMA,
            )
            return self._parse_anamorphism_json(raw)
        except Exception as err:
            logger.debug("[UNFOLDER] Anamorphism SLM query failed: %s", err)
            return None

    def unfold(
        self,
        goal: str,
        orchestrator: Any,
        model_mgr: Any = None,
        tunnel: Optional[Sequence[Any]] = None,
        relevance_map: Optional[Dict[str, float]] = None,
        max_depth: Optional[int] = None,
        force_root_unfold: bool = False,
    ) -> UnfoldingResult:
        """
        Recursively unfolds `goal` until every leaf has kappa >= theta_closed
        or recursion depth reaches `max_depth`.
        """
        depth_limit = self.max_depth if max_depth is None else max(1, int(max_depth))
        trace: List[Dict[str, Any]] = []
        used_combinators: List[str] = []
        valid_combinators = discover_combinators(orchestrator, scope_cells=tunnel)
        summary_dict = extract_lattice_summary(
            orchestrator, goal=goal, tunnel=tunnel, relevance_map=relevance_map
        )
        summary_text = format_lattice_summary(summary_dict)
        max_seen_depth = 0

        can_synth = (
            model_mgr is not None
            and getattr(model_mgr, "profile", None) is not None
            and bool(model_mgr.can_synthesize())
        )

        def _resolve_valid_slot_map(
            comb_id: str,
            raw_slots: Any,
            fallback_steps: List[Any],
        ) -> Dict[str, List[Any]]:
            comb_cell = valid_combinators[comb_id]
            declared_slots = get_macro_slot_names(comb_cell)
            resolved: Dict[str, List[Any]] = {}
            if isinstance(raw_slots, dict) and raw_slots:
                for sk, sv in raw_slots.items():
                    sk_str = str(sk).strip()
                    if sk_str in declared_slots:
                        target_slot = sk_str
                    elif len(declared_slots) == 1:
                        target_slot = declared_slots[0]
                    else:
                        continue
                    items = sv if isinstance(sv, list) else [sv]
                    resolved.setdefault(target_slot, []).extend(items)
            elif fallback_steps and declared_slots:
                resolved[declared_slots[0]] = list(fallback_steps)
            return resolved

        def _recurse(
            cur_goal: str,
            depth: int,
            ancestors: Tuple[str, ...],
        ) -> CoalgebraNode:
            nonlocal max_seen_depth
            max_seen_depth = max(max_seen_depth, depth)

            clean_goal = str(cur_goal or "").strip()
            kappa = self.scorer.score(clean_goal, orchestrator, tunnel=tunnel, relevance_map=relevance_map)
            g_key = _normalize_goal_key(clean_goal)

            should_expand = (
                can_synth
                and depth < depth_limit
                and g_key
                and (force_root_unfold and depth == 0 or kappa < self.theta_closed)
            )
            if not should_expand:
                trace.append({
                    "event": "unfold_stop",
                    "goal": clean_goal,
                    "depth": depth,
                    "kappa": round(kappa, 4),
                    "reason": "depth_limit" if depth >= depth_limit else ("closed" if kappa >= self.theta_closed else "no_synthesizer"),
                })
                return CoalgebraNode(goal=clean_goal, kind="atomic", kappa=kappa, depth=depth)

            parsed = self._query_anamorphism(
                clean_goal, orchestrator, model_mgr, summary_text, valid_combinators
            )
            if not parsed:
                trace.append({
                    "event": "unfold_empty_response",
                    "goal": clean_goal,
                    "depth": depth,
                    "kappa": round(kappa, 4),
                })
                return CoalgebraNode(goal=clean_goal, kind="atomic", kappa=kappa, depth=depth)

            next_ancestors = ancestors + (g_key,)
            top_comb = parsed.get("combinator") or parsed.get("combinator_cell_id")
            top_slots = parsed.get("slots") or parsed.get("slot_subgoals")
            raw_steps = parsed.get("steps") or parsed.get("subgoals") or []
            if not isinstance(raw_steps, list):
                raw_steps = [raw_steps]

            # Case 1: Whole goal wraps a declared macro combinator
            if isinstance(top_comb, str) and top_comb.strip() in valid_combinators:
                comb_id = top_comb.strip()
                slot_map = _resolve_valid_slot_map(comb_id, top_slots, raw_steps)
                if slot_map:
                    slot_children: Dict[str, List[CoalgebraNode]] = {}
                    for sk, items in slot_map.items():
                        ch_list: List[CoalgebraNode] = []
                        for item in items[: self.max_subgoals_per_step]:
                            sub_node = _build_child(item, depth + 1, next_ancestors)
                            if sub_node is not None:
                                ch_list.append(sub_node)
                        if ch_list:
                            slot_children[sk] = ch_list
                    if slot_children:
                        used_combinators.append(comb_id)
                        trace.append({
                            "event": "unfold_combinator",
                            "goal": clean_goal,
                            "depth": depth,
                            "kappa": round(kappa, 4),
                            "combinator": comb_id,
                            "slots": {k: len(v) for k, v in slot_children.items()},
                        })
                        return CoalgebraNode(
                            goal=clean_goal,
                            kind="combinator",
                            kappa=kappa,
                            depth=depth,
                            combinator_cell_id=comb_id,
                            slot_children=slot_children,
                        )

            # Case 2: Ordered sequence of steps (strings and/or combinator steps)
            seq_children: List[CoalgebraNode] = []
            for item in raw_steps[: self.max_subgoals_per_step]:
                sub_node = _build_child(item, depth + 1, next_ancestors)
                if sub_node is not None:
                    seq_children.append(sub_node)

            if len(seq_children) >= 2 or (
                len(seq_children) == 1 and seq_children[0].kind == "combinator"
            ):
                trace.append({
                    "event": "unfold_sequence",
                    "goal": clean_goal,
                    "depth": depth,
                    "kappa": round(kappa, 4),
                    "child_count": len(seq_children),
                })
                if len(seq_children) == 1 and seq_children[0].kind == "combinator":
                    return seq_children[0]
                return CoalgebraNode(
                    goal=clean_goal,
                    kind="seq",
                    kappa=kappa,
                    depth=depth,
                    children=seq_children,
                )

            trace.append({
                "event": "unfold_trivial",
                "goal": clean_goal,
                "depth": depth,
                "kappa": round(kappa, 4),
            })
            return CoalgebraNode(goal=clean_goal, kind="atomic", kappa=kappa, depth=depth)

        def _build_child(
            item: Any,
            depth: int,
            ancestors: Tuple[str, ...],
        ) -> Optional[CoalgebraNode]:
            if isinstance(item, str):
                text = item.strip()
                if not text:
                    return None
                c_key = _normalize_goal_key(text)
                if not c_key or c_key in ancestors:
                    return None
                return _recurse(text, depth, ancestors)

            if isinstance(item, dict):
                step_comb = item.get("combinator") or item.get("combinator_cell_id")
                step_goal = str(item.get("goal") or item.get("subgoal") or item.get("description") or "").strip()
                step_slots = item.get("slots") or item.get("slot_subgoals")
                step_sub = item.get("steps") or item.get("subgoals") or []

                if isinstance(step_comb, str) and step_comb.strip() in valid_combinators:
                    comb_id = step_comb.strip()
                    slot_map = _resolve_valid_slot_map(
                        comb_id,
                        step_slots,
                        step_sub if isinstance(step_sub, list) else [step_sub],
                    )
                    c_key = _normalize_goal_key(step_goal) if step_goal else f"__comb_{comb_id}_{depth}"
                    next_anc = ancestors + (c_key,)
                    slot_children: Dict[str, List[CoalgebraNode]] = {}
                    for sk, s_items in slot_map.items():
                        ch_list: List[CoalgebraNode] = []
                        for si in s_items[: self.max_subgoals_per_step]:
                            sub_node = _build_child(si, depth + 1, next_anc)
                            if sub_node is not None:
                                ch_list.append(sub_node)
                        if ch_list:
                            slot_children[sk] = ch_list
                    if slot_children:
                        nonlocal max_seen_depth
                        max_seen_depth = max(max_seen_depth, depth)
                        used_combinators.append(comb_id)
                        c_kappa = self.scorer.score(step_goal or comb_id, orchestrator, tunnel=tunnel, relevance_map=relevance_map)
                        trace.append({
                            "event": "unfold_inline_combinator",
                            "goal": step_goal or comb_id,
                            "depth": depth,
                            "combinator": comb_id,
                            "slots": {k: len(v) for k, v in slot_children.items()},
                        })
                        return CoalgebraNode(
                            goal=step_goal or comb_id,
                            kind="combinator",
                            kappa=c_kappa,
                            depth=depth,
                            combinator_cell_id=comb_id,
                            slot_children=slot_children,
                        )

                if step_goal:
                    c_key = _normalize_goal_key(step_goal)
                    if c_key and c_key not in ancestors:
                        return _recurse(step_goal, depth, ancestors)
            return None

        root = _recurse(goal, depth=0, ancestors=())
        return UnfoldingResult(
            root=root,
            flat_leaf_goals=root.leaf_goals(),
            max_depth_reached=max_seen_depth,
            used_combinators=used_combinators,
            trace=trace,
        )

    def ground_tree(
        self,
        node: CoalgebraNode,
        orchestrator: Any,
        route_fn: Callable[[str, Optional[Any]], List[Any]],
        start_sig: Optional[Any] = None,
        bridge_max_hops: int = 2,
    ) -> List[Any]:
        """
        Recursively grounds a CoalgebraNode tree into verified lattice Cell sequences,
        populating `bound_slots` on combinator nodes and inserting shortest verified
        bridge paths where needed.
        """
        loaded = getattr(orchestrator, "loaded_cells", {}) or {} if orchestrator is not None else {}
        find_bridges = getattr(orchestrator, "find_shortest_bridge_path", None)

        def _bridge_extend(acc: List[Any], new_cells: Sequence[Any]) -> None:
            for c in new_cells:
                if acc and callable(find_bridges):
                    try:
                        bridges = find_bridges(acc[-1], c, max_hops=bridge_max_hops)
                        if bridges:
                            acc.extend(bridges)
                    except Exception:
                        pass
                acc.append(c)

        def _call_route(goal_text: str, s_sig: Optional[Any], gamma_sigs: Tuple[Any, ...]) -> List[Any]:
            try:
                return list(route_fn(goal_text, s_sig, initial_gamma=gamma_sigs) or [])
            except TypeError:
                return list(route_fn(goal_text, s_sig) or [])

        acc_base: Tuple[Any, ...] = (start_sig,) if start_sig is not None else ()
        if kwargs_gamma := getattr(node, "_gamma_context", None):
            acc_base = tuple(kwargs_gamma)

        def _collect_cell_sigs(cell_seq: Sequence[Any], existing: List[Any]) -> None:
            for c_item in cell_seq:
                outs = getattr(c_item, "outputs", {}) or {}
                if isinstance(outs, dict):
                    for op in outs.values():
                        sig = getattr(op, "signature", op)
                        if sig is not None:
                            existing.append(sig)
                pout = getattr(c_item, "primary_output", None)
                if pout is not None:
                    psig = getattr(pout, "signature", pout)
                    if psig is not None and psig not in existing:
                        existing.append(psig)

        if node.kind == "atomic":
            raw_cells = _call_route(node.goal, start_sig, acc_base)
            sub_toks = set(CellTokenizer.tokenize_prompt(str(node.goal or "").lower()))
            cells = []
            for c_item in raw_cells:
                c_copy = c_item.clone() if hasattr(c_item, "clone") else c_item
                if sub_toks:
                    existing_st = getattr(c_copy, "matched_subgoal_tokens", None) or set()
                    c_copy.matched_subgoal_tokens = set(existing_st) | sub_toks
                cells.append(c_copy)
            node.grounded_cells = list(cells)
            return cells

        if node.kind == "seq":
            composed: List[Any] = []
            cur_sig = start_sig
            seq_sigs: List[Any] = list(acc_base)
            for ch in node.children:
                if composed:
                    last_out = getattr(composed[-1], "primary_output", None)
                    cur_sig = getattr(last_out, "signature", last_out) if last_out else cur_sig
                ch._gamma_context = tuple(seq_sigs)
                sub_cells = self.ground_tree(
                    ch,
                    orchestrator=orchestrator,
                    route_fn=route_fn,
                    start_sig=cur_sig,
                    bridge_max_hops=bridge_max_hops,
                )
                if sub_cells:
                    _bridge_extend(composed, sub_cells)
                    _collect_cell_sigs(sub_cells, seq_sigs)
            node.grounded_cells = list(composed)
            return composed

        if node.kind == "combinator" and node.combinator_cell_id in loaded:
            proto = loaded[node.combinator_cell_id]
            comb_cell = proto.clone() if hasattr(proto, "clone") else proto
            if not isinstance(getattr(comb_cell, "bound_slots", None), dict):
                comb_cell.bound_slots = {}

            for slot_name, child_nodes in node.slot_children.items():
                slot_cells: List[Any] = []
                cur_sig = start_sig
                slot_sigs: List[Any] = list(acc_base)
                for ch in child_nodes:
                    if slot_cells:
                        last_out = getattr(slot_cells[-1], "primary_output", None)
                        cur_sig = getattr(last_out, "signature", last_out) if last_out else cur_sig
                    ch._gamma_context = tuple(slot_sigs)
                    ch_cells = self.ground_tree(
                        ch,
                        orchestrator=orchestrator,
                        route_fn=route_fn,
                        start_sig=cur_sig,
                        bridge_max_hops=bridge_max_hops,
                    )
                    if ch_cells:
                        _bridge_extend(slot_cells, ch_cells)
                        _collect_cell_sigs(ch_cells, slot_sigs)
                if slot_cells:
                    comb_cell.bound_slots[slot_name] = slot_cells

            node.grounded_cells = [comb_cell]
            return [comb_cell]

        return []


__all__ = [
    "CoalgebraNode",
    "CoalgebraicUnfolder",
    "EpistemicGroundingScorer",
    "UnfoldingResult",
    "discover_combinators",
    "extract_lattice_summary",
    "format_lattice_summary",
    "get_macro_slot_names",
    "is_macro_combinator_cell",
]
