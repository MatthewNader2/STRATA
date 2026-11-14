"""
src/route_methods/m4_llm_stepwise.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)

M4: Unconstrained Stepwise LLM Transition Selection.
At each step, the LLM chooses the next cell from the unconstrained tunnel shortlist
(bare/minimal cell IDs ranked by retrieval relevance, WITHOUT poset type-mask edge
filtering E_valid(Gamma_t)). Post-selection, transitions are verified with Robinson
unification so unconstrained type errors fail visibly rather than hiding behind M1.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Set

from log_config import get_logger
from .base import RouteMethod, is_macro_cell

try:
    from ..lattice import Cell, LatticeOrchestrator, TypeRegistry
    from ..unification import ExecutionContext, UnificationGate
    from ..inference import ModelManager
    from ..utils import extract_json_object, tokenize_alphanumeric
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator, TypeRegistry
    from unification import ExecutionContext, UnificationGate
    from inference import ModelManager
    from utils import extract_json_object, tokenize_alphanumeric

logger = get_logger("route_methods.m4")

STEP_SCHEMA_M4 = {
    "type": "object",
    "required": ["next_node"],
    "properties": {
        "next_node": {"type": "string"},
    },
}

SYSTEM_PROMPT_M4 = (
    "You are a pipeline builder. Given the user's request, the operations selected so far, "
    "and an unconstrained list of candidate cell IDs from the tunnel, choose the single best "
    "next cell ID to append, or output 'FINISH' if the request is complete.\n"
    "Respond ONLY with JSON: {\"next_node\": \"CELL_ID\"} or {\"next_node\": \"FINISH\"}."
)


def _parse_step_m4(raw_text: str) -> Optional[str]:
    if not raw_text:
        return None
    data = extract_json_object(raw_text)
    if data and isinstance(data.get("next_node"), str):
        return data["next_node"].strip()
    toks = tokenize_alphanumeric(raw_text)
    if "finish" in toks:
        return "FINISH"
    return toks[0] if toks else None


class M4LLMStepwiseRouteMethod(RouteMethod):
    """
    RouteMethod M4: Unconstrained Stepwise LLM Transition Selection over tunnel T.
    Distinct from M8 (type-mask-constrained Markovian poset transitions over E_valid(Gamma_t)).
    """
    name: str = "m4_llm_stepwise"

    def plan(
        self,
        prompt: str,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
        orchestrator: Optional[LatticeOrchestrator] = None,
        ctx: Optional[ExecutionContext] = None,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        max_transforms: int = 6,
        **kwargs
    ) -> List[Cell]:
        self._current_prompt = prompt
        mm = ModelManager.get_instance()
        if not mm.can_synthesize():
            reason = "M4 requires a generative LLM"
            self.fallback_reason = reason
            self._trace("method_failure", requested=type(self).__name__, reason=reason)
            return []

        orch = orchestrator or self.orchestrator
        if not tunnel:
            return []
        if len(tunnel) == 1:
            return [tunnel[0]]

        candidates = [
            c for c in tunnel
            if getattr(c, "node_type", "") != "constant"
            and (
                bool(getattr(c, "slots", None))
                or (
                    not is_macro_cell(c)
                    and not (
                        getattr(c, "is_combinator", False)
                        or getattr(c, "node_role", "") == "combinator"
                        or getattr(c, "role", "") == "combinator"
                        or getattr(c, "node_type", "") == "combinator"
                    )
                )
            )
        ] or [tunnel[0]]

        # Rank tunnel pool purely by retrieval relevance (no E_valid type-edge prefiltering)
        ranked_pool = sorted(
            candidates,
            key=lambda c: relevance_map.get(c.cell_id, 0.0),
            reverse=True,
        )[:25]
        cand_map = {c.cell_id.lower(): c for c in ranked_pool}

        path: List[Cell] = []
        visited: Set[str] = set()
        max_steps = max(4, min(14, max_transforms + 4))
        reg = TypeRegistry.get_instance()
        stop_tokens = {str(t).upper() for t in (reg.get_verification_semantics("stopping_tokens") or ())}

        for step_idx in range(max_steps):
            unvisited = [c for c in ranked_pool if c.cell_id not in visited]
            if not unvisited:
                break
            shortlist = unvisited[:15]
            cand_lines = "\n".join(f"- {c.cell_id}" for c in shortlist)
            prefix_str = " -> ".join(c.cell_id for c in path) if path else "(empty)"
            finish_note = " or FINISH" if path else ""
            user_msg = (
                f"Request: {prompt}\n"
                f"Current Pipeline: {prefix_str}\n\n"
                f"Candidate Cell IDs:\n{cand_lines}\n\n"
                f"Select next_node{finish_note}:"
            )
            try:
                raw = mm.generate_text(
                    user_msg, max_tokens=64, schema=STEP_SCHEMA_M4, system_prompt=SYSTEM_PROMPT_M4
                )
            except Exception as exc:
                reason = f"LLM step failed: {type(exc).__name__}: {exc}"
                self.fallback_reason = reason
                self._trace("method_failure", requested=type(self).__name__, reason=reason)
                return []

            choice = _parse_step_m4(raw)
            if not choice:
                break
            if choice.upper() in stop_tokens and path:
                break

            chosen_cell = cand_map.get(choice.lower())
            if chosen_cell is None or chosen_cell.cell_id in visited:
                break

            path.append(chosen_cell)
            visited.add(chosen_cell.cell_id)
            if self.is_sink_node(chosen_cell, orch) and len(path) >= 2:
                break

        if not path:
            reason = "LLM selected no valid cells"
            self.fallback_reason = reason
            self._trace("method_failure", requested=type(self).__name__, reason=reason)
            return []

        # Verify unconstrained LLM steps against Robinson type unification
        for i in range(1, len(path)):
            prev_c, nxt_c = path[i - 1], path[i]
            if not (self.step_unifies(prev_c, nxt_c, prev_path=path[:i]) or self.step_unifies_dag(nxt_c, path[:i], ctx=ctx)):
                reason = f"Unconstrained LLM edge {prev_c.cell_id} -> {nxt_c.cell_id} violates type unification"
                self.fallback_reason = reason
                self._trace("method_failure", requested=type(self).__name__, reason=reason)
                return []

        try:
            try:
                from unification import UNRESOLVED_PORT
            except (ImportError, ValueError):
                from ..unification import UNRESOLVED_PORT
            _gate = UnificationGate(orchestrator=orch)
            _tctx = ctx if ctx is not None else ExecutionContext(prompt=prompt)
            _ur = _gate.unify_pipeline(path, _tctx, extract_llm_slots=False)
            has_unresolved = bool(
                getattr(_ur, "value", None)
                and any(v is UNRESOLVED_PORT for _, b in _ur.value for v in b.values())
            )
            if _ur.is_bottom() or has_unresolved:
                reason = "M4 unconstrained path does not unify or left unresolved input ports"
                self.fallback_reason = reason
                self._trace("method_failure", requested=type(self).__name__, reason=reason)
                return []
        except Exception:
            pass

        return path
