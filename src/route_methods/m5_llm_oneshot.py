from __future__ import annotations
import json
from typing import Dict, List, Optional, Any
from .base import RouteMethod, is_macro_cell
from lattice import Cell, LatticeOrchestrator
from unification import ExecutionContext
from inference import ModelManager


class M5LLMOneShotRouteMethod(RouteMethod):
    name = "m5_llm_oneshot"

    def plan(self, prompt, tunnel, relevance_map, orchestrator=None, ctx=None,
             start_sig=None, goal_sig=None, max_transforms=6, **kwargs):
        self._current_prompt = prompt
        mm = ModelManager.get_instance()
        if not mm.can_synthesize():
            return []
        orch = orchestrator or self.orchestrator
        if not tunnel:
            return []
        candidates = [
            c for c in tunnel
            if getattr(c, "node_type", "") != "constant"
            and (
                bool(getattr(c, "slots", None))
                or (
                    not is_macro_cell(c)
                    and not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
                )
            )
        ] or [tunnel[0]]
        cand_map = {c.cell_id.lower(): c for c in candidates}

        def _fail(reason: str) -> List[Cell]:
            self.fallback_reason = reason
            self._trace("method_failure", requested=type(self).__name__, reason=reason)
            return []

        proposed: List[Cell] = []
        _llm_error = None
        try:
            top_str = "\n".join(
                f"- {c.cell_id}"
                for c in sorted(candidates, key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)[:25]
            )
            resp = mm.generate_text(
                f"Intent: {prompt}\nCells:\n{top_str}\nReturn JSON list of cell IDs in order:",
                max_tokens=192,
                system_prompt="Output ONLY a JSON list.",
            ).strip()
            parsed_list = None
            if "[" in resp and "]" in resp:
                lbracket = resp.find("[")
                rbracket = resp.rfind("]")
                try:
                    parsed_list = json.loads(resp[lbracket : rbracket + 1])
                except Exception:
                    pass
            if isinstance(parsed_list, list):
                for cid in parsed_list:
                    key = str(cid).strip().lower()
                    if key in cand_map:
                        proposed.append(cand_map[key])
        except Exception as _llm_err:
            proposed = []
            _llm_error = f"{type(_llm_err).__name__}: {_llm_err}"

        if not proposed:
            return _fail(
                f"LLM proposal raised {_llm_error}" if _llm_error else "LLM one-shot proposed no usable cells"
            )

        chain = [proposed[0]]
        for nxt in proposed[1:]:
            if any(c.cell_id == nxt.cell_id for c in chain):
                continue
            curr = chain[-1]
            if self.step_unifies(curr, nxt, prev_path=chain) or self.step_unifies_dag(nxt, chain, ctx=ctx):
                chain.append(nxt)
            else:
                bridge = self.find_bridge(curr, nxt, candidates, orch, prev_path=chain)
                if bridge:
                    chain.extend([bridge, nxt])
                else:
                    connected = False
                    for anc in reversed(chain[:-1]):
                        if self.step_unifies(anc, nxt, prev_path=chain) or self.step_unifies_dag(nxt, chain, ctx=ctx):
                            chain.append(nxt)
                            connected = True
                            break
                        anc_brg = self.find_bridge(anc, nxt, candidates, orch, prev_path=chain)
                        if anc_brg:
                            chain.extend([anc_brg, nxt])
                            connected = True
                            break
                    if not connected:
                        return _fail(f"proposed step {nxt.cell_id} could not be connected or bridged")

        if len(chain) < 2 and len(proposed) >= 2:
            return _fail("verified chain shorter than 2 cells")

        self.tag_cells_with_clause_indices(chain, prompt, orchestrator=orch)

        try:
            from unification import UnificationGate, ExecutionContext
            from preflight import PreflightLinter
            _gate = UnificationGate(orchestrator=orch)
            _tctx = ctx if ctx is not None else ExecutionContext(prompt=prompt)
            _ur = _gate.unify_pipeline(chain, _tctx, extract_llm_slots=False)
            if _ur.is_bottom():
                return _fail("LLM proposed path does not unify (bottom)")
            _code = _gate.emit_code(_ur.value, _tctx, extract_llm_slots=False)
            _lint = PreflightLinter.lint(_ur.value, prompt=prompt, code_str=_code)
            if not _lint.is_valid:
                return _fail(f"LLM proposed path has pre-flight violations: {_lint.violations[0][:100]}")
        except Exception:
            pass

        clauses = self.segment_prompt_clauses(prompt, ctx=ctx)
        max_allowed = max(32, len(clauses) * 4 + 4, max_transforms + 8)
        return chain[:max_allowed]


