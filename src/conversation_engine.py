"""
src/conversation_engine.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Track 9: Conversational State & Surgical Graph Delta Morphisms (Chat Engine).

Maintains the active verified DAG G_{t-1}, variable bindings, and emitted
artifacts across conversational turns. Supports three surgical graph delta
operations without full-pipeline recompilation regressions:
  - Delta_param   : updates literal parameters in O(1) (~0ms recompilation).
  - Delta_replace : swaps a sub-morphism while preserving surrounding wire contracts.
  - Delta_kernel  : refines logic inside a macro slot / typed hole without touching outer wiring.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from log_config import get_logger

try:
    from .lattice import Cell, LatticeOrchestrator
    from .unification import ExecutionContext, UnificationGate, Substitution
    from .dag_plan import DAGPlan, OpenPortFrontier
    from .synthesis import InductiveKernelGenerator
except (ImportError, ValueError):
    from lattice import Cell, LatticeOrchestrator
    from unification import ExecutionContext, UnificationGate, Substitution
    from dag_plan import DAGPlan, OpenPortFrontier
    from synthesis import InductiveKernelGenerator

logger = get_logger("conversation_engine")


@dataclass
class GraphDeltaMorphism:
    """
    Track 9.2: Surgical Graph Delta Morphism (Delta: G_{t-1} -> G_t).
    """
    kind: str  # "param" | "replace" | "kernel"
    target_cell_id: str
    port_or_slot: Optional[str] = None
    new_value: Any = None
    replacement_cell: Optional[Cell] = None


@dataclass
class TurnRecord:
    turn_id: int
    prompt: str
    delta: Optional[GraphDeltaMorphism]
    dag_plan: DAGPlan
    pipeline_bindings: List[Tuple[Cell, Dict[str, str]]]
    emitted_code: str
    mutated_node_ids: Set[str] = field(default_factory=set)


class ConversationSession:
    """
    Track 9.1, 9.2 & 9.3: Stateful Conversational Graph Surgery Session.
    Preserves verified DAG state across turns and re-verifies only mutated subgraphs.
    """

    def __init__(
        self,
        orchestrator: LatticeOrchestrator,
        gate: Optional[UnificationGate] = None,
        kernel_generator: Optional[InductiveKernelGenerator] = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.gate = gate or UnificationGate(orchestrator=orchestrator)
        self.kernel_generator = kernel_generator or InductiveKernelGenerator()
        self.turns: List[TurnRecord] = []
        self.active_dag: Optional[DAGPlan] = None
        self.active_bindings: List[Tuple[Cell, Dict[str, str]]] = []
        self.active_ctx: Optional[ExecutionContext] = None
        self.active_code: str = ""
        self.emitted_artifacts: List[str] = []

    def commit_initial_plan(
        self,
        prompt: str,
        cells: List[Cell],
        ctx: Optional[ExecutionContext] = None,
    ) -> TurnRecord:
        """Commits the initial turn's verified pipeline into the conversational session."""
        run_ctx = ctx or ExecutionContext(prompt=prompt)
        code = self.gate.emit_code(cells, run_ctx)
        bindings = list(self.gate.last_pipeline_bindings)
        dag = OpenPortFrontier.build_dag_plan([c for c, _ in bindings])

        self.active_dag = dag
        self.active_bindings = bindings
        self.active_ctx = run_ctx
        self.active_code = code
        self.emitted_artifacts = self.gate.get_egress_paths()

        rec = TurnRecord(
            turn_id=len(self.turns) + 1,
            prompt=prompt,
            delta=None,
            dag_plan=dag,
            pipeline_bindings=bindings,
            emitted_code=code,
            mutated_node_ids={c.cell_id for c in dag.nodes},
        )
        self.turns.append(rec)
        return rec

    def apply_delta(self, delta: GraphDeltaMorphism, follow_up_prompt: str = "") -> TurnRecord:
        """
        Track 9.2 & 9.3: Applies a surgical GraphDeltaMorphism (Delta_param,
        Delta_replace, or Delta_kernel) to the active DAG and incrementally
        re-verifies only the mutated subgraph.
        """
        if self.active_dag is None or not self.active_bindings:
            raise RuntimeError("Cannot apply GraphDeltaMorphism: no active DAG in session.")

        if delta.kind == "param":
            return self._apply_param_delta(delta, follow_up_prompt)
        elif delta.kind == "replace":
            return self._apply_replace_delta(delta, follow_up_prompt)
        elif delta.kind == "kernel":
            return self._apply_kernel_delta(delta, follow_up_prompt)
        else:
            raise ValueError(f"Unsupported GraphDeltaMorphism kind: '{delta.kind}'")

    def _apply_param_delta(self, delta: GraphDeltaMorphism, prompt: str) -> TurnRecord:
        """
        Delta_param: Updates literal parameter bindings on `target_cell_id` in O(1)
        without re-running topological routing or re-unifying untouched upstream nodes.
        """
        new_bindings: List[Tuple[Cell, Dict[str, str]]] = []
        mutated: Set[str] = set()

        for cell, bnd in self.active_bindings:
            bnd_copy = dict(bnd)
            if cell.cell_id.lower() == delta.target_cell_id.lower():
                port_key = delta.port_or_slot
                if not port_key or port_key not in cell.inputs:
                    # Locate first non-carrier parameter port
                    port_key = next(
                        (
                            k for k, p in cell.inputs.items()
                            if not str(bnd_copy.get(k, "")).startswith("var_")
                        ),
                        next(iter(cell.inputs.keys()), None),
                    )
                if port_key is None:
                    raise KeyError(f"Cell '{cell.cell_id}' has no parameter port to update.")
                val_str = repr(delta.new_value) if isinstance(delta.new_value, str) and not (
                    delta.new_value.startswith(("'", '"'))
                ) else str(delta.new_value)
                bnd_copy[port_key] = val_str
                mutated.add(cell.cell_id)
            new_bindings.append((cell, bnd_copy))

        if not mutated:
            raise KeyError(f"Target cell '{delta.target_cell_id}' not found in active DAG.")

        # Incremental re-emission from pre-verified bindings (0ms routing/unification)
        code = self.gate.emit_code(new_bindings, self.active_ctx, extract_llm_slots=False)
        self.active_bindings = list(self.gate.last_pipeline_bindings)
        self.active_code = code

        rec = TurnRecord(
            turn_id=len(self.turns) + 1,
            prompt=prompt,
            delta=delta,
            dag_plan=self.active_dag,
            pipeline_bindings=self.active_bindings,
            emitted_code=code,
            mutated_node_ids=mutated,
        )
        self.turns.append(rec)
        return rec

    def _apply_replace_delta(self, delta: GraphDeltaMorphism, prompt: str) -> TurnRecord:
        """
        Delta_replace: Swaps `target_cell_id` with `replacement_cell` while verifying
        that incoming and outgoing wire contracts with adjacent nodes remain sound.
        """
        rep = delta.replacement_cell
        if rep is None:
            rep = self.orchestrator.loaded_cells.get(str(delta.new_value or ""))
        if rep is None:
            raise ValueError(f"Replacement cell for '{delta.target_cell_id}' not found.")

        new_cells: List[Cell] = []
        target_idx: Optional[int] = None
        for idx, (cell, _bnd) in enumerate(self.active_bindings):
            if cell.cell_id.lower() == delta.target_cell_id.lower() and target_idx is None:
                target_idx = idx
                new_cells.append(rep.clone() if hasattr(rep, "clone") else copy.copy(rep))
            else:
                new_cells.append(cell)

        if target_idx is None:
            raise KeyError(f"Target cell '{delta.target_cell_id}' not found in active DAG.")

        # Incremental boundary verification: verify predecessor -> rep and rep -> successor
        if target_idx > 0:
            pred = new_cells[target_idx - 1]
            u_in = self.gate.unify_transition(pred, rep, Substitution(), context=self.active_ctx)
            if u_in.is_bottom():
                raise ValueError(
                    f"Delta_replace wire contract violation: predecessor '{pred.cell_id}' "
                    f"cannot feed replacement '{rep.cell_id}'."
                )
        if target_idx + 1 < len(new_cells):
            succ = new_cells[target_idx + 1]
            u_out = self.gate.unify_transition(rep, succ, Substitution(), context=self.active_ctx)
            if u_out.is_bottom():
                raise ValueError(
                    f"Delta_replace wire contract violation: replacement '{rep.cell_id}' "
                    f"cannot feed successor '{succ.cell_id}'."
                )

        run_ctx = self.active_ctx.clone() if self.active_ctx else ExecutionContext(prompt=prompt)
        code = self.gate.emit_code(new_cells, run_ctx, extract_llm_slots=False)
        bindings = list(self.gate.last_pipeline_bindings)
        dag = OpenPortFrontier.build_dag_plan(new_cells)

        self.active_dag = dag
        self.active_bindings = bindings
        self.active_code = code

        rec = TurnRecord(
            turn_id=len(self.turns) + 1,
            prompt=prompt,
            delta=delta,
            dag_plan=dag,
            pipeline_bindings=bindings,
            emitted_code=code,
            mutated_node_ids={rep.cell_id},
        )
        self.turns.append(rec)
        return rec

    def _apply_kernel_delta(self, delta: GraphDeltaMorphism, prompt: str) -> TurnRecord:
        """
        Delta_kernel: Refines or replaces the inner inductive kernel of a macro slot
        without altering outer envelope port bindings.
        """
        mutated: Set[str] = set()
        new_bindings: List[Tuple[Cell, Dict[str, str]]] = []

        for cell, bnd in self.active_bindings:
            if cell.cell_id.lower() == delta.target_cell_id.lower():
                cell_copy = cell.clone() if hasattr(cell, "clone") else copy.copy(cell)
                slot_name = delta.port_or_slot or next(iter(getattr(cell_copy, "slots", {}) or {"body": {}}))
                kernel_code = str(delta.new_value or "").strip()
                is_safe, err = self.kernel_generator.validate_safety(kernel_code)
                if not is_safe:
                    raise ValueError(f"Delta_kernel capability check failed: {err}")
                if not isinstance(getattr(cell_copy, "bound_slots", None), dict):
                    cell_copy.bound_slots = {}
                cell_copy.bound_slots[slot_name] = kernel_code
                mutated.add(cell_copy.cell_id)
                new_bindings.append((cell_copy, dict(bnd)))
            else:
                new_bindings.append((cell, dict(bnd)))

        if not mutated:
            raise KeyError(f"Target macro cell '{delta.target_cell_id}' not found in active DAG.")

        code = self.gate.emit_code(new_bindings, self.active_ctx, extract_llm_slots=False)
        self.active_bindings = list(self.gate.last_pipeline_bindings)
        self.active_code = code

        rec = TurnRecord(
            turn_id=len(self.turns) + 1,
            prompt=prompt,
            delta=delta,
            dag_plan=self.active_dag,
            pipeline_bindings=self.active_bindings,
            emitted_code=code,
            mutated_node_ids=mutated,
        )
        self.turns.append(rec)
        return rec


__all__ = [
    "GraphDeltaMorphism",
    "TurnRecord",
    "ConversationSession",
]
