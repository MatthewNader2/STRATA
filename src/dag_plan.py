"""
src/dag_plan.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Solves Issue 2 (First-Class DAGPlan), Issue 3 (SemanticBlackboard), and Issue 5 (TypedHoleCell).
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

try:
    from .lattice import Cell, MicroCell, PortSignature, AlgebraicSignature, TypeRegistry
    from .unification import unify, Substitution
except (ImportError, ValueError):
    from lattice import Cell, MicroCell, PortSignature, AlgebraicSignature, TypeRegistry
    from unification import unify, Substitution


# ============================================================================
# 1. Semantic Blackboard (Solves Issue 3: Neural vs. Symbolic Authority)
# ============================================================================

@dataclass
class SemanticBlackboard:
    """
    Single authoritative source of truth for prompt literals, projected keys,
    and slot bindings across Layer 0, Layer 1 (IRCompiler), Layer 2 (Planner),
    and Layer 3 (UnificationGate).
    When `locked_by_neural` is True, heuristic character scanners may ONLY append
    unclaimed literals and can never overwrite locked slots or target sinks.
    """
    locked_by_neural: bool = False
    target_sink: Optional[str] = None
    projected_keys: List[str] = field(default_factory=list)
    mutated_keys: Dict[str, Set[str]] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_files: List[str] = field(default_factory=list)
    dest_files: List[str] = field(default_factory=list)
    cell_slot_locks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        return self.parameters

    @hyperparameters.setter
    def hyperparameters(self, val: Dict[str, Any]) -> None:
        self.parameters = dict(val)

    def lock_from_intent(self, intent_data: Dict[str, Any]) -> None:
        if not intent_data or not isinstance(intent_data, dict):
            return
        self.locked_by_neural = True
        tgt = (
            intent_data.get("target_sink")
            or intent_data.get("sink")
        )
        if tgt:
            self.target_sink = str(tgt).strip()
        keys = (
            intent_data.get("projected_keys")
            or intent_data.get("keys")
        )
        if isinstance(keys, list) and keys:
            self.projected_keys = [str(c).strip() for c in keys if str(c).strip()]
        muts = intent_data.get("mutated_keys") or intent_data.get("mutations")
        if isinstance(muts, dict):
            for k, s in muts.items():
                if isinstance(s, (set, list, tuple)):
                    self.mutated_keys.setdefault(str(k), set()).update(str(x) for x in s)
        if isinstance(intent_data.get("source_files"), list) and intent_data["source_files"]:
            self.source_files = [str(f).strip() for f in intent_data["source_files"] if str(f).strip()]
        if isinstance(intent_data.get("dest_files"), list) and intent_data["dest_files"]:
            self.dest_files = [str(f).strip() for f in intent_data["dest_files"] if str(f).strip()]
        if isinstance(intent_data.get("slots"), dict):
            for cid, s_map in intent_data["slots"].items():
                if isinstance(s_map, dict):
                    self.cell_slot_locks.setdefault(cid, {}).update(s_map)
        params = (
            intent_data.get("parameters")
            or intent_data.get("hyperparameters")
            or intent_data.get("config")
        )
        if isinstance(params, dict):
            self.parameters.update(params)


# ============================================================================
# 2. Open-World TypedHole & TypedHoleCell (Solves Issue 5)
# ============================================================================

@dataclass(frozen=True)
class TypedHole:
    """Formal typestate hole specification H: T_in -> T_out."""
    hole_id: str
    task_description: str
    input_ports: Dict[str, AlgebraicSignature]
    output_ports: Dict[str, AlgebraicSignature]
    invariants: Tuple[str, ...] = ()
    topology_kind: str = "pure_morphism"
    synthesized_code: Optional[str] = None

    def is_filled(self) -> bool:
        return bool(self.synthesized_code and self.synthesized_code.strip())


class TypedHoleCell(MicroCell):
    """
    A structural placeholder `Cell` representing a typed subgoal hole H: T_in -> T_out
    awaiting recursive decomposition into verified domain / control-flow cells.
    Raw unconstrained LLM code emission for unfilled holes is strictly prohibited.
    """

    def __init__(
        self,
        hole_id: str,
        task_description: str,
        in_sig: AlgebraicSignature,
        out_sig: AlgebraicSignature,
        in_port_name: Optional[str] = None,
        out_port_name: Optional[str] = None,
        stage: Optional[int] = None,
    ) -> None:
        if in_port_name is None or out_port_name is None:
            reg_dp = TypeRegistry.get_instance()
            if in_port_name is None:
                t_in = str(getattr(in_sig, "type_name", "in") or "in").lower().replace("-", "_")
                in_port_name = f"{t_in}_in" if not reg_dp.is_declared_top(t_in) else "in_0"
            if out_port_name is None:
                t_out = str(getattr(out_sig, "type_name", "out") or "out").lower().replace("-", "_")
                out_port_name = f"{t_out}_out" if not reg_dp.is_declared_top(t_out) else "out_0"
        if stage is None:
            try:
                from .planner import _active_vocab
                vocab = _active_vocab()
                stages = vocab.stages
                stage = stages[1] if len(stages) >= 3 else 2
            except Exception:
                stage = 2

        inputs = {
            in_port_name: PortSignature(
                name=in_port_name,
                signature=in_sig,
                required=True,
            )
        }
        outputs = {
            out_port_name: PortSignature(
                name=out_port_name,
                signature=out_sig,
                required=True,
            )
        }
        super().__init__(
            cell_id=hole_id,
            stage=stage,
            inputs=inputs,
            outputs=outputs,
            code_template=f"{{{out_port_name}}} = {{{in_port_name}}}",
            keywords=list(task_description.lower().split()[:8]),
            node_role="typed_hole",
            summary=task_description,
        )
        self.is_typed_hole: bool = True
        self.task_description: str = task_description


# ============================================================================
# 3. First-Class DAGPlan(list) & OpenPortFrontier (Solves Issue 2)
# ============================================================================

@dataclass
class DAGEdge:
    """Directed port-to-port wire in a monoidal DAGPlan."""
    src_node_id: str
    src_port: str
    dst_node_id: str
    dst_port: str
    carrier_type: str = "any"

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.src_node_id, f"{self.src_port}->{self.dst_port}", self.dst_node_id)


class DAGPlan(list):
    """
    Solves Issue 2:
    First-class Monoidal DAG Plan that subclasses `list[Cell]`.
    Every legacy caller expecting `List[Cell]` (indexing, slicing, iteration, bool)
    works without modification, while `DAGPlan` preserves explicit port-to-port
    `detailed_edges`, `holes`, and synchronizes `cell.planned_port_wires` across
    post-planning graph surgery (`prune_unrequested`, `insert_column_commits`,
    `insert_property_bridges`).
    """

    def __init__(
        self,
        nodes: Sequence[Cell] = (),
        edges: Optional[List[Tuple[str, str, str]]] = None,
        detailed_edges: Optional[List[DAGEdge]] = None,
        holes: Optional[Dict[str, TypedHole]] = None,
    ) -> None:
        super().__init__(nodes)
        self.edges: List[Tuple[str, str, str]] = list(edges or [])
        self.detailed_edges: List[DAGEdge] = list(detailed_edges or [])
        self.holes: Dict[str, TypedHole] = dict(holes or {})
        self.sync_wires_from_nodes()

    @property
    def nodes(self) -> List[Cell]:
        return list(self)

    def cell_ids(self) -> List[str]:
        return [c.cell_id for c in self]

    def sync_wires_from_nodes(self) -> None:
        """Synchronizes `self.detailed_edges` and `self.edges` from `cell.planned_port_wires`."""
        rebuilt_detailed: List[DAGEdge] = []
        for cell in self:
            pw = getattr(cell, "planned_port_wires", None) or {}
            for dst_port, target in pw.items():
                if isinstance(target, tuple) and len(target) == 2:
                    prod_cell, src_port = target
                    src_id = getattr(prod_cell, "cell_id", str(prod_cell))
                    rebuilt_detailed.append(
                        DAGEdge(
                            src_node_id=src_id,
                            src_port=str(src_port),
                            dst_node_id=cell.cell_id,
                            dst_port=str(dst_port),
                        )
                    )
        if rebuilt_detailed:
            self.detailed_edges = rebuilt_detailed
            self.edges = [e.as_tuple() for e in rebuilt_detailed]

    def rewire_after_surgery(self) -> "DAGPlan":
        """
        Re-verifies and repairs `planned_port_wires` after cells are pruned or
        inserted by `prune_unrequested`, `insert_column_commits`, or `insert_property_bridges`.
        """
        active_ids = {c.cell_id for c in self}
        for cell in self:
            pw = getattr(cell, "planned_port_wires", None)
            if isinstance(pw, dict):
                cell.planned_port_wires = {
                    k: v for k, v in pw.items()
                    if getattr(v[0] if isinstance(v, tuple) else v, "cell_id", str(v[0] if isinstance(v, tuple) else v)) in active_ids
                }
            bp = getattr(cell, "bound_parent_ids", None)
            if isinstance(bp, set):
                cell.bound_parent_ids = bp & active_ids

        rebuilt = OpenPortFrontier.build_dag_plan(list(self))
        self.edges = rebuilt.edges
        self.detailed_edges = rebuilt.detailed_edges
        return self

    def topological_order(self) -> List[Cell]:
        """Returns nodes in a valid topological execution order."""
        if len(self) <= 1 or not self.edges:
            return list(self)

        id_to_cell = {c.cell_id: c for c in self}
        in_degree: Dict[str, int] = {c.cell_id: 0 for c in self}
        adj: Dict[str, List[str]] = {c.cell_id: [] for c in self}

        for src_id, _port_spec, dst_id in self.edges:
            if src_id in id_to_cell and dst_id in id_to_cell and src_id != dst_id:
                adj[src_id].append(dst_id)
                in_degree[dst_id] = in_degree.get(dst_id, 0) + 1

        ready = [c.cell_id for c in self if in_degree[c.cell_id] == 0]
        ordered: List[Cell] = []
        visited: Set[str] = set()

        while ready:
            curr_id = ready.pop(0)
            if curr_id in visited:
                continue
            visited.add(curr_id)
            ordered.append(id_to_cell[curr_id])
            for nxt_id in adj.get(curr_id, []):
                in_degree[nxt_id] -= 1
                if in_degree[nxt_id] <= 0 and nxt_id not in visited:
                    ready.append(nxt_id)

        for c in self:
            if c.cell_id not in visited:
                ordered.append(c)
        return ordered


@dataclass(frozen=True)
class FrontierPort:
    node_id: str
    port_name: str
    signature: AlgebraicSignature
    step_idx: int
    cell_ref: Optional[Any] = None

    @property
    def port_id(self) -> str:
        return f"{self.node_id}.{self.port_name}"


class OpenPortFrontier:
    """Active Open-Port Frontier F = {p_1: T_1, p_2: T_2, ...}."""

    def __init__(self, initial_ports: Optional[Sequence[FrontierPort]] = None) -> None:
        self._ports: Dict[str, FrontierPort] = {}
        self._consumed: Set[str] = set()
        for fp in initial_ports or ():
            self._ports[fp.port_id] = fp

    @property
    def open_ports(self) -> List[FrontierPort]:
        unconsumed = [p for pid, p in self._ports.items() if pid not in self._consumed]
        consumed = [p for pid, p in self._ports.items() if pid in self._consumed]
        unconsumed.sort(key=lambda p: p.step_idx, reverse=True)
        consumed.sort(key=lambda p: p.step_idx, reverse=True)
        return unconsumed + consumed

    def add_ingress_node(self, cell: Cell, step_idx: int) -> List[FrontierPort]:
        added: List[FrontierPort] = []
        for out_name, out_port in getattr(cell, "outputs", {}).items():
            sig = getattr(out_port, "signature", out_port)
            if not isinstance(sig, AlgebraicSignature):
                sig = AlgebraicSignature(
                    str(getattr(out_port, "type_name", "any")),
                    str(getattr(out_port, "state", "any")),
                )
            fp = FrontierPort(
                node_id=cell.cell_id,
                port_name=out_name,
                signature=sig,
                step_idx=step_idx,
                cell_ref=cell,
            )
            self._ports[fp.port_id] = fp
            added.append(fp)
        return added

    def match_morphism(
        self,
        cell: Cell,
        sigma: Optional[Substitution] = None,
    ) -> Optional[Tuple[Substitution, List[DAGEdge], Dict[str, Tuple[Cell, str]]]]:
        sub = Substitution(sigma.mappings if sigma else {})
        reg = TypeRegistry.get_instance()
        wired_edges: List[DAGEdge] = []
        port_wire_map: Dict[str, Tuple[Cell, str]] = {}
        used_frontier_pids: Set[str] = set()

        # Honor existing valid planned_port_wires first if their source is still in the frontier
        existing_wires = getattr(cell, "planned_port_wires", None) or {}
        active_node_ids = {fp.node_id for fp in self._ports.values()}

        for in_name, in_port in getattr(cell, "inputs", {}).items():
            in_sig = getattr(in_port, "signature", in_port)
            matched_fp: Optional[FrontierPort] = None

            if in_name in existing_wires:
                prod_c, out_p = existing_wires[in_name]
                prod_id = getattr(prod_c, "cell_id", str(prod_c))
                cand_pid = f"{prod_id}.{out_p}"
                if cand_pid in self._ports and prod_id in active_node_ids:
                    is_stale_wire = False
                    prod_step = self._ports[cand_pid].step_idx
                    if cand_pid in self._consumed:
                        is_stale_wire = True
                    else:
                        for op in self.open_ports:
                            if op.step_idx > prod_step and op.port_id not in self._consumed and op.node_id != cell.cell_id:
                                if unify(op.signature, in_sig, sub) is not None:
                                    is_stale_wire = True
                                    break
                    if not is_stale_wire:
                        fp_cand = self._ports[cand_pid]
                        u = unify(fp_cand.signature, in_sig, sub)
                        if u is not None:
                            sub = u
                            matched_fp = fp_cand

            if matched_fp is None:
                for fp in self.open_ports:
                    if fp.port_id in used_frontier_pids or fp.node_id == cell.cell_id:
                        continue
                    u = unify(fp.signature, in_sig, sub)
                    if u is not None:
                        sub = u
                        matched_fp = fp
                        break

            if matched_fp is not None:
                used_frontier_pids.add(matched_fp.port_id)
                wired_edges.append(
                    DAGEdge(
                        src_node_id=matched_fp.node_id,
                        src_port=matched_fp.port_name,
                        dst_node_id=cell.cell_id,
                        dst_port=in_name,
                        carrier_type=matched_fp.signature.type_name,
                    )
                )
                if matched_fp.cell_ref is not None:
                    port_wire_map[in_name] = (matched_fp.cell_ref, matched_fp.port_name)
            elif getattr(in_port, "required", False) and getattr(in_port, "default_value", None) is None:
                t_low = str(getattr(in_sig, "type_name", "")).lower()
                has_enum = bool(getattr(in_port, "enum_values", None))
                if not reg.is_literal_groundable(t_low, has_enum=has_enum):
                    return None

        return sub, wired_edges, port_wire_map

    def apply_morphism(
        self,
        cell: Cell,
        wired_edges: Sequence[DAGEdge],
        step_idx: int,
    ) -> List[FrontierPort]:
        for edge in wired_edges:
            pid = f"{edge.src_node_id}.{edge.src_port}"
            self._consumed.add(pid)
        return self.add_ingress_node(cell, step_idx=step_idx)

    @classmethod
    def build_dag_plan(cls, cells: Sequence[Cell]) -> DAGPlan:
        frontier = cls()
        nodes: List[Cell] = list(cells)
        detailed_edges: List[DAGEdge] = []
        sigma = Substitution()

        try:
            from .planner import _active_vocab
            vocab = _active_vocab()
            is_ingress_fn = vocab.is_ingress
        except Exception:
            try:
                from planner import _active_vocab
                vocab = _active_vocab()
                is_ingress_fn = vocab.is_ingress
            except Exception:
                is_ingress_fn = lambda c: getattr(c, "stage", None) == 1 or str(getattr(c, "node_role", "")).lower() == "source"

        for step_idx, cell in enumerate(nodes):
            prefix_ids = {nodes[k].cell_id for k in range(step_idx)}
            pw = getattr(cell, "planned_port_wires", None)
            if isinstance(pw, dict):
                cell.planned_port_wires = {
                    k: v for k, v in pw.items()
                    if getattr(v[0] if isinstance(v, tuple) else v, "cell_id", str(v[0] if isinstance(v, tuple) else v)) in prefix_ids
                }
            bp = getattr(cell, "bound_parent_ids", None)
            if isinstance(bp, set):
                cell.bound_parent_ids = bp & prefix_ids

            is_ing = is_ingress_fn(cell)
            if is_ing and step_idx == 0:
                frontier.add_ingress_node(cell, step_idx=step_idx)
                continue

            match = frontier.match_morphism(cell, sigma=sigma)
            if match is not None:
                sigma, edges, port_wire_map = match
                cell.planned_port_wires = port_wire_map
                cell.bound_parent_ids = {e.src_node_id for e in edges}
                detailed_edges.extend(edges)
                frontier.apply_morphism(cell, edges, step_idx=step_idx)
            else:
                frontier.add_ingress_node(cell, step_idx=step_idx)

        tuple_edges = [e.as_tuple() for e in detailed_edges]
        return DAGPlan(nodes=nodes, edges=tuple_edges, detailed_edges=detailed_edges)



@dataclass
class GoalSpecification:
    """
    Track 6.1: Underspecified Goal Specification G = (I, O, Phi).
    """
    prompt: str
    initial_ports: Dict[str, AlgebraicSignature] = field(default_factory=dict)
    target_outputs: Dict[str, AlgebraicSignature] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)


def fill_typed_hole(
    hole: TypedHole,
    kernel_generator: Optional[Any] = None,
    sandbox: Optional[Any] = None,
    test_inputs: Optional[Dict[str, Any]] = None,
) -> TypedHole:
    """Raw unconstrained inductive kernel generation is disabled in STRATA."""
    raise RuntimeError(
        f"Cannot fill typed hole '{hole.hole_id}' via raw LLM code generation: "
        "all programs must compose verified lattice and control-flow cells."
    )


__all__ = [
    "SemanticBlackboard",
    "TypedHole",
    "TypedHoleCell",
    "GoalSpecification",
    "DAGEdge",
    "DAGPlan",
    "FrontierPort",
    "OpenPortFrontier",
    "fill_typed_hole",
]
