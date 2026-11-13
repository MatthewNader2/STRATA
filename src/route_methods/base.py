"""
src/route_methods/base.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Abstract base class and shared utilities for pluggable RouteMethods (M0 - M6).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import copy
import math
from typing import Dict, List, Optional, Set, Tuple, Any

from log_config import get_logger

try:
    from ..lattice import Cell, MicroCell, MacroCell, LatticeOrchestrator, TypeRegistry, is_path_port, is_macro_cell
    from ..unification import unify, Substitution, ExecutionContext, unify_cell_with_scope
    from ..tokenizer import CellTokenizer
    from ..planner import LatticePlanner
    from ..utils import tokenize_alphanumeric
    from ..dag_plan import DAGPlan, OpenPortFrontier
except (ImportError, ValueError):
    from lattice import Cell, MicroCell, MacroCell, LatticeOrchestrator, TypeRegistry, is_path_port, is_macro_cell
    from unification import unify, Substitution, ExecutionContext, unify_cell_with_scope
    from tokenizer import CellTokenizer
    from planner import LatticePlanner
    from utils import tokenize_alphanumeric
    from dag_plan import DAGPlan, OpenPortFrontier

logger = get_logger("route_methods")


_ACTIVE_ORCHESTRATOR: List[Any] = []


def _fmt_port_sig(p: Any) -> str:
    """Formats a port signature compactly as `type[state]` when state is non-top and non-void."""
    sig = getattr(p, "signature", None)
    tn = str(getattr(sig, "type_name", "") or getattr(p, "abstract_type", "") or "any")
    st = str(getattr(sig, "state", "") or "").strip()
    if not st:
        return tn
    st_low = st.lower()
    reg = TypeRegistry.get_instance()
    if reg.is_declared_top(st_low) or reg.is_void_type(st_low):
        return tn
    return f"{tn}[{st}]"


def _bridge_roles() -> Set[str]:
    """Declared bridge/adapter role names come from the trees' verification_semantics."""
    try:
        from lattice import TypeRegistry
        return {str(r).strip().lower() for r in TypeRegistry.get_instance().get_verification_semantics("bridge_roles")}
    except Exception:
        return set()


def _clause_has_egress_intent(clause: str) -> bool:
    """Language-level materialization intent for a clause: declared egress
    verbs (planner.EGRESS_INTENT_TOKENS) when not an in-memory variable target."""
    try:
        from ..planner import EGRESS_INTENT_TOKENS
    except (ImportError, ValueError):
        from planner import EGRESS_INTENT_TOKENS
    target = ExecutionContext._extract_target_sink(clause)
    toks = set(CellTokenizer.tokenize_prompt(clause.lower()))
    return bool(toks & EGRESS_INTENT_TOKENS) and not target


def is_terminal_sink(cell: Any, orch: Optional[Any] = None) -> bool:
    """Check if cell is a terminal/sink morphism dynamically via LatticeVocabulary, TypeRegistry, or structure."""
    if getattr(cell, "is_sink", False):
        return True
    try:
        from planner import _active_vocab
        vocab = _active_vocab(orch)
        if vocab is not None and vocab.is_terminal(cell):
            return True
    except Exception:
        pass
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        term_roles = {str(r).lower() for r in (reg.get_verification_semantics("terminal_roles") or ())}
        term_roles.update(str(r).lower() for r in (reg.get_verification_semantics("sink_roles") or ()))
        role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
        if role in term_roles:
            return True
        term_stages = {str(s).strip() for s in (reg.get_verification_semantics("terminal_stages") or ())}
        c_st = str(getattr(cell, "stage", "") or "").strip()
        if term_stages and c_st in term_stages:
            return True
    except Exception:
        term_roles = set()
    if getattr(cell, "is_endable", False):
        return True
    outs = getattr(cell, "outputs", {}) or {}
    if not outs:
        return True
    role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
    return role in term_roles


def is_ingress_source(cell: Any, orch: Optional[Any] = None) -> bool:
    """Check if cell is an ingress/source morphism dynamically via LatticeVocabulary, TypeRegistry, or structure."""
    if getattr(cell, "is_ingress", False):
        return True
    try:
        from planner import _active_vocab
        vocab = _active_vocab(orch)
        if vocab is not None and vocab.is_ingress(cell):
            return True
    except Exception:
        pass
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        ing_roles = {str(r).lower() for r in (reg.get_verification_semantics("ingress_roles") or ())}
        ing_roles.update(str(r).lower() for r in (reg.get_verification_semantics("source_roles") or ()))
        role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
        if role in ing_roles:
            return True
        ing_stages = {str(s).strip() for s in (reg.get_verification_semantics("ingress_stages") or ())}
        c_st = str(getattr(cell, "stage", "") or "").strip()
        if ing_stages and c_st in ing_stages:
            return True
    except Exception:
        ing_roles = set()
    ins = getattr(cell, "inputs", {}) or {}
    required_data_ins = [
        p for p in ins.values()
        if getattr(p, "required", True) and getattr(p, "structure_kind", None) != "scalar"
    ]
    if not required_data_ins:
        return True
    role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
    return role in ing_roles


def is_intermediate_transform(cell: Any, orch: Optional[Any] = None) -> bool:
    """Check if cell is an intermediate/transform morphism dynamically via LatticeVocabulary, TypeRegistry, or structure."""
    try:
        from planner import _active_vocab
        vocab = _active_vocab(orch)
        if vocab is not None and vocab.is_transform(cell):
            return True
    except Exception:
        pass
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        tf_roles = {str(r).lower() for r in (reg.get_verification_semantics("transform_roles") or ())}
        role = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
        if role in tf_roles:
            return True
        tf_stages = {str(s).strip() for s in (reg.get_verification_semantics("transform_stages") or ())}
        c_st = str(getattr(cell, "stage", "") or "").strip()
        if tf_stages and c_st in tf_stages:
            return True
    except Exception:
        pass
    return not is_ingress_source(cell, orch) and not is_terminal_sink(cell, orch)


class RouteMethod(ABC):
    """
    Abstract base class for all STRATA RouteMethods (M0 - M6).
    """
    name: str = "base"

    def __init__(self, orchestrator: Optional[LatticeOrchestrator] = None, **kwargs):
        self.orchestrator = orchestrator
        self.kwargs = kwargs
        self.trace_events: List[Dict[str, Any]] = []

    def is_ingress_node(self, cell: Any, orch: Optional[Any] = None) -> bool:
        return is_ingress_source(cell, orch or self.orchestrator)

    def is_sink_node(self, cell: Any, orch: Optional[Any] = None) -> bool:
        return is_terminal_sink(cell, orch or self.orchestrator)

    def is_transform_node(self, cell: Any, orch: Optional[Any] = None) -> bool:
        return is_intermediate_transform(cell, orch or self.orchestrator)

    effective_method: Optional[str] = None      # set when this method handed the request to another
    fallback_reason: Optional[str] = None

    def _trace(self, event: str, **fields: Any) -> None:
        """Structured planner telemetry surfaced by `--debug` (never affects planning)."""
        if not hasattr(self, "trace_events"):
            self.trace_events = []
        self.trace_events.append({"event": event, **fields})

    @abstractmethod
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
        """
        Plans a type-valid sequence of Cell morphisms from the semantic tunnel.
        """
        pass

    def _blend_latent_transition(self, src_cell: Cell, dst_cell: Cell, base_aff: float) -> float:
        """Applies learned continuous transition compatibility when Profile AE is active."""
        try:
            try:
                from inference import ModelManager
            except (ImportError, ValueError):
                from ..inference import ModelManager
            mm = ModelManager.get_instance()
            if mm is None or getattr(mm, "current_profile_name", None) == "0":
                return base_aff
            prof = getattr(mm, "active_profile", None)
            if prof is None or not hasattr(prof, "score_transition"):
                return base_aff

            src_in = getattr(src_cell, "primary_input", None)
            src_in_sig = getattr(src_in, "signature", src_in) if src_in is not None else "__start__"
            src_out = getattr(src_cell, "primary_output", None)
            src_out_sig = getattr(src_out, "signature", src_out) if src_out is not None else "__end__"
            dst_in = getattr(dst_cell, "primary_input", None)
            dst_in_sig = getattr(dst_in, "signature", dst_in) if dst_in is not None else "__start__"
            dst_out = getattr(dst_cell, "primary_output", None)
            dst_out_sig = getattr(dst_out, "signature", dst_out) if dst_out is not None else "__end__"

            s_fwd = float(mm.score_transition(src_out_sig, dst_cell.cell_id, dst_out_sig))
            s_chain = float(mm.score_transition(src_in_sig, src_cell.cell_id, dst_in_sig))
            t_ae = max(0.0, min(1.0, 0.5 * (0.5 * (s_fwd + s_chain) + 1.0)))
            return (0.7 * base_aff + 0.3 * t_ae) if base_aff > 0.0 else (0.5 * t_ae)
        except Exception:
            return base_aff

    def calculate_edge_affinity(
        self,
        src_cell: Cell,
        dst_cell: Cell,
        orchestrator: Optional[LatticeOrchestrator] = None,
        relevance_map: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculates empirical edge affinity score between two cells.
        AST-mined edges from real code snippets receive the highest affinity,
        followed by LLM seed edges, egress completion, and reachability.
        """
        orch = orchestrator or self.orchestrator

        # Synaptic Macro-Goal Edge Reinforcement (Synapses in the Brain):
        try:
            from config import settings
            macros_enabled = bool(settings.macros_enabled)
        except Exception:
            macros_enabled = True

        if macros_enabled and orch:
            if hasattr(orch, "macro_edges"):
                if (src_cell.cell_id, dst_cell.cell_id) in orch.macro_edges:
                    return 0.85
            elif hasattr(orch, "loaded_cells"):
                for m in orch.loaded_cells.values():
                    if getattr(m, "cell_type", "") == "macro" or getattr(m, "sub_cells", None):
                        topo = getattr(m, "internal_topology", {}) or {}
                        is_macro_edge = dst_cell.cell_id in topo.get(src_cell.cell_id, ())
                        if not is_macro_edge and getattr(m, "sub_cells", None):
                            subs = m.sub_cells
                            for i in range(len(subs) - 1):
                                if subs[i] == src_cell.cell_id and subs[i + 1] == dst_cell.cell_id:
                                    is_macro_edge = True
                                    break
                        if is_macro_edge:
                            return 0.85

        # 1. Forward declared edges on src_cell
        dst_id_lower = dst_cell.cell_id.lower()
        for edge in getattr(src_cell, "edges", []):
            tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt_id and (tgt_id == dst_cell.cell_id or str(tgt_id).lower() == dst_id_lower):
                aff = float(edge.get("affinity_score", 0.5) if isinstance(edge, dict) else getattr(edge, "affinity_score", 0.5))
                prov = edge.get("score_provenance") if isinstance(edge, dict) else getattr(edge, "score_provenance", "")
                if prov == "ast_mined":
                    return min(1.0, aff * 1.25)
                return aff

        # 2. Reverse declared edges on dst_cell
        src_id_lower = src_cell.cell_id.lower()
        for edge in getattr(dst_cell, "edges", []):
            tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt_id and (tgt_id == src_cell.cell_id or str(tgt_id).lower() == src_id_lower):
                aff = float(edge.get("affinity_score", 0.3) if isinstance(edge, dict) else getattr(edge, "affinity_score", 0.3))
                return aff * 0.75

        # 3. Intermediate -> Terminal Sink Egress Completion
        src_domain = getattr(src_cell, "domain_name", "")
        dst_domain = getattr(dst_cell, "domain_name", "")
        is_src_intermediate = is_intermediate_transform(src_cell, orch)
        is_dst_sink = is_terminal_sink(dst_cell, orch)
        if is_src_intermediate and is_dst_sink and src_domain and dst_domain and src_domain == dst_domain:
            # Gate the +0.75 egress completion bonus on src_cell having positive prompt relevance
            rel_map = relevance_map if relevance_map is not None else getattr(self, "relevance_map", None)
            if rel_map is None or rel_map.get(src_cell.cell_id, 0.0) > 0.0:
                return 0.75

        # 4. Topological reachability in orchestrator if built
        if orch:
            adj = getattr(orch, "_adjacency", None) or getattr(orch, "adjacency", None)
            if adj and dst_cell.cell_id in adj.get(src_cell.cell_id, ()):
                return self._blend_latent_transition(src_cell, dst_cell, 0.35)

        # 5. Same-domain morphism continuity
        if src_domain and dst_domain and src_domain == dst_domain:
            return self._blend_latent_transition(src_cell, dst_cell, 0.20)

        return self._blend_latent_transition(src_cell, dst_cell, 0.0)

    @staticmethod
    def is_file_format_compatible(cell: Cell, file_path_str: str) -> bool:
        """
        Checks if a cell's file/path port is compatible with the given file path's extension.
        100% domain-agnostic: derives declared format from generic signature File[Carrier, Format]
        or declared cell metadata.
        """
        if not file_path_str or not isinstance(file_path_str, str) or "." not in file_path_str:
            return True
        ext = file_path_str.rpartition(".")[2].strip().lower()
        if not ext:
            return True

        path_ports = [
            p for p in list(cell.inputs.values()) + list(cell.outputs.values())
            if is_path_port(p)
        ]
        if not path_ports:
            return True

        reg = TypeRegistry.get_instance()
        fmt_groups = reg.get_format_groups()
        loc_ctors = set(reg.get_verification_semantics("locator_constructors") or ())
        for p in path_ports:
            tn = str(getattr(p, "type_name", "") or "").strip()
            fmt = ""
            has_concrete_fmt = False
            if "[" in tn and tn.endswith("]"):
                ctor, _, inner = tn[:-1].partition("[")
                if ctor.strip().lower() in loc_ctors:
                    parts = [part.strip().lower() for part in inner.split(",") if part.strip()]
                    if len(parts) >= 2:
                        fmt = parts[1]
                        if fmt and fmt != "generic" and not reg.is_declared_top(fmt):
                            has_concrete_fmt = fmt not in fmt_groups
                            if has_concrete_fmt:
                                if (
                                    ext == fmt
                                    or (len(ext) >= 2 and fmt.startswith(ext))
                                    or reg.is_subtype(ext, fmt)
                                    or reg.is_subtype(fmt, ext)
                                ):
                                    return True
                                return False
                            # Group-level format (e.g. TabularFormat, ImageFormat): match subtypes in that group.
                            if reg.is_subtype(ext, fmt) or ext == fmt:
                                return True
                            for anc in fmt_groups:
                                if reg.is_subtype(ext, anc) and (reg.is_subtype(fmt, anc) or fmt == anc):
                                    return True
                            return False
            port_fmts = getattr(p, "supported_formats", None)
            if port_fmts:
                clean_fmts = {str(f).strip().lower() for f in port_fmts}
                if "*" not in clean_fmts and ext not in clean_fmts:
                    return False
            if "[" in tn and tn.endswith("]"):
                ctor, _, inner = tn[:-1].partition("[")
                if ctor.strip().lower() in loc_ctors:
                    parts = [part.strip().lower() for part in inner.split(",") if part.strip()]
                    if len(parts) >= 2:
                        fmt = parts[1]
                        if not fmt or fmt == "generic" or reg.is_declared_top(fmt):
                            carrier = parts[0]
                            try:
                                placeholder = reg.get_asset_placeholder(carrier, "")
                                if placeholder and "." in placeholder:
                                    carrier_ext = placeholder.rpartition(".")[2].strip().lower()
                                    if carrier_ext and carrier_ext != "dat":
                                        if ext == carrier_ext or reg.is_subtype(ext, carrier_ext) or reg.is_subtype(carrier_ext, ext):
                                            return True
                                        for anc in fmt_groups:
                                            if reg.is_subtype(ext, anc) and reg.is_subtype(carrier_ext, anc):
                                                return True
                                        return False
                            except Exception:
                                pass
                            return True
                    elif len(parts) == 1:
                        target = parts[0]
                        if not target or target == "generic" or reg.is_declared_top(target):
                            return True
                        if ext == target or target in ext or ext in target or reg.is_subtype(ext, target) or reg.is_subtype(target, ext):
                            return True
                        for anc in reg.get_format_groups():
                            if reg.is_subtype(ext, anc) and (reg.is_subtype(target, anc) or target in anc):
                                return True
                        try:
                            placeholder = reg.get_asset_placeholder(target, "")
                            if placeholder and "." in placeholder:
                                carrier_ext = placeholder.rpartition(".")[2].strip().lower()
                                if carrier_ext and carrier_ext != "dat":
                                    if ext == carrier_ext or reg.is_subtype(ext, carrier_ext) or reg.is_subtype(carrier_ext, ext):
                                        return True
                                    for anc in reg.get_format_groups():
                                        if reg.is_subtype(ext, anc) and reg.is_subtype(carrier_ext, anc):
                                            return True
                                    return False
                        except Exception:
                            pass
                        return False
        return True

    def _get_prompt_literals(self, prompt: Optional[str] = None) -> Tuple[List[Any], List[Any], List[Any]]:
        p = prompt or getattr(self, "_current_prompt", "") or ""
        if not p:
            return [], [], []
        cached = getattr(self, "_cached_literals", None)
        if cached is not None and cached[0] == p:
            return cached[1]
        universal = ExecutionContext._extract_universal_literals(p)
        id_lits = [v for _, k, v in universal if k == "identifier"]
        qs_lits = [v for _, k, v in universal if k == "quoted_str"]
        num_lits = [v for _, k, v in universal if k == "numeric"]
        res = (id_lits, qs_lits, num_lits)
        self._cached_literals = (p, res)
        return res

    def step_unifies(self, producer: Cell, consumer: Cell, prev_path: Optional[List[Cell]] = None, prompt: Optional[str] = None) -> bool:
        """
        Checks if consumer cell can validly execute in the multi-carrier environment.
        Enforces:
          1. Stage 1 cells cannot be appended as intermediate transitions.
          2. Direct morphism condition: producer must have an output edge to consumer.
          3. Type-monadic verification via LatticePlanner._verify_transition with prompt literals.
        """
        if self.is_ingress_node(consumer):
            return False

        from unification import is_edge_valid
        producer_feeds_consumer = any(
            is_edge_valid(out, inp, producer, consumer)
            for out in producer.outputs.values()
            for inp in consumer.inputs.values()
        )
        if not producer_feeds_consumer:
            return False

        orch = self.orchestrator
        chain = prev_path if prev_path else [producer]
        if orch:
            planner = getattr(self, "_cached_planner", None)
            if planner is None:
                planner = LatticePlanner(orchestrator=orch)
                self._cached_planner = planner
            id_lits, qs_lits, num_lits = self._get_prompt_literals(prompt)
            sigma = planner._verify_transition(
                chain,
                consumer,
                Substitution(),
                identifier_literals=id_lits,
                quoted_str_literals=qs_lits,
                numeric_literals=num_lits,
            )
            if sigma is not None:
                return True
            # Multi-carrier DAG scope verification:
            dag_res = unify_cell_with_scope(consumer, chain, Substitution())
            return dag_res is not None

        # Fallback if orchestrator not provided
        req_inputs = [p for p in consumer.inputs.values() if p.required]
        prod_outputs = list(producer.outputs.values())
        if not prod_outputs:
            return False
        cand_inputs = req_inputs if req_inputs else list(consumer.inputs.values())
        for out_port in prod_outputs:
            for in_port in cand_inputs:
                if unify(out_port.signature, in_port.signature) is not None:
                    return True
        if prev_path and len(prev_path) > 1:
            for anc in reversed(prev_path[:-1]):
                for out_port in anc.outputs.values():
                    for in_port in cand_inputs:
                        if unify(out_port.signature, in_port.signature) is not None:
                            return True
        dag_res = unify_cell_with_scope(consumer, chain, Substitution())
        return dag_res is not None

    def step_unifies_dag(
        self,
        consumer: Cell,
        available_cells: List[Cell],
        sigma: Optional[Substitution] = None,
        ctx: Optional[ExecutionContext] = None,
    ) -> bool:
        """
        DAG-frontier unification check: verifies if consumer can validly execute
        given the full multi-carrier scope of available ancestor cells (fork/join).
        """
        if self.is_ingress_node(consumer) or not available_cells:
            return False
        res = unify_cell_with_scope(consumer, available_cells, sigma=sigma, context=ctx)
        return res is not None

    def find_bridge_chain(
        self,
        src_cell: Cell,
        dst_cell: Cell,
        candidates: Optional[List[Cell]] = None,
        max_hops: int = 4,
        prev_path: Optional[List[Cell]] = None,
    ) -> List[Cell]:
        """
        Unified Shortest-Path Precondition Bridge Closure:
        When linking src_cell to dst_cell:
          - If src_cell unifies directly with dst_cell: returns []
          - If direct connection fails: runs shortest-path search over the
            precomputed transition graph in LatticeOrchestrator to find the
            minimal bridge morphism chain [b_1, ..., b_k].
        """
        if self.step_unifies(src_cell, dst_cell):
            return []
        orch = self.orchestrator
        if prev_path:
            single = self.find_bridge(src_cell, dst_cell, candidates or [], orchestrator=orch, prev_path=prev_path)
            if single:
                b_cloned = single.clone() if hasattr(single, "clone") else copy.copy(single)
                b_cloned.matched_clause_idx = getattr(dst_cell, "matched_clause_idx", None)
                return [b_cloned]
        if orch and hasattr(orch, "find_shortest_bridge_path"):
            chain = orch.find_shortest_bridge_path(src_cell, dst_cell, max_hops=max_hops, candidate_pool=candidates)
            if chain:
                cloned_chain = []
                for b in chain:
                    b_cloned = b.clone() if hasattr(b, "clone") else copy.copy(b)
                    b_cloned.matched_clause_idx = getattr(dst_cell, "matched_clause_idx", None)
                    cloned_chain.append(b_cloned)
                if prev_path:
                    full_scope = list(prev_path) + cloned_chain
                    if not (
                        self.step_unifies(cloned_chain[-1], dst_cell, prev_path=full_scope)
                        or self.step_unifies_dag(dst_cell, full_scope)
                    ):
                        cloned_chain = []
                if cloned_chain:
                    return cloned_chain
        single = self.find_bridge(src_cell, dst_cell, candidates or [], orchestrator=orch, prev_path=prev_path)
        return [single] if single else []

    def find_bridge(
        self,
        src_cell: Cell,
        dst_cell: Cell,
        candidates: List[Cell],
        orchestrator: Optional[LatticeOrchestrator] = None,
        prev_path: Optional[List[Cell]] = None,
        allow_holes: bool = False
    ) -> Optional[Cell]:
        """
        Finds a bridging cell B in candidates such that:
        B unifies with the current scope/src_cell AND dst_cell unifies after B.
        Returns a cloned instance parameter-grounded to dst_cell's clause literals.
        """
        orch = orchestrator or self.orchestrator
        best_b: Optional[Cell] = None
        best_score = -1.0
        current_scope = prev_path if prev_path else [src_cell]

        reg = getattr(orch, "lattice", None) or getattr(orch, "registry", None)
        if reg is None or not hasattr(reg, "get_verification_semantics"):
            try:
                from lattice import TypeRegistry
                reg = TypeRegistry.get_instance()
            except Exception:
                reg = None

        prompt_str = getattr(self, "prompt", "") or (getattr(orchestrator, "prompt", "") if orchestrator else "")
        prompt_l = str(prompt_str).lower()

        for cand in candidates:
            if cand.cell_id in (src_cell.cell_id, dst_cell.cell_id):
                continue
            if is_ingress_source(cand, orchestrator):
                continue
            if getattr(cand, "is_combinator", False) or getattr(cand, "node_role", "") == "combinator" or getattr(cand, "role", "") == "combinator" or getattr(cand, "node_type", "") == "combinator":
                continue
            if is_terminal_sink(cand, orchestrator):
                continue

            cand_role = str(getattr(cand, "node_role", "") or getattr(cand, "role", "") or "").lower()
            cand_tags = {str(t).lower() for t in (getattr(cand, "semantic_tags", None) or [])}
            bridge_tokens = set(reg.get_verification_semantics("bridge_tokens") or ())
            assigned_val_roles = set(reg.get_verification_semantics("assigned_value_roles") or ())
            assigned_val_binds = set(reg.get_verification_semantics("assigned_value_binds") or ())
            carrier_binds = set(reg.get_verification_semantics("carrier_binds") or ())
            df_roles = set(reg.get_verification_semantics("dataflow_roles") or ())
            compound_sk = set(reg.get_verification_semantics("compound_structure_kinds") or ())
            # Domain-agnostic Lens Injection: a morphism that writes an assigned value/field
            # back into a composite carrier (mu : Record x Field -> Record) is an eligible bridge.
            is_lens_setter = (
                any(getattr(p, "binds", None) in assigned_val_binds or getattr(p, "port_role", None) in assigned_val_roles for p in cand.inputs.values())
                and any(getattr(p, "binds", None) in carrier_binds or getattr(p, "port_role", None) in df_roles for p in cand.inputs.values())
                and any(getattr(p, "structure_kind", None) in compound_sk or getattr(p, "abstract_type", None) in reg.get_composite_carrier_roots() for p in cand.outputs.values())
            )

            is_declared_bridge = (
                cand_role in _bridge_roles()
                or getattr(cand, "node_type", "") in _bridge_roles()
                or bool(bridge_tokens & cand_tags)
                or is_lens_setter
            )
            resolves_property_mismatch = False

            # Check if cand resolves a property precondition of dst_cell violated by src_cell / current_scope
            dst_pres = getattr(dst_cell, "preconditions", None) or []
            src_posts = getattr(src_cell, "postconditions", None) or []
            cand_posts = getattr(cand, "postconditions", None) or []

            def _c_prop(c, key, default=None):
                if isinstance(c, dict):
                    return c.get(key, default)
                return getattr(c, key, default)

            if dst_pres and cand_posts:
                for dp in dst_pres:
                    dp_prop = _c_prop(dp, "property")
                    dp_val = _c_prop(dp, "value")
                    dp_op = _c_prop(dp, "operator")
                    if dp_op == "==" and dp_prop is not None:
                        src_conflict = any(
                            _c_prop(sp, "property") == dp_prop
                            and _c_prop(sp, "operator") == "=="
                            and _c_prop(sp, "value") != dp_val
                            for sp in src_posts
                        )
                        cand_resolves = any(
                            _c_prop(cp, "property") == dp_prop
                            and _c_prop(cp, "operator") == "=="
                            and _c_prop(cp, "value") == dp_val
                            for cp in cand_posts
                        )
                        if src_conflict and cand_resolves:
                            resolves_property_mismatch = True
                            break

            resolves_qualifier_mismatch = False
            dst_rejs = set()
            for inp in dst_cell.inputs.values():
                r_raw = getattr(inp, "rejected_qualifiers", None) or ()
                sig_inner = getattr(inp, "signature", None)
                if sig_inner:
                    r_raw = r_raw or getattr(sig_inner, "rejected_qualifiers", None) or ()
                for r in r_raw:
                    if isinstance(r, (tuple, list, set)):
                        dst_rejs.update(str(x).lower() for x in r)
                    else:
                        dst_rejs.add(str(r).lower())

            if dst_rejs:
                src_quals = set()
                for scope_c in (current_scope or [src_cell]):
                    for o in getattr(scope_c, "outputs", {}).values():
                        q_raw = getattr(o, "qualifiers", None) or getattr(getattr(o, "signature", None), "qualifiers", None) or ()
                        for q in q_raw:
                            if isinstance(q, (tuple, list, set)):
                                src_quals.update(str(x).lower() for x in q)
                            else:
                                src_quals.add(str(q).lower())
                conflict = src_quals & dst_rejs
                if conflict:
                    c_out = getattr(cand, "primary_output", None) or (next(iter(cand.outputs.values())) if getattr(cand, "outputs", None) else None)
                    if c_out is not None:
                        c_out_quals = set()
                        raw_co = getattr(c_out, "qualifiers", None) or getattr(getattr(c_out, "signature", None), "qualifiers", None) or ()
                        for q in raw_co:
                            if isinstance(q, (tuple, list, set)):
                                c_out_quals.update(str(x).lower() for x in q)
                            else:
                                c_out_quals.add(str(q).lower())
                        if not (c_out_quals & dst_rejs):
                            for p_in in cand.inputs.values():
                                in_acc = set()
                                raw_acc = getattr(p_in, "accepted_states", None) or getattr(getattr(p_in, "signature", None), "accepted_states", None) or ()
                                in_st = str(getattr(p_in, "state", "") or getattr(getattr(p_in, "signature", None), "state", "") or "")
                                for a in raw_acc:
                                    in_acc.update(CellTokenizer.tokenize_identifier(str(a).lower()))
                                in_acc.update(CellTokenizer.tokenize_identifier(in_st.lower()))
                                if in_acc & conflict:
                                    resolves_qualifier_mismatch = True
                                    break

            if not is_declared_bridge and not resolves_property_mismatch and not resolves_qualifier_mismatch:
                continue
            from unification import is_edge_valid
            can_src_to_cand = self.step_unifies(src_cell, cand, prev_path=current_scope) or self.step_unifies_dag(cand, current_scope)
            # A candidate is only a bridge to dst_cell if dst_cell actually consumes an output from cand
            cand_feeds_dst = any(
                is_edge_valid(out, inp, cand, dst_cell)
                for out in cand.outputs.values()
                for inp in dst_cell.inputs.values()
            )
            can_cand_to_dst = cand_feeds_dst and (self.step_unifies(cand, dst_cell, prev_path=current_scope + [cand]) or self.step_unifies_dag(dst_cell, current_scope + [cand]))
            if can_src_to_cand and can_cand_to_dst:
                aff1 = self.calculate_edge_affinity(src_cell, cand, orchestrator)
                aff2 = self.calculate_edge_affinity(cand, dst_cell, orchestrator)
                score = aff1 + aff2
                if resolves_property_mismatch:
                    score += 5.0
                    req_in = sum(1 for p in cand.inputs.values() if p.required)
                    opt_in = len(cand.inputs) - req_in
                    if req_in == 1 and opt_in == 0:
                        score += 3.0  # Clean unary element-wise transform preserving element count
                if score > best_score:
                    best_score = score
                    best_b = cand

        if best_b:
            b_inst = best_b.clone() if hasattr(best_b, "clone") else best_b
            b_inst.clause_literals = getattr(dst_cell, "clause_literals", None)
            b_inst.matched_clause_idx = getattr(dst_cell, "matched_clause_idx", None)
            return b_inst

        if orch and hasattr(orch, "find_shortest_bridge_path"):
            chain = orch.find_shortest_bridge_path(src_cell, dst_cell, max_hops=3, candidate_pool=candidates)
            if chain:
                b = chain[0].clone() if hasattr(chain[0], "clone") else chain[0]
                b.matched_clause_idx = getattr(dst_cell, "matched_clause_idx", None)
                return b

        return None

    @classmethod
    def is_handle_port(cls, sig: Any) -> bool:
        """Category-theoretic linear handle: structure_kind == 'handle' or declared object receiver."""
        if sig is None:
            return False
        sk = str(getattr(sig, "structure_kind", "") or "").lower()
        if sk == "handle":
            return True
        pr = str(getattr(sig, "port_role", "") or getattr(sig, "derived_role", "") or "").lower()
        try:
            from preflight import PreflightLinter
            if PreflightLinter._is_object_receiver_role(pr):
                return True
        except Exception:
            pass
        try:
            from lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
            tn = str(getattr(sig, "type_name", "") or "").lower()
            if reg and (
                reg.is_subtype(pr, "handle")
                or reg.is_subtype(tn, "handle")
                or reg.is_subtype(pr, "model")
                or reg.is_subtype(tn, "model")
            ):
                return True
        except Exception:
            pass
        return False

    @classmethod
    def active_unconsumed_handles(cls, prefix: List[Cell]) -> List[Tuple[Cell, str, Any]]:
        """
        Category-theoretic affine/linear logic:
        Finds any output in `prefix` typed as a stateful handle / object receiver
        that has not yet been consumed by any subsequent step in `prefix`.
        """
        if not prefix:
            return []
        handles: List[Tuple[Cell, str, Any]] = []
        for i, cell in enumerate(prefix):
            for out_name, out_sig in (getattr(cell, "outputs", {}) or {}).items():
                if cls.is_handle_port(out_sig):
                    consumed = False
                    for sub in prefix[i + 1:]:
                        for inp_sig in (getattr(sub, "inputs", {}) or {}).values():
                            if cls.is_handle_port(inp_sig):
                                consumed = True
                                break
                        if consumed:
                            break
                    if not consumed:
                        handles.append((cell, out_name, out_sig))
        return handles

    @classmethod
    def linear_handle_score(cls, cand: Cell, prefix: List[Cell]) -> int:
        """
        Affine linear score:
        +1 if candidate consumes an active unconsumed handle in scope.
        -1 if candidate introduces a competing handle while an active handle is dangling.
        0 otherwise.
        """
        active = cls.active_unconsumed_handles(prefix)
        cand_consumes_handle = False
        for inp_sig in (getattr(cand, "inputs", {}) or {}).values():
            if cls.is_handle_port(inp_sig):
                cand_consumes_handle = True
                break

        if active and cand_consumes_handle:
            return 1

        cand_produces_handle = False
        for out_sig in (getattr(cand, "outputs", {}) or {}).values():
            if cls.is_handle_port(out_sig):
                cand_produces_handle = True
                break

        if active and cand_produces_handle:
            return -1

        return 0

    @classmethod
    def extract_file_literals(cls, prompt: str, ctx: Optional[Any] = None) -> Tuple[List[str], List[str]]:
        """
        Extracts source and destination file literals from prompt and/or ExecutionContext.
        Preserves typed intent across layers without brittle re-scanning.
        """
        if ctx:
            ctx_src = list(getattr(ctx, "parameters", {}).get("source_uris") or getattr(ctx, "source_files", []) or [])
            ctx_dest = list(getattr(ctx, "parameters", {}).get("dest_uris") or getattr(ctx, "dest_files", []) or [])
            if ctx_src or ctx_dest:
                return ctx_src, ctx_dest

        extracted = ExecutionContext._extract_universal_literals(prompt or "")
        src_lits = [
            val for pos, kind, val in extracted
            if (kind == "file_asset" or (kind == "quoted_str" and ExecutionContext._is_path_string(str(val))))
            and ExecutionContext._asset_direction(prompt or "", pos) != "dest"
        ]
        dest_lits = [
            val for pos, kind, val in extracted
            if (kind == "file_asset" or (kind == "quoted_str" and ExecutionContext._is_path_string(str(val))))
            and ExecutionContext._asset_direction(prompt or "", pos) == "dest"
        ]
        return src_lits, dest_lits

    @classmethod
    def segment_prompt_clauses(cls, prompt: str, ctx: Optional[Any] = None) -> List[str]:
        """
        Partitions user prompt into sequential procedural clauses.
        Syntactic punctuation and coordination segmentation grounded directly in the user specification.
        """
        try:
            from .planner import _segment_prompt_clauses
        except (ImportError, ValueError):
            from planner import _segment_prompt_clauses
        return _segment_prompt_clauses(prompt)

    @classmethod
    def tag_cells_with_clause_indices(
        cls,
        path: List[Cell],
        prompt: str,
        orchestrator: Any = None,
        ctx: Optional[Any] = None,
        clause_semantic_scores: Optional[List[Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tag path cells with the prompt clause they SERVE (matched_clause_idx).

        Evidence comes from token_evidence (corpus IDF, identity precision x
        recall), plus calibrated semantic scores when enabled (Profiles B/A+).
        Enforces monotonic causal ordering: primary evidence winners
        are assigned in causal sequence, and intermediate connectors/bridges
        are topologically bounded within [c_prev, c_next], eliminating spurious
        lexical overlaps that violate causality.
        """
        decisions: List[Dict[str, Any]] = []
        if not path or not prompt:
            return decisions
        from token_evidence import get_evidence, _resolve_clause_semantic_scores
        orch = orchestrator
        if orch is None:
            for c in path:
                orch = getattr(c, "_orchestrator", None) or orch
        if orch is None:
            orch = _ACTIVE_ORCHESTRATOR[0] if _ACTIVE_ORCHESTRATOR else None
        ev = get_evidence(orch)
        clauses = cls.segment_prompt_clauses(prompt, ctx=ctx)
        if ev is None or not clauses:
            return decisions
        clause_toks = [ev.clause_tokens(cl) for cl in clauses]
        clause_sem = None if ev._pure_symbolic else _resolve_clause_semantic_scores(orch, prompt, clause_semantic_scores)
        if clause_sem is not None and len(clause_sem) != len(clauses):
            clause_sem = None

        # Distinct morphism instances: each step in the path must have independent state
        seen_ids = set()
        for idx in range(len(path)):
            if id(path[idx]) in seen_ids:
                path[idx] = path[idx].clone() if hasattr(path[idx], "clone") else copy.copy(path[idx])
            seen_ids.add(id(path[idx]))

        for cell in path:
            cell.matched_clause_idx = None
            cell.clause_literals = None

        # F matrix: path cell x clause
        table = []
        for cell in path:
            row = [ev.identity_overlap(ct, cell) if ct else {"f": 0.0, "hit": 0.0, "rank": (0.0, 0.0, 0.0), "tokens": []} for ct in clause_toks]
            table.append(row)

        # Pass 1: Primary evidence winners across all clauses
        # Greedy highest-rank matching: pair each clause with its strongest lexical-identity cell
        # (or calibrated semantic winner when lexical hit is absent in non-P0 profiles)
        # subject to causal monotonicity (prod_clause <= cons_clause along linear pipeline order).
        candidates = []
        for j in range(len(clauses)):
            sem_map_j = (clause_sem[j] or {}) if clause_sem is not None else {}
            max_sem_j = max(sem_map_j.values(), default=0.0) if sem_map_j else 0.0
            for i in range(len(path)):
                if table[i][j]["hit"] > 0.0:
                    candidates.append(((1, table[i][j]["rank"]), j, i))
                elif clause_sem is not None and max_sem_j >= 0.35:
                    sem_sc = float(sem_map_j.get(path[i].cell_id, 0.0) or 0.0)
                    if sem_sc >= 0.35 and sem_sc >= 0.70 * max_sem_j:
                        candidates.append(((0, (sem_sc, 0.0, sem_sc)), j, i))
        candidates.sort(key=lambda x: x[0], reverse=True)

        assigned_clauses = set()
        assigned_cells = set()
        for rank, j, i in candidates:
            if j not in assigned_clauses and i not in assigned_cells:
                violates = False
                for k in assigned_cells:
                    k_cl = path[k].matched_clause_idx
                    if k_cl is not None:
                        if i < k and j > k_cl:
                            violates = True
                            break
                        if i > k and j < k_cl:
                            violates = True
                            break
                if violates:
                    continue
                path[i].matched_clause_idx = j
                assigned_clauses.add(j)
                assigned_cells.add(i)

        # Ensure any remaining unassigned clause gets its best available unassigned cell
        for j in range(len(clauses)):
            if j in assigned_clauses:
                continue
            best_i, best_rank = None, (0.0, 0.0, 0.0)
            for i in range(len(path)):
                if i not in assigned_cells and table[i][j]["hit"] > 0.0 and table[i][j]["rank"] > best_rank:
                    violates = False
                    for k in assigned_cells:
                        k_cl = path[k].matched_clause_idx
                        if k_cl is not None:
                            if i < k and j > k_cl:
                                violates = True
                                break
                            if i > k and j < k_cl:
                                violates = True
                                break
                    if not violates:
                        best_i, best_rank = i, table[i][j]["rank"]
            if best_i is not None:
                path[best_i].matched_clause_idx = j
                assigned_clauses.add(j)
                assigned_cells.add(best_i)

        # Pass 2: Remaining cells (bridges, connectors, secondary operations)
        # Bounded by nearest preceding and succeeding anchors, or residual clause explanation.
        for i, cell in enumerate(path):
            if getattr(cell, "matched_clause_idx", None) is not None:
                continue

            prec = [path[k].matched_clause_idx for k in range(i) if getattr(path[k], "matched_clause_idx", None) is not None]
            succ = [path[k].matched_clause_idx for k in range(i + 1, len(path)) if getattr(path[k], "matched_clause_idx", None) is not None]
            c_min = prec[-1] if prec else 0
            c_max = succ[0] if succ else (len(clauses) - 1)
            lo, hi = min(c_min, c_max), max(c_min, c_max)

            best_j, best_res = None, (0.0, 0.0, 0.0)
            for j in range(lo, hi + 1):
                if table[i][j]["hit"] <= 0.0 or not clause_toks[j]:
                    continue
                w_tokens = set()
                for k in range(len(path)):
                    if getattr(path[k], "matched_clause_idx", None) == j:
                        w_tokens |= set(table[k][j]["tokens"])
                res = ev.identity_overlap(clause_toks[j] - w_tokens, cell)
                if res["hit"] > 0.0 and res["rank"] > best_res:
                    best_j, best_res = j, res["rank"]

            if best_j is not None:
                cell.matched_clause_idx = best_j
            else:
                best_j_raw, best_rank_raw = None, (0.0, 0.0, 0.0)
                for j in range(lo, hi + 1):
                    if table[i][j]["hit"] > 0.0 and table[i][j]["rank"] > best_rank_raw:
                        best_j_raw, best_rank_raw = j, table[i][j]["rank"]
                if best_j_raw is not None:
                    cell.matched_clause_idx = best_j_raw
                # Leave matched_clause_idx as None for non-lexical connectors/bridges rather
                # than stamping a fallback neighbour clause index onto them.

            decisions.append({
                "cell": cell.cell_id, "tagged_clause": cell.matched_clause_idx,
                "best_rank": tuple(round(x, 3) for x in best_res),
                "outranked_on": [j for j in range(len(clauses))
                                 if table[i][j]["hit"] > 0.0 and table[i][j]["rank"] < (best_res or (0.0, 0.0, 0.0))],
            })

        for i, cell in enumerate(path):
            if getattr(cell, "matched_clause_idx", None) is None:
                if i > 0 and getattr(path[i - 1], "matched_clause_idx", None) is not None:
                    cell.matched_clause_idx = path[i - 1].matched_clause_idx
                else:
                    cell.matched_clause_idx = 0

        # Clause-scoped literals (column names, quoted strings) travel with the clause a cell
        # serves. Only M1/M6 used to attach them, so literal-scoped binding (target column,
        # feature columns, provenance gates) was silently off for every other method.
        try:
            from unification import ExecutionContext as _EC
            for cell in path:
                j = getattr(cell, "matched_clause_idx", None)
                if j is None or j >= len(clauses) or getattr(cell, "clause_literals", None):
                    continue
                cell.clause_literals = [v for _, k, v in _EC._extract_universal_literals(clauses[j])
                                        if k in ("identifier", "quoted_str")]
        except Exception as _lit_err:
            decisions.append({"cell": None, "error": f"clause literal attachment failed: {_lit_err}"})
        return decisions


    def _find_type_producer(self, anchor, routed_path, candidates, clauses, orch):
        """Type-directed repair: the anchor needs a type no cell in scope produces
        (e.g. predict needs a Regressor). Pick the candidate that (1) has identity
        evidence on the SAME prompt clause, (2) unifies from the current scope and
        (3) unifies into the anchor. Ranked by the shared evidence rank. No
        vocabulary, no thresholds: a producer must be demanded by types AND named
        by the clause."""
        from token_evidence import get_evidence
        ev = get_evidence(orch or self.orchestrator)
        idx = getattr(anchor, "matched_clause_idx", None)
        if idx is None and clauses:
            try:
                idx = max(range(len(clauses)), key=lambda i: self.clause_fit(clauses[i], anchor))
            except Exception:
                idx = len(clauses) - 1
        if ev is None or idx is None or not clauses or not routed_path:
            return None
        if idx >= len(clauses):
            idx = len(clauses) - 1
        ctoks = ev.clause_tokens(clauses[idx])
        in_path = {c.cell_id for c in routed_path}
        best, best_rank = None, None
        for cand in candidates:
            if cand.cell_id in in_path or cand.cell_id == anchor.cell_id:
                continue
            o = ev.identity_overlap(ctoks, cand)
            if o["hit"] <= 0.0:
                continue
            direct_cand = (
                self.step_unifies(routed_path[-1], cand, prev_path=routed_path)
                or self.step_unifies_dag(cand, routed_path)
            )
            p_bridges: List[Cell] = []
            if direct_cand and (
                self.step_unifies(cand, anchor, prev_path=routed_path + [cand])
                or self.step_unifies_dag(anchor, routed_path + [cand])
            ):
                cand_rank = (1, o["rank"])
            else:
                # Try bridging routed_path[-1] -> cand (e.g. DataFrame -> ndarray before LinearRegression.fit -> predict)
                p_bridges = self.find_bridge_chain(
                    routed_path[-1], cand, candidates, prev_path=routed_path
                )
                cand_after_bridge = bool(p_bridges) and (
                    self.step_unifies(p_bridges[-1], cand, prev_path=list(routed_path) + p_bridges)
                    or self.step_unifies_dag(cand, list(routed_path) + p_bridges)
                )
                if not cand_after_bridge:
                    continue
                scope_after = list(routed_path) + p_bridges + [cand]
                if not (
                    self.step_unifies(cand, anchor, prev_path=scope_after)
                    or self.step_unifies_dag(anchor, scope_after)
                ):
                    continue
                cand_rank = (0, o["rank"])
            if best_rank is None or cand_rank > best_rank:
                best, best_rank = cand, cand_rank
        if best is None:
            return None
        inst = best.clone() if hasattr(best, "clone") else best
        inst.matched_clause_idx = idx
        inst.clause_literals = getattr(anchor, "clause_literals", {})
        return inst


    def chain_is_valid(self, path: List[Cell], prompt: Optional[str] = None) -> bool:
        """Every step unifies with its predecessor or with the DAG scope of the prefix."""
        p = prompt or getattr(self, "_current_prompt", "") or getattr(self, "prompt", "")
        for i in range(1, len(path)):
            if not (self.step_unifies(path[i - 1], path[i], prev_path=path[:i], prompt=p)
                    or self.step_unifies_dag(path[i], path[:i], ctx=ExecutionContext(prompt=p) if p else None)):
                return False
        return True

    def insert_column_commits(self, path: List[Cell], trace: bool = True) -> List[Cell]:
        """No-op pass-through: Universal morphism graphs do not perform tabular lens commit surgery."""
        return path

    def insert_property_bridges(
        self,
        path: List[Cell],
        prompt: str = "",
        ctx: Optional[Any] = None,
        trace: bool = True,
    ) -> List[Cell]:
        """
        Inserts property-resolving bridges (e.g. converting complex frequency outputs to real
        magnitudes via NUMPY_ABS) when an upstream cell produces a property that conflicts
        with a downstream consumer's declared preconditions or column contract.
        """
        if not path or self.orchestrator is None:
            return path

        # Step 1: Precondition & Typestate Bridge Closure via indexed find_shortest_bridge_path
        bridged_path: List[Cell] = []
        for c in path:
            if not bridged_path:
                bridged_path.append(c)
            else:
                prev_c = bridged_path[-1]
                if not self.orchestrator.cells_directly_unify(prev_c, c):
                    bridges = self.orchestrator.find_shortest_bridge_path(prev_c, c)
                    bridged_path.extend(bridges)
                bridged_path.append(c)
        path = bridged_path

        from dag_plan import OpenPortFrontier
        try:
            try:
                from ..unification import analyze_property_contracts as _prop_analyze
            except (ImportError, ValueError):
                from unification import analyze_property_contracts as _prop_analyze
            from unification import UnificationGate, ExecutionContext
            resolved_prompt = prompt or getattr(ctx, "prompt", "") or getattr(self, "_current_prompt", "") or getattr(self, "prompt", "")
            probe_ctx = ExecutionContext(
                prompt=resolved_prompt,
                parameters=dict(getattr(ctx, "parameters", {}) or {}),
            )

            gate = UnificationGate(orchestrator=self.orchestrator)
            res = gate.unify_pipeline(path, probe_ctx, extract_llm_slots=False)
            if res.is_bottom() or not gate.last_pipeline_bindings:
                return path
            issues = _prop_analyze(gate.last_pipeline_bindings, self.orchestrator)
            if not issues:
                return path

            out = list(path)
            for iss in issues:
                prod_id = iss.get("established_by")
                candidates = iss.get("bridge_candidates", [])
                prop = iss.get("property")
                if not prod_id or not candidates:
                    continue
                prod_idx = next((i for i, c in enumerate(out) if c.cell_id == prod_id), None)
                if prod_idx is None:
                    continue

                # Advance past any intermediate cells that preserve the offending property
                while prod_idx + 1 < len(out) and any(
                    str(eff).lower() == f"preserves:{str(prop).lower()}"
                    for eff in (getattr(out[prod_idx + 1], "effects", None) or [])
                ):
                    prod_idx += 1

                prod_cell = out[prod_idx]
                prod_domain = getattr(prod_cell, "domain", "")
                cons_domain = getattr(out[min(prod_idx + 1, len(out) - 1)], "domain", "")

                best_cand = None
                best_score = -1.0
                for cid in candidates:
                    cand_cell = self.orchestrator.loaded_cells.get(cid)
                    if cand_cell:
                        c_score = 0.0
                        if self.orchestrator.cells_directly_unify(prod_cell, cand_cell):
                            c_score += 10.0
                        if getattr(cand_cell, "domain", "") in (prod_domain, cons_domain):
                            c_score += 5.0
                        if len(cand_cell.inputs) == 1:
                            c_score += 2.0
                        if c_score > best_score:
                            best_score = c_score
                            best_cand = cand_cell

                if best_cand:
                    if prod_idx + 1 < len(out) and out[prod_idx + 1].cell_id == best_cand.cell_id:
                        continue
                    b_cell = best_cand.clone() if hasattr(best_cand, "clone") else best_cand
                    b_cell.matched_clause_idx = getattr(out[prod_idx], "matched_clause_idx", None)

                    # Wire b_cell to receive output from prod_cell
                    b_out_name = getattr(b_cell, "primary_out", None) or (
                        list(b_cell.outputs.keys())[0] if getattr(b_cell, "outputs", None) else "output_var"
                    )
                    b_in_name = getattr(b_cell, "primary_in", None) or (
                        list(b_cell.inputs.keys())[0] if getattr(b_cell, "inputs", None) else None
                    )
                    prod_cell = out[prod_idx]
                    prod_out_name = getattr(prod_cell, "primary_out", None) or (
                        list(prod_cell.outputs.keys())[0] if getattr(prod_cell, "outputs", None) else "output_var"
                    )
                    if b_in_name:
                        b_cell.planned_port_wires = {b_in_name: (prod_cell, prod_out_name)}
                        b_cell.bound_parent_ids = {prod_id}

                    trial = out[: prod_idx + 1] + [b_cell] + out[prod_idx + 1 :]

                    # Rewire downstream consumers in trial[prod_idx + 2:] that were wired to prod_id
                    for downstream in trial[prod_idx + 2 :]:
                        dw_wires = getattr(downstream, "planned_port_wires", None)
                        if isinstance(dw_wires, dict):
                            new_wires = {}
                            for p_name, tgt in dw_wires.items():
                                if isinstance(tgt, tuple) and len(tgt) == 2:
                                    t_cell, t_port = tgt
                                    t_cid = getattr(t_cell, "cell_id", str(t_cell))
                                    if t_cid == prod_id:
                                        new_wires[p_name] = (b_cell, b_out_name)
                                    else:
                                        new_wires[p_name] = tgt
                                else:
                                    t_cid = getattr(tgt, "cell_id", str(tgt))
                                    if t_cid == prod_id:
                                        new_wires[p_name] = (b_cell, b_out_name)
                                    else:
                                        new_wires[p_name] = tgt
                            downstream.planned_port_wires = new_wires
                        dw_parents = getattr(downstream, "bound_parent_ids", None)
                        if isinstance(dw_parents, set) and prod_id in dw_parents:
                            dw_parents.discard(prod_id)
                            dw_parents.add(b_cell.cell_id)

                    trial_unified = False
                    try:
                        probe_ctx_trial = ExecutionContext(
                            prompt=resolved_prompt,
                            parameters=dict(getattr(ctx, "parameters", {}) or {}),
                        )
                        u_trial = gate.unify_pipeline(trial, probe_ctx_trial, extract_llm_slots=False)
                        trial_unified = not u_trial.is_bottom()
                    except Exception:
                        trial_unified = False

                    if not trial_unified:
                        trial_unified = self.chain_is_valid(trial, prompt=resolved_prompt)

                    if trial_unified:
                        if trace:
                            self._trace("property_bridge_inserted", after=prod_id, bridge=b_cell.cell_id,
                                        property=iss.get("property"), reason="resolves downstream property precondition")
                        out = trial
            try:
                return OpenPortFrontier.build_dag_plan(out)
            except Exception:
                return out
        except Exception as _p_err:
            logger.debug(f"[ROUTE] insert_property_bridges skipped: {_p_err}")
            return path


    def prune_unrequested(self, path: List[Cell], prompt: str) -> List[Cell]:
        """
        Post-planning pass shared by every method. Remove cells the coverage audit calls UNREQUESTED
        (outranked on every clause they touch, explaining no leftover clause token) when the chain
        stays type-valid without them. Each removal is traced. Never removes the first cell.
        """
        try:
            from ..token_evidence import audit
        except (ImportError, ValueError):
            from token_evidence import audit
        try:
            from config import settings
        except (ImportError, ValueError):
            from ..config import settings

        max_ops = settings.prune_unrequested_max_ops
        ops_count = 0

        chain_valid_cache: Dict[Tuple[str, ...], bool] = {}
        lint_cache: Dict[Tuple[str, ...], bool] = {}
        pass1_failed_removals: Set[Tuple[str, ...]] = set()
        pass2_failed_removals: Set[Tuple[str, ...]] = set()

        current = list(path)
        changed = True
        outer_iters = 0
        budget_exceeded = False
        while changed and outer_iters < len(current):
            outer_iters += 1
            changed = False
            # Pass 1: Unrequested cell pruning
            for _ in range(len(current)):
                ops_count += 1
                if ops_count >= max_ops:
                    self._trace("prune_unrequested_budget_exceeded", ops=ops_count, max_ops=max_ops)
                    budget_exceeded = True
                    break
                res = audit(current, prompt, self.orchestrator)
                if not res.get("available"):
                    break
                pruned_any = False
                for victim in res.get("issues", []):
                    if victim.get("kind") != "unrequested_cell":
                        continue
                    idx = next((k for k, c in enumerate(current) if c.cell_id == victim["cell"]), None)
                    if idx is None or idx == 0:
                        continue
                    trial = current[:idx] + current[idx + 1:]
                    trial_key = tuple(c.cell_id for c in trial)
                    if trial_key in pass1_failed_removals:
                        continue
                    if trial_key in chain_valid_cache:
                        valid = chain_valid_cache[trial_key]
                    else:
                        valid = self.chain_is_valid(trial, prompt=prompt)
                        chain_valid_cache[trial_key] = valid
                    if not valid:
                        pass1_failed_removals.add(trial_key)
                        self._trace("prune_blocked", cell=victim["cell"], reason="chain would not unify without it",
                                    winners=victim.get("winners"))
                        continue
                    self._trace("pruned_unrequested", cell=victim["cell"], winners=victim.get("winners"),
                                reason="outranked on every clause it touches and explains no leftover clause token")
                    current = trial
                    pruned_any = True
                    changed = True
                    break
                if not pruned_any:
                    break

            if budget_exceeded:
                break

            # Pass 2: Verifier-guided bridge minimization.
            # Any explicit bridge cell is redundant iff removing it still unifies and
            # passes all static preflight contracts (including property preconditions).
            b_roles = _bridge_roles()
            from lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
            est_roles = {str(r).lower() for r in reg.get_verification_semantics("estimator_roles")}
            est_roles.update(str(r).lower() for r in reg.get_verification_semantics("model_roles"))
            from planner import _active_vocab
            vocab = _active_vocab(getattr(self, "orchestrator", None))
            bridge_roles = set(b_roles) | set(reg.get_verification_semantics("bridge_roles"))
            idx = 1
            while idx < len(current) - 1:
                ops_count += 1
                if ops_count >= max_ops:
                    self._trace("prune_unrequested_budget_exceeded", ops=ops_count, max_ops=max_ops)
                    budget_exceeded = True
                    break
                bridge_pruned = False
                trial_key = None
                c = current[idx]
                c_role = str(getattr(c, "node_role", "") or getattr(c, "role", "") or "").lower()
                c_tags = {str(t).lower() for t in (getattr(c, "semantic_tags", None) or [])}
                is_exempt = (
                    is_ingress_source(c, getattr(self, "orchestrator", None))
                    or is_terminal_sink(c, getattr(self, "orchestrator", None))
                    or c_role in est_roles
                )
                is_candidate_bridge = not is_exempt and (
                    c_role in bridge_roles
                    or getattr(c, "node_type", "") in bridge_roles
                    or bool(bridge_roles & c_tags)
                    or (vocab is not None and hasattr(vocab, "is_bridge") and vocab.is_bridge(c))
                )
                if is_candidate_bridge:
                    trial = [x.clone() if hasattr(x, "clone") else x for x in (current[:idx] + current[idx + 1:])]
                    trial_key = tuple(x.cell_id for x in trial)
                    if trial_key in pass2_failed_removals:
                        idx += 1
                        continue
                    for x in trial[idx:]:
                        x.planned_port_wires = {}
                        x.bound_parent_ids = set()
                    try:
                        from unification import UnificationGate, ExecutionContext
                        from preflight import PreflightLinter
                        gate = UnificationGate(orchestrator=self.orchestrator)
                        probe_ctx = ExecutionContext(prompt=prompt)
                        probe_trial = self.insert_property_bridges(trial, prompt=prompt, ctx=probe_ctx, trace=False)
                        u_res = gate.unify_pipeline(probe_trial, probe_ctx, extract_llm_slots=False)
                        if not u_res.is_bottom():
                            trial_sig = tuple(x.cell_id for x in probe_trial)
                            if trial_sig in lint_cache:
                                is_valid = lint_cache[trial_sig]
                            else:
                                emitted = gate.emit_code(probe_trial, probe_ctx, extract_llm_slots=False)
                                lint_res = PreflightLinter.lint(gate.last_pipeline_bindings, prompt=prompt, code_str=emitted, orchestrator=self.orchestrator)
                                is_valid = lint_res.is_valid
                                lint_cache[trial_sig] = is_valid
                            if is_valid:
                                self._trace("pruned_redundant_bridge", cell=c.cell_id, reason="pipeline unifies and passes preflight without explicit bridge")
                                current = probe_trial
                                changed = True
                                idx = 1
                                bridge_pruned = True
                                continue
                    except Exception:
                        pass
                    if not bridge_pruned and trial_key is not None:
                        pass2_failed_removals.add(trial_key)
                idx += 1
            if budget_exceeded:
                break
        try:
            from dag_plan import OpenPortFrontier
            active_ids = {c.cell_id for c in current}
            for c in current:
                pw = getattr(c, "planned_port_wires", None)
                if isinstance(pw, dict):
                    c.planned_port_wires = {
                        k: v for k, v in pw.items()
                        if getattr(v[0] if isinstance(v, tuple) else v, "cell_id", str(v[0] if isinstance(v, tuple) else v)) in active_ids
                    }
                bp = getattr(c, "bound_parent_ids", None)
                if isinstance(bp, set):
                    c.bound_parent_ids = bp & active_ids
            return OpenPortFrontier.build_dag_plan(current)
        except Exception:
            return current

    # ------------------------------------------------------------------
    # Shared evidence primitives (identical for every route method)
    # ------------------------------------------------------------------
    def _evidence(self) -> Any:
        from token_evidence import get_evidence
        return get_evidence(self.orchestrator)

    def clause_fit(self, clause_text: str, cell: Cell) -> Tuple[float, float, float]:
        """Identity evidence rank (recall, name-precision, F) of `cell` for a prompt clause; zeros if none."""
        ev = self._evidence()
        if ev is None or not clause_text:
            return (0.0, 0.0, 0.0)
        toks = ev.clause_tokens(clause_text)
        if not toks:
            return (0.0, 0.0, 0.0)
        o = ev.identity_overlap(toks, cell)
        return o["rank"] if o["hit"] > 0.0 else (0.0, 0.0, 0.0)

    @staticmethod
    def _is_path_consumer(cell: Cell) -> bool:
        """True if the cell reads/writes a filesystem path (tree-declared roles / abstract types)."""
        try:
            from lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
            roles = {str(r).lower() for r in reg.get_verification_semantics("path_consumer_roles")}
            abstracts = {str(r).lower() for r in reg.get_verification_semantics("path_abstract_types")}
        except Exception:
            roles, abstracts = set(), set()
        for p in cell.inputs.values():
            sig = getattr(p, "signature", p)
            if str(getattr(p, "abstract_type", "") or "").lower() in abstracts or str(getattr(sig, "abstract_type", "") or "").lower() in abstracts:
                return True
            if str(getattr(p, "port_role", "") or "").lower() in roles or str(getattr(p, "derived_role", "") or "").lower() in roles:
                return True
        return False

    def file_compat(self, cell: Cell, literal: Any) -> int:
        """
        +2 explicit declared format match (supported_formats or File[..., ext]),
        +1 wildcard match,
        -1 incompatible,
        0 not applicable (no literal / not a path consumer).
        """
        if literal is None or not self._is_path_consumer(cell):
            return 0
        file_path_str = str(literal)
        if not file_path_str or "." not in file_path_str:
            return 0
        ext = file_path_str.rpartition(".")[2].strip().lower()
        if not ext:
            return 0
        if not self.is_file_format_compatible(cell, file_path_str):
            return -1

        path_ports = [
            p for p in list(cell.inputs.values()) + list(cell.outputs.values())
            if is_path_port(p)
        ]
        reg = TypeRegistry.get_instance()
        fmt_groups = reg.get_format_groups()
        loc_ctors = set(reg.get_verification_semantics("locator_constructors") or ())
        for p in path_ports:
            tn = str(getattr(p, "type_name", "") or "").strip()
            has_concrete_fmt = False
            if "[" in tn and tn.endswith("]"):
                ctor, _, inner = tn[:-1].partition("[")
                if ctor.strip().lower() in loc_ctors:
                    parts = [part.strip().lower() for part in inner.split(",") if part.strip()]
                    if len(parts) >= 2:
                        fmt = parts[1]
                        if fmt and fmt != "generic" and not reg.is_declared_top(fmt):
                            has_concrete_fmt = fmt not in fmt_groups
                            if (
                                ext == fmt
                                or (len(ext) >= 2 and fmt.startswith(ext))
                                or reg.is_subtype(ext, fmt)
                                or (has_concrete_fmt and reg.is_subtype(fmt, ext))
                            ):
                                return 2
                    elif len(parts) == 1:
                        target = parts[0]
                        if target and target != "generic" and not reg.is_declared_top(target):
                            has_concrete_fmt = target not in fmt_groups
                            if (
                                ext == target
                                or (len(ext) >= 2 and target.startswith(ext))
                                or reg.is_subtype(ext, target)
                            ):
                                return 2
            if not has_concrete_fmt:
                port_fmts = getattr(p, "supported_formats", None)
                if port_fmts:
                    clean_fmts = {str(f).strip().lower() for f in port_fmts}
                    if ext in clean_fmts:
                        return 2
        return 1

    def requested_sink(self, prompt: str, candidates: List[Cell]) -> Optional[Cell]:
        """
        A sink (egress/terminal) cell is requested only if some clause is better explained by a sink
        than by ANY non-sink candidate. Replaces the "prompt contains the word write" egress guess,
        which turned "write it to a new column" into a file sink.
        """
        ev = self._evidence()
        if ev is None:
            return None
        from unification import _is_sink_cell
        _, dest_lits = self.extract_file_literals(prompt)
        sinks = [c for c in candidates if _is_sink_cell(c)]
        if not dest_lits:
            sinks = [s for s in sinks if not self._is_path_consumer(s)]
        others = [c for c in candidates if not _is_sink_cell(c)]
        best, best_rank = None, (0.0, 0.0, 0.0)
        for cl in self.segment_prompt_clauses(prompt):
            toks = ev.clause_tokens(cl)
            if not toks:
                continue
            top_other = max((ev.identity_overlap(toks, c)["rank"] for c in others), default=(0.0, 0.0, 0.0))
            for s in sinks:
                o = ev.identity_overlap(toks, s)
                if o["hit"] > 0.0 and o["rank"] > top_other and o["rank"] > best_rank:
                    best, best_rank = s, o["rank"]
        return best

    def select_stratified_candidates(
        self,
        prompt: str,
        candidates: List[Cell],
        relevance_map: Dict[str, float],
        orchestrator: Optional[LatticeOrchestrator] = None,
        max_shown: int = 40,
    ) -> List[Cell]:
        """
        Chronological clause-stratified candidate selection:
        Guarantees that every procedural clause of a compound prompt contributes its
        most precise matching operations, plus carrier bridges, preventing clause starvation
        and keeping the candidate list clean and focused for local LLM reasoning.
        """
        self._current_prompt = prompt
        orch = orchestrator or self.orchestrator
        clauses = self.segment_prompt_clauses(prompt)
        prompt_lower = prompt.lower()

        selected_cells: List[Cell] = []
        selected_ids: Set[str] = set()

        filtered_cands: List[Cell] = []
        for c in candidates:
            cid = c.cell_id
            if getattr(c, "node_type", "") == "constant":
                continue
            if not getattr(c, "slots", None):
                if is_macro_cell(c):
                    continue
                if getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator":
                    continue
            filtered_cands.append(c)

        try:
            from token_evidence import get_evidence, anchors_for_clause
        except (ImportError, ValueError):
            from ..token_evidence import get_evidence, anchors_for_clause

        ev = get_evidence(orch)

        primary_domains: Set[str] = set()
        primary_cells: List[Cell] = []
        for idx, cl in enumerate(clauses):
            if idx == 0 and len(clauses) > 1:
                stage_pool = [c for c in filtered_cands if self.is_ingress_node(c)] or filtered_cands
            else:
                stage_pool = [c for c in filtered_cands if not self.is_ingress_node(c)] or filtered_cands

            if ev is not None:
                anchors = anchors_for_clause(ev, cl, stage_pool, relevance_map, clause_idx=idx)
            else:
                anchors = []

            if not anchors:
                continue

            primary_c, _ = anchors[0]
            primary_cells.append(primary_c)
            p_dom = getattr(primary_c, "domain_name", "")
            if p_dom:
                primary_domains.add(p_dom)

        for primary_c in primary_cells:
            if primary_c.cell_id not in selected_ids and len(selected_cells) < max_shown:
                selected_cells.append(primary_c)
                selected_ids.add(primary_c.cell_id)
            try:
                prod = self._find_type_producer(primary_c, selected_cells, filtered_cands, clauses, orch)
                if prod is not None and prod.cell_id not in selected_ids and len(selected_cells) < max_shown:
                    selected_cells.append(prod)
                    selected_ids.add(prod.cell_id)
            except Exception:
                pass

        # Secondary matching candidates in the primary workflow domains
        if len(selected_cells) < max_shown and ev is not None:
            for idx, cl in enumerate(clauses):
                if len(selected_cells) >= max_shown:
                    break
                stage_pool = [c for c in filtered_cands if not self.is_ingress_node(c)] or filtered_cands
                cl_tokens = ev.clause_tokens(cl)
                for c in stage_pool:
                    if len(selected_cells) >= max_shown:
                        break
                    if c.cell_id in selected_ids:
                        continue
                    c_dom = getattr(c, "domain_name", "")
                    if primary_domains and c_dom and c_dom not in primary_domains:
                        continue
                    o = ev.identity_overlap(cl_tokens, c)
                    if o.get("hit", 0.0) > 0.0 and o.get("f", 0.0) >= 0.25:
                        selected_cells.append(c)
                        selected_ids.add(c.cell_id)

        # Carrier bridges in active workflow domains
        if len(selected_cells) < max_shown:
            candidate_bridges = [
                c for c in filtered_cands
                if (getattr(c, "node_role", "") == "bridge" or getattr(c, "node_type", "") == "tunnel")
                and c.cell_id not in selected_ids
                and (not getattr(c, "domain_name", "") or not primary_domains or getattr(c, "domain_name", "") in primary_domains)
            ]
            candidate_bridges.sort(key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)
            for c in candidate_bridges:
                if len(selected_cells) >= max_shown:
                    break
                selected_cells.append(c)
                selected_ids.add(c.cell_id)

        # Top globally relevant cells fill up only if sparse
        if len(selected_cells) < max_shown:
            sorted_by_rel = sorted(filtered_cands, key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)
            for c in sorted_by_rel:
                if len(selected_cells) >= max_shown:
                    break
                if c.cell_id not in selected_ids:
                    c_dom = getattr(c, "domain_name", "")
                    if primary_domains and c_dom and c_dom not in primary_domains:
                        continue
                    selected_cells.append(c)
                    selected_ids.add(c.cell_id)

        return selected_cells

