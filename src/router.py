"""
src/router.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Semantic Tunneling Router based on Dense Vector Embeddings and Softmax Temperature.

Conforms strictly to Section 3.3 of the STRATA paper:
  e_x = Embed(prompt)
  P(C_k | e_x) = softmax_k( cos(e_x, mu_k) / gamma )
  Tunnel T = { v in V | P(v | e_x) >= epsilon }

Contains ZERO keyword-sniffing regexes, ZERO manual score boosts, and ZERO domain hardcodes.

Tunables previously baked into this module as magic literals now resolve through
`config.settings` when present (see `_cfg`). In-module fallbacks keep the file
runnable standalone; production profiles should override via settings.
"""

from __future__ import annotations
import ast
import os
import json
import time
import math
from typing import Optional, List, Dict, Set, Tuple, Any, Union
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from log_config import get_logger

try:
    from .lattice import LatticeOrchestrator, Cell, MacroCell, TypeRegistry
    from .internal_rag import LocalRAG
    from .planner import LatticePlanner, _segment_prompt_clauses
    from .tokenizer import CellTokenizer
    from .inference import ModelManager
    from .route_methods import get_route_method, RouteMethod
    try:
        from .reranker import LocalReranker
    except (ImportError, ValueError):
        from reranker import LocalReranker
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell, MacroCell, TypeRegistry
    from internal_rag import LocalRAG
    from planner import LatticePlanner, _segment_prompt_clauses
    from tokenizer import CellTokenizer
    from inference import ModelManager
    from route_methods import get_route_method, RouteMethod
    try:
        from reranker import LocalReranker
    except ImportError:
        LocalReranker = None

logger = get_logger('router')


@dataclass(frozen=True)
class PlanResult:
    """Immutable result of a routing/planning pass, avoiding global mutable state on LatticeRouter."""
    cells: List[Cell] = field(default_factory=list)
    candidate_ids: Set[str] = field(default_factory=set)
    requested_route: str = "M0"
    effective_route: str = "M0"
    fallback_reason: Optional[str] = None
    route_trace: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    ir_literals: Dict[str, Any] = field(default_factory=dict)
    tagging_decisions: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    def __iter__(self):
        return iter(self.cells)

    def __len__(self):
        return len(self.cells)

    def __getitem__(self, idx):
        return self.cells[idx]

    def __bool__(self):
        return bool(self.cells)


# ---------------------------------------------------------------------------
# Settings access with in-module fallbacks
# ---------------------------------------------------------------------------
# Everything previously hardcoded in this file (span caps, delimiter sets,
# stratified-pruning caps, macro-promotion scaling, dense-RAG budget) is now
# resolved here. If `config.settings` is unavailable or lacks a given key, the
# documented fallback is used so the router stays runnable in isolation.

def _load_settings():
    try:
        from .config import settings as _s
        return _s
    except (ImportError, ValueError):
        try:
            from config import settings as _s
            return _s
        except Exception:
            return None


_SETTINGS = _load_settings()


def _cfg(name: str, default):
    if _SETTINGS is not None:
        return getattr(_SETTINGS, name, default)
    return default


# --- Query span generation ------------------------------------------------
# Bound on the number of query spans emitted per routing call. This directly
# caps dense-RAG fan-out (one embed + FAISS lookup per span).
MAX_QUERY_SPANS: int = int(_cfg("max_query_spans", 12))
# Prompts with <= this many tokens skip sliding-window generation: clause-level
# decomposition already covers their structure.
SHORT_PROMPT_WINDOW_SKIP: int = int(_cfg("short_prompt_window_skip", 6))

# --- Stratified pruning ---------------------------------------------------
# Caps scale with the number of semantic clauses (n) discovered from the IR
# or clause segmenter. Asymmetry across stages is intentional: stage 2 is the
# transform band where intermediates live and where recall matters most.
PRUNE_S1_MIN: int = int(_cfg("prune_s1_min", 12))
PRUNE_S1_PER_CLAUSE: int = int(_cfg("prune_s1_per_clause", 2))
PRUNE_S2_MIN: int = int(_cfg("prune_s2_min", 38))
PRUNE_S2_PER_CLAUSE: int = int(_cfg("prune_s2_per_clause", 6))
PRUNE_S3_MIN: int = int(_cfg("prune_s3_min", 8))
PRUNE_S3_PER_CLAUSE: int = int(_cfg("prune_s3_per_clause", 2))

# --- Macro participation (Section 3.5) -------------------------------------
# IDF-weighted concept coverage gate: a macro must own at least this fraction
# of the prompt's semantic mass before it can enter the tunnel. Its relevance
# is its measured coverage — no priority inflation, no constituent boosting.
MACRO_CONCEPT_COVERAGE_GATE: float = float(_cfg("macro_concept_coverage_gate", 0.35))

# --- Dense-RAG fusion -------------------------------------------------------
# The dense pass ALWAYS runs when an indexed RAG is available (Profile A/C/D/E
# architecture contract): lexical stage-1 evidence and dense embedding
# evidence are fused, never short-circuited by a lexical heuristic. There is
# intentionally NO lexical-strength bypass here — such a gate silently
# degraded Profile A/C/D benchmarks to Profile 0 behavior on prompts with
# high lexical overlap.
# Per-span top_k bounds for the dense blend.
DENSE_RAG_MIN_PER_SPAN: int = int(_cfg("dense_rag_min_per_span", 15))
DENSE_RAG_MAX_PER_SPAN: int = int(_cfg("dense_rag_max_per_span", 30))


# --- Clause delimiter fallback -------------------------------------------
# Sub-clause split points are DECLARED language vocabulary: the union of the
# registry's sentence connectives and relational preposition triggers, both
# harvested from the loaded domain trees. The engine carries no language
# literals of its own; with no trees loaded the delimiter set is empty and
# only punctuation-based splitting remains.
def _get_clause_delims() -> Set[str]:
    delims = getattr(CellTokenizer, "CLAUSE_DELIMITERS", None)
    if delims:
        return set(delims)
    try:
        reg = TypeRegistry.get_instance()
        return set(reg.get_sentence_connectives()) | set(reg.get_preposition_triggers())
    except Exception:
        return set()


def _extract_id(item):
    if hasattr(item, "cell_id"):
        return item.cell_id
    if isinstance(item, (tuple, list)):
        for x in item:
            if hasattr(x, "cell_id"):
                return x.cell_id
    return str(item)


class IdiomSubgraph:
    """
    Candidate workflow idiom / subgraph in the lattice, discovered via Stage 1 retrieval.
    Represents an empirical pipeline path or cluster connecting prompt intent clauses.
    """
    def __init__(
        self,
        subgraph_id: str,
        cells: List[Cell],
        edges: List[Tuple[str, str, float]],
        covered_clauses: Set[int],
        lexical_score: float = 0.0,
        affinity_score: float = 0.0,
        entry_node: Optional[Cell] = None
    ):
        self.subgraph_id = subgraph_id
        self.cells = list(cells)
        self.cell_ids = {c.cell_id for c in cells}
        self.edges = list(edges)
        self.covered_clauses = set(covered_clauses)
        self.lexical_score = lexical_score
        self.affinity_score = affinity_score
        self.entry_node = entry_node

    @property
    def total_score(self) -> float:
        # Multi-clause coverage bonus + lexical relevance + transition affinities
        w_cov = _cfg("idiom_coverage_weight", 3.0)
        w_aff = _cfg("idiom_affinity_weight", 1.2)
        coverage_bonus = w_cov * len(self.covered_clauses)
        return self.lexical_score + w_aff * self.affinity_score + coverage_bonus

    def __repr__(self) -> str:
        entry_id = self.entry_node.cell_id if self.entry_node else "None"
        return f"<IdiomSubgraph {self.subgraph_id}: {len(self.cells)} cells, entry={entry_id}, score={self.total_score:.2f}>"


LEN_NORM_TABLE = [1.0 / (math.log2(2 + i) ** 0.1) for i in range(500)]


class LatticeRouter:
    """
    Semantic Tunneling Router (Section 3.3).
    Implements Phase 4 Two-Stage Retrieval:
      - Stage 1: Discovers candidate idioms/subgraphs using the lexical/IDF scorer
                 combined with empirical graph edges.
      - Stage 2: Identifies the entry node within the candidate idiom.
    """
    def __init__(
        self,
        orchestrator: LatticeOrchestrator,
        internal_rag: Optional[LocalRAG] = None,
        gamma: float = 0.15,
        epsilon: float = 0.001,
        rag_engine: Optional[LocalRAG] = None,
        **kwargs
    ):
        self.orchestrator = orchestrator
        self.internal_rag = internal_rag or rag_engine
        self.gamma = gamma          # Temperature parameter scaling cosine similarity
        self.epsilon = epsilon      # Cutoff threshold for tunnel inclusion
        # Accept both the current kwarg name (`default_route_method`, used by
        # every call site) and the legacy name (`route_method`) that this
        # line used to read exclusively -- the mismatch meant a caller's
        # explicit default_route_method=... was silently dropped into
        # **kwargs and never read, always falling back to "m0" regardless of
        # what was requested. An explicit method still wins; when neither is
        # given, prefer an LLM-driven default (m4) over the pure-symbolic
        # trellis (m0) whenever the active model profile actually has a
        # usable generation model -- there is no reason to fall back to
        # regex/heuristic-only routing when a model capable of reading the
        # prompt is already loaded. Profiles that intentionally have no
        # usable LLM (0, A) or that intentionally disable synthesis for
        # ablation purposes (e.g. a routing-only benchmark profile) report
        # can_synthesize() == False and are unaffected.
        _explicit_method = kwargs.get("default_route_method", kwargs.get("route_method"))
        if _explicit_method:
            self.default_route_method = str(_explicit_method).strip().lower()
        else:
            try:
                _mm_rt = ModelManager.get_instance()
                if getattr(_mm_rt, "has_coalgebraic_unfolder", lambda: False)():
                    self.default_route_method = "omega"
                elif _mm_rt.can_synthesize():
                    self.default_route_method = "m4"
                else:
                    self.default_route_method = "m0"
            except Exception:
                self.default_route_method = "m0"
        # Macro-goal routing (Section 3.5). Defaults to the global settings
        # value; an explicit kwarg wins. Toggling only affects routers created
        # afterwards unless mutated directly (see CLI `set macros on|off`).
        macros_val = kwargs.get("macros_enabled")
        if macros_val is None:
            macros_val = _cfg("macros_enabled", True)
        self._macros_enabled = bool(macros_val)

        topo_val = kwargs.get("topology_mode")
        if topo_val is None:
            topo_val = _cfg("topology_mode", "frontier")
        self._topology_mode = str(topo_val).lower()
        self.planner = LatticePlanner(
            orchestrator=self.orchestrator,
            rag=self.internal_rag,
            macros_enabled=self._macros_enabled,
            topology_mode=self._topology_mode
        )
        # (apply_fixes_v7) IR compiler hook — auto-enabled when active profile provides has_ir_compiler() (e.g. Profile S)
        is_p0 = False
        has_ir = False
        try:
            mm_ir = ModelManager.get_instance()
            is_p0 = (getattr(mm_ir, "current_profile_name", None) == "0")
            has_ir = bool(mm_ir.has_ir_compiler())
        except Exception:
            is_p0 = False
            has_ir = False
        self._use_ir_compiler = False if is_p0 else (bool(_cfg("use_ir_compiler", False)) or has_ir)
        self._ir_compiler = None
        if self._use_ir_compiler:
            try:
                from ir_compiler import IRCompiler
                self._ir_compiler = IRCompiler(self.orchestrator)
            except Exception as _e:
                try:
                    from .ir_compiler import IRCompiler
                    self._ir_compiler = IRCompiler(self.orchestrator)
                except Exception as _e2:
                    logger.warning("IR compiler unavailable: %s", _e2)
        self._last_ir_steps = 0
        self.priority_map: Dict[str, float] = {}

        # Neural Reranker configuration:
        # Profile A enables cross-encoder/listwise reranking by default (`uses_reranker() == True`).
        # Profile 0 and Profile B never enable the reranker.
        mm_inst = None
        profile_name_now = None
        profile_uses_reranker = False
        try:
            mm_inst = ModelManager.get_instance()
            profile_name_now = getattr(mm_inst, "current_profile_name", None)
            profile_uses_reranker = bool(mm_inst.uses_reranker()) if mm_inst else False
        except Exception:
            pass
        reranker_val = kwargs.get("use_reranker")
        if is_p0 or profile_name_now == "B":
            self.use_reranker = False
        elif reranker_val is not None:
            self.use_reranker = bool(reranker_val)
        else:
            self.use_reranker = bool(_cfg("use_reranker", False)) or profile_uses_reranker
        self.reranker_model = kwargs.get("reranker_model") or _cfg("reranker_model", None)
        self.reranker = None if (is_p0 or profile_name_now == "B") else kwargs.get("reranker")
        if self.use_reranker and self.reranker is None and LocalReranker is not None:
            try:
                self.reranker = LocalReranker(model_name_or_path=self.reranker_model)
            except Exception as _e:
                logger.warning(f"[ROUTER] Could not initialize LocalReranker: {_e}")
                self.reranker = None
        self.raw_rag_tunnel: List[Cell] = []
        self.raw_rag_relevance_map: Dict[str, float] = {}
        self.last_reranker_telemetry: List[Dict[str, Any]] = []
        self.last_clause_semantic_scores: List[Dict[str, float]] = []
        self._route_cache: Dict[Any, Any] = {}
        self.enable_route_cache: bool = True

    @staticmethod
    def _is_profile_zero() -> bool:
        try:
            return getattr(ModelManager.get_instance(), "current_profile_name", None) == "0"
        except Exception:
            return False

    def clear_route_cache(self) -> None:
        """Clears memoized Layer 1 semantic routing results."""
        if hasattr(self, "_route_cache"):
            self._route_cache.clear()

    @property
    def macros_enabled(self) -> bool:
        return self._macros_enabled

    @macros_enabled.setter
    def macros_enabled(self, val: bool):
        self._macros_enabled = bool(val)
        if hasattr(self, "planner") and self.planner is not None:
            self.planner.macros_enabled = bool(val)

    @property
    def topology_mode(self) -> str:
        return getattr(self.planner, "topology_mode", getattr(self, "_topology_mode", "frontier"))

    @topology_mode.setter
    def topology_mode(self, val: str):
        self._topology_mode = str(val).lower()
        if hasattr(self, "planner") and self.planner is not None:
            self.planner.topology_mode = self._topology_mode

    def _score_clause_lexical_idf(
        self,
        text: str,
        token_index: Optional[Dict[str, List[Cell]]],
        N: int
    ) -> Dict[Cell, float]:
        """Computes length-normalized, IDF-weighted lexical scores for a query clause."""
        q_tokens = CellTokenizer.tokenize_prompt(text)
        eval_tokens = q_tokens
        q_len = max(len(text.split()), len(eval_tokens), 1)
        inv_q_len = 1.0 / q_len
        clause_scores: Dict[Cell, float] = {}

        if self._is_profile_zero():
            # Strict Profile 0: direct formal set membership against {cell_id} | canonical_op_names ONLY.
            # Zero semantic tags, zero keywords, zero docstring matching, zero zero-IDF stripping.
            raw_set = {w.strip().lower() for w in text.replace("(", " ").replace(")", " ").replace(",", " ").split() if w.strip()} | eval_tokens
            denom = max(len(raw_set), q_len, 1)
            for cell in self.orchestrator.loaded_cells.values():
                f_ids = getattr(cell, "formal_identifiers", None) or {cell.cell_id.lower()}
                hits = raw_set & f_ids
                if hits:
                    clause_scores[cell] = float(len(hits)) / float(denom)
            return clause_scores

        if not token_index or N == 0:
            for cell in self.orchestrator.loaded_cells.values():
                cell_tokens = cell.token_set
                intersection_len = len(eval_tokens & cell_tokens)
                if intersection_len > 0:
                    c_len = max(len(cell_tokens), 1)
                    sc = (intersection_len * inv_q_len) * LEN_NORM_TABLE[min(c_len, 499)]
                    clause_scores[cell] = float(sc)
            return clause_scores

        def _idf(tok: str) -> float:
            df = len(token_index.get(tok, ()))
            return math.log(1.0 + (N + 1) / (df + 1.0))

        match_weights: Dict[Cell, float] = {}
        cells_with_identity: Set[Cell] = set()
        idf_ratio = float(_cfg("lexical_idf_max_freq_ratio", 0.3))
        idf_body_w = float(_cfg("lexical_idf_body_weight", 0.3))
        q_idf_mass = 0.0
        for tok in eval_tokens:
            cells = token_index.get(tok, [])
            w = _idf(tok)
            q_idf_mass += w
            if N < 20 or len(cells) < idf_ratio * N:
                for cell in cells:
                    if tok in getattr(cell, "identity_tokens", ()):
                        cells_with_identity.add(cell)
                        match_weights[cell] = match_weights.get(cell, 0.0) + w
                    else:
                        match_weights[cell] = match_weights.get(cell, 0.0) + idf_body_w * w

        q_idf_denom = q_idf_mass if q_idf_mass > 0.0 else float(max(q_len, 1))
        for cell, weight_sum in match_weights.items():
            if cells_with_identity and cell not in cells_with_identity:
                continue
            c_len = max(getattr(cell, "token_count", len(cell.token_set)), 1)
            sc = (weight_sum / q_idf_denom) * LEN_NORM_TABLE[min(c_len, 499)]
            clause_scores[cell] = min(1.0, float(sc))

        return clause_scores

    def _discover_candidate_idioms(
        self,
        prompt: str,
        clauses: List[str],
        max_idioms: int = 10
    ) -> List[IdiomSubgraph]:
        """
        Stage 1 Retrieval:
        Discovers candidate idioms/subgraphs in the lattice using:
          1. Clause-level lexical/IDF token matching across prompt intents.
          2. Expansion along empirical AST-mined directed edges and typestate transitions.
          3. Aggregation of path transition affinities and multi-clause coverage.
        """
        token_index = getattr(self.orchestrator, "token_index", None)
        N = len(self.orchestrator.loaded_cells)
        if N == 0:
            return []

        # Step 1.1: Lexical scoring per clause
        clause_matches: List[List[Tuple[Cell, float]]] = []
        for cl in clauses:
            scores = self._score_clause_lexical_idf(cl, token_index, N)
            top_clause = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:8]
            clause_matches.append(top_clause)

        # Step 1.2: Graph Edge Expansion along directed transitions
        idiom_candidates: List[IdiomSubgraph] = []

        if len(clauses) == 1 or not clause_matches:
            top_seeds = clause_matches[0] if clause_matches else []
            for seed_cell, s_score in top_seeds[:5]:
                cluster_cells = [seed_cell]
                cluster_edges: List[Tuple[str, str, float]] = []
                for e in getattr(seed_cell, "edges", [])[:6]:
                    tgt_id = e.get("target_cell_id") if isinstance(e, dict) else getattr(e, "target_cell_id", "")
                    tgt_cell = self.orchestrator.loaded_cells.get(tgt_id)
                    if tgt_cell and tgt_cell not in cluster_cells:
                        cluster_cells.append(tgt_cell)
                        aff = e.get("affinity_score", 0.5) if isinstance(e, dict) else getattr(e, "affinity_score", 0.5)
                        cluster_edges.append((seed_cell.cell_id, tgt_id, aff))

                idiom = IdiomSubgraph(
                    subgraph_id=f"idiom_{seed_cell.cell_id}",
                    cells=cluster_cells,
                    edges=cluster_edges,
                    covered_clauses={0},
                    lexical_score=s_score,
                    affinity_score=sum(aff for _, _, aff in cluster_edges)
                )
                idiom_candidates.append(idiom)
        else:
            # Multi-clause sequential pipeline chaining
            for s0, sc0 in clause_matches[0][:4]:
                chain_cells: List[Cell] = [s0]
                chain_edges: List[Tuple[str, str, float]] = []
                covered: Set[int] = {0}
                total_lex = sc0
                curr_node = s0

                for c_idx in range(1, len(clauses)):
                    next_seeds = clause_matches[c_idx]
                    found_next = False
                    # 1. Direct edge check across DAG history in chain_cells
                    for anc in reversed(chain_cells):
                        anc_out = [e.get("target_cell_id") for e in getattr(anc, "edges", [])]
                        anc_adj = self.orchestrator._adjacency.get(anc.cell_id, [])
                        for cand, cand_sc in next_seeds[:6]:
                            if cand.cell_id in anc_out or cand.cell_id in anc_adj:
                                edge_rec = next((e for e in getattr(anc, "edges", []) if e.get("target_cell_id") == cand.cell_id), None)
                                aff = edge_rec.get("affinity_score", 0.5) if edge_rec else 0.5
                                chain_edges.append((anc.cell_id, cand.cell_id, aff))
                                if cand not in chain_cells:
                                    chain_cells.append(cand)
                                covered.add(c_idx)
                                total_lex += cand_sc
                                curr_node = cand
                                found_next = True
                                break
                        if found_next:
                            break

                    # 2. 1-hop bridge transition check across DAG history
                    if not found_next:
                        for anc in reversed(chain_cells):
                            anc_out = [e.get("target_cell_id") for e in getattr(anc, "edges", [])]
                            anc_adj = self.orchestrator._adjacency.get(anc.cell_id, [])
                            mid_candidates = anc_out[:6] if anc_out else anc_adj[:6]
                            for mid_id in mid_candidates:
                                mid_cell = self.orchestrator.loaded_cells.get(mid_id)
                                if not mid_cell:
                                    continue
                                mid_out = [e.get("target_cell_id") for e in getattr(mid_cell, "edges", [])]
                                mid_adj = self.orchestrator._adjacency.get(mid_id, [])
                                for cand, cand_sc in next_seeds[:6]:
                                    if cand.cell_id in mid_out or cand.cell_id in mid_adj:
                                        e1 = next((e for e in getattr(anc, "edges", []) if e.get("target_cell_id") == mid_id), None)
                                        aff1 = e1.get("affinity_score", 0.5) if e1 else 0.5
                                        e2 = next((e for e in getattr(mid_cell, "edges", []) if e.get("target_cell_id") == cand.cell_id), None)
                                        aff2 = e2.get("affinity_score", 0.5) if e2 else 0.5

                                        chain_edges.append((anc.cell_id, mid_id, aff1))
                                        chain_edges.append((mid_id, cand.cell_id, aff2))
                                        if mid_cell not in chain_cells:
                                            chain_cells.append(mid_cell)
                                        if cand not in chain_cells:
                                            chain_cells.append(cand)
                                        covered.add(c_idx)
                                        total_lex += cand_sc
                                        curr_node = cand
                                        found_next = True
                                        break
                                if found_next:
                                    break
                            if found_next:
                                break

                    # 3. Soft anchor inclusion if path not fully bridged
                    if not found_next and next_seeds:
                        best_cand, best_sc = next_seeds[0]
                        if best_cand not in chain_cells:
                            chain_cells.append(best_cand)
                        covered.add(c_idx)
                        total_lex += best_sc
                        curr_node = best_cand

                idiom = IdiomSubgraph(
                    subgraph_id=f"pipeline_{s0.cell_id}",
                    cells=chain_cells,
                    edges=chain_edges,
                    covered_clauses=covered,
                    lexical_score=total_lex,
                    affinity_score=sum(aff for _, _, aff in chain_edges)
                )
                idiom_candidates.append(idiom)

        idiom_candidates.sort(key=lambda x: x.total_score, reverse=True)
        return idiom_candidates[:max_idioms]

    def _select_entry_node(
        self,
        idiom: IdiomSubgraph,
        first_clause: str,
        token_index: Optional[Dict[str, List[Cell]]],
        N: int
    ) -> Optional[Cell]:
        """
        Stage 2 Retrieval:
        Pinpoints the entry node within the candidate idiom:
          1. Subgraph in-degree: nodes with 0 incoming edges within the idiom.
          2. Stage classification: Stage 1 source/ingestion nodes (e.g. read_csv, imread).
          3. Alignment with initial prompt intent / first clause.
        """
        if not idiom.cells:
            return None

        # Compute internal in-degrees within this idiom
        internal_in_deg: Dict[str, int] = defaultdict(int)
        for src, tgt, _ in idiom.edges:
            if src in idiom.cell_ids and tgt in idiom.cell_ids:
                internal_in_deg[tgt] += 1

        c1_scores = self._score_clause_lexical_idf(first_clause, token_index, N) if first_clause else {}

        best_cell = idiom.cells[0]
        best_score = -1e9

        w_c1 = float(_cfg("entry_c1_score_weight", 1.5))
        w_root = float(_cfg("entry_root_bonus", 2.0))
        w_ingress = float(_cfg("entry_ingress_bonus", 2.0))
        w_source = float(_cfg("entry_source_bonus", 1.0))
        w_no_req = float(_cfg("entry_no_req_inputs_bonus", 1.0))
        try:
            from lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
            ingress_stage = reg.get_ingress_stage()
            ingress_roles = reg.get_ingress_roles_set()
        except Exception:
            ingress_stage = 1
            ingress_roles = set()

        for cell in idiom.cells:
            cid = cell.cell_id
            c1_sc = c1_scores.get(cell, 0.0)
            is_root = 1.0 if internal_in_deg[cid] == 0 else 0.0
            is_ingress = 1.0 if getattr(cell, "stage", 2) == ingress_stage else 0.0
            is_source = 1.0 if str(getattr(cell, "node_role", "") or "").lower() in ingress_roles else 0.0

            # Nodes with no required input ports receive initiation bonus
            no_req_inputs = 1.0 if not any(getattr(p, "required", True) for p in getattr(cell, "inputs", {}).values()) else 0.0

            entry_score = (
                c1_sc * w_c1
                + w_root * is_root
                + w_ingress * is_ingress
                + w_source * is_source
                + w_no_req * no_req_inputs
            )
            if entry_score > best_score:
                best_score = entry_score
                best_cell = cell

        return best_cell


    def route(
        self,
        prompt: str,
        top_k: int = 150
    ) -> Tuple[List[Cell], Dict[str, float]]:
        """
        Two-Stage Retrieval Architecture (Phase 4):
          - Stage 1: Discovers candidate idioms/subgraphs using the lexical/IDF scorer
                     and graph edges (AST-mined and typestate transitions).
          - Stage 2: Identifies the entry node within the winning idiom(s).
        Returns:
          (tunnel_cells, cell_relevance_probabilities)
        """
        if not prompt or not prompt.strip() or len(self.orchestrator.loaded_cells) == 0:
            self.last_clause_semantic_scores = []
            if self.orchestrator is not None:
                self.orchestrator._last_clause_semantic_scores = None
            return [], {}

        self.last_clause_semantic_scores = []
        if self.orchestrator is not None:
            self.orchestrator._last_clause_semantic_scores = None
        is_p0 = self._is_profile_zero()
        if is_p0:
            prog_cells, _ = self._resolve_program_0(prompt)
            rel_p0 = {c.cell_id: 1.0 for c in prog_cells}
            return prog_cells, rel_p0
        cur_prof_name = ""
        try:
            cur_prof_name = str(getattr(ModelManager.get_instance(), "current_profile_name", "") or "")
        except Exception:
            pass

        cache_key = (
            prompt.strip(),
            int(top_k),
            bool(getattr(self, "use_reranker", False)) and not is_p0,
            str(getattr(self, "reranker_model", "") or "") if not is_p0 else "",
            id(getattr(self, "internal_rag", None)) if not is_p0 else 0,
            cur_prof_name,
            bool(getattr(self, "_use_ir_compiler", False)) and not is_p0,
        )
        if getattr(self, "enable_route_cache", True) and hasattr(self, "_route_cache") and cache_key in self._route_cache:
            entry = self._route_cache[cache_key]
            self.last_compiled_ir = entry.get("last_compiled_ir")
            self.last_ir_literals = dict(entry.get("last_ir_literals") or {})
            self._last_ir_steps = entry.get("last_ir_steps", 0)
            self.raw_rag_tunnel = list(entry.get("raw_rag_tunnel") or [])
            self.raw_rag_relevance_map = dict(entry.get("raw_rag_relevance_map") or {})
            self.last_reranker_telemetry = list(entry.get("last_reranker_telemetry") or [])
            self.last_clause_semantic_scores = [dict(d) for d in (entry.get("last_clause_semantic_scores") or [])]
            if self.orchestrator is not None:
                self.orchestrator._last_clause_semantic_scores = (
                    (prompt.strip(), [dict(d) for d in self.last_clause_semantic_scores])
                    if self.last_clause_semantic_scores
                    else None
                )
            for _cid, _s in entry.get("priority_map", {}).items():
                self.priority_map[_cid] = _s
            return list(entry["final_tunnel"]), dict(entry["relevance_map"])

        # IR compilation: when the loaded model successfully compiles the
        # prompt into a typed IR, that IR -- not a regex/punctuation split of
        # the raw text -- is the source of truth for the pipeline's semantic
        # clauses. The IR was produced by a model that actually read the
        # prompt. A successfully compiled IR with >=1 step supplies the
        # clauses directly (one per step); the embedding text itself is left
        # untouched. The regex/punctuation segmenter is kept ONLY as the
        # fallback for when no model is loaded or IR compilation did not
        # produce anything usable.
        #
        # Profile S (Semantic Compiler): the STRUCTURED IR is preserved, not
        # discarded. `self.last_compiled_ir` and `self.last_ir_literals` keep
        # the typed steps and extracted literals for downstream consumers:
        #   - the structured steps deterministically constrain the candidate
        #     tunnel (see the IR tunnel-constraint pass below), and
        #   - the literals are injected into ExecutionContext.parameters by
        #     the executors, letting the Unification Gate bind arguments
        #     deterministically instead of re-scanning the prompt.
        num_semantic_clauses = 0
        ir_clauses: List[str] = []
        self.last_compiled_ir = None
        self.last_ir_literals = {}
        if not is_p0 and getattr(self, "_use_ir_compiler", False) and self._ir_compiler is not None:
            try:
                _ir = self._ir_compiler.compile(prompt,)
                if _ir is not None and _ir.steps:
                    self.last_compiled_ir = _ir
                    self.last_ir_literals = dict(getattr(_ir, "literals", {}) or {})
                    all_ir_steps = _ir.all_steps() if hasattr(_ir, "all_steps") else _ir.steps
                    for s in all_ir_steps:
                        op = str(s.get("op", "")).strip()
                        lib = str(s.get("library", "")).strip()
                        params = s.get("params") or {}
                        param_words = " ".join(
                            str(v) for v in params.values()
                            if isinstance(v, (str, int, float)) and str(v).strip()
                        )
                        piece = " ".join(w for w in (op, lib, param_words) if w).strip()
                        if piece:
                            ir_clauses.append(piece)
                    num_semantic_clauses = len(all_ir_steps)
                    self._last_ir_steps = num_semantic_clauses
            except Exception as _e:
                logger.debug("[IR] compilation failed: %s", _e)

        token_index = getattr(self.orchestrator, "token_index", None)
        N = len(self.orchestrator.loaded_cells)

        # Decompose prompt into constituent clauses (sub-goals) from the authorative
        # user prompt. Model-compiled IR step fragments are included in search_texts
        # as auxiliary retrieval queries without displacing full user clause semantics.
        raw_clauses = _segment_prompt_clauses(prompt)
        clauses = raw_clauses
        if not num_semantic_clauses:
            num_semantic_clauses = len(clauses)
        primary_texts = [prompt.strip()]
        for c in clauses:
            cs = c.strip()
            if cs and cs not in primary_texts:
                primary_texts.append(cs)
        for ic in ir_clauses:
            if ic and ic not in primary_texts:
                primary_texts.append(ic)
        window_texts = [q for q in self._generate_query_spans(prompt.strip()) if q not in primary_texts]
        search_texts = primary_texts + window_texts
        first_clause = clauses[0] if clauses else prompt.strip()

        # Step 1: Stage 1 Idiom / Subgraph Discovery
        candidate_idioms = self._discover_candidate_idioms(prompt, clauses, max_idioms=10)

        # Step 2: Stage 2 Entry Node Selection
        for idiom in candidate_idioms:
            entry = self._select_entry_node(idiom, first_clause, token_index, N)
            idiom.entry_node = entry

        # Step 3: Candidate Group Assembly
        grouped_candidates: List[Dict[Cell, float]] = []
        primary_text_set = set(primary_texts)

        for text in search_texts:
            clause_scores = self._score_clause_lexical_idf(text, token_index, N)
            if not clause_scores:
                continue

            lex_scale = 1.0 if text in primary_text_set else 0.65
            kept = sorted(clause_scores.items(), key=lambda x: x[1], reverse=True)[:100]
            grouped_candidates.append({c: float(sc) * lex_scale for c, sc in kept})

        # Step 3.1: Candidate Subgraph / Idiom Group Preservation
        # Discovered idioms represent coherent topological paths across clauses.
        # Normalize summed clause lexical scores into [0, 1] and weight by
        # topological edge coherence so disconnected cross-domain coincidences
        # cannot clamp weak lexical hits to 1.0 over strong dense semantic matches.
        n_cl_norm = float(max(len(clauses), 1))
        for idiom in candidate_idioms[:3]:
            if not idiom.cells:
                continue
            idiom_group: Dict[Cell, float] = {}
            raw_lex = float(getattr(idiom, "lexical_score", 0.0) or 0.0)
            if raw_lex <= 0.0:
                continue
            edge_ratio = (
                min(1.0, float(len(getattr(idiom, "edges", []) or [])) / float(max(len(clauses) - 1, 1)))
                if len(clauses) > 1
                else 1.0
            )
            base_score = min(1.0, (raw_lex / n_cl_norm) * (0.5 + 0.5 * edge_ratio))
            entry_bonus = float(_cfg("idiom_entry_bonus", 0.2))
            for c in idiom.cells:
                sc_i = base_score * (1.0 + (entry_bonus if c == idiom.entry_node else 0.0))
                idiom_group[c] = min(1.0, sc_i)
            if idiom_group:
                grouped_candidates.append(idiom_group)

        # Step 3.2: Corpus-derived query expansion (pseudo-relevance feedback).
        # One bounded round: the first-pass candidate distribution reveals the
        # lattice vocabulary the prompt could not name itself; high-support
        # declared tokens absent from the prompt become one extra retrieval
        # span applied to BOTH the lexical and dense channels below. No
        # synonyms are ever hardcoded: the association is measured from the
        # loaded cells' own declared identity vocabulary.
        prf_spans: List[str] = []
        if not is_p0:
            try:
                prelim = self._tunnel_from_groups(grouped_candidates)
                exp_tokens = self._harvest_expansion_tokens(prompt, prelim)
                if exp_tokens:
                    span = " ".join(exp_tokens)
                    prf_spans.append(span)
                    exp_scores = self._score_clause_lexical_idf(span, token_index, N)
                    if exp_scores:
                        kept_exp = sorted(exp_scores.items(), key=lambda x: x[1], reverse=True)[:100]
                        grouped_candidates.append(dict(kept_exp))
                    logger.debug("[ROUTER] PRF expansion span: %r", span)
            except Exception as _prf_err:
                logger.debug("[ROUTER] PRF expansion skipped: %s", _prf_err)

        # Step 4: Dense vector blending from edge-context embeddings.
        # Architecture contract (Profile A/C/D/E): the dense pass ALWAYS runs
        # when an indexed RAG is present. Lexical stage-1 idiom evidence and
        # dense embedding evidence are complementary modalities; a lexical
        # heuristic must never short-circuit the embed + FAISS pass, so the
        # reported dense-retrieval behavior in every profile is genuine.
        clause_dense_scores: List[Dict[str, float]] = [{} for _ in clauses]
        if not is_p0 and self.internal_rag is not None and self.internal_rag.index is not None:
            clause_spans = [c.strip() for c in primary_texts if c and c.strip()]
            query_spans = clause_spans + [
                q for q in (prf_spans + window_texts)
                if q and q not in clause_spans
            ]
            span_to_idx = {sp: idx for idx, sp in enumerate(query_spans)}
            # Divide the retrieval budget across spans instead of granting every
            # span the full top_k, so total dense work is bounded by top_k
            # regardless of span count.
            n_spans = max(len(query_spans), 1)
            per_span_k = max(
                DENSE_RAG_MIN_PER_SPAN,
                min(DENSE_RAG_MAX_PER_SPAN, top_k // n_spans),
            )
            rag_results = self.internal_rag.get_relevant_context_batch(query_spans, top_k=per_span_k)
            for cl_idx, cl_text in enumerate(clauses):
                sp_idx = span_to_idx.get(cl_text.strip())
                if sp_idx is not None and sp_idx < len(rag_results):
                    for item in rag_results[sp_idx]:
                        cid = item.get("cell_id")
                        dsc = float(item.get("score", 0.0) or 0.0)
                        if cid and dsc > 0.0:
                            clause_dense_scores[cl_idx][cid] = max(
                                clause_dense_scores[cl_idx].get(cid, 0.0), dsc
                            )
            self.last_clause_semantic_scores = [dict(d) for d in clause_dense_scores]
            for span_results in rag_results:
                span_group: Dict[Cell, float] = {}
                for item in span_results:
                    cid = item.get("cell_id")
                    if not cid:
                        continue
                    dense_sc = float(item.get("score", 0.0) or 0.0)
                    rrf_sc = float(item.get("rrf_score", 0.0) or 0.0)
                    # Commensurate [0, 1] fusion weight per clause/span: preserve real
                    # dense cosine similarity alongside normalized RRF fusion weight without
                    # allowing weak lexical-only overlap to match strong dense cosine matches.
                    if dense_sc > 0.05:
                        weight = min(1.0, max(dense_sc, 0.75 * dense_sc + 0.25 * rrf_sc))
                    else:
                        weight = min(1.0, 0.6 * rrf_sc)
                    if weight <= 0.0:
                        continue
                    cell_obj = self.orchestrator.loaded_cells.get(cid)
                    if cell_obj is None:
                        continue
                    prev_w = span_group.get(cell_obj, 0.0)
                    if weight > prev_w:
                        span_group[cell_obj] = weight
                if span_group:
                    top_span = sorted(span_group.items(), key=lambda x: x[1], reverse=True)[:40]
                    grouped_candidates.append(dict(top_span))

        candidates_with_scores = self._tunnel_from_groups(grouped_candidates)

        if not candidates_with_scores:
            return [], {}


        final_tunnel: List[Cell] = [c for c, _ in candidates_with_scores]
        relevance_map = {c.cell_id: s for c, s in candidates_with_scores}
        tunnel_ids = set(c.cell_id for c in final_tunnel)

        # Step 4.1 (Profile S): Structured IR tunnel constraint. Each compiled
        # IR step deterministically contributes the cells whose DECLARED
        # operation/identity vocabulary matches the step's op (and, when the
        # step names a library, that domain). This keeps the typed-IR output
        # in its structured form all the way into the tunnel instead of
        # flattening it back to unstructured text and hoping lexical recall
        # recovers it.
        if getattr(self, "last_compiled_ir", None) is not None:
            for s in getattr(self.last_compiled_ir, "steps", []) or []:
                op_tok = CellTokenizer.tokenize_identifier(str(s.get("op", "")))
                if not op_tok:
                    continue
                lib_name = str(s.get("library", "")).strip().lower()
                for cell in self.orchestrator.loaded_cells.values():
                    if cell.cell_id in tunnel_ids:
                        continue
                    if lib_name and str(getattr(cell, "domain_name", "")).lower() not in (lib_name, ""):
                        continue
                    identity = getattr(cell, "identity_tokens", getattr(cell, "token_set", set()))
                    if op_tok & identity:
                        final_tunnel.append(cell)
                        tunnel_ids.add(cell.cell_id)
                        relevance_map[cell.cell_id] = max(
                            relevance_map.get(cell.cell_id, 0.0), self.epsilon * 2.0
                        )

        # Category-Theoretic Bridge Morphisms are discovered dynamically during Layer 2
        # topological routing (via find_shortest_bridge_path in the lattice graph), preserving
        # Layer 1 semantic query-grounding purity. Zero ungrounded bridge injection in Layer 1.
        reg = TypeRegistry.get_instance()
        carriers_out = set()
        carriers_in = set()
        for c in final_tunnel:
            out_t = getattr(c.primary_output, "type_name", "")
            in_t = getattr(c.primary_input, "type_name", "")
            if out_t:
                ot_low = out_t.lower()
                if not (reg.is_declared_top(ot_low) or reg.is_void_type(ot_low)):
                    carriers_out.add(out_t)
            if in_t:
                it_low = in_t.lower()
                if not (reg.is_declared_top(it_low) or reg.is_void_type(it_low)):
                    carriers_in.add(in_t)

        if not is_p0 and carriers_out and carriers_in:
            # Property & Qualifier Categorical Boundary Closure:
            # If an active producer establishes a property/qualifier (e.g. is_complex=True)
            # that conflicts with an active consumer's precondition/rejected qualifier on a
            # compatible carrier, discover and inject the resolving morphism h : (T, phi_1) -> (T, phi_2).
            active_cells = list(final_tunnel)
            produced_props = []
            for c in active_cells:
                c_out = getattr(c.primary_output, "type_name", "")
                if not c_out:
                    continue
                for post in (getattr(c, "postconditions", None) or []):
                    p_name = post.get("property") if isinstance(post, dict) else getattr(post, "property", None)
                    p_val = post.get("value") if isinstance(post, dict) else getattr(post, "value", None)
                    p_op = post.get("operator", "==") if isinstance(post, dict) else getattr(post, "operator", "==")
                    if p_name is not None:
                        produced_props.append((c, c_out.lower(), p_name, p_op, p_val))
                for q in (getattr(c.primary_output, "qualifiers", None) or []):
                    produced_props.append((c, c_out.lower(), f"qualifier:{str(q).lower()}", "==", True))

            required_props = []
            for c in active_cells:
                ins = getattr(c, "inputs", {}) or {}
                input_list = ins.values() if isinstance(ins, dict) else ins
                for inp in input_list:
                    c_in = getattr(inp, "type_name", "")
                    if not c_in:
                        continue
                    for pre in (getattr(c, "preconditions", None) or []):
                        p_name = pre.get("property") if isinstance(pre, dict) else getattr(pre, "property", None)
                        p_val = pre.get("value") if isinstance(pre, dict) else getattr(pre, "value", None)
                        p_op = pre.get("operator", "==") if isinstance(pre, dict) else getattr(pre, "operator", "==")
                        if p_name is not None:
                            required_props.append((c, c_in.lower(), p_name, p_op, p_val))
                    for rq in (getattr(inp, "rejected_qualifiers", None) or []):
                        required_props.append((c, c_in.lower(), f"qualifier:{str(rq).lower()}", "==", False))

            conflicts = []
            seen_conflicts = set()
            for prod_cell, prod_carrier, p_name, p_op, p_val in produced_props:
                for cons_cell, cons_carrier, req_name, req_op, req_val in required_props:
                    if prod_cell.cell_id == cons_cell.cell_id:
                        continue
                    if p_name == req_name and p_op == "==" and req_op == "==":
                        if prod_carrier == cons_carrier or (reg and (reg.is_subtype(prod_carrier, cons_carrier) or reg.is_subtype(cons_carrier, prod_carrier))):
                            if p_val != req_val:
                                req_hashable = tuple(req_val) if isinstance(req_val, (list, set)) else req_val
                                c_key = (prod_carrier, p_name, req_hashable)
                                if c_key not in seen_conflicts:
                                    seen_conflicts.add(c_key)
                                    conflicts.append((prod_carrier, p_name, req_val))

            if conflicts:
                for carrier, prop, target_val in conflicts:
                    for cand in self.orchestrator.loaded_cells.values():
                        if cand.cell_id in tunnel_ids:
                            continue
                        if getattr(cand, "is_ingress", False) or getattr(cand, "is_sink", False):
                            continue
                        cand_in = str(getattr(cand.primary_input, "type_name", "") or "").lower()
                        cand_out = str(getattr(cand.primary_output, "type_name", "") or "").lower()
                        if not (cand_in == carrier or (reg and reg.is_subtype(carrier, cand_in))):
                            continue
                        if not (cand_out == carrier or (reg and reg.is_subtype(cand_out, carrier))):
                            continue

                        resolves = False
                        if prop.startswith("qualifier:"):
                            q_name = prop.split(":", 1)[1]
                            cand_out_quals = [str(q).lower() for q in (getattr(cand.primary_output, "qualifiers", None) or [])]
                            if target_val is False and q_name not in cand_out_quals:
                                resolves = True
                        else:
                            for cp in (getattr(cand, "postconditions", None) or []):
                                cp_name = cp.get("property") if isinstance(cp, dict) else getattr(cp, "property", None)
                                cp_val = cp.get("value") if isinstance(cp, dict) else getattr(cp, "value", None)
                                cp_op = cp.get("operator", "==") if isinstance(cp, dict) else getattr(cp, "operator", "==")
                                if cp_name == prop and cp_op == "==" and cp_val == target_val:
                                    resolves = True
                                    break

                        if resolves:
                            final_tunnel.append(cand)
                            tunnel_ids.add(cand.cell_id)
                            # 0.15 >= 0.05 guarantees inclusion in protected_ids during stratified pruning
                            relevance_map[cand.cell_id] = max(relevance_map.get(cand.cell_id, 0.0), 0.15)
                            logger.info("[ROUTER] Injected resolving morphism %s for refinement conflict (%s, %s == %s)", cand.cell_id, carrier, prop, target_val)

        if not is_p0 and (carriers_out or carriers_in):
            prompt_toks = CellTokenizer.tokenize_prompt(prompt) if prompt else set()
            active_domains = set(c.domain_name for c in final_tunnel if c.domain_name)
            # Carrier-completion augmentation is intentionally TIGHT: only cells in
            # an active domain whose carriers fit the active boundary AND share at
            # least content tokens with the prompt. An unbounded same-domain
            # flood drowns the trellis in noise cells (measured: +571 cells on a
            # 150-cell tunnel in the 38K-node corpus).
            ingress_stage = reg.get_ingress_stage() if hasattr(reg, "get_ingress_stage") else 1
            ingress_roles = set(reg.get_verification_semantics("ingress_roles")) if hasattr(reg, "get_verification_semantics") else {"source"}
            if not ingress_roles:
                ingress_roles = {"source"}
            carrier_augment_max = int(_cfg("carrier_augment_max", 35))
            carrier_min_overlap = int(_cfg("carrier_min_overlap", 2))
            augment: List[Cell] = []
            for cell in self.orchestrator.loaded_cells.values():
                if cell.domain_name in active_domains and cell.cell_id not in tunnel_ids:
                    c_in = getattr(cell.primary_input, "type_name", "")
                    c_out = getattr(cell.primary_output, "type_name", "")
                    is_ingress_src = getattr(cell, "stage", None) == ingress_stage and getattr(cell, "node_role", "") in ingress_roles
                    min_ov = 1 if is_ingress_src else carrier_min_overlap
                    tok_ov = len(getattr(cell, "identity_tokens", cell.token_set) & prompt_toks)
                    if (c_in in carriers_out or c_out in carriers_in) and tok_ov >= min_ov:
                        augment.append(cell)
            augment.sort(key=lambda c: (
                len(getattr(c, "identity_tokens", c.token_set) & prompt_toks),
                relevance_map.get(c.cell_id, 0.0),
                1 if getattr(c, "is_public", True) else 0,
                -str(getattr(c, "cell_id", "")).count("."),
            ), reverse=True)
            for cell in augment[:carrier_augment_max]:
                final_tunnel.append(cell)
                tunnel_ids.add(cell.cell_id)
                relevance_map[cell.cell_id] = max(relevance_map.get(cell.cell_id, 0.0), self.epsilon * 2.0)

        # Macro-goal routing toggle: macro-goal cells participate in the
        # tunnel as ordinary lexical members when enabled; when disabled they
        # are excluded entirely, restoring the pre-macro routing distribution
        # exactly (clean A/B baseline for benchmarks).
        if not self.macros_enabled:
            final_tunnel = [c for c in final_tunnel if not (isinstance(c, MacroCell) or getattr(c, "is_macro_goal", False) or bool(getattr(c, "sub_cells", None)))]
            tunnel_ids = {c.cell_id for c in final_tunnel}
            relevance_map = {
                cid: s for cid, s in relevance_map.items()
                if cid in tunnel_ids
            }
        # Macro-goal promotion (Section 3.5): known-good composite paths.
        # When the prompt is relevant to a MacroCell (a predefined, verified
        # path over existing micro-cells — no new functionality), the macro
        # enters the tunnel and scores ABOVE every one of its constituent
        # micro-cells, sharpening the routing distribution toward the
        # known-good path. Skipped entirely when macros are disabled.
        if self.macros_enabled and not is_p0:
            self._promote_macro_goals(prompt, final_tunnel, tunnel_ids, relevance_map)
        final_tunnel = self._prune_tunnel_stratified(
            final_tunnel, relevance_map,
            num_semantic_clauses=num_semantic_clauses,
        )

        for cid, s in relevance_map.items():
            if cid not in self.priority_map:
                self.priority_map[cid] = s

        # Capture pure RAG baseline before neural reranking
        self.raw_rag_tunnel = list(final_tunnel)
        self.raw_rag_relevance_map = dict(relevance_map)
        self.last_reranker_telemetry = []
        if not is_p0 and clause_dense_scores:
            self.last_clause_semantic_scores = [dict(d) for d in clause_dense_scores]

        # Neural Reranking Pass (Optional / Experimental)
        if not is_p0 and getattr(self, "use_reranker", False) and getattr(self, "reranker", None) is not None:
            final_tunnel, relevance_map, self.last_reranker_telemetry = self.reranker.rerank(
                prompt,
                final_tunnel,
                relevance_map,
                top_k=min(max(top_k, 50), 80),
            )
            tunnel_by_id = {c.cell_id: c for c in final_tunnel}
            clause_sem = [dict(d) for d in clause_dense_scores]
            if len(clauses) <= 1 and clause_sem:
                calib_map = {
                    t["cell_id"]: float(t.get("calibrated_score", t.get("new_score", 0.0)) or 0.0)
                    for t in (self.last_reranker_telemetry or [])
                    if "cell_id" in t
                }
                for cid, calib_sc in calib_map.items():
                    if cid in clause_sem[0]:
                        clause_sem[0][cid] = 0.5 * float(clause_sem[0][cid]) + 0.5 * calib_sc
                    else:
                        clause_sem[0][cid] = calib_sc
            elif len(clauses) > 1 and clause_sem:
                for cl_idx, cl_text in enumerate(clauses):
                    cl_clean = cl_text.strip()
                    if not cl_clean:
                        continue
                    cl_pool: List[Cell] = []
                    cl_pool_ids: Set[str] = set()
                    for cid, _dsc in sorted(
                        clause_dense_scores[cl_idx].items(), key=lambda kv: kv[1], reverse=True
                    )[:15]:
                        c_obj = tunnel_by_id.get(cid)
                        if c_obj is not None and cid not in cl_pool_ids:
                            cl_pool.append(c_obj)
                            cl_pool_ids.add(cid)
                    for c_obj in final_tunnel:
                        if len(cl_pool) >= 25:
                            break
                        if c_obj.cell_id not in cl_pool_ids:
                            cl_pool.append(c_obj)
                            cl_pool_ids.add(c_obj.cell_id)
                    if not cl_pool:
                        continue
                    _, cl_rel, cl_telem = self.reranker.rerank(
                        cl_clean,
                        cl_pool,
                        {c.cell_id: relevance_map.get(c.cell_id, 0.0) for c in cl_pool},
                        top_k=len(cl_pool),
                    )
                    calib_map = {
                        t["cell_id"]: float(t.get("calibrated_score", t.get("new_score", 0.0)) or 0.0)
                        for t in (cl_telem or [])
                        if "cell_id" in t
                    }
                    for cid, calib_sc in calib_map.items():
                        if cid in clause_dense_scores[cl_idx]:
                            clause_sem[cl_idx][cid] = 0.5 * float(clause_dense_scores[cl_idx][cid]) + 0.5 * calib_sc
                        else:
                            clause_sem[cl_idx][cid] = calib_sc
                    for cid, rel_sc in (cl_rel or {}).items():
                        relevance_map[cid] = max(relevance_map.get(cid, 0.0), float(rel_sc or 0.0))
                final_tunnel = sorted(
                    final_tunnel,
                    key=lambda c: relevance_map.get(c.cell_id, 0.0),
                    reverse=True,
                )
            self.last_clause_semantic_scores = clause_sem
            for cid, s in relevance_map.items():
                self.priority_map[cid] = s

        # Profile Omega Stage 2: Calibrated Sigmoid Hybrid Retrieval + Backward Precondition Cone
        try:
            mm_st2 = ModelManager.get_instance()
            is_omega_st2 = bool(not is_p0 and getattr(mm_st2, "has_coalgebraic_unfolder", lambda: False)())
        except Exception:
            is_omega_st2 = False
        if not is_p0 and final_tunnel:
            if is_omega_st2 and not self.last_reranker_telemetry:
                self._calibrate_hybrid_sigmoid(relevance_map)
            added_cone = self._expand_backward_precondition_cone(
                prompt, final_tunnel, tunnel_ids, relevance_map
            )
            if added_cone:
                final_tunnel = sorted(
                    final_tunnel,
                    key=lambda c: relevance_map.get(c.cell_id, 0.0),
                    reverse=True,
                )

        if self.orchestrator is not None:
            self.orchestrator._last_clause_semantic_scores = (
                (prompt.strip(), [dict(d) for d in self.last_clause_semantic_scores])
                if (not is_p0 and self.last_clause_semantic_scores)
                else None
            )

        if getattr(self, "enable_route_cache", True) and hasattr(self, "_route_cache"):
            self._route_cache[cache_key] = {
                "final_tunnel": list(final_tunnel),
                "relevance_map": dict(relevance_map),
                "last_compiled_ir": self.last_compiled_ir,
                "last_ir_literals": dict(self.last_ir_literals or {}),
                "last_ir_steps": getattr(self, "_last_ir_steps", 0),
                "raw_rag_tunnel": list(self.raw_rag_tunnel or []),
                "raw_rag_relevance_map": dict(self.raw_rag_relevance_map or {}),
                "last_reranker_telemetry": list(self.last_reranker_telemetry or []),
                "last_clause_semantic_scores": [dict(d) for d in (self.last_clause_semantic_scores or [])],
                "priority_map": dict(self.priority_map or {}),
            }

        return final_tunnel, relevance_map

    @staticmethod
    def _calibrate_hybrid_sigmoid(relevance_map: Dict[str, float]) -> None:
        """Logistic sigmoid calibration sigma((s - mu) / tau) for hybrid scores when no reranker ran."""
        if not relevance_map:
            return
        vals = [float(v) for v in relevance_map.values()]
        mean_v = sum(vals) / max(len(vals), 1)
        var_v = sum((v - mean_v) ** 2 for v in vals) / max(len(vals), 1)
        tau = max(math.sqrt(var_v), 0.18)
        for cid, raw in list(relevance_map.items()):
            z = max(-8.0, min(8.0, (float(raw) - mean_v) / tau))
            relevance_map[cid] = 1.0 / (1.0 + math.exp(-z))

    def _expand_backward_precondition_cone(
        self,
        prompt: str,
        final_tunnel: List[Cell],
        tunnel_ids: Set[str],
        relevance_map: Dict[str, float],
        max_hops: int = 2,
        max_added: int = 24,
    ) -> int:
        """
        Profile Omega Stage 2 Backward Precondition Cone:
        Ensures every high-relevance cell in `final_tunnel` has upstream producer
        morphisms in `final_tunnel` capable of supplying its required carrier types
        and preconditions (up to `max_hops` reverse unification hops).
        """
        try:
            from unification import unify, _pointed_default
            from lattice import is_path_port
        except (ImportError, ValueError):
            from .unification import unify, _pointed_default
            from .lattice import is_path_port

        prompt_toks = CellTokenizer.tokenize_prompt(prompt) if prompt else set()
        loaded = list(getattr(self.orchestrator, "loaded_cells", {}).values())
        if not loaded:
            return 0
        reg = TypeRegistry.get_instance()

        # Collect currently producible output signatures in tunnel
        frontier_goals: List[Tuple[Cell, float]] = [
            (c, float(relevance_map.get(c.cell_id, 0.0)))
            for c in final_tunnel[:12]
        ]
        added = 0

        for _hop in range(max_hops):
            if added >= max_added or not frontier_goals:
                break
            produced_sigs = []
            for tc in final_tunnel:
                for op in (getattr(tc, "outputs", {}) or {}).values():
                    osig = getattr(op, "signature", op)
                    if osig is not None:
                        produced_sigs.append(osig)

            next_frontier: List[Tuple[Cell, float]] = []
            for target_c, target_sc in frontier_goals:
                ins = getattr(target_c, "inputs", {}) or {}
                if not isinstance(ins, dict):
                    continue
                for ip in ins.values():
                    if not getattr(ip, "required", True) or getattr(ip, "default_value", None) is not None:
                        continue
                    role = str(getattr(ip, "port_role", "") or getattr(ip, "role", "") or "").lower()
                    if reg.is_config_role(role) or reg.is_hof_role(role) or role in set(reg.get_verification_semantics("parameter_roles") or ()):
                        continue
                    if is_path_port(ip) or _pointed_default(ip, target_c) is not None:
                        continue
                    isig = getattr(ip, "signature", ip)
                    if isig is None:
                        continue
                    # Already satisfied by an active output carrier in tunnel?
                    if any(unify(psig, isig) is not None for psig in produced_sigs):
                        continue

                    # Find highest-affinity upstream producer outside tunnel unifying with `isig`
                    best_prod = None
                    best_sc = -1.0
                    for cand in loaded:
                        if cand.cell_id in tunnel_ids or getattr(cand, "node_type", "") == "constant":
                            continue
                        outs = (getattr(cand, "outputs", {}) or {}).values()
                        if not any(unify(getattr(op, "signature", op), isig) is not None for op in outs):
                            continue
                        id_toks = set(getattr(cand, "identity_tokens", ())) | set(CellTokenizer.tokenize_identifier(cand.cell_id))
                        ov = len(prompt_toks & id_toks)
                        dom_match = 0.25 if getattr(cand, "domain_name", "") == getattr(target_c, "domain_name", "") else 0.0
                        depth_pen = 0.001 * str(getattr(cand, "cell_id", "")).count(".") + (0.0 if getattr(cand, "is_public", True) else 0.05)
                        sc = 0.20 * ov + dom_match - depth_pen
                        if sc > best_sc:
                            best_sc = sc
                            best_prod = cand
                    if best_prod is not None and best_prod.cell_id not in tunnel_ids:
                        final_tunnel.append(best_prod)
                        tunnel_ids.add(best_prod.cell_id)
                        new_rel = max(float(relevance_map.get(best_prod.cell_id, 0.0)), max(0.08, 0.78 * target_sc))
                        relevance_map[best_prod.cell_id] = new_rel
                        next_frontier.append((best_prod, new_rel))
                        added += 1
                        if added >= max_added:
                            break
                if added >= max_added:
                    break
            frontier_goals = next_frontier
        return added


    def _prune_tunnel_stratified(self, tunnel_cells, relevance_map,
                                 num_semantic_clauses: int = 1,
                                 max_s1=None, max_s2=None, max_s3=None):
        """(apply_fixes_v7) Stratified pruning; caps scale with task size and
        resolve through config (see PRUNE_* module constants)."""
        n = max(int(num_semantic_clauses or 1), 1)
        max_s1 = max_s1 if max_s1 is not None else max(PRUNE_S1_MIN, n * PRUNE_S1_PER_CLAUSE)
        max_s2 = max_s2 if max_s2 is not None else max(PRUNE_S2_MIN, n * PRUNE_S2_PER_CLAUSE)
        max_s3 = max_s3 if max_s3 is not None else max(PRUNE_S3_MIN, n * PRUNE_S3_PER_CLAUSE)

        def _stage_num(c):
            st = getattr(c, 'stage', -1)
            if isinstance(st, int) and st >= 0:
                return st
            st_str = str(st).strip().upper()
            if st_str in ('1', 'S1', 'INGRESS'):
                return 1
            if st_str in ('2', 'S2', 'TRANSFORM'):
                return 2
            if st_str in ('3', 'S3', 'EGRESS', 'SINK'):
                return 3
            clean = st_str.lstrip('S')
            if clean.isdigit():
                return int(clean)
            return 0

        # Protect candidates whose relevance earned a genuine cutoff above the epsilon/tau noise floor
        cutoff = max(self.epsilon * 2.0, 0.02)
        protected_ids = {c.cell_id for c in tunnel_cells if relevance_map.get(c.cell_id, 0.0) >= cutoff}
        from collections import defaultdict
        stage_cells = defaultdict(list)
        for c in tunnel_cells:
            stage_cells[_stage_num(c)].append(c)

        keep_stage_cells = []
        for st_id, cells in stage_cells.items():
            if st_id == 1:
                cap = max_s1
            elif st_id == 2:
                cap = max_s2
            elif st_id == 3:
                cap = max_s3
            else:
                cap = max(PRUNE_S2_MIN, n * PRUNE_S2_PER_CLAUSE)
            sorted_st = sorted(cells, key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)[:cap]
            keep_stage_cells.extend(sorted_st)

        bridge_ids = {c.cell_id for c in tunnel_cells if getattr(c, "node_role", "") == "bridge" or getattr(c, "node_type", "") == "tunnel" }
        keep_ids = {c.cell_id for c in keep_stage_cells} | protected_ids | bridge_ids
        return [c for c in tunnel_cells if c.cell_id in keep_ids] or tunnel_cells

    def _promote_macro_goals(
        self,
        prompt: str,
        final_tunnel: List[Cell],
        tunnel_ids: Set[str],
        relevance_map: Dict[str, float]
    ) -> None:
        """
        Enters matched MacroCells into the tunnel with HONEST, EARNED scores.

        A macro is a macro-goal: an empirically verified path over existing
        micro-cells (synapses forming a known-good macro-path). Its evidence
        is IDF-weighted CONCEPT COVERAGE: the macro's declared identity
        vocabulary (cell id + keywords) must cover a substantial fraction of
        the prompt's semantic mass. Generic I/O vocabulary ('csv', 'save')
        carries near-zero IDF and can therefore never enter on its own.

        The macro's relevance is exactly its measured concept coverage — it
        competes on the router's own distribution like every other cell. NO
        fabricated priority inflation is applied: the macro is never
        artificially forced above its constituent micro-cells regardless of
        routing probabilities, and constituent micro-cells are never injected
        with boosted scores they did not earn. Downstream, the Unification
        Gate expands a promoted macro back into its micro-cell sequence (the
        macro carries no new functionality), so routing fidelity is preserved
        without rigging the distribution.

        All scaling constants resolve through config (see MACRO_* module
        constants); see settings for tuning semantics.
        """
        prompt_toks = CellTokenizer.tokenize_prompt(prompt) if prompt else set()
        if not prompt_toks:
            return

        token_index = getattr(self.orchestrator, "token_index", None) or {}
        n_cells = max(len(self.orchestrator.loaded_cells), 1)

        def _idf(tok: str) -> float:
            return math.log(1.0 + (n_cells + 1) / (len(token_index.get(tok, ())) + 1.0))

        prompt_mass = sum(_idf(t) for t in prompt_toks)
        if prompt_mass <= 0:
            return

        promoted: List[Tuple[float, Cell]] = []
        for cell in self.orchestrator.loaded_cells.values():
            if not (isinstance(cell, MacroCell) and getattr(cell, "sub_cells", None)):
                continue
            identity_hits = getattr(cell, "identity_tokens", set()) & prompt_toks
            if not identity_hits:
                continue

            idf_mass = sum(_idf(t) for t in identity_hits)
            concept_coverage = idf_mass / prompt_mass
            if concept_coverage < MACRO_CONCEPT_COVERAGE_GATE:
                continue

            # Earned relevance: the macro's measured coverage of the prompt's
            # semantic mass, bounded to [0, 1]. Strictly derived from declared
            # identity vocabulary and the routing token statistics.
            earned_rel = min(1.0, concept_coverage)
            promoted.append((earned_rel, cell))

        for earned_rel, cell in promoted:
            if cell.cell_id not in tunnel_ids:
                final_tunnel.append(cell)
                tunnel_ids.add(cell.cell_id)
            if earned_rel > relevance_map.get(cell.cell_id, 0.0):
                relevance_map[cell.cell_id] = earned_rel
            self.priority_map[cell.cell_id] = max(
                self.priority_map.get(cell.cell_id, 0.0), earned_rel
            )

        if promoted:
            logger.debug(
                f"[ROUTER] Macro-goal participation (earned coverage): "
                f"{[_extract_id(c) for _, c in promoted]}"
            )

    def _harvest_expansion_tokens(
        self,
        prompt: str,
        prelim: List[Tuple[Cell, float]],
        max_cells: int = 12,
        max_tokens: int = 6,
    ) -> List[str]:
        """
        Corpus-derived query expansion (pseudo-relevance feedback), ONE round.

        The first retrieval pass returns the cells the prompt's own vocabulary
        can reach. Their DECLARED identity vocabulary (cell ids, keywords,
        semantic tags) is the lattice's own way of naming the prompt's intent;
        tokens that recur across MANY independently retrieved cells while being
        ABSENT from the prompt are exactly the vocabulary the prompt is missing
        (e.g. a prompt saying "train" retrieves estimators whose declared
        identity vocabulary says "fit"). Those tokens become one expansion
        query span, giving both the lexical and dense channels a second chance
        at the cells the prompt's own wording could not name.

        Fully domain-agnostic: tokens, IDF and support all come from the loaded
        lattice and the registry's harvested function-word statistics.
        """
        if not prelim:
            return []
        prompt_toks = CellTokenizer.tokenize_prompt(prompt) if prompt else set()
        try:
            reg = TypeRegistry.get_instance()
            stop = set(reg.get_function_words()) | set(reg.get_sentence_connectives())
        except Exception:
            stop = set()
        try:
            ns = getattr(self.planner, "_vocab", None)
            namespace = set(ns.namespace_tokens) if ns is not None else set()
        except Exception:
            namespace = set()

        token_index = getattr(self.orchestrator, "token_index", None) or {}
        n_cells = max(len(self.orchestrator.loaded_cells), 1)

        def _idf(tok: str) -> float:
            return math.log(1.0 + (n_cells + 1) / (len(token_index.get(tok, ())) + 1.0))

        top_cells = [(c, s) for c, s in prelim[:max_cells] if s > 0.0]
        if len(top_cells) < 2:
            return []

        support: Dict[str, int] = {}
        assoc: Dict[str, float] = {}
        for cell, score in top_cells:
            id_toks = set(getattr(cell, "identity_tokens", cell.token_set) or set())
            fresh = {
                t for t in id_toks
                if len(t) >= 3
                and t not in prompt_toks
                and t not in stop
                and t not in namespace
            }
            for t in fresh:
                support[t] = support.get(t, 0) + 1
                assoc[t] = assoc.get(t, 0.0) + float(score)

        ranked = sorted(
            ((t, a) for t, a in assoc.items() if support[t] >= 2),
            key=lambda kv: (kv[1] * _idf(kv[0]), support[kv[0]]),
            reverse=True,
        )
        return [t for t, _ in ranked[:max_tokens]]

    def _tunnel_from_groups(
        self,
        groups: List[Dict[Cell, float]]
    ) -> List[Tuple[Cell, float]]:
        """
        Computes the group-relative softmax tunnel:
          P(v | group_g) = exp(s_v / gamma) / Z_g,  keep v iff P >= tau * max_u P(u | g)
        A cell survives if it clears the threshold in ANY group; its relevance is
        the maximum normalized probability across groups. This preserves every
        sub-goal of a compound prompt at the distribution level.
        """
        tau = max(self.epsilon, 0.01)
        relevance: Dict[str, Tuple[Cell, float]] = {}

        for group in groups:
            if not group:
                continue
            items = list(group.items())
            raw_scores = np.array([s for _, s in items], dtype=np.float64)
            max_s = np.max(raw_scores)
            if max_s <= 0.0:
                continue
            # Universal Clause-Relative Activation:
            # Prevents group-size dilution where large candidate pools are penalized by 1/N.
            # Relative decay from the group peak is governed by temperature gamma, anchored to peak score.
            shifted = (raw_scores - max_s) / max(self.gamma, 1e-5)
            calibrated_probs = np.exp(shifted) * min(1.0, float(max_s))
            cutoff = tau * min(1.0, float(max_s))
            sorted_indices = np.argsort(raw_scores)[::-1]
            top_k_guaranteed = int(_cfg("group_top_k_guaranteed", 8))
            top_indices = set(sorted_indices[:top_k_guaranteed])

            for i, ((cell, _), p) in enumerate(zip(items, calibrated_probs)):
                if p < cutoff and i not in top_indices:
                    continue
                effective_p = max(float(p), float(cutoff)) if i in top_indices else float(p)
                cid = cell.cell_id
                prev = relevance.get(cid)
                if prev is None or effective_p > prev[1]:
                    relevance[cid] = (cell, effective_p)

        ranked = sorted(relevance.values(), key=lambda x: x[1], reverse=True)
        return ranked

    @staticmethod
    def _generate_query_spans(text: str) -> List[str]:
        """
        Generates continuous sliding semantic window queries across the prompt.
        Multi-scale representation: decomposes compound sentences and spans across
        the entire prompt uniformly so trailing and embedded operations are never truncated.

        Bounded: the returned list is capped at MAX_QUERY_SPANS so the downstream
        dense-RAG stage cannot blow up on long prompts. Dedup uses a set for
        O(1) membership rather than O(n) list scans. Short prompts
        (<= SHORT_PROMPT_WINDOW_SKIP tokens) skip the sliding-window pass
        entirely: their clause-level decomposition already covers their
        structure, and sliding windows on 4-6 tokens just re-emit substrings.

        The delimiter set used to break compound clauses comes from
        CellTokenizer.CLAUSE_DELIMITERS when present; a module-local fallback
        keeps this function usable before that lands.
        """
        tokens = text.strip().split()
        if len(tokens) <= 3:
            return [text.strip()]

        seen: Set[str] = set()
        queries: List[str] = []

        def _add(q: str) -> None:
            q = q.strip(" ,.;:'\"")
            if len(q) >= 2 and q not in seen:
                seen.add(q)
                queries.append(q)

        _add(text.strip())

        # 1. Clause-level decomposition and sub-action split on delimiters
        clauses = _segment_prompt_clauses(text)
        delims = _get_clause_delims()
        for cl in clauses:
            _add(cl)
            current_chunk: List[str] = []
            for w in cl.split():
                cw = w.lower().strip(" ,.;:'\"")
                if cw in delims:
                    if current_chunk:
                        _add(" ".join(current_chunk))
                        current_chunk = []
                else:
                    current_chunk.append(w)
            if current_chunk:
                _add(" ".join(current_chunk))

        # Short prompts: clause-level decomposition is sufficient; sliding
        # windows would only re-emit overlapping substrings of the same few
        # tokens.
        if len(tokens) <= SHORT_PROMPT_WINDOW_SKIP:
            return queries[:MAX_QUERY_SPANS]

        # 2. Sliding multi-scale windows spanning uniformly across the full prompt
        n = len(tokens)
        for window_size in (2, 3, 4):
            if window_size >= n:
                continue
            step = max(1, (n - window_size) // 8)
            for i in range(0, n - window_size + 1, step):
                _add(" ".join(tokens[i : i + window_size]))

        _add(" ".join(tokens[max(0, n - 3) :]))

        # Bounded retrieval fan-out; see MAX_QUERY_SPANS.
        return queries[:MAX_QUERY_SPANS]

    @staticmethod
    def _parse_program_0_steps(prompt: str) -> Tuple[Optional[List[Tuple[str, Dict[str, Any]]]], Optional[str]]:
        """
        Strict Profile 0 Formal Composition Grammar Parser:
          Program_0 ::= Step ("->" Step)*
          Step      ::= <cell_id | op_name> "(" (<port_name> "=" <literal> ("," ...)*)? ")"
        Uses deterministic character scanning and Python AST literal verification (zero regex).
        """
        if not prompt or not prompt.strip():
            return None, "Empty prompt under Profile 0"

        text = prompt.strip()
        segments: List[str] = []
        start = 0
        i = 0
        n = len(text)
        in_quote: Optional[str] = None
        paren_depth = 0
        brack_depth = 0
        brace_depth = 0

        while i < n:
            ch = text[i]
            if in_quote is not None:
                if ch == "\\":
                    i += 2
                    continue
                if ch == in_quote:
                    in_quote = None
                i += 1
                continue
            if ch in ("'", '"'):
                in_quote = ch
                i += 1
                continue
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
                if paren_depth < 0:
                    return None, "Profile 0 syntax error: unbalanced ')'"
            elif ch == "[":
                brack_depth += 1
            elif ch == "]":
                brack_depth -= 1
                if brack_depth < 0:
                    return None, "Profile 0 syntax error: unbalanced ']'"
            elif ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth < 0:
                    return None, "Profile 0 syntax error: unbalanced '}'"
            elif paren_depth == 0 and brack_depth == 0 and brace_depth == 0 and text[i : i + 2] == "->":
                segments.append(text[start:i])
                i += 2
                start = i
                continue
            i += 1

        if in_quote is not None or paren_depth != 0 or brack_depth != 0 or brace_depth != 0:
            return None, "Profile 0 syntax error: unclosed quote or delimiter"
        segments.append(text[start:])

        parsed_steps: List[Tuple[str, Dict[str, Any]]] = []
        for idx, raw_seg in enumerate(segments):
            seg = raw_seg.strip()
            if not seg:
                return None, f"Profile 0 syntax error: empty step at position {idx}"
            try:
                node = ast.parse(seg, mode="eval").body
            except SyntaxError:
                return None, (
                    "Profile 0 requires formal composition syntax: "
                    "<cell_id|op_name>(port=literal, ...) [-> <cell_id|op_name>(...)]*"
                )
            if not isinstance(node, ast.Call):
                return None, (
                    "Profile 0 step must be <cell_id|op_name>(port=literal, ...)"
                )
            # Extract dotted or simple callable identifier (no arbitrary expressions)
            fn_node = node.func
            fn_parts: List[str] = []
            while isinstance(fn_node, ast.Attribute):
                fn_parts.append(fn_node.attr)
                fn_node = fn_node.value
            if not isinstance(fn_node, ast.Name):
                return None, (
                    "Profile 0 step must be <cell_id|op_name>(port=literal, ...)"
                )
            fn_parts.append(fn_node.id)
            ident_str = ".".join(reversed(fn_parts))

            if node.args:
                return None, (
                    "Profile 0 requires named port=literal arguments (no positional args)"
                )
            kwargs_dict: Dict[str, Any] = {}
            for kw in node.keywords:
                if kw.arg is None:
                    return None, "Profile 0 forbids **kwargs unpacking"
                if kw.arg in kwargs_dict:
                    return None, f"Profile 0 duplicate port binding '{kw.arg}' in step '{ident_str}'"
                try:
                    lit_val = ast.literal_eval(kw.value)
                except Exception:
                    return None, (
                        f"Profile 0 argument '{kw.arg}' in '{ident_str}' must be a literal constant"
                    )
                kwargs_dict[kw.arg] = lit_val
            parsed_steps.append((ident_str, kwargs_dict))

        return parsed_steps, None

    def _resolve_program_0(
        self,
        prompt: str,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
    ) -> Tuple[List[Cell], Optional[str]]:
        """
        Deterministically resolves a parsed Program_0 step sequence against
        formal {cell_id} / canonical_op_names and declared port schemas.
        """
        steps, parse_err = self._parse_program_0_steps(prompt)
        if steps is None:
            return [], parse_err

        loaded = getattr(self.orchestrator, "loaded_cells", {}) or {}
        if not loaded:
            return [], "No lattice cells loaded"

        by_cid: Dict[str, Cell] = {}
        by_op: Dict[str, List[Cell]] = defaultdict(list)
        for cell in loaded.values():
            by_cid[cell.cell_id.lower()] = cell
            for op in getattr(cell, "canonical_op_names", ()):
                by_op[op].append(cell)

        reg = TypeRegistry.get_instance()
        try:
            from unification import UnificationGate, unify, _is_key_or_complement_port, _is_valid_slot_value
            from preflight import PreflightLinter
        except (ImportError, ValueError):
            from .unification import UnificationGate, unify, _is_key_or_complement_port, _is_valid_slot_value
            from .preflight import PreflightLinter

        role_carriers = reg.get_declared_role_carriers()
        step_candidates: List[List[Tuple[Tuple[int, int, int], Cell, Dict[str, Any]]]] = []

        for step_idx, (ident, kwargs_dict) in enumerate(steps):
            ident_l = ident.strip().lower()
            exact_cell = by_cid.get(ident_l)
            if exact_cell is not None:
                pool = [(0, exact_cell)]
            else:
                op_cands = by_op.get(ident_l, [])
                if not op_cands:
                    return [], f"Profile 0 unknown formal cell_id or op_name: '{ident}'"
                pool = [(1, c) for c in op_cands]

            surviving: List[Tuple[Tuple[int, int, int], Cell, Dict[str, Any]]] = []
            saw_port_match = False
            for match_tier, cand in pool:
                if not set(kwargs_dict.keys()).issubset(set(cand.inputs.keys())):
                    continue
                saw_port_match = True

                # Literal schema / type / enum / format check on explicit kwargs
                fmt_mismatch = False
                for p_name, lit_val in kwargs_dict.items():
                    p_sig = cand.inputs[p_name]
                    tn_lit = str(getattr(getattr(p_sig, "signature", p_sig), "type_name", "") or "").strip().lower()
                    enums_lit = {str(e).strip().strip("'\"").lower() for e in (getattr(p_sig, "enum_values", None) or [])}
                    if lit_val is None:
                        if getattr(p_sig, "required", True) and "none" not in enums_lit:
                            fmt_mismatch = True
                            break
                        continue
                    if isinstance(lit_val, bool):
                        if tn_lit and tn_lit != "object" and not reg.is_declared_top(tn_lit) and not reg.is_subtype(tn_lit, "bool"):
                            fmt_mismatch = True
                            break
                        continue
                    if enums_lit:
                        if str(lit_val).strip().strip("'\"").lower() not in enums_lit:
                            fmt_mismatch = True
                            break
                    elif tn_lit and tn_lit != "object" and not reg.is_declared_top(tn_lit):
                        if (reg.is_subtype(tn_lit, "str") or reg.is_subtype(tn_lit, "path")) and not (
                            reg.is_subtype(tn_lit, "int") or reg.is_subtype(tn_lit, "numeric")
                        ):
                            if not isinstance(lit_val, str):
                                fmt_mismatch = True
                                break
                        elif not _is_valid_slot_value(lit_val, p_sig):
                            fmt_mismatch = True
                            break
                    sup_fmts = getattr(p_sig, "supported_formats", None) or []
                    if sup_fmts and isinstance(lit_val, str) and "." in lit_val:
                        ext = lit_val.rsplit(".", 1)[-1].strip().lower()
                        norm_fmts = {str(s).strip().lstrip(".").lower() for s in sup_fmts if s}
                        if ext and norm_fmts and ext not in norm_fmts:
                            fmt_mismatch = True
                            break
                if fmt_mismatch:
                    continue

                # Require every non-carrier required port to have explicit binding or non-empty valid default
                missing_non_carrier = 0
                carrier_eligible_left = 1 if (step_idx > 0 or start_sig is not None) else 0
                prim_in_name = getattr(cand.primary_input, "name", None) if getattr(cand, "primary_input", None) else None
                for p_name, p_sig in cand.inputs.items():
                    if not getattr(p_sig, "required", True):
                        continue
                    if p_name in kwargs_dict:
                        continue
                    tn_l = str(getattr(getattr(p_sig, "signature", p_sig), "type_name", "") or "").lower()
                    p_role = str(getattr(p_sig, "port_role", "") or getattr(p_sig, "derived_role", "") or "").lower()
                    is_carrier = (
                        p_name == prim_in_name
                        or p_role in role_carriers
                        or reg.is_composite_carrier(tn_l)
                        or reg.is_abstract_carrier(tn_l)
                    ) and not reg.is_subtype(tn_l, "path") and not reg.is_subtype(tn_l, "str")
                    if is_carrier and carrier_eligible_left > 0:
                        carrier_eligible_left -= 1
                        continue
                    def_val = getattr(p_sig, "default_value", None)
                    has_usable_default = def_val is not None
                    if has_usable_default and _is_key_or_complement_port(cand, p_sig, reg):
                        if str(def_val).strip("'\"") in ("", "[]", "()"):
                            has_usable_default = False
                    if not has_usable_default:
                        missing_non_carrier += 1
                if missing_non_carrier > 0:
                    continue

                req_unfilled = sum(
                    1 for pn, ps in cand.inputs.items()
                    if getattr(ps, "required", True) and pn not in kwargs_dict and getattr(ps, "default_value", None) is None
                )
                vis_rank = (0 if getattr(cand, "is_public", True) else 1, str(getattr(cand, "cell_id", "")).count("."))
                surviving.append(((match_tier, req_unfilled, vis_rank), cand, kwargs_dict))

            if not surviving:
                if not saw_port_match:
                    return [], f"Profile 0 invalid port name(s) {sorted(kwargs_dict.keys())} for '{ident}'"
                return [], f"Profile 0 missing required argument(s) or incompatible format for '{ident}'"

            surviving.sort(key=lambda x: x[0])
            step_candidates.append(surviving[:8])

        def _pair_compatible(prev_c: Cell, next_c: Cell, prefix: List[Cell]) -> bool:
            if hasattr(self.orchestrator, "cells_directly_unify"):
                try:
                    if self.orchestrator.cells_directly_unify(prev_c, next_c):
                        return True
                except Exception:
                    pass
            # Also allow DAG ancestor output unification into next_c
            req_ins = [p for p in next_c.inputs.values() if getattr(p, "required", True)]
            in_ports = req_ins if req_ins else list(next_c.inputs.values())
            for anc in reversed(prefix):
                for out_p in anc.outputs.values():
                    for in_p in in_ports:
                        if unify(out_p.signature, in_p.signature) is not None:
                            return True
            return False

        def _validate_sequence(seq_items: List[Tuple[Cell, Dict[str, Any]]]) -> Optional[List[Cell]]:
            cloned_seq: List[Cell] = []
            for s_idx, (c_obj, kw) in enumerate(seq_items):
                c_clone = c_obj.clone() if hasattr(c_obj, "clone") else c_obj
                c_clone.explicit_port_bindings = dict(kw)
                c_clone.formal_program_step = True
                c_clone.matched_clause_idx = s_idx
                cloned_seq.append(c_clone)
            try:
                gate = UnificationGate(orchestrator=self.orchestrator)
                code = gate.unify_and_emit(cloned_seq, prompt)
                if gate.last_pipeline_bindings:
                    lint_res = PreflightLinter.lint(
                        gate.last_pipeline_bindings,
                        prompt=prompt,
                        code_str=code,
                        orchestrator=self.orchestrator,
                    )
                    if not lint_res.is_valid:
                        return None
                return cloned_seq
            except Exception:
                return None

        # Backtracking DFS over candidate combinations (small branching per formal op)
        chosen: List[Tuple[Cell, Dict[str, Any]]] = []

        def _dfs(pos: int) -> Optional[List[Cell]]:
            if pos == len(step_candidates):
                if goal_sig is not None and chosen:
                    last_out = getattr(chosen[-1][0], "primary_output", None)
                    if last_out is not None and unify(getattr(last_out, "signature", last_out), goal_sig) is None:
                        return None
                return _validate_sequence(chosen)

            for _, cand, kw in step_candidates[pos]:
                if pos == 0 and start_sig is not None:
                    first_in = getattr(cand, "primary_input", None)
                    if first_in is not None and unify(start_sig, getattr(first_in, "signature", first_in)) is None:
                        continue
                if pos > 0:
                    prefix_cells = [x[0] for x in chosen]
                    if not _pair_compatible(chosen[-1][0], cand, prefix_cells):
                        continue
                chosen.append((cand, kw))
                res = _dfs(pos + 1)
                if res is not None:
                    return res
                chosen.pop()
            return None

        resolved = _dfs(0)
        if not resolved:
            return [], "Profile 0 pipeline failed type unification or preflight validation"
        return resolved, None

    def plan_path(
        self,
        prompt: str,
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        return_tuple: bool = True,
        top_k: int = 400,
        route_method: Optional[str] = None,
        ctx: Optional[Any] = None,
        **kwargs
    ) -> Union[List[Cell], Tuple[List[Cell], Set[str]]]:
        """
        End-to-end routing & topological pathfinding:
          1. Computes active semantic tunnel T for prompt.
          2. Uses selected RouteMethod (M0 - M6) to find valid monadic composition through T.
        """
        if self._is_profile_zero():
            self.last_clause_semantic_scores = []
            if self.orchestrator is not None:
                self.orchestrator._last_clause_semantic_scores = None
            prog_cells, prog_err = self._resolve_program_0(prompt, start_sig=start_sig, goal_sig=goal_sig)
            cand_ids = {c.cell_id for c in prog_cells}
            rel_map = {c.cell_id: 1.0 for c in prog_cells}
            self.raw_rag_tunnel = list(prog_cells)
            self.raw_rag_relevance_map = dict(rel_map)
            self.last_reranker_telemetry = []
            plan_res = PlanResult(
                cells=prog_cells,
                candidate_ids=cand_ids,
                requested_route="PROGRAM_0",
                effective_route="PROGRAM_0",
                fallback_reason=prog_err,
                route_trace=tuple([{"event": "program_0_parse", "ok": bool(prog_cells), "error": prog_err}]),
                ir_literals={},
                tagging_decisions=(),
            )
            self.last_requested_route = "PROGRAM_0"
            self.last_effective_route = "PROGRAM_0"
            self.last_fallback_reason = prog_err
            self.last_route_trace = list(plan_res.route_trace)
            self.last_tagging_decisions = []
            if not prog_cells and prog_err:
                logger.warning(f"[ROUTER/P0] Strict formal Program_0 rejected prompt '{prompt}': {prog_err}")
            return (plan_res, cand_ids) if return_tuple else plan_res
        tunnel_cells, relevance_map = self.route(prompt, top_k=top_k)
        # A caller-supplied relevance_map arrives inside **kwargs; if it were
        # re-expanded into method.plan(...) below it would collide with the
        # explicit `relevance_map` argument (TypeError: got multiple values),
        # which the dispatch try/except silently converts into an M0 fallback.
        # Merge the caller's scores instead of crashing (external retrieval
        # signals may only raise, never lower, tunnel scores).
        caller_relevance_map = kwargs.pop("relevance_map", None)
        if caller_relevance_map:
            for _cid, _score in caller_relevance_map.items():
                relevance_map[_cid] = max(relevance_map.get(_cid, 0.0), _score)
        # Dispatch to selected RouteMethod
        method_name = str(route_method or self.default_route_method or "m0").strip().lower()
        if not tunnel_cells:
            logger.warning(f"[ROUTER] Empty tunnel for prompt: '{prompt}'")
            empty_res = PlanResult(
                cells=[],
                candidate_ids=set(),
                requested_route=method_name.upper(),
                effective_route=method_name.upper(),
                fallback_reason="Empty tunnel",
                route_trace=(),
                ir_literals=dict(getattr(self, "last_ir_literals", None) or {}),
                tagging_decisions=(),
            )
            return (empty_res, set()) if return_tuple else empty_res

        # Bound the planning trellis: the highest-likelihood tunnel prefix is
        # searched exhaustively; deeper tail cells remain reachable through
        # carrier-completion augmentation inside the planner.
        if top_k and len(tunnel_cells) > top_k:
            tunnel_cells = tunnel_cells[:top_k]

        requested_route = method_name.upper()
        effective_route = method_name.upper()
        fallback_reason: Optional[str] = None
        route_trace: List[Dict[str, Any]] = []
        tagging_decisions: List[Dict[str, Any]] = []

        clause_sem_scores = getattr(self, "last_clause_semantic_scores", None) or None

        # Registry-driven dispatch decision (no alias duplication here):
        # the M0 trellis planner is the in-router baseline; everything else
        # delegates to its registered RouteMethod class.
        from route_methods import ROUTE_METHOD_REGISTRY as _ROUTE_REGISTRY
        _method_cls = _ROUTE_REGISTRY.get(method_name)
        if _method_cls is not None:
            try:
                method = get_route_method(method_name, orchestrator=self.orchestrator)
                _method_kwargs = dict(kwargs)
                for _coalg_kw in ("max_depth", "recursion_depth", "epsilon", "theta_closed"):
                    _method_kwargs.pop(_coalg_kw, None)
                _rag_arg = _method_kwargs.pop("rag", self.internal_rag)
                if "clause_semantic_scores" not in _method_kwargs and clause_sem_scores is not None:
                    _method_kwargs["clause_semantic_scores"] = clause_sem_scores
                path = method.plan(
                    prompt=prompt,
                    tunnel=tunnel_cells,
                    relevance_map=relevance_map,
                    orchestrator=self.orchestrator,
                    ctx=ctx,
                    start_sig=start_sig,
                    goal_sig=goal_sig,
                    rag=_rag_arg,
                    **_method_kwargs
                )
                effective_route = (getattr(method, "effective_method", None) or method_name).upper()
                if getattr(method, "fallback_reason", None):
                    fallback_reason = f"{method_name.upper()} -> {effective_route}: {method.fallback_reason}"
                    logger.warning(f"[ROUTER] Route method fallback detected: {fallback_reason}")
                route_trace = list(getattr(method, "trace_events", []) or [])
            except Exception as e:
                logger.error(
                    f"[ROUTER] RouteMethod '{method_name}' failed: {e}",
                    exc_info=True,
                )
                effective_route = method_name.upper()
                fallback_reason = f"{type(e).__name__}: {str(e)}"
                path = []
        else:
            # Unknown route method name falls back to Trellis baseline
            effective_route = "M0"
            fallback_reason = f"Unknown route method '{method_name}'"
            m0 = get_route_method("m0", orchestrator=self.orchestrator)
            path = m0.plan(
                prompt=prompt,
                tunnel=tunnel_cells,
                relevance_map=relevance_map,
                orchestrator=self.orchestrator,
                ctx=ctx,
                start_sig=start_sig,
                goal_sig=goal_sig,
                clause_semantic_scores=clause_sem_scores,
            )

        # ------------------------------------------------------------------ #
        # Recursive Coalgebraic Goal Unfolding (Epistemic Grounding + Fixpoint)
        # ------------------------------------------------------------------ #
        residual = 1.0
        if path:
            try:
                try:
                    from token_evidence import audit as _audit
                except (ImportError, ValueError):
                    from .token_evidence import audit as _audit
                audit_res = _audit(path, prompt, self.orchestrator, clause_semantic_scores=clause_sem_scores)
                if audit_res.get("available"):
                    clauses_info = audit_res.get("clauses", [])
                    if clauses_info:
                        served_count = sum(1 for c in clauses_info if c.get("status") == "served")
                        coverage_frac = served_count / len(clauses_info)
                        residual = max(0.0, 1.0 - coverage_frac)
                    else:
                        residual = 0.0
                else:
                    residual = 0.0
            except Exception:
                residual = 0.0 if path else 1.0
        else:
            residual = 1.0

        recursion_depth = kwargs.get("recursion_depth", 0)
        max_depth = kwargs.get("max_depth", _cfg("macro_router_max_depth", 3))
        epsilon = kwargs.get("epsilon", 0.25)
        theta_closed = kwargs.get("theta_closed", 0.55)

        kappa = 1.0
        if recursion_depth < max_depth:
            try:
                try:
                    from coalgebraic_unfolder import EpistemicGroundingScorer, CoalgebraicUnfolder
                except (ImportError, ValueError):
                    from .coalgebraic_unfolder import EpistemicGroundingScorer, CoalgebraicUnfolder
                kappa = EpistemicGroundingScorer(theta_closed=theta_closed).score(
                    prompt, self.orchestrator, tunnel=tunnel_cells, relevance_map=relevance_map
                )
            except Exception as _ku_err:
                logger.debug(f"[ROUTER] Epistemic grounding scorer skipped: {_ku_err}")
                kappa = 1.0 if path else 0.0

        has_unexpanded_macros = False
        try:
            try:
                from coalgebraic_unfolder import is_macro_combinator_cell
            except (ImportError, ValueError):
                from .coalgebraic_unfolder import is_macro_combinator_cell
            has_unexpanded_macros = bool(
                path and any(
                    is_macro_combinator_cell(c) and not getattr(c, "bound_slots", None)
                    for c in path
                )
            )
        except Exception:
            has_unexpanded_macros = False

        # Unified Fixpoint Rule:
        # Unfold via recursive coalgebraic anamorphism psi_C(g) whenever epistemic grounding
        # kappa(g, C) < theta_closed, residual R(g, pi) > epsilon, or the path contains
        # unexpanded macro combinators needing subgoal slot decomposition (bounded by max_depth).
        if (residual > epsilon or kappa < theta_closed or has_unexpanded_macros) and recursion_depth < max_depth:
            try:
                try:
                    from inference import ModelManager
                except (ImportError, ValueError):
                    from .inference import ModelManager
                mm = ModelManager.get_instance()
                has_generative = (
                    mm is not None
                    and mm.profile is not None
                    and mm.can_synthesize()
                )
            except Exception:
                has_generative = False
                mm = None

            if not has_generative:
                # When generative synthesis is disabled, psi(g) is a null operator (empty set).
                # Preserves scientific negative-control integrity.
                route_trace.append({
                    "event": "coalgebraic_unfolding_null",
                    "residual": residual,
                    "kappa": round(float(kappa), 4),
                    "reason": "Generative synthesis disabled (null decomposition operator)",
                })
            else:
                try:
                    try:
                        from coalgebraic_unfolder import CoalgebraicUnfolder
                    except (ImportError, ValueError):
                        from .coalgebraic_unfolder import CoalgebraicUnfolder
                    unfolder = CoalgebraicUnfolder(
                        theta_closed=theta_closed,
                        max_depth=max(1, int(max_depth) - int(recursion_depth)),
                    )
                    unfold_res = unfolder.unfold(
                        prompt,
                        orchestrator=self.orchestrator,
                        model_mgr=mm,
                        tunnel=tunnel_cells,
                        relevance_map=relevance_map,
                        force_root_unfold=has_unexpanded_macros,
                    )
                    if (
                        unfold_res.root.kind != "atomic"
                        or len(unfold_res.flat_leaf_goals) > 1
                        or bool(unfold_res.used_combinators)
                    ):
                        logger.info(
                            f"[ROUTER] Coalgebraic unfolding: leaves={len(unfold_res.flat_leaf_goals)}, "
                            f"combinators={unfold_res.used_combinators}, max_depth={unfold_res.max_depth_reached}"
                        )
                        sub_kwargs = dict(kwargs)
                        sub_kwargs["recursion_depth"] = max_depth
                        sub_kwargs["max_depth"] = max_depth

                        def _route_unfolded_leaf(sub_g: str, cur_sig: Optional[Any], initial_gamma: Optional[Tuple[Any, ...]] = None) -> List[Cell]:
                            call_kwargs = dict(sub_kwargs)
                            if initial_gamma is not None:
                                call_kwargs["initial_gamma"] = initial_gamma
                            sub_res = self.plan_path(
                                sub_g,
                                start_sig=cur_sig,
                                return_tuple=False,
                                top_k=top_k,
                                route_method=route_method,
                                ctx=ctx,
                                **call_kwargs,
                            )
                            return list(sub_res.cells if hasattr(sub_res, "cells") else sub_res)

                        bridge_hops = _cfg("bridge_chain_max_hops", 2)
                        composed_path = unfolder.ground_tree(
                            unfold_res.root,
                            orchestrator=self.orchestrator,
                            route_fn=_route_unfolded_leaf,
                            start_sig=start_sig,
                            bridge_max_hops=bridge_hops,
                        )
                        if composed_path:
                            path = composed_path
                            effective_route = f"{effective_route}+COALG"
                            route_trace.append({
                                "event": "coalgebraic_unfolding_success",
                                "kappa": round(float(kappa), 4),
                                "residual_before": residual,
                                "subgoals": unfold_res.flat_leaf_goals,
                                "used_combinators": unfold_res.used_combinators,
                                "max_depth_reached": unfold_res.max_depth_reached,
                                "composed_len": len(composed_path),
                                "unfold_tree": unfold_res.root.to_dict(),
                            })
                except Exception as _cu_err:
                    logger.debug(f"[ROUTER] Coalgebraic unfolding skipped: {_cu_err}")

        if path:
            # 1. Multiplicity expansion: distribute morphisms over declared prompt identifier groups
            try:
                path = self.planner._expand_identifier_multiplicity(path, prompt)
            except Exception as _em_err:
                logger.debug(f"[ROUTER] Multiplicity expansion skipped: {_em_err}")

            # 2. Universal clause tagging (metadata-only for clause-scoped literal binding;
            # never overrides or mutates the route method's chosen cells or bridges).
            if effective_route != "M0":
                try:
                    from route_methods.base import RouteMethod as _RM
                    tagging_decisions = _RM.tag_cells_with_clause_indices(
                        path, prompt, orchestrator=self.orchestrator,
                        clause_semantic_scores=clause_sem_scores)
                except Exception as _tc_err:
                    logger.warning(f"[ROUTER] Clause tagging FAILED ({type(_tc_err).__name__}): {_tc_err}")
                    tagging_decisions = []

            # 2b. Autonomous Semantic Adapter Injection under --llm-feedback
            if os.environ.get("NSTL_LLM_FEEDBACK") == "1" or getattr(ctx, "llm_feedback", False):
                try:
                    path = self._apply_llm_semantic_adapters(path, prompt)
                except Exception as _adapt_err:
                    logger.debug(f"[ROUTER] LLM semantic adapter injection skipped: {_adapt_err}")

            # 3. Sub-lattice recursive planning for cells with slots (control-flow, macros)
            try:
                from unification import Substitution
                for cell in path:
                    if getattr(cell, "slots", None):
                        for slot_name, slot_contract in getattr(cell, "slots", {}).items():
                            if slot_name not in getattr(cell, "bound_slots", {}):
                                sub_plan = self.planner.plan_sublattice(
                                    cell, slot_name, slot_contract, tunnel_cells, relevance_map, Substitution(), prompt
                                )
                                if sub_plan:
                                    cell.bound_slots[slot_name] = sub_plan
            except Exception as _sl_err:
                logger.debug(f"[ROUTER] Sublattice slot planning skipped: {_sl_err}")

        # Telemetry: planner and orchestrator events merge into route_trace
        if self.orchestrator and hasattr(self.orchestrator, "trace_events"):
            route_trace = list(route_trace) + list(self.orchestrator.trace_events)
            self.orchestrator.trace_events = []
        if self.planner and hasattr(self.planner, "trace_events"):
            route_trace = list(route_trace) + list(self.planner.trace_events)
            self.planner.trace_events = []

        cand_ids = set(c.cell_id for c in tunnel_cells) if tunnel_cells else set()
        plan_result = PlanResult(
            cells=path,
            candidate_ids=cand_ids,
            requested_route=requested_route,
            effective_route=effective_route,
            fallback_reason=fallback_reason,
            route_trace=tuple(route_trace),
            ir_literals=dict(getattr(self, "last_ir_literals", None) or {}),
            tagging_decisions=tuple(tagging_decisions),
        )

        # Retain instance attributes for backward compatibility
        self.last_requested_route = requested_route
        self.last_effective_route = effective_route
        self.last_fallback_reason = fallback_reason
        self.last_route_trace = list(route_trace)
        self.last_tagging_decisions = list(tagging_decisions)

        if return_tuple:
            return plan_result, cand_ids
        return plan_result

    def _apply_llm_semantic_adapters(self, path: List[Cell], prompt: str) -> List[Cell]:
        """
        Autonomous semantic adapter injection protocol under --llm-feedback.
        When adjacent nodes have an explicit typestate/format conflict across domains,
        prompts the LLM to suggest an intermediate bridging adapter from the loaded lattice.
        Enforces strict category-theoretic compositionality (step_unifies) and acyclicity.
        """
        if len(path) < 2:
            return path
        try:
            from inference import ModelManager
        except (ImportError, ValueError):
            from .inference import ModelManager
        mm = ModelManager.get_instance()
        if not (mm.profile and (mm.can_synthesize() or mm.can_feedback_check())):
            return path

        def _can_compose(c1: Cell, c2: Cell, prior_path: List[Cell]) -> bool:
            if getattr(c2, "stage", None) == 1:
                return False
            chain = prior_path if prior_path else [c1]
            if hasattr(self, "planner") and self.planner:
                try:
                    from unification import Substitution, unify_cell_with_scope
                    sigma = self.planner._verify_transition(chain, c2, Substitution())
                    if sigma is not None:
                        return True
                    if unify_cell_with_scope(c2, chain, Substitution()) is not None:
                        return True
                except Exception:
                    pass
            try:
                from unification import unify
                req_inputs = [p for p in c2.inputs.values() if getattr(p, "required", True)]
                cand_inputs = req_inputs if req_inputs else list(c2.inputs.values())
                for out_port in c1.outputs.values():
                    for in_port in cand_inputs:
                        if unify(out_port.signature, in_port.signature) is not None:
                            return True
            except Exception:
                pass
            return False

        reg = TypeRegistry.get_instance()
        new_path: List[Cell] = []
        for i in range(len(path)):
            new_path.append(path[i])
            if i + 1 < len(path):
                curr_cell = path[i]
                next_cell = path[i + 1]
                curr_domain = getattr(curr_cell, "domain_name", "") or ""
                next_domain = getattr(next_cell, "domain_name", "") or ""

                # Evaluate cross-domain transitions or format shifts
                if curr_domain and next_domain and curr_domain != next_domain:
                    # If direct transition already composes cleanly without typestate conflict, bypass adapter query
                    direct_valid = _can_compose(curr_cell, next_cell, new_path)
                    has_state_mismatch = False
                    if direct_valid:
                        for out_p in curr_cell.outputs.values():
                            for in_p in next_cell.inputs.values():
                                o_st = str(getattr(out_p, "state", "") or "").lower()
                                i_st = str(getattr(in_p, "state", "") or "").lower()
                                if (
                                    o_st
                                    and i_st
                                    and o_st != i_st
                                    and o_st != "generic"
                                    and i_st != "generic"
                                    and not reg.is_declared_top(o_st)
                                    and not reg.is_declared_top(i_st)
                                ):
                                    acc = [str(s).lower() for s in getattr(in_p, "accepted_states", []) or []]
                                    if acc and o_st not in acc:
                                        has_state_mismatch = True
                                        break
                        if not has_state_mismatch:
                            continue

                    query = (
                        f"User Prompt: {prompt}\n"
                        f"Candidate Pipeline Transition:\n"
                        f"Producer: {curr_cell.cell_id} (domain: {curr_domain}, outputs: {[p.type_name for p in curr_cell.outputs.values()]})\n"
                        f"Consumer: {next_cell.cell_id} (domain: {next_domain}, inputs: {[p.type_name for p in next_cell.inputs.values()]})\n"
                        f"Question: Does this transition require an intermediate semantic adapter or format conversion (e.g. BGR to RGB)?\n"
                        f"If yes, return JSON: {{\"adapter_cell\": \"<cell_id>\"}} or {{\"adapter_cell\": null}} if direct connection is correct."
                    )
                    try:
                        resp = mm.generate_text(
                            query,
                            max_tokens=128,
                            system_prompt="You are a neuro-symbolic pipeline optimizer. Recommend semantic adapters only when strictly necessary."
                        )
                        if resp and "adapter_cell" in resp:
                            import json
                            cleaned = resp.strip()
                            if "```json" in cleaned:
                                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
                            elif "```" in cleaned:
                                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
                            data = json.loads(cleaned)
                            adapter_id = data.get("adapter_cell")
                            if adapter_id and adapter_id in self.orchestrator.loaded_cells:
                                # 1. Identity & duplicate guard
                                if adapter_id == curr_cell.cell_id or adapter_id == next_cell.cell_id:
                                    continue
                                # 2. Acyclic graph guard: adapter must not already be in the pipeline
                                if any(c.cell_id == adapter_id for c in path) or any(c.cell_id == adapter_id for c in new_path):
                                    continue
                                adapter_cell = self.orchestrator.loaded_cells[adapter_id].clone()
                                adapter_cell.matched_clause_idx = getattr(curr_cell, "matched_clause_idx", None)

                                # 3. Compositionality / Unification guard: adapter must unify with both boundaries
                                if _can_compose(curr_cell, adapter_cell, new_path) and _can_compose(adapter_cell, next_cell, new_path + [adapter_cell]):
                                    new_path.append(adapter_cell)
                                    logger.info(f"[ROUTER] LLM feedback injected verified semantic adapter '{adapter_id}' between '{curr_cell.cell_id}' and '{next_cell.cell_id}'")
                    except Exception as _e:
                        logger.debug(f"[ROUTER] LLM adapter query failed for transition {curr_cell.cell_id}->{next_cell.cell_id}: {_e}")
        return new_path

    def _decompose_goal_coalgebraic(
        self,
        prompt: str,
        orchestrator: Any,
        model_mgr: Any,
        tunnel: Optional[List[Any]] = None,
        relevance_map: Optional[Dict[str, float]] = None,
        max_depth: Optional[int] = None,
    ) -> List[str]:
        """
        Subgoal Decomposition Operator psi_C(g) -> [g_1, ..., g_m]:
        Delegates to CoalgebraicUnfolder with dynamic domain-agnostic lattice
        schema extraction and bounded recursive unfolding.
        """
        if not model_mgr or not orchestrator:
            return []
        try:
            try:
                from coalgebraic_unfolder import CoalgebraicUnfolder
            except (ImportError, ValueError):
                from .coalgebraic_unfolder import CoalgebraicUnfolder
            depth = _cfg("macro_router_max_depth", 3) if max_depth is None else max_depth
            unfolder = CoalgebraicUnfolder(max_depth=depth)
            res = unfolder.unfold(
                prompt,
                orchestrator=orchestrator,
                model_mgr=model_mgr,
                tunnel=tunnel,
                relevance_map=relevance_map,
                force_root_unfold=True,
            )
            return list(res.flat_leaf_goals)
        except Exception as e:
            logger.debug(f"[ROUTER] Coalgebraic decomposition error: {e}")
        return []


class HardwareProfiler:
    """Device configuration profiler for inference accelerators."""
    _cached_device: Optional[str] = None
    _config: Dict[str, str] = {
        'embedder': 'auto',
        'llm': 'auto',
        'trees': 'ram'
    }

    @classmethod
    def set_config(cls, embedder_device: str = "auto", llm_device: str = "auto", trees_storage: str = "ram"):
        cls._config['embedder'] = (embedder_device or 'auto').lower()
        cls._config['llm'] = (llm_device or 'auto').lower()
        cls._config['trees'] = (trees_storage or 'ram').lower()

    @classmethod
    def get_embedder_device(cls) -> str:
        if cls._config.get('embedder', 'auto') != 'auto':
            return cls._config['embedder']
        return cls.get_optimal_device()

    @classmethod
    def get_llm_device(cls) -> str:
        if cls._config.get('llm', 'auto') != 'auto':
            return cls._config['llm']
        return cls.get_optimal_device()

    @classmethod
    def get_optimal_device(cls) -> str:
        if cls._cached_device is not None:
            return cls._cached_device
        try:
            import torch
            if torch.cuda.is_available():
                cls._cached_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                cls._cached_device = "mps"
            else:
                cls._cached_device = "cpu"
        except Exception:
            cls._cached_device = "cpu"
        return cls._cached_device


class MCTSEngine:
    """
    Gap-bridging proxy kept for backward compatibility with older tests. It is NOT Monte Carlo: it tries a
    1-step direct transition, then a 2-step BFS. Nothing in the planner calls it.
    """
    def __init__(self, orchestrator: LatticeOrchestrator):
        self.orchestrator = orchestrator
        self.planner = LatticePlanner(orchestrator)

    def search(self, arg1: Any, arg2: Any = None) -> Optional[List[Cell]]:
        try:
            from lattice import AlgebraicSignature
        except (ImportError, ValueError):
            from .lattice import AlgebraicSignature

        if isinstance(arg1, AlgebraicSignature) and isinstance(arg2, AlgebraicSignature):
            start_sig, goal_sig = arg1, arg2
            # 1-step direct transition:
            for cell in self.orchestrator.cells:
                if cell.primary_input.unifies_with(start_sig) and cell.primary_output.unifies_with(goal_sig):
                    return [cell]
            # 2-step BFS:
            for c1 in self.orchestrator.cells:
                if c1.primary_input.unifies_with(start_sig):
                    for c2 in self.orchestrator.cells:
                        if c2.cell_id == c1.cell_id:
                            continue  # no self-composition: a cell cannot consume its own output
                        if c1.primary_output.unifies_with(c2.primary_input) and c2.primary_output.unifies_with(goal_sig):
                            return [c1, c2]
            return []

        if isinstance(arg1, list) and isinstance(arg2, dict):
            return self.planner._monte_carlo_chain_search("", arg1, arg2)
        if isinstance(arg1, str) and isinstance(arg2, list):
            return self.planner._monte_carlo_chain_search(arg1, arg2, {})
        return []


def log_coverage_gap(prompt: str, domain_guess: str = "", score: float = 0.0, node_id: str = ""):
    """Telemetry logging for below-threshold query gaps."""
    import os
    import json
    import time
    os.makedirs("logs", exist_ok=True)
    log_file = os.path.join("logs", "coverage_gaps.log")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "prompt": prompt,
            "domain_guess": domain_guess,
            "score": score,
            "node_id": node_id,
            "timestamp": time.time()
        }) + "\n")


class SemanticRouteOptimizer:
    """
    Route optimizer wrapping LatticeRouter for profiling and benchmarking across
    methods and profiles.
    """
    def __init__(
        self,
        orchestrator: LatticeOrchestrator,
        profile: Optional[Any] = None,
        use_reranker: bool = False,
        **kwargs
    ):
        self.orchestrator = orchestrator
        self.profile = profile
        if profile is not None:
            p_name = profile.value if hasattr(profile, "value") else str(profile)
            try:
                from inference import ModelManager
                mm = ModelManager.get_instance()
                if getattr(mm, "current_profile_name", None) != p_name.upper():
                    mm.initialize_profile(p_name)
                if not hasattr(orchestrator, "rag") or orchestrator.rag is None:
                    from internal_rag import LocalRAG
                    orchestrator.rag = LocalRAG(trees_dir="trees", orchestrator=orchestrator)
            except Exception as e:
                logger.warning(f"[SemanticRouteOptimizer] Failed to initialize profile {p_name}: {e}")

        self.router = LatticeRouter(
            orchestrator=self.orchestrator,
            internal_rag=getattr(orchestrator, "rag", None),
            use_reranker=use_reranker,
            **kwargs
        )

    def route(
        self,
        prompt: str,
        method_name: str = "m0",
        relevance_map: Optional[Dict[str, float]] = None,
        ctx: Optional[Any] = None,
        **kwargs
    ) -> List[Cell]:
        if ctx is None:
            try:
                from unification import ExecutionContext
                ctx = ExecutionContext(prompt=prompt)
            except Exception:
                ctx = None
        res = self.router.plan_path(
            prompt=prompt,
            route_method=method_name,
            return_tuple=False,
            ctx=ctx,
            relevance_map=relevance_map,
            **kwargs
        )
        return res.cells if isinstance(res, PlanResult) else (res if isinstance(res, list) else [])

    @property
    def last_effective_route(self) -> str:
        return getattr(self.router, "last_effective_route", "M0")

    @property
    def last_fallback_reason(self) -> Optional[str]:
        return getattr(self.router, "last_fallback_reason", None)

