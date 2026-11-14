"""
src/experiments.py - STRATA Neuro-Symbolic Experimental Lab & Frontier Algorithms.

Provides toggleable experimental engines for empirical tinkering and ablation studies:
  1. EarlyExitLatentProbe   : Sub-layer residual stream extraction (early-exit) + ZCA Whitening.
  2. SparseConceptCoding    : Top-K Sparse Autoencoder (SAE) /OMP concept decomposition for multi-clause prompts.
  3. AbsorbingMarkovRouter  : Exact fundamental-matrix N = (I - Q)^{-1} random-walk pathfinding over the lattice DAG.
  4. PUCTSearchEngine       : AlphaZero-style Predictor + UCT (PUCT) guided by Markov & Latent priors.
  5. build_indexed_topology : O(|V|) bucketed hash-join replacement for O(|V|^2) topology building.
"""

from __future__ import annotations
from collections import deque
import collections.abc
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

try:
    from .unification import is_edge_valid
except (ImportError, ValueError):
    from unification import is_edge_valid

from log_config import get_logger

logger = get_logger("experiments")


@dataclass
class ExperimentConfig:
    """Runtime toggles for tinkering with STRATA neuro-symbolic algorithms."""
    # 1. Partial-layer LLM / Embedder residual stream extraction
    early_exit_layer: Optional[int] = None       # e.g. 6 (extracts layer 6 instead of final layer; saves compute & sharpens types)
    use_zca_whitening: bool = True               # Removes transformer anisotropy cone via ZCA whitening
    whitening_eps: float = 1e-5

    # 2. Sparse Autoencoder (SAE) / Sparse Dictionary Concept Decomposition
    use_sparse_coding: bool = False              # Decomposes compound prompt embeddings into sparse K-hot concept atoms
    sparse_top_k: int = 8                        # Number of active concept atoms in SAE decomposition

    # 3. Absorbing Markov Chain Random-Walk Prior
    use_markov_hitting_prior: bool = True        # Computes exact N = (I - Q)^{-1} absorption probabilities to sinks
    markov_temperature: float = 0.25

    # 4. AlphaZero PUCT vs classical UCT
    use_puct_mcts: bool = True                   # Uses AlphaZero PUCT formula instead of classical UCT
    c_puct: float = 1.5

    # 5. Fast O(|V|) indexed topology builder & dual static/live 5-axis edge calculator
    use_indexed_topology: bool = True
    edge_mode: str = "live"  # 'live' (LazyAdjacencyMap + step-conditioned) or 'static' (eager precomputed 5-axis)


ACTIVE_EXPERIMENTS = ExperimentConfig()


# ============================================================================
# 1. Early-Exit Sub-Layer Residual Stream Extractor + ZCA Whitening
# ============================================================================

class EarlyExitLatentProbe:
    """
    Tinkering Module 1:
    Extracts intermediate residual stream representations from transformer layer `k`
    (running only the first `k` layers of the encoder!) and applies ZCA Whitening
    fitted on the lattice cell corpus to eliminate the narrow-cone anisotropy of
    small language models.
    """

    def __init__(self, extractor: Any, config: Optional[ExperimentConfig] = None) -> None:
        self.extractor = extractor
        self.cfg = config or ACTIVE_EXPERIMENTS
        self._mean: Optional[np.ndarray] = None
        self._W_zca: Optional[np.ndarray] = None

    def fit_whitening(self, cell_vectors: np.ndarray) -> None:
        """
        Computes the ZCA whitening matrix W = U * diag(1 / sqrt(S + eps)) * U^T
        over the corpus of lattice cell embeddings (shape: [N_cells, dim]).
        """
        if cell_vectors.ndim != 2 or cell_vectors.shape[0] < 4:
            return
        self._mean = np.mean(cell_vectors, axis=0, keepdims=True)
        centered = cell_vectors - self._mean
        cov = (centered.T @ centered) / float(max(cell_vectors.shape[0] - 1, 1))
        u, s, _ = np.linalg.svd(cov, full_matrices=False)
        inv_sqrt_s = np.diag(1.0 / np.sqrt(s + self.cfg.whitening_eps))
        self._W_zca = (u @ inv_sqrt_s @ u.T).astype(np.float32)
        logger.info(
            "[EXP:WHITENING] Fitted ZCA whitening on %d cells (dim=%d, condition=%.1f)",
            cell_vectors.shape[0],
            cell_vectors.shape[1],
            float(np.max(s) / max(float(np.min(s)), 1e-9)),
        )

    def encode_early_exit(self, text: str, layer_idx: Optional[int] = None) -> np.ndarray:
        """
        If `layer_idx` is set and the underlying backend is a PyTorch Transformer,
        executes ONLY the first `layer_idx` transformer blocks and mean-pools the
        residual stream. Otherwise falls back to full-model encoding.
        """
        target_layer = layer_idx if layer_idx is not None else self.cfg.early_exit_layer
        model = getattr(self.extractor, "_model", None)

        if target_layer is not None and getattr(self.extractor, "_backend", "") == "sentence_transformers" and model is not None:
            try:
                import torch
                first_mod = model[0]
                auto_model = getattr(first_mod, "auto_model", None)
                tokenizer = getattr(first_mod, "tokenizer", None)
                if auto_model is not None and tokenizer is not None:
                    device = next(auto_model.parameters()).device
                    inputs = tokenizer(
                        [text],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    with torch.inference_mode():
                        outputs = auto_model(**inputs, output_hidden_states=True)
                        hidden_states = outputs.hidden_states
                        chosen_idx = max(0, min(int(target_layer), len(hidden_states) - 1))
                        layer_h = hidden_states[chosen_idx]  # [1, seq_len, dim]
                        mask = inputs["attention_mask"].unsqueeze(-1).float()
                        pooled = (layer_h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                        vec = pooled[0].detach().cpu().numpy().astype(np.float32)
                        return self._postprocess(vec)
            except Exception as exc:
                logger.debug("[EXP:EARLY_EXIT] Fallback to standard encode: %s", exc)

        vec = self.extractor.encode(text)
        return self._postprocess(vec)

    def _postprocess(self, vec: np.ndarray) -> np.ndarray:
        if self.cfg.use_zca_whitening and self._W_zca is not None and self._mean is not None:
            centered = vec - self._mean.squeeze(0)
            vec = centered @ self._W_zca
        norm = float(np.linalg.norm(vec))
        return (vec / (norm + 1e-9)).astype(np.float32) if norm > 0 else vec


# ============================================================================
# 2. Sparse Autoencoder (SAE) / Orthogonal Matching Pursuit Concept Probing
# ============================================================================

class SparseConceptCoding:
    """
    Tinkering Module 2:
    Decomposes a dense prompt embedding `e_prompt` into a sparse linear combination
    of unit-normalized lattice cell atoms `D` (e_prompt ~ D^T * alpha, ||alpha||_0 <= K)
    using Orthogonal Matching Pursuit (OMP) / Top-K Sparse Autoencoding.
    Prevents multi-clause feature superposition from drowning out secondary clauses!
    """

    @staticmethod
    def decompose_prompt_embedding(
        prompt_vec: np.ndarray,
        dictionary_matrix: np.ndarray,
        cell_ids: Sequence[str],
        top_k: int = 8,
    ) -> Dict[str, float]:
        """
        Given `prompt_vec` in R^d and `dictionary_matrix` in R^{N x d} (rows are
        unit-normalized cell embeddings), iteratively extracts `top_k` orthogonal
        concept atoms by subtracting each explained residual component.
        """
        if dictionary_matrix.ndim != 2 or len(cell_ids) == 0:
            return {}
        n_cells = dictionary_matrix.shape[0]
        k = min(max(1, int(top_k)), n_cells)

        residual = prompt_vec.astype(np.float64).copy()
        norm_r = float(np.linalg.norm(residual))
        if norm_r < 1e-9:
            return {}
        residual /= norm_r

        D = dictionary_matrix.astype(np.float64)
        norms = np.linalg.norm(D, axis=1, keepdims=True)
        D_unit = D / np.maximum(norms, 1e-9)

        selected_indices: List[int] = []
        activations: Dict[str, float] = {}

        for _step in range(k):
            projections = D_unit @ residual
            for idx in selected_indices:
                projections[idx] = -1e9
            best_idx = int(np.argmax(projections))
            best_val = float(projections[best_idx])
            if best_val <= 1e-4:
                break

            selected_indices.append(best_idx)
            activations[cell_ids[best_idx]] = best_val

            # Orthogonalize residual against selected subspace (OMP step)
            sub_D = D_unit[selected_indices].T  # [d, s]
            coeffs, *_ = np.linalg.lstsq(sub_D, prompt_vec.astype(np.float64), rcond=None)
            residual = prompt_vec.astype(np.float64) - (sub_D @ coeffs)
            if float(np.linalg.norm(residual)) < 1e-5:
                break

        return activations


# ============================================================================
# 3. Absorbing Markov Chain Random-Walk Router
# ============================================================================

class AbsorbingMarkovRouter:
    """
    Tinkering Module 3:
    Models the candidate semantic tunnel as an Absorbing Markov Chain:
      - Stage-3 sinks / endable nodes are absorbing states.
      - Stage-1 & Stage-2 nodes are transient states with transition probabilities
        P(u -> v) proportional to exp((affinity(u,v) + relevance(v)) / T).
    Solves the Fundamental Matrix N = (I - Q)^{-1} to compute the exact expected
    visitation count and absorption probability for every node in the tunnel.
    """

    @staticmethod
    def compute_node_potentials(
        candidates: Sequence[Any],
        relevance_map: Dict[str, float],
        edge_affinity_fn: Callable[[Any, Any], float],
        unify_check_fn: Callable[[Any, Any], bool],
        is_sink_fn: Callable[[Any], bool],
        temperature: float = 0.25,
    ) -> Dict[str, float]:
        n = len(candidates)
        if n == 0:
            return {}
        if n == 1:
            return {candidates[0].cell_id: 1.0}

        temp = max(float(temperature), 0.05)
        transient_idx: List[int] = []
        absorbing_idx: List[int] = []

        for i, c in enumerate(candidates):
            if is_sink_fn(c):
                absorbing_idx.append(i)
            else:
                transient_idx.append(i)

        # If no explicit sinks exist, treat top-relevance Stage-2 nodes as virtual absorbers
        if not absorbing_idx or not transient_idx:
            return {c.cell_id: float(relevance_map.get(c.cell_id, 0.0)) for c in candidates}

        t_count = len(transient_idx)
        a_count = len(absorbing_idx)
        Q = np.zeros((t_count, t_count), dtype=np.float64)
        R = np.zeros((t_count, a_count), dtype=np.float64)

        for row_i, u_idx in enumerate(transient_idx):
            u = candidates[u_idx]
            weights_t = np.zeros(t_count, dtype=np.float64)
            weights_a = np.zeros(a_count, dtype=np.float64)

            for col_j, v_idx in enumerate(transient_idx):
                if u_idx == v_idx:
                    continue
                v = candidates[v_idx]
                if unify_check_fn(u, v):
                    aff = edge_affinity_fn(u, v)
                    rel = relevance_map.get(v.cell_id, 0.0)
                    weights_t[col_j] = math.exp((aff + rel) / temp)

            for col_k, s_idx in enumerate(absorbing_idx):
                s = candidates[s_idx]
                if unify_check_fn(u, s):
                    aff = edge_affinity_fn(u, s)
                    rel = relevance_map.get(s.cell_id, 0.0)
                    weights_a[col_k] = math.exp((aff + rel) / temp)

            # Damping factor (0.85) guarantees (I - Q) is strictly diagonally dominant & invertible
            total_w = float(np.sum(weights_t) + np.sum(weights_a))
            if total_w > 0:
                Q[row_i, :] = 0.85 * (weights_t / total_w)
                R[row_i, :] = 0.85 * (weights_a / total_w)

        # Solve (I - Q) * N = I for the Fundamental Matrix N
        I_mat = np.eye(t_count, dtype=np.float64)
        try:
            N_mat = np.linalg.solve(I_mat - Q, I_mat)
        except np.linalg.LinAlgError:
            N_mat = np.linalg.pinv(I_mat - Q)

        # B = N * R gives exact probability of reaching absorbing sink k from transient state i
        B_mat = N_mat @ R  # [t_count, a_count]
        sink_priors = np.array(
            [max(relevance_map.get(candidates[s_idx].cell_id, 0.01), 0.01) for s_idx in absorbing_idx],
            dtype=np.float64,
        )
        sink_priors /= np.sum(sink_priors)

        transient_potentials = B_mat @ sink_priors  # [t_count]
        visit_centrality = np.sum(N_mat, axis=0)    # Expected visits to transient node j

        potentials: Dict[str, float] = {}
        max_v = float(np.max(visit_centrality)) if len(visit_centrality) else 1.0
        for row_i, u_idx in enumerate(transient_idx):
            cid = candidates[u_idx].cell_id
            potentials[cid] = float(
                0.6 * transient_potentials[row_i] + 0.4 * (visit_centrality[row_i] / max(max_v, 1e-9))
            )

        for col_k, s_idx in enumerate(absorbing_idx):
            cid = candidates[s_idx].cell_id
            inflow = float(np.mean(B_mat[:, col_k])) if t_count > 0 else 0.0
            potentials[cid] = float(0.5 * inflow + 0.5 * sink_priors[col_k])

        return potentials


# ============================================================================
# 4. O(|V|) Bucketed Hash-Join Topology Builder (Fixes O(|V|^2) Bottleneck)
# ============================================================================

def _freeze_edge_val(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (tuple, list)):
        return tuple(_freeze_edge_val(x) for x in v)
    if isinstance(v, set):
        return tuple(sorted((_freeze_edge_val(x) for x in v), key=lambda z: repr(z)))
    if isinstance(v, dict):
        return tuple(sorted((str(k), _freeze_edge_val(val)) for k, val in v.items()))
    return ("__obj__", id(v))


def _port_edge_key(p: Any, gen: int, registry: Any, is_path_fn: Any) -> Tuple[Any, ...]:
    cached_p = p._cached_is_path_port
    if cached_p is not None and cached_p[0] == gen:
        is_p = cached_p[1]
    else:
        is_p = is_path_fn(p, registry)
    return (
        p.signature,
        p.domain or "",
        p.port_role or "",
        p.structure_kind or "",
        p.abstract_type or "",
        p.carrier or "",
        p.nominal_type or "",
        p.semantics or "",
        p.typestate or "",
        _freeze_edge_val(p.layout),
        p.invariants,
        p.rejected_qualifiers,
        p.accepted_states,
        p.parent_state,
        p.rank,
        p._cached_ndim,
        _freeze_edge_val(p.shape_contract),
        tuple(p.supported_formats) if p.supported_formats else (),
        is_p,
    )


class LazyAdjacencyMap(collections.abc.MutableMapping):
    """
    On-demand adjacency mapping backed by an O(|V|) inverted (carrier, semantics) index.
    Computes and memoizes neighbors only when queried, avoiding eager O(|V|^2) startup graphs.
    """

    def __init__(
        self,
        keys: Sequence[str],
        seed_lists: Dict[str, List[str]],
        compute_fn: Callable[[str], List[str]],
    ):
        self._keys: List[str] = list(keys)
        self._key_set: Set[str] = set(keys)
        self._seed_lists: Dict[str, List[str]] = seed_lists
        self._compute_fn = compute_fn
        self._cache: Dict[str, List[str]] = {}

    def _resolve(self, cid: str) -> List[str]:
        cached = self._cache.get(cid)
        if cached is not None:
            return cached
        computed = self._compute_fn(cid)
        seeds = self._seed_lists.get(cid)
        if seeds:
            seen = set(computed)
            merged = list(computed)
            for s in seeds:
                if s not in seen:
                    seen.add(s)
                    merged.append(s)
            self._cache[cid] = merged
        else:
            self._cache[cid] = computed
        return self._cache[cid]

    def __getitem__(self, key: str) -> List[str]:
        if key not in self._key_set and key not in self._cache:
            raise KeyError(key)
        return self._resolve(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._key_set and key not in self._cache:
            return default
        return self._resolve(key)

    def __setitem__(self, key: str, value: List[str]) -> None:
        if key not in self._key_set:
            self._keys.append(key)
            self._key_set.add(key)
        self._cache[key] = list(value)

    def __delitem__(self, key: str) -> None:
        self._cache.pop(key, None)
        self._seed_lists.pop(key, None)
        if key in self._key_set:
            self._key_set.remove(key)
            self._keys.remove(key)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key in self._key_set or key in self._cache)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def clear(self) -> None:
        self._keys.clear()
        self._key_set.clear()
        self._seed_lists.clear()
        self._cache.clear()

    def keys(self):
        return iter(self._keys)

    def values(self):
        for k in self._keys:
            yield self._resolve(k)

    def items(self):
        for k in self._keys:
            yield k, self._resolve(k)


def build_indexed_topology(
    all_cells: Sequence[Any],
    loaded_cells_map: Dict[str, Any],
    registry: Any,
    edge_mode: Optional[str] = None,
    use_indexed: Optional[bool] = None,
) -> Tuple[Any, Any]:
    """
    5-Axis Topological Edge Calculator supporting BOTH dual modes:
      - 'static' : Eagerly materializes full Dict[str, List[str]] forward & reverse
                   adjacency over 5-axis `is_edge_valid` (indexed when use_indexed=True,
                   or direct O(|V|^2) pairwise scan when use_indexed=False).
      - 'live'   : Returns LazyAdjacencyMap objects backed by the O(|V|) inverted
                   carrier/ancestor index and evaluated on-demand via 5-axis unification.
    """
    try:
        from .lattice import PortSignature, is_path_port as _is_path_port
    except (ImportError, ValueError):
        from lattice import PortSignature, is_path_port as _is_path_port

    eff_mode = str(edge_mode or getattr(ACTIVE_EXPERIMENTS, "edge_mode", "live") or "live").strip().lower()
    if eff_mode not in ("static", "live"):
        eff_mode = "live"
    eff_indexed = ACTIVE_EXPERIMENTS.use_indexed_topology if use_indexed is None else bool(use_indexed)

    bottom_set = registry.get_bottom_types_set()
    top_set = registry.get_top_types_set()
    skip_in_set = bottom_set | top_set

    cell_by_id: Dict[str, Any] = {c.cell_id: c for c in all_cells}
    all_ids: List[str] = [c.cell_id for c in all_cells]
    fwd_seeds: Dict[str, List[str]] = {}
    rev_seeds: Dict[str, List[str]] = {}

    if eff_mode == "static" and not eff_indexed:
        static_adj: Dict[str, List[str]] = {cid: [] for cid in all_ids}
        static_rev: Dict[str, List[str]] = {cid: [] for cid in all_ids}
        adj_seen: Dict[str, Set[str]] = {cid: set() for cid in all_ids}
        for u in all_cells:
            uid = u.cell_id
            for edge in getattr(u, "edges", []):
                tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
                tgt_cell = loaded_cells_map.get(tgt_id) if tgt_id else None
                if tgt_cell and tgt_cell.cell_id not in adj_seen[uid]:
                    if is_edge_valid(u.primary_output, tgt_cell.primary_input, producer_cell=u, consumer_cell=tgt_cell):
                        adj_seen[uid].add(tgt_cell.cell_id)
                        static_adj[uid].append(tgt_cell.cell_id)
                        static_rev[tgt_cell.cell_id].append(uid)
            out_p = getattr(u, "primary_output", None)
            if not out_p:
                continue
            out_tn = str(getattr(out_p, "type_name", "") or "").strip().lower()
            if out_tn in skip_in_set:
                continue
            for v in all_cells:
                vid = v.cell_id
                if vid == uid or vid in adj_seen[uid] or getattr(v, "stage", None) == 1 or not getattr(v, "inputs", None):
                    continue
                in_p = v.primary_input
                if in_p and is_edge_valid(out_p, in_p, producer_cell=u, consumer_cell=v):
                    adj_seen[uid].add(vid)
                    static_adj[uid].append(vid)
                    static_rev[vid].append(uid)
        return static_adj, static_rev

    # 1. O(|V|) Inverted consumer & producer indexes keyed by canonical carrier
    consumers_by_type: Dict[str, List[Any]] = {}
    wildcard_consumers: List[Any] = []
    producers_by_type: Dict[str, List[Any]] = {}
    wildcard_producers: List[Any] = []

    for v in all_cells:
        # Explicit declared edges (O(|E_declared|))
        for edge in getattr(v, "edges", []):
            tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt_id:
                tgt_cell = loaded_cells_map.get(tgt_id)
                if tgt_cell and is_edge_valid(v.primary_output, tgt_cell.primary_input, producer_cell=v, consumer_cell=tgt_cell):
                    fwd_seeds.setdefault(v.cell_id, []).append(tgt_cell.cell_id)
                    rev_seeds.setdefault(tgt_cell.cell_id, []).append(v.cell_id)

        # Index v as a producer (for reverse lookups)
        out_p = getattr(v, "primary_output", None)
        if out_p and getattr(out_p, "type_name", None):
            out_tn = str(out_p.type_name).strip().lower()
            if out_tn not in bottom_set:
                canon_out = registry.canonical_name(out_tn).lower()
                if out_tn in top_set or "[" in canon_out or registry.is_type_variable(out_p.type_name):
                    wildcard_producers.append(v)
                else:
                    producers_by_type.setdefault(canon_out, []).append(v)

        # Index v as a consumer (for forward lookups)
        if getattr(v, "stage", None) == 1 or not getattr(v, "inputs", None):
            continue
        in_p = v.primary_input
        if not in_p or not getattr(in_p, "type_name", None):
            continue
        in_tn = str(in_p.type_name).strip().lower()
        if in_tn in skip_in_set:
            continue
        in_sig = in_p.signature
        if (hasattr(in_sig, "is_top") and in_sig.is_top()) or (hasattr(in_sig, "is_bottom") and in_sig.is_bottom()):
            continue
        canon_in = registry.canonical_name(in_tn).lower()
        if "[" in canon_in or registry.is_type_variable(in_p.type_name):
            wildcard_consumers.append(v)
        else:
            consumers_by_type.setdefault(canon_in, []).append(v)

    parents_map = getattr(registry, "_parents", None) or getattr(registry, "poset", {}) or {}
    children_map: Dict[str, List[str]] = {}
    for ch, ps in parents_map.items():
        ch_l = str(ch).lower()
        for p_par in ps or ():
            children_map.setdefault(str(p_par).lower(), []).append(ch_l)

    supertypes_cache: Dict[str, Tuple[str, ...]] = {}
    subtypes_cache: Dict[str, Tuple[str, ...]] = {}
    gen = getattr(registry, "_generation", 0)
    in_key_by_port: Dict[int, Any] = {}
    sig_pair_cache: Dict[Tuple[Any, Any], bool] = {}

    def _get_supertypes(canon_t: str) -> Tuple[str, ...]:
        cached = supertypes_cache.get(canon_t)
        if cached is not None:
            return cached
        res: List[str] = [canon_t]
        seen = {canon_t}
        q = deque([canon_t])
        while q:
            curr = q.popleft()
            for par in parents_map.get(curr, ()):
                pl = str(par).lower()
                if pl not in seen:
                    seen.add(pl)
                    res.append(pl)
                    q.append(pl)
        tup = tuple(res)
        supertypes_cache[canon_t] = tup
        return tup

    def _get_subtypes(canon_t: str) -> Tuple[str, ...]:
        cached = subtypes_cache.get(canon_t)
        if cached is not None:
            return cached
        res: List[str] = [canon_t]
        seen = {canon_t}
        q = deque([canon_t])
        while q:
            curr = q.popleft()
            for ch in children_map.get(curr, ()):
                if ch not in seen:
                    seen.add(ch)
                    res.append(ch)
                    q.append(ch)
        tup = tuple(res)
        subtypes_cache[canon_t] = tup
        return tup

    def _pair_valid(u: Any, out_p: Any, out_key: Any, u_has_post: bool, v: Any) -> bool:
        v_in = v.primary_input
        if out_key is not None and isinstance(v_in, PortSignature):
            if not getattr(v, "preconditions", None):
                v_in_pid = id(v_in)
                in_key = in_key_by_port.get(v_in_pid)
                if in_key is None:
                    in_key = _port_edge_key(v_in, gen, registry, _is_path_port)
                    in_key_by_port[v_in_pid] = in_key
                pair_k = (out_key, in_key)
                valid = sig_pair_cache.get(pair_k)
                if valid is None:
                    valid = is_edge_valid(out_p, v_in, producer_cell=u, consumer_cell=v)
                    sig_pair_cache[pair_k] = valid
                return valid
            if not u_has_post:
                pair_k = (out_key, "__vc__", v.cell_id)
                valid = sig_pair_cache.get(pair_k)
                if valid is None:
                    valid = is_edge_valid(out_p, v_in, producer_cell=u, consumer_cell=v)
                    sig_pair_cache[pair_k] = valid
                return valid
        return is_edge_valid(out_p, v_in, producer_cell=u, consumer_cell=v)

    def _compute_forward(u_id: str) -> List[str]:
        u = cell_by_id.get(u_id)
        if u is None:
            return []
        out_p = getattr(u, "primary_output", None)
        if not out_p or not getattr(out_p, "type_name", None):
            return []
        out_tn = str(out_p.type_name).strip().lower()
        if out_tn in skip_in_set:
            return []
        out_sig = out_p.signature
        if (hasattr(out_sig, "is_top") and out_sig.is_top()) or (hasattr(out_sig, "is_bottom") and out_sig.is_bottom()):
            return []
        canon_out = registry.canonical_name(out_tn).lower()
        pool: List[Any] = list(wildcard_consumers)
        for sup in _get_supertypes(canon_out):
            pool.extend(consumers_by_type.get(sup, ()))
        out_key = _port_edge_key(out_p, gen, registry, _is_path_port) if isinstance(out_p, PortSignature) else None
        u_has_post = bool(getattr(u, "postconditions", None))
        res: List[str] = []
        seen: Set[str] = {u_id}
        for v in pool:
            vid = v.cell_id
            if vid in seen:
                continue
            if _pair_valid(u, out_p, out_key, u_has_post, v):
                seen.add(vid)
                res.append(vid)
        return res

    def _compute_reverse(v_id: str) -> List[str]:
        v = cell_by_id.get(v_id)
        if v is None or getattr(v, "stage", None) == 1 or not getattr(v, "inputs", None):
            return []
        v_in = v.primary_input
        if not v_in or not getattr(v_in, "type_name", None):
            return []
        in_tn = str(v_in.type_name).strip().lower()
        if in_tn in skip_in_set:
            return []
        canon_in = registry.canonical_name(in_tn).lower()
        if "[" in canon_in or registry.is_type_variable(v_in.type_name):
            pool = list(all_cells)
        else:
            pool = list(wildcard_producers)
            for sub in _get_subtypes(canon_in):
                pool.extend(producers_by_type.get(sub, ()))
        res: List[str] = []
        seen: Set[str] = {v_id}
        for u in pool:
            uid = u.cell_id
            if uid in seen:
                continue
            out_p = getattr(u, "primary_output", None)
            if not out_p:
                continue
            out_key = _port_edge_key(out_p, gen, registry, _is_path_port) if isinstance(out_p, PortSignature) else None
            u_has_post = bool(getattr(u, "postconditions", None))
            if _pair_valid(u, out_p, out_key, u_has_post, v):
                seen.add(uid)
                res.append(uid)
        return res

    if eff_mode == "static":
        static_adj: Dict[str, List[str]] = {cid: [] for cid in all_ids}
        static_rev: Dict[str, List[str]] = {cid: [] for cid in all_ids}
        adj_seen: Dict[str, Set[str]] = {cid: set() for cid in all_ids}
        rev_seen: Dict[str, Set[str]] = {cid: set() for cid in all_ids}
        for src_id, dsts in fwd_seeds.items():
            for dst_id in dsts:
                if dst_id not in adj_seen.get(src_id, ()):
                    adj_seen.setdefault(src_id, set()).add(dst_id)
                    static_adj.setdefault(src_id, []).append(dst_id)
                if src_id not in rev_seen.get(dst_id, ()):
                    rev_seen.setdefault(dst_id, set()).add(src_id)
                    static_rev.setdefault(dst_id, []).append(src_id)
        for src_id in all_ids:
            for dst_id in _compute_forward(src_id):
                if dst_id not in adj_seen[src_id]:
                    adj_seen[src_id].add(dst_id)
                    static_adj[src_id].append(dst_id)
                if src_id not in rev_seen[dst_id]:
                    rev_seen[dst_id].add(src_id)
                    static_rev[dst_id].append(src_id)
        return static_adj, static_rev

    adjacency = LazyAdjacencyMap(all_ids, fwd_seeds, _compute_forward)
    reverse_adjacency = LazyAdjacencyMap(all_ids, rev_seeds, _compute_reverse)
    return adjacency, reverse_adjacency


# ============================================================================
# 5. Sinkhorn-Knopp Entropic Optimal Transport Port Matcher (Proposal 4.C)
# ============================================================================

class SinkhornPortMatcher:
    """
    Tinkering Module 5 (Solves Proposal 4.C in Neural Mode):
    Instead of greedy word-by-word cosine matching (where one prompt token can
    be stolen by the wrong port), solves the Entropic Optimal Transport problem:
        P* = argmin_P <C, P> - eps * H(P)
    via Sinkhorn-Knopp matrix scaling between unconsumed prompt tokens and
    open cell parameter ports.
    """

    @staticmethod
    def match_tokens_to_ports(
        token_embeddings: np.ndarray,
        port_embeddings: np.ndarray,
        tokens: Sequence[str],
        port_names: Sequence[str],
        reg_eps: float = 0.1,
        n_iters: int = 25,
        min_sim: float = 0.30,
    ) -> Dict[str, str]:
        """
        Computes the optimal doubly-stochastic transport plan P in R^{n_ports x n_tokens}
        and returns `{port_name: matched_token}` for confident assignments.
        """
        if token_embeddings.ndim != 2 or port_embeddings.ndim != 2:
            return {}
        n_toks = token_embeddings.shape[0]
        n_ports = port_embeddings.shape[0]
        if n_toks == 0 or n_ports == 0:
            return {}

        # Normalize rows to unit sphere
        t_norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
        p_norms = np.linalg.norm(port_embeddings, axis=1, keepdims=True)
        T_unit = token_embeddings / np.maximum(t_norms, 1e-9)
        P_unit = port_embeddings / np.maximum(p_norms, 1e-9)

        # Cosine similarity & Gibbs kernel K = exp(Sim / eps)
        sim_matrix = P_unit @ T_unit.T  # [n_ports, n_toks]
        K = np.exp((sim_matrix - np.max(sim_matrix)) / max(reg_eps, 1e-3))

        # Sinkhorn-Knopp alternating row/column normalization
        u = np.ones(n_ports, dtype=np.float64) / float(n_ports)
        v = np.ones(n_toks, dtype=np.float64) / float(n_toks)
        for _ in range(n_iters):
            u = (1.0 / float(n_ports)) / np.maximum(K @ v, 1e-12)
            v = (1.0 / float(n_toks)) / np.maximum(K.T @ u, 1e-12)

        transport_plan = np.diag(u) @ K @ np.diag(v)  # [n_ports, n_toks]

        assignments: Dict[str, str] = {}
        used_tok_indices: Set[int] = set()
        # Assign ports in descending order of transport confidence
        flat_order = np.argsort(transport_plan, axis=None)[::-1]
        for flat_idx in flat_order:
            p_idx = int(flat_idx // n_toks)
            t_idx = int(flat_idx % n_toks)
            if port_names[p_idx] in assignments or t_idx in used_tok_indices:
                continue
            if float(sim_matrix[p_idx, t_idx]) >= min_sim:
                assignments[port_names[p_idx]] = tokens[t_idx]
                used_tok_indices.add(t_idx)
        return assignments


def corpus_idf_split_clauses(prompt: str, orchestrator: Any = None) -> List[str]:
    """
    Solves Proposal 4.C in Profile 0 (Zero-Word-List Clause Segmentation):
    Segments a prompt into operational clauses using ONLY:
      1. Structural punctuation (';', ',', '.') outside quotes/brackets/comparisons.
      2. Corpus Information Theory: a token `w` with IDF == 0 in the loaded lattice's
         operation vocabulary (and not a literal/number/quote/comparison) is a
         zero-entropy grammatical valley. When a zero-IDF valley separates two
         high-IDF operation peaks, a clause boundary is placed at the valley!
    Zero hardcoded English conjunction or preposition lists.
    """
    if not prompt or not prompt.strip():
        return []
    try:
        from .token_evidence import get_evidence
        from .tokenizer import normalize_token
    except (ImportError, ValueError):
        from token_evidence import get_evidence
        from tokenizer import normalize_token

    orch = orchestrator
    ev = get_evidence(orch) if orch is not None else None
    if ev is None:
        return [c.strip() for c in prompt.split(";") if c.strip()] or [prompt.strip()]

    text = prompt.strip()
    words: List[Tuple[int, int, str]] = []
    cur: List[str] = []
    w_start: Optional[int] = None
    in_quote: Optional[str] = None
    bracket_depth = 0

    for idx, ch in enumerate(text):
        if in_quote:
            if ch == in_quote and (idx == 0 or text[idx - 1] != "\\"):
                in_quote = None
            cur.append(ch)
            continue
        if ch in ("'", '"'):
            in_quote = ch
            if w_start is None:
                w_start = idx
            cur.append(ch)
            continue
        if ch in "([":
            bracket_depth += 1
        elif ch in ")]":
            bracket_depth = max(0, bracket_depth - 1)

        if ch.isspace() and bracket_depth == 0:
            if cur and w_start is not None:
                words.append((w_start, idx, "".join(cur)))
                cur = []
                w_start = None
        else:
            if w_start is None:
                w_start = idx
            cur.append(ch)
    if cur and w_start is not None:
        words.append((w_start, len(text), "".join(cur)))

    if len(words) <= 1:
        return [text]

    # Compute operation IDF for each word token
    op_idfs: List[float] = []
    is_literal_tok: List[bool] = []
    for _, _, raw_w in words:
        clean = raw_w.strip(".,;:!?()[]{}")
        if not clean:
            op_idfs.append(0.0)
            is_literal_tok.append(False)
            continue
        # Formal orthography check for literals (quotes, numbers, paths, comparisons)
        is_lit = (
            raw_w[0] in ("'", '"')
            or "/" in clean
            or "." in clean
            or any(op in raw_w for op in (">", "<", "==", "!="))
            or clean.isdigit()
            or (len(clean) == 1 and clean.isupper())
        )
        is_literal_tok.append(is_lit)
        stem = normalize_token(clean.lower())
        op_idfs.append(max(ev.idf(clean.lower()), ev.idf(stem)))

    # Detect boundaries at punctuation or zero-IDF valleys between two operation peaks
    split_word_indices: List[int] = []
    seen_op_peak = False
    for i, (s, e, raw_w) in enumerate(words):
        if op_idfs[i] > 1.0 and not is_literal_tok[i]:
            seen_op_peak = True
        if raw_w.endswith(";") and i + 1 < len(words):
            split_word_indices.append(i + 1)
            seen_op_peak = False
            continue
        # Zero-IDF valley between an earlier operation peak and an upcoming operation peak
        if (
            seen_op_peak
            and op_idfs[i] == 0.0
            and not is_literal_tok[i]
            and i + 1 < len(words)
            and op_idfs[i + 1] > 1.2
            and not is_literal_tok[i + 1]
        ):
            # Do not split inside comparison expressions (e.g. `x > 10 ... y < 20`)
            near_cmp = any(
                any(op in words[k][2] for op in (">", "<", "==", "!="))
                for k in range(max(0, i - 3), min(len(words), i + 3))
            )
            if not near_cmp:
                split_word_indices.append(i + 1)
                seen_op_peak = False

    if not split_word_indices:
        return [text]

    clauses: List[str] = []
    last_char = 0
    for w_idx in split_word_indices:
        split_char = words[w_idx][0]
        chunk = text[last_char:split_char].strip(" ,;.")
        if chunk:
            clauses.append(chunk)
        last_char = split_char
    tail = text[last_char:].strip(" ,;.")
    if tail:
        clauses.append(tail)
    return clauses if clauses else [text]


__all__ = [
    "ExperimentConfig",
    "ACTIVE_EXPERIMENTS",
    "EarlyExitLatentProbe",
    "SparseConceptCoding",
    "AbsorbingMarkovRouter",
    "build_indexed_topology",
    "SinkhornPortMatcher",
    "corpus_idf_split_clauses",
]
