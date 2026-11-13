"""
src/latent_autoencoder.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Self-Supervised Structural Transition Autoencoder (TransE / Koopman Manifold).

Learns a continuous vector manifold where typestates tau are mapped to points z_tau in R^d
and cell morphisms c are mapped to continuous translation vectors t_c in R^d.
Transitions satisfy:
    z_{tau_in} + t_c ≈ z_{tau_out}

Trained self-supervised via contrastive negative sampling over tree edges in <2 seconds.
"""

from __future__ import annotations
import math
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Collection
import numpy as np

from log_config import get_logger

logger = get_logger("latent_autoencoder")


def _hash_u64(text: str, salt: int = 0) -> int:
    """Deterministic 64-bit FNV-1a + SplitMix64 integer hash (zero RNG, zero MD5)."""
    h = (0xCBF29CE484222325 ^ (salt * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return h ^ (h >> 31)


def _split_camel_and_punct(text: str) -> List[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [t.lower() for t in re.split(r"[^A-Za-z0-9]+", expanded) if t]


def _extract_structural_features(text: str, is_typestate: bool = True) -> List[Tuple[str, float]]:
    """
    Decomposes a typestate string, cell identifier, or phrase into weighted
    compositional structural + subword + character n-gram features.
    """
    s = (text or "").strip() or "__empty__"
    feats: List[Tuple[str, float]] = []

    if is_typestate:
        # Canonical format: BaseType[state]{q1,q2} or general phrase
        state_match = re.search(r"\[([^\]]+)\]", s)
        qual_match = re.search(r"\{([^}]+)\}", s)
        base_part = re.sub(r"\[[^\]]*\]|\{[^}]*\}", " ", s).strip()

        if base_part:
            if " " not in base_part:
                feats.append((f"base:{base_part.lower()}", 1.0))
            base_toks = _split_camel_and_punct(base_part)
            for tok in base_toks:
                feats.append((f"tok:{tok}", 1.2))
                if len(tok) >= 3:
                    for i in range(len(tok) - 2):
                        feats.append((f"tri:{tok[i:i+3]}", 0.30))
            for i in range(len(base_toks) - 1):
                feats.append((f"big:{base_toks[i]}_{base_toks[i+1]}", 0.65))

        if state_match:
            st_raw = state_match.group(1).strip().lower()
            feats.append((f"state_raw:{st_raw}", 1.2))
            for st_tok in _split_camel_and_punct(st_raw):
                feats.append((f"state:{st_tok}", 0.9))
                feats.append((f"tok:{st_tok}", 0.5))

        if qual_match:
            for q in qual_match.group(1).split(","):
                qn = q.strip().lower()
                if qn:
                    feats.append((f"qual_raw:{qn}", 1.0))
                    for qt in _split_camel_and_punct(qn):
                        feats.append((f"qual:{qt}", 0.7))
    else:
        feats.append((f"cid:{s.lower()}", 1.0))
        toks = _split_camel_and_punct(s)
        for tok in toks:
            feats.append((f"ctok:{tok}", 1.0))
            if len(tok) >= 3:
                for i in range(len(tok) - 2):
                    feats.append((f"ctri:{tok[i:i+3]}", 0.30))
        for i in range(len(toks) - 1):
            feats.append((f"cbig:{toks[i]}_{toks[i+1]}", 0.70))

    if not feats:
        feats.append((f"raw:{s.lower()}", 1.0))
    return feats


def _project_compositional_features(
    features: List[Tuple[str, float]],
    dimension: int,
) -> np.ndarray:
    """
    Deterministic Weinberger signed feature projection into R^d, normalized to the unit sphere.
    Preserves compositional similarity without random MD5 seeds.
    """
    vec = np.zeros(dimension, dtype=np.float32)
    if dimension <= 0:
        return vec
    for feat, weight in features:
        h1 = _hash_u64(feat, 1)
        h2 = _hash_u64(feat, 2)
        idx1 = h1 % dimension
        idx2 = h2 % dimension
        sign1 = 1.0 if ((h1 >> 63) & 1) else -1.0
        sign2 = 1.0 if ((h2 >> 63) & 1) else -1.0
        vec[idx1] += np.float32(weight * sign1)
        vec[idx2] += np.float32(0.70710678 * weight * sign2)
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 1e-9 else vec


def _canonical_typestate_str(sig: Any) -> str:
    """Extracts a normalized canonical string representation for a typestate."""
    if sig is None:
        return "__any__"
    if isinstance(sig, str):
        s = sig.strip()
        return s if s else "__any__"
    raw = getattr(sig, "signature", sig)
    t_name = getattr(raw, "type_name", None) or getattr(sig, "type_name", None) or "any"
    state = getattr(raw, "state", None) or getattr(sig, "state", None)
    quals = getattr(raw, "qualifiers", None) or getattr(sig, "qualifiers", None)

    parts = [str(t_name)]
    if state and str(state).lower() not in ("any", "none", "unknown"):
        parts.append(f"[{state}]")
    if quals and isinstance(quals, (list, tuple, set)):
        valid_q = sorted(str(q) for q in quals if q and str(q).lower() not in ("any", "none"))
        if valid_q:
            parts.append(f"{{{','.join(valid_q)}}}")
    return "".join(parts)


class TransitionAutoencoder:
    """
    Self-Supervised Structural Transition Autoencoder (TransE / Koopman style).
    Maps typestates tau to points z_tau in R^d and cell morphisms c to translation vectors t_c in R^d.
    """

    def __init__(
        self,
        dimension: int = 64,
        margin: float = 1.0,
        lr: float = 0.05,
        seed: int = 42,
    ):
        self.dimension = dimension
        self.margin = margin
        self.lr = lr
        self.seed = seed

        self.typestate_to_idx: Dict[str, int] = {}
        self.idx_to_typestate: List[str] = []
        self.cell_to_idx: Dict[str, int] = {}
        self.idx_to_cell: List[str] = []

        self.Z: Optional[np.ndarray] = None  # (N_types, d)
        self.T: Optional[np.ndarray] = None  # (N_cells, d)
        self.is_trained: bool = False

    def extract_triples_from_cells(
        self,
        cells: Union[Dict[str, Any], List[Any], Collection[Any]],
    ) -> List[Tuple[str, str, str]]:
        """
        Extracts directed transition triples (tau_in, cell_id, tau_out) from cell definitions.
        """
        cell_list = list(cells.values()) if isinstance(cells, dict) else list(cells)
        triples: List[Tuple[str, str, str]] = []

        for c in cell_list:
            cid = getattr(c, "cell_id", None) or getattr(c, "name", "unknown")
            in_states = []
            inputs = getattr(c, "inputs", {}) or {}
            for p_name, p_sig in inputs.items():
                if getattr(p_sig, "required", False) or getattr(p_sig, "default_value", None) is None:
                    in_states.append(_canonical_typestate_str(p_sig))

            if not in_states:
                in_states = ["__start__"]

            out_states = []
            outputs = getattr(c, "outputs", {}) or {}
            for p_name, p_sig in outputs.items():
                out_states.append(_canonical_typestate_str(p_sig))

            if not out_states:
                out_states = ["__end__"]

            for in_s in in_states:
                for out_s in out_states:
                    triples.append((in_s, cid, out_s))

        return triples

    def train(
        self,
        triples: List[Tuple[str, str, str]],
        epochs: int = 60,
        negative_rate: int = 2,
    ) -> None:
        """
        Vectorized self-supervised training over morphism transition triples in <2 seconds.
        Optimizes ||z_{tau_in} + t_c - z_{tau_out}||_2^2 + margin-contrastive negative sampling.
        """
        if not triples:
            # Fallback canonical triples if graph is completely empty
            triples = [
                ("__start__", "init_op", "data"),
                ("data", "transform_op", "processed"),
                ("processed", "sink_op", "__end__"),
            ]

        # 1. Build index mappings
        type_set = set()
        cell_set = set()
        for h, r, t in triples:
            type_set.add(h)
            cell_set.add(r)
            type_set.add(t)

        self.idx_to_typestate = sorted(type_set)
        self.typestate_to_idx = {s: i for i, s in enumerate(self.idx_to_typestate)}

        self.idx_to_cell = sorted(cell_set)
        self.cell_to_idx = {c: i for i, c in enumerate(self.idx_to_cell)}

        N_E = len(self.idx_to_typestate)
        N_R = len(self.idx_to_cell)

        # 2. Initialize latent representations structurally on unit hypersphere
        rng = np.random.RandomState(self.seed)
        self.Z = np.vstack([
            _project_compositional_features(_extract_structural_features(ts, is_typestate=True), self.dimension)
            for ts in self.idx_to_typestate
        ]).astype(np.float32)
        self.T = np.vstack([
            _project_compositional_features(_extract_structural_features(cid, is_typestate=False), self.dimension)
            for cid in self.idx_to_cell
        ]).astype(np.float32)
        if N_E > 0:
            self.Z += rng.normal(0, 0.05 / math.sqrt(self.dimension), (N_E, self.dimension)).astype(np.float32)
        if N_R > 0:
            self.T += rng.normal(0, 0.05 / math.sqrt(self.dimension), (N_R, self.dimension)).astype(np.float32)
        self.Z /= (np.linalg.norm(self.Z, axis=1, keepdims=True) + 1e-9)
        self.T /= (np.linalg.norm(self.T, axis=1, keepdims=True) + 1e-9)

        # 3. Vectorized contrastive training
        h_idx = np.array([self.typestate_to_idx[h] for h, _, _ in triples], dtype=np.int32)
        r_idx = np.array([self.cell_to_idx[r] for _, r, _ in triples], dtype=np.int32)
        t_idx = np.array([self.typestate_to_idx[t] for _, _, t in triples], dtype=np.int32)
        n_samples = len(triples)

        lr = self.lr
        margin = self.margin

        for epoch in range(epochs):
            for _ in range(negative_rate):
                t_neg = rng.randint(0, N_E, size=n_samples)

                # Positive distance: ||z_h + t_r - z_t||_2
                d_pos = self.Z[h_idx] + self.T[r_idx] - self.Z[t_idx]
                norm_pos = np.linalg.norm(d_pos, axis=1, keepdims=True)

                # Negative distance: ||z_h + t_r - z_{t_neg}||_2
                d_neg = self.Z[h_idx] + self.T[r_idx] - self.Z[t_neg]
                norm_neg = np.linalg.norm(d_neg, axis=1, keepdims=True)

                # Margin loss: max(0, margin + norm_pos - norm_neg)
                loss = np.maximum(0.0, margin + norm_pos - norm_neg)
                mask = (loss > 0).squeeze()

                if np.any(mask):
                    g_pos = d_pos / (norm_pos + 1e-9)
                    g_neg = d_neg / (norm_neg + 1e-9)
                    grad_comb = g_pos - g_neg

                    # Efficient vectorized updates via np.add.at
                    np.add.at(self.Z, h_idx[mask], -lr * grad_comb[mask])
                    np.add.at(self.T, r_idx[mask], -lr * grad_comb[mask])
                    np.add.at(self.Z, t_idx[mask], lr * g_pos[mask])
                    np.add.at(self.Z, t_neg[mask], -lr * g_neg[mask])

                    # Project back to unit ball
                    self.Z /= np.maximum(1.0, np.linalg.norm(self.Z, axis=1, keepdims=True))
                    self.T /= np.maximum(1.0, np.linalg.norm(self.T, axis=1, keepdims=True))

        self.is_trained = True
        logger.debug(
            f"[TRANSITION AUTOENCODER] Trained self-supervised manifold on {n_samples} triples "
            f"({N_E} states, {N_R} cells, dim={self.dimension}) across {epochs} epochs."
        )

    def encode_signature(self, sig: Union[Any, str]) -> np.ndarray:
        """Projects a discrete typestate signature tau into the continuous manifold z_tau in R^d."""
        s = _canonical_typestate_str(sig)
        if self.is_trained and self.Z is not None and s in self.typestate_to_idx:
            return self.Z[self.typestate_to_idx[s]].copy()

        # Deterministic compositional structural/vocabulary projection (zero MD5 RNG)
        return _project_compositional_features(
            _extract_structural_features(s, is_typestate=True),
            self.dimension,
        )

    def encode_cell(self, cell_id: str) -> np.ndarray:
        """Projects a cell morphism c into the continuous translation manifold t_c in R^d."""
        if self.is_trained and self.T is not None and cell_id in self.cell_to_idx:
            return self.T[self.cell_to_idx[cell_id]].copy()

        # Deterministic compositional identifier/token projection (zero MD5 RNG)
        return _project_compositional_features(
            _extract_structural_features(cell_id, is_typestate=False),
            self.dimension,
        )

    def predict_transition(
        self,
        state_vec: Union[np.ndarray, str, Any],
        cell_id: str,
    ) -> np.ndarray:
        """Computes continuous forward transition: z_pred = z_state + t_cell."""
        z_s = self.encode_signature(state_vec) if not isinstance(state_vec, np.ndarray) else state_vec
        t_c = self.encode_cell(cell_id)
        z_pred = z_s + t_c
        norm = np.linalg.norm(z_pred)
        return z_pred / (norm + 1e-9) if norm > 0 else z_pred

    def score_transition(
        self,
        state_vec: Union[np.ndarray, str, Any],
        cell_id: str,
        target_vec: Union[np.ndarray, str, Any],
    ) -> float:
        """
        Scores transition compatibility in continuous latent space.
        Computes cosine alignment between (z_state + t_cell) and z_target.
        Returns score in [-1.0, 1.0] where 1.0 indicates perfect continuous morphism alignment.
        """
        z_s = self.encode_signature(state_vec) if not isinstance(state_vec, np.ndarray) else state_vec
        z_t = self.encode_signature(target_vec) if not isinstance(target_vec, np.ndarray) else target_vec
        t_c = self.encode_cell(cell_id)

        z_pred = z_s + t_c
        norm_pred = np.linalg.norm(z_pred)
        norm_target = np.linalg.norm(z_t)

        if norm_pred < 1e-9 or norm_target < 1e-9:
            return 0.0

        cos_sim = float(np.dot(z_pred, z_t) / (norm_pred * norm_target))
        return max(-1.0, min(1.0, cos_sim))

    def find_nearest_transitions(
        self,
        state_vec: Union[np.ndarray, str, Any],
        target_vec: Union[np.ndarray, str, Any],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Continuous inverse transition search:
        Finds candidate cells whose translation vector t_c aligns with (z_target - z_state).
        """
        z_s = self.encode_signature(state_vec) if not isinstance(state_vec, np.ndarray) else state_vec
        z_t = self.encode_signature(target_vec) if not isinstance(target_vec, np.ndarray) else target_vec

        ideal_t = z_t - z_s
        norm_ideal = np.linalg.norm(ideal_t)
        if norm_ideal < 1e-9 or not self.idx_to_cell or self.T is None:
            return []

        ideal_unit = ideal_t / norm_ideal
        sims = np.dot(self.T, ideal_unit)
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.idx_to_cell[i], float(sims[i])) for i in top_indices]


def train_autoencoder_from_orchestrator(
    orchestrator: Any,
    dimension: int = 64,
    epochs: int = 50,
) -> TransitionAutoencoder:
    """Extracts all cell transitions from orchestrator and trains the TransitionAutoencoder."""
    cells = list(orchestrator.loaded_cells.values()) if hasattr(orchestrator, "loaded_cells") else []
    ae = TransitionAutoencoder(dimension=dimension)
    triples = ae.extract_triples_from_cells(cells)
    ae.train(triples, epochs=epochs)
    return ae
