"""
src/cognitive_pathfinder.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Pillar 3: Cognitive Latent Pathfinding.

Extracts continuous latent representations from local sub-1B models
(EmbeddingGemma-300M, Qwen2.5-Coder-0.5B, Jina-Nano) and guides A* / beam
search over the lattice DAG via latent geodesic alignment and directional
progress projection.
"""

from __future__ import annotations
import os
import threading
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    from log_config import get_logger
except ImportError:
    import logging
    get_logger = logging.getLogger

try:
    from .config import MODELS_DIR, settings
except (ImportError, ValueError):
    from config import MODELS_DIR, settings

try:
    from .lattice import TypeRegistry
except (ImportError, ValueError):
    from lattice import TypeRegistry

logger = get_logger("cognitive_pathfinder")


class ModelUnavailableError(RuntimeError):
    """
    Raised when cognitive latent extraction is invoked but no local neural
    embedding model (SentenceTransformer or GGUF) is loaded or available.
    Prevents silent fallback to pseudo-random Gaussian noise.
    """
    pass


def get_latent_scoring_weights() -> Dict[str, float]:
    """
    Resolves geodesic scoring and state-update weights from TypeRegistry
    or settings without hardcoded magic numbers in the scoring loop.
    """
    defaults = {
        "sim_weight": float(settings.latent_sim_weight),
        "progress_weight": float(settings.latent_progress_weight),
        "clause_weight": float(settings.latent_clause_weight),
        "goal_weight": float(settings.latent_goal_weight),
        "structural_weight": float(settings.latent_structural_weight),
        "state_momentum": float(settings.latent_state_momentum),
    }
    try:
        reg = TypeRegistry.get_instance()
        sem = reg.get_verification_semantics("latent_scoring_weights")
        if sem:
            for item in sem:
                if isinstance(item, str) and "=" in item:
                    k, v = item.split("=", 1)
                    k = k.strip().lower()
                    if k in defaults:
                        defaults[k] = float(v.strip())
    except Exception as e:
        logger.debug("[COGNITIVE PATHFINDER] Weight config lookup fallback: %s", e)
    return defaults


class LatentVectorExtractor:
    """
    Thread-safe continuous latent vector extractor using local sub-1B models.
    Supports SentenceTransformer (EmbeddingGemma-300M, Jina-Nano) and llama.cpp GGUF.
    Strictly fails loudly with ModelUnavailableError if no neural model is available.
    """
    _instance: Optional[LatentVectorExtractor] = None
    _lock = threading.RLock()

    def __init__(self, model_name: Optional[str] = None):
        self._lock = threading.RLock()
        self._cache: Dict[str, np.ndarray] = {}
        self._model = None
        self._backend: str = "unavailable"
        self._dim: int = 0
        self._init_error: Optional[str] = None
        self._init_model(model_name)

    @classmethod
    def get_instance(cls) -> LatentVectorExtractor:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def is_available(self) -> bool:
        """Returns True iff a real neural embedding backend is loaded and active."""
        return self._backend in ("sentence_transformers", "llama_cpp") and self._model is not None

    def _init_model(self, model_name: Optional[str] = None) -> None:
        emb_dir = os.path.join(MODELS_DIR, "embeddings")
        llm_dir = os.path.join(MODELS_DIR, "llms")
        errors: List[str] = []

        # 0. Check if Profile 0 (Pure Symbolic, zero neural models) is active
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            mm = ModelManager.get_instance()
            if mm.profile is not None:
                prof_id = str(getattr(mm.profile, "profile_id", "") or "")
                if prof_id == "0":
                    self._model = None
                    self._backend = "unavailable"
                    self._init_error = "Profile 0 is pure symbolic: neural embeddings are disabled."
                    logger.info("[COGNITIVE PATHFINDER] Profile 0 active: neural latent extractor disabled.")
                    return
        except Exception as e:
            errors.append(f"Profile 0 check: {e}")

        if settings.profile == "0" or os.environ.get("NSTL_PROFILE") == "0":
            self._model = None
            self._backend = "unavailable"
            self._init_error = "Profile 0 is pure symbolic: neural embeddings are disabled."
            logger.info("[COGNITIVE PATHFINDER] Profile 0 configured: neural latent extractor disabled.")
            return

        # 1. First check if ModelManager already has an active profile with an embedder
        try:
            try:
                from .inference import ModelManager
            except (ImportError, ValueError):
                from inference import ModelManager
            mm = ModelManager.get_instance()
            if mm.profile is not None:
                active_emb = getattr(mm.profile, "model", None) or getattr(mm.profile, "embedder", None)
                if active_emb is not None:
                    self._model = active_emb
                    self._backend = "sentence_transformers"
                    self._dim = int(mm.embedding_dimension)
                    logger.info(
                        "[COGNITIVE PATHFINDER] Reusing active ModelManager embedder (dim=%d)",
                        self._dim,
                    )
                    return
        except Exception as e:
            errors.append(f"ModelManager reuse: {e}")

        # 2. Try optimal local SentenceTransformer embedding model
        try:
            try:
                from .inference import select_optimal_embedder
            except (ImportError, ValueError):
                from inference import select_optimal_embedder
            opt_emb = select_optimal_embedder(model_name or "")
            emb_path = os.path.join(emb_dir, opt_emb)
            if os.path.isdir(emb_path):
                from sentence_transformers import SentenceTransformer
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = SentenceTransformer(emb_path, device=device, trust_remote_code=True)
                self._backend = "sentence_transformers"
                sample = self._model.encode("probe", normalize_embeddings=True)
                self._dim = int(sample.shape[-1])
                logger.info(
                    "[COGNITIVE PATHFINDER] Loaded optimal embedder '%s' on %s (dim=%d)",
                    opt_emb,
                    device,
                    self._dim,
                )
                return
            else:
                errors.append(f"Embedder directory not found: {emb_path}")
        except Exception as e:
            errors.append(f"SentenceTransformer load failed: {e}")
            logger.debug("[COGNITIVE PATHFINDER] Failed loading optimal embedder: %s", e)

        # 3. Fallback to optimal local GGUF model in embedding mode
        try:
            try:
                from .inference import select_optimal_llm
            except (ImportError, ValueError):
                from inference import select_optimal_llm
            best_llm = select_optimal_llm()
            llm_cand_dir = os.path.join(llm_dir, best_llm)
            if os.path.isdir(llm_cand_dir):
                ggufs = [f for f in os.listdir(llm_cand_dir) if f.endswith(".gguf")]
                if ggufs:
                    chosen_gguf = next((f for f in ggufs if "shard" not in f), ggufs[0])
                    model_file = os.path.join(llm_cand_dir, chosen_gguf)
                    from llama_cpp import Llama
                    self._model = Llama(model_path=model_file, embedding=True, verbose=False, n_gpu_layers=0)
                    self._backend = "llama_cpp"
                    res = self._model.create_embedding("probe")
                    raw = np.array(res["data"][0]["embedding"], dtype=np.float32)
                    pooled = np.mean(raw, axis=0) if raw.ndim > 1 else raw
                    self._dim = int(pooled.shape[-1])
                    logger.info(
                        "[COGNITIVE PATHFINDER] Loaded optimal LLM '%s' GGUF embedding on CPU (dim=%d)",
                        best_llm,
                        self._dim,
                    )
                    return
                else:
                    errors.append(f"No .gguf files in {llm_cand_dir}")
            else:
                errors.append(f"LLM directory not found: {llm_cand_dir}")
        except Exception as e:
            errors.append(f"GGUF load failed: {e}")
            logger.debug("[COGNITIVE PATHFINDER] Failed loading GGUF: %s", e)

        # 4. Fail-Closed: Never fall back to pseudo-random hash noise
        self._backend = "unavailable"
        self._model = None
        self._dim = 0
        self._init_error = "; ".join(errors) if errors else "No local embedding or GGUF models found"
        logger.warning(
            "[COGNITIVE PATHFINDER] No neural models available (%s); M11 latent pathfinding disabled.",
            self._init_error,
        )

    def _require_available(self) -> None:
        if not self.is_available():
            raise ModelUnavailableError(
                f"LatentVectorExtractor has no neural model loaded ({self._init_error or 'unavailable'}). "
                f"Refusing to generate pseudo-random noise embeddings."
            )

    def encode(self, text: str) -> np.ndarray:
        """
        Encodes text to a unit-normalized latent vector on the hypersphere S^{d-1}.
        Raises ModelUnavailableError if no neural model is loaded.
        """
        self._require_available()
        if not text or not text.strip():
            return np.zeros(self._dim, dtype=np.float32)

        with self._lock:
            if text in self._cache:
                return self._cache[text]

            if self._backend == "sentence_transformers" and self._model is not None:
                vec = self._model.encode(text, normalize_embeddings=True)
                raw_vec = np.array(vec, dtype=np.float32)
                norm = float(np.linalg.norm(raw_vec))
                norm_vec = raw_vec / (norm + 1e-9) if norm > 0 else np.zeros(self._dim, dtype=np.float32)

            elif self._backend == "llama_cpp" and self._model is not None:
                res = self._model.create_embedding(text)
                raw = np.array(res["data"][0]["embedding"], dtype=np.float32)
                pooled = np.mean(raw, axis=0) if raw.ndim > 1 else raw
                norm = float(np.linalg.norm(pooled))
                norm_vec = pooled / (norm + 1e-9) if norm > 0 else np.zeros(self._dim, dtype=np.float32)

            else:
                raise ModelUnavailableError("Neural embedding backend is not initialized.")

            self._cache[text] = norm_vec
            return norm_vec

    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encodes a batch of texts to unit-normalized latent vectors.
        Raises ModelUnavailableError if no neural model is loaded.
        """
        self._require_available()
        if not texts:
            return []

        uncached_indices: List[int] = []
        uncached_texts: List[str] = []
        results: List[Optional[np.ndarray]] = [None] * len(texts)

        with self._lock:
            for i, t in enumerate(texts):
                if not t or not t.strip():
                    results[i] = np.zeros(self._dim, dtype=np.float32)
                elif t in self._cache:
                    results[i] = self._cache[t]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(t)

            if uncached_texts:
                if self._backend == "sentence_transformers" and self._model is not None:
                    vecs = self._model.encode(uncached_texts, normalize_embeddings=True)
                    for idx, vec in zip(uncached_indices, vecs):
                        raw_vec = np.array(vec, dtype=np.float32)
                        norm = float(np.linalg.norm(raw_vec))
                        norm_vec = raw_vec / (norm + 1e-9) if norm > 0 else np.zeros(self._dim, dtype=np.float32)
                        self._cache[texts[idx]] = norm_vec
                        results[idx] = norm_vec
                else:
                    for idx in uncached_indices:
                        results[idx] = self.encode(texts[idx])

        return [r if r is not None else np.zeros(self._dim, dtype=np.float32) for r in results]


class CognitivePathfinder:
    """
    Cognitive Latent Pathfinder with Experimental Lab Integration:
      - Early-Exit Sub-Layer Residual Stream Extraction & ZCA Whitening
      - Sparse Autoencoder (SAE) / OMP Multi-Clause Concept Decomposition
      - Spherical Geodesic Energy Heuristics on S^{d-1}
    """

    def __init__(self, extractor: Optional[LatentVectorExtractor] = None):
        self.extractor = extractor or LatentVectorExtractor.get_instance()
        self._cell_vec_cache: Dict[str, np.ndarray] = {}
        try:
            from .experiments import EarlyExitLatentProbe, SparseConceptCoding, ACTIVE_EXPERIMENTS
        except (ImportError, ValueError):
            from experiments import EarlyExitLatentProbe, SparseConceptCoding, ACTIVE_EXPERIMENTS
        self.exp_cfg = ACTIVE_EXPERIMENTS
        self.probe = EarlyExitLatentProbe(self.extractor, config=self.exp_cfg)
        self.sae = SparseConceptCoding()
        self._whitening_fitted = False

    def is_available(self) -> bool:
        return self.extractor.is_available()

    def get_cell_representation_text(self, cell: Any) -> str:
        cid = getattr(cell, "cell_id", "")
        keywords = " ".join(getattr(cell, "keywords", []) or [])
        summary = getattr(cell, "summary", "") or ""
        doc = getattr(cell, "documentation", "") or getattr(cell, "docstring", "") or ""
        code = (getattr(cell, "code_template", "") or "")[:120]
        in_types = " ".join(
            str(getattr(getattr(p, "signature", None), "type_name", "") or getattr(p, "type_name", ""))
            for p in getattr(cell, "inputs", {}).values()
        )
        out_types = str(
            getattr(getattr(getattr(cell, "primary_output", None), "signature", None), "type_name", "")
            or getattr(getattr(cell, "primary_output", None), "type_name", "")
        )
        return f"{cid} {keywords} {summary} {doc} in:{in_types} out:{out_types} {code}".strip()

    def encode_text(self, text: str) -> np.ndarray:
        """Encodes text using Early-Exit + ZCA Whitening when enabled in ACTIVE_EXPERIMENTS."""
        if self.exp_cfg.early_exit_layer is not None or self.exp_cfg.use_zca_whitening:
            return self.probe.encode_early_exit(text, layer_idx=self.exp_cfg.early_exit_layer)
        return self.extractor.encode(text)

    def ensure_corpus_whitening(self, cells: List[Any]) -> None:
        """Fits ZCA whitening matrix over the active candidate cell embeddings."""
        if self._whitening_fitted or not self.exp_cfg.use_zca_whitening or len(cells) < 4:
            return
        texts = [self.get_cell_representation_text(c) for c in cells]
        raw_vecs = np.array(self.extractor.encode_batch(texts), dtype=np.float32)
        self.probe.fit_whitening(raw_vecs)
        self._whitening_fitted = True
        self._cell_vec_cache.clear()

    def get_cell_vector(self, cell: Any) -> np.ndarray:
        cid = getattr(cell, "cell_id", id(cell))
        if cid in self._cell_vec_cache:
            return self._cell_vec_cache[cid]
        text = self.get_cell_representation_text(cell)
        vec = self.encode_text(text)
        self._cell_vec_cache[cid] = vec
        return vec

    def compute_sparse_concept_boosts(self, prompt: str, candidates: List[Any]) -> Dict[str, float]:
        """
        Decomposes `prompt` into sparse K-hot concept activations over `candidates`
        via Orthogonal Matching Pursuit (SAE) when `use_sparse_coding` is enabled.
        """
        if not self.exp_cfg.use_sparse_coding or not candidates:
            return {}
        self.ensure_corpus_whitening(candidates)
        p_vec = self.encode_text(prompt)
        c_ids = [getattr(c, "cell_id", "") for c in candidates]
        D_mat = np.array([self.get_cell_vector(c) for c in candidates], dtype=np.float32)
        return self.sae.decompose_prompt_embedding(
            prompt_vec=p_vec,
            dictionary_matrix=D_mat,
            cell_ids=c_ids,
            top_k=self.exp_cfg.sparse_top_k,
        )

    def compute_latent_heuristic(
        self,
        current_vec: np.ndarray,
        candidate_vec: np.ndarray,
        goal_vec: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        w = weights or get_latent_scoring_weights()
        w_sim = w.get("sim_weight", 0.5)
        w_prog = w.get("progress_weight", 0.5)
        norm_w = (w_sim + w_prog) or 1.0

        norm_cand = float(np.linalg.norm(candidate_vec))
        norm_goal = float(np.linalg.norm(goal_vec))
        if norm_cand < 1e-9 or norm_goal < 1e-9:
            return 0.0

        u_cand = candidate_vec / norm_cand
        u_goal = goal_vec / norm_goal
        sim = float(np.clip(np.dot(u_cand, u_goal), -1.0, 1.0))

        norm_curr = float(np.linalg.norm(current_vec))
        if norm_curr > 1e-6:
            u_curr = current_vec / norm_curr
            tangent_goal = u_goal - float(np.dot(u_goal, u_curr)) * u_curr
            tangent_step = u_cand - u_curr
            norm_tg = float(np.linalg.norm(tangent_goal))
            norm_ts = float(np.linalg.norm(tangent_step))
            if norm_tg > 1e-6 and norm_ts > 1e-6:
                progress = float(np.clip(np.dot(tangent_step / norm_ts, tangent_goal / norm_tg), -1.0, 1.0))
            else:
                progress = sim
        else:
            progress = sim

        geodesic_sim = 1.0 - (float(np.arccos(sim)) / np.pi)
        geodesic_prog = 1.0 - (float(np.arccos(progress)) / np.pi)
        return (w_sim * geodesic_sim + w_prog * geodesic_prog) / norm_w

    def update_state_vector(
        self,
        current_vec: np.ndarray,
        next_vec: np.ndarray,
        momentum: Optional[float] = None,
    ) -> np.ndarray:
        if momentum is None:
            momentum = get_latent_scoring_weights().get("state_momentum", 0.7)
        momentum = float(np.clip(momentum, 0.0, 1.0))

        if float(np.linalg.norm(current_vec)) < 1e-9:
            norm_n = float(np.linalg.norm(next_vec))
            return (next_vec / norm_n).astype(np.float32) if norm_n > 1e-9 else next_vec

        blended = momentum * current_vec + (1.0 - momentum) * next_vec
        norm_b = float(np.linalg.norm(blended))
        if norm_b > 1e-9:
            return (blended / norm_b).astype(np.float32)
        return blended.astype(np.float32)

    def rank_candidates(
        self,
        candidates: List[Any],
        current_vec: np.ndarray,
        goal_vec: np.ndarray,
        relevance_map: Optional[Dict[str, float]] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> List[Tuple[Any, float]]:
        w = get_latent_scoring_weights()
        a = alpha if alpha is not None else (w.get("clause_weight", 0.5) + w.get("goal_weight", 0.3))
        b = beta if beta is not None else w.get("structural_weight", 0.2)
        total = (a + b) or 1.0

        self.ensure_corpus_whitening(candidates)
        scored: List[Tuple[Any, float]] = []
        for cell in candidates:
            c_vec = self.get_cell_vector(cell)
            h_score = self.compute_latent_heuristic(current_vec, c_vec, goal_vec, weights=w)
            r_score = (relevance_map or {}).get(getattr(cell, "cell_id", ""), 0.0)
            composite = (a * h_score + b * r_score) / total
            scored.append((cell, composite))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored


__all__ = [
    "ModelUnavailableError",
    "LatentVectorExtractor",
    "CognitivePathfinder",
    "get_latent_scoring_weights",
]
