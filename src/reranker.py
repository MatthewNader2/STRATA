"""
src/reranker.py — Pluggable neural cross-encoder / listwise reranker for STRATA Layer 1.

Ranks neutral API documentation strings cleanly via `score_documents(query, documents)`
with principled score calibration and zero model-specific magic constants.
"""
from __future__ import annotations

import inspect
import math
import os
import threading
from typing import Dict, List, Optional, Tuple

try:
    from .config import MODELS_DIR, settings
    from .log_config import get_logger
except (ImportError, ValueError):
    from config import MODELS_DIR, settings
    from log_config import get_logger

torch = None
AutoModel = None
_TORCH_AVAILABLE: Optional[bool] = None

_SHARED_RERANKERS: Dict[Tuple[str, str, bool], "LocalReranker"] = {}
_SHARED_RERANKER_LOCK = threading.RLock()

logger = get_logger("reranker")


def _ensure_reranker_imports() -> bool:
    """Lazily import torch and transformers.AutoModel only when reranking is active."""
    global torch, AutoModel, _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is not None:
        return _TORCH_AVAILABLE
    try:
        import torch as _torch
        from transformers import AutoModel as _AutoModel
        torch = _torch
        AutoModel = _AutoModel
        _TORCH_AVAILABLE = True
    except ImportError:
        torch = None
        AutoModel = None
        _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


def get_available_rerankers() -> List[str]:
    """Scan configured rerankers directory cleanly for available local reranker models."""
    r_dir = settings.rerankers_dir or os.path.join(MODELS_DIR, "rerankers")
    if not os.path.exists(r_dir):
        return []
    discovered = sorted(
        d for d in os.listdir(r_dir)
        if os.path.isdir(os.path.join(r_dir, d)) and not d.startswith(".")
    )
    prefs = [str(p).strip() for p in (settings.preferred_rerankers or ()) if str(p).strip()]
    if not prefs:
        return discovered
    ordered = [d for p in prefs for d in discovered if p == d or p.lower() in d.lower()]
    ordered.extend(d for d in discovered if d not in ordered)
    return list(dict.fromkeys(ordered))


class LocalReranker:
    """Pluggable local cross-encoder / listwise neural document reranker."""

    @classmethod
    def get_shared(cls, model_name_or_path: Optional[str] = None, device: str = "auto") -> "LocalReranker":
        """Return a process-wide cached LocalReranker instance."""
        with _SHARED_RERANKER_LOCK:
            norm_name = model_name_or_path or settings.reranker_model or ""
            if not norm_name:
                avail = get_available_rerankers()
                if not avail:
                    raise FileNotFoundError(
                        f"No reranker model specified and none found in '{settings.rerankers_dir}'."
                    )
                norm_name = avail[0]

            norm_dev = device
            if device == "auto":
                env_dev = os.environ.get("NSTL_RERANKER_DEVICE", "").strip().lower()
                if env_dev:
                    norm_dev = env_dev
                else:
                    norm_dev = "cuda" if (_ensure_reranker_imports() and torch.cuda.is_available()) else "cpu"

            key = (norm_name, norm_dev, bool(os.environ.get("NSTL_RERANKER_CPU_OFFLOAD") == "1"))
            inst = _SHARED_RERANKERS.get(key)
            if inst is None or getattr(inst, "model", None) is None:
                inst = cls(model_name_or_path=norm_name, device=norm_dev)
                _SHARED_RERANKERS[key] = inst
            return inst

    @classmethod
    def release_shared(cls) -> None:
        """Release cached shared reranker instances to reclaim VRAM/RAM."""
        with _SHARED_RERANKER_LOCK:
            for inst in _SHARED_RERANKERS.values():
                try:
                    if hasattr(inst, "model"):
                        inst.model = None
                except Exception:
                    pass
            _SHARED_RERANKERS.clear()

    @staticmethod
    def _resolve_device_and_dtype(device: str) -> Tuple[str, object, bool]:
        _ensure_reranker_imports()
        cuda_available = bool(torch is not None and torch.cuda.is_available())
        req = str(device or "auto").strip().lower()
        env_dev = os.environ.get("NSTL_RERANKER_DEVICE", "").strip().lower() if req == "auto" else ""
        eff_req = env_dev or req
        eff_dev = ("cuda" if cuda_available else "cpu") if eff_req == "auto" else eff_req

        if not cuda_available or eff_dev == "cpu":
            return ("cpu", "auto", False)
        if eff_dev == "cuda" and os.environ.get("NSTL_RERANKER_CPU_OFFLOAD") == "1":
            return ("cpu", "auto", True)
        if eff_dev == "cuda":
            fp16_dtype = torch.float16 if torch is not None else "float16"
            return ("cuda", fp16_dtype, True)
        return (eff_dev, "auto", False)

    def __init__(self, model_name_or_path: Optional[str] = None, device: str = "auto") -> None:
        if not _ensure_reranker_imports():
            logger.warning("[RERANKER] torch/transformers unavailable; neural reranking disabled.")
            self.device = "cpu"
            self.target_dtype: object = "auto"
            self.offload_to_cpu = False
            self.model_name = model_name_or_path or self._resolve_default_model()
            self.model = None
            self._supports_max_query_len = False
            self._lock = threading.Lock()
            return

        self.device, self.target_dtype, self.offload_to_cpu = self._resolve_device_and_dtype(device)
        self.model_name = model_name_or_path or self._resolve_default_model()
        self.model = None
        self._supports_max_query_len = False
        self._lock = threading.Lock()
        self._load_model()

    def _resolve_default_model(self) -> str:
        if settings.reranker_model:
            return settings.reranker_model
        available = get_available_rerankers()
        if available:
            return available[0]
        raise FileNotFoundError(
            f"No reranker model configured or present in '{settings.rerankers_dir}'."
        )

    def _load_model(self) -> None:
        r_dir = settings.rerankers_dir or os.path.join(MODELS_DIR, "rerankers")
        local_path = os.path.join(r_dir, self.model_name)
        model_path = local_path if os.path.exists(local_path) else self.model_name
        load_target = "cpu" if self.offload_to_cpu else self.device

        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=self.target_dtype,
            ).to(load_target)
            self.model.eval()
            try:
                self._supports_max_query_len = "max_query_length" in inspect.signature(self.model.rerank).parameters
            except Exception:
                self._supports_max_query_len = False
        except Exception as e:
            logger.error("[RERANKER] Failed to load reranker '%s': %s", self.model_name, e)
            self.model = None
            self._supports_max_query_len = False

    @staticmethod
    def _calibrate_score(raw_sc: float, is_cosine_model: bool = False) -> float:
        """
        Map raw reranker scores monotonically onto (0, 1):
          - Bounded similarity scores [-1, 1] scale linearly to [0, 1].
          - Unbounded logit scores map via standard logistic sigmoid.
        """
        val = float(raw_sc)
        if is_cosine_model:
            return (max(-1.0, min(1.0, val)) + 1.0) / 2.0
        logit = max(-60.0, min(60.0, val))
        return 1.0 / (1.0 + math.exp(-logit))

    def _infer_raw_scores(self, query: str, documents: List[str]) -> Optional[List[Tuple[int, float]]]:
        moved_to_gpu = False
        try:
            if self.offload_to_cpu and self.device == "cuda":
                self.model.to("cuda")
                moved_to_gpu = True
            rerank_kwargs = {"top_n": len(documents)}
            if self._supports_max_query_len:
                rerank_kwargs["max_query_length"] = 512
            if torch is not None:
                with torch.inference_mode():
                    results = self.model.rerank(query, documents, **rerank_kwargs)
            else:
                results = self.model.rerank(query, documents, **rerank_kwargs)
        except Exception as e:
            logger.error("[RERANKER] Execution failed: %s", e)
            return None
        finally:
            if moved_to_gpu and self.model is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass

        parsed: List[Tuple[int, float]] = []
        for idx, res in enumerate(results or ()):
            if isinstance(res, dict):
                doc_idx = int(res.get("index", idx))
                raw_score = float(res.get("relevance_score", res.get("score", 0.0)))
            else:
                doc_idx = idx
                raw_score = float(res)
            parsed.append((doc_idx, raw_score))
        return parsed

    def score_documents(self, query: str, documents: List[str]) -> List[float]:
        """Score arbitrary neutral text documents against `query` in input order."""
        if not documents:
            return []
        if self.model is None:
            return [0.0] * len(documents)

        with self._lock:
            parsed = self._infer_raw_scores(query, documents)
        if parsed is None:
            return [0.0] * len(documents)

        is_cosine_model = bool(parsed) and all(-1.05 <= raw_sc <= 1.05 for _, raw_sc in parsed)
        scores = [0.0] * len(documents)
        for doc_idx, raw_sc in parsed:
            if 0 <= doc_idx < len(documents):
                scores[doc_idx] = self._calibrate_score(raw_sc, is_cosine_model=is_cosine_model)
        return scores


__all__ = ["get_available_rerankers", "LocalReranker"]
