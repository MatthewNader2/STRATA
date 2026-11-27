"""
src/hardware.py — Lightweight device configuration profiler for inference accelerators.

Extracted from legacy router.py so unified runtime modules can detect CPU/CUDA/MPS
devices without importing the legacy routing stack.
"""
from __future__ import annotations

from typing import Dict, Optional


class HardwareProfiler:
    """Device configuration profiler for inference accelerators."""

    _cached_device: Optional[str] = None
    _config: Dict[str, str] = {
        "embedder": "auto",
        "llm": "auto",
        "trees": "ram",
    }

    @classmethod
    def set_config(
        cls,
        embedder_device: str = "auto",
        llm_device: str = "auto",
        trees_storage: str = "ram",
    ) -> None:
        cls._config["embedder"] = (embedder_device or "auto").lower()
        cls._config["llm"] = (llm_device or "auto").lower()
        cls._config["trees"] = (trees_storage or "ram").lower()

    @classmethod
    def get_embedder_device(cls) -> str:
        if cls._config.get("embedder", "auto") != "auto":
            return cls._config["embedder"]
        return cls.get_optimal_device()

    @classmethod
    def get_llm_device(cls) -> str:
        if cls._config.get("llm", "auto") != "auto":
            return cls._config["llm"]
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


__all__ = ["HardwareProfiler"]
