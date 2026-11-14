"""
src/inference.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Multi-Profile Inference Engine with Dynamic Hardware Introspection,
Agnostic Semantic Compilation, and Thread-Safe Generation.
"""

from __future__ import annotations
import gc
import inspect
import json
import os
import threading
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Union

try:
    from utils import is_split_gguf_shard
except ImportError:
    from .utils import is_split_gguf_shard

import numpy as np

# Torch is required only by neural inference profiles; the symbolic toolchain
# (CLI compile/validate/harvest) runs without it.
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

try:
    from log_config import get_logger
except ImportError:
    try:
        from .log_config import get_logger
    except ImportError:
        import logging
        get_logger = logging.getLogger

try:
    from .config import MODELS_DIR, settings
    from .utils import extract_code_from_llm_response
except (ImportError, ValueError):
    from config import MODELS_DIR, settings
    from utils import extract_code_from_llm_response

# Configure PyTorch CUDA memory allocator to prevent memory fragmentation
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = get_logger("inference")


def resolve_embedding_dimension(model: Any, fallback: int = 384) -> int:
    """
    Resolves the embedding dimension of a loaded SentenceTransformer model
    across library versions.
    """
    modern = getattr(model, "get_embedding_dimension", None)
    if callable(modern):
        try:
            dim = int(modern())
            if dim > 0:
                return dim
        except Exception:
            pass

    legacy = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(legacy):
        try:
            dim = int(legacy())
            if dim > 0:
                return dim
        except TypeError:
            name = getattr(model, "_model_name", None) or getattr(model, "model_name_or_path", None)
            if isinstance(name, str) and name:
                try:
                    dim = int(legacy(name))
                    if dim > 0:
                        return dim
                except Exception:
                    pass
        except Exception:
            pass

    try:
        probe = model.encode(["nstl dimension probe"], convert_to_numpy=True)
        dim = int(probe.shape[-1])
        if dim > 0:
            return dim
    except Exception:
        pass

    logger.warning(f"[INFERENCE] Could not resolve embedding dimension; using fallback {fallback}.")
    return fallback


def get_adaptive_batch_size(device: str = "cuda", min_batch: int = 8, max_batch: int = 128) -> int:
    """
    Computes an optimal batch size dynamically based on available device memory headroom.
    Fully hardware-agnostic: tracks continuous free VRAM headroom without hardcoded card thresholds.
    """
    if not TORCH_AVAILABLE or "cuda" not in str(device).lower() or not torch.cuda.is_available():
        return 32

    try:
        free_mem, _ = torch.cuda.mem_get_info()
        free_gb = free_mem / (1024 ** 3)
        # Continuous logarithmic scaling mapped to powers of 2 (approx. 8 items per 1 GB of free headroom)
        power = int(np.clip(np.floor(np.log2(max(1.0, free_gb * 8))), 3, 7))
        calculated_bs = int(2 ** power)
        return int(max(min_batch, min(max_batch, calculated_bs)))
    except Exception as e:
        logger.debug(f"[HARDWARE TELEMETRY] Adaptive batch size fallback: {e}")
        return 32


class InferenceProfile(ABC):
    @abstractmethod
    def load_models(self, embedder_name: str, llm_name: str):
        pass

    @abstractmethod
    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        pass

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        return [self.get_embedding(t, mode=mode) for t in texts]

    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        pass

    @abstractmethod
    def can_synthesize(self) -> bool:
        pass

    @abstractmethod
    def can_feedback_check(self) -> bool:
        pass

    def has_translator_pass(self) -> bool:
        return False

    def has_semantic_compiler(self) -> bool:
        return False

    def uses_reranker(self) -> bool:
        return False

    def can_extract_slots(self) -> bool:
        return self.can_synthesize()

    def has_ir_compiler(self) -> bool:
        return False

    def has_coalgebraic_unfolder(self) -> bool:
        return False

    def has_constrained_choice_scorer(self) -> bool:
        return False

    def score_choices(
        self,
        prompt: str,
        candidates: List[str],
        context_desc: str = "",
    ) -> Dict[str, float]:
        return {}

    def has_embedder(self) -> bool:
        return self.embedding_dimension > 0

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        pass

    @abstractmethod
    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        pass


def select_optimal_embedder(requested_name: str = "") -> str:
    """
    Selects the optimal local embedding model based on precomputed cache presence and footprint efficiency.
    Scoring is dynamically normalized across available candidate models without hardcoded architecture
    substrings, fixed cache naming prefixes, or arbitrary byte constants.
    """
    emb_base_dir = os.path.join(MODELS_DIR, "embeddings")
    if not os.path.exists(emb_base_dir):
        return requested_name or "default"

    available = sorted([
        d for d in os.listdir(emb_base_dir)
        if os.path.isdir(os.path.join(emb_base_dir, d)) and not d.endswith("-GGUF")
    ])
    if not available:
        return requested_name or "default"

    req = (requested_name or "").strip()
    if req and req not in ("auto", "default"):
        if req in available:
            return req
        for cand in available:
            if req.lower() in cand.lower():
                return cand
    else:
        # Default / auto selection prioritizes the nano embedder when present
        m_nano = next((m for m in available if "nano" in m.lower()), None)
        if m_nano is not None:
            return m_nano

    # Locate cache directories
    possible_cache_dirs = [
        os.path.join(os.path.dirname(MODELS_DIR), ".rag_cache"),
        os.path.join(MODELS_DIR, ".rag_cache"),
        os.path.abspath(".rag_cache")
    ]
    cache_dir = next((cd for cd in possible_cache_dirs if os.path.isdir(cd)), None)

    model_disk_sizes: Dict[str, int] = {}
    model_cache_sizes: Dict[str, int] = {}

    for cand in available:
        cand_path = os.path.join(emb_base_dir, cand)
        try:
            total_size = sum(
                os.path.getsize(os.path.join(root, f))
                for root, _, files in os.walk(cand_path)
                for f in files
            )
        except Exception:
            total_size = 1
        model_disk_sizes[cand] = max(1, total_size)

        cand_cache_size = 0
        if cache_dir and os.path.isdir(cache_dir):
            cand_norm = cand.lower().replace("-", "_")
            try:
                for f in os.listdir(cache_dir):
                    f_lower = f.lower()
                    if (cand.lower() in f_lower or cand_norm in f_lower) and "cache" in f_lower:
                        cand_cache_size = max(cand_cache_size, os.path.getsize(os.path.join(cache_dir, f)))
            except Exception:
                pass
        model_cache_sizes[cand] = cand_cache_size

    min_disk = min(model_disk_sizes.values()) if model_disk_sizes else 1
    max_cache = max(model_cache_sizes.values()) if model_cache_sizes else 0

    best_model = available[0]
    best_score = -1.0

    for cand in available:
        efficiency = float(min_disk / model_disk_sizes[cand])
        coverage = float(model_cache_sizes[cand] / max_cache) if max_cache > 0 else 0.0

        score = 0.85 * coverage + 0.15 * efficiency
        logger.debug(f"[EMBEDDER EVAL] Model: {cand} | Coverage: {coverage:.2f} | Efficiency: {efficiency:.2f} | Score: {score:.3f}")
        if score > best_score:
            best_score = score
            best_model = cand

    logger.info(f"[EMBEDDER SELECTION] Selected optimal model '{best_model}' (readiness score: {best_score:.3f})")
    return best_model


def select_optimal_llm(requested_name: str = "") -> str:
    """
    Selects the optimal local GGUF LLM model.
    Defaults to the 0.5b model when available, or matches the requested name.
    """
    llm_base_dir = os.path.join(MODELS_DIR, "llms")
    if not os.path.exists(llm_base_dir):
        return requested_name or "qwen2.5-coder-0.5b-instruct"

    available = sorted([
        d for d in os.listdir(llm_base_dir)
        if os.path.isdir(os.path.join(llm_base_dir, d)) and any(f.endswith(".gguf") for f in os.listdir(os.path.join(llm_base_dir, d)))
    ])
    if not available:
        return requested_name or "qwen2.5-coder-0.5b-instruct"

    req = (requested_name or "").strip()
    if req and req not in ("auto", "default"):
        if req in available:
            return req
        for cand in available:
            if req.lower() in cand.lower():
                return cand

    # Prioritize 0.8B (Qwen3.5-0.8B), then 0.6B (Qwen3-0.6B), then 0.5B (Qwen2.5-0.5B)
    m_08b = next((m for m in available if "0.8b" in m.lower()), None)
    if m_08b is not None:
        return m_08b
    m_06b = next((m for m in available if "0.6b" in m.lower()), None)
    if m_06b is not None:
        return m_06b
    m_05b = next((m for m in available if "0.5b" in m.lower()), None)
    if m_05b is not None:
        return m_05b

    return available[0]


_GLOBAL_EMBED_CACHE: Dict[Tuple[Any, str, str], List[float]] = {}
_GLOBAL_EMBED_CACHE_LOCK = threading.RLock()


def clear_embedding_cache() -> None:
    """Clears the global in-memory embedding cache."""
    with _GLOBAL_EMBED_CACHE_LOCK:
        _GLOBAL_EMBED_CACHE.clear()


def _encode_with_modes(
    model: Any,
    texts: List[str],
    mode: str = "query",
    batch_size: int = 32,
    lock: Optional[threading.Lock] = None
) -> List[List[float]]:
    """
    Robust embedding encoder applying model-specific prompt modes (query vs document)
    with adaptive fallback across SentenceTransformer models.
    Properly isolates CUDA OOM handling from non-OOM runtime exceptions.
    Uses in-memory LRU caching to eliminate redundant forward passes across route methods.
    """
    if not texts:
        return []

    if not hasattr(model, "_model_cache_id"):
        try:
            import uuid
            model._model_cache_id = f"{type(model).__name__}_{id(model)}_{uuid.uuid4().hex}"
        except Exception:
            model._model_cache_id = f"{type(model).__name__}_{id(model)}"
    m_id = model._model_cache_id
    cached_map: Dict[int, List[float]] = {}
    missing_indices: List[int] = []
    missing_texts: List[str] = []

    with _GLOBAL_EMBED_CACHE_LOCK:
        for idx, txt in enumerate(texts):
            key = (m_id, txt, mode)
            if key in _GLOBAL_EMBED_CACHE:
                cached_map[idx] = _GLOBAL_EMBED_CACHE[key]
            else:
                missing_indices.append(idx)
                missing_texts.append(txt)

    if not missing_texts:
        return [cached_map[i] for i in range(len(texts))]

    prompt_name = "query" if mode == "query" else "document"

    def _invoke(p_name: Optional[str], t_name: Optional[str], b_size: int, input_texts: List[str]):
        kwargs: Dict[str, Any] = {"convert_to_numpy": True, "batch_size": b_size}
        if p_name:
            kwargs["prompt_name"] = p_name
        if t_name:
            kwargs["task"] = t_name

        if lock is not None:
            with lock:
                if TORCH_AVAILABLE:
                    with torch.inference_mode():
                        return model.encode(input_texts, **kwargs)
                return model.encode(input_texts, **kwargs)
        else:
            if TORCH_AVAILABLE:
                with torch.inference_mode():
                    return model.encode(input_texts, **kwargs)
            return model.encode(input_texts, **kwargs)

    attempts = [
        (prompt_name, "retrieval"),
        (prompt_name, None),
        (None, "retrieval"),
        (None, None),
    ]

    last_sig_error = None
    computed: Optional[List[List[float]]] = None

    for p_name, t_name in attempts:
        try:
            res = _invoke(p_name, t_name, batch_size, missing_texts)
            computed = res.tolist() if hasattr(res, "tolist") else [list(r) for r in res]
            break
        except (TypeError, ValueError) as arg_err:
            last_sig_error = arg_err
            continue
        except Exception as err:
            is_oom = TORCH_AVAILABLE and isinstance(err, torch.cuda.OutOfMemoryError)
            if not is_oom and "out of memory" not in str(err).lower():
                logger.error(f"[ENCODER] Fatal encoding error: {err}")
                raise err

            logger.warning(f"[ENCODER] CUDA OOM encountered at batch_size={batch_size}. Retrying with micro-batches.")
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            micro_bs = max(1, batch_size // 4)
            micro_results: List[List[float]] = []
            try:
                for i in range(0, len(missing_texts), micro_bs):
                    sub_batch = missing_texts[i : i + micro_bs]
                    sub_res = _invoke(p_name, t_name, micro_bs, sub_batch)
                    if hasattr(sub_res, "tolist"):
                        micro_results.extend(sub_res.tolist())
                    else:
                        micro_results.extend([list(r) for r in sub_res])
                computed = micro_results
                break
            except (TypeError, ValueError) as arg_err:
                last_sig_error = arg_err
                continue
            except Exception as sub_err:
                is_sub_oom = TORCH_AVAILABLE and isinstance(sub_err, torch.cuda.OutOfMemoryError)
                if is_sub_oom or "out of memory" in str(sub_err).lower():
                    logger.warning(f"[ENCODER] Micro-batch size {micro_bs} triggered OOM. Attempting next signature candidate.")
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        gc.collect()
                        torch.cuda.empty_cache()
                    continue
                logger.error(f"[ENCODER] Fatal error in micro-batch encoding: {sub_err}")
                raise sub_err

    if computed is None:
        # Final guarded fallback if model signature rejected prompt/task arguments
        try:
            if lock is not None:
                with lock:
                    if TORCH_AVAILABLE:
                        with torch.inference_mode():
                            res = model.encode(missing_texts, convert_to_numpy=True)
                    else:
                        res = model.encode(missing_texts, convert_to_numpy=True)
            else:
                if TORCH_AVAILABLE:
                    with torch.inference_mode():
                        res = model.encode(missing_texts, convert_to_numpy=True)
                else:
                    res = model.encode(missing_texts, convert_to_numpy=True)
            computed = res.tolist() if hasattr(res, "tolist") else [list(r) for r in res]
        except Exception as final_err:
            logger.error(f"[ENCODER] Fallback encoding failed: {final_err}")
            raise final_err from last_sig_error

    with _GLOBAL_EMBED_CACHE_LOCK:
        max_size = settings.max_embedding_cache_size
        for idx, txt, emb in zip(missing_indices, missing_texts, computed):
            key = (m_id, txt, mode)
            _GLOBAL_EMBED_CACHE[key] = emb
            cached_map[idx] = emb
        while len(_GLOBAL_EMBED_CACHE) > max_size:
            try:
                first_key = next(iter(_GLOBAL_EMBED_CACHE))
                del _GLOBAL_EMBED_CACHE[first_key]
            except (StopIteration, KeyError):
                break

    return [cached_map[i] for i in range(len(texts))]


class BenchmarkProfile_0(InferenceProfile):
    """
    Profile 0: Pure Symbolic Baseline.
    Pure lexical (BM25 / token intersection). Zero embeddings, zero LLM, zero English hacks.
    """
    def __init__(self):
        self.embedder_name = "none"
        self.llm_name = "none"
        self._dim = 0

    def load_models(self, embedder_name: str = "", llm_name: str = ""):
        pass

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        return []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        return []

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        raise RuntimeError("Profile 0 (Pure Symbolic) does not support neural text generation.")

    def can_synthesize(self) -> bool:
        return False

    def can_extract_slots(self) -> bool:
        return False

    def can_feedback_check(self) -> bool:
        return False

    def has_embedder(self) -> bool:
        return False

    def has_semantic_compiler(self) -> bool:
        return False

    def has_translator_pass(self) -> bool:
        return False

    def uses_reranker(self) -> bool:
        return False

    def has_ir_compiler(self) -> bool:
        return False

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        return failing_code

    @property
    def embedding_dimension(self) -> int:
        return 0


class BenchmarkProfile_B(InferenceProfile):
    """Profile B: Bi-Encoder Dense Embeddings Only (no reranker, no LLM)."""
    def __init__(self):
        self.model = None
        self.embedder = None
        self.llm = None
        self._dim = 384
        self.embedder_name = "default"
        self.llm_name = "none"
        self._lock = threading.Lock()

    def load_models(self, embedder_name: str, llm_name: str = ""):
        from sentence_transformers import SentenceTransformer
        try:
            from .router import HardwareProfiler
        except (ImportError, ValueError):
            from router import HardwareProfiler

        self.embedder_name = select_optimal_embedder(embedder_name)
        self.llm_name = "none"
        self.llm = None
        emb_path = os.path.join(MODELS_DIR, "embeddings", self.embedder_name)
        device = HardwareProfiler.get_optimal_device()
        is_jina = "jina" in self.embedder_name.lower()
        model_kwargs = {"default_task": "retrieval"} if is_jina else {}
        try:
            self.model = SentenceTransformer(emb_path, device=device, trust_remote_code=True, model_kwargs=model_kwargs)
        except Exception:
            self.model = SentenceTransformer(emb_path, device=device, trust_remote_code=True)
        self.embedder = self.model

        self._dim = resolve_embedding_dimension(self.model)
        logger.info(f"[PROFILE B] Loaded embedder '{self.embedder_name}' (dim={self._dim}, no LLM, no reranker) on {device.upper()}")

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        res = self.get_embeddings([text], mode=mode)
        return res[0] if res else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        if not texts or self.model is None:
            return []
        device = getattr(self.model, "device", None)
        dev_str = str(device) if device else "cpu"
        batch_size = get_adaptive_batch_size(dev_str)
        return _encode_with_modes(self.model, texts, mode=mode, batch_size=batch_size, lock=self._lock)

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        raise RuntimeError("Profile B (Dense Embeddings Only) does not support text generation.")

    def can_synthesize(self) -> bool:
        return False

    def can_extract_slots(self) -> bool:
        return False

    def can_feedback_check(self) -> bool:
        return False

    def has_semantic_compiler(self) -> bool:
        return False

    def has_translator_pass(self) -> bool:
        return False

    def has_ir_compiler(self) -> bool:
        return False

    def uses_reranker(self) -> bool:
        return False

    def has_embedder(self) -> bool:
        return True

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        return failing_code

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_A(BenchmarkProfile_B):
    """Profile A: Bi-Encoder Dense Embeddings + Cross-Encoder / Listwise Reranker (no LLM)."""

    def uses_reranker(self) -> bool:
        return True

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        raise RuntimeError("Profile A (Dense Embeddings + Reranker) does not support text generation.")


class BenchmarkProfile_C(InferenceProfile):
    """Profile C: Dedicated Embedder + GGUF LLM."""
    def __init__(self):
        self.embedder = None
        self.llm = None
        self._dim = 384
        self.embedder_name = "default"
        self.llm_name = "default"
        self._embedder_lock = threading.Lock()
        self._llm_lock = threading.Lock()
        self._lock = self._llm_lock  # backward compatibility alias

    def load_models(self, embedder_name: str, llm_name: str):
        from sentence_transformers import SentenceTransformer
        from llama_cpp import Llama
        try:
            from .router import HardwareProfiler
        except (ImportError, ValueError):
            from router import HardwareProfiler

        device = HardwareProfiler.get_optimal_device()

        # 1. Load Embedder
        self.embedder_name = select_optimal_embedder(embedder_name)
        emb_path = os.path.join(MODELS_DIR, "embeddings", self.embedder_name)
        is_jina = "jina" in self.embedder_name.lower()
        model_kwargs = {"default_task": "retrieval"} if is_jina else {}

        try:
            self.embedder = SentenceTransformer(emb_path, device=device, trust_remote_code=True, model_kwargs=model_kwargs)
        except Exception:
            self.embedder = SentenceTransformer(emb_path, device=device, trust_remote_code=True)

        self._dim = resolve_embedding_dimension(self.embedder)

        # 2. Load LLM
        self.llm_name = select_optimal_llm(llm_name)
        llm_base_dir = os.path.join(MODELS_DIR, "llms")
        llm_dir = os.path.join(llm_base_dir, self.llm_name)
        if not os.path.isdir(llm_dir):
            raise FileNotFoundError(f"LLM directory not found: {llm_dir}")

        ggufs = [f for f in os.listdir(llm_dir) if f.endswith(".gguf")]
        if not ggufs:
            raise FileNotFoundError(f"No GGUF model file found in {llm_dir}")
        unified_ggufs = [f for f in ggufs if not is_split_gguf_shard(f)]
        chosen_gguf = unified_ggufs[0] if unified_ggufs else ggufs[0]
        model_file = os.path.join(llm_dir, chosen_gguf)

        gpu_layers = -1 if device in ("cuda", "mps", "metal", "rocm") else 0
        n_threads = max(1, (os.cpu_count() or 4) - 1)
        self.llm = Llama(
            model_path=model_file,
            n_ctx=settings.llm_context_length,
            n_gpu_layers=gpu_layers,
            n_threads=n_threads,
            verbose=False
        )
        logger.info(f"[PROFILE C] Loaded Embedder '{self.embedder_name}' + LLM '{self.llm_name}' on {device.upper()} (gpu_layers={gpu_layers}, n_threads={n_threads})")

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        res = self.get_embeddings([text], mode=mode)
        return res[0] if res else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        if not texts or self.embedder is None:
            return []
        device = getattr(self.embedder, "device", None)
        dev_str = str(device) if device else "cpu"
        batch_size = get_adaptive_batch_size(dev_str)
        return _encode_with_modes(self.embedder, texts, mode=mode, batch_size=batch_size, lock=self._embedder_lock)

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        if self.llm is None:
            raise RuntimeError("LLM is not loaded.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": settings.llm_temperature,
            "top_p": settings.llm_top_p,
            "repeat_penalty": 1.15,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        }
        if schema:
            kwargs["response_format"] = {"type": "json_object", "schema": schema}

        # Concurrency safety: synchronize queries to llama_cpp instance
        with self._llm_lock:
            response = self.llm.create_chat_completion(**kwargs)
        return response['choices'][0]['message']['content'].strip()

    def can_synthesize(self) -> bool:
        return True

    def can_extract_slots(self) -> bool:
        return True

    def can_feedback_check(self) -> bool:
        return True

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        prompt = f"Fix the runtime error in this code and return ONLY the corrected code inside ```python ```:\n\nERROR:\n{traceback_error}\n\nCODE:\n```python\n{failing_code}\n```"
        try:
            raw = self.generate_text(prompt, max_tokens=1024)
            return extract_code_from_llm_response(raw)
        except Exception as e:
            logger.error(f"[FEEDBACK CHECK ERROR] {e}")
            return failing_code

    def has_semantic_compiler(self) -> bool:
        return False

    def has_translator_pass(self) -> bool:
        return False

    def has_ir_compiler(self) -> bool:
        return False

    def compile_semantic_intent(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extracts structured semantic intent, parameters, and topological bindings using the LLM.
        Domain-agnostic with customizable schema and prompt contracts.
        """
        target_schema = schema or {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "domain": {"type": "string"},
                "deliverable": {
                    "type": "string",
                    "description": "The primary terminal entity, variable, model, or metric requested by the user."
                },
                "inputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit input file names, asset paths, or external sources mentioned in the specification."
                },
                "outputs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit output file names or destination paths mentioned in the specification (empty list if deliverable is in-memory)."
                },
                "operations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Strictly ordered sequence of atomic, irreducible computational morphisms. Each step must represent a single discrete transformation or calculation."
                },
                "variables": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "All explicit variable, column, feature, channel, and symbol identifiers mentioned in the specification."
                },
                "parameters": {
                    "type": "object",
                    "description": "Key-value mapping of explicit execution parameters, hyperparameters, and file references extracted from the specification."
                },
                "slots": {
                    "type": "object",
                    "description": "Mapping of named entities, symbolic variables, field/feature identifiers, or target variables referenced in the specification."
                },
                "effective_prompt": {"type": "string"},
            },
            "required": ["task", "domain", "deliverable", "operations", "variables", "inputs"]
        }

        target_system_prompt = system_prompt or (
            "You are a formal semantic compiler for computational pipelines.\n"
            "Analyze the given natural language user specification and decompose it strictly into a JSON object adhering to the schema:\n"
            "- task: high-level intent.\n"
            "- domain: problem domain.\n"
            "- deliverable: the primary terminal entity, variable, model, or metric requested.\n"
            "- operations: strictly ordered sequence of atomic, single-step computational morphisms representing each sequential transformation. "
            "You MUST exhaustively include every operation from the prompt up to and including the terminal deliverable (e.g. data loading, transformations, aggregations, feature extraction, model training, evaluation, predictions, or persistence).\n"
            "- variables: exhaustively list all distinct field names, variable names, target variables, parameters, and symbols explicitly mentioned in the user specification (never omit target or output identifiers, and never append synthetic suffixes unless written verbatim in the text).\n"
            "- inputs: explicit input file names, asset paths, or data sources mentioned in the text (empty list if none).\n"
            "- outputs: explicit destination file paths mentioned in the text (empty list if deliverable is in-memory).\n\n"
            "Guidelines:\n"
            "1. Exhaustive Coverage: operations must NEVER be truncated or omitted; every sequential clause in the prompt must have its corresponding atomic step in operations.\n"
            "2. Extract ALL variables, identifiers, and files that are explicitly mentioned in the user specification without omitting any identifier or target entity.\n"
            "3. Do not invent or hallucinate placeholder names or synthetic suffixes; preserve identifier strings verbatim.\n"
            "4. Output valid JSON only."
        )

        try:
            raw = self.generate_text(prompt, max_tokens=1024, schema=target_schema, system_prompt=target_system_prompt)
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                try:
                    from utils import extract_json_object
                    data = extract_json_object(raw)
                except Exception:
                    pass
            if isinstance(data, dict):
                # Strict literal grounding: destination files must be grounded in prompt text
                raw_outs = data.get("outputs") or []
                valid_outs = []
                prompt_lower = prompt.lower()
                for out_f in raw_outs:
                    if isinstance(out_f, str) and out_f.strip():
                        clean_out = out_f.strip()
                        if clean_out.lower() in prompt_lower:
                            valid_outs.append(clean_out)
                        else:
                            try:
                                from unification import ExecutionContext
                                if any(clean_out.lower() in lit_val.lower() for _, lit_k, lit_val in ExecutionContext._extract_universal_literals(prompt)):
                                    valid_outs.append(clean_out)
                            except Exception:
                                pass
                data["outputs"] = valid_outs
                data["dest_files"] = list(valid_outs)

                # Ground input source_files against prompt text/literals symmetrically
                # so speculative filenames from the small compiler never poison Layer 2 Step 0.
                raw_ins = data.get("source_files") if "source_files" in data else (data.get("inputs") or [])
                if not isinstance(raw_ins, list):
                    raw_ins = [raw_ins] if raw_ins else []
                valid_ins = []
                extracted_lits = None
                for in_f in raw_ins:
                    if isinstance(in_f, str) and in_f.strip():
                        clean_in = in_f.strip()
                        if clean_in.lower() in prompt_lower:
                            valid_ins.append(clean_in)
                        else:
                            try:
                                if extracted_lits is None:
                                    from unification import ExecutionContext
                                    extracted_lits = ExecutionContext._extract_universal_literals(prompt)
                                if any(clean_in.lower() in str(lit_val).lower() for _, _, lit_val in (extracted_lits or ())):
                                    valid_ins.append(clean_in)
                            except Exception:
                                pass
                data["inputs"] = list(valid_ins)
                data["source_files"] = list(valid_ins)
                if "parameters" in data and "hyperparameters" not in data:
                    data["hyperparameters"] = data["parameters"]
                if "variables" in data and isinstance(data["variables"], list):
                    if "slots" not in data or not isinstance(data["slots"], dict):
                        data["slots"] = {}
                    data["slots"]["variables"] = [str(v).strip() for v in data["variables"] if str(v).strip()]
                if "effective_prompt" not in data or not data["effective_prompt"]:
                    data["effective_prompt"] = prompt
                else:
                    try:
                        from utils import ensure_comprehensive_prompt
                        data["effective_prompt"] = ensure_comprehensive_prompt(prompt, data["effective_prompt"])
                    except Exception:
                        pass
                return data
        except Exception as e:
            logger.debug(f"[LLM] Semantic intent compilation fallback: {e}")

        return {
            "task": "generic",
            "domain": "generic",
            "inputs": [],
            "outputs": [],
            "source_files": [],
            "dest_files": [],
            "operations": [],
            "parameters": {},
            "hyperparameters": {},
            "slots": {},
            "effective_prompt": prompt,
        }

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_D(BenchmarkProfile_C):
    """Profile D: Embeddings + Schema-Constrained SLM Slot Extraction Only."""

    def can_extract_slots(self) -> bool:
        return True

    def can_synthesize(self) -> bool:
        return False

    def can_feedback_check(self) -> bool:
        return False

    def has_semantic_compiler(self) -> bool:
        return False

    def has_translator_pass(self) -> bool:
        return False

    def has_ir_compiler(self) -> bool:
        return False


class BenchmarkProfile_E(BenchmarkProfile_C):
    """Profile E: Translator Pass + Semantic Intent Compiler + Neuro-Symbolic Synthesis."""

    def has_semantic_compiler(self) -> bool:
        return True

    def has_translator_pass(self) -> bool:
        return True

    def has_ir_compiler(self) -> bool:
        return False


class BenchmarkProfile_S(BenchmarkProfile_E):
    """Profile S: Typed IR Compiler + Semantic Compiler + Translator + Neuro-Symbolic Synthesis."""

    def has_semantic_compiler(self) -> bool:
        return True

    def has_translator_pass(self) -> bool:
        return True

    def has_ir_compiler(self) -> bool:
        return True

    def compile_semantic_intent(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        return super().compile_semantic_intent(prompt, schema=schema, system_prompt=system_prompt)


class BenchmarkProfile_U(InferenceProfile):
    """
    Profile U: Single-Model Early-Exit Layer Tapping.
    Cuts VRAM footprint in half by loading only ONE LLM.
    Uses intermediate layer k (e.g. k = floor(N/2)) for vector embeddings/RAG,
    and all N layers for generation/unfolding.
    """

    def __init__(self, early_exit_layer: Optional[int] = None):
        self.llm = None
        self.model = None
        self.tokenizer = None
        self._dim = 896
        self.early_exit_layer = early_exit_layer
        self.k_layer = 0
        self.embedder_name = "llm_early_exit"
        self.llm_name = "default"
        self.is_torch = False
        self._lock = threading.Lock()

    def load_models(self, embedder_name: str, llm_name: str):
        self.llm_name = select_optimal_llm(llm_name)
        llm_base_dir = os.path.join(MODELS_DIR, "llms")
        llm_dir = os.path.join(llm_base_dir, self.llm_name)

        device = "cpu"
        try:
            from .router import HardwareProfiler
            device = HardwareProfiler.get_optimal_device()
        except Exception:
            try:
                from router import HardwareProfiler
                device = HardwareProfiler.get_optimal_device()
            except Exception:
                pass

        loaded = False
        # 1. Attempt GGUF / llama_cpp loading with embedding=True
        if os.path.isdir(llm_dir):
            ggufs = [f for f in os.listdir(llm_dir) if f.endswith(".gguf")]
            if ggufs:
                try:
                    import llama_cpp
                    from llama_cpp import Llama
                    unified_ggufs = [f for f in ggufs if not is_split_gguf_shard(f)]
                    chosen_gguf = unified_ggufs[0] if unified_ggufs else ggufs[0]
                    gguf_path = os.path.join(llm_dir, chosen_gguf)
                    n_gpu = -1 if device in ("cuda", "mps", "metal", "rocm") else 0
                    n_threads = max(1, (os.cpu_count() or 4) - 1)
                    mean_pooling = getattr(llama_cpp, "LLAMA_POOLING_TYPE_MEAN", 1)
                    self.llm = Llama(
                        model_path=gguf_path,
                        n_ctx=settings.llm_context_length,
                        n_gpu_layers=n_gpu,
                        n_threads=n_threads,
                        embedding=True,  # Enable dual embedding + completion from single instance
                        pooling_type=mean_pooling,
                        verbose=False,
                    )
                    try:
                        test_emb = self.llm.create_embedding("probe")
                        probe_arr = np.asarray(test_emb["data"][0]["embedding"], dtype=np.float32)
                        self._dim = int(probe_arr.shape[-1]) if probe_arr.ndim >= 1 and probe_arr.size > 0 else 896
                    except Exception:
                        self._dim = getattr(self.llm, "n_embd", lambda: 896)() or 896
                    self.is_torch = False
                    loaded = True
                    logger.info(f"[PROFILE U] Loaded single GGUF LLM '{self.llm_name}' (dim={self._dim}, embedding=True) on {device.upper()}")
                except Exception as e:
                    logger.debug(f"[PROFILE U] GGUF loading skipped: {e}")

        # 2. Attempt PyTorch / HuggingFace model loading
        if not loaded and TORCH_AVAILABLE:
            try:
                if os.path.isdir(llm_dir) and any(os.path.exists(os.path.join(llm_dir, f)) for f in ("config.json", "model.safetensors")):
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    self.tokenizer = AutoTokenizer.from_pretrained(llm_dir, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_pretrained(llm_dir, trust_remote_code=True).to(device)
                    self.model.eval()
                    total_layers = len(getattr(getattr(self.model, "model", self.model), "layers", []))
                    self.k_layer = self.early_exit_layer or max(1, total_layers // 2)
                    self._dim = getattr(self.model.config, "hidden_size", 896)
                    self.is_torch = True
                    loaded = True
                    logger.info(f"[PROFILE U] Loaded single PyTorch LLM '{self.llm_name}' (k={self.k_layer}/{total_layers}, dim={self._dim}) on {device.upper()}")
            except Exception as e:
                logger.debug(f"[PROFILE U] PyTorch model loading skipped: {e}")

        if not loaded:
            # Standalone test/simulation mode: maintain lightweight layer-slicing simulator
            self._dim = 896
            self.k_layer = self.early_exit_layer or 12
            logger.info(f"[PROFILE U] Initialized layer-tapping manifold (k={self.k_layer}, dim={self._dim})")

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        res = self.get_embeddings([text], mode=mode)
        return res[0] if res else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        if not texts:
            return []
        with self._lock:
            # 1. PyTorch early-exit forward pass through layers[:k]
            if self.is_torch and self.model is not None and self.tokenizer is not None:
                device = getattr(self.model, "device", "cpu")
                inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                with torch.no_grad():
                    core_model = getattr(self.model, "model", self.model)
                    hidden = core_model.embed_tokens(inputs.input_ids)
                    seq_len = inputs.input_ids.shape[1]
                    batch_size = inputs.input_ids.shape[0]
                    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, seq_len)
                    rotary_pos_emb = None
                    if hasattr(core_model, "rotary_emb"):
                        try:
                            rotary_pos_emb = core_model.rotary_emb(hidden, position_ids)
                        except Exception:
                            rotary_pos_emb = None
                    causal_mask = None
                    attn_2d = getattr(inputs, "attention_mask", None)
                    if attn_2d is not None:
                        min_val = torch.finfo(hidden.dtype).min if hidden.dtype.is_floating_point else -1e9
                        causal_base = torch.triu(
                            torch.full((seq_len, seq_len), min_val, dtype=hidden.dtype, device=device),
                            diagonal=1,
                        ).unsqueeze(0).unsqueeze(0)
                        pad_mask = (1.0 - attn_2d.to(dtype=hidden.dtype)).unsqueeze(1).unsqueeze(2) * min_val
                        causal_mask = causal_base + pad_mask

                    k = self.k_layer
                    layers = getattr(core_model, "layers", [])
                    for layer in layers[:k]:
                        layer_kwargs: Dict[str, Any] = {}
                        try:
                            sig = inspect.signature(layer.forward)
                            params = sig.parameters
                            if "position_ids" in params:
                                layer_kwargs["position_ids"] = position_ids
                            if "position_embeddings" in params and rotary_pos_emb is not None:
                                layer_kwargs["position_embeddings"] = rotary_pos_emb
                            if "attention_mask" in params and causal_mask is not None:
                                layer_kwargs["attention_mask"] = causal_mask
                        except (TypeError, ValueError):
                            layer_kwargs = {"position_ids": position_ids}
                        layer_out = layer(hidden, **layer_kwargs)
                        hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                    if hasattr(core_model, "norm"):
                        hidden = core_model.norm(hidden)
                    mask = inputs.attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                    sum_emb = torch.sum(hidden * mask, dim=1)
                    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                    pooled = sum_emb / sum_mask
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    return normalized.detach().cpu().numpy().tolist()

            # 2. GGUF / llama_cpp native embedding pooling endpoint
            if self.llm is not None and hasattr(self.llm, "create_embedding"):
                embeddings = []
                for t in texts:
                    res = self.llm.create_embedding(t)
                    emb = res["data"][0]["embedding"]
                    arr = np.asarray(emb, dtype=np.float32)
                    if arr.ndim > 1:
                        arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
                    norm = float(np.linalg.norm(arr))
                    if norm > 1e-9:
                        arr = arr / norm
                    embeddings.append(arr.tolist())
                return embeddings

            # 3. Deterministic compositional projection fallback for offline/test mode (zero MD5 RNG)
            try:
                from .latent_autoencoder import _extract_structural_features, _project_compositional_features
            except (ImportError, ValueError):
                from latent_autoencoder import _extract_structural_features, _project_compositional_features
            return [
                _project_compositional_features(
                    _extract_structural_features(t, is_typestate=True),
                    self._dim,
                ).tolist()
                for t in texts
            ]

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        with self._lock:
            if self.is_torch and self.model is not None and self.tokenizer is not None:
                device = getattr(self.model, "device", "cpu")
                full_p = f"{system_prompt}\n{prompt}" if system_prompt else prompt
                inputs = self.tokenizer(full_p, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = self.model.generate(**inputs, max_new_tokens=max_tokens)
                return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            if self.llm is not None:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                kwargs: Dict[str, Any] = {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": settings.llm_temperature,
                    "top_p": settings.llm_top_p,
                    "repeat_penalty": 1.15,
                    "stop": ["<|im_end|>", "<|endoftext|>"],
                }
                if schema:
                    kwargs["response_format"] = {"type": "json_object", "schema": schema}
                res = self.llm.create_chat_completion(**kwargs)
                return res["choices"][0]["message"]["content"].strip()
            return ""

    def can_synthesize(self) -> bool:
        return True

    def can_extract_slots(self) -> bool:
        return True

    def can_feedback_check(self) -> bool:
        return True

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        prompt = (
            "Fix the runtime error in this code and return ONLY the corrected code "
            f"inside ```python ```:\n\nERROR:\n{traceback_error}\n\nCODE:\n```python\n{failing_code}\n```"
        )
        try:
            raw = self.generate_text(prompt, max_tokens=1024)
            repaired = extract_code_from_llm_response(raw)
            return repaired if repaired else failing_code
        except Exception as e:
            logger.error(f"[PROFILE U FEEDBACK CHECK ERROR] {e}")
            return failing_code

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_AE(InferenceProfile):
    """
    Profile AE: Self-Supervised Structural Transition Autoencoder Manifold.
    Projects discrete node signatures and typestates into a continuous transition manifold.
    Enables routing via fast continuous vector arithmetic: z_{tau_in} + t_c ≈ z_{tau_out}.
    """

    def __init__(self, dimension: int = 64):
        self.autoencoder: Optional[Any] = None
        self._dim = dimension
        self.embedder_name = "transition_autoencoder"
        self.llm_name = "none"
        self._lock = threading.Lock()

    def load_models(self, embedder_name: str = "", llm_name: str = ""):
        try:
            from .latent_autoencoder import TransitionAutoencoder, train_autoencoder_from_orchestrator
        except (ImportError, ValueError):
            from latent_autoencoder import TransitionAutoencoder, train_autoencoder_from_orchestrator

        try:
            from .main import get_orchestrator
            orch = get_orchestrator()
        except Exception:
            orch = None

        if orch is not None and getattr(orch, "loaded_cells", None):
            self.autoencoder = train_autoencoder_from_orchestrator(orch, dimension=self._dim)
        else:
            self.autoencoder = TransitionAutoencoder(dimension=self._dim)
            self.autoencoder.train([], epochs=30)

        logger.info(f"[PROFILE AE] Initialized Structural Transition Autoencoder (dim={self._dim})")

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        res = self.get_embeddings([text], mode=mode)
        return res[0] if res else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        if not texts or self.autoencoder is None:
            return []
        with self._lock:
            return [self.autoencoder.encode_signature(t).tolist() for t in texts]

    def score_transition(
        self,
        state_vec: Union[np.ndarray, str, Any],
        cell_id: str,
        target_vec: Union[np.ndarray, str, Any],
    ) -> float:
        if self.autoencoder is None:
            return 0.0
        with self._lock:
            return self.autoencoder.score_transition(state_vec, cell_id, target_vec)

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        raise RuntimeError("Profile AE is a continuous transition manifold and does not support generative text.")

    def can_synthesize(self) -> bool:
        return False

    def can_feedback_check(self) -> bool:
        return False

    def has_embedder(self) -> bool:
        return True

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        return failing_code

    @property
    def embedding_dimension(self) -> int:
        return self._dim


class BenchmarkProfile_Omega(BenchmarkProfile_S):
    """
    Profile Omega (O / Ω): Full STRATA 4-Stage Formal Neuro-Symbolic Pipeline.
      - Stage 1: Recursive Coalgebraic Subgoal Unfolding (open -> semi-closed -> closed)
      - Stage 2: Calibrated Sigmoid Hybrid Retrieval + Backward Precondition Cone
      - Stage 3: Type-Directed Bidirectional A* Search over Symmetric Monoidal Category
                 with schema-constrained SLM ranking over E_valid(Gamma_t)
      - Stage 4: Exact Robinson Unification + Profunctor Lens Lifting + GBNF Slot Extraction
                 + Pointed Monoidal Defaults eta_tau : 1 -> tau
    """

    def uses_reranker(self) -> bool:
        return True

    def has_coalgebraic_unfolder(self) -> bool:
        return True

    def has_constrained_choice_scorer(self) -> bool:
        return True

    def score_choices(
        self,
        prompt: str,
        candidates: List[str],
        context_desc: str = "",
    ) -> Dict[str, float]:
        """
        Schema-constrained SLM choice scorer restricted strictly to E_valid(Gamma_t).
        Returns {cell_id: score in [0, 1]} only for candidate IDs in `candidates`.
        """
        clean_cands = [str(c).strip() for c in (candidates or []) if c]
        has_custom_gen = type(self).generate_text is not BenchmarkProfile_C.generate_text
        if not clean_cands or (self.llm is None and not has_custom_gen):
            return {}
        capped = clean_cands[:12]
        choice_schema = {
            "type": "object",
            "properties": {
                "rankings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cell_id": {"type": "string", "enum": capped},
                            "score": {"type": "number"},
                        },
                        "required": ["cell_id", "score"],
                    },
                }
            },
            "required": ["rankings"],
        }
        ctx_line = f"\nCurrent typed context Gamma_t:\n{context_desc}\n" if context_desc else "\n"
        sys_p = (
            "You rank ONLY type-valid outgoing morphisms from the provided candidate list.\n"
            "Do not invent new cell IDs. Assign scores in [0.0, 1.0] reflecting how well each "
            "candidate advances the goal."
        )
        usr_p = (
            f"Goal: {prompt}{ctx_line}"
            f"Type-valid candidate morphisms E_valid: {', '.join(capped)}\n"
            'Return JSON: {"rankings": [{"cell_id": "<id>", "score": <0..1>}, ...]}'
        )
        try:
            raw = self.generate_text(usr_p, max_tokens=256, schema=choice_schema, system_prompt=sys_p)
            data = None
            try:
                data = json.loads(raw)
            except Exception:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end > start:
                    data = json.loads(raw[start : end + 1])
            if not isinstance(data, dict):
                return {}
            valid_set = set(capped)
            out: Dict[str, float] = {}
            for item in data.get("rankings", []) or []:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("cell_id", "")).strip()
                if cid in valid_set:
                    try:
                        sc = max(0.0, min(1.0, float(item.get("score", 0.0))))
                        out[cid] = max(out.get(cid, 0.0), sc)
                    except Exception:
                        pass
            return out
        except Exception as err:
            logger.debug(f"[OMEGA] Constrained choice scoring skipped: {err}")
            return {}


class ModelManager:
    _instance = None
    _lock = threading.RLock()

    def __init__(self):
        self.active_profile: Optional[InferenceProfile] = None
        self.current_profile_name: Optional[str] = None

    @classmethod
    def get_instance(cls) -> ModelManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def profile(self) -> Optional[InferenceProfile]:
        with self._lock:
            return self.active_profile

    def cleanup(self):
        """Forces complete teardown of active models, releasing all GPU VRAM and CPU RAM."""
        with self._lock:
            if self.active_profile is not None:
                if hasattr(self.active_profile, 'llm') and self.active_profile.llm is not None:
                    try:
                        self.active_profile.llm.close()
                    except Exception:
                        pass
                    self.active_profile.llm = None
                if hasattr(self.active_profile, 'embedder') and self.active_profile.embedder is not None:
                    self.active_profile.embedder = None
                if hasattr(self.active_profile, 'model') and self.active_profile.model is not None:
                    self.active_profile.model = None
                self.active_profile = None
            self.current_profile_name = None
            try:
                from cognitive_pathfinder import LatentVectorExtractor
                LatentVectorExtractor.reset_instance()
            except Exception:
                pass

            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info("[MODEL MANAGER] Completed VRAM cleanup and memory reclamation.")

    def initialize_profile(self, profile_type: str, embedder_name: str = "", llm_name: str = ""):
        with self._lock:
            self.cleanup()

            p_type = profile_type.upper()
            if p_type in ("0", "SYMBOLIC", "ZERO", "PURE"):
                prof = BenchmarkProfile_0()
                p_type = "0"
            elif p_type == "A":
                prof = BenchmarkProfile_A()
            elif p_type == "B":
                prof = BenchmarkProfile_B()
            elif p_type == "C":
                prof = BenchmarkProfile_C()
            elif p_type == "D":
                prof = BenchmarkProfile_D()
            elif p_type == "E":
                prof = BenchmarkProfile_E()
            elif p_type == "S":
                prof = BenchmarkProfile_S()
            elif p_type == "U":
                prof = BenchmarkProfile_U()
            elif p_type == "AE":
                prof = BenchmarkProfile_AE()
            elif p_type in ("O", "OMEGA", "Ω"):
                prof = BenchmarkProfile_Omega()
                p_type = "O"
            else:
                raise ValueError(f"Unknown profile type: {profile_type}")

            try:
                prof.load_models(embedder_name, llm_name)
                self.active_profile = prof
                self.current_profile_name = p_type
            except Exception as e:
                self.cleanup()
                raise e

    def get_embedding(self, text: str, mode: str = "query") -> List[float]:
        with self._lock:
            return self.active_profile.get_embedding(text, mode=mode) if self.active_profile else []

    def get_embeddings(self, texts: List[str], mode: str = "query") -> List[List[float]]:
        with self._lock:
            return self.active_profile.get_embeddings(texts, mode=mode) if self.active_profile else []

    def get_embeddings_batch(self, texts: List[str], mode: str = "query") -> np.ndarray:
        with self._lock:
            if not self.active_profile or not texts:
                dim = getattr(self.active_profile, "_dim", 384) if self.active_profile else 384
                return np.empty((0, dim), dtype=np.float32)
            res = self.active_profile.get_embeddings(texts, mode=mode)
            return np.array(res, dtype=np.float32)

    def generate_text(self, prompt: str, max_tokens: int = 1024, schema: Optional[dict] = None, system_prompt: Optional[str] = None) -> str:
        with self._lock:
            if not self.active_profile:
                return ""
            return self.active_profile.generate_text(prompt, max_tokens, schema, system_prompt=system_prompt)

    def can_synthesize(self) -> bool:
        with self._lock:
            return self.active_profile.can_synthesize() if self.active_profile else False

    def can_extract_slots(self) -> bool:
        with self._lock:
            if not self.active_profile:
                return False
            if hasattr(self.active_profile, "can_extract_slots"):
                return self.active_profile.can_extract_slots()
            return self.active_profile.can_synthesize()

    def is_pure_symbolic(self) -> bool:
        with self._lock:
            return self.current_profile_name == "0"

    def uses_reranker(self) -> bool:
        with self._lock:
            return getattr(self.active_profile, "uses_reranker", lambda: False)() if self.active_profile else False

    def has_ir_compiler(self) -> bool:
        with self._lock:
            return getattr(self.active_profile, "has_ir_compiler", lambda: False)() if self.active_profile else False

    def has_coalgebraic_unfolder(self) -> bool:
        with self._lock:
            return getattr(self.active_profile, "has_coalgebraic_unfolder", lambda: False)() if self.active_profile else False

    def has_constrained_choice_scorer(self) -> bool:
        with self._lock:
            return getattr(self.active_profile, "has_constrained_choice_scorer", lambda: False)() if self.active_profile else False

    def score_choices(
        self,
        prompt: str,
        candidates: List[str],
        context_desc: str = "",
    ) -> Dict[str, float]:
        with self._lock:
            if self.active_profile and hasattr(self.active_profile, "score_choices"):
                return self.active_profile.score_choices(prompt, candidates, context_desc=context_desc)
            return {}

    def has_embedder(self) -> bool:
        with self._lock:
            if not self.active_profile:
                return False
            if hasattr(self.active_profile, "has_embedder"):
                return self.active_profile.has_embedder()
            return getattr(self.active_profile, "embedding_dimension", 0) > 0

    def can_feedback_check(self) -> bool:
        with self._lock:
            return self.active_profile.can_feedback_check() if self.active_profile else False

    def has_translator_pass(self) -> bool:
        with self._lock:
            return self.active_profile.has_translator_pass() if self.active_profile else False

    def has_semantic_compiler(self) -> bool:
        with self._lock:
            return getattr(self.active_profile, "has_semantic_compiler", lambda: False)() if self.active_profile else False

    def compile_semantic_intent(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        with self._lock:
            if self.active_profile and hasattr(self.active_profile, "compile_semantic_intent"):
                return self.active_profile.compile_semantic_intent(prompt, schema=schema, system_prompt=system_prompt)
            return {"effective_prompt": prompt}

    def score_transition(
        self,
        state_vec: Union[np.ndarray, str, Any],
        cell_id: str,
        target_vec: Union[np.ndarray, str, Any],
    ) -> float:
        with self._lock:
            if self.active_profile and hasattr(self.active_profile, "score_transition"):
                return self.active_profile.score_transition(state_vec, cell_id, target_vec)
            return 0.0

    def feedback_check(self, failing_code: str, traceback_error: str) -> str:
        with self._lock:
            return self.active_profile.feedback_check(failing_code, traceback_error) if self.active_profile else failing_code

    @property
    def embedding_dimension(self) -> int:
        with self._lock:
            return self.active_profile.embedding_dimension if self.active_profile else 384
