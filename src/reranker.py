"""
src/reranker.py - Neural Document Reranking for Layer 1 Semantic Tunneling
Provides pluggable neural cross-encoder / listwise reranking (e.g. Jina Reranker v3.5).
"""
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import inspect
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

torch = None  # Lazily imported on first neural reranker instantiation
AutoModel = None
_TORCH_AVAILABLE: Optional[bool] = None
_SHARED_RERANKERS: Dict[Tuple[str, str], "LocalReranker"] = {}
_SHARED_RERANKER_LOCK = threading.RLock()


def _ensure_reranker_imports() -> bool:
    """Lazily imports torch and transformers.AutoModel only when reranking is active."""
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

from config import settings, MODELS_DIR, RERANKERS_DIR
from lattice import Cell
try:
    from log_config import get_logger
except ImportError:
    from .log_config import get_logger

logger = get_logger("reranker")


def get_available_rerankers() -> List[str]:
    """Scans MODELS_DIR/rerankers for available reranker models (sorted, optional config preference)."""
    r_dir = settings.rerankers_dir or os.path.join(MODELS_DIR, 'rerankers')
    if os.path.exists(str(r_dir)):
        raw = sorted([
            d for d in os.listdir(str(r_dir))
            if os.path.isdir(os.path.join(str(r_dir), d))
        ])
        prefs = [str(p).strip() for p in (getattr(settings, "preferred_rerankers", None) or ()) if str(p).strip()]
        if prefs:
            pref_list = [d for p in prefs for d in raw if d == p]
            rest = [d for d in raw if d not in pref_list]
            return pref_list + rest
        return raw
    return []


class LocalReranker:
    """
    Neural cross-encoder / listwise reranker that refines RAG candidate ranking
    prior to topological planning.
    """
    @classmethod
    def get_shared(cls, model_name_or_path: Optional[str] = None, device: str = 'auto') -> "LocalReranker":
        """Returns a process-wide cached LocalReranker so multiple shells share one VRAM copy."""
        with _SHARED_RERANKER_LOCK:
            norm_name = model_name_or_path or settings.reranker_model or ''
            if not norm_name:
                avail = get_available_rerankers()
                if not avail:
                    raise FileNotFoundError(
                        f"No reranker model specified and none found in '{settings.rerankers_dir}'."
                    )
                norm_name = avail[0]
            norm_dev = device
            if device == 'auto':
                norm_dev = 'cuda' if (_ensure_reranker_imports() and torch.cuda.is_available()) else 'cpu'
            key = (norm_name, norm_dev)
            inst = _SHARED_RERANKERS.get(key)
            if inst is None or inst.model is None:
                inst = cls(model_name_or_path=model_name_or_path, device=device)
                _SHARED_RERANKERS[key] = inst
            return inst

    @classmethod
    def release_shared(cls) -> None:
        """Releases cached shared reranker instances to reclaim VRAM/RAM."""
        with _SHARED_RERANKER_LOCK:
            for inst in _SHARED_RERANKERS.values():
                inst.model = None
            _SHARED_RERANKERS.clear()
        try:
            import gc as _gc
            _gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                from .inference import _trim_host_memory
            except (ImportError, ValueError):
                from inference import _trim_host_memory
            _trim_host_memory()
        except Exception:
            pass

    def __init__(self, model_name_or_path: Optional[str] = None, device: str = 'auto'):
        if not _ensure_reranker_imports():
            logger.warning("[RERANKER] torch/transformers unavailable; neural reranking disabled (no-op).")
            self.device = 'cpu'
            self.model_name = model_name_or_path or self._resolve_default_model()
            self.model = None
            self._supports_max_query_len = False
            self._doc_cache: Dict[str, str] = {}
            self._score_cache: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[int, float]]] = {}
            return
        self.device = (
            'cuda' if (device == 'auto' and torch.cuda.is_available())
            else ('cpu' if device == 'auto' else device)
        )
        self.model_name = model_name_or_path or self._resolve_default_model()
        self.model = None
        self._supports_max_query_len = False
        self._doc_cache: Dict[str, str] = {}
        self._score_cache: Dict[Tuple[str, Tuple[str, ...]], List[Tuple[int, float]]] = {}
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

    def _load_model(self):
        r_dir = settings.rerankers_dir or os.path.join(MODELS_DIR, 'rerankers')
        local_cand = os.path.join(str(r_dir), self.model_name)
        model_path = local_cand if os.path.exists(local_cand) else self.model_name
        logger.info(f"[RERANKER] Loading reranker '{self.model_name}' from {model_path} onto {self.device}...")
        try:
            try:
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype='auto',
                    low_cpu_mem_usage=True,
                ).to(self.device)
            except TypeError:
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype='auto',
                ).to(self.device)
            self.model.eval()
            try:
                import gc as _gc
                _gc.collect()
                try:
                    from .inference import _trim_host_memory
                except (ImportError, ValueError):
                    from inference import _trim_host_memory
                _trim_host_memory()
            except Exception:
                pass
            try:
                self._supports_max_query_len = "max_query_length" in inspect.signature(self.model.rerank).parameters
            except Exception:
                self._supports_max_query_len = False
            logger.info(f"[RERANKER] Successfully initialized {self.model_name}.")
        except Exception as e:
            logger.error(f"[RERANKER] Failed to load reranker model '{self.model_name}': {e}")
            self.model = None
            self._supports_max_query_len = False

    @staticmethod
    def _cell_to_document(cell: Cell) -> str:
        """Constructs a concise semantic text representation of a lattice cell for fast reranking."""
        raw_doc = (getattr(cell, 'docstring', '') or getattr(cell, 'doc', '') or '').strip()
        if raw_doc:
            first_line = " ".join(raw_doc.split())
            if len(first_line) > 160:
                cut = first_line[:160].rsplit(' ', 1)[0]
                first_line = cut if cut else first_line[:160]
        else:
            first_line = ""
        domain = getattr(cell, 'domain_name', '') or ''
        tags = getattr(cell, 'semantic_tags', ()) or ()
        keywords = getattr(cell, 'keywords', ()) or ()
        in_t = getattr(cell.primary_input, 'type_name', '') if getattr(cell, 'primary_input', None) else ''
        out_t = getattr(cell.primary_output, 'type_name', '') if getattr(cell, 'primary_output', None) else ''

        parts = [f'Cell {cell.cell_id}: {first_line}' if first_line else f'Cell {cell.cell_id}']
        if domain:
            parts.append(f'Domain: {domain}')
        if in_t or out_t:
            parts.append(f'Sig: {in_t} -> {out_t}')
        seen_terms: List[str] = []
        seen_low: set = set()
        for src in (keywords, tags):
            seq = sorted(str(k) for k in src) if isinstance(src, set) else [str(k) for k in src]
            for term in seq:
                tl = term.strip().lower()
                if tl and tl not in seen_low:
                    seen_low.add(tl)
                    seen_terms.append(term.strip())
                    if len(seen_terms) >= 10:
                        break
            if len(seen_terms) >= 10:
                break
        if seen_terms:
            parts.append(f'Terms: {", ".join(seen_terms)}')
        return ' | '.join(parts)

    @staticmethod
    def _calibrate_score(raw_sc: float, is_cosine_model: bool = False) -> float:
        """
        Batch-independent monotonic sigmoid calibration mapping raw reranker scores to (0, 1).
        Replaces batch min-max normalization so irrelevant batches are never inflated to 1.0.
        For bounded cosine-based listwise rerankers (e.g. jina-reranker-v3.5 outputting [-1, 1]),
        applies temperature-calibrated logistic scaling around empirical cosine separation.
        For unbounded logit cross-encoders, applies standard logistic sigmoid sigma(logit).
        """
        val = float(raw_sc)
        logit = (10.0 * (val - 0.05)) if is_cosine_model else val
        logit = max(-60.0, min(60.0, logit))
        return 1.0 / (1.0 + math.exp(-logit))

    def rerank(
        self,
        query: str,
        candidate_cells: List[Cell],
        relevance_map: Dict[str, float],
        top_k: int = 50,
    ) -> Tuple[List[Cell], Dict[str, float], List[Dict[str, Any]]]:
        """
        Reranks candidate cells against the prompt query.
        Returns:
            - reranked_cells: New ordered list of cells.
            - reranked_relevance_map: Dict[cell_id, new_score].
            - telemetry: List of per-cell diff entries:
                {'cell_id': cid, 'old_rank': r_old, 'new_rank': r_new,
                 'old_score': s_old, 'new_score': s_new, 'delta_rank': r_old - r_new,
                 'delta_score': s_new - s_old}
        """
        if not self.model or not candidate_cells:
            return candidate_cells, relevance_map, []

        pool_size = min(len(candidate_cells), max(top_k, 50))
        target_cells = candidate_cells[:pool_size]
        remainder_cells = candidate_cells[pool_size:]

        target_cids = tuple(getattr(c, 'cell_id', '') for c in target_cells)
        cache_key = (query, target_cids)
        cached_pairs = self._score_cache.get(cache_key)
        if cached_pairs is None:
            documents = []
            for c, cid in zip(target_cells, target_cids):
                doc = self._doc_cache.get(cid)
                if doc is None:
                    doc = self._cell_to_document(c)
                    if cid:
                        self._doc_cache[cid] = doc
                documents.append(doc)
            try:
                rerank_kwargs = {"top_n": len(documents)}
                if getattr(self, "_supports_max_query_len", False):
                    rerank_kwargs["max_query_length"] = 512
                if torch is not None:
                    with torch.inference_mode():
                        results = self.model.rerank(
                            query,
                            documents,
                            **rerank_kwargs,
                        )
                else:
                    results = self.model.rerank(
                        query,
                        documents,
                        **rerank_kwargs,
                    )
            except Exception as e:
                logger.warning(f'[RERANKER] Reranking inference failed: {e}. Falling back to RAG.')
                return candidate_cells, relevance_map, []
            cached_pairs = [(int(res['index']), float(res['relevance_score'])) for res in results]
            if len(self._score_cache) >= 256:
                try:
                    del self._score_cache[next(iter(self._score_cache))]
                except (StopIteration, KeyError):
                    self._score_cache.clear()
            self._score_cache[cache_key] = cached_pairs

        old_ranks = {c.cell_id: i + 1 for i, c in enumerate(candidate_cells)}
        old_scores = {c.cell_id: float(relevance_map.get(c.cell_id, 0.0)) for c in candidate_cells}

        is_cosine_model = hasattr(self.model, "_calculate_cosine_scores")
        new_scores: Dict[str, float] = dict(relevance_map)
        scored_cells: List[Cell] = []
        calib_by_id: Dict[str, float] = {}

        for orig_doc_idx, raw_sc in cached_pairs:
            cell = target_cells[orig_doc_idx]
            calib_sc = self._calibrate_score(raw_sc, is_cosine_model=is_cosine_model)
            calib_by_id[cell.cell_id] = round(calib_sc, 4)
            blended_sc = round(0.7 * calib_sc + 0.3 * old_scores.get(cell.cell_id, 0.0), 4)
            new_scores[cell.cell_id] = blended_sc
            scored_cells.append(cell)

        reranked_target_cells = sorted(
            scored_cells,
            key=lambda c: new_scores.get(c.cell_id, 0.0),
            reverse=True,
        )
        telemetry: List[Dict[str, Any]] = []
        for new_idx, cell in enumerate(reranked_target_cells):
            old_r = old_ranks.get(cell.cell_id, new_idx + 1)
            new_r = new_idx + 1
            blended_sc = new_scores.get(cell.cell_id, 0.0)
            telemetry.append({
                'cell_id': cell.cell_id,
                'domain': getattr(cell, 'domain_name', ''),
                'old_rank': old_r,
                'new_rank': new_r,
                'delta_rank': old_r - new_r,
                'old_score': old_scores.get(cell.cell_id, 0.0),
                'calibrated_score': calib_by_id.get(cell.cell_id, blended_sc),
                'new_score': blended_sc,
                'delta_score': round(blended_sc - old_scores.get(cell.cell_id, 0.0), 4),
            })

        final_cells = reranked_target_cells + remainder_cells
        return final_cells, new_scores, telemetry
