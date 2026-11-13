"""
src/reranker.py - Neural Document Reranking for Layer 1 Semantic Tunneling
Provides pluggable neural cross-encoder / listwise reranking (e.g. Jina Reranker v3.5).
"""
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    from transformers import AutoModel
    _TORCH_AVAILABLE = True
except ImportError:  # torch/transformers are optional heavy deps; reranking degrades to no-op
    torch = None
    AutoModel = None
    _TORCH_AVAILABLE = False

from config import settings, MODELS_DIR, RERANKERS_DIR
from lattice import Cell
try:
    from log_config import get_logger
except ImportError:
    from .log_config import get_logger

logger = get_logger("reranker")


def get_available_rerankers() -> List[str]:
    """Scans MODELS_DIR/rerankers for available reranker models."""
    r_dir = settings.rerankers_dir or os.path.join(MODELS_DIR, 'rerankers')
    if os.path.exists(str(r_dir)):
        raw = sorted([
            d for d in os.listdir(str(r_dir))
            if os.path.isdir(os.path.join(str(r_dir), d))
        ])
        jina = [d for d in raw if 'jina' in d.lower()]
        rest = [d for d in raw if 'jina' not in d.lower()]
        return jina + rest
    return []


class LocalReranker:
    """
    Neural cross-encoder / listwise reranker that refines RAG candidate ranking
    prior to topological planning.
    """
    def __init__(self, model_name_or_path: Optional[str] = None, device: str = 'auto'):
        if not _TORCH_AVAILABLE:
            logger.warning("[RERANKER] torch/transformers unavailable; neural reranking disabled (no-op).")
            self.device = 'cpu'
            self.model_name = model_name_or_path or self._resolve_default_model()
            self.model = None
            return
        self.device = (
            'cuda' if (device == 'auto' and torch.cuda.is_available())
            else ('cpu' if device == 'auto' else device)
        )
        self.model_name = model_name_or_path or self._resolve_default_model()
        self.model = None
        self._load_model()

    def _resolve_default_model(self) -> str:
        available = get_available_rerankers()
        if available:
            return available[0]
        return 'jinaai/jina-reranker-v3.5'

    def _load_model(self):
        r_dir = settings.rerankers_dir or os.path.join(MODELS_DIR, 'rerankers')
        local_cand = os.path.join(str(r_dir), self.model_name)
        model_path = local_cand if os.path.exists(local_cand) else self.model_name
        logger.info(f"[RERANKER] Loading reranker '{self.model_name}' from {model_path} onto {self.device}...")
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype='auto',
            ).to(self.device)
            self.model.eval()
            logger.info(f"[RERANKER] Successfully initialized {self.model_name}.")
        except Exception as e:
            logger.error(f"[RERANKER] Failed to load reranker model '{self.model_name}': {e}")
            self.model = None

    @staticmethod
    def _cell_to_document(cell: Cell) -> str:
        """Constructs a rich text representation of a lattice cell for reranking."""
        doc = getattr(cell, 'docstring', '') or getattr(cell, 'doc', '') or ''
        domain = getattr(cell, 'domain_name', '') or ''
        tags = getattr(cell, 'semantic_tags', []) or []
        keywords = getattr(cell, 'keywords', []) or []
        in_t = getattr(cell.primary_input, 'type_name', '') if getattr(cell, 'primary_input', None) else ''
        out_t = getattr(cell.primary_output, 'type_name', '') if getattr(cell, 'primary_output', None) else ''

        parts = [f'Cell {cell.cell_id}: {doc}']
        if domain:
            parts.append(f'Domain: {domain}')
        if in_t or out_t:
            parts.append(f'Type signature: {in_t} -> {out_t}')
        if keywords:
            kw_list = sorted(str(k) for k in keywords) if isinstance(keywords, set) else [str(k) for k in keywords]
            parts.append(f'Keywords: {", ".join(kw_list)}')
        if tags:
            tag_list = sorted(str(t) for t in tags) if isinstance(tags, set) else [str(t) for t in tags]
            parts.append(f'Tags: {", ".join(tag_list)}')
        inputs = getattr(cell, 'inputs', {}) or {}
        if isinstance(inputs, dict) and inputs:
            in_descs = []
            for p_sig in inputs.values():
                p_desc = (getattr(p_sig, 'description', '') or getattr(p_sig, 'doc', '') or '').strip()
                if p_desc:
                    in_descs.append(p_desc)
            if in_descs:
                parts.append(f'Parameters: {" ".join(in_descs)}')
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

        documents = [self._cell_to_document(c) for c in target_cells]
        try:
            import inspect
            rerank_kwargs = {"top_n": len(documents)}
            sig = inspect.signature(self.model.rerank)
            if "max_query_length" in sig.parameters:
                rerank_kwargs["max_query_length"] = 512
            results = self.model.rerank(
                query,
                documents,
                **rerank_kwargs,
            )
        except Exception as e:
            logger.warning(f'[RERANKER] Reranking inference failed: {e}. Falling back to RAG.')
            return candidate_cells, relevance_map, []

        old_ranks = {c.cell_id: i + 1 for i, c in enumerate(candidate_cells)}
        old_scores = {c.cell_id: float(relevance_map.get(c.cell_id, 0.0)) for c in candidate_cells}

        is_cosine_model = hasattr(self.model, "_calculate_cosine_scores") or "jina" in str(self.model_name).lower()
        new_scores: Dict[str, float] = dict(relevance_map)
        scored_cells: List[Cell] = []

        for res in results:
            orig_doc_idx = res['index']
            cell = target_cells[orig_doc_idx]
            calib_sc = self._calibrate_score(float(res['relevance_score']), is_cosine_model=is_cosine_model)
            blended_sc = round(0.7 * calib_sc + 0.3 * old_scores.get(cell.cell_id, 0.0), 4)
            new_scores[cell.cell_id] = blended_sc
            scored_cells.append(cell)

        calib_by_id: Dict[str, float] = {}
        for res in results:
            cell = target_cells[res['index']]
            calib_by_id[cell.cell_id] = round(
                self._calibrate_score(float(res['relevance_score']), is_cosine_model=is_cosine_model), 4
            )

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
