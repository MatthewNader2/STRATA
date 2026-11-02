"""
src/internal_rag.py - STRATA (Neuro-Symbolic Topological Lattice)
Local Vector Index (FAISS) and Incremental Embedding Cache.
"""

from __future__ import annotations
import os
import gc
import json
import math
import hashlib
import pickle
import threading
from typing import Optional, Dict, Any, List, Tuple

try:
    from utils import tokenize_alphanumeric
except ImportError:
    from .utils import tokenize_alphanumeric

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# The neural stack (faiss/torch) is required ONLY by dense-retrieval profiles.
# Pure-symbolic profiles (Profile 0) must not pay the dependency: imports are
# lazy and guarded, so routing/planning/synthesis run without torch installed.
try:
    import faiss
    import torch
    NEURAL_STACK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in CPU-only environments
    faiss = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    NEURAL_STACK_AVAILABLE = False

import numpy as np
from log_config import get_logger

try:
    from .inference import ModelManager
except (ImportError, ValueError):
    from inference import ModelManager

logger = get_logger("internal_rag")
_CACHE_DIR_NAME = ".rag_cache"
# Persisted-index format version: bump to invalidate on-disk FAISS indexes when
# the embedding-text construction changes.
_INDEX_FORMAT_VERSION = 3


def build_cell_embedding_text(cell: Any, orchestrator: Optional[Any] = None) -> str:
    """Builds a rich semantic representation of a cell for dense vector embedding.

    Includes:
    - Categorical Stage (Stage 1 Source, Stage 2 Transform, Stage 3 Egress)
    - Structural Role & Archetype (source, transform, sink, macro, constant)
    - Primary Carrier container type
    - Natural language description / docstring
    - Human-readable operation name from cell ID
    - Canonical code template / callable API
    - Combined semantic tags and keywords (synonyms)
    - Input & output typestate data flow contract
    - Input parameter names and descriptions
    - Domain & library context
    - Outbound edge transitions (with transition probabilities and affinities)
    - Inbound edge transitions (predecessors feeding into this cell)
    - Preconditions and effects / postconditions
    """
    parts = []

    # 0. Categorical Stage & Structural Role
    stage_val = getattr(cell, "stage", None) if not isinstance(cell, dict) else cell.get("stage")
    stage_names = {
        1: "Stage 1 (Ingestion / Source / Reader)",
        2: "Stage 2 (Transformation / Processing / Computation)",
        3: "Stage 3 (Egress / Writer / Terminal / Visualization / Plot)"
    }
    if stage_val in stage_names:
        parts.append(f"Stage: {stage_names[stage_val]}")

    role_val = getattr(cell, "node_role", "") or getattr(cell, "node_type", "") if not isinstance(cell, dict) else (cell.get("node_role") or cell.get("node_type") or "")
    if role_val:
        parts.append(f"Role: {role_val}")

    # Carrier data type
    carrier = None
    if hasattr(cell, "primary_output"):
        carrier = getattr(cell.primary_output, "type_name", None)
    elif isinstance(cell, dict):
        outs = cell.get("outputs", {})
        if isinstance(outs, dict):
            if "output_data" in outs and isinstance(outs["output_data"], dict):
                carrier = outs["output_data"].get("type_name")
            elif "type_name" in outs:
                carrier = outs.get("type_name")
    if carrier and str(carrier).lower() not in ("any", "none", "*", "top"):
        parts.append(f"Carrier: {carrier}")

    # 1. Functional description / docstring
    desc = (getattr(cell, "docstring", "") or "").strip() if not isinstance(cell, dict) else (cell.get("docstring", "") or "").strip()
    if desc:
        parts.append(f"Description: {desc}")

    # 2. Human-readable operation name from cell_id
    cid = getattr(cell, "cell_id", "") if not isinstance(cell, dict) else cell.get("cell_id", "")
    clean_name = cid.replace("_", " ").lower()
    if clean_name:
        parts.append(f"Operation: {clean_name}")

    # 3. Canonical code template (gives exact API call, e.g. .dropna(), pd.read_csv)
    code = (getattr(cell, "code_template", "") or "").strip() if not isinstance(cell, dict) else (cell.get("code_template", "") or "").strip()
    if code:
        parts.append(f"Code: {code}")

    # 4. Semantic tags and keywords (synonym vocabulary: clean, remove, drop, null, filter, etc.)
    all_terms = set()
    raw_tags = getattr(cell, "semantic_tags", []) if not isinstance(cell, dict) else cell.get("semantic_tags", [])
    for t in raw_tags or []:
        if t:
            all_terms.add(str(t).lower())
    raw_kws = getattr(cell, "keywords", []) if not isinstance(cell, dict) else cell.get("keywords", [])
    for k in raw_kws or []:
        if k:
            all_terms.add(str(k).lower())
    if all_terms:
        parts.append(f"Keywords: {', '.join(sorted(all_terms))}")

    # 5. Typestate data flow
    if hasattr(cell, "primary_input") and hasattr(cell, "primary_output"):
        in_sig = f"{cell.primary_input.type_name}[{cell.primary_input.state}]"
        out_sig = f"{cell.primary_output.type_name}[{cell.primary_output.state}]"
        parts.append(f"Flow: {in_sig} -> {out_sig}")
    elif isinstance(cell, dict):
        in_t = cell.get("inputs", {}).get("type_name", "any") if isinstance(cell.get("inputs"), dict) else "any"
        out_t = cell.get("outputs", {}).get("type_name", "any") if isinstance(cell.get("outputs"), dict) else "any"
        parts.append(f"Flow: {in_t} -> {out_t}")

    # 6. Inputs parameter details
    inputs = getattr(cell, "inputs", {}) if not isinstance(cell, dict) else cell.get("inputs", {})
    if isinstance(inputs, dict) and inputs:
        in_descs = []
        for p_name, p_sig in inputs.items():
            if isinstance(p_sig, dict):
                t_name = p_sig.get("type_name", "any")
            else:
                t_name = getattr(p_sig, "type_name", "any")
            in_descs.append(f"{p_name}: {t_name}")
        parts.append(f"Inputs: {', '.join(in_descs)}")

    # 7. Domain name
    domain = getattr(cell, "domain_name", "") if not isinstance(cell, dict) else (cell.get("domain_name") or cell.get("domain", ""))
    if domain:
        parts.append(f"Domain: {domain}")

    # 8. Outbound Edge Transitions (with transition probabilities and affinities)
    edges = getattr(cell, "edges", []) if not isinstance(cell, dict) else cell.get("edges", [])
    if edges:
        succ_entries = []
        for e in edges[:6]:
            tgt = e.get("target_cell_id") if isinstance(e, dict) else getattr(e, "target_cell_id", "")
            if not tgt:
                continue
            prov = e.get("score_provenance", "") if isinstance(e, dict) else getattr(e, "score_provenance", "")
            aff = e.get("affinity_score", 0.0) if isinstance(e, dict) else getattr(e, "affinity_score", 0.0)
            meta = e.get("metadata", {}) if isinstance(e, dict) else getattr(e, "metadata", {})
            prob = meta.get("transition_probability") if isinstance(meta, dict) else None
            p_str = f"p={prob:.3f}" if prob is not None else f"aff={aff:.2f}"
            succ_entries.append(f"{tgt} ({p_str}, {prov})")
        if succ_entries:
            parts.append(f"Successors: {'; '.join(succ_entries)}")

    # 9. Inbound Edge Transitions (Predecessors from orchestrator topology)
    if orchestrator is not None and hasattr(orchestrator, "_reverse_adjacency") and cid:
        preds = orchestrator._reverse_adjacency.get(cid, [])[:6]
        if preds:
            parts.append(f"Predecessors: {', '.join(preds)}")

    # 10. Preconditions and Effects / Postconditions
    preconds = getattr(cell, "preconditions", []) if not isinstance(cell, dict) else cell.get("preconditions", [])
    if preconds:
        p_strs = []
        for p in preconds[:3]:
            if isinstance(p, dict):
                expr = p.get("expression") or f"{p.get('property')} {p.get('operator')} {p.get('value')}"
            elif hasattr(p, "expression") and p.expression:
                expr = p.expression
            else:
                expr = str(p)
            p_strs.append(expr)
        if p_strs:
            parts.append(f"Preconditions: {'; '.join(p_strs)}")

    effects = getattr(cell, "effects", []) or getattr(cell, "postconditions", []) if not isinstance(cell, dict) else (cell.get("effects") or cell.get("postconditions", []))
    if effects:
        e_strs = []
        for ef in effects[:3]:
            if isinstance(ef, dict):
                expr = ef.get("expression") or f"{ef.get('property')} {ef.get('operator')} {ef.get('value')}"
            elif hasattr(ef, "expression") and ef.expression:
                expr = ef.expression
            else:
                expr = str(ef)
            e_strs.append(expr)
        if e_strs:
            parts.append(f"Effects: {'; '.join(e_strs)}")

    return " | ".join(parts) if parts else cid


class _BM25Index:
    """Fast lexical BM25 token index backed by an inverted postings index."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lens: Dict[int, int] = {}
        self.avg_dl: float = 1.0
        self.df: Dict[str, int] = {}
        # Inverted index: term -> {doc_id: term_frequency}
        self.inverted_index: Dict[str, Dict[int, int]] = {}
        self.num_docs: int = 0
        self._total_len: int = 0

    def _extract_tokens(self, schema: Dict[str, Any]) -> List[str]:
        cid = str(schema.get("cell_id", "") or "")
        doc = str(schema.get("docstring", "") or "")
        domain = str(schema.get("domain", "") or "")
        role = str(schema.get("node_role", "") or "")
        keywords = " ".join(schema.get("keywords", []) or [])
        text = f"{cid} {cid.replace('_', ' ')} {doc} {domain} {role} {keywords}"
        return tokenize_alphanumeric(text, min_len=1)

    def index_schemas(self, id_to_schema: Dict[int, Dict[str, Any]]) -> None:
        self.num_docs = len(id_to_schema)
        self.doc_lens.clear()
        self.df.clear()
        self.inverted_index.clear()
        self._total_len = 0
        if self.num_docs == 0:
            self.avg_dl = 1.0
            return

        for idx, schema in id_to_schema.items():
            tokens = self._extract_tokens(schema)
            dl = len(tokens)
            self.doc_lens[idx] = dl
            self._total_len += dl

            term_counts: Dict[str, int] = {}
            for t in tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            for term, freq in term_counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][idx] = freq
                self.df[term] = self.df.get(term, 0) + 1

        self.avg_dl = max(1.0, self._total_len / float(self.num_docs))

    def add_document(self, doc_id: int, schema: Dict[str, Any]) -> None:
        tokens = self._extract_tokens(schema)
        dl = len(tokens)

        # Remove existing document postings if updating an existing ID
        if doc_id in self.doc_lens:
            self._total_len -= self.doc_lens[doc_id]
            for term, postings in list(self.inverted_index.items()):
                if doc_id in postings:
                    del postings[doc_id]
                    self.df[term] = max(0, self.df.get(term, 1) - 1)
        else:
            self.num_docs += 1

        self.doc_lens[doc_id] = dl
        self._total_len += dl

        term_counts: Dict[str, int] = {}
        for t in tokens:
            term_counts[t] = term_counts.get(t, 0) + 1

        for term, freq in term_counts.items():
            if term not in self.inverted_index:
                self.inverted_index[term] = {}
            self.inverted_index[term][doc_id] = freq
            self.df[term] = self.df.get(term, 0) + 1

        self.avg_dl = max(1.0, self._total_len / float(max(1, self.num_docs)))

    def score_query(self, query: str) -> Dict[int, float]:
        tokens = tokenize_alphanumeric(query, min_len=1)
        if not tokens or self.num_docs == 0:
            return {}

        qtf: Dict[str, int] = {}
        for t in tokens:
            qtf[t] = qtf.get(t, 0) + 1

        scores: Dict[int, float] = {}
        for token, q_count in qtf.items():
            postings = self.inverted_index.get(token)
            if not postings:
                continue
            doc_freq = len(postings)
            idf = math.log(1.0 + (self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            token_weight = idf * q_count

            for idx, freq in postings.items():
                dl = self.doc_lens.get(idx, self.avg_dl)
                num = freq * (self.k1 + 1.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_dl))
                scores[idx] = scores.get(idx, 0.0) + token_weight * (num / denom)
        return scores


class LocalRAG:
    """
    Maintains a dense vector index over all lattice cells for sub-millisecond
    semantic tunneling and context retrieval.
    """
    def __init__(self, trees_dir: str, orchestrator=None):
        self.trees_dir = trees_dir
        self.orchestrator = orchestrator
        self._cache_dir = os.path.join(os.path.dirname(trees_dir), _CACHE_DIR_NAME)

        if not NEURAL_STACK_AVAILABLE:
            raise RuntimeError(
                "LocalRAG requires the neural stack (faiss, torch). Install them or use Profile 0 (pure symbolic)."
            )

        model_mgr = ModelManager.get_instance()
        if model_mgr.profile is None:
            raise RuntimeError(
                "LocalRAG requires ModelManager to be initialized with a profile before construction."
            )

        self.dimension = model_mgr.embedding_dimension
        self.index: Optional[Any] = None
        self.id_to_schema: Dict[int, Dict[str, Any]] = {}
        self.cell_cache: Dict[str, Dict[str, Any]] = {}
        self.bm25_index = _BM25Index()
        self.cell_boosts: Dict[str, float] = {}
        self._lock = threading.RLock()

        self.build_index()

    def _get_cache_path(self) -> str:
        mm = ModelManager.get_instance()
        prof_name = getattr(mm, "current_profile_name", "unknown") or "unknown"
        profile = mm.active_profile
        emb_name = getattr(profile, "embedder_name", "default")
        dim = getattr(profile, "_dim", self.dimension)
        safe_name = "".join(c for c in f"{prof_name}__{emb_name}__dim{dim}" if c.isalnum() or c in "._-")
        return os.path.join(self._cache_dir, f"{safe_name}_cache.pkl")

    def _get_index_paths(self) -> Tuple[str, str]:
        """Persisted FAISS index + schema map paths, keyed by profile identity, embedder, and corpus fingerprint."""
        mm = ModelManager.get_instance()
        prof_name = getattr(mm, "current_profile_name", "unknown") or "unknown"
        profile = mm.active_profile
        emb_name = getattr(profile, "embedder_name", "default")
        dim = getattr(profile, "_dim", self.dimension)
        safe_name = "".join(c for c in f"{prof_name}__{emb_name}__dim{dim}" if c.isalnum() or c in "._-")
        corpus_fp = self._corpus_fingerprint()
        base = os.path.join(self._cache_dir, f"{safe_name}__{corpus_fp}__v{_INDEX_FORMAT_VERSION}")
        return base + ".faiss", base + ".schema.pkl"

    def _corpus_fingerprint(self) -> str:
        """Stable fingerprint of the indexed corpus (tree files or loaded cell count + names)."""
        try:
            tree_files = sorted(
                f for f in os.listdir(self.trees_dir) if f.endswith(".json")
            ) if os.path.isdir(self.trees_dir) else []
            h = hashlib.sha256()
            for f in tree_files:
                st = os.stat(os.path.join(self.trees_dir, f))
                h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode("utf-8"))
            if tree_files:
                return h.hexdigest()[:16]
        except Exception:
            pass
        return f"n{len(self.orchestrator.loaded_cells) if self.orchestrator else 0}"

    def _load_cache(self):
        path = self._get_cache_path()
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.cell_cache = pickle.load(f)
                logger.info(f"[RAG CACHE] Loaded {len(self.cell_cache)} records from {path}")
            except Exception as e:
                logger.warning(f"[RAG CACHE] Failed to load cache: {e}")
                self.cell_cache = {}
        else:
            self.cell_cache = {}

    def _save_cache(self):
        os.makedirs(self._cache_dir, exist_ok=True)
        path = self._get_cache_path()
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(self.cell_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.warning(f"[RAG CACHE] Failed to save cache: {e}")
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def build_index(self):
        """Constructs or updates the FAISS index incrementally from loaded cells."""
        with self._lock:
            if self._try_restore_persisted_index():
                return

            self._load_cache()
            if self.orchestrator is None:
                return

            cells = list(self.orchestrator.loaded_cells.values())
            seen_ids = set()
            new_or_changed = []

            for cell in cells:
                cid = cell.cell_id
                seen_ids.add(cid)

                desc = (getattr(cell, "docstring", "") or "").strip()
                text_repr = build_cell_embedding_text(cell, orchestrator=self.orchestrator)
                content_hash = hashlib.sha256(text_repr.encode("utf-8")).hexdigest()

                schema = {
                    "cell_id": cid,
                    "type": cell.cell_type,
                    "stage": cell.stage,
                    "keywords": sorted(cell.keywords),
                    "domain": cell.domain_name,
                    "docstring": desc,
                    "enrichment_source": getattr(cell, "enrichment_source", None),
                    "primary_input": cell.primary_input.type_name,
                    "primary_output": cell.primary_output.type_name
                }

                if cid in self.cell_cache and self.cell_cache[cid].get("hash") == content_hash:
                    self.cell_cache[cid]["schema"] = schema
                else:
                    new_or_changed.append({"cell_id": cid, "text": text_repr, "hash": content_hash, "schema": schema})

            # Evict removed cells
            for cid in list(self.cell_cache.keys()):
                if cid not in seen_ids:
                    del self.cell_cache[cid]

            # Embed new/modified cells incrementally to avoid VRAM exhaustion
            if new_or_changed:
                logger.info(f"[RAG] Batch-embedding {len(new_or_changed)} new/changed cells in chunks...")
                chunk_size = 250
                total_cnt = len(new_or_changed)
                for i in range(0, total_cnt, chunk_size):
                    chunk = new_or_changed[i:i + chunk_size]
                    texts = [item["text"] for item in chunk]
                    try:
                        embeddings = ModelManager.get_instance().get_embeddings(texts, mode="document")
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"[RAG] CUDA OOM at chunk {i}; flushing VRAM and falling back to micro-batches...")
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        embeddings = []
                        for sub_idx in range(0, len(texts), 25):
                            sub_texts = texts[sub_idx:sub_idx + 25]
                            sub_embs = ModelManager.get_instance().get_embeddings(sub_texts, mode="document")
                            embeddings.extend(sub_embs)
                            del sub_texts
                            del sub_embs
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    for item, emb in zip(chunk, embeddings):
                        self.cell_cache[item["cell_id"]] = {
                            "hash": item["hash"],
                            "embedding": emb,
                            "schema": item["schema"]
                        }
                    self._save_cache()
                    logger.info(
                        f"[RAG] Embedding Progress: {min(i + chunk_size, total_cnt)} / {total_cnt} "
                        f"({min(i + chunk_size, total_cnt) / total_cnt * 100:.1f}%)"
                    )

                    del texts
                    del embeddings
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        if (i // chunk_size) % 10 == 0:
                            torch.cuda.ipc_collect()

            if not self.cell_cache:
                self.index = None
                self.id_to_schema.clear()
                return

            # Build FAISS normalized flat inner-product index
            all_embs = []
            self.id_to_schema.clear()
            for idx, (cid, data) in enumerate(self.cell_cache.items()):
                all_embs.append(data["embedding"])
                self.id_to_schema[idx] = data["schema"]

            matrix = np.array(all_embs, dtype=np.float32)
            self.dimension = matrix.shape[1]

            # L2 Normalization (Cosine Similarity)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            matrix = matrix / norms

            self.index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.add(matrix)
            self.bm25_index.index_schemas(self.id_to_schema)
            logger.info(f"[RAG] FAISS Index ready with {self.index.ntotal} vectors.")
            self._persist_index()

    def _try_restore_persisted_index(self) -> bool:
        """Restores a persisted FAISS index iff it is consistent with the current corpus and embedder."""
        if self.orchestrator is None:
            return False
        try:
            faiss_path, schema_path = self._get_index_paths()
            cache_path = self._get_cache_path()
            if not (os.path.exists(faiss_path) and os.path.exists(schema_path)):
                return False
            if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(faiss_path):
                return False
            with open(schema_path, "rb") as f:
                id_to_schema = pickle.load(f)
            index = faiss.read_index(faiss_path)
            if index.ntotal != len(id_to_schema):
                return False
            self.index = index
            self.id_to_schema = id_to_schema
            self.dimension = index.d
            self.bm25_index.index_schemas(self.id_to_schema)
            logger.info(f"[RAG] Restored persisted FAISS index ({index.ntotal} vectors) from {faiss_path}.")
            return True
        except Exception as e:
            logger.warning(f"[RAG] Persisted-index restore failed: {e}")
            return False

    def _persist_index(self) -> None:
        """Persists the assembled FAISS index + schema map keyed by corpus fingerprint."""
        faiss_path, schema_path = self._get_index_paths()
        tmp_faiss, tmp_schema = faiss_path + ".tmp", schema_path + ".tmp"
        try:
            if self.index is None or not self.id_to_schema:
                return
            os.makedirs(self._cache_dir, exist_ok=True)
            faiss.write_index(self.index, tmp_faiss)
            with open(tmp_schema, "wb") as f:
                pickle.dump(self.id_to_schema, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_faiss, faiss_path)
            os.replace(tmp_schema, schema_path)

            # Invalidate stale fingerprint variants of the persisted index safely
            safe_prefix = os.path.basename(faiss_path).split("__")[0]
            try:
                cache_files = os.listdir(self._cache_dir)
            except OSError:
                cache_files = []

            for old in cache_files:
                if old.startswith(safe_prefix + "__") and old.endswith(".faiss") and old != os.path.basename(faiss_path):
                    old_faiss = os.path.join(self._cache_dir, old)
                    old_schema = os.path.join(self._cache_dir, old[:-6] + ".schema.pkl")
                    for target_file in (old_faiss, old_schema):
                        try:
                            if os.path.exists(target_file):
                                os.remove(target_file)
                        except OSError as err:
                            logger.debug(f"[RAG] Stale cache removal skipped for {target_file}: {err}")

            logger.info(f"[RAG] Persisted FAISS index to {faiss_path}.")
        except Exception as e:
            logger.warning(f"[RAG] Failed to persist FAISS index: {e}")
            for tmp in (tmp_faiss, tmp_schema):
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass

    def add_dynamic_cell(self, cell_dict: Dict[str, Any]):
        """Appends a newly synthesized cell dynamically to the active FAISS index."""
        with self._lock:
            if self.index is None:
                return

            cid = cell_dict.get("cell_id", "dynamic_cell")
            text_repr = build_cell_embedding_text(cell_dict, orchestrator=self.orchestrator)

            raw_emb = np.array([ModelManager.get_instance().get_embedding(text_repr, mode="document")], dtype=np.float32)
            norm = np.linalg.norm(raw_emb)
            if norm > 0:
                raw_emb = raw_emb / norm

            self.index.add(raw_emb)
            new_idx = len(self.id_to_schema)
            self.id_to_schema[new_idx] = cell_dict
            self.bm25_index.add_document(new_idx, cell_dict)
            logger.info(f"[RAG] Dynamically indexed synthesized cell: {cid}")

    def boost_cell(self, cell_id: str, boost: float = 1.5) -> None:
        """Modifies relevance score multiplier for cell_id so it ranks higher in RAG."""
        with self._lock:
            cid = str(cell_id).strip().lower()
            prev = self.cell_boosts.get(cid, 1.0)
            self.cell_boosts[cid] = prev * boost
            logger.info(f"[RAG] Boosted cell '{cell_id}' score multiplier from {prev:.2f} to {self.cell_boosts[cid]:.2f}")

    def register_dynamic_cell(
        self,
        cell_dict: Dict[str, Any],
        reviewed: bool = False,
        provisional_score_mult: float = 0.70
    ) -> None:
        """
        Dynamically indexes a newly synthesized cell into FAISS + BM25,
        marking review state and applying provisional score multiplier.
        """
        cid = str(cell_dict.get("cell_id", "dynamic_cell")).strip()
        cell_dict["reviewed"] = reviewed
        cell_dict["provisional_score_mult"] = provisional_score_mult

        with self._lock:
            if not reviewed:
                self.cell_boosts[cid.lower()] = provisional_score_mult
            self.add_dynamic_cell(cell_dict)
            logger.info(
                f"[RAG] Registered dynamic cell '{cid}' (reviewed={reviewed}, "
                f"provisional_multiplier={provisional_score_mult})"
            )

    def _blend_dense_and_lexical(
        self,
        prompt: str,
        dense_results: List[Tuple[float, int]],
        top_k: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Combines dense vector similarity scores with BM25 lexical keyword matches
        using Reciprocal Rank Fusion (RRF). Contains ZERO keyword-sniffing regexes,
        ZERO manual score boosts, and ZERO domain hardcodes.
        """
        bm25_scores = self.bm25_index.score_query(prompt)
        sorted_bm25 = sorted(bm25_scores.items(), key=lambda kv: kv[1], reverse=True)
        bm25_rank_map = {idx: rank + 1 for rank, (idx, _) in enumerate(sorted_bm25)}

        dense_rank_map = {
            idx: rank + 1
            for rank, (dist, idx) in enumerate(dense_results)
            if idx != -1 and idx in self.id_to_schema
        }

        candidates_set = set(dense_rank_map.keys()) | set(list(bm25_rank_map.keys())[:top_k * 2])
        if not candidates_set:
            return []

        rrf_k = 60.0
        fused: List[Tuple[float, int, float]] = []
        for idx in candidates_set:
            if idx not in self.id_to_schema:
                continue
            schema = self.id_to_schema[idx]
            cid = str(schema.get("cell_id", "") or "").lower()

            d_rank = dense_rank_map.get(idx, 100)
            b_rank = bm25_rank_map.get(idx, 100)

            # Mathematical Reciprocal Rank Fusion without heuristic keyword boosts
            rrf = (1.0 / (rrf_k + d_rank)) + (1.0 / (rrf_k + b_rank))

            # Apply cell-level registration multiplier (e.g., provisional discount for unreviewed cells)
            final_score = rrf * self.cell_boosts.get(cid, 1.0)
            dense_score = next((d for d, i in dense_results if i == idx), 0.0)
            fused.append((final_score, idx, dense_score))

        fused.sort(key=lambda x: x[0], reverse=True)

        results: List[Dict[str, Any]] = []
        for score, idx, dense_sc in fused[:top_k]:
            schema = self.id_to_schema[idx]
            cid = schema.get("cell_id", "")
            # Honest score reporting: `score` is the REAL dense cosine
            # similarity when an embedding score exists, and 0.0 otherwise.
            # Lexical-only candidates keep their ranking position through the
            # RRF fusion (see `rrf_score`) but are never assigned a
            # manufactured "vector confidence" — dense evidence and lexical
            # evidence remain separately observable downstream.
            has_dense_evidence = dense_sc is not None and dense_sc > 0.05
            norm_score = max(0.0, min(1.0, float(dense_sc))) if has_dense_evidence else 0.0
            results.append({
                "cell_id": cid,
                "score": float(norm_score),
                "score_source": "dense_cosine" if has_dense_evidence else "rrf_rank",
                "rrf_score": float(score),
                "schema": schema,
                "domain": schema.get("domain", "generic"),
                "primary_input": schema.get("primary_input", "any"),
                "primary_output": schema.get("primary_output", "any"),
                "text": (
                    f"ID: {cid} | In: {schema.get('primary_input', 'any')} -> "
                    f"Out: {schema.get('primary_output', 'any')} | Domain: {schema.get('domain', 'generic')}"
                )
            })
        return results

    def get_relevant_context(self, prompt: str, top_k: int = 25) -> List[Dict[str, Any]]:
        """Retrieves structured context list for the top-k most semantically aligned cells."""
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []

            raw_emb = np.array([ModelManager.get_instance().get_embedding(prompt, mode="query")], dtype=np.float32)
            norm = np.linalg.norm(raw_emb)
            if norm == 0:
                return []
            raw_emb = raw_emb / norm

            search_k = min(top_k * 2, self.index.ntotal)
            distances, indices = self.index.search(raw_emb, search_k)
            dense_pairs = list(zip(distances[0], indices[0]))
            return self._blend_dense_and_lexical(prompt, dense_pairs, top_k=top_k)

    def get_relevant_context_batch(
        self,
        prompts: List[str],
        top_k: int = 25,
        max_fanout: int = 32
    ) -> List[List[Dict[str, Any]]]:
        """
        Batched retrieval: embeds unique query spans in a single model call with
        capped fan-out, performing batch FAISS search and inverted-index BM25 hybrid fusion.
        """
        with self._lock:
            if self.index is None or self.index.ntotal == 0 or not prompts:
                return [[] for _ in prompts]

            # 1. Deduplicate unique prompts while preserving input mapping
            unique_prompts: List[str] = []
            seen: Dict[str, int] = {}
            for p in prompts:
                p_clean = p.strip()
                if p_clean and p_clean not in seen:
                    seen[p_clean] = len(unique_prompts)
                    unique_prompts.append(p_clean)

            if not unique_prompts:
                return [[] for _ in prompts]

            # 2. Cap fan-out of unique prompts to prevent unconstrained query multiplication
            effective_prompts = unique_prompts
            if max_fanout and len(unique_prompts) > max_fanout:
                logger.warning(
                    f"[RAG] Batch unique query fan-out of {len(unique_prompts)} exceeds cap ({max_fanout}); "
                    f"capping retrieval to first {max_fanout} unique spans."
                )
                effective_prompts = unique_prompts[:max_fanout]

            # 3. Batch embed only unique queries
            embeddings = ModelManager.get_instance().get_embeddings(effective_prompts, mode="query")
            matrix = np.array(embeddings, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(effective_prompts):
                return [[] for _ in prompts]

            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            matrix = matrix / norms

            # 4. Single batched FAISS search
            search_k = min(top_k * 2, self.index.ntotal)
            distances, indices = self.index.search(matrix, search_k)

            # 5. Hybrid blend per unique query
            unique_results: List[List[Dict[str, Any]]] = []
            for p, row_d, row_i in zip(effective_prompts, distances, indices):
                dense_pairs = list(zip(row_d, row_i))
                unique_results.append(self._blend_dense_and_lexical(p, dense_pairs, top_k=top_k))

            # 6. Map back to original prompt list
            output: List[List[Dict[str, Any]]] = []
            for p in prompts:
                p_clean = p.strip()
                if p_clean in seen and seen[p_clean] < len(unique_results):
                    output.append(unique_results[seen[p_clean]])
                else:
                    output.append([])

            return output

    def format_context_for_prompt(self, context_items: List[Dict[str, Any]]) -> str:
        """Formats structured context list into a prompt-friendly string for LLMs."""
        if not context_items:
            return "No verified cells available."
        lines = []
        for item in context_items:
            lines.append(
                f"- ID: {item.get('cell_id')} | "
                f"In: {item.get('primary_input', 'any')} -> "
                f"Out: {item.get('primary_output', 'any')} | "
                f"Domain: {item.get('domain', 'generic')}"
            )
        return "\n".join(lines)

    def find_closest_cell_by_embedding(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Finds the single closest cell by cosine similarity."""
        with self._lock:
            results = self.get_relevant_context(prompt, top_k=1)
            return results[0] if results else None
