"""
src/subgoal_retrieval.py — Neutral per-subgoal candidate retrieval.

Retrieves cells strictly via official neutral API signatures and docstrings,
supporting asymmetric query/document embeddings, pre-normalized vector scoring,
pure calibrated reranking, and honest Profile A/B retrieval on formal intents.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from .schema import derive_value_class, get_attr
    from .typed_ir import IRNode, TypedIR
except (ImportError, ValueError):
    from schema import derive_value_class, get_attr
    from typed_ir import IRNode, TypedIR


def _cell_has_external_asset_port(cell: Any) -> bool:
    raw_ins = get_attr(cell, "inputs", {}) or {}
    if not isinstance(raw_ins, dict):
        return False
    return any(derive_value_class(pv) == "external_asset" for pv in raw_ins.values())


def _format_ports(raw_ports: Any) -> str:
    if not isinstance(raw_ports, dict) or not raw_ports:
        return "none"
    parts: List[str] = []
    for pname, pval in raw_ports.items():
        carrier = str(get_attr(pval, "carrier", None) or get_attr(pval, "type_name", None) or "any").strip()
        ts = str(get_attr(pval, "typestate", None) or get_attr(pval, "state", None) or "any").strip()
        parts.append(f"{pname}: {carrier}[{ts}]")
    return ", ".join(parts)


def build_neutral_api_document(cell: Any) -> str:
    """
    Construct canonical retrieval text exclusively from neutral API metadata:
    domain, cell_id, input/output port signatures, and official docstring.
    """
    dom = str(get_attr(cell, "domain_name", None) or "").strip()
    cid = str(get_attr(cell, "cell_id", "") or "").strip()
    doc = str(get_attr(cell, "docstring", None) or "").strip()
    ins_str = _format_ports(get_attr(cell, "inputs", {}))
    outs_str = _format_ports(get_attr(cell, "outputs", {}))
    lines = [
        f"domain: {dom}",
        f"operation: {cid}",
        f"inputs: {ins_str}",
        f"outputs: {outs_str}",
    ]
    if doc:
        lines.append(f"summary: {doc}")
    return "\n".join(lines)


def _normalize_vec(vec: Sequence[float]) -> Tuple[float, ...]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm <= 1e-12:
        return tuple(0.0 for _ in vec)
    inv = 1.0 / norm
    return tuple(float(x) * inv for x in vec)


def _dot_unit(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


@dataclass(frozen=True)
class CandidateMatch:
    cell: Any
    cell_id: str
    domain: str
    score: float
    match_mode: str


class NeutralSubgoalRetriever:
    """Per-subgoal retriever parameterized cleanly by embedding and reranking callables."""

    def __init__(
        self,
        cells: Iterable[Any],
        embed_fn: Optional[Callable[..., List[List[float]]]] = None,
        rerank_fn: Optional[Callable[[str, List[str]], List[float]]] = None,
        domain_hint_bonus: float = 0.0,
    ) -> None:
        self._cells: List[Any] = []
        self._cell_ids: List[str] = []
        self._cell_domains: List[str] = []
        self._docs: List[str] = []
        for c in cells:
            cid = str(get_attr(c, "cell_id", "") or "").strip()
            if not cid:
                continue
            dom = str(get_attr(c, "domain_name", None) or "").strip()
            self._cells.append(c)
            self._cell_ids.append(cid)
            self._cell_domains.append(dom)
            self._docs.append(build_neutral_api_document(c))

        self.embed_fn = embed_fn
        self.rerank_fn = rerank_fn
        self.domain_hint_bonus = max(0.0, float(domain_hint_bonus))
        self._doc_unit_vecs: Optional[List[Tuple[float, ...]]] = None

    def _invoke_embed(self, texts: List[str], *, mode: str) -> List[List[float]]:
        if self.embed_fn is None or not texts:
            return []
        try:
            sig = inspect.signature(self.embed_fn)
            if "mode" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                return list(self.embed_fn(texts, mode=mode))
        except (TypeError, ValueError):
            pass
        return list(self.embed_fn(texts))

    def _ensure_doc_embeddings(self) -> Optional[List[Tuple[float, ...]]]:
        if self.embed_fn is None:
            return None
        if self._doc_unit_vecs is None:
            raw_vecs = self._invoke_embed(self._docs, mode="document") if self._docs else []
            self._doc_unit_vecs = [_normalize_vec(v) for v in raw_vecs]
        return self._doc_unit_vecs

    def retrieve_for_node(
        self,
        node: IRNode,
        source_mode: str,
        top_k: int = 8,
        min_dense_sim: float = 1e-6,
        *,
        allowed_domains: Optional[Set[str]] = None,
        strict_domain: Optional[str] = None,
        requires_asset_port: bool = False,
    ) -> List[CandidateMatch]:
        if not self._cells or top_k <= 0 or node.kind != "call":
            return []

        hint_dom = (node.domain_hint or "").strip().lower()
        strict_l = (strict_domain or "").strip().lower()
        if not strict_l and source_mode == "formal_syntax" and hint_dom:
            strict_l = hint_dom
        allowed_l = {str(d).strip().lower() for d in allowed_domains} if allowed_domains is not None else None
        target_op = str(node.intent or "").strip().lower()

        def _domain_permitted(dom_val: str) -> bool:
            dom_l = dom_val.strip().lower()
            if strict_l:
                return dom_l == strict_l
            if allowed_l is not None:
                return dom_l in allowed_l
            return True

        if source_mode == "formal_syntax":
            exact: List[CandidateMatch] = []
            for cell, cid, dom in zip(self._cells, self._cell_ids, self._cell_domains):
                if cid.lower() != target_op:
                    continue
                if not _domain_permitted(dom):
                    continue
                exact.append(
                    CandidateMatch(
                        cell=cell,
                        cell_id=cid,
                        domain=dom,
                        score=1.0,
                        match_mode="formal_exact",
                    )
                )
            if exact:
                exact.sort(key=lambda m: (-m.score, m.cell_id))
                return exact[:top_k]
            # Profile 0 (embed_fn=None) allows strictly exact formal matches;
            # Profiles A/B fall through to dense/reranked retrieval on semantic intents.
            if self.embed_fn is None:
                return []

        doc_vecs = self._ensure_doc_embeddings()
        if self.embed_fn is None or doc_vecs is None or len(doc_vecs) != len(self._cells):
            return []

        q_parts = [node.intent.strip()]
        if hint_dom:
            q_parts.append(f"domain: {hint_dom}")
        if node.params:
            q_parts.append(f"params: {', '.join(sorted(node.params.keys()))}")
        query = " | ".join(p for p in q_parts if p)

        q_vecs = self._invoke_embed([query], mode="query")
        if not q_vecs:
            return []
        q_unit = _normalize_vec(q_vecs[0])

        def _collect_dense_pool(asset_only: bool) -> List[Tuple[float, int]]:
            out_scores: List[Tuple[float, int]] = []
            for idx, (cell, cid, dom, d_unit) in enumerate(
                zip(self._cells, self._cell_ids, self._cell_domains, doc_vecs)
            ):
                if not _domain_permitted(dom):
                    continue
                if asset_only and not _cell_has_external_asset_port(cell):
                    continue
                sim = _dot_unit(q_unit, d_unit)
                if sim <= min_dense_sim:
                    continue
                if self.domain_hint_bonus > 0.0 and hint_dom and dom.lower() == hint_dom:
                    sim += self.domain_hint_bonus
                out_scores.append((sim, idx))
            return out_scores

        scored_indices: List[Tuple[float, int]] = _collect_dense_pool(asset_only=requires_asset_port)
        if requires_asset_port and not scored_indices:
            scored_indices = _collect_dense_pool(asset_only=False)

        scored_indices.sort(key=lambda item: (-item[0], self._cell_ids[item[1]]))
        pool_k = max(top_k * 2, 16)
        pool = scored_indices[:pool_k]

        if self.rerank_fn is not None and pool:
            pool_docs = [self._docs[idx] for _, idx in pool]
            r_scores = list(self.rerank_fn(query, pool_docs))
            mode_tag = "formal_reranked" if source_mode == "formal_syntax" else "neural_reranked"
            reranked = [
                CandidateMatch(
                    cell=self._cells[idx],
                    cell_id=self._cell_ids[idx],
                    domain=self._cell_domains[idx],
                    score=float(r_sc),
                    match_mode=mode_tag,
                )
                for (_, idx), r_sc in zip(pool, r_scores)
            ]
            reranked.sort(key=lambda m: (-m.score, m.cell_id))
            return reranked[:top_k]

        dense_tag = "formal_dense" if source_mode == "formal_syntax" else "neural_dense"
        dense_matches = [
            CandidateMatch(
                cell=self._cells[idx],
                cell_id=self._cell_ids[idx],
                domain=self._cell_domains[idx],
                score=float(dense_sc),
                match_mode=dense_tag,
            )
            for dense_sc, idx in pool[:top_k]
        ]
        dense_matches.sort(key=lambda m: (-m.score, m.cell_id))
        return dense_matches

    def retrieve_for_ir(
        self,
        ir: TypedIR,
        top_k: int = 8,
        *,
        allowed_domains: Optional[Set[str]] = None,
        strict_domains_by_node: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[CandidateMatch]]:
        strict_map = strict_domains_by_node or {}
        grounded_assets = ir.grounded_assets()
        is_slm = ir.source_mode == "slm_compiled"
        results: Dict[str, List[CandidateMatch]] = {}
        for node in ir.flatten_nodes():
            if node.kind != "call":
                continue
            has_asset = is_slm and (
                any(inp in grounded_assets for inp in node.inputs)
                or any(
                    pv.value_class == "external_asset" and pv.provenance == "prompt" and pv.value is not None
                    for pv in node.params.values()
                )
            )
            results[node.id] = self.retrieve_for_node(
                node,
                ir.source_mode,
                top_k=top_k,
                allowed_domains=allowed_domains,
                strict_domain=strict_map.get(node.id),
                requires_asset_port=has_asset,
            )
        return results


__all__ = [
    "CandidateMatch",
    "build_neutral_api_document",
    "NeutralSubgoalRetriever",
]
