"""
src/bridge_graph.py — Formal 4-Component Port Unification & Coherent Coercion Subcategory.

Discovers k-hop representation conversions dynamically across whatever domain trees
are loaded, restricted strictly to declared coercion morphisms (`bridge_profile.is_bridge`)
with two-tier coercion coherence (Tier-1 lossless isomorphisms vs Tier-2 intent/lossy bridges).
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

try:
    from .schema import (
        DESTRUCTIVE_OR_CLEAN_EFFECT_KINDS,
        TOP_CARRIERS,
        TOP_STATES,
        cell_has_effect_kind,
        derive_value_class,
        get_attr,
        normalize_typestate,
    )
    from .type_system import BoundedPoset, RefinementSet, join_shapes, unify_shapes
except (ImportError, ValueError):
    from schema import (
        DESTRUCTIVE_OR_CLEAN_EFFECT_KINDS,
        TOP_CARRIERS,
        TOP_STATES,
        cell_has_effect_kind,
        derive_value_class,
        get_attr,
        normalize_typestate,
    )
    from type_system import BoundedPoset, RefinementSet, join_shapes, unify_shapes

SimpleTypePoset = BoundedPoset
_get = get_attr
_derive_value_class = derive_value_class


@dataclass(frozen=True)
class CanonicalPortType:
    """4-Component Graded Port Type T = <carrier, layout, (typestate, resource_mode), invariants>."""

    carrier: str
    typestate: str = "any"
    accepted_states: Tuple[str, ...] = ()
    domain: str = ""
    rank: Optional[Union[int, str]] = None
    shape: Optional[Tuple[Union[int, str, None], ...]] = None
    dtype: Optional[str] = None
    channel_count: Optional[int] = None
    channel_axis: Optional[int] = None
    length: Optional[Union[int, str]] = None
    rank_polymorphic: bool = False
    invariants: Tuple[str, ...] = ()
    resource_mode: str = "value"

    @classmethod
    def from_port(cls, port: Any, cell_domain: str = "") -> "CanonicalPortType":
        dom = str(get_attr(port, "domain", None) or cell_domain or "").strip().lower()
        carrier = str(get_attr(port, "carrier", None) or get_attr(port, "type_name", None) or "any").strip()

        raw_ts = get_attr(port, "typestate", None)
        if not raw_ts:
            raw_ts = get_attr(port, "state", "any")
        ts = normalize_typestate(str(raw_ts or "any"), dom)

        raw_acc = get_attr(port, "accepted_states", ()) or ()
        acc_list = [normalize_typestate(str(s), dom) for s in raw_acc if str(s or "").strip()]
        accepted_states = tuple(dict.fromkeys(acc_list))

        layout = get_attr(port, "layout", None)
        rank = get_attr(layout, "rank", None)
        if rank is None:
            rank = get_attr(port, "rank", None)
        if isinstance(rank, str) and rank.isdigit():
            rank = int(rank)

        raw_shape = get_attr(layout, "shape", None)
        shape: Optional[Tuple[Union[int, str, None], ...]] = (
            tuple(raw_shape) if isinstance(raw_shape, (list, tuple)) else None
        )

        dtype = get_attr(layout, "dtype", None)
        if dtype is None:
            dtype = get_attr(port, "element_type", None)
        if dtype is not None:
            dtype = str(dtype).strip() or None

        channel_count = get_attr(layout, "channel_count", None)
        channel_axis = get_attr(layout, "channel_axis", None)
        length = get_attr(layout, "length", None)
        if isinstance(length, str) and length.isdigit():
            length = int(length)
        rank_polymorphic = bool(get_attr(layout, "rank_polymorphic", False))

        raw_inv = get_attr(port, "invariants", ()) or ()
        invs = tuple(sorted({str(x).strip().lower() for x in raw_inv if str(x).strip()}))

        res_mode = str(get_attr(port, "resource_mode", "value") or "value").strip().lower()
        if res_mode not in ("value", "acquire", "borrow", "release"):
            res_mode = "value"

        return cls(
            carrier=carrier,
            typestate=ts,
            accepted_states=accepted_states,
            domain=dom,
            rank=rank,
            shape=shape,
            dtype=dtype,
            channel_count=channel_count,
            channel_axis=channel_axis,
            length=length,
            rank_polymorphic=rank_polymorphic,
            invariants=invs,
            resource_mode=res_mode,
        )

    def state_key(self) -> Tuple[Any, ...]:
        return (
            self.carrier.lower(),
            self.typestate.lower(),
            self.rank,
            self.shape,
            (self.dtype or "").lower(),
            self.channel_count,
            self.channel_axis,
            self.length,
            self.invariants,
        )


def _is_symbolic_dtype(raw_dtype: Optional[str]) -> bool:
    """Return True if dtype is absent, wildcard ('any'/'*'), or an uppercase type variable (e.g. 'D')."""
    s = str(raw_dtype or "").strip()
    if not s or s.lower() in ("any", "*"):
        return True
    return len(s) == 1 and s.isupper()


def _is_symbolic_rank(raw_rank: Optional[Union[int, str]]) -> bool:
    """Return True if rank is absent or a symbolic rank variable (non-int, e.g. 'R')."""
    return raw_rank is None or isinstance(raw_rank, bool) or not isinstance(raw_rank, int)


def signatures_unify(prod: CanonicalPortType, cons: CanonicalPortType, poset: Any) -> bool:
    """
    Formal 4-component unification check:
      1. Parametric/multi-parent carrier subtyping
      2. Rank, dtype, channels, symbolic shape & length compatibility
      3. Multi-parent typestate DAG + disjunctive acceptance set
      4. Decidable numeric interval & predicate refinement entailment
    """
    if cons.carrier.lower() not in TOP_CARRIERS and not poset.is_subtype(prod.carrier, cons.carrier):
        return False

    if isinstance(cons.rank, int) and not cons.rank_polymorphic:
        if isinstance(prod.rank, int) and prod.rank != cons.rank:
            return False

    if not _is_symbolic_dtype(cons.dtype) and not _is_symbolic_dtype(prod.dtype):
        if (prod.dtype or "").strip().lower() != (cons.dtype or "").strip().lower():
            return False

    if cons.channel_count is not None and prod.channel_count is not None:
        if prod.channel_count != cons.channel_count:
            return False

    if cons.channel_axis is not None and prod.channel_axis is not None:
        if prod.channel_axis != cons.channel_axis:
            return False

    if not unify_shapes(
        prod.shape,
        cons.shape,
        prod_length=prod.length,
        cons_length=cons.length,
        cons_rank_poly=cons.rank_polymorphic,
    ):
        return False

    if not poset.is_state_compatible(
        prod.typestate,
        cons.typestate,
        cons_accepted=cons.accepted_states,
        prod_domain=prod.domain,
        cons_domain=cons.domain,
    ):
        return False

    if cons.invariants:
        prod_ref = RefinementSet.from_invariants(prod.invariants)
        cons_ref = RefinementSet.from_invariants(cons.invariants)
        if not prod_ref.entails(cons_ref, prod_dtype=prod.dtype):
            return False

    return True


def propagate_polymorphic_signature(
    src_sig: Optional[CanonicalPortType],
    dst_sig: CanonicalPortType,
    cell: Any,
) -> CanonicalPortType:
    """Preserve layout, dtype, typestate, or nominal carrier axes per `cell.polymorphic_propagation`."""
    if src_sig is None:
        return dst_sig
    pp = get_attr(cell, "polymorphic_propagation", None)
    if pp is None:
        return dst_sig

    pres_layout = bool(get_attr(pp, "preserves_layout", False))
    pres_dtype = bool(get_attr(pp, "preserves_dtype", False))
    pres_ts = bool(get_attr(pp, "preserves_typestate", False))
    pres_sem = bool(get_attr(pp, "preserves_semantics", False))
    pres_nom = bool(get_attr(pp, "preserves_nominal", False))

    out_carrier = (
        src_sig.carrier
        if (pres_nom and dst_sig.carrier.lower() in TOP_CARRIERS)
        else dst_sig.carrier
    )
    out_ts = (
        src_sig.typestate
        if ((pres_ts or pres_sem) and dst_sig.typestate == "any")
        else dst_sig.typestate
    )
    out_dom = src_sig.domain if (out_ts == src_sig.typestate and not dst_sig.domain) else dst_sig.domain

    out_rank = src_sig.rank if (pres_layout and _is_symbolic_rank(dst_sig.rank)) else dst_sig.rank
    out_shape = src_sig.shape if (pres_layout and dst_sig.shape is None) else dst_sig.shape
    out_dtype = src_sig.dtype if ((pres_layout or pres_dtype) and _is_symbolic_dtype(dst_sig.dtype)) else dst_sig.dtype
    out_ch = src_sig.channel_count if (pres_layout and dst_sig.channel_count is None) else dst_sig.channel_count
    out_ax = src_sig.channel_axis if (pres_layout and dst_sig.channel_axis is None) else dst_sig.channel_axis
    out_len = src_sig.length if (pres_layout and dst_sig.length is None) else dst_sig.length

    return CanonicalPortType(
        carrier=out_carrier,
        typestate=out_ts,
        accepted_states=dst_sig.accepted_states,
        domain=out_dom,
        rank=out_rank,
        shape=out_shape,
        dtype=out_dtype,
        channel_count=out_ch,
        channel_axis=out_ax,
        length=out_len,
        rank_polymorphic=dst_sig.rank_polymorphic,
        invariants=dst_sig.invariants,
        resource_mode=dst_sig.resource_mode,
    )


def join_port_types(
    a: CanonicalPortType,
    b: CanonicalPortType,
    poset: Any,
) -> CanonicalPortType:
    """Compute lattice join `a \\sqcup b` across carrier, layout, typestate, and refinements."""
    j_carrier = poset.join_carriers(a.carrier, b.carrier)
    j_ts = poset.join_states(a.typestate, b.typestate, a.domain, b.domain)
    j_rank = a.rank if a.rank == b.rank else None
    a_dt = (a.dtype or "").strip().lower()
    b_dt = (b.dtype or "").strip().lower()
    j_dtype = a.dtype if (a_dt and a_dt == b_dt) else None
    j_shape = join_shapes(a.shape, b.shape)
    j_len = a.length if a.length == b.length else None
    j_cc = a.channel_count if a.channel_count == b.channel_count else None
    j_ca = a.channel_axis if a.channel_axis == b.channel_axis else None
    j_poly = a.rank_polymorphic or b.rank_polymorphic
    j_invs = RefinementSet.from_invariants(a.invariants).join(
        RefinementSet.from_invariants(b.invariants)
    ).to_invariants()
    j_acc = tuple(s for s in a.accepted_states if s in b.accepted_states)
    j_dom = a.domain if a.domain == b.domain else ""
    j_rm = a.resource_mode if a.resource_mode == b.resource_mode else "value"

    return CanonicalPortType(
        carrier=j_carrier,
        typestate=j_ts,
        accepted_states=j_acc,
        domain=j_dom,
        rank=j_rank,
        shape=j_shape,
        dtype=j_dtype,
        channel_count=j_cc,
        channel_axis=j_ca,
        length=j_len,
        rank_polymorphic=j_poly,
        invariants=j_invs,
        resource_mode=j_rm,
    )


@dataclass(frozen=True)
class MorphismEdge:
    cell: Any
    cell_id: str
    domain: str
    in_port: str
    out_port: str
    src: CanonicalPortType
    dst: CanonicalPortType
    cost: float
    lossy: bool
    deltas: Tuple[str, ...]
    requires_intent: bool = False


@dataclass(frozen=True)
class LiveBridgePath:
    edges: Tuple[MorphismEdge, ...]
    total_cost: float
    lossy: bool
    deltas: Tuple[str, ...]

    @property
    def hop_count(self) -> int:
        return len(self.edges)

    @property
    def cell_ids(self) -> Tuple[str, ...]:
        return tuple(e.cell_id for e in self.edges)

    @property
    def is_tier2(self) -> bool:
        return bool(self.lossy or any(e.lossy or e.requires_intent for e in self.edges))


def _compute_deltas(src: CanonicalPortType, dst: CanonicalPortType) -> Tuple[str, ...]:
    deltas: List[str] = []
    if src.carrier.lower() != dst.carrier.lower():
        deltas.append("carrier")
    src_layout = (src.rank, src.shape, (src.dtype or "").lower(), src.channel_count, src.channel_axis, src.length)
    dst_layout = (dst.rank, dst.shape, (dst.dtype or "").lower(), dst.channel_count, dst.channel_axis, dst.length)
    if src_layout != dst_layout:
        deltas.append("layout")
    if src.typestate.lower() != dst.typestate.lower():
        deltas.append("typestate")
    if src.invariants != dst.invariants:
        deltas.append("invariants")
    return tuple(deltas)


def _apply_edge(cur_sig: CanonicalPortType, edge: MorphismEdge) -> CanonicalPortType:
    return propagate_polymorphic_signature(cur_sig, edge.dst, edge.cell)


def _allows_tier2_step(edge: MorphismEdge, dst: CanonicalPortType, poset: Any, *, allow_lossy: bool) -> bool:
    """Coercion coherence guard for Tier-2 (lossy or intent-bearing) conversions."""
    if not (edge.lossy or edge.requires_intent):
        return True
    if not allow_lossy:
        return False
    if "carrier" in edge.deltas and dst.carrier.lower() not in TOP_CARRIERS and not poset.is_subtype(edge.src.carrier, dst.carrier):
        return True
    if "typestate" in edge.deltas and dst.typestate != "any":
        return True
    if "layout" in edge.deltas and (
        not _is_symbolic_rank(dst.rank)
        or not _is_symbolic_dtype(dst.dtype)
        or dst.shape is not None
        or dst.length is not None
    ):
        return True
    if "invariants" in edge.deltas and bool(dst.invariants):
        return True
    return False


class LiveBridgeGraph:
    """
    Coherent runtime subcategory C_coerce of representation-coercion morphisms.

    Only admits pure/immutable unary bridges explicitly marked with
    `bridge_profile.is_bridge = True`, enforcing Tier-1 vs Tier-2 coercion coherence.
    """

    def __init__(self, edges: Tuple[MorphismEdge, ...], poset: Any) -> None:
        self.edges: Tuple[MorphismEdge, ...] = edges
        self.poset = poset

    @classmethod
    def build(
        cls,
        cells: Iterable[Any],
        poset: Any,
        *,
        include_lossy: bool = True,
    ) -> "LiveBridgeGraph":
        edges: List[MorphismEdge] = []
        for cell in cells:
            bp = get_attr(cell, "bridge_profile", None)
            if bp is None or not bool(get_attr(bp, "is_bridge", False)):
                continue

            mut = str(get_attr(cell, "mutation_type", "pure") or "pure").strip().lower()
            if mut not in ("pure", "immutable"):
                continue
            if cell_has_effect_kind(cell, DESTRUCTIVE_OR_CLEAN_EFFECT_KINDS):
                continue

            raw_inputs = get_attr(cell, "inputs", {}) or {}
            if not isinstance(raw_inputs, dict) or not raw_inputs:
                continue

            has_required_asset = False
            req_data: List[Tuple[str, Any]] = []
            req_non_data_no_default: List[str] = []

            for pname, pval in raw_inputs.items():
                is_req = bool(get_attr(pval, "required", True))
                if not is_req:
                    continue
                vc = derive_value_class(pval)
                if vc == "external_asset":
                    has_required_asset = True
                    break
                has_def = get_attr(pval, "default_value", None) is not None
                if vc == "data":
                    if not has_def:
                        req_data.append((str(pname), pval))
                elif not has_def:
                    req_non_data_no_default.append(str(pname))

            if has_required_asset or req_non_data_no_default or len(req_data) != 1:
                continue

            raw_outputs = get_attr(cell, "outputs", {}) or {}
            if not isinstance(raw_outputs, dict) or not raw_outputs:
                continue

            prim_out_name = get_attr(cell, "primary_out", None)
            if prim_out_name and prim_out_name in raw_outputs:
                out_pname, out_port = str(prim_out_name), raw_outputs[prim_out_name]
            else:
                first_k = next(iter(raw_outputs))
                out_pname, out_port = str(first_k), raw_outputs[first_k]

            in_pname, in_port = req_data[0]
            dom = str(get_attr(cell, "domain_name", None) or "").strip().lower()
            cid = str(get_attr(cell, "cell_id", "") or "").strip()
            if not cid:
                continue

            src = CanonicalPortType.from_port(in_port, dom)
            dst = CanonicalPortType.from_port(out_port, dom)
            if src.resource_mode not in ("value", "borrow") or dst.resource_mode not in ("value", "borrow"):
                continue

            deltas = _compute_deltas(src, dst)
            if not deltas:
                continue

            bp_lossy = bool(get_attr(bp, "lossy", False))
            req_intent = bool(get_attr(bp, "requires_intent", False))
            rank_drop = isinstance(src.rank, int) and isinstance(dst.rank, int) and dst.rank < src.rank
            lossy = bool(bp_lossy or rank_drop)
            is_tier2 = bool(lossy or req_intent)

            if not include_lossy and is_tier2:
                continue

            bp_cost = get_attr(bp, "cost_tier", None)
            if isinstance(bp_cost, (int, float)) and float(bp_cost) > 0.0:
                cost = float(bp_cost)
            else:
                cost = 0.5 if "carrier" not in deltas else 1.0
            if is_tier2 and cost < 10.0:
                cost += 10.0

            edges.append(
                MorphismEdge(
                    cell=cell,
                    cell_id=cid,
                    domain=dom,
                    in_port=in_pname,
                    out_port=out_pname,
                    src=src,
                    dst=dst,
                    cost=cost,
                    lossy=lossy,
                    deltas=deltas,
                    requires_intent=req_intent,
                )
            )

        edges.sort(key=lambda e: (e.lossy or e.requires_intent, e.cost, e.cell_id))
        return cls(tuple(edges), poset)

    def find_bridges(
        self,
        src: CanonicalPortType,
        dst: CanonicalPortType,
        max_hops: int = 3,
        max_results: int = 3,
        *,
        allow_lossy: bool = True,
        allowed_domains: Optional[Set[str]] = None,
    ) -> List[LiveBridgePath]:
        if signatures_unify(src, dst, self.poset):
            return [LiveBridgePath(edges=(), total_cost=0.0, lossy=False, deltas=())]

        if max_hops <= 0 or max_results <= 0 or not self.edges:
            return []

        results: List[LiveBridgePath] = []
        seen_chains: Set[Tuple[str, ...]] = set()
        best_seen: Dict[Tuple[Tuple[Any, ...], bool], Tuple[float, int]] = {
            (src.state_key(), False): (0.0, 0)
        }

        seq = 0
        pq: List[Tuple[bool, float, int, Tuple[str, ...], int, CanonicalPortType, Tuple[MorphismEdge, ...], bool, Tuple[str, ...]]] = [
            (False, 0.0, 0, (), seq, src, (), False, ())
        ]

        while pq and len(results) < max_results:
            cur_t2, cur_cost, cur_hops, cur_cids, _, cur_sig, cur_edges, cur_lossy, cur_deltas = heapq.heappop(pq)
            if cur_hops >= max_hops:
                continue

            for edge in self.edges:
                if allowed_domains is not None and edge.domain and edge.domain.lower() not in allowed_domains:
                    continue
                if edge.cell_id in cur_cids:
                    continue
                if not _allows_tier2_step(edge, dst, self.poset, allow_lossy=allow_lossy):
                    continue
                if not signatures_unify(cur_sig, edge.src, self.poset):
                    continue

                nxt_sig = _apply_edge(cur_sig, edge)
                nxt_cost = cur_cost + edge.cost
                nxt_hops = cur_hops + 1
                nxt_cids = cur_cids + (edge.cell_id,)
                nxt_edges = cur_edges + (edge,)
                nxt_lossy = cur_lossy or edge.lossy
                nxt_t2 = cur_t2 or edge.lossy or edge.requires_intent
                nxt_deltas = tuple(dict.fromkeys(cur_deltas + edge.deltas))

                if signatures_unify(nxt_sig, dst, self.poset):
                    if nxt_cids not in seen_chains:
                        seen_chains.add(nxt_cids)
                        results.append(
                            LiveBridgePath(
                                edges=nxt_edges,
                                total_cost=round(nxt_cost, 6),
                                lossy=nxt_lossy,
                                deltas=nxt_deltas,
                            )
                        )
                        if len(results) >= max_results:
                            break
                    continue

                if nxt_hops < max_hops:
                    prune_key = (nxt_sig.state_key(), nxt_t2)
                    prev = best_seen.get(prune_key)
                    if prev is not None and (prev[0] < nxt_cost - 1e-9 or (abs(prev[0] - nxt_cost) <= 1e-9 and prev[1] <= nxt_hops)):
                        continue
                    best_seen[prune_key] = (nxt_cost, nxt_hops)
                    seq += 1
                    heapq.heappush(
                        pq,
                        (nxt_t2, nxt_cost, nxt_hops, nxt_cids, seq, nxt_sig, nxt_edges, nxt_lossy, nxt_deltas),
                    )

        results.sort(key=lambda r: (r.is_tier2, r.total_cost, r.hop_count, r.cell_ids))
        return results[:max_results]


__all__ = [
    "TOP_CARRIERS",
    "TOP_STATES",
    "BoundedPoset",
    "CanonicalPortType",
    "SimpleTypePoset",
    "signatures_unify",
    "propagate_polymorphic_signature",
    "join_port_types",
    "MorphismEdge",
    "LiveBridgePath",
    "LiveBridgeGraph",
]
