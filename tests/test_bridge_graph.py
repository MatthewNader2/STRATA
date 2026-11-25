"""Deterministic tests for STRATA's live cross-domain coercion subcategory LiveBridgeGraph."""
from __future__ import annotations

from bridge_graph import CanonicalPortType, LiveBridgeGraph, LiveBridgePath, MorphismEdge, _allows_tier2_step
from schema import CellSchema, PortSchema
from type_system import BoundedPoset


def test_default_lattice_admits_all_22_bridges(compiled_lattice) -> None:
    poset = BoundedPoset(compiled_lattice.type_parents, compiled_lattice.state_parents, compiled_lattice.top_types)
    graph = LiveBridgeGraph.build(compiled_lattice.cells, poset, include_lossy=True)

    declared_bridges = {c.cell_id for c in compiled_lattice.cells if c.bridge_profile and c.bridge_profile.is_bridge}
    loaded_bridges = {e.cell_id for e in graph.edges}

    assert declared_bridges == loaded_bridges
    assert len(loaded_bridges) == 22
    tier2 = [e for e in graph.edges if e.lossy or e.requires_intent]
    tier1 = [e for e in graph.edges if not (e.lossy or e.requires_intent)]
    assert len(tier1) == 16 and len(tier2) == 6
    assert all(e.cost >= 10.0 for e in tier2)

    for non_bridge in ("NUMPY_MEAN_ALL", "CRYPTO_SHA256_ONESHOT_HEX", "CV2_ERODE", "sns_facetgrid_init"):
        assert non_bridge not in loaded_bridges


def test_tier1_strictly_precedes_tier2_paths(compiled_lattice) -> None:
    poset = BoundedPoset(compiled_lattice.type_parents, compiled_lattice.state_parents, compiled_lattice.top_types)
    graph = LiveBridgeGraph.build(compiled_lattice.cells, poset, include_lossy=True)

    bgr = CanonicalPortType(carrier="ndarray", typestate="numpy::color_bgr", domain="numpy", rank=3, dtype="uint8")
    rgb = CanonicalPortType(carrier="ndarray", typestate="numpy::color_rgb", domain="numpy", rank=3, dtype="uint8")
    gray = CanonicalPortType(carrier="ndarray", typestate="numpy::gray", domain="numpy", rank=2, dtype="uint8")
    pil_rgb = CanonicalPortType(carrier="Image", typestate="pillow::mode_rgb", domain="pillow", rank=3, dtype="uint8")

    # Lossless BGR -> RGB conversion
    p_rgb = graph.find_bridges(bgr, rgb, max_hops=2, allow_lossy=False)
    assert p_rgb and not p_rgb[0].is_tier2 and p_rgb[0].cell_ids == ("CV2_CVT_COLOR_BGR2RGB",)

    # Lossy BGR -> Gray forbidden under allow_lossy=False, permitted under allow_lossy=True
    assert graph.find_bridges(bgr, gray, max_hops=2, allow_lossy=False) == []
    p_gray = graph.find_bridges(bgr, gray, max_hops=2, allow_lossy=True)
    assert p_gray and p_gray[0].is_tier2 and p_gray[0].total_cost >= 10.0

    # Cross-domain ndarray -> PIL.Image bridge
    p_pil = graph.find_bridges(bgr, pil_rgb, max_hops=2, allow_lossy=True)
    assert p_pil and not p_pil[0].is_tier2 and p_pil[0].cell_ids[0] == "PILLOW_FROM_NDARRAY_BGR"

    # Multi-hop cross-domain PIL.RGB -> ndarray gray (2 hops) and Gray -> PIL.RGB (2 Tier-1 hops)
    p_pil_gray = graph.find_bridges(pil_rgb, gray, max_hops=3, allow_lossy=True)
    assert p_pil_gray and p_pil_gray[0].hop_count == 2
    p_gray_pil = graph.find_bridges(gray, pil_rgb, max_hops=3, allow_lossy=False)
    assert p_gray_pil and not p_gray_pil[0].is_tier2 and p_gray_pil[0].hop_count == 2


def test_tier2_sort_key_and_carrier_delta_guard() -> None:
    poset = BoundedPoset(type_parents={"bool": {"int"}, "int": {"any"}})
    dummy_cell = CellSchema(
        cell_id="LOSSY_CAST",
        domain_name="test",
        code_template="{out} = int({x})",
        inputs={"x": PortSchema(carrier="bool", typestate="any")},
        outputs={"out": PortSchema(carrier="int", typestate="any")},
    )
    t1_edge = MorphismEdge(
        cell=dummy_cell,
        cell_id="TIER1_HOP",
        domain="test",
        in_port="x",
        out_port="out",
        src=CanonicalPortType(carrier="bool"),
        dst=CanonicalPortType(carrier="int"),
        cost=50.0,
        lossy=False,
        deltas=("carrier",),
        requires_intent=False,
    )
    t2_edge = MorphismEdge(
        cell=dummy_cell,
        cell_id="TIER2_HOP",
        domain="test",
        in_port="x",
        out_port="out",
        src=CanonicalPortType(carrier="bool"),
        dst=CanonicalPortType(carrier="int"),
        cost=0.1,
        lossy=True,
        deltas=("carrier",),
        requires_intent=False,
    )

    p_t1 = LiveBridgePath(edges=(t1_edge,), total_cost=50.0, lossy=False, deltas=("carrier",))
    p_t2 = LiveBridgePath(edges=(t2_edge,), total_cost=0.1, lossy=True, deltas=("carrier",))
    ordered = sorted([p_t2, p_t1], key=lambda p: (p.is_tier2, p.total_cost, p.hop_count, p.cell_ids))
    assert ordered[0] is p_t1

    # Carrier delta guard rejects lossy carrier bridge when edge.src.carrier <= dst.carrier
    assert _allows_tier2_step(t2_edge, CanonicalPortType(carrier="int"), poset, allow_lossy=True) is False
