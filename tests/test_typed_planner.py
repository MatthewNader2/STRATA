"""Deterministic unit tests for TypedIRPlanner resource tracking, borrows, and domain scoping."""
from __future__ import annotations

import json
from pathlib import Path

from bridge_graph import LiveBridgeGraph
from schema import CellSchema, LayoutContract, PortSchema, TreeSchema
from strata_pipeline import StrataPipeline
from subgoal_retrieval import CandidateMatch
from type_system import BoundedPoset
from typed_ir import IRNode, TypedIR
from typed_planner import TypedIRPlanner, _file_format_mismatch_penalty

ROOT_DIR = Path(__file__).resolve().parent.parent


def _cm(cell: CellSchema) -> list[CandidateMatch]:
    return [CandidateMatch(cell_id=cell.cell_id, domain=cell.domain_name, match_mode="exact", score=1.0, cell=cell)]


def _simple_scalar_cells(compiled_lattice):
    open_cell = compiled_lattice.loaded_cells["SQLITE_CONNECT_MEMORY"]
    close_cell = compiled_lattice.loaded_cells["SQLITE_CLOSE"]
    const_list = CellSchema(
        cell_id="SYNTH_INT_LIST",
        domain_name="python_core",
        code_template="{out} = [1, 2]",
        inputs={},
        outputs={"out": PortSchema(carrier="List[int]", typestate="any", layout=LayoutContract(rank=1, dtype="int"))},
    )
    const_int = CellSchema(
        cell_id="SYNTH_INT",
        domain_name="python_core",
        code_template="{out} = 1",
        inputs={},
        outputs={"out": PortSchema(carrier="int", typestate="any", layout=LayoutContract(rank=0))},
    )
    const_float = CellSchema(
        cell_id="SYNTH_FLOAT",
        domain_name="python_core",
        code_template="{out} = 1.0",
        inputs={},
        outputs={"out": PortSchema(carrier="float", typestate="any", layout=LayoutContract(rank=0))},
    )
    const_bool = CellSchema(
        cell_id="SYNTH_BOOL",
        domain_name="python_core",
        code_template="{out} = True",
        inputs={},
        outputs={"out": PortSchema(carrier="bool", typestate="any", layout=LayoutContract(rank=0))},
    )
    return open_cell, close_cell, const_list, const_int, const_float, const_bool


def test_linear_acquire_release_and_use_after_release(compiled_lattice) -> None:
    pipe = StrataPipeline.from_compiled(compiled_lattice)

    res_leak = pipe.run(
        "db = sqlite3::SQLITE_CONNECT_MEMORY()\n"
        'df = pandas::PD_READ_SQL(sql="SELECT 1", con=$db)'
    )
    assert not res_leak.ok and res_leak.refusal_kind == "UNCLOSED_RESOURCE"

    res_uaf = pipe.run(
        "db = sqlite3::SQLITE_CONNECT_MEMORY()\n"
        "done = sqlite3::SQLITE_CLOSE(connection=$db)\n"
        'df = pandas::PD_READ_SQL(sql="SELECT 1", con=$db)'
    )
    assert not res_uaf.ok and res_uaf.refusal_kind is not None

    res_ok = pipe.run(
        "db = sqlite3::SQLITE_CONNECT_MEMORY()\n"
        'df = pandas::PD_READ_SQL(sql="SELECT 1", con=$db)\n'
        "done = sqlite3::SQLITE_CLOSE(connection=$db)"
    )
    assert res_ok.ok


def test_borrow_owner_lifetime_safety_cryopulse_and_matplotlib(compiled_lattice) -> None:
    core_tree = json.loads((ROOT_DIR / "trees" / "python_core.json").read_text(encoding="utf-8"))
    cryo_tree = json.loads((ROOT_DIR / "benchmarks" / "fixtures" / "cryopulse" / "cryopulse.json").read_text(encoding="utf-8"))
    cryo_pipe = StrataPipeline.from_trees([core_tree, cryo_tree])

    res_cryo_dangling = cryo_pipe.run(
        'vessel = cryopulse::CRYOPULSE_OPEN_CRYO_VESSEL(chamber_tag="DR-1", target_mk=12.0)\n'
        'channel = cryopulse::CRYOPULSE_ENERGIZE_FLUX_COIL(vessel=$vessel, rail_id="R1", bias_ua=30.0)\n'
        "cryopulse::CRYOPULSE_VENT_CRYO_VESSEL(vessel=$vessel)\n"
        "trace = cryopulse::CRYOPULSE_SAMPLE_SQUID_FLUX(channel=$channel, window_ns=256, shots=1024)"
    )
    assert not res_cryo_dangling.ok
    assert "dangling_borrow" in (res_cryo_dangling.error or "")

    mpl_pipe = StrataPipeline.from_compiled(compiled_lattice)
    res_mpl_dangling = mpl_pipe.run(
        "arr = numpy::NUMPY_LINSPACE(start=0.0, stop=10.0, num=20)\n"
        "fig, ax = matplotlib::matplotlib_plt_subplots_single()\n"
        "matplotlib::matplotlib_plt_close(fig=$fig)\n"
        "matplotlib::matplotlib_ax_plot(ax=$ax, y=$arr)"
    )
    assert not res_mpl_dangling.ok
    assert "dangling_borrow" in (res_mpl_dangling.error or "")


def test_loop_and_branch_condition_liveness_and_joins(compiled_lattice) -> None:
    poset = BoundedPoset(compiled_lattice.type_parents, compiled_lattice.state_parents, compiled_lattice.top_types)
    planner = TypedIRPlanner(LiveBridgeGraph([], poset), poset)
    open_cell, close_cell, c_list, c_int, c_float, c_bool = _simple_scalar_cells(compiled_lattice)

    # Loop releasing outer handle must fail
    ir_rel = TypedIR(
        raw_prompt="loop_rel",
        nodes=[
            IRNode(id="xs", kind="call", intent="SYNTH_INT_LIST", outputs=["xs"]),
            IRNode(id="db", kind="call", intent="SQLITE_CONNECT_MEMORY", outputs=["db"]),
            IRNode(
                id="lp",
                kind="loop",
                intent="for x in xs",
                inputs=["xs"],
                condition="x in xs",
                body=[IRNode(id="cl", kind="call", intent="SQLITE_CLOSE", inputs=["db"], outputs=["cl"])],
            ),
        ],
        source_mode="formal_syntax",
    )
    r_rel = planner.plan(ir_rel, {"xs": _cm(c_list), "db": _cm(open_cell), "cl": _cm(close_cell)})
    assert not r_rel.ok and r_rel.refusal_kind == "UNBALANCED_LOOP_RESOURCES"

    # Branch condition referencing a released handle must be rejected cleanly
    ir_cond_dead = TypedIR(
        raw_prompt="cond_dead",
        nodes=[
            IRNode(id="db", kind="call", intent="SQLITE_CONNECT_MEMORY", outputs=["db"]),
            IRNode(id="cl", kind="call", intent="SQLITE_CLOSE", inputs=["db"], outputs=["cl"]),
            IRNode(
                id="br",
                kind="branch",
                intent="if db",
                inputs=["db"],
                condition="$db is not None",
                body=[IRNode(id="t1", kind="call", intent="SYNTH_INT", outputs=["t1"])],
            ),
        ],
        source_mode="formal_syntax",
    )
    r_dead = planner.plan(ir_cond_dead, {"db": _cm(open_cell), "cl": _cm(close_cell), "t1": _cm(c_int)})
    assert not r_dead.ok

    # Two-arm branch joins int + float to carrier LUB and 1-arm widens to Optional[int]
    ir_join = TypedIR(
        raw_prompt="br_join",
        nodes=[
            IRNode(id="c", kind="call", intent="SYNTH_BOOL", outputs=["c"]),
            IRNode(
                id="br",
                kind="branch",
                intent="if c",
                inputs=["c"],
                condition="c",
                outputs=["res"],
                body=[IRNode(id="t1", kind="call", intent="SYNTH_INT", outputs=["t1"])],
                else_body=[IRNode(id="e1", kind="call", intent="SYNTH_FLOAT", outputs=["e1"])],
            ),
        ],
        source_mode="formal_syntax",
    )
    r_join = planner.plan(ir_join, {"c": _cm(c_bool), "t1": _cm(c_int), "e1": _cm(c_float)})
    assert r_join.ok and r_join.chosen_plan is not None
    assert r_join.chosen_plan.steps[-1].output_sig.carrier == poset.join_carriers("int", "float")


def test_file_format_mismatch_penalty_extension_equivalence() -> None:
    png_cell = CellSchema(
        cell_id="SAVE_IMG",
        domain_name="img",
        code_template="save({dst})",
        inputs={"dst": PortSchema(carrier="File[Image, JPEG]", typestate="any", value_class="external_asset", required=True)},
        outputs={"out": PortSchema(carrier="File[Image, JPEG]", typestate="any", value_class="external_asset")},
    )
    pq_cell = CellSchema(
        cell_id="SAVE_PQ",
        domain_name="tab",
        code_template="save({dst})",
        inputs={"dst": PortSchema(carrier="File[Tabular, Parquet]", typestate="any", value_class="external_asset", required=True)},
        outputs={"out": PortSchema(carrier="File[Tabular, Parquet]", typestate="any", value_class="external_asset")},
    )

    from typed_ir import IRParamValue

    node_jpg = IRNode(
        id="n1",
        kind="call",
        intent="SAVE_IMG",
        params={"dst": IRParamValue(value="photo.jpg", provenance="prompt", value_class="external_asset")},
    )
    node_png = IRNode(
        id="n2",
        kind="call",
        intent="SAVE_IMG",
        params={"dst": IRParamValue(value="photo.png", provenance="prompt", value_class="external_asset")},
    )
    node_pq = IRNode(
        id="n3",
        kind="call",
        intent="SAVE_PQ",
        params={"dst": IRParamValue(value="data.pq", provenance="prompt", value_class="external_asset")},
    )

    assert _file_format_mismatch_penalty(node_jpg, png_cell, {}) == 0.0
    assert _file_format_mismatch_penalty(node_png, png_cell, {}) > 0.0
    assert _file_format_mismatch_penalty(node_pq, pq_cell, {}) == 0.0


def test_domain_scope_isolation_refuses_unloaded_pillow() -> None:
    core_tree = TreeSchema(**json.loads((ROOT_DIR / "trees" / "python_core.json").read_text(encoding="utf-8")))
    np_tree = TreeSchema(**json.loads((ROOT_DIR / "trees" / "numpy.json").read_text(encoding="utf-8")))
    cv2_tree = TreeSchema(**json.loads((ROOT_DIR / "trees" / "cv2.json").read_text(encoding="utf-8")))
    pipe = StrataPipeline.from_trees([core_tree, np_tree, cv2_tree])

    res = pipe.run('img = pillow::PILLOW_IMAGE_OPEN(image_path=@"input.png")')
    assert not res.ok and res.refusal_kind == "UNAVAILABLE_DOMAIN"
