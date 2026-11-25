"""End-to-end Profile 0 synthesis + live GEVRSandbox execution and manifest invariants."""
from __future__ import annotations

import json
from pathlib import Path

from schema import TreeSchema
from strata_pipeline import StrataPipeline
from tests.conftest import ROOT_DIR


def test_profile0_cv2_canny_and_live_bridges_execute_cleanly(profile0_pipeline) -> None:
    res_edges = profile0_pipeline.run(
        'img = cv2::CV2_IMREAD(image_path=@"input.png", flags=1)\n'
        "gray = cv2::CV2_CVT_COLOR_BGR2GRAY(src=$img)\n"
        "blur = cv2::CV2_GAUSSIAN_BLUR(src=$gray, ksize=(5, 5), sigma_x=1.5)\n"
        "edges = cv2::CV2_CANNY(src=$blur, threshold1=50.0, threshold2=150.0)\n"
        'out = cv2::CV2_IMWRITE(output_path=@"edges_out.png", image=$edges)',
        execute=True,
    )
    assert res_edges.ok and res_edges.sandbox_result and res_edges.sandbox_result["success"] is True

    res_bridge = profile0_pipeline.run(
        'img = cv2::CV2_IMREAD(image_path=@"input.png", flags=1)\n'
        "edges = cv2::CV2_CANNY(src=$img, threshold1=50.0, threshold2=150.0)",
        execute=True,
    )
    assert res_bridge.ok and res_bridge.live_bridge_count >= 1
    assert res_bridge.sandbox_result and res_bridge.sandbox_result["success"] is True

    res_cross = profile0_pipeline.run(
        'pil_img = pillow::PILLOW_IMAGE_OPEN(image_path=@"input.png")\n'
        "edges = cv2::CV2_CANNY(src=$pil_img, threshold1=50.0, threshold2=150.0)",
        execute=True,
    )
    assert res_cross.ok and res_cross.live_bridge_count >= 1
    assert res_cross.sandbox_result and res_cross.sandbox_result["success"] is True


def test_profile0_sqlite_pandas_and_matplotlib_multiout_execute_cleanly(profile0_pipeline) -> None:
    res_sql = profile0_pipeline.run(
        "db = sqlite3::SQLITE_CONNECT_MEMORY()\n"
        'df = pandas::PD_READ_SQL(sql="SELECT 1 AS a, 2 AS b", con=$db)\n'
        "done = sqlite3::SQLITE_CLOSE(connection=$db)",
        execute=True,
    )
    assert res_sql.ok and res_sql.sandbox_result and res_sql.sandbox_result["success"] is True

    res_mpl = profile0_pipeline.run(
        "arr = numpy::NUMPY_LINSPACE(start=0.0, stop=10.0, num=20)\n"
        "fig, ax = matplotlib::matplotlib_plt_subplots_single()\n"
        "matplotlib::matplotlib_ax_plot(ax=$ax, y=$arr)\n"
        "matplotlib::matplotlib_ax_legend(ax=$ax)\n"
        "matplotlib::matplotlib_plt_close(fig=$fig)",
        execute=True,
    )
    assert res_mpl.ok and res_mpl.sandbox_result and res_mpl.sandbox_result["success"] is True


def test_profile0_blind_unseen_sdks_and_leak_refusals() -> None:
    core_tree = json.loads((ROOT_DIR / "trees" / "python_core.json").read_text(encoding="utf-8"))
    cryo_tree = json.loads((ROOT_DIR / "benchmarks" / "fixtures" / "cryopulse" / "cryopulse.json").read_text(encoding="utf-8"))
    ae_tree = json.loads((ROOT_DIR / "benchmarks" / "fixtures" / "aetheris" / "aetheris.json").read_text(encoding="utf-8"))

    pipe = StrataPipeline.from_trees([core_tree, cryo_tree, ae_tree])

    res_ae_ok = pipe.run(
        'b = aetheris::AETHERIS_MOUNT_OPTICAL_TABLE(rack_id="R1", wavelength_nm=1550.0)\n'
        "drift = aetheris::AETHERIS_MEASURE_BENCH_DRIFT(bench=$b)\n"
        "done = aetheris::AETHERIS_PARK_OPTICAL_TABLE(bench=$b)"
    )
    assert res_ae_ok.ok

    res_ae_leak = pipe.run(
        'b = aetheris::AETHERIS_MOUNT_OPTICAL_TABLE(rack_id="R1", wavelength_nm=1550.0)\n'
        "drift = aetheris::AETHERIS_MEASURE_BENCH_DRIFT(bench=$b)"
    )
    assert not res_ae_leak.ok and res_ae_leak.refusal_kind == "UNCLOSED_RESOURCE"

    res_cp_leak = pipe.run(
        'v = cryopulse::CRYOPULSE_OPEN_CRYO_VESSEL(chamber_tag="DR-1", target_mk=12.0)\n'
        'c = cryopulse::CRYOPULSE_ENERGIZE_FLUX_COIL(vessel=$v, rail_id="R1", bias_ua=30.0)'
    )
    assert not res_cp_leak.ok and res_cp_leak.refusal_kind == "UNCLOSED_RESOURCE"


def test_all_domain_trees_validate_and_maintain_wave2_neutrality() -> None:
    tree_paths = sorted((ROOT_DIR / "trees").glob("*.json"))
    assert len(tree_paths) == 15

    fixture_paths = [
        ROOT_DIR / "benchmarks" / "fixtures" / "cryopulse" / "cryopulse.json",
        ROOT_DIR / "benchmarks" / "fixtures" / "aetheris" / "aetheris.json",
    ]
    for p in tree_paths + fixture_paths:
        raw = json.loads(Path(p).read_text(encoding="utf-8"))
        ts = TreeSchema(**raw)
        if not ts.is_core and ts.domain not in ("control_flow", "cryopulse", "aetheris"):
            assert len(ts.aliases) >= 1, f"Domain '{ts.domain}' missing aliases"
        for cell in ts.cells:
            assert not cell.cell_id.startswith("SHUTIL_")
            assert not cell.cell_id.startswith("pickle.")
            for port_name in list(cell.inputs.keys()) + list(cell.outputs.keys()):
                assert not port_name.startswith("port_"), f"Opaque port '{port_name}' in {ts.domain}:{cell.cell_id}"
