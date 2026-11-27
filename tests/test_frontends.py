"""Deterministic unit tests for STRATA formal_regex and slm_compiler frontends."""
from __future__ import annotations

import json

from frontends.formal_regex import parse_formal_prompt
from frontends.slm_compiler import (
    SLM_SYSTEM_PROMPT_TEMPLATE,
    _is_concrete_external_asset,
    _is_path_or_uri_shape,
    _parse_numeric_with_unit_suffix,
    compile_slm_typed_ir,
)


def test_formal_regex_assignment_dag_and_pipe_syntax() -> None:
    dag = (
        'img = cv2::CV2_IMREAD(image_path=@"input.png", flags=1)\n'
        "blur = cv2::CV2_GAUSSIAN_BLUR(src=$img, ksize=(5, 5), sigma_x=1.5)"
    )
    res_dag = parse_formal_prompt(dag)
    assert res_dag.ok and res_dag.ir is not None
    assert len(res_dag.ir.nodes) == 2
    assert res_dag.ir.nodes[0].domain_hint == "cv2"
    assert res_dag.ir.nodes[0].params["image_path"].value_class == "external_asset"
    assert res_dag.ir.nodes[1].params["ksize"].value == (5, 5)
    assert res_dag.ir.nodes[1].inputs == ["img"]

    res_pipe = parse_formal_prompt('OP_A(k=3, flag=true) |> OP_B(scale=2.5)')
    assert res_pipe.ok and res_pipe.ir is not None
    assert [n.intent for n in res_pipe.ir.nodes] == ["OP_A", "OP_B"]
    assert res_pipe.ir.nodes[0].params["flag"].value is True
    assert res_pipe.ir.nodes[1].inputs == ["v1"]


def test_formal_regex_rejects_natural_language_and_invalid_syntax() -> None:
    res_nl = parse_formal_prompt('load image "input.png", convert to grayscale, and save to "out.png"')
    assert not res_nl.ok and res_nl.ir is None

    res_mix = parse_formal_prompt("a = OP1() |> OP2()")
    assert not res_mix.ok and res_mix.ir is None


def test_slm_prompt_has_no_benchmark_leakage() -> None:
    prompt_words = {
        w.strip(".,;:()[]{}/\"'")
        for w in SLM_SYSTEM_PROMPT_TEMPLATE.replace("/", " ").split()
    }
    for leaked in ("vent", "park", "quit", "cryopulse", "aetheris", "nm", "mK", "mW", "GHz", "ns"):
        assert leaked not in prompt_words


def test_asset_shape_classifier_and_unit_stripping() -> None:
    assert _is_path_or_uri_shape("input.png") is True
    assert _is_path_or_uri_shape("sub/out.csv") is True
    assert _is_path_or_uri_shape("https://example.com/item") is True
    assert _is_path_or_uri_shape("user:123") is False
    assert _is_path_or_uri_shape("localhost:8080") is False
    assert _is_path_or_uri_shape("v1.2") is False
    assert _is_path_or_uri_shape("127.0.0.1") is False

    assert _is_concrete_external_asset("input.png", 'read "input.png"') is True
    assert _is_concrete_external_asset("user:123", "fetch user:123") is False
    assert _is_concrete_external_asset("ungrounded.png", 'read "other.png"') is False

    assert _parse_numeric_with_unit_suffix("4.2K") == 4.2
    assert _parse_numeric_with_unit_suffix("50px") == 50
    assert _parse_numeric_with_unit_suffix("12.5 ms") == 12.5
    assert _parse_numeric_with_unit_suffix("abc") is None


def test_compile_slm_typed_ir_recovers_quoted_literals_and_drops_non_path_assets() -> None:
    raw_prompt = 'using demo_pkg, open device tag "K7" at 4.2 and save to "result.bin"'
    llm_payload = {
        "nodes": [
            {
                "id": "n1",
                "kind": "call",
                "intent": 'open device tag "K7"',
                "inputs": [],
                "outputs": ["dev"],
                "params": {"level": {"value": 4.2, "provenance": "prompt", "value_class": "literal"}},
            },
            {
                "id": "n2",
                "kind": "call",
                "intent": "save output",
                "inputs": ["dev"],
                "outputs": ["saved"],
                "params": {"dest": {"value": "result.bin", "provenance": "prompt", "value_class": "external_asset"}},
            },
        ],
        "assets": {
            "dest_asset": {"value": "result.bin", "provenance": "prompt", "value_class": "external_asset"},
            "bogus": {"value": "user:123", "provenance": "prompt", "value_class": "external_asset"},
        },
        "deliverable": "saved",
    }

    res = compile_slm_typed_ir(
        raw_prompt,
        lambda p, s, sp: json.dumps(llm_payload),
        loaded_domains=["demo_pkg"],
    )
    assert res.ok and res.ir is not None
    assert "bogus" not in res.ir.assets
    recovered_vals = [pv.value for pv in res.ir.nodes[0].params.values()]
    assert "K7" in recovered_vals
    assert any("Dropped non-path asset entry" in w for w in res.warnings)
