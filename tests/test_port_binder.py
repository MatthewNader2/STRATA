"""Deterministic unit tests for Symbolic Port Binding and Backend-Agnostic Language Lowering."""
from __future__ import annotations

import json

from bridge_graph import LiveBridgeGraph
from language_backend import LexicalLanguageBackend, get_active_language_spec, set_active_language_spec
from port_binder import bind_and_emit
from schema import LanguageSpec
from strata_pipeline import StrataPipeline
from type_system import BoundedPoset


def test_port_binder_enum_and_refinement_enforcement(compiled_lattice) -> None:
    pipe = StrataPipeline.from_compiled(compiled_lattice)

    res_bad_k = pipe.run("a = numpy::NUMPY_ZEROS_1D(length=6)\nb = numpy::NUMPY_RESHAPE_NCOL(array=$a, n_columns=0)")
    assert not res_bad_k.ok and res_bad_k.refusal_kind == "REFINEMENT_VIOLATION"

    res_bad_enum = pipe.run('img = cv2::CV2_IMREAD(image_path=@"input.png", flags=999)')
    assert not res_bad_enum.ok and res_bad_enum.refusal_kind == "INVALID_ENUM_VALUE"


def test_input_assets_vs_egress_paths_separation(compiled_lattice) -> None:
    pipe = StrataPipeline.from_compiled(compiled_lattice)
    res = pipe.run(
        'img = cv2::CV2_IMREAD(image_path=@"input.png", flags=1)\n'
        "gray = cv2::CV2_CVT_COLOR_BGR2GRAY(src=$img)\n"
        "edges = cv2::CV2_CANNY(src=$gray, threshold1=50.0, threshold2=150.0)\n"
        'out = cv2::CV2_IMWRITE(output_path=@"edges_out.png", image=$edges)'
    )
    assert res.ok and res.plan is not None and res.ir is not None
    bound_res = bind_and_emit(res.plan, res.ir)
    assert bound_res.ok
    assert "input.png" in bound_res.input_assets
    assert "edges_out.png" in bound_res.egress_paths
    assert "edges_out.png" not in bound_res.input_assets


def test_lexical_extract_lhs_placeholders_ignores_kwargs_and_handles_multi_lhs() -> None:
    backend = LexicalLanguageBackend(LanguageSpec(name="python", ast_backend="lexical", assignment_operator="=", line_comment="#"))

    assert backend.extract_lhs_placeholders(
        "{data}.to_csv({filepath}, index=False)",
        ["data", "filepath", "output_var"],
    ) == ["output_var"]
    assert backend.extract_lhs_placeholders(
        "pd.read_sql(sql={sql}, con={con})",
        ["sql", "con", "output_var"],
    ) == ["output_var"]
    assert backend.extract_lhs_placeholders(
        "{fig}.savefig({fname}, dpi=150)",
        ["fig", "fname", "output_var"],
    ) == ["output_var"]
    assert backend.extract_lhs_placeholders(
        "{fig}, {ax} = plt.subplots(figsize=({w}, {h}))",
        ["fig", "ax", "w", "h"],
    ) == ["fig", "ax"]


def test_non_python_language_spec_lowering_and_restoration() -> None:
    prev_spec = get_active_language_spec()
    rust_tree = {
        "domain": "rust_core",
        "version": "2.0.0",
        "is_core": True,
        "language_spec": {
            "name": "rust",
            "file_extension": ".rs",
            "ast_backend": "lexical",
            "assignment_operator": "=",
            "statement_terminator": ";",
            "line_comment": "//",
            "dependency_prefixes": ["use "],
            "loop_keyword": "for",
            "loop_in_keyword": "in",
            "block_suffix": "{",
            "null_literal": "None",
            "bool_literals": ["true", "false"],
            "noop_statement": "()",
            "return_keyword": "return",
        },
        "cells": [
            {
                "cell_id": "RUST_INIT_FLAG",
                "domain_name": "rust_core",
                "docstring": "Initialize boolean flag.",
                "code_template": "let {out} = true;",
                "dependencies": ["std::io"],
                "inputs": {},
                "outputs": {"out": {"carrier": "bool", "typestate": "any", "value_class": "data", "resource_mode": "value"}},
                "mutation_type": "pure",
            }
        ],
    }

    ir_payload = {
        "nodes": [{"id": "n1", "kind": "call", "intent": "RUST_INIT_FLAG", "inputs": [], "outputs": ["ok_flag"], "params": {}}],
        "assets": {},
        "deliverable": "ok_flag",
    }

    try:
        pipe = StrataPipeline.from_trees(
            [rust_tree],
            has_llm=True,
            generate_fn=lambda *a, **kw: json.dumps(ir_payload),
            embed_fn=lambda texts, mode="document": [[1.0] for _ in texts],
        )
        res = pipe.run("emit rust code", execute=False)
        assert res.ok
        assert "use std::io;" in res.code
        assert "let ok_flag = true;" in res.code
        assert get_active_language_spec() == prev_spec
    finally:
        set_active_language_spec(prev_spec)

    from bridge_graph import CanonicalPortType
    from schema import CellSchema, PortSchema
    from typed_ir import IRNode, IRParamValue, TypedIR
    from typed_planner import CandidatePlan, PlannedStep

    poset = BoundedPoset()
    assert LiveBridgeGraph([], poset).edges == []
    step_cell = CellSchema(
        cell_id="ECHO_LIT",
        domain_name="test",
        code_template="{out} = {val}",
        inputs={"val": PortSchema(carrier="int", typestate="any", value_class="literal")},
        outputs={"out": PortSchema(carrier="int", typestate="any", value_class="data")},
    )
    plan = CandidatePlan(
        steps=(
            PlannedStep(
                ir_node_id="n1",
                cell=step_cell,
                cell_id="ECHO_LIT",
                is_live_bridge=False,
                input_wiring={},
                output_var="y",
                output_sig=CanonicalPortType(carrier="int"),
                step_cost=0.01,
                output_vars={"out": "y"},
            ),
        ),
        total_cost=0.01,
        posterior_prob=1.0,
        open_resources=(),
    )
    ir_lit = TypedIR(
        source_mode="formal_syntax",
        raw_prompt="y = 42",
        nodes=[
            IRNode(
                id="n1",
                kind="call",
                intent="ECHO_LIT",
                outputs=["y"],
                params={"val": IRParamValue(value=42, provenance="prompt", value_class="literal")},
            )
        ],
    )
    bound = bind_and_emit(plan, ir_lit)
    assert bound.ok and "y = 42" in bound.code
