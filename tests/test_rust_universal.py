"""
tests/test_rust_universal.py - End-to-End Non-Python (Rust Core + Crate) Universality Proof

Verifies STRATA's domain-agnostic and target-language-agnostic architecture by loading
ONLY a synthetic `rust_core.json` (`LanguageSpec.name == "rust"`, `ast_backend == "lexical"`)
and an arbitrary Rust crate tree (`kv_crate.json`), completely isolated from `python_core.json`
and all Python trees.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TOOLS_DIR = PROJECT_ROOT / "tools"
for p in (str(PROJECT_ROOT), str(SRC_DIR), str(TOOLS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.language_backend import get_active_language_backend, set_active_language_spec
from src.lattice import LatticeOrchestrator, TypeRegistry
from src.preflight import PreflightLinter
from src.unification import ExecutionContext, UnificationGate
from src.gevr_sandbox import GEVRSandbox
from tools.compile_trees import compile_database
from tools.validate_trees import validate_tree_json


RUST_CORE_TREE: Dict[str, Any] = {
    "domain": "rust_core",
    "version": "1.0.0",
    "is_core": True,
    "target_language": "rust",
    "description": "Core Rust language specification and primitive/collection type lattice.",
    "language_spec": {
        "name": "rust",
        "file_extension": ".rs",
        "ast_backend": "lexical",
        "runtime": "rust",
        "assignment_operator": "=",
        "statement_terminator": ";",
        "line_comment": "//",
        "dependency_prefixes": ["use ", "extern crate "],
        "declaration_prefixes": ["let", "let mut", "const", "static"],
        "function_keyword": "fn",
        "subscript_open": "[",
        "subscript_close": "]",
        "attr_separator": ".",
        "block_keywords": ["fn", "if", "else", "match", "for", "while", "loop", "struct", "impl"],
        "forbidden_constructs": ["unsafe"],
        "forbidden_identifiers": ["transmute"],
        "permitted_builtins": ["println!", "vec!"],
        "output_targets": ["output_var"],
    },
    "top_types": ["any", "generic", "*"],
    "top_states": ["any", "*"],
    "data_bearing_roles": ["receiver", "carrier", "data_input", "source_data"],
    "carrier_roles": {
        "data": ["receiver", "carrier", "data_carrier", "data_input", "source_data"],
        "receiver": ["receiver", "carrier"],
        "key": ["column_key", "key", "field_key"],
        "projection": ["projection", "field_key"],
        "single_projection": ["column", "single", "field"],
        "multi_projection": ["columns", "multi", "fields"],
        "optic": ["getter", "setter", "complement"],
        "clean": ["clean", "cleaner"],
        "transform": ["transform", "transformation"],
        "source": ["source", "ingress", "source_data"],
        "sink": ["sink", "terminal", "export"],
        "destination": ["dest_path", "out_path", "destination"],
        "path": ["source_path", "dest_path", "file_path", "path"],
        "literal": ["literal", "constant"],
    },
    "types": {
        "object": {"parent": None, "kind": "top"},
        "scalar": {"parent": "object", "kind": "scalar", "properties": {"is_atomic": True}},
        "i64": {"parent": "scalar", "kind": "numeric", "default_value": 0, "properties": {"is_atomic": True}},
        "f64": {"parent": "scalar", "kind": "numeric", "default_value": 0.0, "properties": {"is_atomic": True}},
        "bool": {"parent": "scalar", "kind": "boolean", "default_value": "false", "properties": {"is_atomic": True}},
        "String": {"parent": "scalar", "kind": "string", "default_value": "\"\"", "properties": {"is_atomic": True}},
        "PathBuf": {"parent": "String", "kind": "path"},
        "Vec": {"parent": "object", "kind": "collection", "empty_expr": "Vec::new()", "properties": {"is_composite": True}},
        "RecordMap": {"parent": "object", "kind": "composite", "empty_expr": "RecordMap::new()", "properties": {"is_composite": True}},
        "Void": {"parent": "object", "kind": "void"},
        "Never": {"parent": "Void", "kind": "bottom"},
    },
    "typestates": {
        "domain": "rust_core",
        "initial_state": "uninitialized",
        "terminal_states": ["serialized_file"],
        "states": [
            {"name": "uninitialized", "parent_state": None, "properties": {}},
            {"name": "loaded_map", "parent_state": None, "properties": {}},
            {"name": "focused_series", "parent_state": None, "properties": {}},
            {"name": "cleaned_series", "parent_state": "focused_series", "properties": {}},
            {"name": "source_path", "parent_state": None, "properties": {"is_source_identifier": True}},
            {"name": "dest_path", "parent_state": None, "properties": {"is_destination_identifier": True}},
            {"name": "field_key", "parent_state": None, "properties": {"is_projection": True}},
            {"name": "serialized_file", "parent_state": None, "properties": {"is_terminal": True, "is_materialized": True}},
        ],
    },
    "cells": [],
}


KV_CRATE_TREE: Dict[str, Any] = {
    "domain": "kv_crate",
    "version": "1.0.0",
    "target_language": "rust",
    "description": "Arbitrary Rust key-value columnar store crate.",
    "cells": [
        {
            "cell_id": "KV_LOAD_MAP",
            "stage": 1,
            "domain_name": "kv_crate",
            "node_type": "function",
            "node_role": "source",
            "docstring": "Load a RecordMap from an on-disk kv file path.",
            "semantic_tags": ["load", "read", "kv", "map", "file"],
            "dependencies": ["use kv_crate::RecordMap;"],
            "inputs": {
                "source_path": {
                    "type_name": "PathBuf",
                    "state": "source_path",
                    "required": True,
                    "port_role": "source_data",
                    "structure_kind": "path_spec",
                    "rank": 0,
                    "supported_formats": ["kv"],
                    "description": "Input kv file path.",
                }
            },
            "outputs": {
                "output_var": {
                    "type_name": "RecordMap",
                    "state": "loaded_map",
                    "port_role": "data_input",
                    "structure_kind": "composite",
                    "rank": 2,
                    "description": "Loaded RecordMap container.",
                }
            },
            "code_template": "let {output_var} = kv_crate::load_map({source_path});",
        },
        {
            "cell_id": "KV_GET_FIELD",
            "stage": 2,
            "domain_name": "kv_crate",
            "node_type": "method",
            "node_role": "transform",
            "optic_role": "getter",
            "container_port": "map",
            "key_port": "field",
            "focus_port": "output_var",
            "docstring": "Project a named numeric Vec field from a RecordMap.",
            "semantic_tags": ["select", "field", "get", "vec"],
            "dependencies": ["use kv_crate::RecordMap;"],
            "inputs": {
                "map": {
                    "type_name": "RecordMap",
                    "state": "loaded_map",
                    "required": True,
                    "port_role": "data_input",
                    "binds": "data_carrier",
                    "structure_kind": "composite",
                    "rank": 2,
                    "description": "Source RecordMap.",
                },
                "field": {
                    "type_name": "String",
                    "state": "field_key",
                    "required": True,
                    "port_role": "key",
                    "binds": "column_key",
                    "structure_kind": "scalar",
                    "rank": 0,
                    "description": "Field key to project.",
                },
            },
            "outputs": {
                "output_var": {
                    "type_name": "Vec",
                    "state": "focused_series",
                    "port_role": "data_input",
                    "structure_kind": "collection",
                    "rank": 1,
                    "description": "Projected field vector.",
                }
            },
            "code_template": "let {output_var} = {map}.get_vec({field});",
        },
        {
            "cell_id": "KV_TRIM_NAN",
            "stage": 2,
            "domain_name": "kv_crate",
            "node_type": "function",
            "node_role": "clean",
            "docstring": "Trim missing or NaN entries from a field Vec.",
            "semantic_tags": ["trim", "clean", "filter", "nan"],
            "dependencies": ["use kv_crate::clean_vec;"],
            "inputs": {
                "vec_in": {
                    "type_name": "Vec",
                    "state": "focused_series",
                    "required": True,
                    "port_role": "data_input",
                    "structure_kind": "collection",
                    "rank": 1,
                    "description": "Vector to clean.",
                }
            },
            "outputs": {
                "output_var": {
                    "type_name": "Vec",
                    "state": "cleaned_series",
                    "port_role": "data_input",
                    "structure_kind": "collection",
                    "rank": 1,
                    "description": "Cleaned vector.",
                }
            },
            "effects": [
                {
                    "target": "output_var",
                    "property": "has_nans",
                    "operator": "==",
                    "value": False,
                    "effect_kind": "clean",
                }
            ],
            "code_template": "let {output_var} = kv_crate::clean_vec({vec_in});",
        },
        {
            "cell_id": "KV_SET_FIELD",
            "stage": 2,
            "domain_name": "kv_crate",
            "node_type": "method",
            "node_role": "transform",
            "optic_role": "setter",
            "container_port": "map",
            "key_port": "field",
            "focus_port": "values",
            "docstring": "Commit updated Vec values back into RecordMap for a field.",
            "semantic_tags": ["set", "update", "commit", "field", "map"],
            "dependencies": ["use kv_crate::RecordMap;"],
            "inputs": {
                "map": {
                    "type_name": "RecordMap",
                    "state": "loaded_map",
                    "required": True,
                    "port_role": "data_input",
                    "binds": "data_carrier",
                    "structure_kind": "composite",
                    "rank": 2,
                    "description": "Target RecordMap.",
                },
                "field": {
                    "type_name": "String",
                    "state": "field_key",
                    "required": True,
                    "port_role": "key",
                    "binds": "column_key",
                    "structure_kind": "scalar",
                    "rank": 0,
                    "description": "Field key to write.",
                },
                "values": {
                    "type_name": "Vec",
                    "state": "focused_series",
                    "required": True,
                    "port_role": "assigned_value",
                    "binds": "assigned_value",
                    "structure_kind": "collection",
                    "rank": 1,
                    "description": "Updated field vector.",
                },
            },
            "outputs": {
                "output_var": {
                    "type_name": "RecordMap",
                    "state": "loaded_map",
                    "port_role": "data_input",
                    "structure_kind": "composite",
                    "rank": 2,
                    "description": "Updated RecordMap container.",
                }
            },
            "code_template": "let {output_var} = {map}.with_vec({field}, {values});",
        },
        {
            "cell_id": "KV_SAVE_MAP",
            "stage": 3,
            "domain_name": "kv_crate",
            "node_type": "function",
            "node_role": "sink",
            "is_endable": True,
            "docstring": "Persist a RecordMap to disk at dest_path.",
            "semantic_tags": ["save", "write", "export", "kv"],
            "dependencies": ["use kv_crate::write_map;"],
            "inputs": {
                "map": {
                    "type_name": "RecordMap",
                    "state": "loaded_map",
                    "required": True,
                    "port_role": "data_input",
                    "structure_kind": "composite",
                    "rank": 2,
                    "description": "RecordMap to serialize.",
                },
                "dest_path": {
                    "type_name": "PathBuf",
                    "state": "dest_path",
                    "required": True,
                    "port_role": "dest_path",
                    "structure_kind": "path_spec",
                    "rank": 0,
                    "supported_formats": ["kv"],
                    "description": "Destination kv file path.",
                },
            },
            "outputs": {},
            "code_template": "kv_crate::write_map({map}, {dest_path});",
        },
    ],
}


class TestRustUniversalPipeline(unittest.TestCase):
    """Proves end-to-end target-language independence using an isolated Rust tree pair."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="strata_rust_test_"))
        self.core_path = self.temp_dir / "rust_core.json"
        self.crate_path = self.temp_dir / "kv_crate.json"
        self.db_path = self.temp_dir / "rust_lattice.db"

        self.core_path.write_text(json.dumps(RUST_CORE_TREE, indent=2), encoding="utf-8")
        self.crate_path.write_text(json.dumps(KV_CRATE_TREE, indent=2), encoding="utf-8")

        TypeRegistry._instance = None
        LatticeOrchestrator._active_instance = None
        set_active_language_spec(None)

    def tearDown(self) -> None:
        TypeRegistry._instance = None
        LatticeOrchestrator._active_instance = None
        set_active_language_spec(None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_rust_tree_validation_and_compilation(self) -> None:
        """Rust trees pass TreeSchema + validate_trees + SQLite edge synthesis cleanly."""
        for path in (self.core_path, self.crate_path):
            res = validate_tree_json(path)
            self.assertTrue(res.is_valid, f"{path.name} invalid: {res.errors}")

        written = compile_database(output_db=str(self.db_path), trees_dir=self.temp_dir)
        self.assertEqual(written, 5)
        self.assertTrue(self.db_path.exists())

    def test_rust_isolated_load_unify_preflight_and_emit(self) -> None:
        """Isolated load of rust_core + kv_crate synthesizes valid Rust with zero Python leakage."""
        compile_database(output_db=str(self.db_path), trees_dir=self.temp_dir)
        orch = LatticeOrchestrator(trees_directory=str(self.temp_dir), db_path=str(self.db_path))

        reg = TypeRegistry.get_instance()
        spec = reg.get_language_spec()
        self.assertIsNotNone(spec)
        self.assertEqual(getattr(spec, "name", None), "rust")
        backend = get_active_language_backend(reg)
        self.assertEqual(backend.runtime, "rust")
        self.assertFalse(backend.is_python_ast)

        # Ensure python_core types were NOT injected into the isolated Rust registry
        self.assertNotIn("dataframe", reg.get_registered_types())

        pipeline = [
            orch.loaded_cells["KV_LOAD_MAP"],
            orch.loaded_cells["KV_GET_FIELD"],
            orch.loaded_cells["KV_TRIM_NAN"],
            orch.loaded_cells["KV_SET_FIELD"],
            orch.loaded_cells["KV_SAVE_MAP"],
        ]
        ctx = ExecutionContext(
            prompt='Load "in.kv", trim field "price", and save to "out.kv"'
        )
        ctx.parameters["source_path"] = '"in.kv"'
        ctx.parameters["field"] = '"price"'
        ctx.parameters["dest_path"] = '"out.kv"'

        gate = UnificationGate(orchestrator=orch)
        res = gate.unify_pipeline(pipeline, ctx)
        self.assertFalse(res.is_bottom(), f"Rust unification failed: {getattr(res, 'reason', res)}")

        bindings = res.value
        code = gate.emit_code(pipeline, ctx, extract_llm_slots=False)
        self.assertTrue(code.strip(), "Emitted Rust code must be non-empty")

        ok_syn, err_syn = backend.validate_syntax(code)
        self.assertTrue(ok_syn, f"Rust lexical syntax check failed: {err_syn}\n{code}")

        # Must contain Rust declarations and crate calls, never Python-only imports
        self.assertIn("let ", code)
        self.assertIn("kv_crate::load_map", code)
        self.assertIn("kv_crate::clean_vec", code)
        self.assertIn(".with_vec(", code)
        self.assertIn("kv_crate::write_map", code)
        self.assertNotIn("import ", code)
        self.assertNotIn("None", code)

        # Preflight lint must validate Rust code using active LanguageBackend without SyntaxError
        lint_res = PreflightLinter.lint(
            pipeline_bindings=bindings,
            prompt=ctx.prompt,
            code_str=code,
            orchestrator=orch,
        )
        syntax_viols = [v for v in lint_res.structured_violations if v.check_id == "syntax_error"]
        self.assertEqual(syntax_viols, [], f"Preflight flagged syntax_error on Rust code: {syntax_viols}")

        # Sandbox must gracefully skip non-Python runtime rather than crashing Python ast.parse
        sb_res = GEVRSandbox().execute(code)
        self.assertTrue(sb_res.get("skipped"), f"Expected non-Python sandbox skip, got: {sb_res}")


if __name__ == "__main__":
    unittest.main()
