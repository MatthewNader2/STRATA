"""
tests/test_unknown_tree.py - Pure Universal, Domain-Agnostic Engine Verification

Verifies that the STRATA engine can compile, route, unify, and execute arbitrary,
unseen non-tabular domain trees with:
- ZERO tabular bias (no DataFrame, Series, pandas, or column fallbacks)
- ZERO hardcoded stage gates (supports 2-stage and 5-stage topologies)
- ZERO pre-baked edges (dynamic structural edge synthesis from rank, kind, element_type)
- ZERO regex / string pattern heuristics

Test Topologies:
1. 2-Stage Audio Signal Tree:
   AUDIO_BUFFER_SOURCE (AudioBufferSource) -> FFT_SPECTRUM_SINK (FFTSpectrumSink)
   Rank-1 float32 tensor morphism.

2. 5-Stage Network Security Tree:
   PACKET_CAPTURE -> PROTOCOL_DECODER -> PAYLOAD_DECRYPTOR -> THREAT_CLASSIFIER -> ALERT_SINK
   Handle -> Composite Record -> Composite Record -> Composite Record -> Scalar Bool.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

# Ensure project root and src/ are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TOOLS_DIR = PROJECT_ROOT / "tools"
for p in (str(PROJECT_ROOT), str(SRC_DIR), str(TOOLS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.compile_trees import compile_database
from tools.validate_trees import validate_tree_json
from src.lattice import LatticeOrchestrator, Cell, TypeRegistry
from src.unification import UnificationGate, ExecutionContext
from src.gevr_sandbox import GEVRSandbox, PostconditionRegistry


AUDIO_DSP_TREE: Dict[str, Any] = {
    "domain": "audio_dsp",
    "version": "1.0.0",
    "description": "Digital signal processing, time-frequency morphisms, and spectral transformations.",
    "types": [
        {"name": "object", "parent": None},
        {"name": "scalar", "parent": "object"},
        {"name": "string", "parent": "scalar"},
        {"name": "path_spec", "parent": "string"},
        {"name": "audio_path", "parent": "path_spec"},
        {"name": "tensor", "parent": "object"},
        {"name": "audio_samples", "parent": "tensor"},
        {"name": "freq_spectrum", "parent": "tensor"},
    ],
    "typestates": {
        "states": [
            {"name": "uninitialized", "parent_state": None},
            {"name": "raw_signal", "parent_state": None},
            {"name": "transformed_signal", "parent_state": None},
        ]
    },
    "supported_formats": ["wav", "flac", "ogg"],
    "capabilities": {
        "fs.read": {
            "modules": ["os", "pathlib"]
        }
    },
    "cells": [
        {
            "cell_id": "AUDIO_BUFFER_SOURCE",
            "stage": 1,
            "domain_name": "audio_dsp",
            "node_type": "function",
            "node_role": "source",
            "topology_type": "sequential",
            "docstring": "Loads an audio signal file into a 1D floating-point sample tensor.",
            "semantic_tags": ["audio", "load", "wav", "buffer", "samples"],
            "inputs": {
                "audio_file": {
                    "type_name": "audio_path",
                    "state": "uninitialized",
                    "abstract_type": "path_spec",
                    "required": True,
                    "port_role": "source_data",
                    "description": "Path to input audio file.",
                    "structure_kind": "path_spec",
                    "rank": 0,
                    "element_type": "string",
                    "supported_formats": ["wav", "flac", "ogg"],
                }
            },
            "outputs": {
                "audio_buffer": {
                    "type_name": "audio_samples",
                    "state": "raw_signal",
                    "abstract_type": "tensor",
                    "port_role": "data_output",
                    "description": "1D audio sample array.",
                    "structure_kind": "tensor",
                    "rank": 1,
                    "element_type": "float32",
                    "supported_formats": [],
                }
            },
            "code_template": "{audio_buffer} = load_audio_buffer({audio_file})",
            "dependencies": [],
            "primary_in": "audio_file",
            "primary_out": "audio_buffer",
        },
        {
            "cell_id": "FFT_SPECTRUM_SINK",
            "stage": 2,
            "domain_name": "audio_dsp",
            "node_type": "function",
            "node_role": "sink",
            "topology_type": "sequential",
            "docstring": "Computes Discrete Fourier Transform spectrum and terminates audio pipeline.",
            "semantic_tags": ["fft", "spectrum", "fourier", "frequency", "sink"],
            "inputs": {
                "audio_in": {
                    "type_name": "audio_samples",
                    "state": "raw_signal",
                    "abstract_type": "tensor",
                    "required": True,
                    "port_role": "feature_input",
                    "description": "1D audio sample array.",
                    "structure_kind": "tensor",
                    "rank": 1,
                    "element_type": "float32",
                    "supported_formats": [],
                }
            },
            "outputs": {
                "spectrum": {
                    "type_name": "freq_spectrum",
                    "state": "transformed_signal",
                    "abstract_type": "tensor",
                    "port_role": "data_output",
                    "description": "Computed frequency magnitudes.",
                    "structure_kind": "tensor",
                    "rank": 1,
                    "element_type": "float32",
                    "supported_formats": [],
                }
            },
            "code_template": "{spectrum} = compute_fft_spectrum({audio_in})",
            "dependencies": [],
            "primary_in": "audio_in",
            "primary_out": "spectrum",
        },
    ],
}


NET_SEC_TREE: Dict[str, Any] = {
    "domain": "net_sec",
    "version": "1.0.0",
    "description": "Network telemetry capture, packet decoding, decryption, anomaly classification, and alert dispatch.",
    "types": [
        {"name": "object", "parent": None},
        {"name": "handle", "parent": "object"},
        {"name": "scalar", "parent": "object"},
        {"name": "string", "parent": "scalar"},
        {"name": "bool", "parent": "scalar"},
        {"name": "composite", "parent": "object"},
        {"name": "iface_name", "parent": "string"},
        {"name": "raw_packets", "parent": "handle"},
        {"name": "decoded_frames", "parent": "composite"},
        {"name": "decrypted_payloads", "parent": "composite"},
        {"name": "threat_alerts", "parent": "composite"},
        {"name": "alert_status", "parent": "bool"},
    ],
    "typestates": {
        "states": [
            {"name": "uninitialized", "parent_state": None},
            {"name": "capturing", "parent_state": None},
            {"name": "decoded", "parent_state": None},
            {"name": "decrypted", "parent_state": None},
            {"name": "classified", "parent_state": None},
            {"name": "dispatched", "parent_state": None},
        ]
    },
    "supported_formats": ["pcap", "pcapng"],
    "capabilities": {
        "net.packet": {
            "modules": ["socket"]
        }
    },
    "cells": [
        {
            "cell_id": "PACKET_CAPTURE",
            "stage": 1,
            "domain_name": "net_sec",
            "node_type": "function",
            "node_role": "source",
            "topology_type": "sequential",
            "docstring": "Captures raw packet handles from an interface specification.",
            "semantic_tags": ["packet", "capture", "sniff", "network", "raw"],
            "inputs": {
                "interface_spec": {
                    "type_name": "iface_name",
                    "state": "uninitialized",
                    "abstract_type": "scalar",
                    "required": True,
                    "port_role": "source_data",
                    "description": "Network interface identifier.",
                    "structure_kind": "scalar",
                    "rank": 0,
                    "element_type": "string",
                    "supported_formats": [],
                }
            },
            "outputs": {
                "raw_stream": {
                    "type_name": "raw_packets",
                    "state": "capturing",
                    "abstract_type": "handle",
                    "port_role": "data_output",
                    "description": "Active packet stream handle.",
                    "structure_kind": "handle",
                    "rank": None,
                    "element_type": "opaque",
                    "supported_formats": [],
                }
            },
            "code_template": "{raw_stream} = capture_live_packets({interface_spec})",
            "dependencies": [],
            "primary_in": "interface_spec",
            "primary_out": "raw_stream",
        },
        {
            "cell_id": "PROTOCOL_DECODER",
            "stage": 2,
            "domain_name": "net_sec",
            "node_type": "function",
            "node_role": "transformer",
            "topology_type": "sequential",
            "docstring": "Decodes raw packet stream into structured network frames.",
            "semantic_tags": ["decode", "protocol", "dissect", "frames", "headers"],
            "inputs": {
                "packets_in": {
                    "type_name": "raw_packets",
                    "state": "capturing",
                    "abstract_type": "handle",
                    "required": True,
                    "port_role": "receiver",
                    "description": "Raw packet stream handle.",
                    "structure_kind": "handle",
                    "rank": None,
                    "element_type": "opaque",
                    "supported_formats": [],
                }
            },
            "outputs": {
                "frames": {
                    "type_name": "decoded_frames",
                    "state": "decoded",
                    "abstract_type": "composite",
                    "port_role": "data_output",
                    "description": "Structured frame collection.",
                    "structure_kind": "composite",
                    "rank": 1,
                    "element_type": "composite_record",
                    "supported_formats": [],
                }
            },
            "code_template": "{frames} = decode_protocol_frames({packets_in})",
            "dependencies": [],
            "primary_in": "packets_in",
            "primary_out": "frames",
        },
        {
            "cell_id": "PAYLOAD_DECRYPTOR",
            "stage": 3,
            "domain_name": "net_sec",
            "node_type": "function",
            "node_role": "transformer",
            "topology_type": "sequential",
            "docstring": "Decrypts encrypted application payloads across decoded frames.",
            "semantic_tags": ["decrypt", "tls", "payload", "plaintext", "security"],
            "inputs": {
                "enc_frames": {
                    "type_name": "decoded_frames",
                    "state": "decoded",
                    "abstract_type": "composite",
                    "required": True,
                    "port_role": "feature_input",
                    "description": "Encrypted frame records.",
                    "structure_kind": "composite",
                    "rank": 1,
                    "element_type": "composite_record",
                    "supported_formats": [],
                }
            },
            "outputs": {
                "plain_payloads": {
                    "type_name": "decrypted_payloads",
                    "state": "decrypted",
                    "abstract_type": "composite",
                    "port_role": "data_output",
                    "description": "Decrypted payload records.",
                    "structure_kind": "composite",
                    "rank": 1,
                    "element_type": "composite_record",
                    "supported_formats": [],
                }
            },
            "code_template": "{plain_payloads} = decrypt_payload_stream({enc_frames})",
            "dependencies": [],
            "primary_in": "enc_frames",
            "primary_out": "plain_payloads",
        },
        {
            "cell_id": "THREAT_CLASSIFIER",
            "stage": 4,
            "domain_name": "net_sec",
            "node_type": "function",
            "node_role": "transformer",
            "topology_type": "sequential",
            "docstring": "Classifies threat signatures and anomalies over decrypted payloads.",
            "semantic_tags": ["classify", "threat", "anomaly", "detection", "ids"],
            "inputs": {
                "payload_stream": {
                    "type_name": "decrypted_payloads",
                    "state": "decrypted",
                    "abstract_type": "composite",
                    "required": True,
                    "port_role": "feature_input",
                    "description": "Plain payload stream.",
                    "structure_kind": "composite",
                    "rank": 1,
                    "element_type": "composite_record",
                    "supported_formats": [],
                }
            },
            "outputs": {
                "threat_events": {
                    "type_name": "threat_alerts",
                    "state": "classified",
                    "abstract_type": "composite",
                    "port_role": "data_output",
                    "description": "Detected threat alert records.",
                    "structure_kind": "composite",
                    "rank": 1,
                    "element_type": "composite_record",
                    "supported_formats": [],
                }
            },
            "code_template": "{threat_events} = classify_threat_signatures({payload_stream})",
            "dependencies": [],
            "primary_in": "payload_stream",
            "primary_out": "threat_events",
        },
        {
            "cell_id": "ALERT_SINK",
            "stage": 5,
            "domain_name": "net_sec",
            "node_type": "function",
            "node_role": "sink",
            "topology_type": "sequential",
            "docstring": "Dispatches high-priority security alerts to SIEM sink.",
            "semantic_tags": ["alert", "dispatch", "siem", "sink", "notify"],
            "inputs": {
                "events_in": {
                    "type_name": "threat_alerts",
                    "state": "classified",
                    "abstract_type": "composite",
                    "required": True,
                    "port_role": "feature_input",
                    "description": "Threat event records to dispatch.",
                    "structure_kind": "composite",
                    "rank": 1,
                    "element_type": "composite_record",
                    "supported_formats": [],
                }
            },
            "outputs": {
                "status": {
                    "type_name": "alert_status",
                    "state": "dispatched",
                    "abstract_type": "scalar",
                    "port_role": "data_output",
                    "description": "Dispatch success status.",
                    "structure_kind": "scalar",
                    "rank": 0,
                    "element_type": "bool",
                    "supported_formats": [],
                }
            },
            "code_template": "{status} = dispatch_security_alert({events_in})",
            "dependencies": [],
            "primary_in": "events_in",
            "primary_out": "status",
        },
    ],
}


class TestUnknownDomainTrees(unittest.TestCase):
    """Verifies that STRATA handles arbitrary, unseen non-tabular trees without heuristics."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="nstl_unknown_trees_"))
        cls.db_path = cls.temp_dir / "lattice.db"

        # Write out both unseen domain tree JSON files
        cls.audio_json_path = cls.temp_dir / "audio_dsp.json"
        with open(cls.audio_json_path, "w", encoding="utf-8") as f:
            json.dump(AUDIO_DSP_TREE, f, indent=2)

        cls.net_json_path = cls.temp_dir / "net_sec.json"
        with open(cls.net_json_path, "w", encoding="utf-8") as f:
            json.dump(NET_SEC_TREE, f, indent=2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_schema_validation(self):
        """Task 4.1: Validate both novel trees with tools/validate_trees.py."""
        res_audio = validate_tree_json(self.audio_json_path)
        self.assertTrue(res_audio.is_valid, f"audio_dsp validation errors: {res_audio.errors}")
        self.assertEqual(res_audio.structural_coverage, 100.0)
        self.assertEqual(len(res_audio.degradation_warnings), 0, f"Unexpected degradation warnings: {res_audio.degradation_warnings}")

        res_net = validate_tree_json(self.net_json_path)
        self.assertTrue(res_net.is_valid, f"net_sec validation errors: {res_net.errors}")
        self.assertEqual(res_net.structural_coverage, 100.0)
        self.assertEqual(len(res_net.degradation_warnings), 0, f"Unexpected degradation warnings: {res_net.degradation_warnings}")

    def test_02_dynamic_edge_compilation(self):
        """Task 4.1: Compile both trees and verify dynamic edge synthesis without pre-baked edges."""
        written = compile_database(output_db=str(self.db_path), trees_dir=self.temp_dir)
        self.assertEqual(written, 7, f"Expected 7 compiled cells, got {written}")
        self.assertTrue(self.db_path.exists())

        conn = sqlite3.connect(str(self.db_path))
        cur = conn.cursor()

        # Check nodes count
        cur.execute("SELECT cell_id, domain_name, stage, configuration_schema FROM nodes")
        rows = cur.fetchall()
        self.assertEqual(len(rows), 7)

        edges_map = {}
        for cid, dom, stage, cfg_json in rows:
            cfg = json.loads(cfg_json) if cfg_json else {}
            edges_map[cid] = cfg.get("edges", [])

        conn.close()

        # Audio DSP dynamic edge synthesis:
        # AUDIO_BUFFER_SOURCE (1D tensor float32) -> FFT_SPECTRUM_SINK (1D tensor float32)
        audio_edges = edges_map.get("AUDIO_BUFFER_SOURCE", [])
        self.assertIn("FFT_SPECTRUM_SINK", audio_edges, "Dynamic edge AUDIO_BUFFER_SOURCE -> FFT_SPECTRUM_SINK was not synthesized!")

        # Network Security dynamic edge synthesis:
        # PACKET_CAPTURE (handle opaque) -> PROTOCOL_DECODER (handle opaque)
        pcap_edges = edges_map.get("PACKET_CAPTURE", [])
        self.assertIn("PROTOCOL_DECODER", pcap_edges, "Dynamic edge PACKET_CAPTURE -> PROTOCOL_DECODER was not synthesized!")

        # PROTOCOL_DECODER (composite record) -> PAYLOAD_DECRYPTOR (composite record)
        dec_edges = edges_map.get("PROTOCOL_DECODER", [])
        self.assertIn("PAYLOAD_DECRYPTOR", dec_edges, "Dynamic edge PROTOCOL_DECODER -> PAYLOAD_DECRYPTOR was not synthesized!")

        # PAYLOAD_DECRYPTOR -> THREAT_CLASSIFIER
        crypto_edges = edges_map.get("PAYLOAD_DECRYPTOR", [])
        self.assertIn("THREAT_CLASSIFIER", crypto_edges, "Dynamic edge PAYLOAD_DECRYPTOR -> THREAT_CLASSIFIER was not synthesized!")

        # THREAT_CLASSIFIER -> ALERT_SINK
        threat_edges = edges_map.get("THREAT_CLASSIFIER", [])
        self.assertIn("ALERT_SINK", threat_edges, "Dynamic edge THREAT_CLASSIFIER -> ALERT_SINK was not synthesized!")

    def test_03_orchestrator_structural_roles(self):
        """Task 4.1: Verify LatticeOrchestrator loads arbitrary trees and derives roles structurally."""
        orch = LatticeOrchestrator(trees_directory=str(self.temp_dir), db_path=str(self.db_path))
        self.assertEqual(len(orch.cells), 7)

        # Check source and sink structural roles
        audio_src = orch.loaded_cells["AUDIO_BUFFER_SOURCE"]
        fft_sink = orch.loaded_cells["FFT_SPECTRUM_SINK"]
        self.assertEqual(audio_src.node_role, "source")
        self.assertEqual(fft_sink.node_role, "sink")

        pcap = orch.loaded_cells["PACKET_CAPTURE"]
        alert = orch.loaded_cells["ALERT_SINK"]
        self.assertEqual(pcap.node_role, "source")
        self.assertEqual(alert.node_role, "sink")

    def test_04_audio_pipeline_unification_and_emission(self):
        """Task 4.1: Unify and emit code for 2-stage audio pipeline with zero tabular assumptions."""
        orch = LatticeOrchestrator(trees_directory=str(self.temp_dir), db_path=str(self.db_path))
        gate = UnificationGate(orchestrator=orch)

        audio_src = orch.loaded_cells["AUDIO_BUFFER_SOURCE"]
        fft_sink = orch.loaded_cells["FFT_SPECTRUM_SINK"]
        pipeline = [audio_src, fft_sink]

        ctx = ExecutionContext(prompt="Compute FFT spectrum from audio file test.wav")
        ctx.parameters["audio_file"] = "'test.wav'"

        code = gate.emit_code(pipeline, ctx, extract_llm_slots=False)
        self.assertTrue(len(code.strip()) > 0, "Emitted audio code should not be empty")

        # Must parse as clean Python AST
        tree = ast.parse(code)
        self.assertEqual(len(tree.body), 2, "Expected exactly 2 statement nodes for 2-stage audio pipeline")

        # Verify variable chaining:
        # stmt 0: var0 = load_audio_buffer('test.wav')
        # stmt 1: var1 = compute_fft_spectrum(var0)
        stmt0, stmt1 = tree.body[0], tree.body[1]
        self.assertIsInstance(stmt0, ast.Assign)
        self.assertIsInstance(stmt1, ast.Assign)

        out_var = stmt0.targets[0].id
        in_var = stmt1.value.args[0].id
        self.assertEqual(out_var, in_var, f"Audio pipeline broke dataflow chaining: {out_var} != {in_var}")

        # Assert zero tabular bias
        lowered_code = code.lower()
        self.assertNotIn("dataframe", lowered_code)
        self.assertNotIn("pandas", lowered_code)
        self.assertNotIn("column", lowered_code)
        self.assertNotIn("fillna", lowered_code)

    def test_05_netsec_pipeline_unification_and_emission(self):
        """Task 4.1: Unify and emit code for 5-stage network security pipeline with zero stage gates."""
        orch = LatticeOrchestrator(trees_directory=str(self.temp_dir), db_path=str(self.db_path))
        gate = UnificationGate(orchestrator=orch)

        pipeline = [
            orch.loaded_cells["PACKET_CAPTURE"],
            orch.loaded_cells["PROTOCOL_DECODER"],
            orch.loaded_cells["PAYLOAD_DECRYPTOR"],
            orch.loaded_cells["THREAT_CLASSIFIER"],
            orch.loaded_cells["ALERT_SINK"],
        ]

        ctx = ExecutionContext(prompt="Capture packets on eth0, decode, decrypt, classify threat, and dispatch alert")
        ctx.parameters["interface_spec"] = "'eth0'"

        code = gate.emit_code(pipeline, ctx, extract_llm_slots=False)
        self.assertTrue(len(code.strip()) > 0, "Emitted net_sec code should not be empty")

        # Must parse as clean Python AST
        tree = ast.parse(code)
        self.assertEqual(len(tree.body), 5, f"Expected exactly 5 statements for 5-stage pipeline, got {len(tree.body)}")

        # Verify sequential dataflow chaining across all 5 stages:
        # v0 = capture(eth0)
        # v1 = decode(v0)
        # v2 = decrypt(v1)
        # v3 = classify(v2)
        # v4 = alert(v3)
        assigned_vars = []
        for i, stmt in enumerate(tree.body):
            self.assertIsInstance(stmt, ast.Assign, f"Statement {i} is not an assignment: {ast.dump(stmt)}")
            assigned_vars.append(stmt.targets[0].id)
            if i > 0:
                consumed_var = stmt.value.args[0].id
                prev_var = assigned_vars[i - 1]
                self.assertEqual(
                    consumed_var,
                    prev_var,
                    f"Stage {i} broken dataflow: consumed {consumed_var} but previous stage assigned {prev_var}",
                )

        # Assert zero tabular bias
        lowered_code = code.lower()
        self.assertNotIn("dataframe", lowered_code)
        self.assertNotIn("pandas", lowered_code)
        self.assertNotIn("column", lowered_code)
        self.assertNotIn("fillna", lowered_code)

    def test_06_sandbox_execution_and_postconditions(self):
        """Task 4.1 & 4.3: Execute emitted pipelines in GEVRSandbox and evaluate generic postconditions."""
        orch = LatticeOrchestrator(trees_directory=str(self.temp_dir), db_path=str(self.db_path))
        gate = UnificationGate(orchestrator=orch)

        # 1. Audio execution
        pipeline_audio = [orch.loaded_cells["AUDIO_BUFFER_SOURCE"], orch.loaded_cells["FFT_SPECTRUM_SINK"]]
        ctx_audio = ExecutionContext(prompt="Process audio")
        ctx_audio.parameters["audio_file"] = "'sample.wav'"
        audio_code = gate.emit_code(pipeline_audio, ctx_audio, extract_llm_slots=False)

        # Mock runtime definitions for the domain operations
        mock_audio_runtime = (
            "def load_audio_buffer(path):\n"
            "    class Signal:\n"
            "        ndim = 1\n"
            "        sample_rate = 44100\n"
            "    return Signal()\n\n"
            "def compute_fft_spectrum(signal):\n"
            "    class Spectrum:\n"
            "        ndim = 1\n"
            "        peak_freq = 440.0\n"
            "    return Spectrum()\n\n"
        )
        full_audio_code = mock_audio_runtime + audio_code

        # Run in sandbox with local execution
        sandbox = GEVRSandbox(timeout=5.0)
        res_audio = sandbox.execute(full_audio_code)
        self.assertEqual(res_audio.exit_code, 0, f"Audio sandbox execution failed: {res_audio.stderr}")

        # Postcondition evaluation on the custom non-tabular domain object
        reg = PostconditionRegistry()
        spectrum_obj = res_audio.results[list(res_audio.results.keys())[-1]]
        self.assertTrue(reg.evaluate("ndim", spectrum_obj, 1))
        self.assertTrue(reg.evaluate("peak_freq", spectrum_obj, 440.0))

        # 2. NetSec execution
        pipeline_net = [
            orch.loaded_cells["PACKET_CAPTURE"],
            orch.loaded_cells["PROTOCOL_DECODER"],
            orch.loaded_cells["PAYLOAD_DECRYPTOR"],
            orch.loaded_cells["THREAT_CLASSIFIER"],
            orch.loaded_cells["ALERT_SINK"],
        ]
        ctx_net = ExecutionContext(prompt="Process netsec")
        ctx_net.parameters["interface_spec"] = "'eth0'"
        net_code = gate.emit_code(pipeline_net, ctx_net, extract_llm_slots=False)

        mock_net_runtime = (
            "def capture_live_packets(iface): return {'pcap_handle': iface}\n"
            "def decode_protocol_frames(pkts): return [{'proto': 'TCP', 'payload': b'\\x01\\x02'}]\n"
            "def decrypt_payload_stream(frames): return [{'proto': 'TCP', 'plain': 'GET /login'}]\n"
            "def classify_threat_signatures(stream): return [{'severity': 'HIGH', 'rule': 'SQLi'}]\n"
            "def dispatch_security_alert(threats):\n"
            "    class AlertResult:\n"
            "        dispatched = True\n"
            "        threat_count = len(threats)\n"
            "    return AlertResult()\n\n"
        )
        full_net_code = mock_net_runtime + net_code
        res_net = sandbox.execute(full_net_code)
        self.assertEqual(res_net.exit_code, 0, f"NetSec sandbox execution failed: {res_net.stderr}")

        alert_obj = res_net.results[list(res_net.results.keys())[-1]]
        self.assertTrue(reg.evaluate("dispatched", alert_obj, True))
        self.assertTrue(reg.evaluate("threat_count", alert_obj, 1))


if __name__ == "__main__":
    unittest.main()
