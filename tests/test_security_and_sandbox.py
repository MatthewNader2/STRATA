"""Deterministic tests for AST SecurityPolicy and GEVRSandbox isolation."""
from __future__ import annotations

import ast

import pytest

from gevr_sandbox import GEVRSandbox
from security_policy import SecurityPolicy, SecurityViolationError
from tests.conftest import MEDIA_FIXTURES_DIR


def test_security_policy_blocks_dangerous_constructs_and_allows_safe_code() -> None:
    policy = SecurityPolicy(allowed_capabilities=set())

    unsafe_snippets = (
        "import subprocess\nx = 1",
        "import shutil\nx = 1",
        "import pickle\nx = 1",
        "x = eval('1 + 1')",
        "exec('x = 1')",
        "from os import system\nx = 1",
        "x = ().__class__.__subclasses__()",
        "import socket\nx = 1",
    )
    for snippet in unsafe_snippets:
        with pytest.raises(SecurityViolationError):
            policy.validate_ast(ast.parse(snippet))

    policy.validate_ast(ast.parse("import math\ny = math.sqrt(9)"))


def test_sandbox_executes_safe_code_captures_egress_and_stages_asset_dirs(sandbox, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    res = sandbox.execute(
        "with open('input.png', 'rb') as f:\n"
        "    nbytes = len(f.read())\n"
        "with open('out.txt', 'w', encoding='utf-8') as out:\n"
        "    out.write(str(nbytes))\n",
        input_assets=["input.png"],
        egress_paths=["out.txt"],
        capture_egress=True,
    )
    assert res["success"] is True, res
    assert res["results"].get("nbytes", 0) > 0
    assert res["egress"].get("out.txt") == "ok"
    assert res["egress_bytes"].get("out.txt", b"").decode("utf-8") == str(res["results"]["nbytes"])
    assert not (tmp_path / "out.txt").exists()


def test_sandbox_skips_missing_and_blocks_traversal_and_network(sandbox) -> None:
    res_miss = sandbox.execute("x = 1", input_assets=["_missing_input_file_xyz.bin"])
    assert not res_miss["success"] and res_miss.get("skipped") is True

    res_trav_in = sandbox.execute("s = open('../etc/passwd').read()", input_assets=["../etc/passwd"])
    assert not res_trav_in["success"]

    res_trav_out = sandbox.execute("x = 1", egress_paths=["../escape.txt"])
    assert not res_trav_out["success"] and res_trav_out["error_kind"] == "security"

    res_etc = sandbox.execute("s = open('/etc/hostname').read()")
    assert not res_etc["success"] and res_etc["error_kind"] == "security"

    res_net = sandbox.execute("import socket\ns = socket.socket()")
    assert not res_net["success"] and res_net["error_kind"] == "security"


def test_sandbox_enforces_timeout_and_runtime_setups() -> None:
    sb = GEVRSandbox(default_timeout=1.5, allowed_capabilities=set(), asset_dirs=[MEDIA_FIXTURES_DIR])

    res_timeout = sb.execute("while True:\n    pass", timeout=0.6)
    assert not res_timeout["success"] and res_timeout["error_kind"] == "timeout"

    res_env = sb.execute(
        "import os\nval = os.environ.get('STRATA_SANDBOX_PROBE', '')\n",
        runtime_setups=[{"env_vars": {"STRATA_SANDBOX_PROBE": "ok_42"}}],
    )
    assert res_env["success"] is True
    assert res_env["results"].get("val") == "ok_42"
