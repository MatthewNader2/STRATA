"""
src/gevr_sandbox.py
Hardened, isolated, domain-agnostic sandboxed execution environment with
AST security inspection, POSIX resource limits (RLIMIT_AS / RLIMIT_CPU),
ephemeral filesystem confinement, network socket blocking, and closed-loop
CEGIS self-repair.
"""

from __future__ import annotations

import abc
import ast
import builtins
import contextlib
import io as _io
import logging
import math
import multiprocessing
import os
import pickle as _pickle
import queue as _pyqueue
import re
import shutil
import signal
import socket as _real_socket
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

try:
    import resource as _resource
except ImportError:  # pragma: no cover (non-POSIX fallback)
    _resource = None  # type: ignore[assignment]

try:
    from .config import settings
except (ImportError, ValueError):
    try:
        from config import settings
    except Exception:
        settings = None

try:
    from .utils import extract_template_placeholders
except (ImportError, ValueError):
    try:
        from utils import extract_template_placeholders
    except Exception:
        def extract_template_placeholders(template: str) -> List[str]:
            return []

logger = logging.getLogger(__name__)


# ============================================================================
# Security Exceptions & AST Validator
# ============================================================================

class SecurityViolationError(PermissionError):
    """Raised when executed code violates sandbox security constraints."""
    pass


class SandboxTimeoutError(TimeoutError):
    """Raised when execution exceeds the allocated wall-clock time limit."""
    pass


class _SecurityASTVisitor(ast.NodeVisitor):
    """
    Static analysis pass over synthesized code before execution.
    Detects dunder traversal exploits, bytecode manipulation, and forbidden builtins.
    """

    FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset({
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__code__",
        "__closure__",
        "__builtins__",
        "__import__",
        "__loader__",
        "__spec__",
        "gi_frame",
        "f_locals",
        "f_globals",
        "cr_frame",
    })

    FORBIDDEN_CALLS: frozenset[str] = frozenset({
        "eval",
        "exec",
        "compile",
        "breakpoint",
        "input",
    })

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in self.FORBIDDEN_ATTRIBUTES:
            raise SecurityViolationError(
                f"Access to restricted meta-attribute '{node.attr}' at line {node.lineno} is forbidden."
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in self.FORBIDDEN_CALLS:
            raise SecurityViolationError(
                f"Direct access to forbidden built-in '{node.id}' at line {node.lineno} is blocked."
            )
        self.generic_visit(node)

    @staticmethod
    def _evaluate_constant_str(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = _SecurityASTVisitor._evaluate_constant_str(node.left)
            right = _SecurityASTVisitor._evaluate_constant_str(node.right)
            if left is not None and right is not None:
                return left + right
        if isinstance(node, ast.JoinedStr):
            parts = []
            for val in node.values:
                s = _SecurityASTVisitor._evaluate_constant_str(val)
                if s is None:
                    return None
                parts.append(s)
            return "".join(parts)
        if isinstance(node, ast.FormattedValue):
            return _SecurityASTVisitor._evaluate_constant_str(node.value)
        return None

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Add):
            concat_str = self._evaluate_constant_str(node)
            if concat_str is not None:
                if concat_str in self.FORBIDDEN_ATTRIBUTES or (concat_str.startswith("__") and concat_str.endswith("__")):
                    raise SecurityViolationError(
                        f"Dynamic concatenation producing forbidden dunder attribute '{concat_str}' at line {getattr(node, 'lineno', 0)} is blocked."
                    )
            for side in (node.left, node.right):
                side_str = self._evaluate_constant_str(side)
                if side_str is not None and "__" in side_str:
                    raise SecurityViolationError(
                        f"Dynamic dunder string concatenation involving '{side_str}' at line {getattr(node, 'lineno', 0)} is blocked."
                    )
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        eval_str = self._evaluate_constant_str(node)
        if eval_str is not None and (eval_str in self.FORBIDDEN_ATTRIBUTES or (eval_str.startswith("__") and eval_str.endswith("__"))):
            raise SecurityViolationError(
                f"Dynamic string formatting producing forbidden dunder attribute '{eval_str}' at line {getattr(node, 'lineno', 0)} is blocked."
            )
        for val in node.values:
            s = self._evaluate_constant_str(val)
            if s is not None and "__" in s:
                raise SecurityViolationError(
                    f"Dynamic dunder string formatting involving '{s}' at line {getattr(node, 'lineno', 0)} is blocked."
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in self.FORBIDDEN_CALLS:
                raise SecurityViolationError(
                    f"Direct access to forbidden built-in '{node.func.id}' at line {node.lineno} is blocked."
                )
            if node.func.id in ("getattr", "hasattr", "setattr", "delattr") and len(node.args) >= 2:
                arg_str = self._evaluate_constant_str(node.args[1])
                if arg_str and (arg_str in self.FORBIDDEN_ATTRIBUTES or (arg_str.startswith("__") and arg_str.endswith("__"))):
                    raise SecurityViolationError(
                        f"Dynamic access to forbidden dunder attribute '{arg_str}' via {node.func.id} at line {node.lineno} is blocked."
                    )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        slice_str = self._evaluate_constant_str(node.slice)
        if slice_str and (slice_str in self.FORBIDDEN_ATTRIBUTES or (slice_str.startswith("__") and slice_str.endswith("__"))):
            raise SecurityViolationError(
                f"Subscription access to forbidden meta-attribute '{slice_str}' at line {getattr(node, 'lineno', 0)} is blocked."
            )
        self.generic_visit(node)


# ============================================================================
# Pluggable Postconditions & State Protocols (Domain-Agnostic)
# ============================================================================

class StateProtocolRegistry:
    """Registry determining object readiness or fittedness via pluggable predicates."""

    def __init__(self) -> None:
        self._predicates: List[Callable[[Any], bool]] = []
        self._register_default_predicates()

    def register(self, predicate: Callable[[Any], bool]) -> None:
        self._predicates.append(predicate)

    def is_fitted_or_ready(self, target: Any) -> bool:
        for pred in self._predicates:
            try:
                if pred(target):
                    return True
            except Exception as exc:
                logger.debug("[SANDBOX] State predicate check skipped: %s", exc)
                continue
        return False

    def _register_default_predicates(self) -> None:
        def _callable_check(obj: Any) -> bool:
            for attr in ("is_fitted", "is_ready", "is_converged"):
                method = getattr(obj, attr, None)
                if callable(method):
                    return bool(method())
            return False

        def _attribute_convention_check(obj: Any) -> bool:
            if hasattr(obj, "__dict__"):
                public_state = [k for k in obj.__dict__.keys() if not k.startswith("_")]
                return len(public_state) > 0
            return True

        self.register(_callable_check)
        self.register(_attribute_convention_check)


class PostconditionRegistry:
    """Domain-agnostic registry for evaluating postcondition constraints."""

    def __init__(self) -> None:
        self._evaluators: Dict[str, Callable[[Any, Any], bool]] = {}
        self._register_default_evaluators()

    def register(self, property_name: str, evaluator: Callable[[Any, Any], bool]) -> None:
        self._evaluators[property_name] = evaluator

    def evaluate(self, property_name: str, target_val: Any, expected_val: Any) -> bool:
        evaluator = self._evaluators.get(property_name)
        if not evaluator:
            if hasattr(target_val, property_name):
                try:
                    attr = getattr(target_val, property_name)
                    val = attr() if callable(attr) else attr
                    return bool(val == expected_val)
                except Exception:
                    pass
            elif isinstance(target_val, dict) and property_name in target_val:
                return bool(target_val[property_name] == expected_val)
            logger.warning("No postcondition evaluator registered for '%s'. Falling back to equality.", property_name)
            return bool(target_val == expected_val)
        try:
            return evaluator(target_val, expected_val)
        except Exception as e:
            logger.error("Error evaluating postcondition '%s': %s", property_name, e)
            return False

    def _register_default_evaluators(self) -> None:
        def _ndim_evaluator(val: Any, expected: Any) -> bool:
            if hasattr(val, "ndim"):
                return bool(val.ndim == expected)
            if hasattr(val, "shape"):
                return bool(len(val.shape) == expected)

            def _get_depth(obj: Any) -> int:
                if isinstance(obj, (list, tuple)) and obj:
                    return 1 + _get_depth(obj[0])
                return 0

            return _get_depth(val) == expected

        def _has_nans_evaluator(val: Any, expected: Any) -> bool:
            for method_name in ("isna", "isnull"):
                method = getattr(val, method_name, None)
                if callable(method):
                    res = method()
                    has_nan = bool(getattr(res, "any", lambda: res)())
                    return has_nan == expected
            try:
                import math
                if isinstance(val, float) and math.isnan(val):
                    return True == expected
                if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
                    has_nan = any(isinstance(x, float) and math.isnan(x) for x in val)
                    return has_nan == expected
            except Exception as exc:
                logger.debug("[SANDBOX] NaN evaluation fallback: %s", exc)
            return False == expected

        def _dedup_evaluator(val: Any, expected: Any) -> bool:
            if hasattr(val, "duplicated"):
                dups = val.duplicated()
                is_unique = not bool(getattr(dups, "any", lambda: dups)())
                return is_unique == expected
            try:
                seq = list(val)
                return (len(seq) == len(set(seq))) == expected
            except TypeError:
                return True == expected

        self.register("ndim", _ndim_evaluator)
        self.register("has_nans", _has_nans_evaluator)
        self.register("is_deduped", _dedup_evaluator)


# ============================================================================
# Pluggable Terminal Intent Verifiers (Domain-Agnostic)
# ============================================================================

@dataclass
class IntentVerificationResult:
    valid: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class BaseIntentVerifier(abc.ABC):
    @abc.abstractmethod
    def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
        pass


class TerminalIntentRegistry:
    """Decoupled registry of verifiers validating terminal computation intent."""

    def __init__(self) -> None:
        self._verifiers: Dict[str, BaseIntentVerifier] = {}
        self._register_default_verifiers()

    def register(self, intent_type: str, verifier: BaseIntentVerifier) -> None:
        self._verifiers[intent_type] = verifier

    def verify(self, intent_type: str, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
        verifier = self._verifiers.get(intent_type)
        if not verifier:
            is_valid = output is not None
            return IntentVerificationResult(
                valid=is_valid,
                reason="Default fallback: output is non-None" if is_valid else "Output is None",
            )
        return verifier.verify(output, context)

    def _register_default_verifiers(self) -> None:
        class PartitionIntentVerifier(BaseIntentVerifier):
            def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
                if not isinstance(output, (tuple, list)):
                    return IntentVerificationResult(valid=False, reason="Split intent must produce a collection")
                expected_min = context.get("min_partitions", 2)
                if len(output) < expected_min:
                    return IntentVerificationResult(
                        valid=False,
                        reason=f"Expected at least {expected_min} partitions, received {len(output)}",
                    )
                return IntentVerificationResult(valid=True, reason="Valid partition structure")

        class StructuredDataVerifier(BaseIntentVerifier):
            def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
                if output is None:
                    return IntentVerificationResult(valid=False, reason="Target structured output is None")
                has_len = hasattr(output, "__len__")
                has_shape = hasattr(output, "shape")
                if not (has_len or has_shape):
                    return IntentVerificationResult(valid=False, reason="Output lacks dimension or sequence protocol")
                return IntentVerificationResult(valid=True, reason="Structured carrier contract verified")

        class VisualArtifactVerifier(BaseIntentVerifier):
            def verify(self, output: Any, context: Dict[str, Any]) -> IntentVerificationResult:
                if output is None:
                    return IntentVerificationResult(valid=False, reason="Artifact output is None")
                has_render_hook = any(
                    hasattr(output, attr) for attr in ("canvas", "savefig", "render", "show", "to_image")
                )
                if has_render_hook or type(output).__name__ in ("Figure", "Image", "Plot"):
                    return IntentVerificationResult(valid=True, reason="Visual artifact verified")
                return IntentVerificationResult(valid=False, reason="Object lacks visual artifact rendering protocol")

        self.register("model_fit_split", PartitionIntentVerifier())
        self.register("structured_carrier_egress", StructuredDataVerifier())
        self.register("tabular_egress", StructuredDataVerifier())
        self.register("image_annotation_egress", StructuredDataVerifier())
        self.register("visualization_egress", VisualArtifactVerifier())


# ============================================================================
# Hardened Worker Execution Layer (Track 5: RLIMIT, Network & FS Confinement)
# ============================================================================

_BLOCKED_MODULES = frozenset({
    "subprocess",
    "shutil",
    "socket",
    "ssl",
    "http",
    "urllib",
    "requests",
    "ftplib",
    "telnetlib",
    "smtplib",
    "xmlrpc",
    "ctypes",
    "pty",
    "commands",
    "multiprocessing",
})


def _enforce_posix_resource_limits(max_memory_mb: int = 16384, max_cpu_seconds: int = 10) -> None:
    """
    Track 5.1: Enforces strict OS-level memory (RLIMIT_AS, RLIMIT_DATA) and CPU time
    (RLIMIT_CPU) bounds on the sandbox worker process.
    """
    if _resource is None:
        return
    try:
        env_mem = os.environ.get("NSTL_SANDBOX_MAX_MEMORY_MB")
        if env_mem:
            try:
                max_memory_mb = max(int(env_mem), max_memory_mb)
            except ValueError:
                pass
        mem_bytes = int(max_memory_mb) * 1024 * 1024
        as_bytes = max(32768, int(max_memory_mb)) * 1024 * 1024
        if hasattr(_resource, "RLIMIT_AS"):
            _, hard_as = _resource.getrlimit(_resource.RLIMIT_AS)
            target_as = as_bytes if hard_as == _resource.RLIM_INFINITY else min(as_bytes, hard_as)
            _resource.setrlimit(_resource.RLIMIT_AS, (target_as, target_as))
        if hasattr(_resource, "RLIMIT_DATA"):
            _, hard_data = _resource.getrlimit(_resource.RLIMIT_DATA)
            target_data = mem_bytes if hard_data == _resource.RLIM_INFINITY else min(mem_bytes, hard_data)
            _resource.setrlimit(_resource.RLIMIT_DATA, (target_data, target_data))
    except Exception as exc:
        logger.debug("[SANDBOX] RLIMIT_AS/DATA setup skipped: %s", exc)

    try:
        cpu_s = max(1, int(max_cpu_seconds))
        if hasattr(_resource, "RLIMIT_CPU"):
            _, hard_cpu = _resource.getrlimit(_resource.RLIMIT_CPU)
            target_cpu = cpu_s if hard_cpu == _resource.RLIM_INFINITY else min(cpu_s, hard_cpu)
            _resource.setrlimit(_resource.RLIMIT_CPU, (target_cpu, target_cpu))
    except Exception as exc:
        logger.debug("[SANDBOX] RLIMIT_CPU setup skipped: %s", exc)


def _disable_network_sockets(allow_network: bool = False) -> None:
    """
    Track 5.2: Disables all outbound/inbound network socket creation in the
    sandbox worker process unless the 'net' capability is explicitly allowed.
    """
    if allow_network:
        return

    def _blocked_socket(*args: Any, **kwargs: Any) -> Any:
        raise SecurityViolationError("Network socket creation is prohibited inside the GEVR sandbox without --allow-capabilities net.")

    _real_socket.socket = _blocked_socket  # type: ignore[misc,assignment]
    _real_socket.create_connection = _blocked_socket  # type: ignore[assignment]


def _create_confined_open(
    allowed_read_roots: Set[str],
    allowed_write_roots: Set[str],
    allowed_write_files: Set[str],
) -> Callable[..., Any]:
    """
    Track 5.2: Wraps `builtins.open` so untrusted code can:
      - Read only from the project working tree, ephemeral sandbox dir, or standard Python library paths.
      - Write ONLY inside the ephemeral sandbox directory or explicitly declared `egress_paths`.
    """
    real_open = builtins.open

    def _is_under(path: Path, roots: Set[str]) -> bool:
        p_str = str(path)
        for r in roots:
            if p_str == r or p_str.startswith(r.rstrip(os.sep) + os.sep):
                return True
        return False

    def _confined_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, int):
            raise SecurityViolationError("Opening raw file descriptors is prohibited in the sandbox.")
        raw_str = os.fspath(file)
        if isinstance(raw_str, bytes):
            raw_str = raw_str.decode("utf-8", errors="replace")

        resolved = Path(raw_str).resolve()
        is_write = any(m in mode for m in ("w", "a", "x", "+"))

        if is_write:
            if str(resolved) not in allowed_write_files and not _is_under(resolved, allowed_write_roots):
                raise SecurityViolationError(
                    f"Sandbox write access denied for path '{raw_str}'. "
                    f"Writes are confined to ephemeral sandbox storage and declared egress paths."
                )
        else:
            if not _is_under(resolved, allowed_read_roots) and str(resolved) not in allowed_write_files:
                raise SecurityViolationError(
                    f"Sandbox read access denied for arbitrary filesystem path '{raw_str}'."
                )

        return real_open(file, mode, *args, **kwargs)

    return _confined_open


def _create_sanitized_os():
    """Wrap standard `os` to disable shell execution and destructive file operations."""
    import os as _real_os

    class _SanitizedOS:
        def __getattr__(self, name: str) -> Any:
            blocked_ops = {
                "system", "popen", "spawn", "spawnl", "spawnle", "spawnlp", "spawnlpe",
                "spawnv", "spawnve", "spawnvp", "spawnvpe", "execv", "execve", "execvp",
                "execvpe", "execl", "execle", "execlp", "execlpe", "remove", "unlink",
                "rmdir", "removedirs", "chmod", "chown", "kill", "killpg", "fork", "forkpty",
            }
            if name in blocked_ops:
                raise SecurityViolationError(f"Operating system operation 'os.{name}' is prohibited in sandbox.")
            return getattr(_real_os, name)

    return _SanitizedOS()


def _create_restricted_import(sec_policy: Optional[Any] = None):
    def _restricted_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
        """Guarded import callback preventing module escape."""
        root_module = name.split(".")[0]
        if sec_policy is not None:
            allowed, reason = sec_policy.is_module_allowed(name)
            if not allowed:
                raise SecurityViolationError(f"Import of module '{name}' is blocked: {reason}")
        elif root_module in _BLOCKED_MODULES:
            raise SecurityViolationError(f"Import of restricted module '{root_module}' is blocked.")

        if root_module == "os":
            return _create_sanitized_os()

        real_import = builtins.__import__
        return real_import(name, globals, locals, fromlist, level)
    return _restricted_import


class ASTLoopBounder(ast.NodeTransformer):
    """
    Bounds unbounded while loops to a maximum number of iterations during smoke-test verification.
    """
    def __init__(self, max_iters: int = 5):
        super().__init__()
        self.max_iters = max_iters
        self._loop_count = 0

    def visit_While(self, node: ast.While) -> Any:
        self.generic_visit(node)
        self._loop_count += 1
        counter_name = f"_nstl_iter_{self._loop_count}"
        inc_stmt = ast.AugAssign(
            target=ast.Name(id=counter_name, ctx=ast.Store()),
            op=ast.Add(),
            value=ast.Constant(value=1),
        )
        check_stmt = ast.If(
            test=ast.Compare(
                left=ast.Name(id=counter_name, ctx=ast.Load()),
                ops=[ast.Gt()],
                comparators=[ast.Constant(value=self.max_iters)],
            ),
            body=[ast.Break()],
            orelse=[],
        )
        node.body = [inc_stmt, check_stmt] + node.body
        init_stmt = ast.Assign(
            targets=[ast.Name(id=counter_name, ctx=ast.Store())],
            value=ast.Constant(value=0),
        )
        return [init_stmt, node]


def _sandbox_worker_exec(
    code_str: str,
    context: Dict[str, Any],
    runtime_aliases: Dict[str, str],
    result_conn: Any,
    smoke_test: bool = False,
    egress_paths: Optional[List[str]] = None,
    max_memory_mb: int = 16384,
    max_cpu_seconds: int = 10,
    allowed_capabilities: Optional[List[str]] = None,
) -> None:
    """
    Subprocess worker executing sandboxed code in strict process, resource,
    network, and filesystem isolation with capability-based security.
    """
    allowed_caps_set = set(allowed_capabilities or [])
    if "NSTL_ALLOWED_CAPABILITIES" in os.environ:
        allowed_caps_set.update(c.strip() for c in os.environ["NSTL_ALLOWED_CAPABILITIES"].split(",") if c.strip())

    try:
        from security_policy import SecurityPolicy
        sec_policy = SecurityPolicy(allowed_capabilities=allowed_caps_set)
    except Exception:
        sec_policy = None

    # Step 0: Configure headless environment for GUI, Audio, and Plotting
    try:
        from lattice import TypeRegistry
        env_specs = TypeRegistry.get_instance().get_verification_semantics("headless_environment")
        for spec in env_specs:
            if "=" in spec:
                k, v = spec.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    except Exception as exc:
        logger.debug("[SANDBOX] Headless env setup fallback: %s", exc)

    captured_stdout = _io.StringIO()
    captured_stderr = _io.StringIO()
    def _ipc_safe(value: Any) -> Any:
        try:
            _pickle.dumps(value)
            return value
        except Exception:
            return f"<unpicklable:{type(value).__name__}>"

    def _send_payload(payload_dict: Dict[str, Any]) -> None:
        try:
            if hasattr(result_conn, "send"):
                result_conn.send(payload_dict)
            elif hasattr(result_conn, "put"):
                result_conn.put(payload_dict)
        except Exception:
            try:
                safe_payload = {
                    "success": payload_dict.get("success", False),
                    "returncode": payload_dict.get("returncode", 1),
                    "stdout": str(payload_dict.get("stdout", "")),
                    "stderr": str(payload_dict.get("stderr", "")),
                    "error": str(payload_dict.get("error", "")),
                    "results": {},
                    "egress": payload_dict.get("egress", {}),
                }
                if hasattr(result_conn, "send"):
                    result_conn.send(safe_payload)
                elif hasattr(result_conn, "put"):
                    result_conn.put(safe_payload)
            except Exception:
                pass
        finally:
            try:
                if hasattr(result_conn, "close"):
                    result_conn.close()
            except Exception:
                pass

    orig_cwd = os.getcwd()

    with tempfile.TemporaryDirectory(prefix="nstl_sandbox_") as ephemeral_dir:
        try:
            # Step 1: Pre-execution AST Analysis
            parsed_ast = ast.parse(code_str)
            if sec_policy is not None:
                sec_policy.validate_ast(parsed_ast)
            else:
                _SecurityASTVisitor().visit(parsed_ast)

            if smoke_test:
                try:
                    from lattice import TypeRegistry
                    raw_bound = TypeRegistry.get_instance().get_verification_semantics("smoke_test_loop_bound")
                    loop_limit = int(next(iter(raw_bound))) if raw_bound else 5
                except Exception:
                    loop_limit = 5
                bounder = ASTLoopBounder(max_iters=loop_limit)
                parsed_ast = bounder.visit(parsed_ast)
                ast.fix_missing_locations(parsed_ast)

            # Step 2: Enforce POSIX resource limits & disable sockets (Track 5.1 & 5.2)
            _enforce_posix_resource_limits(max_memory_mb=max_memory_mb, max_cpu_seconds=max_cpu_seconds)
            allow_net = bool(allowed_caps_set & {"net", "net.http", "net.tcp", "net.all"})
            _disable_network_sockets(allow_network=allow_net)

            # Step 3: Configure Ephemeral Filesystem Confinement (Track 5.2)
            ephemeral_resolved = str(Path(ephemeral_dir).resolve())
            cwd_resolved = str(Path(orig_cwd).resolve())
            allowed_write_roots = {ephemeral_resolved, str(Path(tempfile.gettempdir()).resolve())}
            allowed_write_files: Set[str] = set()
            for ep in egress_paths or []:
                if ep and str(ep).strip():
                    ep_path = Path(orig_cwd) / str(ep).strip() if not Path(str(ep)).is_absolute() else Path(str(ep))
                    allowed_write_files.add(str(ep_path.resolve()))
                    allowed_write_roots.add(str(ep_path.resolve().parent))

            allowed_read_roots = {
                cwd_resolved,
                ephemeral_resolved,
                str(Path(sys.prefix).resolve()),
                str(Path(sys.exec_prefix).resolve()),
                "/usr",
                "/lib",
                "/lib64",
                "/etc/ld.so.cache",
                "/proc/self",
            }
            for p in sys.path:
                if p:
                    try:
                        allowed_read_roots.add(str(Path(p).resolve()))
                    except Exception:
                        pass

            os.environ["TMPDIR"] = ephemeral_resolved
            os.environ["TEMP"] = ephemeral_resolved
            os.environ["TMP"] = ephemeral_resolved

            # Step 4: Construct Sanitized Execution Environment
            safe_builtins: Dict[str, Any] = {}
            for k, v in builtins.__dict__.items():
                if k not in _SecurityASTVisitor.FORBIDDEN_CALLS:
                    safe_builtins[k] = v

            safe_builtins["__import__"] = _create_restricted_import(sec_policy)
            safe_builtins["open"] = _create_confined_open(
                allowed_read_roots=allowed_read_roots,
                allowed_write_roots=allowed_write_roots,
                allowed_write_files=allowed_write_files,
            )

            exec_globals: Dict[str, Any] = {
                "__builtins__": safe_builtins,
                "__name__": "__sandbox__",
                "__doc__": None,
            }
            exec_globals.update(context)

            # Step 5: Bytecode Execution under captured streams
            compiled = compile(parsed_ast, filename="<sandbox>", mode="exec")
            with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
                exec(compiled, exec_globals)

            raw_results = {
                k: v for k, v in exec_globals.items()
                if not k.startswith("__") and k not in context
            }
            for alias_name, source_var in (runtime_aliases or {}).items():
                try:
                    if source_var in raw_results:
                        raw_results[alias_name] = raw_results[source_var]
                except Exception as alias_exc:
                    logger.debug("[SANDBOX] Runtime alias bind skipped: %s", alias_exc)

            extracted_results = {k: _ipc_safe(v) for k, v in raw_results.items()}

            _send_payload({
                "success": True,
                "returncode": 0,
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "error": None,
                "results": extracted_results,
            })

        except SecurityViolationError as sec_err:
            _send_payload({
                "success": False,
                "returncode": 1,
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "error": f"SecurityViolation: {str(sec_err)}",
                "results": {},
            })
        except MemoryError as mem_err:
            _send_payload({
                "success": False,
                "returncode": 1,
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "error": f"MemoryError: Sandbox exceeded RLIMIT_AS ({max_memory_mb} MB): {mem_err}",
                "results": {},
            })
        except Exception as exc:
            formatted_exc = traceback.format_exc()
            _send_payload({
                "success": False,
                "returncode": 1,
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "error": f"{type(exc).__name__}: {str(exc)}\n{formatted_exc}",
                "results": {},
            })


# ============================================================================
# Main GEVR Sandbox Manager (with Closed-Loop CEGIS Integration - Track 8.2)
# ============================================================================

_IO_SIGNATURES_CACHE: Optional[Tuple[int, Tuple[Set[str], Set[str], Set[str], Set[str]]]] = None


def _derive_io_call_signatures() -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Introspects code templates of all loaded cells in LatticeOrchestrator to derive
    callables and keyword argument names that accept filesystem paths.
    Returns:
        (sink_callables, sink_kw_args, ingress_callables, ingress_kw_args)
    """
    global _IO_SIGNATURES_CACHE
    try:
        from lattice import LatticeOrchestrator, TypeRegistry, is_path_port
        orch = LatticeOrchestrator.get_active_instance()
        if not orch or not getattr(orch, "loaded_cells", None):
            return set(), set(), set(), set()
        cell_count = len(orch.loaded_cells)
        if _IO_SIGNATURES_CACHE is not None and _IO_SIGNATURES_CACHE[0] == cell_count:
            return _IO_SIGNATURES_CACHE[1]

        sink_callables: Set[str] = set()
        sink_kw_args: Set[str] = set()
        ingress_callables: Set[str] = set()
        ingress_kw_args: Set[str] = set()

        reg = TypeRegistry.get_instance()
        path_tokens = reg.get_path_port_tokens() if reg else set()

        for cell in orch.loaded_cells.values():
            c_role = str(getattr(cell, "node_role", "") or "").lower()
            c_stage = getattr(cell, "stage", None)
            tpl = getattr(cell, "code_template", "") or ""
            if not tpl:
                continue

            path_ports = [
                p for p in (list(getattr(cell, "inputs", {}).values()) + list(getattr(cell, "outputs", {}).values() if isinstance(getattr(cell, "outputs", {}), dict) else []))
                if is_path_port(p) or getattr(p, "abstract_type", "") == "path" or getattr(p, "type_name", "") in path_tokens
            ]
            if not path_ports:
                continue

            # Replace {ph} with _nstl_ph_ph so ast.parse succeeds on code templates without regex
            parsed_tpl_code = tpl
            for ph in extract_template_placeholders(tpl):
                parsed_tpl_code = parsed_tpl_code.replace(f"{{{ph}}}", f"_nstl_ph_{ph}")
            try:
                c_ast = ast.parse(parsed_tpl_code)
            except Exception:
                continue

            is_sink = (
                getattr(cell, "is_endable", False)
                or not getattr(cell, "outputs", None)
                or c_role in ("sink", "egress", "terminal")
                or any(str(getattr(getattr(p, "signature", p), "state", "")).lower() in ("materialization", "destination", "serialized", "sink") for p in getattr(cell, "inputs", {}).values())
            )
            is_ingress = (
                getattr(cell, "is_ingress", False)
                or (len(getattr(cell, "inputs", {})) == 0)
                or c_role in ("source", "ingress")
            )

            for c_node in ast.walk(c_ast):
                if isinstance(c_node, ast.Call):
                    func = c_node.func
                    fn_name = None
                    if isinstance(func, ast.Attribute):
                        fn_name = func.attr
                    elif isinstance(func, ast.Name):
                        fn_name = func.id

                    if not fn_name:
                        continue

                    has_path_arg = False
                    matched_kws: Set[str] = set()

                    for arg in c_node.args:
                        if isinstance(arg, ast.Name) and arg.id.startswith("_nstl_ph_"):
                            has_path_arg = True
                            break
                    for kw in c_node.keywords:
                        if kw.arg and any(getattr(p, "name", "") == kw.arg for p in path_ports):
                            matched_kws.add(kw.arg)
                            has_path_arg = True
                        elif isinstance(kw.value, ast.Name) and kw.value.id.startswith("_nstl_ph_"):
                            if kw.arg:
                                matched_kws.add(kw.arg)
                            has_path_arg = True

                    if has_path_arg:
                        if is_sink:
                            sink_callables.add(fn_name)
                            sink_kw_args.update(matched_kws)
                        elif is_ingress:
                            ingress_callables.add(fn_name)
                            ingress_kw_args.update(matched_kws)

        res = (sink_callables, sink_kw_args, ingress_callables, ingress_kw_args)
        _IO_SIGNATURES_CACHE = (cell_count, res)
        return res
    except Exception:
        return set(), set(), set(), set()


class GEVRSandbox:
    """
    Hardened execution sandbox with strict process isolation, POSIX resource limits,
    network/filesystem confinement, timeout enforcement, and automated CEGIS repair.
    """

    _EXTRINSIC_ERROR_MARKERS: Tuple[str, ...] = (
        "ModuleNotFoundError",
        "ImportError",
        "FileNotFoundError",
        "FileExistsError",
        "PermissionError",
        "No such file or directory",
    )

    def __init__(
        self,
        default_timeout: float = 30.0,
        max_memory_mb: Optional[int] = None,
        max_cpu_seconds: Optional[int] = None,
        postcondition_registry: Optional[PostconditionRegistry] = None,
        intent_registry: Optional[TerminalIntentRegistry] = None,
        state_registry: Optional[StateProtocolRegistry] = None,
        allowed_capabilities: Optional[Set[str]] = None,
    ):
        self.default_timeout = float(
            os.environ.get(
                "NSTL_SANDBOX_TIMEOUT",
                settings.sandbox_timeout if settings else default_timeout,
            )
        )
        self.max_memory_mb = int(
            max_memory_mb
            if max_memory_mb is not None
            else (settings.sandbox_max_memory_mb if settings else 8192)
        )
        self.max_cpu_seconds = int(
            max_cpu_seconds
            if max_cpu_seconds is not None
            else (settings.sandbox_max_cpu_seconds if settings else 30)
        )
        self.postconditions = postcondition_registry or PostconditionRegistry()
        self.intents = intent_registry or TerminalIntentRegistry()
        self.state_protocols = state_registry or StateProtocolRegistry()
        self.allowed_capabilities = set(allowed_capabilities or set())

    @staticmethod
    def _verify_egress_paths(egress_paths: Optional[List[str]]) -> Dict[str, str]:
        verification: Dict[str, str] = {}
        for raw_path in egress_paths or []:
            path = str(raw_path)
            if not path.strip():
                continue
            if not os.path.exists(path):
                verification[path] = "missing"
            elif os.path.getsize(path) <= 0:
                verification[path] = "empty"
            else:
                verification[path] = "ok"
        return verification

    @staticmethod
    def _evaluate_verification_spec(
        verification_spec: Any,
        results: Dict[str, Any],
    ) -> List[str]:
        violations: List[str] = []
        if verification_spec is None:
            return violations

        checks: List[Dict[str, Any]] = []
        if isinstance(verification_spec, dict):
            checks = list(verification_spec.get("terminal_checks") or [])
        else:
            checks = list(getattr(verification_spec, "terminal_checks", None) or [])

        for check in checks:
            if not isinstance(check, dict):
                continue

            if check.get("type") == "observable_deliverable":
                v_name = check.get("var")
                cid = check.get("cell_id", "unknown_cell")
                if not v_name or str(v_name) not in results:
                    violations.append(f"Deliverable variable '{v_name}' produced by '{cid}' was not found in sandbox execution scope.")
                else:
                    v_val = results[str(v_name)]
                    if v_val is None:
                        violations.append(f"Deliverable variable '{v_name}' produced by '{cid}' evaluated to None.")
                    else:
                        try:
                            f_val = float(v_val)
                            if math.isnan(f_val) or math.isinf(f_val):
                                violations.append(f"Deliverable variable '{v_name}' produced by '{cid}' is NaN or infinite ({v_val}).")
                        except (TypeError, ValueError):
                            pass

                        if hasattr(v_val, "__len__"):
                            try:
                                if len(v_val) == 0:
                                    violations.append(f"Deliverable variable '{v_name}' produced by '{cid}' is empty (length 0).")
                            except Exception:
                                pass
                        elif hasattr(v_val, "size"):
                            try:
                                if int(getattr(v_val, "size", 1)) == 0:
                                    violations.append(f"Deliverable variable '{v_name}' produced by '{cid}' is empty (size 0).")
                            except Exception:
                                pass

                        for post in check.get("postconditions", []) or []:
                            prop = getattr(post, "property", None) or (post.get("property") if isinstance(post, dict) else None)
                            if prop:
                                exp = getattr(post, "value", None) if not isinstance(post, dict) else post.get("value")
                                try:
                                    ok = PostconditionRegistry().evaluate(prop, v_val, exp)
                                    if not ok:
                                        violations.append(f"Postcondition '{prop}' on deliverable '{v_name}' ({cid}) was not satisfied.")
                                except Exception as eval_err:
                                    logger.debug("[SANDBOX] deliverable postcondition eval failed: %s", eval_err)
                continue

            expected_var = check.get("expected_var")
            actual_var = check.get("actual_var")
            if expected_var and actual_var:
                expected_val = results.get(str(expected_var))
                actual_val = results.get(str(actual_var))
                if expected_val is not None and actual_val is not None:
                    identical = expected_val is actual_val
                    if not identical:
                        try:
                            identical = bool(expected_val == actual_val)
                        except Exception:
                            identical = False
                        violations.append(
                            f"Terminal check '{check.get('type', 'unknown')}' failed: "
                            f"'{actual_var}' does not carry the value declared for '{expected_var}'."
                        )
                exp_props = check.get("expected_properties")
                if exp_props and actual_val is not None:
                    for prop_name, prop_val in exp_props.items():
                        try:
                            ok = PostconditionRegistry().evaluate(prop_name, actual_val, prop_val)
                        except Exception as eval_err:
                            logger.debug("[SANDBOX] property '%s' evaluation failed: %s", prop_name, eval_err)
                            ok = False
                        if not ok:
                            violations.append(
                                f"Property '{prop_name}' on result '{actual_var}' was not satisfied."
                            )
                continue

            prop = check.get("property") or check.get("postcondition")
            target_var = check.get("target_var") or check.get("var")
            if prop and target_var and target_var in results:
                expected = check.get("expected")
                try:
                    ok = PostconditionRegistry().evaluate(prop, results[target_var], expected)
                except Exception as eval_err:
                    logger.debug("[SANDBOX] postcondition '%s' evaluation failed: %s", prop, eval_err)
                    ok = False
                if not ok:
                    violations.append(
                        f"Postcondition '{prop}' on result '{target_var}' was not satisfied."
                    )

        return violations

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        egress_paths: Optional[List[str]] = None,
        verification_spec: Optional[Any] = None,
        runtime_aliases: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute code in a separate resource- and filesystem-confined process.
        """
        missing_inputs: List[str] = []
        egress_set: Set[str] = {str(p).strip() for p in (egress_paths or []) if p}
        try:
            parsed_ast = ast.parse(code)

            # Incorporate any egress/sink bindings from pipeline_bindings
            for b in kwargs.get("pipeline_bindings") or []:
                if isinstance(b, dict):
                    c_role = str(b.get("role", "") or b.get("node_role", "")).lower()
                    c_stage = b.get("stage")
                    if c_role in ("sink", "egress", "terminal") or (c_stage is not None and c_stage >= 3):
                        for v in b.values():
                            if isinstance(v, str) and not v.startswith("var_"):
                                cln = v.strip("'\"")
                                if "." in cln:
                                    egress_set.add(cln)

            # Dynamically derive ingress vs egress callables and path parameters from loaded cells
            sink_callables, sink_kw_args, ingress_callables, ingress_kw_args = _derive_io_call_signatures()

            # Pass 1: Identify all output/egress paths written in the code
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Call):
                    func = node.func
                    is_sink_call = False
                    if isinstance(func, ast.Attribute) and func.attr in sink_callables:
                        is_sink_call = True
                    elif isinstance(func, ast.Name) and func.id in sink_callables:
                        is_sink_call = True
                    elif isinstance(func, ast.Name) and func.id == "open":
                        mode = None
                        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                            mode = str(node.args[1].value)
                        for kw in node.keywords:
                            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                                mode = str(kw.value.value)
                        if mode and any(m in mode for m in ("w", "a", "x")):
                            is_sink_call = True

                    if is_sink_call:
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                egress_set.add(arg.value.strip())
                        for kw in node.keywords:
                            if (not sink_kw_args or kw.arg in sink_kw_args) and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                egress_set.add(kw.value.value.strip())

            # Pass 2: Check only ingress/loader calls for required input files
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Call):
                    func = node.func
                    is_ingress_call = False
                    if isinstance(func, ast.Attribute) and func.attr in ingress_callables:
                        is_ingress_call = True
                    elif isinstance(func, ast.Name) and func.id in ingress_callables:
                        is_ingress_call = True
                    elif isinstance(func, ast.Name) and func.id == "open":
                        mode = "r"
                        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                            mode = str(node.args[1].value)
                        for kw in node.keywords:
                            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                                mode = str(kw.value.value)
                        if "r" in mode and not any(m in mode for m in ("w", "a", "x")):
                            is_ingress_call = True

                    if is_ingress_call:
                        for arg in node.args:
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                s = arg.value.strip()
                                if s in egress_set:
                                    continue
                                _, raw_ext = os.path.splitext(s)
                                clean_ext = raw_ext.lstrip(".").lower()
                                if clean_ext and clean_ext.isalnum() and len(clean_ext) <= 6:
                                    if not os.path.exists(s) and not os.path.exists(os.path.abspath(s)):
                                        if s not in missing_inputs:
                                            missing_inputs.append(s)
                        for kw in node.keywords:
                            if (not ingress_kw_args or kw.arg in ingress_kw_args) and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                s = kw.value.value.strip()
                                if s in egress_set:
                                    continue
                                _, raw_ext = os.path.splitext(s)
                                clean_ext = raw_ext.lstrip(".").lower()
                                if clean_ext and clean_ext.isalnum() and len(clean_ext) <= 6:
                                    if not os.path.exists(s) and not os.path.exists(os.path.abspath(s)):
                                        if s not in missing_inputs:
                                            missing_inputs.append(s)
        except Exception as ast_err:
            logger.debug("[SANDBOX] AST inspection for input files failed: %s", ast_err)

        if missing_inputs:
            missing_file = missing_inputs[0]
            logger.info("[SANDBOX] Input file '%s' not found on disk. Skipping execution.", missing_file)
            return {
                "success": False,
                "skipped": True,
                "status": "SKIPPED",
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "error": f"SKIPPED: input '{missing_file}' not provided",
                "extrinsic": False,
                "results": {},
                "egress": {},
            }

        exec_timeout = timeout if timeout is not None else self.default_timeout
        context = context or {}
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)

        smoke_test = bool(kwargs.get("smoke_test", False))
        mem_limit = int(kwargs.get("max_memory_mb", self.max_memory_mb))
        cpu_limit = int(kwargs.get("max_cpu_seconds", max(int(exec_timeout) + 2, self.max_cpu_seconds)))
        allowed_caps = kwargs.get("allowed_capabilities")
        if allowed_caps is None:
            allowed_caps = getattr(self, "allowed_capabilities", set())
        if isinstance(allowed_caps, (set, frozenset, list, tuple)):
            allowed_caps = list(allowed_caps)
        else:
            allowed_caps = []
        if not allowed_caps and "NSTL_ALLOWED_CAPABILITIES" in os.environ:
            allowed_caps = [c.strip() for c in os.environ["NSTL_ALLOWED_CAPABILITIES"].split(",") if c.strip()]

        worker = ctx.Process(
            target=_sandbox_worker_exec,
            args=(
                code,
                context,
                dict(runtime_aliases or {}),
                child_conn,
                smoke_test,
                list(egress_set),
                mem_limit,
                cpu_limit,
                allowed_caps,
            ),
            daemon=True,
        )

        worker.start()
        # Close child endpoint in parent so EOF is cleanly detected
        try:
            child_conn.close()
        except Exception:
            pass

        deadline = time.monotonic() + max(float(exec_timeout), 0.05)
        payload: Optional[Dict[str, Any]] = None
        timed_out = False
        try:
            remaining = max(0.05, deadline - time.monotonic())
            if parent_conn.poll(timeout=remaining):
                payload = parent_conn.recv()
            else:
                timed_out = True
        except (EOFError, BrokenPipeError, ConnectionResetError):
            payload = None
        except Exception as recv_err:
            logger.warning("[SANDBOX] Result reception failed: %s", recv_err)
            payload = None
        finally:
            try:
                parent_conn.close()
            except Exception:
                pass

        if payload is None and worker.is_alive():
            logger.warning("Worker PID %s timed out after %ss. Terminating.", worker.pid, exec_timeout)
            timed_out = True
            worker.terminate()
            worker.join(timeout=0.5)

            if worker.is_alive():
                logger.critical("Worker PID %s refused SIGTERM. Issuing SIGKILL.", worker.pid)
                try:
                    if hasattr(signal, "SIGKILL"):
                        os.kill(worker.pid, signal.SIGKILL)
                    else:
                        worker.kill()
                except ProcessLookupError:
                    pass
                worker.join()

        if payload is not None:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.5)

        envelope: Dict[str, Any] = {
            "success": False,
            "returncode": worker.exitcode if worker.exitcode is not None else 1,
            "stdout": "",
            "stderr": "",
            "error": "",
            "extrinsic": False,
            "results": {},
            "egress": {},
        }

        if timed_out:
            envelope["error"] = f"Execution timed out after {exec_timeout} seconds."
            envelope["returncode"] = worker.exitcode if worker.exitcode is not None else 1
            return envelope

        if payload is None:
            envelope["error"] = "Execution worker terminated abruptly without returning a result."
            envelope["extrinsic"] = True
            return envelope

        envelope["success"] = bool(payload.get("success", False))
        envelope["returncode"] = int(payload.get("returncode", 0) or 0)
        envelope["stdout"] = str(payload.get("stdout", "") or "")
        envelope["stderr"] = str(payload.get("stderr", "") or "")
        envelope["error"] = str(payload.get("error", "") or "")
        envelope["results"] = dict(payload.get("results", {}) or {})

        if envelope["error"]:
            envelope["extrinsic"] = any(
                marker in envelope["error"] for marker in self._EXTRINSIC_ERROR_MARKERS
            )

        egress_verification = self._verify_egress_paths(egress_paths)
        envelope["egress"] = egress_verification
        if egress_verification:
            bad = {p: v for p, v in egress_verification.items() if v != "ok"}
            if bad:
                envelope["success"] = False
                detail = "; ".join(f"{p}: {v}" for p, v in sorted(bad.items()))
                envelope["error"] = (
                    (envelope["error"] + "\n" if envelope["error"] else "")
                    + f"Egress verification failed: {detail}"
                )
                envelope["extrinsic"] = True

        spec_violations = self._evaluate_verification_spec(verification_spec, envelope["results"])
        if spec_violations:
            envelope["success"] = False
            envelope["error"] = (
                (envelope["error"] + "\n" if envelope["error"] else "")
                + "Verification contract violated: " + " | ".join(spec_violations)
            )

        return envelope

    def execute_with_cegis(
        self,
        code: str,
        task_description: str,
        state_inputs: Dict[str, str],
        state_outputs: Dict[str, str],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_cegis_rounds: int = 3,
        kernel_generator: Optional[Any] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Track 8.2: Closed-Loop CEGIS (Counter-Example Guided Inductive Synthesis) Gate.
        Executes `code` in the isolated sandbox; if a non-extrinsic runtime exception
        or contract violation occurs, routes the traceback and failing inputs into
        `InductiveKernelGenerator.repair_kernel()` up to `max_cegis_rounds` times.
        Returns `(final_code, final_sandbox_envelope)`.
        """
        current_code = code
        res = self.execute(current_code, context=context, timeout=timeout, **kwargs)
        if res.get("success") or res.get("skipped") or res.get("extrinsic"):
            res["cegis_rounds"] = 0
            return current_code, res

        if kernel_generator is None:
            try:
                from .synthesis import InductiveKernelGenerator
            except (ImportError, ValueError):
                from synthesis import InductiveKernelGenerator
            kernel_generator = InductiveKernelGenerator()

        for round_idx in range(1, max_cegis_rounds + 1):
            err_trace = res.get("error") or res.get("stderr") or "Runtime verification failure"
            try:
                repaired = kernel_generator.repair_kernel(
                    failing_code=current_code,
                    task_description=task_description,
                    traceback_error=err_trace,
                    state_inputs=state_inputs,
                    state_outputs=state_outputs,
                    failing_inputs=context,
                )
            except Exception as repair_exc:
                logger.warning("[CEGIS] Round %d repair aborted: %s", round_idx, repair_exc)
                break

            if not repaired or repaired.strip() == current_code.strip():
                break

            current_code = repaired
            res = self.execute(current_code, context=context, timeout=timeout, **kwargs)
            res["cegis_rounds"] = round_idx
            if res.get("success") or res.get("extrinsic"):
                return current_code, res

        res.setdefault("cegis_rounds", max_cegis_rounds)
        return current_code, res

    def verify_postcondition(self, property_name: str, target: Any, expected: Any) -> bool:
        return self.postconditions.evaluate(property_name, target, expected)

    def verify_intent(
        self, intent_type: str, output: Any, context: Optional[Dict[str, Any]] = None
    ) -> IntentVerificationResult:
        return self.intents.verify(intent_type, output, context or {})

    def is_fitted(self, estimator: Any) -> bool:
        return self.state_protocols.is_fitted_or_ready(estimator)
