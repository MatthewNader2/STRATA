"""
src/gevr_sandbox.py
Hardened, isolated, domain-agnostic sandboxed execution environment with
AST security inspection, POSIX resource limits (RLIMIT_AS / RLIMIT_CPU),
ephemeral filesystem confinement, network socket blocking, and closed-loop
CEGIS self-repair.
"""

from __future__ import annotations

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
import shutil
import signal
import socket as _real_socket
import sys
import tempfile
import threading
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

try:
    from .security_policy import (
        SecurityViolationError,
        _SecurityVisitor as _UnifiedSecurityASTVisitor,
    )
except (ImportError, ValueError):
    from security_policy import (
        SecurityViolationError,
        _SecurityVisitor as _UnifiedSecurityASTVisitor,
    )


class SandboxTimeoutError(TimeoutError):
    """Raised when execution exceeds the allocated wall-clock time limit."""
    pass


class _SecurityASTVisitor(_UnifiedSecurityASTVisitor):
    """Unified AST security visitor backed by security_policy._SecurityVisitor."""
    def __init__(self, policy: Optional[Any] = None) -> None:
        super().__init__(policy=policy)


# ============================================================================
# Pluggable Postconditions (Domain-Agnostic)
# ============================================================================

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

        def _reduce_any_bool(obj: Any) -> bool:
            cur = obj
            for _ in range(4):
                if isinstance(cur, bool) or getattr(cur, "ndim", 1) == 0:
                    break
                any_fn = getattr(cur, "any", None)
                if not callable(any_fn):
                    break
                nxt = any_fn()
                if nxt is cur:
                    break
                cur = nxt
            return bool(cur)

        def _has_nans_evaluator(val: Any, expected: Any) -> bool:
            try:
                from lattice import TypeRegistry
                nan_methods = TypeRegistry.get_instance().get_type_property_values("missing_value_methods")
            except Exception:
                nan_methods = set()
            for method_name in nan_methods:
                method = getattr(val, method_name, None)
                if callable(method):
                    res = method()
                    has_nan = _reduce_any_bool(res)
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
            try:
                try:
                    from .lattice import TypeRegistry
                except ImportError:
                    from lattice import TypeRegistry
                dup_methods = TypeRegistry.get_instance().get_type_property_values("duplicate_check_methods")
            except Exception:
                dup_methods = set()
            for method_name in dup_methods:
                method = getattr(val, method_name, None)
                if callable(method):
                    dups = method()
                    is_unique = not _reduce_any_bool(dups)
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
# Hardened Worker Execution Layer (Track 5: RLIMIT, Network & FS Confinement)
# ============================================================================

try:
    from .security_policy import (
        DEFAULT_CAPABILITY_MODULES as _SP_CAP_MODS,
        NON_OVERRIDABLE_BLOCKED_MODULES as _SP_BLOCKED_MODS,
        NON_OVERRIDABLE_BLOCKED_OS_OPS as _SP_BLOCKED_OS_OPS,
    )
except (ImportError, ValueError):
    from security_policy import (
        DEFAULT_CAPABILITY_MODULES as _SP_CAP_MODS,
        NON_OVERRIDABLE_BLOCKED_MODULES as _SP_BLOCKED_MODS,
        NON_OVERRIDABLE_BLOCKED_OS_OPS as _SP_BLOCKED_OS_OPS,
    )

_BLOCKED_MODULES = tuple(dict.fromkeys((*_SP_BLOCKED_MODS, *_SP_CAP_MODS.keys())))


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
                max_memory_mb = int(env_mem)
            except ValueError:
                pass
        mem_bytes = int(max_memory_mb) * 1024 * 1024
        as_bytes = int(max_memory_mb) * 1024 * 1024
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


_SAFE_OS_PATH_ATTRS: Tuple[str, ...] = (
    "join", "split", "splitext", "basename", "dirname", "normpath",
    "abspath", "relpath", "realpath", "exists", "isfile", "isdir",
    "getsize", "commonpath", "commonprefix", "expanduser", "isabs",
    "sep", "altsep", "extsep", "pathsep", "curdir", "pardir", "devnull",
)


def _create_sanitized_os_path():
    """Wrap standard `os.path` exposing only safe path utilities (no `.os` module escape)."""
    import os as _real_os
    _real_path = _real_os.path

    class _SanitizedOSPath:
        def __getattr__(self, name: str) -> Any:
            if name not in _SAFE_OS_PATH_ATTRS:
                raise SecurityViolationError(
                    f"Access to restricted attribute 'os.path.{name}' is prohibited in sandbox."
                )
            return getattr(_real_path, name)

    return _SanitizedOSPath()


def _create_sanitized_os():
    """Wrap standard `os` to disable shell execution and destructive file operations."""
    import os as _real_os

    class _SanitizedOS:
        path = _create_sanitized_os_path()

        def __getattr__(self, name: str) -> Any:
            if name in _SP_BLOCKED_OS_OPS or name in {"sys", "builtins", "__dict__"}:
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

        if name == "os.path" or name.startswith("os.path."):
            for item in fromlist or ():
                if item == "*" or item not in _SAFE_OS_PATH_ATTRS:
                    raise SecurityViolationError(
                        f"Importing '{item}' from '{name}' is prohibited in sandbox."
                    )
            return _create_sanitized_os_path() if fromlist else _create_sanitized_os()

        if root_module == "os":
            for item in fromlist or ():
                if item == "*" or item in _SP_BLOCKED_OS_OPS or item in {"sys", "builtins"}:
                    raise SecurityViolationError(
                        f"Importing '{item}' from 'os' is prohibited in sandbox."
                    )
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


_HEADLESS_ENV_SPECS_CACHE: Optional[List[str]] = None
_SANDBOX_MP_CTX: Any = None
_SANDBOX_CTX_LOCK = threading.Lock()
_SANDBOX_WARMUP_STARTED = False


def _get_headless_env_specs() -> List[str]:
    """Resolves and caches headless environment specs in the parent process so workers avoid reloading lattice trees."""
    global _HEADLESS_ENV_SPECS_CACHE
    if _HEADLESS_ENV_SPECS_CACHE is not None:
        return _HEADLESS_ENV_SPECS_CACHE
    specs: List[str] = []
    try:
        from lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        specs = list(reg.get_capabilities().get("headless_environment", []) or [])
    except Exception:
        specs = []
    _HEADLESS_ENV_SPECS_CACHE = specs
    return specs


def _get_sandbox_mp_context() -> Any:
    """Returns a cached forkserver context on Linux (preloading pure-Python data libs) or spawn fallback."""
    global _SANDBOX_MP_CTX
    if _SANDBOX_MP_CTX is not None:
        return _SANDBOX_MP_CTX
    with _SANDBOX_CTX_LOCK:
        if _SANDBOX_MP_CTX is not None:
            return _SANDBOX_MP_CTX
        forced = os.environ.get("NSTL_SANDBOX_START_METHOD", "").strip().lower()
        if forced == "spawn":
            _SANDBOX_MP_CTX = multiprocessing.get_context("spawn")
            return _SANDBOX_MP_CTX
        if sys.platform.startswith("linux") and "forkserver" in multiprocessing.get_all_start_methods():
            try:
                ctx = multiprocessing.get_context("forkserver")
                _SANDBOX_MP_CTX = ctx
                return ctx
            except Exception as exc:
                logger.debug("[SANDBOX] forkserver init fallback to spawn: %s", exc)
        _SANDBOX_MP_CTX = multiprocessing.get_context("spawn")
        return _SANDBOX_MP_CTX


def warmup_sandbox_async() -> None:
    """Starts the sandbox forkserver zygote and preloads headless specs on a background daemon thread."""
    global _SANDBOX_WARMUP_STARTED
    if os.environ.get("NSTL_SANDBOX_WARMUP", "1") == "0":
        return
    with _SANDBOX_CTX_LOCK:
        if _SANDBOX_WARMUP_STARTED:
            return
        _SANDBOX_WARMUP_STARTED = True

    def _warm() -> None:
        try:
            _get_headless_env_specs()
            ctx = _get_sandbox_mp_context()
            if getattr(ctx, "_name", "") == "forkserver":
                import multiprocessing.forkserver as _fs
                _fs.ensure_running()
        except Exception:
            pass

    threading.Thread(target=_warm, name="sandbox-forkserver-warmup", daemon=True).start()


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
    ingress_paths: Optional[List[str]] = None,
    headless_env_specs: Optional[List[str]] = None,
    capture_egress: bool = False,
    max_egress_bytes: int = 4 * 1024 * 1024,
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
        if headless_env_specs is None:
            env_specs = _get_headless_env_specs()
        else:
            env_specs = headless_env_specs
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
                    "egress_bytes": payload_dict.get("egress_bytes", {}),
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
    import _io as _c_io
    orig_builtins_open = None
    orig_io_open = None
    orig_cio_open = None

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
                    from config import settings as _sb_settings
                    loop_limit = int(_sb_settings.smoke_test_loop_bound)
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
            ephemeral_root_p = Path(ephemeral_resolved)
            cwd_resolved = str(Path(orig_cwd).resolve())
            allowed_write_roots = {ephemeral_resolved}
            allowed_write_files: Set[str] = set()
            for ep in egress_paths or []:
                if ep and str(ep).strip():
                    ep_s = str(ep).strip()
                    ep_cand = (ephemeral_root_p / ep_s).resolve() if not Path(ep_s).is_absolute() else Path(ep_s).resolve()
                    if not ep_cand.is_relative_to(ephemeral_root_p):
                        raise SecurityViolationError(
                            f"Egress path '{ep_s}' escapes ephemeral sandbox directory and is prohibited."
                        )
                    allowed_write_files.add(str(ep_cand))

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
            # Stage verified relative ingress files (read-only copies) into ephemeral_dir
            # so relative paths like 'input.csv' resolve inside the confined working dir
            # without allowing writes back to the workspace copies.
            for rel_in in ingress_paths or ():
                rel_str = str(rel_in or "").strip()
                if not rel_str or os.path.isabs(rel_str):
                    continue
                src_p = (Path(orig_cwd) / rel_str).resolve()
                dst_p = (ephemeral_root_p / rel_str).resolve()
                if not src_p.is_file() or not dst_p.is_relative_to(ephemeral_root_p):
                    continue
                dst_p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_p), str(dst_p))
            os.chdir(ephemeral_dir)

            # Step 4: Construct Sanitized Execution Environment & Confine ALL Open Entrypoints
            confined_open = _create_confined_open(
                allowed_read_roots=allowed_read_roots,
                allowed_write_roots=allowed_write_roots,
                allowed_write_files=allowed_write_files,
            )
            orig_builtins_open = builtins.open
            orig_io_open = _io.open
            orig_cio_open = _c_io.open
            builtins.open = confined_open
            _io.open = confined_open
            _c_io.open = confined_open

            safe_builtins: Dict[str, Any] = {}
            for k, v in builtins.__dict__.items():
                if k not in _SecurityASTVisitor.FORBIDDEN_CALLS:
                    safe_builtins[k] = v

            safe_builtins["__import__"] = _create_restricted_import(sec_policy)
            safe_builtins["open"] = confined_open

            exec_globals: Dict[str, Any] = {
                "__builtins__": safe_builtins,
                "__name__": "__sandbox__",
                "__doc__": None,
            }
            exec_globals.update(context)

            # Step 5: Bytecode Execution under captured streams (handle explicit sys.exit)
            compiled = compile(parsed_ast, filename="<sandbox>", mode="exec")
            sys_exit_code = 0
            with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
                try:
                    exec(compiled, exec_globals)
                except SystemExit as sys_exit_exc:
                    code_val = sys_exit_exc.code
                    if isinstance(code_val, int):
                        sys_exit_code = code_val
                    elif code_val is not None:
                        sys_exit_code = 1
                        captured_stderr.write(f"\nSystemExit: {code_val}\n")
            if sys_exit_code != 0:
                _send_payload({
                    "success": False,
                    "returncode": sys_exit_code,
                    "stdout": captured_stdout.getvalue(),
                    "stderr": captured_stderr.getvalue(),
                    "error": f"SystemExit({sys_exit_code})",
                    "error_kind": "runtime",
                    "results": {},
                })
                return

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

            worker_egress: Dict[str, str] = {}
            worker_egress_bytes: Dict[str, bytes] = {}
            for raw_ep in egress_paths or ():
                ep_s = str(raw_ep or "").strip()
                if not ep_s:
                    continue
                ep_cand = (ephemeral_root_p / ep_s).resolve() if not Path(ep_s).is_absolute() else Path(ep_s).resolve()
                if not ep_cand.is_relative_to(ephemeral_root_p):
                    worker_egress[ep_s] = "outside_sandbox"
                elif ep_cand.exists():
                    sz = ep_cand.stat().st_size
                    worker_egress[ep_s] = "ok" if sz > 0 else "empty"
                    if capture_egress and ep_cand.is_file() and 0 < sz <= max_egress_bytes:
                        try:
                            worker_egress_bytes[ep_s] = ep_cand.read_bytes()
                        except Exception as read_err:
                            logger.debug("[SANDBOX] Egress capture skipped for %s: %s", ep_s, read_err)
                else:
                    worker_egress[ep_s] = "missing"

            _send_payload({
                "success": True,
                "returncode": 0,
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "error": None,
                "error_kind": None,
                "results": extracted_results,
                "egress": worker_egress,
                "egress_bytes": worker_egress_bytes,
            })

        except SecurityViolationError as sec_err:
            _send_payload({
                "success": False,
                "returncode": 1,
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "error": f"SecurityViolation: {str(sec_err)}",
                "error_kind": "security",
                "results": {},
            })
        except MemoryError as mem_err:
            _send_payload({
                "success": False,
                "returncode": 1,
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "error": f"MemoryError: Sandbox exceeded RLIMIT_AS ({max_memory_mb} MB): {mem_err}",
                "error_kind": "memory",
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
                "error_kind": "runtime",
                "results": {},
            })
        finally:
            if orig_builtins_open is not None:
                builtins.open = orig_builtins_open
            if orig_io_open is not None:
                _io.open = orig_io_open
            if orig_cio_open is not None:
                _c_io.open = orig_cio_open
            try:
                os.chdir(orig_cwd)
            except Exception:
                pass


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
        path_types = (set(reg.get_types_by_kind("path") or ()) | {"path", "file"}) if reg else {"path", "file"}

        for cell in orch.loaded_cells.values():
            c_role = str(getattr(cell, "node_role", "") or "").lower()
            c_stage = getattr(cell, "stage", None)
            tpl = getattr(cell, "code_template", "") or ""
            if not tpl:
                continue

            path_ports = [
                p for p in (list(getattr(cell, "inputs", {}).values()) + list(getattr(cell, "outputs", {}).values() if isinstance(getattr(cell, "outputs", {}), dict) else []))
                if is_path_port(p) or getattr(p, "abstract_type", "") == "path" or str(getattr(p, "type_name", "") or "").strip().lower() in path_types
            ]
            if not path_ports:
                continue

            try:
                from .language_backend import get_active_language_backend
            except (ImportError, ValueError):
                from language_backend import get_active_language_backend
            _be_io = get_active_language_backend()
            if getattr(_be_io, "spec", None) is not None and not _be_io.is_python_ast:
                continue

            # Replace {ph} with _nstl_ph_ph so ast.parse succeeds on code templates without regex
            parsed_tpl_code = tpl
            for ph in extract_template_placeholders(tpl):
                parsed_tpl_code = parsed_tpl_code.replace(f"{{{ph}}}", f"_nstl_ph_{ph}")
            try:
                c_ast = ast.parse(parsed_tpl_code)
            except Exception:
                continue

            try:
                from lattice import TypeRegistry
                reg_io = TypeRegistry.get_instance()
                is_term_role = reg_io.is_terminal_role(c_role)
                is_in_role = reg_io.is_ingress_role(c_role)
                dest_states = (
                    set(reg_io.get_terminal_states() or ())
                    | reg_io.get_states_by_property("is_materialized", True)
                    | reg_io.get_states_by_property("is_terminal", True)
                    | reg_io.get_roles_for_category("destination")
                )
            except Exception:
                is_term_role, is_in_role, dest_states = False, False, set()
            is_sink = (
                not getattr(cell, "outputs", None)
                or is_term_role
                or any(str(getattr(getattr(p, "signature", p), "state", "")).lower() in dest_states for p in getattr(cell, "inputs", {}).values())
            )
            is_ingress = (
                getattr(cell, "is_ingress", False)
                or (len(getattr(cell, "inputs", {})) == 0)
                or is_in_role
            )

            chained_receivers: Set[int] = {
                id(n.func.value)
                for n in ast.walk(c_ast)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Call)
            }

            def _call_has_path_ph(call_node: ast.Call) -> Tuple[bool, Set[str]]:
                found_ph = False
                kws: Set[str] = set()
                cur: Optional[ast.AST] = call_node
                while isinstance(cur, ast.Call):
                    for arg in cur.args:
                        if isinstance(arg, ast.Name) and arg.id.startswith("_nstl_ph_"):
                            found_ph = True
                    for kw in cur.keywords:
                        if kw.arg and any(getattr(p, "name", "") == kw.arg for p in path_ports):
                            kws.add(kw.arg)
                            found_ph = True
                        elif isinstance(kw.value, ast.Name) and kw.value.id.startswith("_nstl_ph_"):
                            if kw.arg:
                                kws.add(kw.arg)
                            found_ph = True
                    cur = cur.func.value if isinstance(cur.func, ast.Attribute) and isinstance(cur.func.value, ast.Call) else None
                return found_ph, kws

            for c_node in ast.walk(c_ast):
                if isinstance(c_node, ast.Call):
                    if id(c_node) in chained_receivers:
                        continue
                    func = c_node.func
                    fn_name = None
                    if isinstance(func, ast.Attribute):
                        fn_name = func.attr
                    elif isinstance(func, ast.Name):
                        fn_name = func.id

                    if not fn_name:
                        continue

                    has_path_arg, matched_kws = _call_has_path_ph(c_node)

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
        self.allowed_capabilities = set(allowed_capabilities or set())
        warmup_sandbox_async()

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

    def _evaluate_verification_spec(
        self,
        verification_spec: Any,
        results: Dict[str, Any],
    ) -> List[str]:
        violations: List[str] = []
        if verification_spec is None:
            return violations

        post_reg = getattr(self, "postconditions", None) or PostconditionRegistry()

        def _eval_prop_op(prop_name: str, target_val: Any, op: str, expected_val: Any) -> bool:
            op_s = str(op or "==").strip()
            if op_s in ("==", "is"):
                return bool(post_reg.evaluate(prop_name, target_val, expected_val))
            if op_s in ("!=", "is not"):
                return not bool(post_reg.evaluate(prop_name, target_val, expected_val))
            # Resolve attribute or dict field for inequality / membership ops
            if hasattr(target_val, prop_name):
                attr_v = getattr(target_val, prop_name)
                actual = attr_v() if callable(attr_v) else attr_v
            elif isinstance(target_val, dict) and prop_name in target_val:
                actual = target_val[prop_name]
            else:
                actual = target_val
            if op_s == "<":
                return bool(actual < expected_val)
            if op_s == "<=":
                return bool(actual <= expected_val)
            if op_s == ">":
                return bool(actual > expected_val)
            if op_s == ">=":
                return bool(actual >= expected_val)
            if op_s == "in":
                return bool(actual in expected_val)
            if op_s == "not in":
                return bool(actual not in expected_val)
            raise ValueError(f"Unsupported verification operator: {op_s}")

        def _values_equal(v1: Any, v2: Any) -> bool:
            if v1 is v2:
                return True
            if hasattr(v1, "equals") and callable(v1.equals):
                return bool(v1.equals(v2))
            if type(v1).__name__ == "ndarray" or type(v2).__name__ == "ndarray":
                import numpy as _np
                return bool(_np.array_equal(v1, v2))
            cmp_res = (v1 == v2)
            if hasattr(cmp_res, "all") and callable(cmp_res.all):
                return bool(cmp_res.all())
            return bool(cmp_res)

        if isinstance(verification_spec, dict):
            cell_checks = list(verification_spec.get("cell_checks") or [])
            checks = list(verification_spec.get("terminal_checks") or [])
        else:
            cell_checks = list(getattr(verification_spec, "cell_checks", None) or [])
            checks = list(getattr(verification_spec, "terminal_checks", None) or [])

        for ccheck in cell_checks:
            if not isinstance(ccheck, dict):
                continue
            prop = ccheck.get("property") or ccheck.get("postcondition")
            if prop and str(prop).strip().lower() == "state":
                continue
            target_var = ccheck.get("target_var") or ccheck.get("var")
            cid = ccheck.get("cell_id", "unknown_cell")
            if not target_var or str(target_var) not in results:
                violations.append(
                    f"Cell check for '{cid}' missing target variable '{target_var}' in sandbox results."
                )
                continue
            if prop:
                expected = ccheck.get("value") if "value" in ccheck else ccheck.get("expected")
                op = ccheck.get("operator", "==")
                try:
                    ok = _eval_prop_op(str(prop), results[str(target_var)], op, expected)
                except Exception as eval_err:
                    ok = False
                    violations.append(
                        f"Cell check '{prop} {op} {expected}' on '{target_var}' ({cid}) failed evaluation: {eval_err}"
                    )
                    continue
                if not ok:
                    violations.append(
                        f"Cell check '{prop} {op} {expected}' on '{target_var}' ({cid}) was not satisfied."
                    )

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
                    elif isinstance(v_val, str) and v_val.startswith("<unpicklable:"):
                        continue
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
                                op = getattr(post, "operator", "==") if not isinstance(post, dict) else post.get("operator", "==")
                                try:
                                    ok = _eval_prop_op(str(prop), v_val, op, exp)
                                    if not ok:
                                        violations.append(f"Postcondition '{prop}' on deliverable '{v_name}' ({cid}) was not satisfied.")
                                except Exception as eval_err:
                                    violations.append(
                                        f"Postcondition '{prop}' on deliverable '{v_name}' ({cid}) failed evaluation: {eval_err}"
                                    )
                continue

            if check.get("type") == "algebraic_partition_coherence":
                if check.get("coherent") is False or (
                    check.get("expected_partition") is not None
                    and check.get("actual_partition") is not None
                    and check.get("expected_partition") != check.get("actual_partition")
                ):
                    violations.append(
                        check.get("violation")
                        or f"Partition coherence failed at '{check.get('cell_id', 'unknown')}' on port '{check.get('port', '?')}'."
                    )
                continue

            expected_var = check.get("expected_var")
            actual_var = check.get("actual_var")
            if expected_var and actual_var:
                if str(expected_var) not in results or str(actual_var) not in results:
                    violations.append(
                        f"Terminal check '{check.get('type', 'unknown')}' missing expected/actual variable "
                        f"('{expected_var}', '{actual_var}') in sandbox results."
                    )
                    continue
                expected_val = results[str(expected_var)]
                actual_val = results[str(actual_var)]
                if expected_val is None or actual_val is None:
                    violations.append(
                        f"Terminal check '{check.get('type', 'unknown')}' evaluated None for "
                        f"'{expected_var}' or '{actual_var}'."
                    )
                    continue
                try:
                    identical = _values_equal(expected_val, actual_val)
                except Exception:
                    identical = False
                if not identical:
                    violations.append(
                        f"Terminal check '{check.get('type', 'unknown')}' failed: "
                        f"'{actual_var}' does not carry the value declared for '{expected_var}'."
                    )
                exp_props = check.get("expected_properties")
                if exp_props:
                    for prop_name, prop_val in exp_props.items():
                        try:
                            ok = bool(post_reg.evaluate(prop_name, actual_val, prop_val))
                        except Exception as eval_err:
                            ok = False
                            violations.append(
                                f"Property '{prop_name}' on result '{actual_var}' failed evaluation: {eval_err}"
                            )
                            continue
                        if not ok:
                            violations.append(
                                f"Property '{prop_name}' on result '{actual_var}' was not satisfied."
                            )
                continue

            prop = check.get("property") or check.get("postcondition")
            target_var = check.get("target_var") or check.get("var")
            if prop and target_var:
                if str(target_var) not in results:
                    violations.append(
                        f"Postcondition '{prop}' missing target variable '{target_var}' in sandbox results."
                    )
                    continue
                expected = check.get("expected") if "expected" in check else check.get("value")
                op = check.get("operator", "==")
                try:
                    ok = _eval_prop_op(str(prop), results[str(target_var)], op, expected)
                except Exception as eval_err:
                    ok = False
                    violations.append(
                        f"Postcondition '{prop}' on result '{target_var}' failed evaluation: {eval_err}"
                    )
                    continue
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
        staged_inputs: List[str] = []
        egress_set: Set[str] = {str(p).strip() for p in (egress_paths or []) if p}

        try:
            try:
                from .language_backend import get_active_language_backend
            except (ImportError, ValueError):
                from language_backend import get_active_language_backend
            be_sb = get_active_language_backend()
            if getattr(be_sb, "spec", None) is not None and not be_sb.is_python_ast:
                ok_sb, err_sb = be_sb.validate_syntax(code)
                if not ok_sb:
                    return {
                        "success": False,
                        "skipped": False,
                        "status": "SYNTAX_ERROR",
                        "returncode": 1,
                        "stdout": "",
                        "stderr": str(err_sb or "Syntax validation failed"),
                        "error": str(err_sb or "Syntax validation failed"),
                        "error_kind": "syntax",
                        "extrinsic": False,
                    }
                return {
                    "success": False,
                    "skipped": True,
                    "status": "SKIPPED",
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "",
                    "error": f"SKIPPED: non-Python runtime '{be_sb.runtime}' sandbox execution not configured",
                    "error_kind": None,
                    "extrinsic": False,
                }
        except Exception:
            pass

        try:
            parsed_ast = ast.parse(code)

            # Incorporate any egress/sink bindings from pipeline_bindings
            try:
                from lattice import TypeRegistry
                _reg_sb = TypeRegistry.get_instance()
            except Exception:
                _reg_sb = None
            produced_vars: Set[str] = set()
            for b in kwargs.get("pipeline_bindings") or []:
                if isinstance(b, dict):
                    ov = b.get("output_var")
                    if isinstance(ov, str) and ov:
                        produced_vars.add(ov)
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    produced_vars.add(node.id)
            def _ast_root_ident(expr_str: str) -> str:
                try:
                    node = ast.parse(expr_str.strip(), mode="eval").body
                    while isinstance(node, (ast.Subscript, ast.Attribute)):
                        node = node.value
                    if isinstance(node, ast.Name):
                        return node.id
                except Exception:
                    pass
                return expr_str.strip()

            for b in kwargs.get("pipeline_bindings") or []:
                if isinstance(b, dict):
                    c_role = str(b.get("role", "") or b.get("node_role", "")).lower()
                    c_stage = b.get("stage")
                    if (_reg_sb is not None and _reg_sb.is_terminal_role(c_role)) or (c_stage is not None and c_stage >= 3):
                        for v in b.values():
                            if isinstance(v, str) and _ast_root_ident(v) not in produced_vars:
                                cln = v.strip("'\"")
                                if "." in cln:
                                    egress_set.add(cln)

            # Dynamically derive ingress vs egress callables and path parameters from loaded cells
            sink_callables, sink_kw_args, ingress_callables, ingress_kw_args = _derive_io_call_signatures()

            exec_chained_receivers: Set[int] = {
                id(n.func.value)
                for n in ast.walk(parsed_ast)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Call)
            }
            path_var_lits: Dict[str, List[str]] = {}
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call):
                    rhs_func = node.value.func
                    rhs_fn = rhs_func.attr if isinstance(rhs_func, ast.Attribute) else (rhs_func.id if isinstance(rhs_func, ast.Name) else "")
                    # Skip data loaders (e.g. pd.read_csv('input.csv')) whose output is an in-memory carrier, not a path handle
                    if rhs_fn and rhs_fn in ingress_callables and "path" not in rhs_fn.lower():
                        continue
                    c_lits = [a.value.strip() for a in node.value.args if isinstance(a, ast.Constant) and isinstance(a.value, str) and a.value.strip()]
                    c_lits.extend(kw.value.value.strip() for kw in node.value.keywords if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str) and kw.value.value.strip())
                    if c_lits:
                        path_var_lits[node.targets[0].id] = c_lits

            def _iter_call_chain_str_args(call_node: ast.Call, kw_filter: Optional[Set[str]]):
                cur: Optional[ast.AST] = call_node
                while isinstance(cur, ast.Call):
                    for arg in cur.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            yield arg.value.strip()
                    for kw in cur.keywords:
                        if (not kw_filter or kw.arg in kw_filter) and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            yield kw.value.value.strip()
                    if isinstance(cur.func, ast.Attribute):
                        if isinstance(cur.func.value, ast.Call):
                            cur = cur.func.value
                            continue
                        if isinstance(cur.func.value, ast.Name) and cur.func.value.id in path_var_lits:
                            for vs in path_var_lits[cur.func.value.id]:
                                yield vs
                    break

            # Pass 1: Track paths written by lattice-declared sink calls or open(..., "w"/"a"/"x")
            # (NEVER promote untrusted code-scraped paths into caller-trusted egress_set)
            code_written_paths: Set[str] = set()
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Call):
                    if id(node) in exec_chained_receivers:
                        continue
                    func = node.func
                    is_sink_call = False
                    if isinstance(func, ast.Name) and func.id == "open":
                        mode = None
                        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                            mode = str(node.args[1].value)
                        for kw in node.keywords:
                            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                                mode = str(kw.value.value)
                        if mode and any(m in mode for m in ("w", "a", "x")):
                            is_sink_call = True
                    elif isinstance(func, ast.Attribute) and func.attr in sink_callables:
                        is_sink_call = True
                    elif isinstance(func, ast.Name) and func.id in sink_callables:
                        is_sink_call = True

                    if is_sink_call:
                        if isinstance(func, ast.Name) and func.id == "open":
                            f_arg = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "file"), None)
                            if isinstance(f_arg, ast.Constant) and isinstance(f_arg.value, str) and f_arg.value.strip():
                                code_written_paths.add(f_arg.value.strip())
                        else:
                            for s_lit in _iter_call_chain_str_args(node, sink_kw_args or None):
                                if s_lit:
                                    code_written_paths.add(s_lit)

            # Pass 2: Check only ingress/loader calls for required input files
            for node in ast.walk(parsed_ast):
                if isinstance(node, ast.Call):
                    if id(node) in exec_chained_receivers:
                        continue
                    func = node.func
                    is_open_call = isinstance(func, ast.Name) and func.id == "open"
                    is_ingress_call = False
                    if is_open_call:
                        mode = "r"
                        if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                            mode = str(node.args[1].value)
                        for kw in node.keywords:
                            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                                mode = str(kw.value.value)
                        if "r" in mode and not any(m in mode for m in ("w", "a", "x")):
                            is_ingress_call = True
                    elif isinstance(func, ast.Attribute) and func.attr in ingress_callables:
                        is_ingress_call = True
                    elif isinstance(func, ast.Name) and func.id in ingress_callables:
                        is_ingress_call = True

                    def _inspect_ingress_str(raw_s: str, flag_missing: bool = True) -> None:
                        s = raw_s.strip()
                        if (
                            not s
                            or s in egress_set
                            or s in code_written_paths
                            or ".." in s
                            or os.path.isabs(s)
                            or "://" in s
                        ):
                            return
                        stem_part, raw_ext = os.path.splitext(s)
                        clean_ext = raw_ext.lstrip(".").lower()
                        if stem_part and not stem_part.endswith("/") and clean_ext and clean_ext.isalnum() and not clean_ext.isdigit() and len(clean_ext) <= 6:
                            if os.path.isfile(s) or os.path.isfile(os.path.abspath(s)):
                                if s not in staged_inputs:
                                    staged_inputs.append(s)
                            elif flag_missing and s not in missing_inputs:
                                missing_inputs.append(s)

                    if is_open_call:
                        if is_ingress_call:
                            f_arg = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "file"), None)
                            if isinstance(f_arg, ast.Constant) and isinstance(f_arg.value, str):
                                _inspect_ingress_str(f_arg.value, flag_missing=True)
                    elif is_ingress_call:
                        for s_lit in _iter_call_chain_str_args(node, ingress_kw_args):
                            _inspect_ingress_str(s_lit, flag_missing=True)
                    else:
                        for s_lit in _iter_call_chain_str_args(node, None):
                            _inspect_ingress_str(s_lit, flag_missing=False)
        except Exception as ast_err:
            logger.debug("[SANDBOX] AST inspection for input files failed: %s", ast_err)

        if missing_inputs:
            missing_file = missing_inputs[0]
            logger.info("[SANDBOX] Input file '%s' not found on disk. Skipping execution.", missing_file)
            return {
                "success": False,
                "skipped": True,
                "status": "SKIPPED",
                "returncode": -1,
                "stdout": "",
                "stderr": "",
                "error": f"SKIPPED: input '{missing_file}' not provided",
                "error_kind": None,
                "extrinsic": False,
                "results": {},
                "egress": {},
                "egress_bytes": {},
            }

        exec_timeout = timeout if timeout is not None else self.default_timeout
        context = context or {}
        ctx = _get_sandbox_mp_context()
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        headless_env_specs = _get_headless_env_specs()

        smoke_test = bool(kwargs.get("smoke_test", False))
        capture_egress = bool(kwargs.get("capture_egress", False))
        mem_limit = int(kwargs.get("max_memory_mb", self.max_memory_mb))
        cpu_limit = int(kwargs.get("max_cpu_seconds", max(int(exec_timeout) + 2, self.max_cpu_seconds)))
        allowed_caps = kwargs.get("allowed_capabilities")
        if allowed_caps is None:
            allowed_caps = getattr(self, "allowed_capabilities", set())
        if isinstance(allowed_caps, (set, list, tuple)):
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
                staged_inputs,
                headless_env_specs,
                capture_egress,
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
            "error_kind": None,
            "extrinsic": False,
            "results": {},
            "egress": {},
            "egress_bytes": {},
        }

        if timed_out:
            envelope["error"] = f"Execution timed out after {exec_timeout} seconds."
            envelope["error_kind"] = "timeout"
            envelope["returncode"] = worker.exitcode if worker.exitcode is not None else 1
            return envelope

        if payload is None:
            envelope["error"] = "Execution worker terminated abruptly without returning a result."
            envelope["error_kind"] = "runtime"
            envelope["extrinsic"] = True
            return envelope

        envelope["success"] = bool(payload.get("success", False))
        envelope["returncode"] = int(payload.get("returncode", 0) or 0)
        envelope["stdout"] = str(payload.get("stdout", "") or "")
        envelope["stderr"] = str(payload.get("stderr", "") or "")
        envelope["error"] = str(payload.get("error", "") or "")
        envelope["error_kind"] = payload.get("error_kind") or (None if envelope["success"] else "runtime")
        envelope["results"] = dict(payload.get("results", {}) or {})
        envelope["egress_bytes"] = dict(payload.get("egress_bytes") or {})

        if envelope["error"]:
            envelope["extrinsic"] = any(
                marker in envelope["error"] for marker in self._EXTRINSIC_ERROR_MARKERS
            )

        payload_egress = payload.get("egress") or {}
        if egress_paths:
            egress_verification = {}
            for raw_p in egress_paths:
                p_s = str(raw_p or "").strip()
                if not p_s:
                    continue
                if p_s in payload_egress:
                    egress_verification[p_s] = payload_egress[p_s]
                else:
                    egress_verification.update(self._verify_egress_paths([p_s]))
        else:
            egress_verification = {}
        envelope["egress"] = egress_verification
        if egress_verification:
            bad = {p: v for p, v in egress_verification.items() if v != "ok"}
            if bad:
                envelope["success"] = False
                if envelope["error_kind"] is None:
                    envelope["error_kind"] = "egress"
                detail = "; ".join(f"{p}: {v}" for p, v in sorted(bad.items()))
                envelope["error"] = (
                    (envelope["error"] + "\n" if envelope["error"] else "")
                    + f"Egress verification failed: {detail}"
                )
                envelope["extrinsic"] = True

        spec_violations = self._evaluate_verification_spec(verification_spec, envelope["results"])
        if spec_violations:
            envelope["success"] = False
            if envelope["error_kind"] is None:
                envelope["error_kind"] = "contract"
            envelope["error"] = (
                (envelope["error"] + "\n" if envelope["error"] else "")
                + "Verification contract violated: " + " | ".join(spec_violations)
            )

        return envelope
