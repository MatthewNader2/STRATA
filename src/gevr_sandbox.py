"""
src/gevr_sandbox.py — Hardened, domain-agnostic sandboxed execution environment.

Enforces AST security inspection (SecurityPolicy), POSIX RLIMIT_AS/DATA/CPU bounds,
network socket isolation, ephemeral filesystem confinement, declarative domain
`runtime_setup`, and explicit caller-declared input asset staging.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import importlib
import io as _io
import logging
import multiprocessing
import os
import pickle as _pickle
import shutil
import signal
import socket as _real_socket
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

try:
    import resource as _resource
except ImportError:  # pragma: no cover
    _resource = None  # type: ignore[assignment]

try:
    from .config import settings
    from .language_backend import get_active_language_backend
    from .security_policy import (
        DEFAULT_CAPABILITY_MODULES as _SP_CAP_MODS,
        NON_OVERRIDABLE_BLOCKED_MODULES as _SP_BLOCKED_MODS,
        NON_OVERRIDABLE_BLOCKED_OS_OPS as _SP_BLOCKED_OS_OPS,
        SecurityPolicy,
        SecurityViolationError,
        _SecurityVisitor as _UnifiedSecurityASTVisitor,
    )
except (ImportError, ValueError):
    from config import settings
    from language_backend import get_active_language_backend
    from security_policy import (
        DEFAULT_CAPABILITY_MODULES as _SP_CAP_MODS,
        NON_OVERRIDABLE_BLOCKED_MODULES as _SP_BLOCKED_MODS,
        NON_OVERRIDABLE_BLOCKED_OS_OPS as _SP_BLOCKED_OS_OPS,
        SecurityPolicy,
        SecurityViolationError,
        _SecurityVisitor as _UnifiedSecurityASTVisitor,
    )

logger = logging.getLogger(__name__)
_BLOCKED_MODULES = tuple(dict.fromkeys((*_SP_BLOCKED_MODS, *_SP_CAP_MODS.keys())))


class SandboxTimeoutError(TimeoutError):
    """Raised when execution exceeds the allocated wall-clock time limit."""


class _SecurityASTVisitor(_UnifiedSecurityASTVisitor):
    """Unified AST security visitor backed by security_policy._SecurityVisitor."""

    def __init__(self, policy: Optional[Any] = None) -> None:
        super().__init__(policy=policy)


def _enforce_posix_resource_limits(max_memory_mb: int = 8192, max_cpu_seconds: int = 30) -> None:
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
        if hasattr(_resource, "RLIMIT_AS"):
            _, hard_as = _resource.getrlimit(_resource.RLIMIT_AS)
            target_as = mem_bytes if hard_as == _resource.RLIM_INFINITY else min(mem_bytes, hard_as)
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
    if allow_network:
        return

    def _blocked_socket(*args: Any, **kwargs: Any) -> Any:
        raise SecurityViolationError(
            "Network socket creation is prohibited in the sandbox unless the 'net' capability is granted."
        )

    _real_socket.socket = _blocked_socket  # type: ignore[misc,assignment]
    _real_socket.create_connection = _blocked_socket  # type: ignore[assignment]


def _create_confined_open(
    allowed_read_roots: Set[str],
    allowed_write_roots: Set[str],
    allowed_write_files: Set[str],
) -> Callable[..., Any]:
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
                    f"Sandbox write access denied for path '{raw_str}'."
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
    import os as _real_os
    _real_path = _real_os.path

    class _SanitizedOSPath:
        def __getattr__(self, name: str) -> Any:
            if name not in _SAFE_OS_PATH_ATTRS:
                raise SecurityViolationError(f"Access to restricted attribute 'os.path.{name}' is prohibited in sandbox.")
            return getattr(_real_path, name)

    return _SanitizedOSPath()


def _create_sanitized_os():
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
                    raise SecurityViolationError(f"Importing '{item}' from '{name}' is prohibited in sandbox.")
            return _create_sanitized_os_path() if fromlist else _create_sanitized_os()

        if root_module == "os":
            for item in fromlist or ():
                if item == "*" or item in _SP_BLOCKED_OS_OPS or item in {"sys", "builtins"}:
                    raise SecurityViolationError(f"Importing '{item}' from 'os' is prohibited in sandbox.")
            return _create_sanitized_os()

        return builtins.__import__(name, globals, locals, fromlist, level)

    return _restricted_import


class ASTLoopBounder(ast.NodeTransformer):
    """Bound unbounded while loops during smoke-test verification."""

    def __init__(self, max_iters: int = 5) -> None:
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


_SANDBOX_MP_CTX: Any = None
_SANDBOX_CTX_LOCK = threading.Lock()
_SANDBOX_WARMUP_STARTED = False


def _get_sandbox_mp_context() -> Any:
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
                _SANDBOX_MP_CTX = multiprocessing.get_context("forkserver")
                return _SANDBOX_MP_CTX
            except Exception as exc:
                logger.debug("[SANDBOX] forkserver init fallback to spawn: %s", exc)
        _SANDBOX_MP_CTX = multiprocessing.get_context("spawn")
        return _SANDBOX_MP_CTX


def warmup_sandbox_async() -> None:
    global _SANDBOX_WARMUP_STARTED
    if os.environ.get("NSTL_SANDBOX_WARMUP", "1") == "0":
        return
    with _SANDBOX_CTX_LOCK:
        if _SANDBOX_WARMUP_STARTED:
            return
        _SANDBOX_WARMUP_STARTED = True

    def _warm() -> None:
        try:
            ctx = _get_sandbox_mp_context()
            if getattr(ctx, "_name", "") == "forkserver":
                import multiprocessing.forkserver as _fs
                _fs.ensure_running()
        except Exception:
            pass

    threading.Thread(target=_warm, name="sandbox-forkserver-warmup", daemon=True).start()


def _apply_declarative_runtime_setups(
    parsed_ast: ast.AST,
    runtime_setups: Optional[List[Dict[str, Any]]],
    sec_policy: Any,
) -> None:
    """Apply domain-declared environment variables and guarded pre-exec hooks."""
    if not runtime_setups:
        return
    imported_roots: Set[str] = set()
    for node in ast.walk(parsed_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    blocked_roots = set(_BLOCKED_MODULES) | {"os", "nt", "posix", "io", "_io"}
    for setup in runtime_setups:
        if not isinstance(setup, dict):
            continue
        env_map = setup.get("env_vars") or {}
        if isinstance(env_map, dict):
            for k, v in env_map.items():
                ks = str(k or "").strip()
                if ks:
                    os.environ.setdefault(ks, str(v))
        for hook_spec in (setup.get("pre_exec_hooks") or ()):
            parts = str(hook_spec or "").strip().split(".")
            if len(parts) < 2 or not all(p.isidentifier() and not p.startswith("_") for p in parts):
                raise SecurityViolationError(f"Invalid pre_exec_hook specification '{hook_spec}'.")
            root_mod = parts[0]
            if root_mod in blocked_roots:
                raise SecurityViolationError(f"Blocked module '{root_mod}' in declarative pre_exec_hook '{hook_spec}'.")
            if sec_policy is not None and hasattr(sec_policy, "is_module_allowed"):
                allowed, reason = sec_policy.is_module_allowed(".".join(parts[:-1]))
                if not allowed:
                    raise SecurityViolationError(f"Declarative pre_exec_hook '{hook_spec}' rejected: {reason}")
            if root_mod not in imported_roots:
                continue
            mod_obj = importlib.import_module(".".join(parts[:-1]))
            fn = getattr(mod_obj, parts[-1], None)
            if callable(fn):
                fn()


def _sandbox_worker_exec(
    code_str: str,
    context: Dict[str, Any],
    runtime_aliases: Dict[str, str],
    result_conn: Any,
    smoke_test: bool = False,
    egress_paths: Optional[List[str]] = None,
    max_memory_mb: int = 8192,
    max_cpu_seconds: int = 30,
    allowed_capabilities: Optional[List[str]] = None,
    ingress_paths: Optional[List[str]] = None,
    capture_egress: bool = False,
    max_egress_bytes: int = 4 * 1024 * 1024,
    runtime_setups: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Subprocess worker executing code under AST, RLIMIT, socket, and FS confinement."""
    allowed_caps_set = set(allowed_capabilities or [])
    if "NSTL_ALLOWED_CAPABILITIES" in os.environ:
        allowed_caps_set.update(c.strip() for c in os.environ["NSTL_ALLOWED_CAPABILITIES"].split(",") if c.strip())

    try:
        sec_policy: Optional[SecurityPolicy] = SecurityPolicy(allowed_capabilities=allowed_caps_set)
    except Exception:
        sec_policy = None

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
            parsed_ast = ast.parse(code_str)
            if sec_policy is not None:
                sec_policy.validate_ast(parsed_ast)
            else:
                _SecurityASTVisitor().visit(parsed_ast)

            _apply_declarative_runtime_setups(parsed_ast, runtime_setups, sec_policy)

            if smoke_test:
                loop_limit = int(settings.smoke_test_loop_bound) if settings else 5
                parsed_ast = ASTLoopBounder(max_iters=loop_limit).visit(parsed_ast)
                ast.fix_missing_locations(parsed_ast)

            _enforce_posix_resource_limits(max_memory_mb=max_memory_mb, max_cpu_seconds=max_cpu_seconds)
            allow_net = bool(allowed_caps_set & {"net", "net.http", "net.tcp", "net.all"})
            _disable_network_sockets(allow_network=allow_net)

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
            for asset_item in ingress_paths or ():
                if isinstance(asset_item, (tuple, list)) and len(asset_item) == 2:
                    src_abs, rel_raw = asset_item
                    rel_str = str(rel_raw or "").strip()
                    if not rel_str or os.path.isabs(rel_str):
                        continue
                    src_p = Path(str(src_abs)).resolve()
                else:
                    rel_str = str(asset_item or "").strip()
                    if not rel_str or os.path.isabs(rel_str):
                        continue
                    src_p = (Path(orig_cwd) / rel_str).resolve()
                dst_p = (ephemeral_root_p / rel_str).resolve()
                if not src_p.is_file() or not dst_p.is_relative_to(ephemeral_root_p):
                    continue
                dst_p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_p), str(dst_p))
            os.chdir(ephemeral_dir)

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

            safe_builtins: Dict[str, Any] = {
                k: v for k, v in builtins.__dict__.items()
                if k not in _SecurityASTVisitor.FORBIDDEN_CALLS
            }
            safe_builtins["__import__"] = _create_restricted_import(sec_policy)
            safe_builtins["open"] = confined_open

            exec_globals: Dict[str, Any] = {
                "__builtins__": safe_builtins,
                "__name__": "__sandbox__",
                "__doc__": None,
            }
            exec_globals.update(context)

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

            raw_results = {k: v for k, v in exec_globals.items() if not k.startswith("__") and k not in context}
            for alias_name, source_var in (runtime_aliases or {}).items():
                if source_var in raw_results:
                    raw_results[alias_name] = raw_results[source_var]

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
                "error": f"SecurityViolation: {sec_err}",
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
            _send_payload({
                "success": False,
                "returncode": 1,
                "stdout": captured_stdout.getvalue(),
                "stderr": captured_stderr.getvalue(),
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
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


class GEVRSandbox:
    """
    Hardened execution sandbox with strict process isolation, POSIX resource limits,
    network/filesystem confinement, timeout enforcement, and declarative runtime setups.
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
        allowed_capabilities: Optional[Set[str]] = None,
        asset_dirs: Optional[List[Union[str, Path]]] = None,
    ) -> None:
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
        self.allowed_capabilities = set(allowed_capabilities or set())
        self.asset_dirs: List[Path] = []
        for raw_dir in (asset_dirs or ()):
            try:
                resolved_dir = Path(raw_dir).resolve()
                if resolved_dir not in self.asset_dirs:
                    self.asset_dirs.append(resolved_dir)
            except Exception:
                pass
        warmup_sandbox_async()

    def execute(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        input_assets: Optional[List[str]] = None,
        egress_paths: Optional[List[str]] = None,
        runtime_aliases: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute code in a resource- and filesystem-confined worker process."""
        be_sb = get_active_language_backend()
        if getattr(be_sb, "spec", None) is not None and not be_sb.is_python_ast:
            ok_sb, err_sb = be_sb.validate_syntax(code, allow_placeholders=False)
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

        egress_set: Set[str] = {str(p).strip() for p in (egress_paths or []) if p}
        staged_inputs: List[Union[str, Tuple[str, str]]] = []
        seen_input_names: Set[str] = set()
        missing_inputs: List[str] = []

        for raw_asset in (input_assets or ()):
            s = str(raw_asset or "").strip()
            if not s or "://" in s or os.path.isabs(s) or s in egress_set:
                continue
            rel_parts = Path(s).parts
            if ".." in rel_parts:
                continue
            if os.path.isfile(s) or os.path.isfile(os.path.abspath(s)):
                if s not in seen_input_names:
                    seen_input_names.add(s)
                    staged_inputs.append(s)
                continue
            matched_src: Optional[Path] = None
            for asset_dir in self.asset_dirs:
                cand = (asset_dir / s).resolve()
                if cand.is_file() and cand.is_relative_to(asset_dir):
                    matched_src = cand
                    break
            if matched_src is not None:
                if s not in seen_input_names:
                    seen_input_names.add(s)
                    staged_inputs.append((str(matched_src), s))
            elif s not in missing_inputs:
                missing_inputs.append(s)

        if missing_inputs:
            missing_file = missing_inputs[0]
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

        smoke_test = bool(kwargs.get("smoke_test", False))
        capture_egress = bool(kwargs.get("capture_egress", False))
        max_egress_bytes = int(kwargs.get("max_egress_bytes", 4 * 1024 * 1024))
        runtime_setups = list(kwargs["runtime_setups"]) if kwargs.get("runtime_setups") is not None else None
        mem_limit = int(kwargs.get("max_memory_mb", self.max_memory_mb))
        cpu_limit = int(kwargs.get("max_cpu_seconds", max(int(exec_timeout) + 2, self.max_cpu_seconds)))
        allowed_caps = kwargs.get("allowed_capabilities")
        if allowed_caps is None:
            allowed_caps = self.allowed_capabilities
        allowed_caps_list = list(allowed_caps) if isinstance(allowed_caps, (set, list, tuple)) else []
        if not allowed_caps_list and "NSTL_ALLOWED_CAPABILITIES" in os.environ:
            allowed_caps_list = [c.strip() for c in os.environ["NSTL_ALLOWED_CAPABILITIES"].split(",") if c.strip()]

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
                allowed_caps_list,
                staged_inputs,
                capture_egress,
                max_egress_bytes,
                runtime_setups,
            ),
            daemon=True,
        )

        worker.start()
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
            timed_out = True
            worker.terminate()
            worker.join(timeout=0.5)
            if worker.is_alive():
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
            envelope["extrinsic"] = any(m in envelope["error"] for m in self._EXTRINSIC_ERROR_MARKERS)

        payload_egress = payload.get("egress") or {}
        if egress_paths:
            egress_verification = {
                str(raw_p).strip(): payload_egress.get(str(raw_p).strip(), "missing")
                for raw_p in egress_paths
                if str(raw_p or "").strip()
            }
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
                envelope["error"] = (envelope["error"] + "\n" if envelope["error"] else "") + f"Egress verification failed: {detail}"
                envelope["extrinsic"] = True

        return envelope
