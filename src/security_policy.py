"""
src/security_policy.py
Non-overridable engine security floor and capability-based access control.
Enforces default-deny for capability-gated modules, with explicit opt-in
via --allow-capabilities. Zero tolerance for low-level OS bypasses.
"""

from __future__ import annotations

import ast
from typing import Dict, Optional, Set, Tuple


class SecurityViolationError(PermissionError):
    """Raised when synthesized code violates sandbox security constraints."""
    pass


# 1. Minimal, Non-Overridable Security Floor
# No tree, capability, or user flag is permitted to override these restrictions.
NON_OVERRIDABLE_BLOCKED_MODULES: Tuple[str, ...] = (
    "ctypes",
    "_ctypes",
    "pty",
    "subprocess",
    "_posixsubprocess",
    "posix",
    "code",
    "codeop",
    "shutil",
    "importlib",
    "pickle",
    "_pickle",
    "marshal",
    "shelve",
    "multiprocessing",
    "sys",
    "builtins",
    "_socket",
    "nt",
    "io",
    "_io",
    "gc",
    "inspect",
    "traceback",
)

NON_OVERRIDABLE_BLOCKED_ATTRS: Tuple[str, ...] = (
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
    "__class__",
    "__dict__",
    "__module__",
    "__self__",
    "__func__",
    "__wrapped__",
    "__qualname__",
    "__reduce__",
    "__reduce_ex__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__init_subclass__",
    "gi_frame",
    "gi_code",
    "f_locals",
    "f_globals",
    "f_back",
    "f_builtins",
    "f_code",
    "cr_frame",
    "cr_code",
    "ag_frame",
    "tb_frame",
    "tb_next",
)

NON_OVERRIDABLE_BLOCKED_CALLS: Tuple[str, ...] = (
    "eval",
    "exec",
    "compile",
    "breakpoint",
    "input",
    "__import__",
    "vars",
    "globals",
    "locals",
)

NON_OVERRIDABLE_BLOCKED_OS_OPS: Tuple[str, ...] = (
    "system", "popen", "spawn", "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe", "execv", "execve", "execvp",
    "execvpe", "execl", "execle", "execlp", "execlpe", "remove", "unlink",
    "rmdir", "removedirs", "chmod", "chown", "kill", "killpg", "fork", "forkpty",
)

# 2. Standard Default Capability Gating
# Mapping of standard modules to their capability categories.
DEFAULT_CAPABILITY_MODULES: Dict[str, str] = {
    "socket": "net",
    "ssl": "net",
    "urllib": "net.http",
    "urllib3": "net.http",
    "http": "net.http",
    "ftplib": "net.ftp",
    "smtplib": "net.smtp",
    "telnetlib": "net.telnet",
    "xmlrpc": "net.http",
    "asyncio": "async",
}


class SecurityPolicy:
    """
    Enforces the security floor and evaluates tree-declared capabilities.
    Default-deny: capability-gated modules require explicit authorization.
    """

    def __init__(
        self,
        allowed_capabilities: Optional[Set[str]] = None,
        tree_capabilities: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        self.allowed_capabilities: Set[str] = set(allowed_capabilities or set())
        # Mapping from module name to required capability set
        self.module_to_capabilities: Dict[str, Set[str]] = {}

        # Seed with default capabilities
        for mod, cap in DEFAULT_CAPABILITY_MODULES.items():
            self.module_to_capabilities.setdefault(mod, set()).add(cap)

        # Merge tree-declared capabilities
        if tree_capabilities:
            for cap_name, modules in tree_capabilities.items():
                for mod in modules:
                    self.module_to_capabilities.setdefault(mod, set()).add(cap_name)

    def is_module_allowed(self, module_name: str) -> Tuple[bool, Optional[str]]:
        """
        Check if importing module_name is permitted.
        Returns (is_allowed, reason_if_denied).
        """
        root_mod = module_name.split(".")[0]

        # 1. Check non-overridable security floor
        if root_mod in NON_OVERRIDABLE_BLOCKED_MODULES or module_name in NON_OVERRIDABLE_BLOCKED_MODULES:
            return False, f"Module '{module_name}' is blocked by the non-overridable engine security floor."

        # 2. Check capability gating
        req_caps = self.module_to_capabilities.get(root_mod) or self.module_to_capabilities.get(module_name)
        if req_caps:
            # Check if any required capability is allowed
            if not any(cap in self.allowed_capabilities for cap in req_caps):
                caps_str = ", ".join(sorted(req_caps))
                return False, f"Module '{module_name}' requires capability [{caps_str}] which was not granted via --allow-capabilities."

        return True, None

    def validate_ast(self, tree: ast.AST) -> None:
        """
        Perform static AST security inspection on synthesized code.
        Raises SecurityViolationError if any violation is detected.
        """
        visitor = _SecurityVisitor(self)
        visitor.visit(tree)


class _SecurityVisitor(ast.NodeVisitor):
    """
    Static analysis pass over synthesized code before execution.
    Detects forbidden imports, dunder traversal exploits, dynamic string
    obfuscation (BinOp/JoinedStr/Subscript/getattr), and forbidden builtins.
    """

    FORBIDDEN_ATTRIBUTES: Tuple[str, ...] = NON_OVERRIDABLE_BLOCKED_ATTRS
    FORBIDDEN_CALLS: Tuple[str, ...] = NON_OVERRIDABLE_BLOCKED_CALLS

    def __init__(self, policy: Optional[SecurityPolicy] = None) -> None:
        self.policy = policy

    def visit_Import(self, node: ast.Import) -> None:
        if self.policy is not None:
            for alias in node.names:
                allowed, reason = self.policy.is_module_allowed(alias.name)
                if not allowed:
                    raise SecurityViolationError(f"Line {node.lineno}: {reason}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.policy is not None and node.module:
            allowed, reason = self.policy.is_module_allowed(node.module)
            if not allowed:
                raise SecurityViolationError(f"Line {node.lineno}: {reason}")
        if node.module == "os.path":
            for alias in node.names:
                if alias.name in {"os", "sys", "genericpath", "posixpath", "ntpath", "*"}:
                    raise SecurityViolationError(
                        f"Line {node.lineno}: Importing '{alias.name}' from 'os.path' is blocked."
                    )
        elif node.module == "os":
            for alias in node.names:
                if alias.name in NON_OVERRIDABLE_BLOCKED_OS_OPS or alias.name == "*":
                    raise SecurityViolationError(
                        f"Line {node.lineno}: Importing dangerous operation '{alias.name}' from 'os' is blocked."
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in NON_OVERRIDABLE_BLOCKED_ATTRS:
            raise SecurityViolationError(
                f"Line {node.lineno}: Access to restricted meta-attribute '{node.attr}' is forbidden."
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and (
            node.id in NON_OVERRIDABLE_BLOCKED_CALLS or node.id == "__builtins__"
        ):
            raise SecurityViolationError(
                f"Line {node.lineno}: Direct access to forbidden built-in '{node.id}' is blocked."
            )
        self.generic_visit(node)

    @staticmethod
    def _evaluate_constant_str(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = _SecurityVisitor._evaluate_constant_str(node.left)
            right = _SecurityVisitor._evaluate_constant_str(node.right)
            if left is not None and right is not None:
                return left + right
        if isinstance(node, ast.JoinedStr):
            parts = []
            for val in node.values:
                s = _SecurityVisitor._evaluate_constant_str(val)
                if s is None:
                    return None
                parts.append(s)
            return "".join(parts)
        if isinstance(node, ast.FormattedValue):
            return _SecurityVisitor._evaluate_constant_str(node.value)
        return None

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Add):
            concat_str = self._evaluate_constant_str(node)
            if concat_str is not None:
                if concat_str in NON_OVERRIDABLE_BLOCKED_ATTRS or (
                    concat_str.startswith("__") and concat_str.endswith("__")
                ):
                    raise SecurityViolationError(
                        f"Line {getattr(node, 'lineno', 0)}: Dynamic concatenation producing forbidden dunder attribute '{concat_str}' is blocked."
                    )
            for side in (node.left, node.right):
                side_str = self._evaluate_constant_str(side)
                if side_str is not None and "__" in side_str:
                    raise SecurityViolationError(
                        f"Line {getattr(node, 'lineno', 0)}: Dynamic dunder string concatenation involving '{side_str}' is blocked."
                    )
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        eval_str = self._evaluate_constant_str(node)
        if eval_str is not None and (
            eval_str in NON_OVERRIDABLE_BLOCKED_ATTRS
            or (eval_str.startswith("__") and eval_str.endswith("__"))
        ):
            raise SecurityViolationError(
                f"Line {getattr(node, 'lineno', 0)}: Dynamic string formatting producing forbidden dunder attribute '{eval_str}' is blocked."
            )
        for val in node.values:
            s = self._evaluate_constant_str(val)
            if s is not None and "__" in s:
                raise SecurityViolationError(
                    f"Line {getattr(node, 'lineno', 0)}: Dynamic dunder string formatting involving '{s}' is blocked."
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in NON_OVERRIDABLE_BLOCKED_CALLS:
                raise SecurityViolationError(
                    f"Line {node.lineno}: Direct access to forbidden built-in '{node.func.id}' is blocked."
                )
            if node.func.id in ("getattr", "hasattr", "setattr", "delattr") and len(node.args) >= 2:
                arg_str = self._evaluate_constant_str(node.args[1])
                if arg_str is None:
                    raise SecurityViolationError(
                        f"Line {node.lineno}: Dynamic non-constant attribute name in {node.func.id} is blocked."
                    )
                if arg_str in NON_OVERRIDABLE_BLOCKED_ATTRS or (
                    arg_str.startswith("__") and arg_str.endswith("__")
                ):
                    raise SecurityViolationError(
                        f"Line {node.lineno}: Dynamic access to forbidden dunder attribute '{arg_str}' via {node.func.id} is blocked."
                    )
            if node.func.id == "type" and (len(node.args) >= 3 or len(node.keywords) > 0):
                raise SecurityViolationError(
                    f"Line {node.lineno}: Dynamic class construction via 3-arg type() is blocked."
                )
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        slice_str = self._evaluate_constant_str(node.slice)
        if slice_str and (
            slice_str in NON_OVERRIDABLE_BLOCKED_ATTRS
            or (slice_str.startswith("__") and slice_str.endswith("__"))
        ):
            raise SecurityViolationError(
                f"Line {getattr(node, 'lineno', 0)}: Subscription access to forbidden meta-attribute '{slice_str}' is blocked."
            )
        self.generic_visit(node)
