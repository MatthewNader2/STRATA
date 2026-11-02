"""
src/library_adapters.py - STRATA (Neuro-Symbolic Topological Lattice)
Universal Domain-Agnostic Library Adapters.

Zero hardcoded library names, types, enums, or function names.
Adapters are selected strictly based on HOW the library is physically implemented:
  1. PythonSourceAdapter: Pure Python packages with accessible source code and AST.
  2. CompiledExtensionAdapter: Compiled C/C++/Rust binary extensions (.so/.pyd) using PEP 561 .pyi stubs and C introspection.
  3. HybridLibraryAdapter: Mixed packages containing both Python wrappers and compiled extensions.
"""

from __future__ import annotations

import abc
import ast
import enum
import functools
import importlib
import inspect
import keyword
import os
import pkgutil
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from utils import tokenize_alphanumeric
except ImportError:
    from .utils import tokenize_alphanumeric

try:
    from log_config import get_logger
    from signature_introspector import (
        resolve_signature,
        _find_stub_file_for_module,
        extract_clean_type_name,
        infer_abstract_carrier,
        extract_enum_domain,
        extract_return_specs,
        extract_param_kind,
        extract_docstring_params,
        extract_param_constraints,
        extract_shape_contract,
        extract_docstring_raises,
        extract_ast_raises,
        extract_type_vars,
        _tier2_text_signature,
    )
except ImportError:
    from .log_config import get_logger
    from .signature_introspector import (
        resolve_signature,
        _find_stub_file_for_module,
        extract_clean_type_name,
        infer_abstract_carrier,
        extract_enum_domain,
        extract_return_specs,
        extract_param_kind,
        extract_docstring_params,
        extract_param_constraints,
        extract_shape_contract,
        extract_docstring_raises,
        extract_ast_raises,
        extract_type_vars,
        _tier2_text_signature,
    )

logger = get_logger("library_adapters")


# =====================================================================
# Domain-Agnostic Configuration Helpers
# =====================================================================

def _get_type_registry():
    try:
        from .lattice import TypeRegistry
        return TypeRegistry.get_instance()
    except (ImportError, ValueError):
        try:
            from lattice import TypeRegistry
            return TypeRegistry.get_instance()
        except (ImportError, ValueError):
            return None


def _get_excluded_module_tokens() -> Set[str]:
    """Retrieves generic internal/test module tokens, dynamically extensible via TypeRegistry."""
    try:
        reg = _get_type_registry()
        if reg is not None:
            tokens = reg.get_excluded_module_tokens()
            if tokens:
                return set(tokens)
    except Exception:
        pass
    return {
        "test", "tests", "testing", "conftest", "testutils",
        "vendor", "vendored", "externals", "third_party",
        "compat", "internal", "internals",
    }


def _get_enum_citation_triggers() -> Set[str]:
    """Retrieves triggers indicating enum/choice citations in docstrings."""
    try:
        reg = _get_type_registry()
        if reg is not None:
            triggers = reg.get_enum_citation_triggers()
            if triggers:
                return set(triggers)
    except Exception:
        pass
    return {"see", "values in", "one of", "choices are", "must be one of", "refer to"}


def _get_enum_suffixes() -> Tuple[str, ...]:
    """Retrieves candidate suffixes for enum lookup stripping."""
    try:
        reg = _get_type_registry()
        if reg is not None:
            suffixes = reg.get_enum_suffixes()
            if suffixes:
                return tuple(suffixes)
    except Exception:
        pass
    return ("flags", "flag", "types", "type", "codes", "code", "mode", "modes", "enum", "enums")


def _get_symbol_stopwords() -> Set[str]:
    """Retrieves common linguistic and docstring noise tokens."""
    try:
        reg = _get_type_registry()
        if reg is not None:
            sw = reg.get_symbol_stopwords()
            if sw:
                return set(sw)
    except Exception:
        pass
    return set()


def _is_python_source_module(module: Any) -> bool:
    """True iff the module is backed by real Python source (.py), not a compiled binary."""
    file_path = getattr(module, "__file__", None) or ""
    if not file_path:
        return False
    return str(file_path).endswith(".py")


def _is_compiled_extension_module(module: Any) -> bool:
    """True iff the module is a compiled extension (.so, .pyd, or built-in)."""
    file_path = getattr(module, "__file__", None) or ""
    if not file_path:
        return True
    lower = str(file_path).lower()
    return lower.endswith(".so") or lower.endswith(".pyd") or ".cpython" in lower


def _is_module_excluded(name: str, package_name: str) -> bool:
    """Determines whether a module is an internal/test module using domain-agnostic patterns."""
    excluded_tokens = _get_excluded_module_tokens()
    parts = name.split(".")
    for p in parts:
        pl = p.lower()
        if pl.startswith("_") and pl != f"_{package_name}":
            return True
        if pl in excluded_tokens or pl.strip("_") in excluded_tokens:
            return True
        if pl.startswith("test_") or pl.endswith("_test"):
            return True
    return False


def _walk_package(root_module: Any) -> Dict[str, Any]:
    """Universal module enumeration for an importable package."""
    package_name = getattr(root_module, "__name__", "")
    modules: Dict[str, Any] = {package_name: root_module}
    pkg_path = getattr(root_module, "__path__", None)

    def _walk(path, prefix):
        try:
            for item in pkgutil.iter_modules(path, prefix):
                if _is_module_excluded(item.name, package_name):
                    continue
                if item.name.split(".")[-1].startswith("_") and item.name != f"_{package_name}":
                    continue
                try:
                    mod = importlib.import_module(item.name)
                    modules[item.name] = mod
                    sub_path = getattr(mod, "__path__", None)
                    if item.ispkg and sub_path:
                        _walk(sub_path, item.name + ".")
                except (Exception, BaseException):
                    continue
        except (Exception, BaseException):
            pass

    if pkg_path:
        _walk(pkg_path, package_name + ".")
    return modules


# =====================================================================
# Base Library Adapter Contract
# =====================================================================

class LibraryAdapter(abc.ABC):
    """
    Contract for library-specific extraction and introspection.
    Adapters adapt to HOW a library is physically constructed.
    """
    paradigm: str = "base"

    def __init__(self, root_module: Any = None):
        self.root_module = root_module

    def enumerate_modules(self, root_module: Any = None) -> Dict[str, Any]:
        return _walk_package(root_module or self.root_module)

    @abc.abstractmethod
    def get_source(self, obj: Any) -> Optional[str]:
        """Extract source code representation where available."""
        pass

    @abc.abstractmethod
    def get_docstring(self, obj: Any) -> Optional[str]:
        """Extract normalized documentation string."""
        pass

    @abc.abstractmethod
    def resolve_callable_signature(
        self,
        obj: Any,
        callable_name: str = "",
        parent_cls_name: str = "",
        mod: Any = None
    ) -> Optional[inspect.Signature]:
        """Resolve callable signature respecting paradigm capabilities."""
        pass

    @abc.abstractmethod
    def get_raises(self, doc: str = "", source: Optional[str] = None) -> List[str]:
        """Extract declared or AST-inferred exceptions."""
        pass

    def get_domain_carriers(self, root_module: Any = None) -> Set[str]:
        """Discovers domain carrier classes dynamically from the package namespace."""
        carriers: Set[str] = set()
        rm = root_module or self.root_module
        if rm:
            pkg_name = getattr(rm, "__name__", "")
            for m in self.enumerate_modules(rm).values():
                for attr in dir(m):
                    if not attr.startswith("_"):
                        val = getattr(m, attr, None)
                        if inspect.isclass(val):
                            cls_mod = getattr(val, "__module__", "") or ""
                            if cls_mod.startswith(pkg_name) or cls_mod.startswith(f"_{pkg_name}"):
                                carriers.add(val.__name__)
        return carriers

    def get_abstract_type(self, type_name_or_obj: Any, param_name: str = "") -> Optional[str]:
        return infer_abstract_carrier(type_name_or_obj, param_name=param_name)

    def get_parameter_domains(
        self,
        anno: Any,
        doc: str = "",
        param_name: str = "",
        mod: Any = None,
    ) -> Optional[List[Any]]:
        """Extracts enum domain choices for a parameter purely from annotations or docstring citations."""
        choices = extract_enum_domain(anno, doc, param_name)
        if choices:
            return choices

        target_mod = mod or self.root_module
        if doc and param_name and target_mod:
            p_desc = ""
            doc_params = extract_docstring_params(doc)
            if param_name in doc_params:
                p_desc = doc_params[param_name].get("desc", "")
            else:
                for line in doc.splitlines():
                    if param_name in line:
                        p_desc = line.strip()
                        break

            if p_desc:
                triggers = _get_enum_citation_triggers()
                stopwords = _get_symbol_stopwords()
                suffixes = _get_enum_suffixes()

                citations: List[str] = []
                p_desc_low = p_desc.lower()
                for trig in triggers:
                    if trig in p_desc_low:
                        idx = p_desc_low.find(trig) + len(trig)
                        after = p_desc[idx:].lstrip(" :#")
                        toks = tokenize_alphanumeric(after)
                        if toks:
                            citations.append(toks[0])

                for enum_ident in citations:
                    if not enum_ident.isidentifier():
                        continue
                    if len(enum_ident) < 2 or keyword.iskeyword(enum_ident.lower()) or enum_ident.lower() in stopwords:
                        continue

                    enum_obj = getattr(target_mod, enum_ident, None) or getattr(self.root_module, enum_ident, None)
                    if enum_obj is not None:
                        if inspect.isclass(enum_obj) and issubclass(enum_obj, enum.Enum):
                            return [m.name for m in enum_obj]
                        if inspect.isclass(enum_obj):
                            constants = [attr for attr in dir(enum_obj) if attr.isupper() and not attr.startswith("_")]
                            if constants:
                                return constants

                    clean_name = enum_ident
                    for suffix in suffixes:
                        if clean_name.lower().endswith(suffix):
                            clean_name = clean_name[:-len(suffix)]
                            break
                    pfx = f"{clean_name.upper()}_"
                    matches = [attr for attr in dir(target_mod) if attr.startswith(pfx)]
                    if matches:
                        return matches

        return None

    def get_return_ports(
        self,
        ret_anno: Any,
        doc: str = ""
    ) -> List[Tuple[str, str, Optional[str]]]:
        specs = extract_return_specs(ret_anno, doc)
        resolved = []
        for name, t_name, abs_t in specs:
            enriched_abs = self.get_abstract_type(t_name) or abs_t
            resolved.append((name, t_name, enriched_abs))
        return resolved

    def get_param_kind(self, param: inspect.Parameter) -> str:
        return extract_param_kind(param)

    def get_docstring_params(self, doc: str) -> Dict[str, Dict[str, Any]]:
        return extract_docstring_params(doc)

    def get_param_constraints(self, text: str) -> Optional[Dict[str, Any]]:
        return extract_param_constraints(text)

    def get_shape_contract(self, text: str) -> Optional[Dict[str, Any]]:
        return extract_shape_contract(text)

    def is_context_manager(self, target: Any) -> bool:
        if target is None:
            return False
        return hasattr(target, "__enter__") and hasattr(target, "__exit__")

    def get_type_vars(self, sig: inspect.Signature) -> List[str]:
        return extract_type_vars(sig)

    def is_ingress_source(
        self,
        fn: Any,
        name: str,
        input_names: List[str],
        produces_domain: bool,
        consumes_domain: bool = False,
    ) -> bool:
        if not produces_domain:
            return False
        return not consumes_domain

    def is_egress_sink(
        self,
        fn: Any,
        name: str,
        input_names: List[str],
        produces_domain: bool,
        consumes_domain: bool = True,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if produces_domain:
            return False

        if inputs:
            import io
            for p in inputs.values():
                p_abs = getattr(p, "abstract_type", None) or (p.get("abstract_type") if isinstance(p, dict) else None)
                if p_abs == "path":
                    return True
                p_type = getattr(p, "type_name", None) or (p.get("type_name") if isinstance(p, dict) else None)
                if p_type and isinstance(p_type, str):
                    try:
                        from signature_introspector import _resolve_type_from_string
                        resolved = _resolve_type_from_string(p_type)
                        if resolved is not None and isinstance(resolved, type):
                            if issubclass(resolved, (os.PathLike, io.IOBase)):
                                return True
                    except Exception:
                        pass

        if fn is not None:
            try:
                import io
                sig = self.resolve_callable_signature(fn, callable_name=name)
                if sig is not None:
                    for param in sig.parameters.values():
                        anno = param.annotation
                        if anno is not inspect.Parameter.empty and isinstance(anno, type):
                            if issubclass(anno, (os.PathLike, io.IOBase)):
                                return True
            except Exception:
                pass

        try:
            from lattice import TypeRegistry
            dest_tokens = TypeRegistry.get_instance().get_dest_port_tokens()
        except Exception:
            dest_tokens = frozenset()

        for p in input_names:
            components = {c for c in tokenize_alphanumeric(str(p).lower()) if c}
            if components & dest_tokens:
                return True
        return False

    def is_parameterized(
        self,
        fn: Any,
        name: str,
        inputs: Dict[str, Any]
    ) -> bool:
        return any(
            (getattr(inp, "type_name", "") == "Enum" or getattr(inp, "enum_values", None) is not None)
            for inp in inputs.values() if getattr(inp, "required", True)
        )

    def is_public_symbol(
        self,
        obj: Any,
        name: str,
        module_name: str,
        root_module: Any = None
    ) -> bool:
        if name.startswith("_") or name in ("TYPE_CHECKING",):
            return False

        excluded_tokens = _get_excluded_module_tokens()
        parts = module_name.split(".")
        is_internal_mod = any(
            p.startswith("_") or p.lower() in excluded_tokens or p.lower().strip("_") in excluded_tokens
            or p.startswith("test_") or p.endswith("_test")
            for p in parts if p != f"_{parts[0]}"
        )

        if is_internal_mod:
            pkg_prefix = ""
            for p in parts:
                if p.startswith("_") or p.lower() in excluded_tokens or p.lower().strip("_") in excluded_tokens:
                    break
                pkg_prefix = f"{pkg_prefix}.{p}" if pkg_prefix else p
                p_mod = sys.modules.get(pkg_prefix)
                if p_mod is not None and hasattr(p_mod, name) and getattr(p_mod, name) is obj:
                    return True
            return False

        mod = sys.modules.get(module_name)
        if mod and hasattr(mod, "__all__") and isinstance(mod.__all__, (list, tuple, set)) and len(mod.__all__) > 0:
            return name in mod.__all__
        return True

    def describe(self) -> Dict[str, Any]:
        return {"paradigm": self.paradigm}


# =====================================================================
# Implementation Paradigm: Python Source Library Adapter
# =====================================================================

class PythonSourceAdapter(LibraryAdapter):
    """
    Adapter for pure Python libraries where source code and AST are accessible.
    Leverages AST inspection for signatures, mutations, and internal call graphs.
    """
    paradigm = "python_source"

    def get_source(self, obj: Any) -> Optional[str]:
        try:
            target = inspect.unwrap(obj) if callable(obj) else obj
            src = inspect.getsource(target)
            return src if src and src.strip() else None
        except Exception:
            return None

    def get_ast(self, obj: Any) -> Optional[ast.AST]:
        """Parses the abstract syntax tree directly from source code."""
        src = self.get_source(obj)
        if not src:
            return None
        try:
            return ast.parse(src)
        except Exception:
            return None

    def get_docstring(self, obj: Any) -> Optional[str]:
        try:
            return inspect.getdoc(obj)
        except Exception:
            raw = getattr(obj, "__doc__", None)
            return str(raw) if raw else None

    def resolve_callable_signature(
        self,
        obj: Any,
        callable_name: str = "",
        parent_cls_name: str = "",
        mod: Any = None
    ) -> Optional[inspect.Signature]:
        target = inspect.unwrap(obj) if callable(obj) else obj
        return resolve_signature(
            target,
            callable_name=callable_name,
            parent_cls_name=parent_cls_name,
            mod=mod or self.root_module
        )

    def get_raises(self, doc: str = "", source: Optional[str] = None) -> List[str]:
        doc_raises = extract_docstring_raises(doc)
        ast_raises = extract_ast_raises(source)
        return list(dict.fromkeys(doc_raises + ast_raises))


# =====================================================================
# Implementation Paradigm: Compiled Binary Extension Adapter
# =====================================================================

class CompiledExtensionAdapter(LibraryAdapter):
    """
    Adapter for compiled C/C++/Rust binary extension libraries (.so, .pyd).
    Extracts signatures and types from PEP 561 .pyi type stubs and C docstrings,
    safely handling objects where inspect.getsource is unavailable.
    """
    paradigm = "compiled_binary"

    def get_source(self, obj: Any) -> Optional[str]:
        """
        Compiled binary extensions have no accessible Python AST source code.
        Safely returns None without triggering OSError/TypeError.
        """
        return None

    def get_docstring(self, obj: Any) -> Optional[str]:
        """Extracts and normalizes C-level docstrings, stripping Argument Clinic headers."""
        raw = getattr(obj, "__doc__", None)
        if not raw or not isinstance(raw, str):
            return None

        doc = raw.strip()
        # Clean Argument Clinic signature separators (e.g., 'func(a, b)\n--\n\nDoc...')
        if "\n--\n" in doc:
            parts = doc.split("\n--\n", 1)
            doc = parts[1].strip() if len(parts) > 1 else doc
        elif "\n---\n" in doc:
            parts = doc.split("\n---\n", 1)
            doc = parts[1].strip() if len(parts) > 1 else doc

        return doc or None

    def resolve_callable_signature(
        self,
        obj: Any,
        callable_name: str = "",
        parent_cls_name: str = "",
        mod: Any = None
    ) -> Optional[inspect.Signature]:
        target_mod = mod or self.root_module

        # 1. Primary mechanism for compiled extensions: PEP 561 Stub Introspection
        stub_sig = resolve_signature(
            obj,
            callable_name=callable_name,
            parent_cls_name=parent_cls_name,
            mod=target_mod
        )
        if stub_sig is not None:
            return stub_sig

        # 2. Introspect __text_signature__ if provided by Argument Clinic
        return _tier2_text_signature(obj)

    def get_raises(self, doc: str = "", source: Optional[str] = None) -> List[str]:
        # Compiled binaries rely strictly on docstring raises and type annotations
        return extract_docstring_raises(doc)


# =====================================================================
# Implementation Paradigm: Hybrid Library Adapter
# =====================================================================

class HybridLibraryAdapter(LibraryAdapter):
    """
    Adapter for mixed packages containing both Python wrapper modules
    and compiled binary extensions. Dynamically routes to the specialized
    adapter mechanism on a per-module and per-callable basis.
    """
    paradigm = "hybrid"

    def __init__(self, root_module: Any = None):
        super().__init__(root_module)
        self._python_adapter = PythonSourceAdapter(root_module)
        self._compiled_adapter = CompiledExtensionAdapter(root_module)

    def _select_adapter(self, obj: Any = None, mod: Any = None) -> LibraryAdapter:
        """Determines whether to delegate to PythonSourceAdapter or CompiledExtensionAdapter."""
        target_mod = mod
        if target_mod is None and obj is not None:
            mod_name = getattr(obj, "__module__", None)
            if mod_name and mod_name in sys.modules:
                target_mod = sys.modules[mod_name]

        if target_mod is not None:
            if _is_compiled_extension_module(target_mod):
                return self._compiled_adapter
            if _is_python_source_module(target_mod):
                return self._python_adapter

        if obj is not None:
            if inspect.isbuiltin(obj) or isinstance(obj, types.BuiltinFunctionType):
                return self._compiled_adapter

        return self._python_adapter

    def get_source(self, obj: Any) -> Optional[str]:
        return self._select_adapter(obj=obj).get_source(obj)

    def get_docstring(self, obj: Any) -> Optional[str]:
        return self._select_adapter(obj=obj).get_docstring(obj)

    def resolve_callable_signature(
        self,
        obj: Any,
        callable_name: str = "",
        parent_cls_name: str = "",
        mod: Any = None
    ) -> Optional[inspect.Signature]:
        return self._select_adapter(obj=obj, mod=mod).resolve_callable_signature(
            obj,
            callable_name=callable_name,
            parent_cls_name=parent_cls_name,
            mod=mod
        )

    def get_raises(self, doc: str = "", source: Optional[str] = None) -> List[str]:
        if source:
            return self._python_adapter.get_raises(doc=doc, source=source)
        return self._compiled_adapter.get_raises(doc=doc, source=source)


# =====================================================================
# Universal Dynamic Factory (Zero Hardcoded Package Names)
# =====================================================================

_adapter_cache: Dict[str, LibraryAdapter] = {}


def get_adapter_for_package(package_name: str) -> LibraryAdapter:
    """
    Dynamically analyzes the physical implementation of any package
    and returns the corresponding specialized paradigm adapter.
    Contains ZERO hardcoded package names.
    """
    if package_name in _adapter_cache:
        return _adapter_cache[package_name]

    try:
        root_module = importlib.import_module(package_name)
    except ImportError:
        cwd = str(Path.cwd())
        if cwd not in sys.path:
            sys.path.insert(0, cwd)
        root_module = importlib.import_module(package_name)

    modules = _walk_package(root_module)
    py_count = 0
    compiled_count = 0

    for m in modules.values():
        if _is_python_source_module(m):
            py_count += 1
        elif _is_compiled_extension_module(m):
            compiled_count += 1

    total = py_count + compiled_count
    if total == 0 or (py_count > 0 and compiled_count == 0):
        adapter = PythonSourceAdapter(root_module)
    elif compiled_count > 0 and py_count == 0:
        adapter = CompiledExtensionAdapter(root_module)
    else:
        adapter = HybridLibraryAdapter(root_module)

    logger.info(
        f"[ADAPTER] Selected '{adapter.paradigm}' adapter for package '{package_name}' "
        f"(Python modules: {py_count}, Compiled modules: {compiled_count})"
    )
    _adapter_cache[package_name] = adapter
    return adapter
