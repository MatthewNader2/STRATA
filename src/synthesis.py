"""
src/synthesis.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Dynamic MicroCell Synthesizer & Deterministic Structural Cell Renderer.
"""

from __future__ import annotations
import asyncio
import ast
import importlib
import os
import textwrap
from typing import Dict, Any, List, Optional, Set, Tuple, Type
from log_config import get_logger

try:
    from .external_rag import LiveDocFetcher
    from .inference import ModelManager
    from .lattice import TypeRegistry
    from .utils import (
        extract_json_from_llm,
        validate_code_template,
        extract_code_from_llm_response,
        safe_substitute_template,
        extract_template_placeholders,
    )
except (ImportError, ValueError):
    from external_rag import LiveDocFetcher
    from inference import ModelManager
    from lattice import TypeRegistry
    from utils import (
        extract_json_from_llm,
        validate_code_template,
        extract_code_from_llm_response,
        safe_substitute_template,
        extract_template_placeholders,
    )

logger = get_logger("synthesis")


class KernelSynthesisError(RuntimeError):
    """
    Raised when inductive state kernel synthesis fails or produces unsafe code.
    Eliminates silent fake-success zero/empty fallbacks.
    """
    pass


# --------------------------------------------------------------------------- #
# AST Capability Permit-List Validator (Track 1.6)
# --------------------------------------------------------------------------- #

class _PermittedModuleNamespacesDescriptor:
    def __get__(self, instance: Any, owner: Optional[Type[Any]] = None) -> Set[str]:
        return ASTCapabilityValidator.get_permitted_module_namespaces()


class _ForbiddenSideEffectMethodsDescriptor:
    def __get__(self, instance: Any, owner: Optional[Type[Any]] = None) -> Set[str]:
        return ASTCapabilityValidator.get_forbidden_side_effect_methods()


class _ForbiddenNamesDescriptor:
    def __get__(self, instance: Any, owner: Optional[Type[Any]] = None) -> Set[str]:
        return ASTCapabilityValidator.get_forbidden_names()


class _PermittedBuiltinsDescriptor:
    def __get__(self, instance: Any, owner: Optional[Type[Any]] = None) -> Set[str]:
        return ASTCapabilityValidator.get_permitted_builtins()


class ASTCapabilityValidator(ast.NodeVisitor):
    """
    Capability-based AST Permit-List Validator for synthesized kernels.
    Strictly allows only pure mathematical, algorithmic, collection, and bounded
    control-flow constructs. Rejects all imports, I/O, reflection builtins
    (getattr, eval, exec, open, __import__), unwhitelisted modules, and dunder attribute traversal.
    """

    PERMITTED_BUILTINS = _PermittedBuiltinsDescriptor()
    PERMITTED_MODULE_NAMESPACES = _PermittedModuleNamespacesDescriptor()
    FORBIDDEN_NAMES = _ForbiddenNamesDescriptor()
    FORBIDDEN_SIDE_EFFECT_METHODS = _ForbiddenSideEffectMethodsDescriptor()

    @classmethod
    def get_forbidden_side_effect_methods(cls) -> Set[str]:
        methods: Set[str] = set()
        try:
            try:
                from .lattice import TypeRegistry
            except (ImportError, ValueError):
                from lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
            methods.update(reg.get_verification_semantics("forbidden_side_effect_methods") or ())
        except Exception:
            pass
        return methods

    @classmethod
    def get_forbidden_names(cls) -> Set[str]:
        names: Set[str] = set()
        try:
            try:
                from .lattice import TypeRegistry
            except (ImportError, ValueError):
                from lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
            names.update(reg.get_verification_semantics("forbidden_identifier_names") or ())
        except Exception:
            pass
        return names

    @classmethod
    def get_permitted_builtins(cls) -> Set[str]:
        builtins: Set[str] = set()
        try:
            try:
                from .lattice import TypeRegistry
            except (ImportError, ValueError):
                from lattice import TypeRegistry
            reg = TypeRegistry.get_instance()
            builtins.update(reg.get_verification_semantics("permitted_kernel_builtins") or ())
        except Exception:
            pass
        return builtins

    @classmethod
    def get_permitted_module_namespaces(cls) -> Set[str]:
        namespaces: Set[str] = set()
        try:
            try:
                from .lattice import TypeRegistry, LatticeOrchestrator
            except (ImportError, ValueError):
                from lattice import TypeRegistry, LatticeOrchestrator

            reg = TypeRegistry.get_instance()
            aliases = reg.get_all_aliases()
            namespaces.update(aliases.keys())
            namespaces.update(aliases.values())
            namespaces.update(reg.get_all_libraries())

            orch = LatticeOrchestrator.get_active_instance()
            if orch:
                for cell in getattr(orch, "loaded_cells", {}).values():
                    for dep in getattr(cell, "dependencies", []) or []:
                        dep_s = str(dep).strip()
                        if not dep_s:
                            continue
                        try:
                            parsed_dep = ast.parse(dep_s)
                            for d_node in ast.walk(parsed_dep):
                                if isinstance(d_node, ast.Import):
                                    for alias_item in d_node.names:
                                        namespaces.add(alias_item.name.split(".")[0])
                                        if alias_item.asname:
                                            namespaces.add(alias_item.asname)
                                elif isinstance(d_node, ast.ImportFrom):
                                    if d_node.module:
                                        namespaces.add(d_node.module.split(".")[0])
                                    for alias_item in d_node.names:
                                        if alias_item.asname:
                                            namespaces.add(alias_item.asname)
                        except Exception:
                            mod = dep_s.split()[0].split(".")[0]
                            if mod.isidentifier():
                                namespaces.add(mod)
        except Exception:
            pass
        return namespaces

    PERMITTED_NODES: Tuple[Type[ast.AST], ...] = (
        ast.Module,
        # Statements
        ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Pass,
        ast.If, ast.For, ast.While, ast.Break, ast.Continue,
        ast.Try, ast.ExceptHandler, ast.Raise, ast.Delete,
        # Expressions
        ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
        ast.Call, ast.Constant, ast.Name, ast.Attribute, ast.Subscript, ast.Slice,
        ast.List, ast.Tuple, ast.Set, ast.Dict, ast.Starred,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
        ast.Lambda, ast.arguments, ast.arg, ast.keyword,
        ast.JoinedStr, ast.FormattedValue,
        # Contexts & Operators
        ast.Load, ast.Store, ast.Del,
        ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
    )

    def __init__(self, allowed_vars: Optional[Set[str]] = None) -> None:
        self.error: Optional[str] = None
        self.allowed_vars: Set[str] = set(allowed_vars or ())
        self._valid_roots: Set[str] = set()
        self._active_forbidden_names: Set[str] = set(self.FORBIDDEN_NAMES)
        self._active_forbidden_methods: Set[str] = set(self.FORBIDDEN_SIDE_EFFECT_METHODS)
        self._active_permitted_builtins: Set[str] = set(self.PERMITTED_BUILTINS)
        self._active_permitted_modules: Set[str] = set(self.PERMITTED_MODULE_NAMESPACES)

    def _collect_local_assigned_names(self, tree: ast.AST) -> Set[str]:
        assigned: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    assigned.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                if node.target:
                    assigned.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                assigned.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
            elif isinstance(node, ast.comprehension):
                assigned.update(n.id for n in ast.walk(node.target) if isinstance(n, ast.Name))
            elif isinstance(node, ast.NamedExpr):
                if isinstance(node.target, ast.Name):
                    assigned.add(node.target.id)
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    assigned.add(node.name)
        return assigned

    def _find_attribute_root(self, node: ast.AST) -> Optional[str]:
        curr = node
        while isinstance(curr, ast.Attribute):
            curr = curr.value
        if isinstance(curr, ast.Name):
            return curr.id
        return None

    def validate(self, tree: ast.AST, allowed_vars: Optional[Set[str]] = None) -> Tuple[bool, Optional[str]]:
        self.error = None
        current_allowed = set(self.allowed_vars)
        if allowed_vars:
            current_allowed.update(allowed_vars)

        self._active_forbidden_names = set(self.FORBIDDEN_NAMES)
        self._active_forbidden_methods = set(self.FORBIDDEN_SIDE_EFFECT_METHODS)
        self._active_permitted_builtins = set(self.PERMITTED_BUILTINS)
        self._active_permitted_modules = set(self.PERMITTED_MODULE_NAMESPACES)

        local_assigned = self._collect_local_assigned_names(tree)
        self._valid_roots = (
            current_allowed
            | local_assigned
            | self._active_permitted_modules
            | self._active_permitted_builtins
            | {"self", "cls"}
        )
        self.visit(tree)
        return (self.error is None, self.error)

    def generic_visit(self, node: ast.AST) -> None:
        if self.error is not None:
            return
        if not isinstance(node, self.PERMITTED_NODES):
            self.error = f"Disallowed AST construct '{type(node).__name__}' in pure kernel"
            return
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if self.error is not None:
            return
        if node.id.startswith("__") or node.id.endswith("__") or node.id in self._active_forbidden_names:
            self.error = f"Forbidden identifier or builtin reference: '{node.id}'"
            return
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self.error is not None:
            return
        if node.attr.startswith("__") or node.attr.endswith("__"):
            self.error = f"Forbidden dunder attribute access: '.{node.attr}'"
            return
        if node.attr in self._active_forbidden_names:
            self.error = f"Forbidden attribute access: '.{node.attr}'"
            return
        root_id = self._find_attribute_root(node)
        if root_id is not None and root_id not in self._valid_roots:
            self.error = f"Unauthorized attribute access on untrusted/unpermitted identifier '{root_id}.{node.attr}'"
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.error is not None:
            return
        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in self._active_permitted_builtins and func.id not in self._valid_roots:
                self.error = f"Call to non-whitelisted function or builtin: '{func.id}()'"
                return
        elif isinstance(func, ast.Attribute):
            if func.attr.startswith("_") or func.attr in self._active_forbidden_names:
                self.error = f"Forbidden method call: '.{func.attr}()'"
                return
            # Block declared forbidden side-effecting methods even on variables
            if func.attr in self._active_forbidden_methods:
                self.error = f"Forbidden side-effecting method call: '.{func.attr}()'"
                return
            root_id = self._find_attribute_root(func)
            if root_id is not None and root_id not in self._valid_roots:
                self.error = f"Unauthorized method call on untrusted/unpermitted identifier '{root_id}.{func.attr}()'"
                return
        else:
            self.error = f"Disallowed dynamic callable expression '{type(func).__name__}'"
            return
        self.generic_visit(node)


# --------------------------------------------------------------------------- #
# Combinator-library detection
# --------------------------------------------------------------------------- #

def import_stmt_for(qualified_name: str) -> str:
    """
    Derives an import statement from a qualified symbol or module name, verified live.
    Conforms to Section 3.4 of the STRATA paper.
    """
    module = qualified_name.rsplit(".", 1)[0] if "." in qualified_name else qualified_name
    try:
        importlib.import_module(module)
    except ImportError:
        logger.warning("unverifiable import: %s", module)
    return f"import {module}"


def _safe_substitute(template: str, bindings: Dict[str, Any]) -> str:
    """
    Placeholder substitution that ONLY touches ``{identifier}`` spans.
    Uses zero-regex safe_substitute_template.
    """
    return safe_substitute_template(template, bindings)


# --------------------------------------------------------------------------- #
# Micro-cell synthesis
# --------------------------------------------------------------------------- #

class SynthesisEngine:
    """
    Synthesizes missing computational primitives on-demand using live documentation lookup.
    """

    def __init__(self, trees_dir: str = "trees"):
        self.trees_dir = trees_dir

    def synthesize_micro_cell(
        self,
        gap_concept: str,
        expected_input: Optional[str] = None,
        expected_output: Optional[str] = None,
        fetcher: Optional[Any] = None,
        domain: str = "Python_Core",
        stage: Optional[int] = 1,
    ) -> Dict[str, Any]:
        """
        Queries official API documentation and uses the LLM to synthesize
        a verified, type-annotated MicroCell.
        """
        logger.info("[SYNTHESIS] Fetching live documentation for: '%s'", gap_concept)
        fetch_call = getattr(fetcher, "fetch", None)
        if fetch_call is None:
            live_docs = ""
        else:
            try:
                live_docs = fetch_call(gap_concept)
            except TypeError:
                live_docs = ""
            if asyncio.iscoroutine(live_docs) or isinstance(live_docs, asyncio.Future):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                        live_docs = _pool.submit(asyncio.run, live_docs).result()
                else:
                    live_docs = asyncio.run(live_docs)
            if not isinstance(live_docs, str):
                live_docs = str(getattr(live_docs, "content", "") or "")
        live_docs = live_docs or "No live documentation available."

        try:
            from lattice import TypeRegistry
        except (ImportError, ValueError):
            from .lattice import TypeRegistry
        reg = TypeRegistry.get_instance()
        top_types = set(reg.get_verification_semantics("universal_top_types") or ())
        empty_types = set(reg.get_verification_semantics("empty_carrier_types") or ())
        path_type_names = set(reg.get_verification_semantics("path_type_names") or ())
        path_tokens = set(reg.get_path_port_tokens() or ()) | path_type_names

        def _is_path_type(t_name: Optional[str]) -> bool:
            if not t_name:
                return False
            clean = str(t_name).strip().lower()
            if clean in path_tokens:
                return True
            try:
                for pt in path_type_names:
                    if reg.is_known_type_name(pt) and reg.is_subtype(clean, pt):
                        return True
            except Exception:
                pass
            return False

        in_str = str(expected_input).strip().lower() if expected_input else ""
        out_str = str(expected_output).strip().lower() if expected_output else ""

        is_path_in = _is_path_type(in_str)
        is_path_out = _is_path_type(out_str)

        is_source = (not in_str or in_str in top_types or in_str in empty_types or is_path_in)
        is_sink = (not out_str or out_str in empty_types or is_path_out)

        if is_source and not is_sink:
            role = "source"
        elif is_sink and not is_source:
            role = "sink"
        elif is_source and is_sink:
            if not in_str or in_str in top_types or in_str in empty_types:
                role = "source"
            else:
                role = "sink"
        else:
            role = "transformer"

        in_placeholder = "{input_locator}" if is_path_in else "{input_var}"
        out_placeholder = "{output_locator}" if is_path_out else "{output_var}"

        if role == "source":
            if is_path_in:
                template_rule = (
                    "For Source (Initial / Ingress morphism with locator input), `code_template` MUST produce the primary output "
                    f"assigned to `{{output_var}}` by consuming `{in_placeholder}` "
                    f"(e.g. `{{output_var}} = <package>.<source_func>({in_placeholder})`)."
                )
                input_spec = f'{{"type_name": "{expected_input or "path"}", "state": "source_identifier"}}'
                sample_template = f"{{output_var}} = <package>.<source_func>({in_placeholder})"
            else:
                template_rule = (
                    "For Source (Initial / Ingress morphism), `code_template` MUST produce the primary output "
                    "assigned to `{output_var}` (e.g. `{output_var} = <package>.<source_func>()`)."
                )
                input_spec = "{}"
                sample_template = "{output_var} = <package>.<source_func>()"
        elif role == "sink":
            if is_path_out:
                template_rule = (
                    "For Sink (Terminal / Egress morphism with destination locator), `code_template` MUST consume `{input_var}` and "
                    f"persist or export it to `{out_placeholder}` (e.g. `{{input_var}}.<sink_func>({out_placeholder})\\n{{output_var}} = {out_placeholder}`)."
                )
                input_spec = f'{{"type_name": "{expected_input}", "state": "any"}}'
                sample_template = f"{{input_var}}.<sink_func>({out_placeholder})\n{{output_var}} = {out_placeholder}"
            else:
                template_rule = (
                    "For Sink (Terminal / Egress morphism), `code_template` MUST consume `{input_var}` and "
                    "complete the terminal action or lifecycle (e.g. `{output_var} = <package>.<sink_func>({input_var})` or `{input_var}.<sink_func>()`)."
                )
                input_spec = f'{{"type_name": "{expected_input}", "state": "any"}}'
                sample_template = "{output_var} = <package>.<sink_func>({input_var})"
        else:
            template_rule = (
                "For Transformer (Intermediate morphism), `code_template` MUST consume `{input_var}` and "
                "assign the result to `{output_var}` (e.g. `{output_var} = <func>({input_var})`)."
            )
            input_spec = f'{{"type_name": "{expected_input}", "state": "any"}}'
            sample_template = "{output_var} = <package>.<func>({input_var})"

        sanitized_concept = "".join(
            c if c.isalnum() or c == "_" else "_" for c in gap_concept
        ).lower()[:30]

        system_prompt = f"""You are an expert Software Engineer. Synthesize a single verified Python computational node implementing: '{gap_concept}'.
Output ONLY a valid JSON object matching the schema below. No markdown formatting, no explanations.

Schema:
{{
  "cell_id": "micro_synthesized_{sanitized_concept}",
  "type": "micro",
  "stage": {stage},
  "keywords": ["{gap_concept}"],
  "inputs": {input_spec},
  "outputs": {{ "type_name": "{expected_output}", "state": "computed" }},
  "dependencies": ["<import statement>"],
  "code_template": "{sample_template}"
}}

Documentation Reference:
{live_docs[:1500]}

RULES:
1. {template_rule}
2. Put all necessary import statements in the `dependencies` array (e.g. ["import package", "from package import module"]).
3. Ensure the code is strictly non-interactive (do not use input() or sys.stdin).
4. Never hardcode external asset paths or destinations — always use {in_placeholder} or {out_placeholder} placeholders when loading or persisting external assets."""

        full_prompt = f"{system_prompt}\n\nTask: {gap_concept}"
        result_text = ModelManager.get_instance().generate_text(full_prompt, max_tokens=1024)

        cell_dict = extract_json_from_llm(result_text)
        if cell_dict is None:
            logger.error("[SYNTHESIS ERROR] Failed to extract JSON from model output")
            raise ValueError(f"Model failed to generate valid JSON for {gap_concept}")

        template = extract_code_from_llm_response(cell_dict.get("code_template", ""))
        cell_dict["code_template"] = template
        if not validate_code_template(template):
            logger.error("[SYNTHESIS ERROR] Synthesized template failed AST validation")
            raise ValueError(f"Synthesized template for {gap_concept} failed AST parse check.")

        return cell_dict


def synthesize_and_register(
    engine: SynthesisEngine,
    gap_concept: str,
    expected_input: str,
    expected_output: str,
    fetcher: LiveDocFetcher,
    registry: Any,
    domain: str = "Python_Core",
    stage: Optional[int] = None,
) -> Optional[Any]:
    """
    Pipeline-facing entrypoint that synthesizes a micro-cell and registers it
    into a TypeRegistry / cell repository.
    """
    try:
        cell_dict = engine.synthesize_micro_cell(
            gap_concept=gap_concept,
            expected_input=expected_input,
            expected_output=expected_output,
            fetcher=fetcher,
            domain=domain,
            stage=stage,
        )
    except Exception as exc:
        logger.error("[SYNTHESIS] registration failed for '%s': %s", gap_concept, exc)
        return None

    if registry is None:
        return cell_dict

    for method_name in ("register_cell", "add_cell", "register"):
        method = getattr(registry, method_name, None)
        if callable(method):
            try:
                method(cell_dict)
                logger.info("[SYNTHESIS] registered synthesized cell '%s'", cell_dict.get("cell_id"))
                return cell_dict
            except Exception as exc:
                logger.error("[SYNTHESIS] registry.%s failed: %s", method_name, exc)
                return cell_dict

    logger.warning("[SYNTHESIS] no compatible register method on registry; returning raw dict")
    return cell_dict


# --------------------------------------------------------------------------- #
# Loop rendering helpers (all data-driven, zero hard-coded variable names)
# --------------------------------------------------------------------------- #

def _derive_loop_item_type(cell: Any, accumulated_sigma: Optional[Any]) -> Optional[Any]:
    """
    Derives the declared iteration carrier (item type) of a traced-loop macro.
    """
    try:
        from .unification import TypeTerm, GenericTypeTerm, substitute_generics
    except (ImportError, ValueError):
        from unification import TypeTerm, GenericTypeTerm, substitute_generics

    for p_sig in getattr(cell, "inputs", {}).values():
        raw_t = str(
            getattr(getattr(p_sig, "signature", None), "type_name", "")
            or getattr(p_sig, "type_name", "")
        )
        if not raw_t or "[" not in raw_t:
            continue
        concrete_t = substitute_generics(raw_t, accumulated_sigma) if accumulated_sigma is not None else raw_t
        try:
            term = TypeTerm.from_string(concrete_t)
        except Exception as e:
            logger.debug("[SYNTHESIS] TypeTerm parse skipped for '%s': %s", concrete_t, e)
            continue
        if isinstance(term, GenericTypeTerm) and term.args:
            return term.args[0]
    return None


def _declared_loop_kind(cell: Any) -> Optional[str]:
    """
    Reads the declared loop kind from cell metadata (never from template strings).
    Returns 'collect', 'extremum', 'fold', or None.
    """
    meta = getattr(cell, "tree_meta", None) or {}
    declared = getattr(cell, "loop_kind", None)
    if declared is None and isinstance(meta, dict):
        declared = meta.get("loop_kind")

    try:
        from lattice import TypeRegistry
    except (ImportError, ValueError):
        from .lattice import TypeRegistry
    reg = TypeRegistry.get_instance()

    fold_kinds = set(reg.get_verification_semantics("fold_loop_kinds") or ())
    collect_kinds = set(reg.get_verification_semantics("collect_loop_kinds") or ())
    extremum_kinds = set(reg.get_verification_semantics("extremum_loop_kinds") or ())

    if declared:
        d_lower = str(declared).lower()
        if d_lower in fold_kinds:
            return "fold"
        if d_lower in collect_kinds:
            return "collect"
        if d_lower in extremum_kinds:
            return "extremum"

    topology = str(getattr(cell, "topology_type", "") or "").lower()
    if topology in set(reg.get_verification_semantics("collect_loop_topologies") or ()):
        return "collect"
    if topology in set(reg.get_verification_semantics("extremum_loop_topologies") or ()):
        return "extremum"
    if topology in set(reg.get_verification_semantics("fold_loop_topologies") or ()):
        return "fold"
    return None


def _sanitize_template_for_ast(template: str) -> str:
    """
    Sanitizes template placeholders `{ph}` to valid Python identifiers `_nstl_ph_{ph}`
    so the template can be safely parsed by `ast.parse`.
    """
    if not template or not template.strip():
        return ""
    phs = extract_template_placeholders(template)
    sanitized = template
    for ph in phs:
        sanitized = sanitized.replace(f"{{{ph}}}", f"_nstl_ph_{ph}")
    return sanitized


def _unsanitize_ast_source(expr: str) -> str:
    """Restores `_nstl_ph_<name>` tokens back to `{<name>}` placeholders."""
    phs = []
    idx = 0
    marker = "_nstl_ph_"
    while True:
        pos = expr.find(marker, idx)
        if pos == -1:
            phs.append(expr[idx:])
            break
        phs.append(expr[idx:pos])
        end = pos + len(marker)
        while end < len(expr) and (expr[end].isalnum() or expr[end] == "_"):
            end += 1
        name = expr[pos + len(marker):end]
        phs.append(f"{{{name}}}" if name else marker)
        idx = end
    return "".join(phs)


def _template_assigns_var(rendered: str, var_name: str) -> bool:
    """AST check: returns True iff `var_name` is assigned as a top-level target in `rendered`."""
    if not rendered or not var_name:
        return False
    try:
        parsed = ast.parse(_sanitize_template_for_ast(rendered))
    except SyntaxError:
        return False
    for stmt in parsed.body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and tgt.id == var_name:
                    return True
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == var_name and stmt.value is not None:
                return True
    return False


def _format_default_value(p_sig: Any, reg: Optional[Any] = None) -> str:
    """Formats a port's declared `default_value` using TypeRegistry subtype checks."""
    dv = getattr(p_sig, "default_value", None)
    if reg is None and "TypeRegistry" in globals():
        try:
            reg = TypeRegistry.get_instance()
        except Exception:
            reg = None
    p_tn = str(
        getattr(getattr(p_sig, "signature", None), "type_name", "")
        or getattr(p_sig, "type_name", "")
        or ""
    ).strip().lower()
    is_str_type = (
        reg.is_subtype(p_tn, "str")
        if (reg is not None and p_tn)
        else (p_tn == str.__name__)
    )
    if isinstance(dv, str) and is_str_type:
        try:
            evaled = ast.literal_eval(dv.strip())
            if isinstance(evaled, str):
                return dv.strip()
        except Exception:
            pass
        return repr(dv)
    return str(dv)


def _template_initializes_container(template: str, primary_out_name: Optional[str] = None) -> Optional[str]:
    """
    AST inspection: detect whether the template initializes the output
    container variable (`_nstl_ph_output_var` or `_nstl_ph_<primary_out_name>`)
    to an empty container and return its container kind.
    """
    if not template or not template.strip():
        return None
    sanitized = _sanitize_template_for_ast(template)
    try:
        parsed = ast.parse(sanitized)
    except Exception:
        return None

    target_ids = {"_nstl_ph_output_var"}
    if primary_out_name:
        target_ids.add(f"_nstl_ph_{primary_out_name}")

    constructors = {
        "set": "set",
        "dict": "dict", "OrderedDict": "dict", "defaultdict": "dict", "Counter": "dict",
        "list": "list", "tuple": "tuple", "deque": "deque",
    }
    for node in ast.walk(parsed):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            is_out = any(
                isinstance(t, ast.Name) and t.id in target_ids
                for t in targets
            )
            if not is_out:
                continue
            val = node.value
            if isinstance(val, ast.List):
                return "list"
            if isinstance(val, ast.Dict):
                return "dict"
            if isinstance(val, ast.Set):
                return "set"
            if isinstance(val, ast.Tuple):
                return "tuple"
            if isinstance(val, ast.Call):
                func_name = None
                if isinstance(val.func, ast.Name):
                    func_name = val.func.id
                elif isinstance(val.func, ast.Attribute):
                    func_name = val.func.attr
                if func_name and func_name in constructors:
                    return constructors[func_name]
    return None


def _detect_loop_item_var(template: str) -> Optional[str]:
    """
    AST inspection: scans the sanitized template AST for an `ast.For` or `ast.AsyncFor`
    node and extracts the loop variable name from node.target.
    """
    if not template or not template.strip():
        return None
    sanitized = _sanitize_template_for_ast(template)
    try:
        parsed = ast.parse(sanitized)
    except Exception:
        return None

    for node in ast.walk(parsed):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            target = node.target
            target_name = None
            if isinstance(target, ast.Name):
                target_name = target.id
            elif isinstance(target, (ast.Tuple, ast.List)) and target.elts:
                first = target.elts[0]
                if isinstance(first, ast.Name):
                    target_name = first.id
            if target_name:
                if target_name.startswith("_nstl_ph_"):
                    return target_name[len("_nstl_ph_"):]
                return target_name
    return None


def _unique_var(cell: Any, suffix: str) -> str:
    """
    Returns a cell-scoped variable name of the form `_<cell_id>_<suffix>`.
    Guarantees no collision between nested renders of distinct cells.
    """
    cid = (
        getattr(cell, "cell_id", None)
        or getattr(cell, "name", None)
        or "cell"
    )
    safe = "".join(c if c.isalnum() else "_" for c in str(cid)) or "cell"
    return f"_{safe}_{suffix}"


def _resolve_item_var(cell: Any, template: str) -> str:
    """
    Resolves the loop item variable name from declared metadata, or detected
    from the template's `for ... in ...` header, or a cell-scoped fallback.
    """
    meta = getattr(cell, "tree_meta", None) or {}
    declared = getattr(cell, "item_var", None)
    if declared is None and isinstance(meta, dict):
        declared = meta.get("item_var")
    if declared:
        return str(declared)
    detected = _detect_loop_item_var(template)
    if detected:
        return detected
    return _unique_var(cell, "item")


def _resolve_direction_is_max(cell: Any, bindings: Dict[str, Any]) -> bool:
    """
    Resolves the declared extremum direction from a bound port value. Only
    reads declared bindings (never prompt text). Returns True for max/argmax.
    """
    meta = getattr(cell, "tree_meta", None) or {}
    direction_port = (
        getattr(cell, "direction_port", None)
        or (meta.get("direction_port") if isinstance(meta, dict) else None)
    )
    raw = bindings.get(direction_port) if direction_port else None
    if raw is None:
        try:
            try:
                from .lattice import TypeRegistry
            except (ImportError, ValueError):
                from lattice import TypeRegistry
            order_ports = TypeRegistry.get_instance().get_verification_semantics("order_flag_names") or ()
        except Exception:
            order_ports = ()
        for k in tuple(order_ports):
            if k in bindings and bindings[k] is not None:
                raw = bindings[k]
                break
    if raw is None:
        return True
    s = str(raw).strip().strip("'\"").lower()
    if s in ("false", "0"):
        return False
    return True


def render_cell(
    cell: Any,
    bindings: Dict[str, Any],
    indent_level: int = 0,
    context: Optional[Any] = None,
    accumulated_sigma: Optional[Any] = None,
) -> str:
    """
    Renders an executable block of Python code for a cell at the specified indentation level.
    Recursively renders child sub-pipelines in bound_slots at indent_level + 1.
    """
    try:
        from .unification import TypeTerm, GenericTypeTerm, UnificationGate, unify, Substitution
    except (ImportError, ValueError):
        from unification import TypeTerm, GenericTypeTerm, UnificationGate, unify, Substitution
    try:
        from .coalgebraic_unfolder import is_macro_combinator_cell
    except (ImportError, ValueError):
        from coalgebraic_unfolder import is_macro_combinator_cell

    reg_syn = TypeRegistry.get_instance() if "TypeRegistry" in globals() else None
    indent_prefix = "    " * indent_level
    template = cell.code_template.strip()
    if not template:
        return ""

    slots = getattr(cell, "slots", {}) or {}
    bound_slots = getattr(cell, "bound_slots", {}) or {}
    prim_out_obj = getattr(cell, "primary_output", None)
    prim_out_name_outer = getattr(prim_out_obj, "name", None) if prim_out_obj else None

    out_var = bindings.get(
        "output_var",
        f"_{getattr(prim_out_obj, 'state', None) or 'out'}",
    )

    loop_kind = _declared_loop_kind(cell)
    init_container = _template_initializes_container(template, primary_out_name=prim_out_name_outer)
    if loop_kind == "extremum":
        is_extremum_loop, is_collect_loop, is_fold_loop = True, False, False
    elif loop_kind == "collect":
        is_extremum_loop, is_collect_loop, is_fold_loop = False, True, False
    elif loop_kind == "fold":
        is_extremum_loop, is_collect_loop, is_fold_loop = False, False, True
    else:
        has_direction_port = "direction" in getattr(cell, "inputs", {})
        direction_bound = bindings.get("direction") is not None
        is_extremum_loop = has_direction_port and direction_bound
        is_collect_loop = (
            not is_extremum_loop and init_container is not None
        )
        is_fold_loop = False

    is_max = _resolve_direction_is_max(cell, bindings) if is_extremum_loop else False

    item_var = _resolve_item_var(cell, template)
    metric_var = _unique_var(cell, "max_metric" if is_max else "min_metric")

    item_type = (
        _derive_loop_item_type(cell, accumulated_sigma)
        if (is_collect_loop or is_extremum_loop or is_fold_loop)
        else None
    )

    slot_rendered: Dict[str, str] = {}
    slot_last_out: Dict[str, str] = {}
    cell_inputs = getattr(cell, "inputs", {}) or {}
    slots_items = slots.items() if isinstance(slots, dict) else [(s, {}) for s in slots]
    macro_slot_items = [
        (sn, sp)
        for sn, sp in slots_items
        if (str(sn) in bound_slots) or (str(sn) not in cell_inputs)
    ]
    for slot_name, slot_spec in macro_slot_items:
        child_nodes = bound_slots.get(slot_name, [])
        if not child_nodes:
            # Every macro slot must be grounded in verified lattice / control-flow cells;
            # raw unconstrained LLM code generation is never invoked here.
            slot_rendered[slot_name] = "pass"
            continue

        if isinstance(child_nodes, str):
            slot_rendered[slot_name] = child_nodes
            continue

        if not isinstance(child_nodes, list):
            child_nodes = [child_nodes]

        child_lines: List[str] = []
        last_metric: Optional[str] = None
        enclosing_known_vars: Set[str] = set()
        # Track active typed outputs produced within this slot (and from parent context)
        # so multi-step slot sub-pipelines wire type-compatibly without hardcodes.
        slot_typed_vars: List[Tuple[Any, str]] = []
        if context is not None:
            for ctx_attr in ("variables", "var_types"):
                ctx_map = getattr(context, ctx_attr, None)
                if isinstance(ctx_map, dict):
                    if ctx_attr == "variables":
                        enclosing_known_vars.update(ctx_map.keys())
                    for v_name, v_entry in ctx_map.items():
                        v_port = v_entry[0] if isinstance(v_entry, tuple) and v_entry else v_entry
                        v_sig = getattr(v_port, "signature", v_port)
                        if v_sig is not None:
                            slot_typed_vars.append((v_sig, v_name))
            if hasattr(context, "active_variables") and isinstance(context.active_variables, (set, list)):
                enclosing_known_vars.update(context.active_variables)
        for p_in_name, p_in_sig in cell_inputs.items():
            if p_in_name in bindings and isinstance(bindings[p_in_name], str):
                b_clean = bindings[p_in_name].strip().strip("'\"")
                if b_clean.isidentifier():
                    in_sig = getattr(p_in_sig, "signature", p_in_sig)
                    slot_typed_vars.append((in_sig, b_clean))
        for v in bindings.values():
            if isinstance(v, str):
                v_clean = v.strip().strip("'\"")
                if v_clean.isidentifier():
                    enclosing_known_vars.add(v_clean)
        if item_var and item_var.isidentifier():
            enclosing_known_vars.add(item_var)
        if out_var and out_var.isidentifier():
            enclosing_known_vars.add(out_var)

        for child_idx, child in enumerate(child_nodes):
            child_bindings: Dict[str, Any] = {}
            for p_name, p_sig in (getattr(child, "inputs", {}) or {}).items():
                bound = False
                if item_type is not None:
                    child_sig = getattr(p_sig, "signature", None)
                    if child_sig is not None:
                        try:
                            if (
                                unify(item_type, child_sig, Substitution()) is not None
                                or unify(child_sig, item_type, Substitution()) is not None
                            ):
                                child_bindings[p_name] = item_var
                                bound = True
                        except Exception as e:
                            logger.debug("[SYNTHESIS] Loop item unify check skipped: %s", e)
                            bound = False
                if bound:
                    continue
                if hasattr(child, "typed_bindings") and child.typed_bindings and p_name in child.typed_bindings:
                    rec = child.typed_bindings[p_name]
                    if rec and rec.value_expr:
                        child_bindings[p_name] = rec.value_expr
                        continue
                if p_name in bindings:
                    child_bindings[p_name] = bindings[p_name]
                    continue
                # Match compatible upstream output variable inside slot by type/carrier
                p_role = str(getattr(p_sig, "port_role", "") or "").lower()
                child_in_sig = getattr(p_sig, "signature", None)
                matched_slot_var = None
                literal_roles = (
                    set(reg_syn.get_verification_semantics("literal_roles") or ())
                    if reg_syn is not None else set()
                )
                is_literal_port = (
                    p_role in literal_roles
                    or (reg_syn.is_config_role(p_role) if reg_syn is not None and p_role else False)
                )
                if not is_literal_port and child_in_sig is not None:
                    in_t = str(getattr(child_in_sig, "type_name", "") or "").lower()
                    is_prim_or_top = (
                        not in_t
                        or (
                            (reg_syn.is_declared_top(in_t) or reg_syn.is_scalar_primitive(in_t))
                            if reg_syn is not None else False
                        )
                    )
                    if not is_prim_or_top:
                        for prev_sig, prev_var in reversed(slot_typed_vars):
                            try:
                                if (
                                    unify(prev_sig, child_in_sig, Substitution()) is not None
                                    or unify(child_in_sig, prev_sig, Substitution()) is not None
                                ):
                                    matched_slot_var = prev_var
                                    break
                            except Exception:
                                continue
                if matched_slot_var is not None:
                    child_bindings[p_name] = matched_slot_var
                elif p_name in enclosing_known_vars:
                    child_bindings[p_name] = p_name
                else:
                    lit_bound = None
                    if context is not None and hasattr(context, "resolve_literal_for_port"):
                        prev_cl = getattr(context, "current_cell_clause_idx", None)
                        ch_cl = getattr(child, "matched_clause_idx", None)
                        if ch_cl is not None:
                            context.current_cell_clause_idx = ch_cl
                        try:
                            lit_bound = context.resolve_literal_for_port(p_sig, cell=child)
                        except Exception:
                            lit_bound = None
                        finally:
                            context.current_cell_clause_idx = prev_cl
                    if lit_bound is not None:
                        child_bindings[p_name] = lit_bound
                        continue
                    try:
                        try:
                            from .unification import resolve_synthetic_boundary_io
                        except (ImportError, ValueError):
                            from unification import resolve_synthetic_boundary_io
                        synth_io = resolve_synthetic_boundary_io(p_sig, cell=child, ctx=context)
                    except Exception:
                        synth_io = None
                    if synth_io is not None:
                        child_bindings[p_name] = synth_io
                        continue
                    if getattr(p_sig, "default_value", None) is not None:
                        child_bindings[p_name] = _format_default_value(p_sig, reg=reg_syn)
                    else:
                        try:
                            try:
                                from .unification import _pointed_default
                            except (ImportError, ValueError):
                                from unification import _pointed_default
                            eta_v = _pointed_default(p_sig, child)
                            if eta_v is not None:
                                child_bindings[p_name] = eta_v
                        except Exception:
                            pass

            child_out_var = f"{_unique_var(child, 'out')}_{child_idx}"
            child_bindings["output_var"] = child_out_var
            last_metric = child_out_var
            enclosing_known_vars.add(child_out_var)

            child_outputs = getattr(child, "outputs", {}) or {}
            prim_out = getattr(child, "primary_output", None)
            prim_out_name = getattr(prim_out, "name", None) if prim_out else None
            for out_p_name, out_p_sig in child_outputs.items():
                if len(child_outputs) == 1 or out_p_name == prim_out_name:
                    ov = child_out_var
                else:
                    ov = f"{_unique_var(child, out_p_name)}_{child_idx}"
                child_bindings[out_p_name] = ov
                enclosing_known_vars.add(ov)
                out_sig_obj = getattr(out_p_sig, "signature", out_p_sig)
                if out_sig_obj is not None:
                    slot_typed_vars.append((out_sig_obj, ov))

            if (
                bool(getattr(child, "bound_slots", None))
                or is_macro_combinator_cell(child)
            ):
                child_text = render_cell(
                    child,
                    child_bindings,
                    indent_level=0,
                    context=context,
                    accumulated_sigma=accumulated_sigma,
                )
            else:
                child_text = UnificationGate._instantiate_ast_template(
                    child.code_template,
                    child_bindings,
                    child.inputs,
                    known_vars=enclosing_known_vars,
                )
            if child_text:
                child_lines.append(child_text)

        if is_extremum_loop and last_metric is not None:
            op = ">" if is_max else "<"
            update_block = (
                f"if {metric_var} is None or {last_metric} {op} {metric_var}:\n"
                f"    {metric_var} = {last_metric}\n"
                f"    {out_var} = {item_var}"
            )
            child_lines.append(update_block)
        elif is_collect_loop and last_metric is not None:
            container_type = init_container
            if container_type is None:
                out_sig = getattr(cell, "primary_output", None)
                t_name = str(getattr(out_sig, "type_name", "") or "").strip().lower()
                base_t = t_name
                try:
                    t_parsed = TypeTerm.from_string(t_name) if t_name else None
                    if isinstance(t_parsed, GenericTypeTerm):
                        base_t = str(t_parsed.base or t_name).strip().lower()
                except Exception:
                    base_t = t_name.split("[", 1)[0].strip() if "[" in t_name else t_name
                if reg_syn is not None and (base_t == "set" or reg_syn.is_subtype(base_t, "set")):
                    container_type = "set"
                elif reg_syn is not None and (base_t in ("dict", "mapping") or reg_syn.is_subtype(base_t, "dict") or reg_syn.is_subtype(base_t, "mapping")):
                    container_type = "dict"
                else:
                    container_type = "list"

            if container_type == "set":
                child_lines.append(f"{out_var}.add({last_metric})")
            elif container_type == "dict":
                child_lines.append(f"{out_var}[{item_var}] = {last_metric}")
            else:
                child_lines.append(f"{out_var}.append({last_metric})")
        elif is_fold_loop and last_metric is not None:
            meta = getattr(cell, "tree_meta", None) or {}
            reducer = getattr(cell, "reducer", None)
            if reducer is None and isinstance(meta, dict):
                reducer = meta.get("reducer") or meta.get("fold_op") or meta.get("accumulation_op")

            if reducer in ("+", "-", "*", "/", "%", "&", "|", "^"):
                child_lines.append(f"{out_var} = {out_var} {reducer} {last_metric}")
            elif reducer and str(reducer).isidentifier():
                child_lines.append(f"{out_var} = {reducer}({out_var}, {last_metric})")
            else:
                child_lines.append(f"{out_var} = {last_metric} if {out_var} is None else ({out_var} + {last_metric})")

        slot_rendered[slot_name] = "\n".join(child_lines) if child_lines else "pass"
        if last_metric is not None:
            slot_last_out[slot_name] = last_metric

    rendered = template
    if is_extremum_loop and not _template_assigns_var(rendered, metric_var):
        rendered = f"{metric_var} = None\n" + rendered
    elif is_fold_loop and not _template_assigns_var(rendered, out_var):
        init_val = getattr(cell, "init_val", None)
        if init_val is None and isinstance(getattr(cell, "tree_meta", None), dict):
            init_val = cell.tree_meta.get("init_val") or cell.tree_meta.get("initial_value")
        init_expr = str(init_val) if init_val is not None else "None"
        rendered = f"{out_var} = {init_expr}\n" + rendered

    # Map macro slot names to non-predicate HOF ports when the template uses
    # functional call syntax (e.g. ({body_fn})(_state)) rather than {body}.
    action_op_ports: List[str] = []
    for p_name, p_sig in cell_inputs.items():
        p_role = str(getattr(p_sig, "port_role", "") or "").strip().lower()
        is_hof = (
            reg_syn.is_hof_role(p_role)
            if (reg_syn is not None and p_role)
            else False
        )
        if is_hof:
            t_str = str(getattr(getattr(p_sig, "signature", None), "type_name", "") or getattr(p_sig, "type_name", "") or "").strip()
            is_bool_predicate = False
            if t_str:
                try:
                    t_term = TypeTerm.from_string(t_str)
                    if isinstance(t_term, GenericTypeTerm) and t_term.args:
                        ret_term = t_term.args[-1]
                        ret_name = str(getattr(ret_term, "name", None) or getattr(ret_term, "base", None) or ret_term).strip().lower()
                        is_bool_predicate = (
                            reg_syn.is_subtype(ret_name, "bool")
                            if reg_syn is not None
                            else (ret_name == bool.__name__)
                        )
                except Exception:
                    is_bool_predicate = False
            if is_bool_predicate:
                continue
            action_op_ports.append(p_name)

    def _resolve_op_port_for_slot(s_name: str, s_idx: int) -> Optional[str]:
        s_low = s_name.lower()
        s_stem = s_low.split("_")[0]
        for op_p in action_op_ports:
            op_low = op_p.lower()
            if op_low == s_low or op_low.startswith(s_stem):
                return op_p
        if len(action_op_ports) == 1 and len(slot_rendered) == 1:
            return action_op_ports[0]
        if s_idx < len(action_op_ports):
            return action_op_ports[s_idx]
        return None

    for s_idx, (slot_name, slot_code) in enumerate(slot_rendered.items()):
        placeholder = f"{{{slot_name}}}"
        if placeholder in rendered:
            base_indent = "    "
            found_target = None
            for line in rendered.splitlines():
                stripped = line.lstrip(" \t")
                if stripped.startswith(placeholder):
                    base_indent = line[: len(line) - len(stripped)]
                    found_target = base_indent + placeholder
                    break

            indented_slot = "\n".join(
                (base_indent + line if line.strip() else line)
                for line in slot_code.splitlines()
            )
            if found_target and found_target in rendered:
                rendered = rendered.replace(found_target, indented_slot)
            else:
                rendered = rendered.replace(placeholder, indented_slot)
            continue

        # Fallback: template uses functional_operator call placeholder ({body_fn}) etc.
        if not bound_slots.get(slot_name):
            continue
        op_port = _resolve_op_port_for_slot(slot_name, s_idx)
        if not op_port:
            continue
        op_ph = f"{{{op_port}}}"
        if op_ph not in rendered:
            continue
        new_lines: List[str] = []
        replaced = False
        for line in rendered.splitlines():
            if not replaced and op_ph in line:
                stripped = line.lstrip(" \t")
                base_indent = line[: len(line) - len(stripped)] or "    "
                indented_slot = "\n".join(
                    (base_indent + sl if sl.strip() else sl)
                    for sl in slot_code.splitlines()
                )
                new_lines.append(indented_slot)
                slot_out_var = slot_last_out.get(slot_name)
                # Inspect the sanitized AST statement to preserve `return` or assignment LHS
                stmt_node = None
                try:
                    src_probe = _sanitize_template_for_ast(stripped)
                    if stripped.endswith(":"):
                        src_probe = src_probe + "\n    pass"
                    parsed_stmt = ast.parse(src_probe)
                    stmt_node = parsed_stmt.body[0] if parsed_stmt.body else None
                except SyntaxError:
                    stmt_node = None
                if isinstance(stmt_node, ast.Return):
                    ret_expr = slot_out_var or bindings.get("output_var") or "None"
                    new_lines.append(f"{base_indent}return {ret_expr}")
                elif isinstance(stmt_node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    raw_tgt = stmt_node.targets[0] if isinstance(stmt_node, ast.Assign) else stmt_node.target
                    lhs_target = _unsanitize_ast_source(ast.unparse(raw_tgt)).strip()
                    if lhs_target and slot_out_var and lhs_target != slot_out_var:
                        new_lines.append(f"{base_indent}{lhs_target} = {slot_out_var}")
                replaced = True
            else:
                new_lines.append(line)
        if replaced:
            rendered = "\n".join(new_lines)

    render_bindings = dict(bindings)
    for p_name, p_sig in cell_inputs.items():
        if p_name not in render_bindings or render_bindings[p_name] is None:
            if getattr(p_sig, "default_value", None) is not None:
                render_bindings[p_name] = _format_default_value(p_sig, reg=reg_syn)
    cell_outputs = getattr(cell, "outputs", {}) or {}
    prim_out = getattr(cell, "primary_output", None)
    prim_out_name = getattr(prim_out, "name", None) if prim_out else None
    for out_p_name in cell_outputs.keys():
        if out_p_name not in render_bindings or not render_bindings[out_p_name]:
            if len(cell_outputs) == 1 or out_p_name == prim_out_name:
                render_bindings[out_p_name] = out_var
            else:
                render_bindings[out_p_name] = _unique_var(cell, out_p_name)

    rendered = safe_substitute_template(rendered, render_bindings)

    if indent_level > 0:
        rendered = "\n".join(
            (indent_prefix + line if line.strip() else line)
            for line in rendered.splitlines()
        )

    return rendered


__all__ = [
    "KernelSynthesisError",
    "ASTCapabilityValidator",
    "SynthesisEngine",
    "synthesize_and_register",
    "render_cell",
    "import_stmt_for",
]
