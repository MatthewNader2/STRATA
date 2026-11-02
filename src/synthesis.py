"""
src/synthesis.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Dynamic MicroCell Synthesizer & Capability-Sandboxed Inductive Kernel Generator.
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
                    live_docs = asyncio.run_coroutine_threadsafe(live_docs, loop).result()
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
# Inductive State Kernel Synthesizer (Pillar 2: Cognitive State Morphisms)
# --------------------------------------------------------------------------- #

class InductiveKernelGenerator:
    """
    Inductive State Kernel Synthesizer.
    Synthesizes pure mathematical / algorithmic state transition kernels (f: State -> State)
    for typed holes within category-theoretic macro envelopes (e.g. game loop update,
    simulation step, algorithmic transformations).

    Guarantees:
      - Capability-based AST Permit-List validation (zero imports, I/O, or dunder bypasses).
      - Explicit statement unwrapping (strips `def` wrappers, `return`, `assert`, and `print()`).
      - Counter-Example Guided Inductive Synthesis (CEGIS) repair hook (`repair_kernel`).
      - Fails loudly with `KernelSynthesisError` instead of emitting dummy zero-values.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self._model_manager = model_manager
        self._validator = ASTCapabilityValidator()

    @property
    def model_manager(self) -> ModelManager:
        if self._model_manager is None:
            self._model_manager = ModelManager.get_instance()
        return self._model_manager

    def validate_safety(self, code: str, allowed_vars: Optional[Set[str]] = None) -> Tuple[bool, Optional[str]]:
        """
        Validates that a code snippet is a pure, capability-safe state transition
        using ASTCapabilityValidator.
        """
        if not code or not code.strip():
            return False, "Empty kernel code"
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError in kernel code: {e}"

        if not tree.body:
            return False, "Kernel AST contains no executable statements"

        return self._validator.validate(tree, allowed_vars=allowed_vars)

    def generate_kernel(
        self,
        task_description: str,
        state_inputs: Dict[str, str],
        state_outputs: Dict[str, str],
        domain_context: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        """
        Synthesizes a pure Python state transition snippet for the given task and state signatures.
        Raises KernelSynthesisError if no valid, capability-safe kernel can be synthesized.
        """
        in_lines = "\n".join(f"- {name}: {t}" for name, t in state_inputs.items()) or "- (none)"
        out_lines = "\n".join(f"- {name}: {t}" for name, t in state_outputs.items()) or "- (none)"
        ctx_text = f"\nContext & Constants:\n{domain_context}" if domain_context else ""

        system_prompt = (
            "You are an expert computational kernel synthesizer.\n"
            "Synthesize a PURE Python code snippet implementing an inductive state transition function.\n"
            "STRICT RULES:\n"
            "1. Output ONLY executable Python code statements. No markdown code fences, no explanations, no def function wrappers.\n"
            "2. PURE LOGIC ONLY: Read from input state variables and update/assign output state variables.\n"
            "3. NO I/O, NO print(), NO assert, NO input(), NO imports of any module.\n"
            "4. Keep code clean, bounded, and mathematically correct."
        )

        user_prompt = (
            f"State Transition Goal: {task_description}\n\n"
            f"Input State Variables:\n{in_lines}\n\n"
            f"Output State Variables to Update/Assign:\n{out_lines}"
            f"{ctx_text}\n\n"
            f"Write the pure Python statements:"
        )

        raw_code = ""
        mgr = self.model_manager
        if mgr.profile is not None:
            try:
                response = mgr.generate_text(
                    user_prompt,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )
                raw_code = extract_code_from_llm_response(response).strip()
            except Exception as e:
                logger.warning("[INDUCTIVE KERNEL] ModelManager generation failed: %s", e)

        if not raw_code:
            raw_code = self._try_direct_local_generation(user_prompt, system_prompt, max_tokens)

        if not raw_code:
            raise KernelSynthesisError(
                f"Inductive synthesis failed for '{task_description}': no LLM output generated."
            )

        clean_code = self._sanitize_and_unwrap_kernel(raw_code, state_outputs)
        allowed_state_vars = set(state_inputs.keys()) | set(state_outputs.keys())
        is_safe, error_msg = self.validate_safety(clean_code, allowed_vars=allowed_state_vars)
        if not is_safe:
            logger.error("[INDUCTIVE KERNEL] Synthesized code failed capability check: %s", error_msg)
            raise KernelSynthesisError(
                f"Synthesized kernel for '{task_description}' violated capability safety: {error_msg}"
            )

        logger.info("[INDUCTIVE KERNEL] Successfully synthesized and validated kernel for '%s'", task_description)
        return clean_code

    def repair_kernel(
        self,
        failing_code: str,
        task_description: str,
        traceback_error: str,
        state_inputs: Dict[str, str],
        state_outputs: Dict[str, str],
        failing_inputs: Optional[Dict[str, Any]] = None,
        max_tokens: int = 512,
    ) -> str:
        """
        Counter-Example Guided Inductive Synthesis (CEGIS) kernel repair (Track 8.1).
        Accepts a failing kernel snippet, target invariant/goal, runtime traceback,
        and optional counter-example input values, returning a surgically repaired
        and capability-validated kernel.
        """
        in_lines = "\n".join(f"- {name}: {t}" for name, t in state_inputs.items()) or "- (none)"
        out_lines = "\n".join(f"- {name}: {t}" for name, t in state_outputs.items()) or "- (none)"
        ce_text = f"\nFailing Counter-Example Inputs:\n{failing_inputs}" if failing_inputs else ""

        system_prompt = (
            "You are an expert CEGIS (Counter-Example Guided Inductive Synthesis) repair engine.\n"
            "Fix the runtime error or invariant violation in the provided pure Python state transition kernel.\n"
            "STRICT RULES:\n"
            "1. Output ONLY the corrected executable Python statements. No markdown, no explanations, no def wrappers.\n"
            "2. Preserve all required output state variable assignments.\n"
            "3. NO imports, NO print(), NO assert, NO I/O, NO dunder attributes."
        )

        user_prompt = (
            f"State Transition Goal: {task_description}\n\n"
            f"Input State Variables:\n{in_lines}\n\n"
            f"Required Output State Variables:\n{out_lines}"
            f"{ce_text}\n\n"
            f"Failing Kernel Code:\n{failing_code}\n\n"
            f"Runtime Exception / Traceback:\n{traceback_error}\n\n"
            f"Write the corrected pure Python statements:"
        )

        raw_code = ""
        mgr = self.model_manager
        if mgr.profile is not None:
            try:
                response = mgr.generate_text(
                    user_prompt,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )
                raw_code = extract_code_from_llm_response(response).strip()
            except Exception as e:
                logger.warning("[INDUCTIVE KERNEL] CEGIS repair generation failed: %s", e)

        if not raw_code:
            raw_code = self._try_direct_local_generation(user_prompt, system_prompt, max_tokens)

        if not raw_code:
            raise KernelSynthesisError(
                f"CEGIS repair failed for '{task_description}': model produced no repair candidate."
            )

        clean_code = self._sanitize_and_unwrap_kernel(raw_code, state_outputs)
        allowed_state_vars = set(state_inputs.keys()) | set(state_outputs.keys())
        is_safe, error_msg = self.validate_safety(clean_code, allowed_vars=allowed_state_vars)
        if not is_safe:
            raise KernelSynthesisError(
                f"CEGIS repaired kernel for '{task_description}' failed capability validation: {error_msg}"
            )

        return clean_code

    def _sanitize_and_unwrap_kernel(self, code: str, state_outputs: Dict[str, str]) -> str:
        """
        Extracts clean, pure executable statements from LLM output.
        If the model wrapped the logic in a `def func(...):` or appended `assert` statements
        or `print()` calls, this strips them and extracts only the core state-transition statements.
        """
        dedented = textwrap.dedent(code).strip()
        try:
            tree = ast.parse(dedented)
        except SyntaxError:
            return code

        out_var = list(state_outputs.keys())[0] if state_outputs else None

        def _filter_statements(stmts: List[ast.stmt]) -> List[ast.stmt]:
            cleaned: List[ast.stmt] = []
            has_out_assign = any(
                isinstance(s, (ast.Assign, ast.AnnAssign))
                and any(
                    (isinstance(t, ast.Name) and t.id == out_var)
                    for t in (s.targets if isinstance(s, ast.Assign) else [s.target])
                )
                for s in stmts
            )
            for stmt in stmts:
                # 1. Strip `assert ...` statements (ast.Assert is a statement, not ast.Call)
                if isinstance(stmt, ast.Assert):
                    continue
                # 2. Convert or strip `return ...` statements from unwrapped functions
                if isinstance(stmt, ast.Return):
                    if stmt.value is not None and out_var and not has_out_assign:
                        cleaned.append(
                            ast.Assign(
                                targets=[ast.Name(id=out_var, ctx=ast.Store())],
                                value=stmt.value,
                            )
                        )
                        has_out_assign = True
                    continue
                # 3. Strip standalone docstrings and `print(...)` call expressions
                if isinstance(stmt, ast.Expr):
                    if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                        continue
                    if isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        if isinstance(func, ast.Name) and func.id == "print":
                            continue
                cleaned.append(stmt)
            return cleaned

        # If top-level contains a FunctionDef, unwrap its body statements
        fn_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if fn_nodes:
            fn = fn_nodes[0]
            body_stmts = _filter_statements(fn.body)
            if body_stmts:
                try:
                    mod = ast.Module(body=body_stmts, type_ignores=[])
                    return ast.unparse(mod)
                except Exception as e:
                    logger.debug("[INDUCTIVE KERNEL] Function unwrap unparse fallback: %s", e)

        clean_stmts = _filter_statements(tree.body)
        try:
            if clean_stmts:
                mod = ast.Module(body=clean_stmts, type_ignores=[])
                return ast.unparse(mod)
            return ""
        except Exception as e:
            logger.debug("[INDUCTIVE KERNEL] Statement unparse fallback: %s", e)
            return code

    def _try_direct_local_generation(self, user_prompt: str, system_prompt: str, max_tokens: int) -> str:
        """Attempts direct generation via local sub-1B GGUF model if ModelManager had no active profile."""
        try:
            from config import MODELS_DIR
            from llama_cpp import Llama
            from inference import select_optimal_llm
            try:
                from .router import HardwareProfiler
            except (ImportError, ValueError):
                try:
                    from router import HardwareProfiler
                except Exception:
                    HardwareProfiler = None

            device = HardwareProfiler.get_llm_device() if HardwareProfiler else "cpu"
            n_gpu = -1 if device in ("cuda", "mps") else 0

            best_llm = select_optimal_llm()
            llm_dir = os.path.join(MODELS_DIR, "llms", best_llm)
            if os.path.isdir(llm_dir):
                ggufs = [f for f in os.listdir(llm_dir) if f.endswith(".gguf")]
                if ggufs:
                    chosen_gguf = next((f for f in ggufs if "shard" not in f), ggufs[0])
                    model_path = os.path.join(llm_dir, chosen_gguf)
                    llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=n_gpu, verbose=False)
                    res = llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.2,
                    )
                    text = res["choices"][0]["message"]["content"]
                    return extract_code_from_llm_response(text).strip()
        except Exception as e:
            logger.debug("[INDUCTIVE KERNEL] Direct local generation fallback: %s", e)
        return ""

    def splice_kernel(self, envelope_template: str, slot_name: str, kernel_code: str) -> str:
        """
        Splices an inductive state kernel into a macro envelope template at `{slot_name}`
        with exact indentation alignment and AST validation.
        """
        placeholder = f"{{{slot_name}}}"
        if placeholder not in envelope_template:
            return envelope_template

        base_indent = ""
        target_line = None
        for line in envelope_template.splitlines():
            stripped = line.lstrip(" \t")
            if stripped.startswith(placeholder):
                base_indent = line[: len(line) - len(stripped)]
                target_line = line
                break

        indented_lines = []
        for line in kernel_code.splitlines():
            if line.strip():
                indented_lines.append(base_indent + line)
            else:
                indented_lines.append("")
        indented_kernel = "\n".join(indented_lines)

        if target_line:
            spliced = envelope_template.replace(target_line, indented_kernel)
        else:
            spliced = envelope_template.replace(placeholder, indented_kernel)

        return spliced


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


def _template_initializes_container(template: str) -> Optional[str]:
    """
    AST inspection: detect whether the template initializes the output
    container variable (_nstl_ph_output_var) to an empty container and return its container kind.
    """
    if not template or "{output_var}" not in template:
        return None
    sanitized = _sanitize_template_for_ast(template)
    try:
        parsed = ast.parse(sanitized)
    except Exception:
        return None

    constructors = {
        "set": "set", "frozenset": "set",
        "dict": "dict", "OrderedDict": "dict", "defaultdict": "dict", "Counter": "dict",
        "list": "list", "tuple": "tuple", "deque": "deque",
    }
    for node in ast.walk(parsed):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            is_out = any(
                isinstance(t, ast.Name) and t.id == "_nstl_ph_output_var"
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
    AST inspection: scans the sanitized template AST for an `ast.For` node
    and extracts the loop variable name from node.target.
    """
    if not template or ("for " not in template and "for\t" not in template):
        return None
    sanitized = _sanitize_template_for_ast(template)
    try:
        parsed = ast.parse(sanitized)
    except Exception:
        return None

    for node in ast.walk(parsed):
        if isinstance(node, ast.For):
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
        from .unification import UnificationGate, unify, Substitution
    except (ImportError, ValueError):
        from unification import UnificationGate, unify, Substitution

    indent_prefix = "    " * indent_level
    template = cell.code_template.strip()
    if not template:
        return ""

    slots = getattr(cell, "slots", {}) or {}
    bound_slots = getattr(cell, "bound_slots", {}) or {}

    out_var = bindings.get(
        "output_var",
        f"_{getattr(cell.primary_output, 'state', None) or 'out'}",
    )

    loop_kind = _declared_loop_kind(cell)
    init_container = _template_initializes_container(template)
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
    slots_items = slots.items() if isinstance(slots, dict) else [(s, {}) for s in slots]
    for slot_name, slot_spec in slots_items:
        child_nodes = bound_slots.get(slot_name, [])
        if not child_nodes:
            # Unified Continuum (Track 6.2): if no lattice cell filled this macro slot,
            # synthesize a capability-sandboxed kernel for the typed hole if an LLM is active.
            prompt_str = getattr(context, "prompt", "") if context is not None else ""
            if prompt_str:
                try:
                    gen = InductiveKernelGenerator()
                    if gen.model_manager.can_synthesize():
                        in_sigs = {k: p.type_name for k, p in getattr(cell, "inputs", {}).items()}
                        out_sigs = {out_var: getattr(cell.primary_output, "type_name", "any")}
                        slot_rendered[slot_name] = gen.generate_kernel(
                            task_description=f"{prompt_str} (slot: {slot_name})",
                            state_inputs=in_sigs,
                            state_outputs=out_sigs,
                        )
                        continue
                except Exception as synth_err:
                    logger.debug("[SYNTHESIS] Slot kernel generation fallback: %s", synth_err)
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
        if context is not None:
            if hasattr(context, "variables") and isinstance(context.variables, dict):
                enclosing_known_vars.update(context.variables.keys())
            if hasattr(context, "active_variables") and isinstance(context.active_variables, (set, list)):
                enclosing_known_vars.update(context.active_variables)
        for v in bindings.values():
            if isinstance(v, str):
                v_clean = v.strip().strip("'\"")
                if v_clean.isidentifier():
                    enclosing_known_vars.add(v_clean)
        if item_var and item_var.isidentifier():
            enclosing_known_vars.add(item_var)
        if out_var and out_var.isidentifier():
            enclosing_known_vars.add(out_var)

        for child in child_nodes:
            child_bindings: Dict[str, Any] = {}
            for p_name, p_sig in child.inputs.items():
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
                elif p_name in enclosing_known_vars:
                    child_bindings[p_name] = p_name
                elif getattr(p_sig, "default_value", None) is not None:
                    child_bindings[p_name] = str(p_sig.default_value)

            child_out_var = _unique_var(child, "out")
            child_bindings["output_var"] = child_out_var
            last_metric = child_out_var
            enclosing_known_vars.add(child_out_var)

            if (
                bool(getattr(child, "bound_slots", None))
                or getattr(child, "node_type", "") == "macro"
                or bool(getattr(child, "slots", None))
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
                t_name = str(getattr(out_sig, "type_name", "") or "").lower()
                if "set" in t_name:
                    container_type = "set"
                elif "dict" in t_name or "map" in t_name:
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

    rendered = template
    if is_extremum_loop and f"{metric_var} = None" not in rendered:
        rendered = f"{metric_var} = None\n" + rendered
    elif is_fold_loop and f"{out_var} = " not in rendered and f"{out_var}=" not in rendered:
        init_val = getattr(cell, "init_val", None)
        if init_val is None and isinstance(getattr(cell, "tree_meta", None), dict):
            init_val = cell.tree_meta.get("init_val") or cell.tree_meta.get("initial_value")
        init_expr = str(init_val) if init_val is not None else "None"
        rendered = f"{out_var} = {init_expr}\n" + rendered

    for slot_name, slot_code in slot_rendered.items():
        placeholder = f"{{{slot_name}}}"
        if placeholder not in rendered:
            continue
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

    rendered = safe_substitute_template(rendered, bindings)

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
    "InductiveKernelGenerator",
]
