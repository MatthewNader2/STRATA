"""
src/preflight.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Static Pre-Flight Linter before GEVR Sandbox Execution.

Structural invariants enforced post-binding:
1. Universal Literal Consumption: Every universal literal extracted at L0 intent
   (identifiers, column names, file paths) is consumed by at least one bound port on
   the final path, or explicitly waived.
2. Data-Bearing Port Integrity: No call position receives a bare None where the bound
   port's role is data-bearing (feature_input, target_input, data_input, model_input,
   source_data, model_sink).
3. Structural Estimator Contract: For any cell whose signature shape matches a
   *supervised* estimator contract (features + target) AND whose prompt carries a
   supervised-learning verb, both feature and target ports must be bound when the
   prompt declares a prediction/training relation (>= 2 data identifiers extracted
   at L0). Unsupervised tasks (clustering, dimensionality reduction, anomaly
   detection) are explicitly exempt.

All role / verb classification is delegated to TypeRegistry. This module contains no
domain-specific identifiers.
"""

from __future__ import annotations
import ast
import functools
import importlib.util
import sys
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass, field

try:
    from .lattice import UNRESOLVED_PORT, TypeRegistry, AlgebraicSignature, PortSignature
    from .tokenizer import CellTokenizer
    from .unification import (
        ExecutionContext, unify, _shape_compatible, _is_sink_cell,
        resolve_typed_binding_record, TypedBindingRecord, _declared_role_semantics,
        _get_p_field, _enrich_port_signature_qualifiers, _collect_propagated_qualifiers,
    )
except (ImportError, ValueError):
    from lattice import UNRESOLVED_PORT, TypeRegistry, AlgebraicSignature, PortSignature
    from tokenizer import CellTokenizer
    from unification import (
        ExecutionContext, unify, _shape_compatible, _is_sink_cell,
        resolve_typed_binding_record, TypedBindingRecord, _declared_role_semantics,
        _get_p_field, _enrich_port_signature_qualifiers, _collect_propagated_qualifiers,
    )


@dataclass
class VariableSchema:
    """Tracks active and invalidated composite component schemas per variable (R5-3)."""
    known_components: Optional[Set[str]] = None
    removed_components: Set[str] = field(default_factory=set)
    added_components: Set[str] = field(default_factory=set)
    removed_by: Dict[str, str] = field(default_factory=dict)



class _DynamicDataBearingRoles(frozenset):
    """Dynamic data-bearing roles proxy backed by TypeRegistry."""
    def __contains__(self, item):
        return str(item).lower() in TypeRegistry.get_instance().get_data_bearing_roles()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_data_bearing_roles())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_data_bearing_roles())


DATA_BEARING_ROLES = _DynamicDataBearingRoles()


class _DynamicConventionalEstimatorVerbs(frozenset):
    """Dynamic conventional estimator verbs proxy backed by TypeRegistry."""
    def __contains__(self, item):
        return str(item).lower() in TypeRegistry.get_instance().get_estimator_verbs()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_estimator_verbs())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_estimator_verbs())


CONVENTIONAL_ESTIMATOR_VERBS = _DynamicConventionalEstimatorVerbs()


# ---------------------------------------------------------------------------
# Unbound-value detection (domain-agnostic)
# ---------------------------------------------------------------------------

# String sentinels that indicate an unbound slot. Case-insensitive.
_UNBOUND_STRING_SENTINELS: Set[str] = {
    "none",
    "null",
    "<unresolved>",
    "<unbound>",
}

try:
    _UNRESOLVED_PORT_SENTINEL = str(UNRESOLVED_PORT).strip().lower()
except Exception:
    _UNRESOLVED_PORT_SENTINEL = ""


@functools.lru_cache(maxsize=2048)
def _cached_find_spec(mod_name: str) -> bool:
    try:
        return importlib.util.find_spec(mod_name) is not None
    except Exception:
        return False


def _is_unbound(value: Any) -> bool:
    """Return True when a port binding value represents an unbound/None slot."""
    if value is None or value is UNRESOLVED_PORT:
        return True
    try:
        s = str(value).strip().lower()
    except Exception:
        return False
    if s in _UNBOUND_STRING_SENTINELS:
        return True
    if _UNRESOLVED_PORT_SENTINEL and s == _UNRESOLVED_PORT_SENTINEL:
        return True
    return False


def _check_callable_callees(tree: ast.AST, violations: List[str]) -> None:
    """
    Host-language sanity check (domain-agnostic): a call whose callee is a
    non-callable CONSTANT (a string, number, bool...) can never execute
    successfully -- it is the signature of an unfilled template placeholder
    reaching emission ("var = 'X'(var_1)"). Catching it here keeps the failure
    loud and honest at synthesis time instead of at sandbox time.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Constant):
            violations.append(
                f"Synthesized code calls a non-callable constant "
                f"{node.func.value!r} as a function (unfilled template placeholder "
                f"reached emission)."
            )


# ---------------------------------------------------------------------------
# Errors / results
# ---------------------------------------------------------------------------

class PreflightLintError(Exception):
    """Raised when synthesized code fails static structural pre-flight validation."""
    def __init__(self, message: str, violations: Optional[List[str]] = None, structured_violations: Optional[List[PreflightViolation]] = None):
        super().__init__(message)
        self.violations = violations or [message]
        self.structured_violations = structured_violations or []


@dataclass
class PreflightViolation:
    check_id: str
    message: str
    cell_id: Optional[str] = None
    port_name: Optional[str] = None
    variable_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightLintResult:
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    structured_violations: List[PreflightViolation] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            msg = "Pre-flight lint validation failed:\n  - " + "\n  - ".join(self.violations)
            raise PreflightLintError(msg, violations=self.violations, structured_violations=self.structured_violations)


# ---------------------------------------------------------------------------
# Linter
# ---------------------------------------------------------------------------

class PreflightLinter:
    """
    Domain-agnostic static validator inspecting pipeline bindings and synthesized
    ASTs strictly through structural contracts and port roles supplied by the
    TypeRegistry. No domain-specific identifier, verb, or role name is hardcoded.
    """

    # ------------------------------------------------------------------
    # Registry accessors (all classification flows through here)
    # ------------------------------------------------------------------

    @staticmethod
    def _registry() -> Any:
        return TypeRegistry.get_instance()

    @classmethod
    def _role_set(cls, *accessor_names: str) -> Set[str]:
        """Union of role-name sets returned by any available registry accessor."""
        reg = cls._registry()
        roles: Set[str] = set()
        for name in accessor_names:
            getter = getattr(reg, name, None)
            if callable(getter):
                try:
                    roles.update(str(r).lower() for r in getter())
                except Exception:
                    continue
        return roles

    @classmethod
    def _verb_set(cls, *accessor_names: str) -> Set[str]:
        """Union of verb sets returned by any available registry accessor."""
        reg = cls._registry()
        verbs: Set[str] = set()
        for name in accessor_names:
            getter = getattr(reg, name, None)
            if callable(getter):
                try:
                    verbs.update(str(v).lower() for v in getter())
                except Exception:
                    continue
        return verbs

    # ------------------------------------------------------------------
    # Structural predicates
    # ------------------------------------------------------------------

    @classmethod
    def _port_role(cls, port_sig: Any) -> str:
        role = getattr(port_sig, "port_role", None) or getattr(port_sig, "derived_role", "") or ""
        return str(role).lower()

    @classmethod
    def _port_is_optional(cls, port_sig: Any) -> bool:
        """A port is optional when its signature explicitly marks it so."""
        if getattr(port_sig, "required", None) is False:
            return True
        if getattr(port_sig, "optional", None) is True:
            return True
        return False

    @classmethod
    def _is_object_receiver_role(cls, role: str) -> bool:
        """True for the declared data-bearing roles that carry a fitted model/estimator
        instance rather than array-like data. The role names are DECLARED tree data
        (verification_semantics.model_roles) -- the same source Check 2 draws on."""
        try:
            model_roles = {
                str(r).lower()
                for r in TypeRegistry.get_instance().get_verification_semantics("model_roles")
            }
        except Exception:
            model_roles = set()
        return bool(role and role in model_roles)

    @classmethod
    def _verify_carrier_extent_epochs(
        cls, pipeline_bindings: List[Tuple[Any, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Universal validation: Carrier extent domains are verified via universal contracts."""
        return []

    @classmethod
    def _verify_causal_key_def_use(
        cls, pipeline_bindings: List[Tuple[Any, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Removed: tabular causal key def-use tracking is excised."""
        return []

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    @classmethod
    def lint(
        cls,
        pipeline_bindings: List[Tuple[Any, Dict[str, Any]]],
        prompt: str = "",
        code_str: Optional[str] = None,
        waived_literals: Optional[Set[str]] = None,
        orchestrator: Optional[Any] = None,
    ) -> PreflightLintResult:
        violations: List[str] = []
        warnings: List[str] = []
        structured_violations: List[PreflightViolation] = []
        waived = set(waived_literals or set())

        bound_pipeline_vars: Set[str] = set()
        for cell_item, b_dict in pipeline_bindings:
            if isinstance(b_dict, dict):
                out_v = b_dict.get("output_var")
                if out_v and not _is_unbound(out_v):
                    bound_pipeline_vars.add(str(out_v))
                for o_name in (getattr(cell_item, "outputs", {}) or {}):
                    o_val = b_dict.get(o_name)
                    if o_val and not _is_unbound(o_val):
                        bound_pipeline_vars.add(str(o_val))
                for k, v in b_dict.items():
                    if isinstance(v, str) and not _is_unbound(v):
                        if v.startswith("var_") or (k == "output_var" and v.isidentifier()) or v in bound_pipeline_vars:
                            bound_pipeline_vars.add(v)

        def _is_var(val: Any) -> bool:
            if not isinstance(val, str) or not val:
                return False
            if _is_unbound(val):
                return False
            v = val.strip()
            if v in bound_pipeline_vars or v.startswith("var_"):
                return True
            root = v.split("[")[0].strip() if "[" in v else v
            if root in bound_pipeline_vars or root.startswith("var_"):
                return True
            return False

        def _add_violation(
            check_id: str,
            message: str,
            cell_id: Optional[str] = None,
            port_name: Optional[str] = None,
            variable_name: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None,
        ) -> None:
            violations.append(message)
            structured_violations.append(
                PreflightViolation(
                    check_id=check_id,
                    message=message,
                    cell_id=cell_id,
                    port_name=port_name,
                    variable_name=variable_name,
                    details=details or {},
                )
            )

        # Extract universal literals from prompt if ExecutionContext is available.
        extracted_literals: List[Tuple[int, str, str]] = []
        try:
            extracted_literals = ExecutionContext._extract_universal_literals(prompt or "")
        except Exception:
            pass

        # -----------------------------------------------------------------
        # Check 1: Universal Literal Consumption
        # -----------------------------------------------------------------
        if extracted_literals:
            bound_values_str: Set[str] = set()
            for _cell, bindings in pipeline_bindings:
                for _p_name, val in bindings.items():
                    if val is not None:
                        bound_values_str.add(str(val).strip("'\""))

            seen_consumed_lits: Set[str] = set()

            for _, kind, lit_val in extracted_literals:
                clean_lit = lit_val.strip("'\"")
                if not clean_lit or clean_lit in waived:
                    continue

                if kind not in ("file_asset", "quoted_str", "numeric"):
                    continue

                if clean_lit in seen_consumed_lits:
                    continue
                seen_consumed_lits.add(clean_lit)

                is_consumed = any(
                    clean_lit == b or clean_lit in b
                    for b in bound_values_str
                )

                if not is_consumed and code_str:
                    try:
                        tree = ast.parse(code_str)
                        clean_lower = clean_lit.lower()
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Constant):
                                val_str = str(node.value).lower()
                                if val_str == clean_lower or clean_lower in val_str:
                                    is_consumed = True
                                    break
                            elif isinstance(node, ast.Name):
                                if node.id.lower() == clean_lower:
                                    is_consumed = True
                                    break
                            elif isinstance(node, ast.keyword):
                                if node.arg and node.arg.lower() == clean_lower:
                                    is_consumed = True
                                    break
                    except SyntaxError:
                        _add_violation("syntax_error", "Synthesized code failed static AST parsing (SyntaxError).")
                        is_consumed = False
                    except Exception:
                        is_consumed = False

                if not is_consumed:
                    msg = (
                        f"Universal literal '{clean_lit}' (kind: {kind}) extracted from prompt "
                        f"was not consumed by any bound port on the path."
                    )
                    _add_violation(
                        "unconsumed_literal",
                        msg,
                        details={"literal": clean_lit, "kind": kind}
                    )

        # -----------------------------------------------------------------
        # Variable origin tracking for prompt alignment
        # -----------------------------------------------------------------
        var_origins: Dict[str, Set[str]] = {}
        for cell, bindings in pipeline_bindings:
            if not isinstance(bindings, dict):
                continue
            out_var = bindings.get("output_var")
            if out_var and isinstance(out_var, str):
                origs: Set[str] = set()
                for inp_name, inp_val in bindings.items():
                    if inp_name == "output_var":
                        continue
                    if isinstance(inp_val, str) and _is_var(inp_val):
                        r_v = inp_val.split("[")[0].strip()
                        origs.update(var_origins.get(r_v, set()))
                    elif isinstance(inp_val, str) and not _is_unbound(inp_val):
                        clean_inp = inp_val.strip("'\"")
                        if clean_inp:
                            origs.add(clean_inp)
                var_origins[out_var] = origs

        # -----------------------------------------------------------------
        # Check 2b: declared dependency names, from the cells' own dependency
        # declarations (same data the emitter renders into import headers).
        # -----------------------------------------------------------------
        declared_dep_names: Set[str] = set()
        try:
            from .lattice import TypeRegistry
            reg_aliases = TypeRegistry.get_instance().get_all_aliases()
            declared_dep_names.update(reg_aliases.keys())
            declared_dep_names.update(reg_aliases.values())
        except Exception:
            try:
                from lattice import TypeRegistry
                reg_aliases = TypeRegistry.get_instance().get_all_aliases()
                declared_dep_names.update(reg_aliases.keys())
                declared_dep_names.update(reg_aliases.values())
            except Exception:
                pass
        import sys
        import importlib.util
        declared_dep_names.update(sys.builtin_module_names)
        for cell, _ in pipeline_bindings:
            for dep in (getattr(cell, "dependencies", None) or []) + (getattr(cell, "imports", None) or []):
                d = str(dep).strip()
                if not d:
                    continue
                if d.startswith("import "):
                    for p in d[len("import "):].split(","):
                        p = p.strip()
                        if " as " in p:
                            alias = p.split(" as ")[1].strip()
                        else:
                            alias = p.split(".")[0].strip()
                        if alias:
                            declared_dep_names.add(alias)
                elif d.startswith("from ") and " import " in d:
                    tail = d.split(" import ", 1)[1]
                    for p in tail.split(","):
                        p = p.strip()
                        if " as " in p:
                            alias = p.split(" as ")[1].strip()
                        else:
                            alias = p
                        if alias:
                            declared_dep_names.add(alias)
                else:
                    # Bare module declaration form used by the trees: "numpy as np",
                    # "pandas as pd" — the emitter renders these as import statements.
                    if " as " in d:
                        alias = d.split(" as ")[1].strip()
                    else:
                        alias = d.split(".")[0].strip()
                    if alias:
                        declared_dep_names.add(alias)

        # -----------------------------------------------------------------
        # Check 2: No bare None in required data-bearing positions
        # -----------------------------------------------------------------
        for cell, bindings in pipeline_bindings:
            for p_name, port_sig in cell.inputs.items():
                p_role = cls._port_role(port_sig) or "standard"
                if p_role not in DATA_BEARING_ROLES:
                    continue
                if cls._port_is_optional(port_sig):
                    # Optional data-bearing ports may legitimately remain unbound
                    # (e.g. an unsupervised estimator's optional target slot).
                    continue
                bound_val = bindings.get(p_name)
                if _is_unbound(bound_val):
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' has data-bearing role '{p_role}' "
                        f"but received bare None or unresolved value."
                    )
                    _add_violation("unbound_data_port", msg, cell_id=cell.cell_id, port_name=p_name, details={"port_role": p_role})

        # -----------------------------------------------------------------
        # Check 2b: Cross-Port Static Type and Carrier Compatibility
        # -----------------------------------------------------------------
        var_signatures: Dict[str, Any] = {}
        var_producers: Dict[str, Tuple[Any, str]] = {}

        for cell, bindings in pipeline_bindings:
            if not isinstance(bindings, dict):
                continue

            cell_outputs = getattr(cell, "outputs", {})
            cell_output_vars = set()
            for out_name, out_sig in cell_outputs.items():
                if out_name in bindings and isinstance(bindings[out_name], str):
                    cell_output_vars.add(bindings[out_name])
            out_v = bindings.get("output_var")
            if isinstance(out_v, str):
                cell_output_vars.add(out_v)

            for p_name, in_port_sig in getattr(cell, "inputs", {}).items():
                # Skip self-declared outputs on the cell
                if p_name in cell_outputs:
                    continue

                bound_val = bindings.get(p_name)
                if bound_val is None or _is_unbound(bound_val) or bound_val is UNRESOLVED_PORT:
                    continue

                if isinstance(bound_val, str) and bound_val in cell_output_vars:
                    continue

                record = getattr(cell, "typed_bindings", {}).get(p_name)
                if record is None:
                    record = resolve_typed_binding_record(
                        cell=cell,
                        port_name=p_name,
                        bound_val=bound_val,
                        port_sig=in_port_sig,
                        var_signatures=var_signatures,
                    )

                if record.kind == "literal" or record.root_var is None:
                    continue

                root_var = record.root_var
                resolved_sig = record.resulting_signature

                if root_var in cell_output_vars:
                    continue

                # Declared module aliases / imported symbols (R6-9): a binding whose
                # root is an alias the cells themselves declare ("numpy as np",
                # "from x import y") is defined by the emission header the emitter
                # generates from those same declarations — referencing it is not an
                # undefined variable.
                if root_var in declared_dep_names or root_var in sys.modules or (root_var.isidentifier() and _cached_find_spec(root_var)):
                    continue

                if resolved_sig is None:
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' references undefined pipeline variable '{root_var}' "
                        f"with no producing cell in DAG history."
                    )
                    _add_violation("undefined_variable", msg, cell_id=cell.cell_id, port_name=p_name, variable_name=root_var)
                    continue

                prod_cell, prod_port = var_producers.get(root_var, (None, None))
                prod_id = prod_cell.cell_id if prod_cell else "unknown"

                # 1. Structural Carrier & Shape Compatibility
                if not _shape_compatible(resolved_sig, in_port_sig):
                    res_quals_raw = _get_p_field(resolved_sig, "qualifiers", None) or getattr(getattr(resolved_sig, "signature", None), "qualifiers", frozenset()) or frozenset()
                    in_rej_raw = _get_p_field(in_port_sig, "rejected_qualifiers", None) or getattr(getattr(in_port_sig, "signature", None), "rejected_qualifiers", frozenset()) or frozenset()

                    flat_res_quals = set()
                    for q in res_quals_raw:
                        if isinstance(q, (tuple, list, set, frozenset)):
                            flat_res_quals.update(str(x).lower() for x in q)
                        else:
                            flat_res_quals.add(str(q).lower())

                    flat_in_rej = set()
                    for q in in_rej_raw:
                        if isinstance(q, (tuple, list, set, frozenset)):
                            flat_in_rej.update(str(x).lower() for x in q)
                        else:
                            flat_in_rej.add(str(q).lower())

                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' has carrier/shape mismatch with "
                        f"bound variable '{bound_val}' (produced by '{prod_id}' port '{prod_port}')."
                    )
                    _add_violation(
                        "shape_carrier_mismatch",
                        msg,
                        cell_id=cell.cell_id,
                        port_name=p_name,
                        variable_name=root_var,
                        details={
                            "bound_val": bound_val,
                            "producer_cell_id": prod_id,
                            "producer_port": prod_port,
                            "resolved_qualifiers": sorted(list(flat_res_quals)),
                            "rejected_qualifiers": sorted(list(flat_in_rej)),
                        },
                    )
                    continue

                # 2. Algebraic Type Unification
                p_term = getattr(resolved_sig, "signature", resolved_sig)
                c_term = getattr(in_port_sig, "signature", in_port_sig)
                sub = unify(p_term, c_term)
                if sub is None:
                    prod_t = getattr(p_term, "type_name", str(p_term))
                    cons_t = getattr(c_term, "type_name", str(c_term))
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' (expected type '{cons_t}') failed "
                        f"type unification with bound variable '{bound_val}' (producer type '{prod_t}' from '{prod_id}')."
                    )
                    _add_violation(
                        "type_unification_failure",
                        msg,
                        cell_id=cell.cell_id,
                        port_name=p_name,
                        variable_name=root_var,
                        details={
                            "bound_val": bound_val,
                            "producer_cell_id": prod_id,
                            "producer_port": prod_port,
                            "producer_type": prod_t,
                            "consumer_type": cons_t,
                        },
                    )

            if _is_sink_cell(cell):
                continue

            # Calculate propagated qualifiers from input variables
            prop_quals = _collect_propagated_qualifiers(cell, bindings, var_signatures)

            # Register cell outputs for subsequent consumers
            prim_out = getattr(cell, "primary_output", None)
            if prop_quals and prim_out is not None:
                prim_out = _enrich_port_signature_qualifiers(prim_out, prop_quals)

            if prim_out is not None:
                p_name = getattr(prim_out, "name", None)
                if p_name and p_name in bindings:
                    v = bindings[p_name]
                    if isinstance(v, str) and v.isidentifier():
                        var_signatures[v] = prim_out
                        var_producers[v] = (cell, p_name)

            for out_name, out_sig in cell_outputs.items():
                if prop_quals:
                    out_sig = _enrich_port_signature_qualifiers(out_sig, prop_quals)
                if out_name in bindings:
                    v = bindings[out_name]
                    if isinstance(v, str) and v.isidentifier():
                        var_signatures[v] = out_sig
                        var_producers[v] = (cell, out_name)

            out_var = bindings.get("output_var")
            if isinstance(out_var, str) and out_var.isidentifier() and out_var not in var_signatures:
                first_sig = prim_out or (next(iter(cell_outputs.values())) if cell_outputs else None)
                if first_sig is not None:
                    if prop_quals:
                        first_sig = _enrich_port_signature_qualifiers(first_sig, prop_quals)
                    var_signatures[out_var] = first_sig
                    var_producers[out_var] = (cell, "output_var")

        # -----------------------------------------------------------------
        # Check 3: Dataflow Output Liveness (R5-2)
        # Every non-goal, non-sink cell MUST have its output consumed downstream.
        # -----------------------------------------------------------------
        n_steps = len(pipeline_bindings)
        _registry_liveness = cls._registry()
        for i, (cell, bindings) in enumerate(pipeline_bindings):
            if not isinstance(bindings, dict):
                continue
            if _is_sink_cell(cell) or not getattr(cell, "outputs", None):
                continue
            # The final step is the pipeline goal / return value
            if i == n_steps - 1:
                continue

            # Output variables produced by this step
            produced_vars: Set[str] = set()
            for out_name in getattr(cell, "outputs", {}).keys():
                v = bindings.get(out_name)
                if isinstance(v, str) and _is_var(v):
                    produced_vars.add(v)
            out_v = bindings.get("output_var")
            if isinstance(out_v, str) and _is_var(out_v):
                produced_vars.add(out_v)

            if not produced_vars:
                continue

            # Check if any downstream cell reads at least one produced variable
            is_consumed = False
            for downstream_cell, down_bindings in pipeline_bindings[i + 1 :]:
                if not isinstance(down_bindings, dict):
                    continue
                for p_k, down_val in down_bindings.items():
                    if down_val is None:
                        continue
                    # Check typed bindings record first
                    d_rec = getattr(downstream_cell, "typed_bindings", {}).get(p_k)
                    if d_rec and d_rec.root_var and d_rec.root_var in produced_vars:
                        is_consumed = True
                        break
                    # String check
                    down_str = str(down_val)
                    for pv in produced_vars:
                        if pv == down_str or f"{pv}[" in down_str or f"{pv}." in down_str or f"({pv}" in down_str:
                            is_consumed = True
                            break
                    if is_consumed:
                        break
                if is_consumed:
                    break

            if not is_consumed:
                # Invariant: Stateful handles or object-receiver carrier outputs MUST be consumed downstream
                # unless this is the final pipeline step. They are NEVER exempt from liveness simply by matching a clause.
                has_handle_output = False
                for out_name, out_sig in getattr(cell, "outputs", {}).items():
                    s_kind = str(getattr(out_sig, "structure_kind", "") or "").lower()
                    p_role = str(getattr(out_sig, "port_role", "") or "").lower()
                    if s_kind == "handle" or cls._is_object_receiver_role(p_role):
                        has_handle_output = True
                        break

                if has_handle_output and i < n_steps - 1:
                    dead_v = next(iter(produced_vars))
                    msg = (
                        f"Cell '{cell.cell_id}' produced linear handle / stateful resource '{dead_v}' "
                        f"which is never consumed downstream (abandoned linear handle)."
                    )
                    _add_violation(
                        "dead_handle_output",
                        msg,
                        cell_id=cell.cell_id,
                        variable_name=dead_v,
                        details={"cell_id": cell.cell_id, "output_var": dead_v, "produced_vars": list(produced_vars)},
                    )
                    continue

                # Terminal observation, goal, sink, or clause witness exemption
                role = str(getattr(cell, "node_role", "") or "").lower()
                term_roles = _declared_role_semantics("terminal_roles")
                is_terminal = (role in term_roles) if term_roles else False
                if not is_terminal:
                    try:
                        from planner import _active_vocab
                        is_terminal = _active_vocab().is_terminal(cell)
                    except Exception:
                        pass
                # In graph-theoretic dataflow liveness:
                # A step is live if its outputs are consumed, OR if it represents a terminal deliverable / sink,
                # OR if it was explicitly synthesized as the witness to a declared user specification clause.
                if (
                    is_terminal
                    or getattr(cell, "is_goal", False)
                    or getattr(cell, "matched_clause_idx", None) is not None
                ):
                    continue
                dead_v = next(iter(produced_vars))
                msg = (
                    f"Cell '{cell.cell_id}' produced output variable '{dead_v}' which is never "
                    f"consumed by any downstream cell (dead output)."
                )
                _add_violation(
                    "dead_output",
                    msg,
                    cell_id=cell.cell_id,
                    variable_name=dead_v,
                    details={"cell_id": cell.cell_id, "output_var": dead_v, "produced_vars": list(produced_vars)},
                )

        # -----------------------------------------------------------------
        # Check 4: Monotonic Clause Ordering & Dependency Inversion (R5-2)
        # -----------------------------------------------------------------
        if prompt:
            path_cells = [c for c, _ in pipeline_bindings]
            try:
                try:
                    from .route_methods.base import RouteMethod
                except (ImportError, ValueError):
                    from route_methods.base import RouteMethod
                RouteMethod.tag_cells_with_clause_indices(path_cells, prompt, orchestrator=orchestrator)
                for idx, c in enumerate(path_cells):
                    if pipeline_bindings[idx][0] is not c:
                        pipeline_bindings[idx] = (c, pipeline_bindings[idx][1])
            except Exception:
                pass

        # Track which variables each cell produces and consumes
        cell_producers: Dict[str, Tuple[int, Any]] = {}  # var_name -> (step_idx, cell)
        for i, (cell, bindings) in enumerate(pipeline_bindings):
            if not isinstance(bindings, dict):
                continue
            for out_name in getattr(cell, "outputs", {}).keys():
                v = bindings.get(out_name)
                if isinstance(v, str) and _is_var(v):
                    cell_producers[v] = (i, cell)
            out_v = bindings.get("output_var")
            if isinstance(out_v, str) and _is_var(out_v):
                cell_producers[out_v] = (i, cell)

        # Check for order inversions across dependent cell pairs
        for j, (cons_cell, cons_bindings) in enumerate(pipeline_bindings):
            if not isinstance(cons_bindings, dict):
                continue
            cons_clause = getattr(cons_cell, "matched_clause_idx", None)
            if cons_clause is None:
                continue

            for p_k, down_val in cons_bindings.items():
                if down_val is None:
                    continue
                d_rec = getattr(cons_cell, "typed_bindings", {}).get(p_k)
                root_v = d_rec.root_var if d_rec else None
                if not root_v and isinstance(down_val, str) and _is_var(down_val):
                    root_v = down_val

                if root_v and root_v in cell_producers:
                    prod_idx, prod_cell = cell_producers[root_v]
                    prod_clause = getattr(prod_cell, "matched_clause_idx", None)
                    if prod_clause is not None and prod_clause > cons_clause:
                        msg = (
                            f"Order inversion: Cell '{prod_cell.cell_id}' (serving clause {prod_clause}) "
                            f"produces variable '{root_v}' consumed by earlier-clause Cell '{cons_cell.cell_id}' "
                            f"(serving clause {cons_clause})."
                        )
                        _add_violation(
                            "order_inversion",
                            msg,
                            cell_id=prod_cell.cell_id,
                            variable_name=root_v,
                            details={
                                "producer": prod_cell.cell_id,
                                "producer_clause": prod_clause,
                                "consumer": cons_cell.cell_id,
                                "consumer_clause": cons_clause,
                                "variable": root_v,
                            },
                        )

            # Also check if cons_cell is an earlier-clause data cleaner/transformer
            # placed AFTER a later-clause estimator or consumer that operated on the uncleaned root carrier
            clean_effects = set()
            try:
                from .lattice import TypeRegistry
                reg_p = TypeRegistry.get_instance()
                clean_effects = {str(e).lower() for e in (set(reg_p.get_verification_semantics("clean_effects")) | set(reg_p.get_verification_semantics("dedup_effects")))}
            except Exception:
                try:
                    from lattice import TypeRegistry
                    reg_p = TypeRegistry.get_instance()
                    clean_effects = {str(e).lower() for e in (set(reg_p.get_verification_semantics("clean_effects")) | set(reg_p.get_verification_semantics("dedup_effects")))}
                except Exception:
                    reg_p = None
                    clean_effects = set()

            if hasattr(cons_cell, "effect_names") and callable(cons_cell.effect_names):
                cons_effects = cons_cell.effect_names()
            else:
                try:
                    from lattice import extract_effect_names
                    cons_effects = extract_effect_names(cons_cell)
                except Exception:
                    cons_effects = set()
                    for eff in getattr(cons_cell, "effects", []) or []:
                        if isinstance(eff, dict):
                            for val in (eff.get("name"), eff.get("type"), eff.get("kind"), eff.get("property"), eff.get("value"), eff.get("description")):
                                if isinstance(val, str):
                                    cons_effects.add(val.lower())
                        elif isinstance(eff, str):
                            cons_effects.add(eff.lower())

            clean_roles = set(r.lower() for r in (reg_p.get_verification_semantics("clean_roles") if reg_p else []))
            clean_roles.update(r.lower() for r in (reg_p.get_verification_semantics("transform_roles") if reg_p else []))
            if not clean_roles:
                clean_roles = {"cleaner", "transformer", "filter"}
            cons_is_cleaner = bool(cons_effects & clean_effects) or (getattr(cons_cell, "node_role", "") in clean_roles)

            cons_in_v = None
            if cons_is_cleaner:
                for port_name, b_val in cons_bindings.items():
                    if isinstance(b_val, str) and _is_var(b_val):
                        port_sig = getattr(cons_cell, "inputs", {}).get(port_name)
                        p_role = getattr(port_sig, "port_role", "") if port_sig else ""
                        p_tn = getattr(getattr(port_sig, "signature", port_sig), "type_name", "") if port_sig else ""
                        if not port_sig or p_role in ("carrier", "data_input", "feature_input") or (reg_p and (reg_p.is_abstract_carrier(p_tn) or reg_p.get_abstract_carrier(p_tn) is not None)):
                            cons_in_v = b_val
                            break
            if cons_in_v:
                for prev_i in range(j):
                    prev_cell, prev_b = pipeline_bindings[prev_i]
                    prev_clause = getattr(prev_cell, "matched_clause_idx", None)
                    if prev_clause is not None and prev_clause > cons_clause:
                        if any(v == cons_in_v or (isinstance(v, str) and cons_in_v in v) for v in prev_b.values()):
                            msg = (
                                f"Order inversion: Cell '{prev_cell.cell_id}' (serving clause {prev_clause}) "
                                f"executed before Cell '{cons_cell.cell_id}' (serving clause {cons_clause}) "
                                f"on shared carrier '{cons_in_v}'."
                            )
                            _add_violation(
                                "order_inversion",
                                msg,
                                cell_id=prev_cell.cell_id,
                                variable_name=cons_in_v,
                                details={
                                    "producer": prev_cell.cell_id,
                                    "producer_clause": prev_clause,
                                    "consumer": cons_cell.cell_id,
                                    "consumer_clause": cons_clause,
                                    "variable": cons_in_v,
                                },
                            )
                            break

        # -----------------------------------------------------------------
        # Check 4c: Clause Adjacency & Provenance Gating (R5-7)
        # A data port bound to the product of a cell covering a different clause
        # than its own without clause adjacency or a matching prompt literal
        # is flagged provenance_mismatch.
        # -----------------------------------------------------------------
        prompt_clauses = []
        try:
            from planner import _segment_prompt_clauses
            prompt_clauses = _segment_prompt_clauses(prompt or "")
        except Exception:
            pass

        for j, (cell, bindings) in enumerate(pipeline_bindings):
            cons_clause = getattr(cell, "matched_clause_idx", None)
            if cons_clause is None:
                continue

            cons_clause_text = prompt_clauses[cons_clause] if cons_clause < len(prompt_clauses) else ""
            cons_clause_toks = {t.lower() for t in cons_clause_text.replace(",", " ").replace('"', " ").replace("'", " ").split()} if cons_clause_text else set()

            for p_name, val_expr in bindings.items():
                if not isinstance(val_expr, str) or not _is_var(val_expr):
                    continue

                p_sig = getattr(cell, "inputs", {}).get(p_name)
                if p_sig is None:
                    continue

                p_binds = getattr(p_sig, "binds", None)
                p_abstract = getattr(p_sig, "abstract_type", "")
                inner_sig = getattr(p_sig, "signature", p_sig)
                p_tn = str(getattr(inner_sig, "type_name", "")).lower()

                # Carriers of any modality are exempt as they carry throughout the entire pipeline
                reg_4c = cls._registry()
                is_carrier_port = (
                    getattr(p_sig, "is_carrier", False)
                    or p_binds in ("data_carrier", "carrier")
                    or reg_4c.is_abstract_carrier(p_abstract)
                    or reg_4c.is_abstract_carrier(p_tn)
                    or reg_4c.get_abstract_carrier(p_tn) is not None
                )
                if is_carrier_port:
                    continue

                # Literal parameters or component keys are exempt
                if p_binds in ("key", "component_key", "field_key") or getattr(p_sig, "port_role", None) in ("literal_parameter", "key"):
                    continue

                # Extract root variable if projection
                root_var = val_expr.split("[")[0].strip() if "[" in val_expr else val_expr.strip()
                prod_cell, _ = var_producers.get(root_var, (None, None))
                if prod_cell is None:
                    continue

                prod_clause = getattr(prod_cell, "matched_clause_idx", None)
                if prod_clause is None or prod_clause == cons_clause:
                    continue

                # Primary-dataflow exemption (R6-9): the producer's PRIMARY output
                # feeding the consumer's PRIMARY input is the pipeline's main data
                # path — the unifier wires it from the freshest value of that
                # carrier. Multi-stage prompts legitimately span clauses here
                # (normalize at clause 1 feeding a mean at clause 3), and flagging
                # the main dataflow produced false positives on correct pipelines.
                # Auxiliary ports (parameters, secondary products) keep the
                # adjacency/literal rules below.
                prod_prim_out = getattr(prod_cell, "primary_output", None)
                prod_prim_var = None
                if prod_prim_out is not None and prod_cell in [c for c, _ in pipeline_bindings]:
                    prod_bindings = next((b for c, b in pipeline_bindings if c is prod_cell), None)
                    if prod_bindings is not None:
                        cand_v = prod_bindings.get(getattr(prod_prim_out, "name", "") or "")
                        if isinstance(cand_v, str):
                            prod_prim_var = cand_v.split("[")[0].strip()
                if prod_prim_var is None:
                    cons_prim_in = getattr(cell, "primary_input", None)
                    if cons_prim_in is not None and p_name == getattr(cons_prim_in, "name", None):
                        # producer var unknown but port is primary: still require the
                        # producer-side check below via fallback below.
                        pass
                if prod_prim_var is not None and prod_prim_var == root_var:
                    cons_prim_in = getattr(cell, "primary_input", None)
                    if cons_prim_in is not None and p_name == getattr(cons_prim_in, "name", None):
                        continue

                # 1. Matching literal in consumer clause
                var_origs = set(var_origins.get(root_var, set()))
                try:
                    rec = (getattr(cell, "typed_bindings", {}) or {}).get(p_name) or resolve_typed_binding_record(cell=cell, port_name=p_name, bound_val=val_expr)
                    if rec and rec.kind == "projection" and rec.projection:
                        k = rec.projection.get("key")
                        if isinstance(k, str):
                            var_origs.add(k)
                        elif isinstance(k, (list, tuple, set)):
                            var_origs.update(str(x) for x in k)
                except Exception:
                    pass
                has_literal_match = False
                if var_origs:
                    for orig in var_origs:
                        if str(orig).lower() in cons_clause_toks:
                            has_literal_match = True
                            break

                if has_literal_match:
                    continue

                # 2. Forward DAG dataflow: producer must precede or equal consumer clause
                is_valid_flow = (prod_clause <= cons_clause)

                if not is_valid_flow:
                    msg = (
                        f"Provenance mismatch: Cell '{cell.cell_id}' port '{p_name}' bound variable '{val_expr}' "
                        f"produced by Cell '{prod_cell.cell_id}' (serving clause {prod_clause}) into earlier clause {cons_clause} "
                        f"violating forward DAG dataflow."
                    )
                    _add_violation(
                        "provenance_mismatch",
                        msg,
                        cell_id=cell.cell_id,
                        port_name=p_name,
                        variable_name=val_expr,
                        details={
                            "consumer": cell.cell_id,
                            "consumer_clause": cons_clause,
                            "producer": prod_cell.cell_id,
                            "producer_clause": prod_clause,
                            "port": p_name,
                            "variable": val_expr,
                        },
                    )


        # -----------------------------------------------------------------
        # Check 3b: Explicit Unresolved Port Detection
        # -----------------------------------------------------------------
        reg = cls._registry()
        key_binds = set(reg.get_verification_semantics("key_binds") or ())
        proj_roles = cls._role_set("get_projection_tokens")
        for cell, bindings in pipeline_bindings:
            for p_name, val in bindings.items():
                if val is UNRESOLVED_PORT:
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' has unresolved binding "
                        f"value: '{val}'"
                    )
                    _add_violation("unresolved_port", msg, cell_id=cell.cell_id, port_name=p_name, details={"value": str(val)})
                elif isinstance(val, str) and val in (
                    str(UNRESOLVED_PORT), "<UNRESOLVED>", "<unbound>"
                ):
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' has unresolved binding "
                        f"value: '{val}'"
                    )
                    _add_violation("unresolved_port", msg, cell_id=cell.cell_id, port_name=p_name, details={"value": str(val)})
                else:
                    port_sig = getattr(cell, "inputs", {}).get(p_name)
                    if port_sig and getattr(port_sig, "required", False):
                        p_role = cls._port_role(port_sig)
                        p_binds = str(getattr(port_sig, "binds", "") or "").lower()
                        if (p_binds in key_binds or p_role in proj_roles) and str(val).strip("'\"") == "":
                            msg = f"Cell '{cell.cell_id}' required key/projection port '{p_name}' has ungrounded empty binding: {val!r}"
                            _add_violation("unresolved_key_port", msg, cell_id=cell.cell_id, port_name=p_name, details={"value": str(val)})

        # -----------------------------------------------------------------
        # Check 3c: Object-Receiver Ports Must Bind a Variable, Not a Literal
        #
        # A `model_input`/`model_sink` port carries a fitted estimator instance, never
        # array-like or scalar data. The emitter's own convention is that a bound
        # reference to a prior cell's output is a bare Python identifier ("var_7");
        # anything else is a literal expression the resolver fell back to (e.g. a
        # tree that misdeclares the port's carrier). This is a syntactic check on the
        # emitted binding, not a check against any particular type name.
        # -----------------------------------------------------------------
        for cell, bindings in pipeline_bindings:
            for p_name, port_sig in cell.inputs.items():
                p_role = cls._port_role(port_sig)
                if not cls._is_object_receiver_role(p_role):
                    continue
                bound_val = bindings.get(p_name)
                if _is_unbound(bound_val) or bound_val is UNRESOLVED_PORT:
                    continue  # already reported by Check 2 / Check 3b
                val_str = str(bound_val).strip()
                if not val_str.isidentifier():
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' has data-bearing role "
                        f"'{p_role}' (a fitted model/estimator instance) but was bound to "
                        f"literal expression '{val_str}' instead of a variable reference."
                    )
                    _add_violation("object_receiver_literal", msg, cell_id=cell.cell_id, port_name=p_name, details={"port_role": p_role, "bound_val": val_str})

        # -----------------------------------------------------------------
        # Check 4: AST Syntax Validation (if code_str provided)
        # -----------------------------------------------------------------
        if code_str:
            try:
                tree = ast.parse(code_str)
                pre_cnt = len(violations)
                _check_callable_callees(tree, violations)
                for v in violations[pre_cnt:]:
                    structured_violations.append(
                        PreflightViolation(
                            check_id="callable_callee_constant",
                            message=v,
                            details={"code": code_str},
                        )
                    )
            except SyntaxError as e:
                msg = f"Synthesized code has syntax error: {e}"
                _add_violation("syntax_error", msg, details={"error": str(e)})

        # -----------------------------------------------------------------
        # Check 10: Extent domain lineage. A value computed on one carrier extent must not be
        # written into a composite carrier living on another (tree-declared extent effects).
        # -----------------------------------------------------------------
        try:
            for _ri in cls._verify_carrier_extent_epochs(pipeline_bindings):
                _add_violation(
                    "carrier_extent_misalignment",
                    f"Cell '{_ri['cell']}' writes {_ri['value_vars']} (extent epoch {_ri['value_epoch']}) into carrier "
                    f"{_ri['table_vars']} (extent epoch {_ri['table_epoch']}): the value was computed across a "
                    f"dimension/extent-altering morphism, resulting in mismatched bounds at runtime.",
                    cell_id=_ri["cell"], details=_ri)
        except Exception as _row_err:
            warnings.append(f"carrier extent analysis unavailable: {type(_row_err).__name__}: {_row_err}")


        # -----------------------------------------------------------------
        # Check 11: Tree-declared value properties. A consumer precondition that a KNOWN
        # producer postcondition contradicts (e.g. a complex FFT result bound to an estimator that
        # rejects complex data) is a static error, not a runtime surprise.
        # -----------------------------------------------------------------
        try:
            try:
                from .unification import analyze_property_contracts as _prop_analyze
            except (ImportError, ValueError):
                from unification import analyze_property_contracts as _prop_analyze
            for _pi in _prop_analyze(pipeline_bindings, orchestrator):
                _hint = (f" Cells declared to establish {_pi['property']} == {_pi['required']}: {', '.join(_pi['bridge_candidates'])}."
                         if _pi["bridge_candidates"] else "")
                _add_violation(
                    "precondition_violated",
                    f"Cell '{_pi['cell']}' port '{_pi['port']}' requires {_pi['property']} {_pi['required_operator']} {_pi['required']}, "
                    f"but the bound value {_pi['bound']} is known to have {_pi['property']} == {_pi['known']} "
                    f"(established by {_pi['established_by']}: {_pi['established_note']}).{_hint}",
                    cell_id=_pi["cell"], details=_pi)
        except Exception as _prop_err:
            warnings.append(f"property analysis unavailable: {type(_prop_err).__name__}: {_prop_err}")

        # -----------------------------------------------------------------
        # Check 9: Prompt-clause coverage (corpus-derived, vocabulary-free).
        # "Sandbox exit code 0" proved nothing about whether the program does
        # what was asked; a path with no FFT or with an unrequested file sink
        # used to lint clean.
        # -----------------------------------------------------------------
        try:
            try:
                from .token_evidence import audit as _cov_audit
            except (ImportError, ValueError):
                from token_evidence import audit as _cov_audit
            _cells_for_cov = [b[0] for b in pipeline_bindings]
            _cov = _cov_audit(_cells_for_cov, prompt, orchestrator=orchestrator)
            for _iss in _cov.get("issues", []):
                if _iss["kind"] == "clause_unserved":
                    _add_violation(
                        "clause_unserved",
                        f"Prompt clause {_iss['clause_idx'] + 1} ({_iss['clause']!r}) is not served by any cell in the "
                        f"pipeline (best on-path cell {_iss['best_on_path']} explains {_iss['path_recall']} of it; the lattice can explain {_iss['lattice_best_recall']}).",
                        details=_iss)
                elif _iss["kind"] == "unrequested_cell":
                    _add_violation(
                        "unrequested_cell",
                        f"Cell '{_iss['cell']}' is not requested by the prompt: every clause it touches is better "
                        f"explained by another cell ({_iss['winners']}) and it explains no remaining clause token.",
                        cell_id=_iss["cell"], details=_iss)
        except Exception as _cov_err:  # an audit crash must never silently pass or kill lint
            warnings.append(f"coverage audit unavailable: {type(_cov_err).__name__}: {_cov_err}")

        is_valid = len(violations) == 0
        return PreflightLintResult(is_valid=is_valid, violations=violations, warnings=warnings, structured_violations=structured_violations)

    @staticmethod
    def audit_lattice(orchestrator: Any) -> Optional[Any]:
        """Runs static reachability, dead-end, and cross-tree auditing on the loaded lattice."""
        try:
            from lattice_auditor import LatticeAuditor
            return LatticeAuditor(orchestrator).audit()
        except Exception:
            return None
