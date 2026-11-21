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
import builtins
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass, field

try:
    from .language_backend import get_active_language_backend
    from .lattice import UNRESOLVED_PORT, TypeRegistry, AlgebraicSignature, PortSignature
    from .schema import DESTRUCTIVE_OR_CLEAN_EFFECT_KINDS, get_effect_kind, is_unresolved_value
    from .tokenizer import CellTokenizer
    from .unification import (
        ExecutionContext, unify, _shape_compatible, _is_sink_cell,
        resolve_typed_binding_record, TypedBindingRecord,
        _get_p_field, _enrich_port_signature_qualifiers, _collect_propagated_qualifiers,
    )
except (ImportError, ValueError):
    from language_backend import get_active_language_backend
    from lattice import UNRESOLVED_PORT, TypeRegistry, AlgebraicSignature, PortSignature
    from schema import DESTRUCTIVE_OR_CLEAN_EFFECT_KINDS, get_effect_kind, is_unresolved_value
    from tokenizer import CellTokenizer
    from unification import (
        ExecutionContext, unify, _shape_compatible, _is_sink_cell,
        resolve_typed_binding_record, TypedBindingRecord,
        _get_p_field, _enrich_port_signature_qualifiers, _collect_propagated_qualifiers,
    )


def _extract_expr_root_var(expr: Any) -> Optional[str]:
    """Extracts the root identifier from a Python variable, attribute, or subscript expression via AST."""
    if not isinstance(expr, str):
        return None
    s = expr.strip()
    if not s or is_unresolved_value(s):
        return None
    if s.startswith("<") and s.endswith(">"):
        return None
    if s.isidentifier():
        return s
    try:
        node = ast.parse(s, mode="eval").body
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        if isinstance(node, ast.Name):
            return node.id
    except SyntaxError:
        pass
    return None


def _contains_word_token(haystack: str, needle: str) -> bool:
    """Return True if *needle* appears in *haystack* delimited by non-identifier chars."""
    if not haystack or not needle:
        return False
    nlen = len(needle)
    hlen = len(haystack)
    start = 0
    while start <= hlen - nlen:
        idx = haystack.find(needle, start)
        if idx == -1:
            return False
        before_ok = idx == 0 or not (haystack[idx - 1].isalnum() or haystack[idx - 1] == "_")
        after_idx = idx + nlen
        after_ok = after_idx == hlen or not (haystack[after_idx].isalnum() or haystack[after_idx] == "_")
        if before_ok and after_ok:
            return True
        start = idx + 1
    return False


@dataclass
class VariableSchema:
    """Tracks active and invalidated composite component schemas per variable (R5-3)."""
    removed_components: Set[str] = field(default_factory=set)
    added_components: Set[str] = field(default_factory=set)
    removed_by: Dict[str, str] = field(default_factory=dict)



import collections.abc


class _DynamicDataBearingRoles(collections.abc.Set):
    """Dynamic data-bearing roles proxy backed by TypeRegistry."""
    def __contains__(self, item):
        return str(item).lower() in TypeRegistry.get_instance().get_data_bearing_roles()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_data_bearing_roles())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_data_bearing_roles())


DATA_BEARING_ROLES = _DynamicDataBearingRoles()


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
    refusal_kind: Optional[str] = None

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            msg = "Pre-flight lint validation failed:\n  - " + "\n  - ".join(self.violations)
            raise PreflightLintError(msg, violations=self.violations, structured_violations=self.structured_violations)


PreflightResult = PreflightLintResult


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
            reg_m = TypeRegistry.get_instance()
            model_roles = (
                reg_m.get_roles_for_category("model")
                | reg_m.get_roles_for_category("estimator")
                | reg_m.get_roles_for_category("model_input")
                | reg_m.get_roles_for_category("model_sink")
            )
        except Exception:
            model_roles = set()
        return bool(role and role in model_roles)

    @classmethod
    def _verify_carrier_extent_epochs(
        cls, pipeline_bindings: List[Tuple[Any, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Check 10: Per-carrier extent epoch tracking.
        Tracks causal root carrier + monotone extent-mutation epoch across steps,
        flagging mismatched-epoch writes in setters and mismatched-epoch co-indexed
        multi-input consumers (e.g. X pre-dropna + y post-dropna).
        """
        _extent_effect_kinds = {
            "extent_mutation", "extent_altering", "cardinality_change",
            "row_filter", "row_set_altering", "resampling", "dedup",
        }
        _extent_effect_names = {
            "extent_mutating", "deduplicates", "filters_rows", "drops_rows", "resamples",
        }
        _reconcile_roles = {"join", "merge", "concat", "reconcile", "alignment", "align"}
        _preserve_tokens = {"", "none", "preserve", "preserving", "id", "identity"}
        reg = cls._registry()
        if callable(getattr(reg, "get_roles_for_category", None)):
            receiver_roles = (
                set(reg.get_roles_for_category("receiver"))
                | set(reg.get_roles_for_category("data"))
            )
        else:
            receiver_roles = {"receiver", "data_input", "df"}
        key_roles_10 = reg.get_roles_for_category("key")
        val_roles_10 = reg.get_roles_for_category("assigned_value")
        scalar_types_10 = {"int", "float", "bool", "str", "number", "numeric", "scalar"}

        def _root_id(val: Any) -> Optional[str]:
            if not isinstance(val, str) or _is_unbound(val):
                return None
            return _extract_expr_root_var(val)

        def _is_scalar_port_sig(p_sig: Any) -> bool:
            if p_sig is None:
                return False
            sk = str(getattr(p_sig, "structure_kind", "") or "").strip().lower()
            if sk == "scalar":
                return True
            if sk in ("composite", "collection", "handle"):
                return False
            rk = getattr(p_sig, "canonical_rank", None)
            if rk is None:
                rk = getattr(p_sig, "rank", None)
            if rk is not None:
                return rk == 0
            tn = str(getattr(getattr(p_sig, "signature", p_sig), "type_name", "") or "").strip().lower()
            return tn in scalar_types_10 or reg.is_subtype(tn, "scalar")

        def _cell_alters_extent(cell_obj: Any) -> bool:
            for eff in (getattr(cell_obj, "effects", None) or ()):
                if isinstance(eff, dict):
                    ek = str(eff.get("effect_kind") or "").strip().lower()
                    en = str(eff.get("name") or "").strip().lower()
                elif isinstance(eff, str):
                    ek = eff.strip().lower()
                    en = ek
                else:
                    ek = str(getattr(eff, "effect_kind", "") or "").strip().lower()
                    en = str(getattr(eff, "name", "") or "").strip().lower()
                if ek in _extent_effect_kinds or en in _extent_effect_names:
                    return True
            for attr_n in ("row_set_effects", "extent_effect"):
                raw_ev = getattr(cell_obj, attr_n, None)
                if raw_ev is not None:
                    if isinstance(raw_ev, str) and raw_ev.strip().lower() not in _preserve_tokens:
                        return True
                    if isinstance(raw_ev, (list, tuple, set, dict)) and bool(raw_ev):
                        return True
            return False

        var_root: Dict[str, str] = {}
        var_epoch: Dict[str, int] = {}
        issues: List[Dict[str, Any]] = []

        for cell, bindings in pipeline_bindings:
            if not isinstance(bindings, dict):
                continue
            inputs = getattr(cell, "inputs", {}) or {}
            outputs = getattr(cell, "outputs", {}) or {}
            out_keys = set(outputs.keys()) | {"output_var"}
            c_id = getattr(cell, "cell_id", str(cell))
            c_role = str(getattr(cell, "node_role", "") or "").strip().lower()
            is_reconciling = c_role in _reconcile_roles
            optic_r = str(getattr(cell, "optic_role", "") or "").strip().lower()

            # Tracked non-scalar input variables for this step
            tracked_inputs: List[Tuple[str, str]] = []
            for p_name, val in bindings.items():
                if p_name in out_keys:
                    continue
                rv = _root_id(val)
                if rv and rv in var_epoch:
                    tracked_inputs.append((p_name, rv))

            # Pattern A: Setter writing value into container across mismatched epochs
            flagged_here = False
            key_p = next(
                (n for n, ps in inputs.items() if str(getattr(ps, "binds", "") or cls._port_role(ps)).lower() in key_roles_10 and n in bindings),
                None,
            )
            val_p = next(
                (n for n, ps in inputs.items() if str(getattr(ps, "binds", "") or cls._port_role(ps)).lower() in val_roles_10 and n in bindings),
                None,
            )
            if (optic_r == "setter" or (key_p and val_p)) and not is_reconciling:
                cont_p = next(
                    (
                        n for n, ps in inputs.items()
                        if n not in (key_p, val_p)
                        and (
                            str(getattr(ps, "binds", "") or "").lower() in receiver_roles
                            or n == getattr(cell, "container_port", None)
                            or n == getattr(getattr(cell, "primary_input", None), "name", None)
                        )
                        and n in bindings
                    ),
                    None,
                )
                if val_p is None and cont_p is not None:
                    val_p = next((n for n in bindings if n not in out_keys and n not in (key_p, cont_p) and _root_id(bindings[n]) in var_epoch), None)
                if cont_p and val_p:
                    c_var = _root_id(bindings.get(cont_p))
                    v_var = _root_id(bindings.get(val_p))
                    if c_var in var_epoch and v_var in var_epoch and var_root[c_var] == var_root[v_var]:
                        if var_epoch[c_var] != var_epoch[v_var]:
                            issues.append({
                                "cell": c_id,
                                "value_vars": [v_var],
                                "value_epoch": var_epoch[v_var],
                                "table_vars": [c_var],
                                "table_epoch": var_epoch[c_var],
                                "root_carrier": var_root[c_var],
                            })
                            flagged_here = True

            # Pattern B: Multi-input co-indexed consumer across mismatched epochs on same root carrier
            if not flagged_here and not is_reconciling and len(tracked_inputs) >= 2:
                by_root: Dict[str, List[str]] = {}
                for p_name, rv in tracked_inputs:
                    p_sig = inputs.get(p_name)
                    if _is_scalar_port_sig(p_sig):
                        continue
                    r_key = var_root[rv]
                    if rv not in by_root.setdefault(r_key, []):
                        by_root[r_key].append(rv)
                for r_key, r_vars in by_root.items():
                    if len(r_vars) < 2:
                        continue
                    epochs_seen = {var_epoch[v] for v in r_vars}
                    if len(epochs_seen) > 1:
                        max_ep = max(epochs_seen)
                        min_ep = min(epochs_seen)
                        hi_vars = [v for v in r_vars if var_epoch[v] == max_ep]
                        lo_vars = [v for v in r_vars if var_epoch[v] == min_ep]
                        issues.append({
                            "cell": c_id,
                            "value_vars": hi_vars,
                            "value_epoch": max_ep,
                            "table_vars": lo_vars,
                            "table_epoch": min_ep,
                            "root_carrier": r_key,
                        })
                        break

            # Propagate (root, epoch) to outputs
            alters = _cell_alters_extent(cell)
            base_in_v: Optional[str] = None
            prim_in = getattr(cell, "primary_input", None)
            if prim_in and prim_in.name in bindings:
                cand_rv = _root_id(bindings[prim_in.name])
                if cand_rv and cand_rv in var_epoch:
                    base_in_v = cand_rv
            if base_in_v is None:
                for p_name, rv in tracked_inputs:
                    if not _is_scalar_port_sig(inputs.get(p_name)):
                        base_in_v = rv
                        break
            if base_in_v is None and tracked_inputs:
                base_in_v = tracked_inputs[0][1]

            out_vars: List[str] = []
            for ok_name in out_keys:
                ov = _root_id(bindings.get(ok_name))
                if ov and ov not in out_vars:
                    out_vars.append(ov)

            if base_in_v is not None:
                b_root = var_root[base_in_v]
                new_ep = var_epoch[base_in_v] + (1 if alters else 0)
                if alters and getattr(cell, "mutation_type", "") == "in_place":
                    var_epoch[base_in_v] = new_ep
                for ov in out_vars:
                    var_root[ov] = b_root
                    var_epoch[ov] = new_ep
            else:
                for ov in out_vars:
                    var_root[ov] = ov
                    var_epoch[ov] = 1 if alters else 0

        return issues

    @classmethod
    def _verify_causal_key_def_use(
        cls, pipeline_bindings: List[Tuple[Any, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Universal Key-Space Causal Def-Use:
        Tracks defined_at: Dict[Tuple[str, str], int] (per carrier root + lowercase key)
        and step_projected_keys (all keys read via projection/subscript at step idx,
        including RHS reads inside setter cells).
        A key projected from a carrier at step idx must not be defined on that carrier
        at a later step defined_at[(root, k)] > idx.
        """
        defined_at: Dict[Tuple[str, str], int] = {}
        var_root: Dict[str, str] = {}
        var_key_lineage: Dict[str, Set[str]] = {}
        step_projected_keys: List[Tuple[int, Any, str, str, Set[str]]] = []
        key_bind_tags = cls._registry().get_roles_for_category("key")
        val_bind_tags = cls._registry().get_roles_for_category("assigned_value")
        recv_bind_tags = cls._registry().get_roles_for_category("receiver") | cls._registry().get_roles_for_category("data")

        def _extract_subscript_keys(expr: Any) -> Set[str]:
            found: Set[str] = set()
            if not isinstance(expr, str) or "[" not in expr:
                return found
            if get_active_language_backend(cls._registry()).is_python_ast:
                try:
                    tree = ast.parse(expr, mode="eval")
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Subscript):
                            for sub in ast.walk(node.slice):
                                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                                    s_val = sub.value.strip()
                                    if s_val:
                                        found.add(s_val)
                except Exception:
                    pass
                return found
            i = 0
            n_expr = len(expr)
            while i < n_expr:
                if expr[i] == "[":
                    depth = 1
                    j = i + 1
                    in_q: Optional[str] = None
                    q_start = -1
                    while j < n_expr and depth > 0:
                        ch = expr[j]
                        if in_q is not None:
                            if ch == "\\" and j + 1 < n_expr:
                                j += 2
                                continue
                            if ch == in_q:
                                s_val = expr[q_start:j].strip()
                                if s_val:
                                    found.add(s_val)
                                in_q = None
                        else:
                            if ch in ("'", '"'):
                                in_q = ch
                                q_start = j + 1
                            elif ch == "[":
                                depth += 1
                            elif ch == "]":
                                depth -= 1
                        j += 1
                    i = j
                else:
                    i += 1
            return found

        for idx, (cell, bindings) in enumerate(pipeline_bindings):
            if not isinstance(bindings, dict):
                continue
            inputs = getattr(cell, "inputs", {}) or {}
            outputs = getattr(cell, "outputs", {}) or {}
            out_keys = set(outputs.keys()) | {"output_var"}
            typed_map = getattr(cell, "typed_bindings", {}) or {}

            proj_roles = (
                cls._registry().get_roles_for_category("projection")
                | set(cls._registry().get_projection_tokens() or ())
                | key_bind_tags
            )

            # Determine input carrier root for this step
            step_carrier_var: Optional[str] = None
            cont_port_name = getattr(cell, "container_port", None)
            if cont_port_name and isinstance(bindings.get(cont_port_name), str) and not _is_unbound(bindings.get(cont_port_name)):
                step_carrier_var = _extract_expr_root_var(bindings[cont_port_name])
            if not step_carrier_var and getattr(cell, "primary_input", None):
                pi_n = cell.primary_input.name
                if isinstance(bindings.get(pi_n), str) and not _is_unbound(bindings.get(pi_n)):
                    step_carrier_var = _extract_expr_root_var(bindings[pi_n])
            if not step_carrier_var:
                for pn, ps in inputs.items():
                    pv = bindings.get(pn)
                    if isinstance(pv, str) and not _is_unbound(pv):
                        rv = _extract_expr_root_var(pv)
                        if rv and (rv in var_root or str(getattr(ps, "binds", "") or "").lower() in recv_bind_tags):
                            step_carrier_var = rv
                            break
            step_carrier_root = var_root.get(step_carrier_var, step_carrier_var) if step_carrier_var else "_global_"

            def _keys_and_root_of_port(p_name: str, expr: Any) -> Tuple[str, Set[str]]:
                p_obj = inputs.get(p_name)
                rec = typed_map.get(p_name) or resolve_typed_binding_record(cell, p_name, expr, p_obj)
                found: Set[str] = set()
                read_var: Optional[str] = None
                if rec and rec.kind == "projection" and rec.projection:
                    k = rec.projection.get("key")
                    if isinstance(k, (list, tuple, set)):
                        found.update(str(x).strip("'\"") for x in k)
                    elif isinstance(k, str):
                        found.add(k.strip("'\""))
                    read_var = rec.root_var
                if not found:
                    found.update(_extract_subscript_keys(expr))
                if not found and p_obj is not None and isinstance(expr, str):
                    p_role = str(getattr(p_obj, "port_role", "") or "").strip().lower()
                    p_state = str(getattr(p_obj, "state", "") or "").strip().lower()
                    p_binds = str(getattr(p_obj, "binds", "") or "").strip().lower()
                    if p_role in proj_roles or p_state in proj_roles or p_binds in proj_roles:
                        if get_active_language_backend(cls._registry()).is_python_ast:
                            try:
                                tree = ast.parse(expr, mode="eval")
                                for sub in ast.walk(tree):
                                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                                        s_val = sub.value.strip()
                                        if s_val:
                                            found.add(s_val)
                            except Exception:
                                pass
                if not found and isinstance(expr, str) and expr.strip() in var_key_lineage:
                    found.update(var_key_lineage[expr.strip()])
                    read_var = expr.strip()
                if not read_var and isinstance(expr, str) and not _is_unbound(expr):
                    cand_rv = _extract_expr_root_var(expr)
                    if cand_rv and cand_rv in var_root:
                        read_var = cand_rv
                r_root = var_root.get(read_var, read_var) if read_var else step_carrier_root
                return r_root, found

            key_port_decl = getattr(cell, "key_port", None)
            key_port = next(
                (
                    n for n, p in inputs.items()
                    if n in bindings
                    and (
                        getattr(p, "binds", None) in key_bind_tags
                        or (key_port_decl and n == key_port_decl)
                        or str(getattr(p, "port_role", "") or "").strip().lower() in key_bind_tags
                    )
                ),
                None,
            )
            val_port = next((n for n, p in inputs.items() if getattr(p, "binds", None) in val_bind_tags and n in bindings), None)
            is_setter = str(getattr(cell, "optic_role", "") or "").strip().lower() == "setter"
            if is_setter and val_port is None and key_port:
                cont_n = getattr(cell, "container_port", None) or (cell.primary_input.name if getattr(cell, "primary_input", None) else None)
                val_port = next((n for n in bindings if n not in out_keys and n not in (key_port, cont_n)), None)

            if key_port and (val_port or is_setter):
                k_rec = typed_map.get(key_port)
                written_key = str(k_rec.value_expr if (k_rec and k_rec.value_expr) else bindings[key_port]).strip().strip("'\"")
                val_lineage: Set[str] = set()
                if val_port and val_port in bindings:
                    _, val_lineage = _keys_and_root_of_port(val_port, bindings[val_port])
                val_lin_low = {str(x).strip().lower() for x in val_lineage}
                wk_low = written_key.lower()
                if wk_low and wk_low not in val_lin_low:
                    defined_at.setdefault((step_carrier_root, wk_low), idx)

            out_v = bindings.get("output_var")
            step_read_lineage: Set[str] = set()
            for p_name, expr in bindings.items():
                if p_name in out_keys or p_name == key_port:
                    continue
                r_root, p_keys = _keys_and_root_of_port(p_name, expr)
                if p_keys:
                    step_read_lineage.update(p_keys)
                    step_projected_keys.append((idx, cell, p_name, r_root, p_keys))

            bound_out_names = [
                str(bindings[ok]).strip()
                for ok in out_keys
                if isinstance(bindings.get(ok), str) and str(bindings.get(ok)).strip()
            ]
            for ov_str in bound_out_names:
                ov_clean = _extract_expr_root_var(ov_str) or ov_str
                if step_carrier_var and step_carrier_var in var_root:
                    var_root.setdefault(ov_clean, var_root[step_carrier_var])
                elif step_carrier_var:
                    var_root.setdefault(step_carrier_var, step_carrier_var)
                    var_root.setdefault(ov_clean, step_carrier_var)
                else:
                    var_root.setdefault(ov_clean, ov_clean)
                if step_read_lineage:
                    var_key_lineage.setdefault(ov_clean, set()).update(step_read_lineage)

        issues: List[Dict[str, Any]] = []
        for idx, cell, p_name, r_root, keys in step_projected_keys:
            for k in keys:
                k_low = str(k).strip().lower()
                def_idx = defined_at.get((r_root, k_low))
                if def_idx is not None and def_idx > idx:
                    issues.append({
                        "cell": getattr(cell, "cell_id", str(cell)),
                        "port": p_name,
                        "key": k,
                        "column": k,
                        "root_carrier": r_root,
                        "read_at": idx,
                        "defined_at": def_idx,
                        "defined_by": getattr(pipeline_bindings[def_idx][0], "cell_id", ""),
                    })
        return issues

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
        clause_semantic_scores: Optional[List[Dict[str, float]]] = None,
    ) -> PreflightLintResult:
        violations: List[str] = []
        warnings: List[str] = []
        structured_violations: List[PreflightViolation] = []
        waived = set(waived_literals or set())

        bound_pipeline_vars: Set[str] = set()
        if code_str:
            for tgt in get_active_language_backend(cls._registry()).extract_assignment_targets(code_str):
                if isinstance(tgt, str) and tgt:
                    bound_pipeline_vars.add(tgt)
        for cell_item, b_dict in pipeline_bindings:
            if isinstance(b_dict, dict):
                out_v = b_dict.get("output_var")
                if out_v and not _is_unbound(out_v):
                    bound_pipeline_vars.add(str(out_v))
                for o_name in (getattr(cell_item, "outputs", {}) or {}):
                    o_val = b_dict.get(o_name)
                    if o_val and not _is_unbound(o_val):
                        bound_pipeline_vars.add(str(o_val))

        def _is_var(val: Any) -> bool:
            if not isinstance(val, str) or not val:
                return False
            if _is_unbound(val):
                return False
            v = val.strip()
            if v in bound_pipeline_vars:
                return True
            root = _extract_expr_root_var(v)
            return bool(root and root in bound_pipeline_vars)

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

        # Extract universal literals from prompt (unless pipeline is explicit formal Program_0 syntax,
        # where literals are explicitly declared per-port and must not be regex-scraped from call wrappers).
        extracted_literals: List[Tuple[int, str, str]] = []
        is_formal_pipeline = bool(pipeline_bindings) and all(
            bool(getattr(_c, "formal_program_step", False)) for _c, _ in pipeline_bindings
        )
        if is_formal_pipeline:
            for _cell_idx, (_cell_item, _b_dict) in enumerate(pipeline_bindings):
                exp_b = getattr(_cell_item, "explicit_port_bindings", None) or {}
                for _pn, _lv in exp_b.items():
                    if isinstance(_lv, str) and _lv.strip():
                        extracted_literals.append((_cell_idx, "quoted_str", _lv.strip()))
                    elif isinstance(_lv, (int, float)) and not isinstance(_lv, bool):
                        extracted_literals.append((_cell_idx, "numeric", str(_lv)))
        else:
            try:
                extracted_literals = ExecutionContext._extract_universal_literals(prompt or "")
            except Exception:
                pass

        # Cache parsed Python AST once across Checks 1, 2b, and 4
        be_lint = get_active_language_backend(cls._registry())
        parsed_code_tree: Optional[ast.AST] = None
        code_syntax_error_reported = False
        if code_str and be_lint.is_python_ast:
            try:
                parsed_code_tree = ast.parse(code_str)
            except SyntaxError as syn_err:
                _add_violation("syntax_error", f"Synthesized code has syntax error: {syn_err}", details={"error": str(syn_err)})
                code_syntax_error_reported = True

        # -----------------------------------------------------------------
        # Check 1: Universal Literal Consumption
        # -----------------------------------------------------------------
        if extracted_literals:
            bound_values_str: Set[str] = set()
            bound_num_values: List[float] = []
            for _cell, bindings in pipeline_bindings:
                if not isinstance(bindings, dict):
                    continue
                for _p_name, val in bindings.items():
                    if val is None or _is_unbound(val):
                        continue
                    str_val = str(val).strip("'\"")
                    if str_val and str_val not in bound_pipeline_vars:
                        bound_values_str.add(str_val)
                    if not isinstance(val, bool):
                        try:
                            bound_num_values.append(float(str_val))
                        except (TypeError, ValueError):
                            if be_lint.is_python_ast and isinstance(val, str):
                                try:
                                    b_ast = ast.parse(val.strip(), mode="eval").body
                                    neg_b = {
                                        id(n.operand) for n in ast.walk(b_ast)
                                        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub)
                                    }
                                    for bn in ast.walk(b_ast):
                                        if isinstance(bn, ast.UnaryOp) and isinstance(bn.op, ast.USub) and isinstance(bn.operand, ast.Constant) and isinstance(bn.operand.value, (int, float)) and not isinstance(bn.operand.value, bool):
                                            bound_num_values.append(-float(bn.operand.value))
                                        elif isinstance(bn, ast.Constant) and id(bn) not in neg_b and isinstance(bn.value, (int, float)) and not isinstance(bn.value, bool):
                                            bound_num_values.append(float(bn.value))
                                except Exception:
                                    pass

            ast_str_constants: Set[str] = set()
            ast_joined_segments: List[str] = []
            ast_num_values: List[float] = []
            ast_names: Set[str] = set()
            ast_kw_args: Set[str] = set()
            non_py_syntax_ok = True
            if parsed_code_tree is not None:
                negated_operand_ids = {
                    id(n.operand)
                    for n in ast.walk(parsed_code_tree)
                    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub)
                }
                for node in ast.walk(parsed_code_tree):
                    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
                        if isinstance(node.operand.value, (int, float)) and not isinstance(node.operand.value, bool):
                            ast_num_values.append(-float(node.operand.value))
                    elif isinstance(node, ast.JoinedStr):
                        js_parts: List[str] = []
                        for val_node in node.values:
                            if isinstance(val_node, ast.Constant) and isinstance(val_node.value, str):
                                seg_low = val_node.value.lower()
                                if seg_low:
                                    ast_joined_segments.append(seg_low)
                                    js_parts.append(seg_low)
                        if js_parts:
                            ast_str_constants.add("".join(js_parts))
                    elif isinstance(node, ast.Constant):
                        if isinstance(node.value, str):
                            ast_str_constants.add(node.value.lower())
                        elif isinstance(node.value, (int, float)) and not isinstance(node.value, bool) and id(node) not in negated_operand_ids:
                            ast_num_values.append(float(node.value))
                    elif isinstance(node, ast.Name):
                        ast_names.add(node.id.lower())
                    elif isinstance(node, ast.keyword) and node.arg:
                        ast_kw_args.add(node.arg.lower())
            elif code_str and not be_lint.is_python_ast:
                non_py_syntax_ok, non_py_err = be_lint.validate_syntax(code_str)
                if not non_py_syntax_ok:
                    _add_violation("syntax_error", f"Synthesized code failed static syntax validation: {non_py_err}")
                    code_syntax_error_reported = True

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
                clean_lower = clean_lit.lower()

                is_consumed = False
                if kind == "numeric":
                    try:
                        lit_f = float(clean_lit)
                        is_consumed = any(bn == lit_f for bn in bound_num_values)
                        if not is_consumed and parsed_code_tree is not None:
                            is_consumed = any(an == lit_f for an in ast_num_values)
                    except (TypeError, ValueError):
                        is_consumed = clean_lit in bound_values_str
                else:
                    is_consumed = any(
                        clean_lit == b or (kind == "file_asset" and b.lower().endswith("/" + clean_lower))
                        for b in bound_values_str
                    )

                if not is_consumed and code_str and kind != "numeric":
                    if parsed_code_tree is not None:
                        if clean_lower in ast_str_constants or clean_lower in ast_names or clean_lower in ast_kw_args:
                            is_consumed = True
                        elif kind == "file_asset":
                            is_consumed = any(
                                sc == clean_lower or sc.endswith("/" + clean_lower) or sc.endswith("\\" + clean_lower)
                                for sc in ast_str_constants
                            )
                        elif kind == "quoted_str":
                            is_consumed = any(_contains_word_token(seg, clean_lower) for seg in ast_joined_segments)
                    elif not be_lint.is_python_ast and non_py_syntax_ok:
                        is_consumed = (
                            _contains_word_token(code_str.lower(), clean_lower)
                            or clean_lower in code_str.lower()
                        )

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
        # Check 1b: Active Component Set Tracking (Schema Removal, R5-3)
        # Tracks composite carrier schemas via declared effects (`schema_removal`
        # / `removes_columns`) and optic setters with zero domain hardcodes.
        # -----------------------------------------------------------------
        var_schemas: Dict[str, VariableSchema] = {}
        reg_1b = cls._registry()
        key_binds_1b = reg_1b.get_roles_for_category("key")
        receiver_binds_1b = reg_1b.get_roles_for_category("receiver") | reg_1b.get_roles_for_category("data")

        def _parse_key_collection(raw_val: Any) -> List[str]:
            out_keys: List[str] = []
            if isinstance(raw_val, (list, tuple, set)):
                for item in raw_val:
                    s = str(item).strip().strip("'\"")
                    if s:
                        out_keys.append(s)
                return out_keys
            if isinstance(raw_val, str):
                s = raw_val.strip()
                if not s or _is_unbound(s):
                    return out_keys
                if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                    s = s[1:-1].strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        parsed_k = ast.literal_eval(s)
                        if isinstance(parsed_k, (list, tuple, set)):
                            for item in parsed_k:
                                it = str(item).strip().strip("'\"")
                                if it:
                                    out_keys.append(it)
                            return out_keys
                    except Exception:
                        pass
                    for item in s[1:-1].split(","):
                        it = item.strip().strip("'\"")
                        if it:
                            out_keys.append(it)
                else:
                    clean_k = s.strip("'\"")
                    if clean_k:
                        out_keys.append(clean_k)
            return out_keys

        for cell, bindings in pipeline_bindings:
            if not isinstance(bindings, dict):
                continue
            c_inputs = getattr(cell, "inputs", {}) or {}
            c_typed = getattr(cell, "typed_bindings", {}) or {}

            carrier_in_var: Optional[str] = None
            prim_in = getattr(cell, "primary_input", None)
            if prim_in and prim_in.name in bindings:
                cand_v = bindings[prim_in.name]
                if isinstance(cand_v, str):
                    root_cand = _extract_expr_root_var(cand_v)
                    if root_cand and (root_cand in bound_pipeline_vars or root_cand in var_schemas):
                        carrier_in_var = root_cand
            if not carrier_in_var:
                for p_n, p_sig in c_inputs.items():
                    cand_v = bindings.get(p_n)
                    if not isinstance(cand_v, str):
                        continue
                    root_cand = _extract_expr_root_var(cand_v)
                    if not root_cand or (root_cand not in bound_pipeline_vars and root_cand not in var_schemas):
                        continue
                    p_binds = str(getattr(p_sig, "binds", "") or getattr(p_sig, "port_role", "") or "").lower()
                    p_tn = str(getattr(getattr(p_sig, "signature", p_sig), "type_name", "") or "").lower()
                    if p_binds in receiver_binds_1b or reg_1b.is_abstract_carrier(p_tn) or root_cand in var_schemas:
                        carrier_in_var = root_cand
                        break

            # 1. Verify referenced projected/removed keys on consumed variables
            for p_name, val in bindings.items():
                if val is None or _is_unbound(val):
                    continue
                if p_name == "output_var" or p_name in (getattr(cell, "outputs", {}) or {}):
                    continue
                rec = c_typed.get(p_name) or resolve_typed_binding_record(
                    cell=cell, port_name=p_name, bound_val=val, port_sig=c_inputs.get(p_name)
                )
                cols_referenced: List[Tuple[str, Optional[str]]] = []
                if rec and rec.kind == "projection" and rec.projection:
                    proj_key = rec.projection.get("key")
                    target_v = rec.root_var or carrier_in_var
                    if isinstance(proj_key, str):
                        cols_referenced.append((proj_key, target_v))
                    elif isinstance(proj_key, (list, tuple, set)):
                        for k in proj_key:
                            cols_referenced.append((str(k), target_v))
                elif isinstance(val, str):
                    p_sig = c_inputs.get(p_name)
                    p_binds = str(getattr(p_sig, "binds", "") or "").lower() if p_sig else ""
                    p_role = cls._port_role(p_sig) if p_sig else ""
                    if p_binds in key_binds_1b or p_role in key_binds_1b:
                        target_v = carrier_in_var
                        for ck in _parse_key_collection(val):
                            cols_referenced.append((ck, target_v))

                for col, target_v in cols_referenced:
                    if not target_v or target_v not in var_schemas:
                        continue
                    sch = var_schemas[target_v]
                    if col in sch.removed_components:
                        remover = sch.removed_by.get(col, "unknown")
                        msg = (
                            f"Cell '{cell.cell_id}' port '{p_name}' references column '{col}' "
                            f"which was provably removed from '{target_v}' by '{remover}'."
                        )
                        _add_violation(
                            "column_absent",
                            msg,
                            cell_id=cell.cell_id,
                            port_name=p_name,
                            variable_name=target_v,
                            details={"column": col, "variable": target_v, "removed_by": remover},
                        )

            # 2. Propagate schema to cell's output variable
            out_var: Optional[str] = bindings.get("output_var")
            if not out_var and getattr(cell, "primary_output", None):
                out_var = bindings.get(cell.primary_output.name)
            if not out_var and carrier_in_var and getattr(cell, "mutation_type", "") == "in_place":
                out_var = carrier_in_var
            if isinstance(out_var, str) and out_var.strip():
                base_sch = var_schemas.get(carrier_in_var) if carrier_in_var else None
                new_sch = VariableSchema(
                    removed_components=set(base_sch.removed_components) if base_sch else set(),
                    added_components=set(base_sch.added_components) if base_sch else set(),
                    removed_by=dict(base_sch.removed_by) if base_sch else {},
                )
                if str(getattr(cell, "optic_role", "") or "").strip().lower() == "setter":
                    kp = next((n for n, ps in c_inputs.items() if str(getattr(ps, "binds", "") or cls._port_role(ps)).lower() in key_binds_1b and n in bindings), None)
                    if kp:
                        for ak in _parse_key_collection(bindings[kp]):
                            new_sch.added_components.add(ak)
                            new_sch.removed_components.discard(ak)

                for eff in (getattr(cell, "effects", []) or []):
                    eff_name = eff.get("name") if isinstance(eff, dict) else getattr(eff, "name", str(eff))
                    eff_kind = eff.get("effect_kind") if isinstance(eff, dict) else getattr(eff, "effect_kind", None)
                    if eff_kind == "schema_removal" or eff_name == "removes_columns":
                        from_port = eff.get("from_port") if isinstance(eff, dict) else getattr(eff, "from_port", None)
                        cols_to_remove: List[str] = []
                        if from_port and from_port in bindings:
                            cols_to_remove.extend(_parse_key_collection(bindings[from_port]))
                        else:
                            for p_n, p_sig in c_inputs.items():
                                pb = str(getattr(p_sig, "binds", "") or cls._port_role(p_sig)).lower()
                                if pb in key_binds_1b and p_n in bindings:
                                    cols_to_remove.extend(_parse_key_collection(bindings[p_n]))
                            c_lits = getattr(cell, "clause_literals", None)
                            if c_lits:
                                cols_to_remove.extend(_parse_key_collection(c_lits))
                        for rem_c in cols_to_remove:
                            if rem_c:
                                new_sch.removed_components.add(rem_c)
                                new_sch.removed_by[rem_c] = getattr(cell, "cell_id", "unknown")

                var_schemas[out_var.strip()] = new_sch

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
                        r_v = _extract_expr_root_var(inp_val)
                        if r_v:
                            origs.update(var_origins.get(r_v, set()))
                    elif isinstance(inp_val, str) and not _is_unbound(inp_val):
                        clean_inp = inp_val.strip("'\"")
                        if clean_inp:
                            origs.add(clean_inp)
                var_origins[out_var] = origs

        # -----------------------------------------------------------------
        # Check 2b: declared dependency names, from the cells' own dependency
        # declarations (plus builtins & explicit imports in code_str).
        # Never whitelist arbitrary stdlib/loaded modules via sys.modules or find_spec.
        # -----------------------------------------------------------------
        declared_dep_names: Set[str] = set(dir(builtins))
        be_pf = be_lint
        for cell, _ in pipeline_bindings:
            for dep in (getattr(cell, "dependencies", None) or []) + (getattr(cell, "imports", None) or []):
                d = str(dep).strip()
                if d:
                    declared_dep_names.update(be_pf.extract_dependency_namespaces(d))
        if parsed_code_tree is not None:
            for imp_node in ast.walk(parsed_code_tree):
                if isinstance(imp_node, ast.Import):
                    for al in imp_node.names:
                        declared_dep_names.add(al.asname or al.name.split(".")[0])
                elif isinstance(imp_node, ast.ImportFrom):
                    for al in imp_node.names:
                        if al.name != "*":
                            declared_dep_names.add(al.asname or al.name)

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

                # Declared module aliases / imported symbols: only names explicitly declared
                # in cell dependencies or imported in synthesized code are permitted.
                if root_var in declared_dep_names:
                    continue

                if root_var not in var_signatures:
                    resolved_sig = None

                if resolved_sig is None:
                    msg = (
                        f"Cell '{cell.cell_id}' port '{p_name}' references undefined pipeline variable '{root_var}' "
                        f"with no producing cell in DAG history."
                    )
                    _add_violation("undefined_variable", msg, cell_id=cell.cell_id, port_name=p_name, variable_name=root_var)
                    continue

                prod_cell, prod_port = var_producers.get(root_var, (None, None))
                prod_id = prod_cell.cell_id if prod_cell else "unknown"
                prod_sig_for_edge = resolved_sig
                if (
                    getattr(record, "kind", None) == "variable"
                    and bound_val == root_var
                    and prod_cell is not None
                    and prod_port
                ):
                    _declared_out = (getattr(prod_cell, "outputs", {}) or {}).get(prod_port)
                    if _declared_out is not None:
                        prod_sig_for_edge = _declared_out

                # 1. Structural Carrier & Shape Compatibility
                if not _shape_compatible(prod_sig_for_edge, in_port_sig, producer_cell=prod_cell, consumer_cell=cell):
                    res_quals_raw = _get_p_field(resolved_sig, "qualifiers", None) or getattr(getattr(resolved_sig, "signature", None), "qualifiers", ()) or ()
                    in_rej_raw = _get_p_field(in_port_sig, "rejected_qualifiers", None) or getattr(getattr(in_port_sig, "signature", None), "rejected_qualifiers", ()) or ()

                    flat_res_quals = set()
                    for q in res_quals_raw:
                        if isinstance(q, (tuple, list, set)):
                            flat_res_quals.update(str(x).lower() for x in q)
                        else:
                            flat_res_quals.add(str(q).lower())

                    flat_in_rej = set()
                    for q in in_rej_raw:
                        if isinstance(q, (tuple, list, set)):
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
                down_out_keys = set((getattr(downstream_cell, "outputs", {}) or {}).keys()) | {"output_var"}
                for p_k, down_val in down_bindings.items():
                    if down_val is None or p_k in down_out_keys:
                        continue
                    # Check typed bindings record first
                    d_rec = getattr(downstream_cell, "typed_bindings", {}).get(p_k)
                    if d_rec and d_rec.root_var and d_rec.root_var in produced_vars:
                        is_consumed = True
                        break
                    down_str = str(down_val).strip()
                    if be_lint.is_python_ast:
                        try:
                            expr_ast = ast.parse(down_str, mode="eval")
                            if any(
                                isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in produced_vars
                                for n in ast.walk(expr_ast)
                            ):
                                is_consumed = True
                                break
                            continue
                        except SyntaxError:
                            pass
                    for pv in produced_vars:
                        if pv == down_str or _contains_word_token(down_str, pv):
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
                c_ntype = str(getattr(cell, "node_type", "") or "").strip().lower()
                for out_name, out_sig in getattr(cell, "outputs", {}).items():
                    s_kind = str(getattr(out_sig, "structure_kind", "") or "").lower()
                    p_role = str(getattr(out_sig, "port_role", "") or "").lower()
                    o_tname = str(getattr(getattr(out_sig, "signature", out_sig), "type_name", "") or "").strip().lower()
                    if c_ntype in ("macro", "combinator") and p_role == "result" and o_tname == "state":
                        continue
                    if s_kind == "handle" or cls._is_object_receiver_role(p_role):
                        has_handle_output = True
                        break

                is_teardown_tail = (
                    getattr(cell, "stage", None) == 2
                    and bool(pipeline_bindings[i + 1 :])
                    and all(_is_sink_cell(dc) for dc, _ in pipeline_bindings[i + 1 :])
                )
                if has_handle_output and i < n_steps - 1 and not is_teardown_tail:
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
                reg_t = cls._registry()
                term_roles = (
                    reg_t.get_terminal_roles_set()
                    | reg_t.get_roles_for_category("terminal_output")
                )
                is_terminal = role in term_roles
                if not is_terminal:
                    try:
                        from planner import _active_vocab
                        is_terminal = _active_vocab().is_terminal(cell)
                    except Exception:
                        pass
                # In graph-theoretic dataflow liveness:
                # A step is live if its outputs are consumed, OR if it represents a terminal deliverable / sink,
                # OR if it was explicitly synthesized as the witness to a declared user specification clause.
                # Serving a clause only excuses an unused result for observations/queries (scalar results such as
                # a mean). A carrier-preserving transform (non-scalar data in, same kind of data out) whose result
                # nothing reads is a silent no-op: the requested operation had no effect on the final output.
                _pi = getattr(cell, "primary_input", None)
                _po = getattr(cell, "primary_output", None)
                _k_in = str(getattr(_pi, "structure_kind", "") or "").lower()
                _k_out = str(getattr(_po, "structure_kind", "") or "").lower()
                _t_in = str(getattr(_pi, "type_name", "") or "").lower()
                _t_out = str(getattr(_po, "type_name", "") or "").lower()
                is_carrier_transform = bool(_k_in and _k_in == _k_out and _t_in == _t_out and _k_out not in ("scalar", "handle"))
                if (
                    is_terminal
                    or getattr(cell, "is_goal", False)
                    or (getattr(cell, "matched_clause_idx", None) is not None and not is_carrier_transform)
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
        if prompt and not is_formal_pipeline:
            path_cells = [c for c, _ in pipeline_bindings]
            # Preserve causal clause tags already established by the planner or Layer-3
            # role-partitioned complement/bridge synthesis; only invoke heuristic re-tagging
            # when no causal tags exist yet, or monotonically interpolate missing bridge tags.
            has_any_tag = any(getattr(c, "matched_clause_idx", None) is not None for c in path_cells)
            if not has_any_tag:
                try:
                    try:
                        from .route_methods.base import RouteMethod
                    except (ImportError, ValueError):
                        from route_methods.base import RouteMethod
                    RouteMethod.tag_cells_with_clause_indices(
                        path_cells, prompt, orchestrator=orchestrator,
                        clause_semantic_scores=clause_semantic_scores,
                    )
                    for idx, c in enumerate(path_cells):
                        if pipeline_bindings[idx][0] is not c:
                            pipeline_bindings[idx] = (c, pipeline_bindings[idx][1])
                except Exception:
                    pass
            elif any(getattr(c, "matched_clause_idx", None) is None for c in path_cells):
                for idx, c in enumerate(path_cells):
                    if getattr(c, "matched_clause_idx", None) is None:
                        prev_cl = next(
                            (getattr(path_cells[k], "matched_clause_idx", None)
                             for k in range(idx - 1, -1, -1)
                             if getattr(path_cells[k], "matched_clause_idx", None) is not None),
                            None,
                        )
                        next_cl = next(
                            (getattr(path_cells[k], "matched_clause_idx", None)
                             for k in range(idx + 1, len(path_cells))
                             if getattr(path_cells[k], "matched_clause_idx", None) is not None),
                            None,
                        )
                        c.matched_clause_idx = prev_cl if prev_cl is not None else next_cl

        # Track which variables each cell produces and consumes
        cell_producers: Dict[str, Tuple[int, Any]] = {}  # var_name -> (step_idx, cell)
        cell_first_producer: Dict[str, Tuple[int, Any]] = {}  # var_name -> (first step_idx, cell)
        for i, (cell, bindings) in enumerate(pipeline_bindings):
            if not isinstance(bindings, dict):
                continue
            for out_name in getattr(cell, "outputs", {}).keys():
                v = bindings.get(out_name)
                if isinstance(v, str) and _is_var(v):
                    cell_producers[v] = (i, cell)
                    if v not in cell_first_producer:
                        cell_first_producer[v] = (i, cell)
            out_v = bindings.get("output_var")
            if isinstance(out_v, str) and _is_var(out_v):
                cell_producers[out_v] = (i, cell)
                if out_v not in cell_first_producer:
                    cell_first_producer[out_v] = (i, cell)

        # Check for DAG causal order violations and clause order inversions across dependent cell pairs
        for j, (cons_cell, cons_bindings) in enumerate(pipeline_bindings):
            if not isinstance(cons_bindings, dict):
                continue
            cons_clause = getattr(cons_cell, "matched_clause_idx", None)
            out_port_keys = set(getattr(cons_cell, "outputs", {}).keys()) | {"output_var"}

            for p_k, down_val in cons_bindings.items():
                if down_val is None or p_k in out_port_keys:
                    continue
                d_rec = getattr(cons_cell, "typed_bindings", {}).get(p_k)
                root_v = d_rec.root_var if d_rec else None
                if not root_v and isinstance(down_val, str) and _is_var(down_val):
                    root_v = down_val

                if root_v and root_v in cell_first_producer:
                    first_prod_idx, first_prod_cell = cell_first_producer[root_v]
                    if first_prod_idx >= j and first_prod_cell is not cons_cell:
                        msg = (
                            f"DAG causal order violation: Cell '{cons_cell.cell_id}' at step {j} "
                            f"consumes variable '{root_v}' before it is produced by "
                            f"Cell '{first_prod_cell.cell_id}' at step {first_prod_idx}."
                        )
                        _add_violation(
                            "dag_causal_order_violation",
                            msg,
                            cell_id=cons_cell.cell_id,
                            variable_name=root_v,
                            details={
                                "consumer": cons_cell.cell_id,
                                "consumer_step": j,
                                "producer": first_prod_cell.cell_id,
                                "producer_step": first_prod_idx,
                                "variable": root_v,
                            },
                        )

                if root_v and root_v in cell_producers and cons_clause is not None:
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
            reg_p = cls._registry()
            has_clean_eff = any(
                get_effect_kind(eff) in DESTRUCTIVE_OR_CLEAN_EFFECT_KINDS
                for eff in (getattr(cons_cell, "effects", []) or [])
            )
            clean_roles = (
                reg_p.get_roles_for_category("clean")
                | reg_p.get_roles_for_category("transform")
            )
            cons_is_cleaner = has_clean_eff or (str(getattr(cons_cell, "node_role", "") or "").lower() in clean_roles)

            cons_in_v = None
            _df_roles_4b = reg_p.get_roles_for_category("data") | reg_p.get_data_bearing_roles()
            if cons_is_cleaner and cons_clause is not None:
                for port_name, b_val in cons_bindings.items():
                    if isinstance(b_val, str) and _is_var(b_val):
                        port_sig = getattr(cons_cell, "inputs", {}).get(port_name)
                        p_role = str(getattr(port_sig, "port_role", "") or "").lower() if port_sig else ""
                        p_tn = getattr(getattr(port_sig, "signature", port_sig), "type_name", "") if port_sig else ""
                        if not port_sig or p_role in _df_roles_4b or reg_p.is_abstract_carrier(p_tn) or reg_p.get_abstract_carrier(p_tn) is not None:
                            cons_in_v = b_val
                            break
            if cons_in_v and cons_clause is not None:
                for prev_i in range(j):
                    prev_cell, prev_b = pipeline_bindings[prev_i]
                    prev_clause = getattr(prev_cell, "matched_clause_idx", None)
                    if prev_clause is not None and prev_clause > cons_clause:
                        if any(v == cons_in_v or (isinstance(v, str) and _contains_word_token(v, cons_in_v)) for v in prev_b.values()):
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
        if is_formal_pipeline:
            prompt_clauses = [s.strip() for s in (prompt or "").split("->") if s.strip()]
        else:
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

                # Container/receiver carriers are exempt as they flow through sequential updates
                reg_4c = cls._registry()
                _cb_set = reg_4c.get_roles_for_category("receiver")
                _assigned_val_roles = reg_4c.get_roles_for_category("assigned_value")
                is_carrier_port = (
                    p_binds not in _assigned_val_roles
                    and (
                        getattr(p_sig, "is_carrier", False)
                        or p_binds in _cb_set
                        or (p_abstract and reg_4c.is_abstract_carrier(p_abstract))
                        or (p_tn and reg_4c.is_composite_carrier(p_tn))
                    )
                )
                if is_carrier_port:
                    continue

                # Literal parameters or component keys are exempt
                _kb_set = reg_4c.get_roles_for_category("key")
                _kr_set = reg_4c.get_roles_for_category("key") | reg_4c.get_roles_for_category("parameter")
                if p_binds in _kb_set or getattr(p_sig, "port_role", None) in _kr_set or reg_4c.is_config_role(getattr(p_sig, "port_role", None)):
                    continue

                # Extract root variable if projection
                root_var = _extract_expr_root_var(val_expr) or val_expr.strip()
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
                if prod_cell in [c for c, _ in pipeline_bindings]:
                    prod_bindings = next((b for c, b in pipeline_bindings if c is prod_cell), None)
                    if prod_bindings is not None:
                        cand_v = (
                            prod_bindings.get(getattr(prod_prim_out, "name", "") or "") if prod_prim_out is not None else None
                        ) or prod_bindings.get("output_var")
                        if isinstance(cand_v, str):
                            prod_prim_var = _extract_expr_root_var(cand_v)
                if prod_prim_var is not None and prod_prim_var == root_var:
                    cons_prim_in = getattr(cell, "primary_input", None)
                    if cons_prim_in is not None and p_name == getattr(cons_prim_in, "name", None):
                        continue

                bound_parents = getattr(cell, "bound_parent_ids", None) or ()
                if getattr(prod_cell, "cell_id", None) in bound_parents or id(prod_cell) in bound_parents:
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

                # 2. Clause adjacency: immediately preceding clause (cons_clause - 1)
                is_adjacent = (prod_clause == cons_clause - 1)

                if not is_adjacent:
                    msg = (
                        f"Provenance mismatch: Cell '{cell.cell_id}' port '{p_name}' bound variable '{val_expr}' "
                        f"produced by Cell '{prod_cell.cell_id}' (serving clause {prod_clause}) into clause {cons_clause} "
                        f"without clause adjacency or matching prompt literal."
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
        # Check 4d: Uncommitted Optic Component Transforms
        #
        # When an optic getter extracts a focus component (e.g. a column or
        # slice) and subsequent non-scalar transforms mutate it into a new
        # variable, that transformed focus must eventually be consumed
        # (either committed back via an optic setter or fed into a downstream
        # reduction/sink). Leaving a mutated optic focus unreferenced means the
        # intended in-place/carrier mutation was silently dropped.
        # -----------------------------------------------------------------
        all_consumed_vars: Set[str] = set()
        for cell_c, bindings_c in pipeline_bindings:
            if not isinstance(bindings_c, dict):
                continue
            out_keys_c = set((getattr(cell_c, "outputs", {}) or {}).keys()) | {"output_var"}
            for p_n, v_e in bindings_c.items():
                if p_n in out_keys_c or not isinstance(v_e, str) or not _is_var(v_e):
                    continue
                rv_c = _extract_expr_root_var(v_e)
                if rv_c:
                    all_consumed_vars.add(rv_c)

        reg_4d = cls._registry()
        _non_terminal_xform_roles = (
            reg_4d.get_roles_for_category("clean")
            | reg_4d.get_roles_for_category("transform")
        )
        sink_roles_4d = (
            reg_4d.get_terminal_roles_set()
            | reg_4d.get_roles_for_category("terminal_output")
            | reg_4d.get_roles_for_category("model_output")
            | (reg_4d.get_roles_for_category("estimator") - _non_terminal_xform_roles)
        )
        open_focus_chains: Dict[str, Dict[str, Any]] = {}
        for cell_c, bindings_c in pipeline_bindings:
            if not isinstance(bindings_c, dict):
                continue
            cell_outs_map = getattr(cell_c, "outputs", {}) or {}
            out_keys_c = set(cell_outs_map.keys()) | {"output_var"}
            step_out_vars: List[Tuple[str, Any]] = []
            for o_k, o_sig in cell_outs_map.items():
                o_v = bindings_c.get(o_k)
                if isinstance(o_v, str) and _is_var(o_v):
                    rv_o = _extract_expr_root_var(o_v) or o_v.strip()
                    step_out_vars.append((rv_o, o_sig))
            if not step_out_vars and isinstance(bindings_c.get("output_var"), str) and _is_var(bindings_c["output_var"]):
                prim_o = getattr(cell_c, "primary_output", None)
                ov_r = _extract_expr_root_var(bindings_c["output_var"]) or bindings_c["output_var"].strip()
                step_out_vars.append((ov_r, prim_o))

            step_in_vars: Set[str] = set()
            for p_n, v_e in bindings_c.items():
                if p_n in out_keys_c or not isinstance(v_e, str) or not _is_var(v_e):
                    continue
                rv_i = _extract_expr_root_var(v_e)
                if rv_i:
                    step_in_vars.add(rv_i)

            optic_r_c = str(getattr(cell_c, "optic_role", "") or "").strip().lower()
            if optic_r_c == "getter":
                for o_v, _ in step_out_vars:
                    open_focus_chains[o_v] = {"getter": cell_c, "last_cell": cell_c, "modified": False}
            elif optic_r_c == "setter":
                for in_v in step_in_vars:
                    open_focus_chains.pop(in_v, None)
            else:
                matched_ins = [iv for iv in step_in_vars if iv in open_focus_chains]
                if matched_ins:
                    entries = [open_focus_chains.pop(iv) for iv in matched_ins]
                    entry = entries[0]
                    c_nr_4d = str(getattr(cell_c, "node_role", "") or "").strip().lower()
                    _optic_r_4d = reg_4d.get_roles_for_category("optic")
                    is_terminal_or_est = (
                        optic_r_c in _optic_r_4d
                        or getattr(cell_c, "stage", None) == 3
                        or c_nr_4d in sink_roles_4d
                    )
                    if not is_terminal_or_est and step_out_vars:
                        o_v, o_sig = step_out_vars[0]
                        o_sk = str(getattr(o_sig, "structure_kind", "") or "").lower()
                        o_rk = getattr(o_sig, "canonical_rank", None)
                        if o_rk is None:
                            o_rk = getattr(o_sig, "rank", None)
                        o_tn = str(getattr(getattr(o_sig, "signature", o_sig), "type_name", "") or getattr(o_sig, "type_name", "") or "").strip().lower()
                        is_scalar_4d = o_rk == 0 or o_sk == "scalar" or (bool(o_tn) and reg_4d.is_subtype(o_tn, "scalar"))
                        if not is_scalar_4d and o_sk != "handle":
                            open_focus_chains[o_v] = {
                                "getter": entry["getter"],
                                "last_cell": cell_c,
                                "modified": True,
                            }

        for focus_v, entry in open_focus_chains.items():
            if entry.get("modified") and focus_v not in all_consumed_vars:
                g_cid = getattr(entry["getter"], "cell_id", "unknown")
                l_cid = getattr(entry["last_cell"], "cell_id", "unknown")
                msg = (
                    f"Optic focus extracted by '{g_cid}' and mutated via '{l_cid}' into '{focus_v}' "
                    f"was never committed back to its carrier or consumed downstream."
                )
                _add_violation(
                    "uncommitted_optic_transform",
                    msg,
                    cell_id=l_cid,
                    variable_name=focus_v,
                    details={"getter": g_cid, "mutator": l_cid, "variable": focus_v},
                )


        # -----------------------------------------------------------------
        # Check 3b: Explicit Unresolved Port Detection
        # -----------------------------------------------------------------
        reg = cls._registry()
        key_binds = reg.get_roles_for_category("key")
        proj_roles = cls._role_set("get_projection_tokens")
        for cell, bindings in pipeline_bindings:
            for p_name, val in bindings.items():
                if is_unresolved_value(val):
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
        # Check 4: AST Syntax Validation (reuses cached AST parsed before Check 1)
        # -----------------------------------------------------------------
        if code_str:
            if be_lint.is_python_ast:
                if parsed_code_tree is not None:
                    pre_cnt = len(violations)
                    _check_callable_callees(parsed_code_tree, violations)
                    for v in violations[pre_cnt:]:
                        structured_violations.append(
                            PreflightViolation(
                                check_id="callable_callee_constant",
                                message=v,
                                details={"code": code_str},
                            )
                        )
                elif not code_syntax_error_reported:
                    try:
                        parsed_code_tree = ast.parse(code_str)
                    except SyntaxError as e:
                        _add_violation("syntax_error", f"Synthesized code has syntax error: {e}", details={"error": str(e)})
            elif not code_syntax_error_reported:
                ok_4, err_4 = be_lint.validate_syntax(code_str)
                if not ok_4:
                    _add_violation("syntax_error", f"Synthesized code has syntax error: {err_4}", details={"error": str(err_4 or "")})

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
        # Check 10b: Key-Space Causal Def-Use (a column/key must not be read before it is defined).
        # -----------------------------------------------------------------
        try:
            for _ci in cls._verify_causal_key_def_use(pipeline_bindings):
                _add_violation(
                    "column_used_before_defined",
                    f"Cell '{_ci['cell']}' reads column {_ci['column']!r} (port '{_ci['port']}', step {_ci['read_at'] + 1}) before "
                    f"'{_ci['defined_by']}' defines it (step {_ci['defined_at'] + 1}); this raises KeyError at runtime.",
                    cell_id=_ci["cell"], details=_ci)
        except Exception as _cd_err:
            warnings.append(f"column definition analysis unavailable: {type(_cd_err).__name__}: {_cd_err}")


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
        # Check 12: Structured Cell Relations (complement_of, disjoint_from, same_provenance)
        # -----------------------------------------------------------------
        for i, (cell, bindings) in enumerate(pipeline_bindings):
            if not isinstance(bindings, dict):
                continue
            for rel in (getattr(cell, "relations", None) or []):
                if not isinstance(rel, dict):
                    continue
                kind = str(rel.get("kind") or "").lower()
                sp = str(rel.get("source_port") or "")
                tp = str(rel.get("target_port") or "")
                if not sp or not tp:
                    continue
                sv = bindings.get(sp)
                tv = bindings.get(tp)
                if kind in ("disjoint_from", "complement_of") and isinstance(sv, str) and isinstance(tv, str) and _is_var(sv) and sv == tv:
                    _add_violation(
                        "relation_violated",
                        f"Cell '{cell.cell_id}' relation '{kind}({sp}, {tp})' violated: both ports bound to identical variable '{sv}'.",
                        cell_id=cell.cell_id,
                        details={"relation": rel, "bound": sv},
                    )
                elif kind == "same_provenance" and isinstance(sv, str) and isinstance(tv, str) and sv in cell_producers and tv in cell_producers:
                    ps_idx, ps_cell = cell_producers[sv]
                    pt_idx, pt_cell = cell_producers[tv]
                    if ps_cell.domain_name != pt_cell.domain_name:
                        warnings.append(
                            f"Cell '{cell.cell_id}' relation 'same_provenance({sp}, {tp})': "
                            f"'{sv}' (from {ps_cell.cell_id}) and '{tv}' (from {pt_cell.cell_id}) span domains."
                        )

        # -----------------------------------------------------------------
        # Check 9: Prompt-clause coverage (corpus-derived, vocabulary-free).
        # "Sandbox exit code 0" proved nothing about whether the program does
        # what was asked; a path with no FFT or with an unrequested file sink
        # used to lint clean.
        # -----------------------------------------------------------------
        if not is_formal_pipeline:
            try:
                try:
                    from .token_evidence import audit as _cov_audit
                except (ImportError, ValueError):
                    from token_evidence import audit as _cov_audit
                _cells_for_cov = [b[0] for b in pipeline_bindings]
                if orchestrator is None:
                    try:
                        from lattice import LatticeOrchestrator as _LO
                        orchestrator = _LO.get_active_instance()
                    except Exception:
                        orchestrator = None
                if orchestrator is None:
                    warnings.append(
                        "coverage audit SKIPPED: no orchestrator available, so unserved/unrequested prompt clauses "
                        "were NOT checked; do not treat a clean lint as evidence the program does what was asked."
                    )
                _cov = _cov_audit(
                    _cells_for_cov,
                    prompt,
                    orchestrator=orchestrator,
                    clause_semantic_scores=clause_semantic_scores,
                )
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
        refusal_kind: Optional[str] = None
        if not is_valid:
            if any(sv.check_id in {"ambiguity", "ambiguous_plan"} for sv in structured_violations):
                refusal_kind = "AMBIGUITY"
            elif structured_violations and all(
                sv.check_id in {"unresolved_port", "unresolved_key_port", "unbound_data_port", "typed_hole"}
                for sv in structured_violations
            ):
                refusal_kind = "TYPED_HOLE"
            else:
                refusal_kind = "VIOLATION"
        return PreflightLintResult(
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
            structured_violations=structured_violations,
            refusal_kind=refusal_kind,
        )

    @staticmethod
    def audit_lattice(orchestrator: Any) -> Optional[Any]:
        """Runs static reachability, dead-end, and cross-tree auditing on the loaded lattice."""
        try:
            from lattice_auditor import LatticeAuditor
            return LatticeAuditor(orchestrator).audit()
        except Exception:
            return None
