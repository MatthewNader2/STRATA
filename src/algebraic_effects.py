"""
src/algebraic_effects.py - STRATA (Neuro-Symbolic Topological Lattice)
Solves Issue 1, Issue 4 & Proposal 4.B:
  1. Compositional Property Meet-Semilattice (Heyting Property Algebra) unifying
     atomic string typestates and postcondition predicates.
  2. Universal Algebraic Port Effects (`mutates`, `partition_index`, `key_of`/`key_effect`)
     replacing hardcoded pandas/sklearn/cv2 branches in verification contracts.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

try:
    from .lattice import Cell, PortSignature, AlgebraicSignature, TypeRegistry, is_path_port
except (ImportError, ValueError):
    from lattice import Cell, PortSignature, AlgebraicSignature, TypeRegistry, is_path_port


# ============================================================================
# 1. Compositional Property Meet-Semilattice (Solves Issue 4)
# ============================================================================

@dataclass(frozen=True)
class PropertyBundle:
    """
    Represents a variable's typestate as a monotonic conjunction of atomic state
    tags and key-value property predicates:
      Phi(v) = (active_states, predicates, key_space, partition_tag)
    When a pure or in-place transform `f` is applied to `v`, `Phi(f(v))` inherits
    all non-invalidated predicates of `v` plus the postconditions of `f`.
    """
    carrier_type: str
    active_states: FrozenSet[str] = field(default_factory=frozenset)
    predicates: FrozenSet[Tuple[str, str, Any]] = field(default_factory=frozenset)
    keys_present: FrozenSet[str] = field(default_factory=frozenset)
    keys_deleted: FrozenSet[str] = field(default_factory=frozenset)
    partition_source: Optional[str] = None
    partition_index: Optional[int] = None
    lineage_root: Optional[str] = None

    def transition(
        self,
        new_carrier: Optional[str] = None,
        new_state: Optional[str] = None,
        added_predicates: Sequence[Tuple[str, str, Any]] = (),
        invalidated_props: Set[str] = frozenset(),
        added_keys: Set[str] = frozenset(),
        removed_keys: Set[str] = frozenset(),
        partition_source: Optional[str] = None,
        partition_index: Optional[int] = None,
    ) -> "PropertyBundle":
        """Monotonic semilattice join with selective property invalidation."""
        states = set(self.active_states)
        if new_state and new_state.lower() not in ("any", "*", "default", ""):
            states.add(new_state.lower())

        kept_preds = {
            (prop, op, val)
            for (prop, op, val) in self.predicates
            if prop not in invalidated_props and not any(ap[0] == prop for ap in added_predicates)
        }
        kept_preds.update(added_predicates)

        k_pres = (set(self.keys_present) | set(added_keys)) - set(removed_keys)
        k_del = (set(self.keys_deleted) | set(removed_keys)) - set(added_keys)

        return PropertyBundle(
            carrier_type=new_carrier or self.carrier_type,
            active_states=frozenset(states),
            predicates=frozenset(kept_preds),
            keys_present=frozenset(k_pres),
            keys_deleted=frozenset(k_del),
            partition_source=partition_source if partition_source is not None else self.partition_source,
            partition_index=partition_index if partition_index is not None else self.partition_index,
            lineage_root=self.lineage_root,
        )

    def entails(self, required_state: str, required_preds: Sequence[Tuple[str, str, Any]] = ()) -> bool:
        """
        Checks whether this bundle satisfies a consumer's required state and preconditions:
          Phi(producer) |- Phi(consumer)
        """
        req_s = (required_state or "").strip().lower()
        reg = TypeRegistry.get_instance()
        is_top = (
            not req_s
            or req_s in ("any", "*", "top", "unknown", "⊤")
            or (hasattr(reg, "is_top_state") and reg.is_top_state(req_s))
            or (hasattr(reg, "is_declared_top_state") and reg.is_declared_top_state(req_s))
        )
        if not is_top:
            active_lower = {str(s).lower() for s in self.active_states}
            state_ok = req_s in active_lower or any(
                (hasattr(reg, "is_state_compatible") and reg.is_state_compatible(s, req_s))
                or (hasattr(reg, "is_substate") and reg.is_substate(s, req_s))
                for s in self.active_states
            )
            if not state_ok:
                return False

        pred_map = {(p, op): v for (p, op, v) in self.predicates}
        for req_p, req_op, req_v in required_preds:
            if (req_p, req_op) not in pred_map or pred_map[(req_p, req_op)] != req_v:
                return False
        return True


# ============================================================================
# 2. Universal Algebraic Port Effect Inference (Solves Proposal 4.B & Issue 1)
# ============================================================================

@dataclass
class PortAlgebraicEffect:
    """
    Universal, domain-agnostic port effect annotations:
      - mutates_port    : name of input port mutated or superseded by this output
      - key_of_port     : name of carrier port indexed by this key port
      - key_effect      : "read" | "write" | "delete"
      - partition_index : 0-based partition index for multi-output splitters (e.g. 0=train, 1=test)
    """
    mutates_port: Optional[str] = None
    key_of_port: Optional[str] = None
    key_effect: Optional[str] = None
    partition_index: Optional[int] = None


def infer_cell_algebraic_effects(cell: Cell) -> Dict[str, PortAlgebraicEffect]:
    """
    Derives algebraic port effects strictly from declared cell and port metadata:
      - Explicit port attributes: `mutates`, `key_of`, `key_effect`, `partition_index`
      - Declared cell attributes: `mutation_type == "in_place"` on identical carrier types
      - Explicit partition contracts in schema
    Zero heuristic string matching, zero ad-hoc hardcoded keyword sets, zero cell_id parsing.
    """
    effects_map: Dict[str, PortAlgebraicEffect] = {}
    reg = TypeRegistry.get_instance()
    inputs = getattr(cell, "inputs", {}) or {}
    outputs = getattr(cell, "outputs", {}) or {}

    # Identify declared carrier input port (if any)
    carrier_in_name: Optional[str] = None
    for p_name, p_sig in inputs.items():
        if getattr(p_sig, "binds", None) in ("data_carrier", "carrier") or getattr(p_sig, "port_role", None) == "receiver":
            carrier_in_name = p_name
            break
    if carrier_in_name is None and getattr(cell, "primary_input", None) is not None:
        carrier_in_name = cell.primary_input.name

    # 1. Input port key effects
    for p_name, p_sig in inputs.items():
        explicit_mut = getattr(p_sig, "mutates", None)
        explicit_key_of = getattr(p_sig, "key_of", None)
        explicit_key_eff = getattr(p_sig, "key_effect", None)

        if explicit_mut or explicit_key_of or explicit_key_eff:
            effects_map[f"in:{p_name}"] = PortAlgebraicEffect(
                mutates_port=explicit_mut,
                key_of_port=explicit_key_of,
                key_effect=explicit_key_eff,
            )

    # 2. Output port mutation & partition effects
    for idx, (out_name, out_sig) in enumerate(outputs.items()):
        mut_target = getattr(out_sig, "mutates", None)
        part_idx = getattr(out_sig, "partition_index", None)

        # In-place endomorphism f: T -> T on declared carrier
        if mut_target is None and carrier_in_name and carrier_in_name in inputs:
            in_t = str(getattr(inputs[carrier_in_name], "type_name", "")).lower()
            out_t = str(getattr(out_sig, "type_name", "")).lower()
            if getattr(cell, "mutation_type", "") == "in_place" or (
                in_t and out_t and (in_t == out_t or (reg and reg.is_subtype(out_t, in_t)))
                and getattr(cell, "stage", None) == 2
            ):
                mut_target = carrier_in_name

        # Declared partition metadata
        if part_idx is None and getattr(out_sig, "partition", None) is not None:
            p_decl = getattr(out_sig, "partition")
            if isinstance(p_decl, int):
                part_idx = p_decl
            elif isinstance(p_decl, dict) and "index" in p_decl:
                try:
                    part_idx = int(p_decl["index"])
                except (ValueError, TypeError):
                    pass

        if mut_target is not None or part_idx is not None:
            effects_map[f"out:{out_name}"] = PortAlgebraicEffect(
                mutates_port=mut_target,
                partition_index=part_idx,
            )

    return effects_map


def build_universal_verification_contract(
    pipeline_bindings: List[Tuple[Cell, Dict[str, str]]],
    contract_cls: Any,
) -> Any:
    """
    Solves Issue 1 & Proposal 4.B:
    Constructs `VerificationContract` purely from algebraic port effects (`mutates`,
    `partition_index`, and `postconditions`) with ZERO domain-specific if/elif branches.
    """
    contract = contract_cls()
    latest_carrier_lineage: Dict[str, str] = {}   # root_var -> latest mutated var
    var_to_root: Dict[str, str] = {}              # var -> root_var
    var_partition: Dict[str, Tuple[str, int]] = {}  # var -> (split_cell_id, partition_index)
    partition_zero_by_role: Dict[str, str] = {}   # role/type -> var with partition_index == 0
    accumulated_props: Dict[str, Dict[str, Any]] = {}  # root_var -> {prop: expected_val}

    for cell, bindings in pipeline_bindings:
        eff_map = infer_cell_algebraic_effects(cell)
        stage = getattr(cell, "stage", None)

        # 1. Record declared cell postconditions
        for post in (getattr(cell, "postconditions", []) or []):
            prop = getattr(post, "property", None) or (post.get("property") if isinstance(post, dict) else None)
            op = getattr(post, "operator", "==") if not isinstance(post, dict) else post.get("operator", "==")
            val = getattr(post, "value", None) if not isinstance(post, dict) else post.get("value")
            expr = getattr(post, "expression", None) if not isinstance(post, dict) else post.get("expression")
            desc = getattr(post, "description", None) if not isinstance(post, dict) else post.get("description")
            t_port = (getattr(post, "target", None) if not isinstance(post, dict) else post.get("target")) or "output_var"
            t_var = bindings.get(t_port) or bindings.get("output_var") or (
                bindings.get(cell.primary_output.name) if getattr(cell, "primary_output", None) else None
            )
            if t_var:
                contract.cell_checks.append({
                    "cell_id": cell.cell_id,
                    "target_var": t_var,
                    "target_port": t_port,
                    "property": prop,
                    "operator": op,
                    "value": val,
                    "expression": expr,
                    "description": desc,
                })
                if prop:
                    root = var_to_root.get(t_var, t_var)
                    accumulated_props.setdefault(root, {})[str(prop)] = val

        # 2. Track Linear Carrier Mutation Lineage & Partition Indices
        for out_name, out_sig in (getattr(cell, "outputs", {}) or {}).items():
            out_v = bindings.get(out_name) or bindings.get("output_var")
            if not out_v or not isinstance(out_v, str) or out_v.startswith("None"):
                continue

            out_eff = eff_map.get(f"out:{out_name}")
            if out_eff and out_eff.mutates_port:
                in_v = bindings.get(out_eff.mutates_port)
                if in_v and isinstance(in_v, str):
                    root = var_to_root.get(in_v, in_v)
                    var_to_root[out_v] = root
                    latest_carrier_lineage[root] = out_v
                    if in_v in accumulated_props:
                        accumulated_props.setdefault(root, {}).update(accumulated_props[in_v])
            else:
                var_to_root.setdefault(out_v, out_v)
                latest_carrier_lineage.setdefault(out_v, out_v)

            if out_eff and out_eff.partition_index is not None:
                var_partition[out_v] = (cell.cell_id, out_eff.partition_index)
                if out_eff.partition_index == 0:
                    role_key = str(getattr(out_sig, "port_role", "") or getattr(out_sig, "type_name", "data")).lower()
                    partition_zero_by_role[role_key] = out_v

        # 3. Universal Co-Initial Partition Verification on Multi-Input Consumers
        bound_split_ports: List[Tuple[str, str, Tuple[str, int]]] = []
        for in_name, in_sig in (getattr(cell, "inputs", {}) or {}).items():
            in_v = bindings.get(in_name)
            if isinstance(in_v, str) and in_v in var_partition:
                bound_split_ports.append((in_name, in_v, var_partition[in_v]))

        if bound_split_ports:
            # Check partition coherence (e.g., X_train paired with y_train, never X_train with y_test or unsplit X)
            for in_name, in_v, (split_cid, p_idx) in bound_split_ports:
                in_role = str(getattr(cell.inputs[in_name], "port_role", "") or "").lower()
                expected_p0 = partition_zero_by_role.get(in_role)
                if expected_p0 and p_idx == 0:
                    contract.terminal_checks.append({
                        "type": "algebraic_partition_coherence",
                        "cell_id": cell.cell_id,
                        "port": in_name,
                        "expected_var": expected_p0,
                        "actual_var": in_v,
                    })

        # 4. Universal Linear Egress Verification on Path-Consuming Sinks
        has_path_in = any(is_path_port(p) for p in (getattr(cell, "inputs", {}) or {}).values())
        if has_path_in and stage != 1:
            path_var = None
            data_var = None
            for in_name, in_sig in cell.inputs.items():
                if is_path_port(in_sig):
                    path_var = bindings.get(in_name)
                elif data_var is None:
                    data_var = bindings.get(in_name)

            if data_var and isinstance(data_var, str):
                root = var_to_root.get(data_var, data_var)
                expected_latest = latest_carrier_lineage.get(root, data_var)
                contract.terminal_checks.append({
                    "type": "algebraic_linear_egress",
                    "cell_id": cell.cell_id,
                    "saved_var": data_var,
                    "ingress_var": root,
                    "expected_var": expected_latest,
                    "actual_var": data_var,
                    "output_path": path_var,
                    "expected_properties": dict(accumulated_props.get(root, {})),
                })

    # 5. Universal In-Memory Observable Deliverable Verification
    all_output_vars: Set[str] = set()
    for cell, bindings in pipeline_bindings:
        for out_name in (getattr(cell, "outputs", {}) or {}).keys():
            ov = bindings.get(out_name) or bindings.get("output_var")
            if isinstance(ov, str) and ov and not ov.startswith("None"):
                all_output_vars.add(ov)
        ov = bindings.get("output_var")
        if isinstance(ov, str) and ov and not ov.startswith("None"):
            all_output_vars.add(ov)

    all_downstream_consumed: Set[str] = set()
    for _, b_map in pipeline_bindings:
        for in_k, in_v in b_map.items():
            if isinstance(in_v, str):
                root_v = in_v.split("[")[0].strip()
                if root_v in all_output_vars or root_v.startswith("var_"):
                    all_downstream_consumed.add(root_v)

    n_steps = len(pipeline_bindings)
    for idx, (cell, bindings) in enumerate(pipeline_bindings):
        is_final_step = (idx == n_steps - 1)
        for out_name, out_sig in (getattr(cell, "outputs", {}) or {}).items():
            out_v = bindings.get(out_name) or bindings.get("output_var")
            if not out_v or not isinstance(out_v, str) or out_v.startswith("None"):
                continue

            is_unconsumed = out_v not in all_downstream_consumed
            has_clause = getattr(cell, "matched_clause_idx", None) is not None
            if is_final_step or (is_unconsumed and has_clause):
                if any(c.get("type") == "observable_deliverable" and c.get("var") == out_v for c in contract.terminal_checks):
                    continue

                cell_posts = getattr(cell, "postconditions", []) or []
                contract.terminal_checks.append({
                    "type": "observable_deliverable",
                    "cell_id": cell.cell_id,
                    "var": out_v,
                    "port": out_name,
                    "type_name": getattr(out_sig, "type_name", None),
                    "structure_kind": getattr(out_sig, "structure_kind", None),
                    "is_final": is_final_step,
                    "clause_idx": getattr(cell, "matched_clause_idx", None),
                    "postconditions": [
                        p for p in cell_posts
                        if (getattr(p, "target", None) if not isinstance(p, dict) else p.get("target")) in (out_name, "output_var", None)
                    ],
                })

    return contract


__all__ = [
    "PropertyBundle",
    "PortAlgebraicEffect",
    "infer_cell_algebraic_effects",
    "build_universal_verification_contract",
]
