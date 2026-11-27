"""
src/schema.py — Canonical v2 domain-agnostic STRATA tree, cell, port, and language specification schemas,
plus single-source structural type/state helpers shared across the unified engine.
"""
from __future__ import annotations

from typing import Any, Callable, Collection, Dict, List, Literal, Optional, Set, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


TOP_CARRIERS: Tuple[str, ...] = ("", "any", "object", "*", "top", "unknown", "generic", "\u22a4")
TOP_STATES: Tuple[str, ...] = ("", "any", "*", "default", "top", "unregistered", "unbound", "\u22a4")
UNIVERSAL_TOP_SYMBOLS: Tuple[str, ...] = ("any", "*", "top", "unknown", "generic", "\u22a4")
DEFAULT_STATE_SYMBOLS: Tuple[str, ...] = TOP_STATES
UNIVERSAL_BOTTOM_SYMBOLS: Tuple[str, ...] = ("void", "unit", "never", "bottom", "\u22a5")

EXTENT_EFFECT_KINDS: Tuple[str, ...] = ("extent_mutation", "dedup", "schema_removal")
COMPONENT_CLEAN_EFFECT_KINDS: Tuple[str, ...] = ("clean", "component_transform")
DESTRUCTIVE_OR_CLEAN_EFFECT_KINDS: Tuple[str, ...] = ("clean", "dedup", "extent_mutation", "schema_removal")

_CONFIG_PORT_ROLES: Tuple[str, ...] = ("config", "configuration", "literal_parameter", "parameter", "dim_size")
_LITERAL_CARRIER_BASES: Tuple[str, ...] = ("int", "float", "bool", "str", "tuple")
_ENFORCED_PRIMITIVE_CARRIER_BASES: Tuple[str, ...] = (
    "int", "integer", "float", "number", "bool", "boolean", "str", "string", "list", "tuple", "dict"
)
_PRIMITIVE_TUPLE_ELEM_TOKENS: Tuple[str, ...] = ("int", "float", "bool", "str", "...", "any", "number", "scalar")
_ALIGN_STOP_TOKENS: Tuple[str, ...] = ("port", "val", "value", "param", "arg", "str", "int", "float", "bool")

_LEGACY_PORT_SHORTHAND_KEYS: Tuple[str, ...] = (
    "type_name",
    "state",
    "default",
    "role",
    "rank",
    "element_type",
    "shape_contract",
    "structure_kind",
    "abstract_type",
    "semantics",
    "parent_state",
    "qualifiers",
    "rejected_qualifiers",
    "name",
    "type",
    "nominal_type",
    "binds",
    "mutates",
    "supported_formats",
    "param_kind",
    "value_constraints",
    "disjoint_from_port",
    "complement_of_port",
    "key_from_upstream_role",
    "effect_suspended",
    "partition",
    "partition_index",
    "polarity",
    "key_of",
    "key_effect",
)


def get_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Safely read attribute or dict key across schema objects, lightweight snapshots, and dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key)
    inner = getattr(obj, "signature", None)
    if inner is not None:
        if isinstance(inner, dict):
            return inner.get(key, default)
        if hasattr(inner, key):
            return getattr(inner, key)
    return default


def normalize_typestate(raw_state: Optional[str], domain: str = "") -> str:
    """Normalize a typestate string to `'any'` or `'domain::state'`."""
    st = str(raw_state or "any").strip().lower()
    if st in TOP_STATES:
        return "any"
    dom = str(domain or "").strip().lower()
    if "::" not in st and dom:
        return f"{dom}::{st}"
    return st


def is_literal_config_carrier(carrier: Any) -> bool:
    """Return True iff `carrier` is a primitive or primitive-tuple configuration carrier."""
    raw = str(carrier or "").strip()
    if not raw:
        return False
    base = raw.split("[", 1)[0].strip().lower()
    if base not in _LITERAL_CARRIER_BASES:
        return False
    if base == "tuple" and "[" in raw and raw.endswith("]"):
        inner = raw[raw.find("[") + 1 : -1].strip()
        if not inner:
            return True
        for part in inner.split(","):
            tok = part.strip().lower()
            if tok and tok not in _PRIMITIVE_TUPLE_ELEM_TOKENS:
                return False
    return True


def derive_value_class(port: Any) -> str:
    """Derive canonical port value_class ('data'|'literal'|'external_asset'|'identifier'|'enum')."""
    vc = get_attr(port, "value_class", None)
    if isinstance(vc, str) and vc.strip():
        return vc.strip().lower()
    enum_vals = get_attr(port, "enum_values", None)
    if enum_vals:
        return "enum"
    sk = str(get_attr(port, "structure_kind", "") or "").strip().lower()
    at = str(get_attr(port, "abstract_type", "") or "").strip().lower()
    role = str(get_attr(port, "port_role", None) or get_attr(port, "role", "") or "").strip().lower()
    if sk == "path_spec" or at == "path":
        return "external_asset"
    if role in ("column_projection", "source_identifier"):
        return "identifier"
    layout = get_attr(port, "layout", None)
    rank = get_attr(layout, "rank", None)
    dtype = str(get_attr(layout, "dtype", None) or "").strip().lower()
    res_mode = str(get_attr(port, "resource_mode", "value") or "value").strip().lower()
    if role in _CONFIG_PORT_ROLES and res_mode == "value":
        carrier = get_attr(port, "carrier", None) or get_attr(port, "type_name", None)
        if (
            sk == "scalar"
            or is_literal_config_carrier(carrier)
            or (rank == 0 and dtype != "composite_record")
        ):
            return "literal"
    return "data"


def _value_compatible_with_port(val: Any, p_obj: Any) -> bool:
    """Return True iff Python runtime value `val` is carrier-compatible with `p_obj`."""
    car = str(get_attr(p_obj, "carrier", None) or "").strip()
    base = car.split("[", 1)[0].strip().lower()
    if base in TOP_CARRIERS:
        return True
    if isinstance(val, bool):
        return base in ("bool", "boolean")
    if isinstance(val, (int, float)):
        if base in ("int", "integer"):
            return isinstance(val, int) or float(val).is_integer()
        return base in ("float", "number", "tuple", "scalar")
    if isinstance(val, str):
        return base in ("str", "string")
    if isinstance(val, (list, tuple)):
        return base in ("list", "tuple", "sequence", "array", "ndarray")
    if isinstance(val, dict):
        return base in ("dict", "mapping", "object")
    return True


def is_value_compatible_with_primitive_carrier(val: Any, p_obj: Any) -> bool:
    """Enforce Python value compatibility on primitive/container carriers while permitting custom subtypes."""
    car = str(get_attr(p_obj, "carrier", None) or "").strip()
    base = car.split("[", 1)[0].strip().lower()
    if base not in _ENFORCED_PRIMITIVE_CARRIER_BASES:
        return True
    return _value_compatible_with_port(val, p_obj)


def _param_tokens(name: str) -> Set[str]:
    return {
        tok
        for tok in str(name or "").strip().lower().split("_")
        if tok and tok not in _ALIGN_STOP_TOKENS and not tok.isdigit()
    }


def _strict_carrier_match(val: Any, p_obj: Any) -> bool:
    """Strictly match Python value type to declared port carrier (distinguishing int, float, str, bool, tuple/list)."""
    car = str(get_attr(p_obj, "carrier", None) or "").strip()
    base = car.split("[", 1)[0].strip().lower()
    if base in TOP_CARRIERS:
        return True
    if isinstance(val, bool):
        return base in ("bool", "boolean")
    if isinstance(val, int):
        return base in ("int", "integer")
    if isinstance(val, float):
        return base in ("float", "number", "scalar")
    if isinstance(val, str):
        return base in ("str", "string")
    if isinstance(val, (list, tuple)):
        return base in ("list", "tuple", "sequence", "array", "ndarray")
    if isinstance(val, dict):
        return base in ("dict", "mapping", "object")
    return True


def _run_bijective_cfg_pass(
    cfg_subset: List[Tuple[str, Any]],
    eligible: Dict[str, Any],
    mapping: Dict[str, str],
    used_pks: Set[str],
    compat_fn: Callable[[Any, Any], bool],
    raw_prompt: str,
) -> None:
    """Bind remaining config ports via unique 1-to-1 type-compatible bijection."""
    rem_ports = [(p_str, p_obj) for p_str, p_obj in cfg_subset if p_str not in mapping]
    rem_pks = [pk for pk in eligible if pk not in used_pks]
    if not rem_ports or not rem_pks:
        return

    prompt_low = str(raw_prompt or "").lower()
    port_cands: Dict[str, List[str]] = {}
    for p_str, p_obj in rem_ports:
        cands = [
            pk for pk in rem_pks
            if compat_fn(get_attr(eligible[pk], "value", None), p_obj)
        ]
        if len(cands) > 1 and prompt_low:
            car_base = str(get_attr(p_obj, "carrier", None) or "").split("[", 1)[0].strip().lower()
            if car_base in ("str", "string"):
                quoted_cands = []
                for pk in cands:
                    val_s = get_attr(eligible[pk], "value", None)
                    if isinstance(val_s, str):
                        v_l = val_s.strip().lower()
                        if v_l and (f'"{v_l}"' in prompt_low or f"'{v_l}'" in prompt_low):
                            quoted_cands.append(pk)
                if len(quoted_cands) == 1:
                    cands = quoted_cands
        port_cands[p_str] = cands

    pk_to_ports: Dict[str, List[str]] = {pk: [] for pk in rem_pks}
    for p_str, cands in port_cands.items():
        for pk in cands:
            pk_to_ports[pk].append(p_str)

    for p_str, _ in rem_ports:
        cands = port_cands.get(p_str, [])
        if len(cands) == 1:
            pk = cands[0]
            if pk not in used_pks and len(pk_to_ports.get(pk, [])) == 1:
                mapping[p_str] = pk
                used_pks.add(pk)


def align_node_params_to_ports(
    node_params: Dict[str, Any],
    raw_ins: Dict[str, Any],
    in_port_to_slot: Optional[Dict[str, str]] = None,
    *,
    allow_single_fallback: bool = False,
    raw_prompt: str = "",
) -> Dict[str, str]:
    """Align node configuration params to cell input ports via exact name, slot, typestate, and token subset."""
    if not isinstance(node_params, dict) or not isinstance(raw_ins, dict) or not node_params or not raw_ins:
        return {}

    eligible: Dict[str, Any] = {}
    for pk, pv in node_params.items():
        val = get_attr(pv, "value", None)
        prov = str(get_attr(pv, "provenance", "") or "").strip().lower()
        vc_p = str(get_attr(pv, "value_class", "") or "").strip().lower()
        if val is not None and prov in ("prompt", "estimated", "default") and vc_p != "external_asset":
            eligible[str(pk)] = pv
    if not eligible:
        return {}

    mapping: Dict[str, str] = {}
    used_pks: Set[str] = set()

    # Pass 1a: exact port name with strict carrier match
    for pname, p_obj in raw_ins.items():
        p_str = str(pname)
        if p_str in eligible and p_str not in used_pks:
            if _strict_carrier_match(get_attr(eligible[p_str], "value", None), p_obj):
                mapping[p_str] = p_str
                used_pks.add(p_str)

    # Pass 1b: exact port name with relaxed carrier compatibility (unless another param strictly fits)
    for pname, p_obj in raw_ins.items():
        p_str = str(pname)
        if p_str not in mapping and p_str in eligible and p_str not in used_pks:
            if not _value_compatible_with_port(get_attr(eligible[p_str], "value", None), p_obj):
                continue
            if allow_single_fallback and any(
                pk != p_str
                and pk not in used_pks
                and _strict_carrier_match(get_attr(pv, "value", None), p_obj)
                for pk, pv in eligible.items()
            ):
                continue
            mapping[p_str] = p_str
            used_pks.add(p_str)

    # Pass 2: exact template placeholder slot match
    if in_port_to_slot:
        for pname, slot in in_port_to_slot.items():
            p_str = str(pname)
            s_str = str(slot)
            if p_str not in mapping and p_str in raw_ins and s_str in eligible and s_str not in used_pks:
                mapping[p_str] = s_str
                used_pks.add(s_str)

    req_cfg: List[Tuple[str, Any]] = []
    opt_cfg: List[Tuple[str, Any]] = []
    for pname, p_obj in raw_ins.items():
        p_str = str(pname)
        if p_str in mapping:
            continue
        if derive_value_class(p_obj) not in ("literal", "enum", "identifier"):
            continue
        is_req = bool(get_attr(p_obj, "required", True)) and get_attr(p_obj, "default_value", None) is None
        if is_req:
            req_cfg.append((p_str, p_obj))
        else:
            opt_cfg.append((p_str, p_obj))
    cfg_ports = req_cfg + opt_cfg
    if not cfg_ports:
        return mapping

    # Pass 3: exact bare typestate match
    for p_str, p_obj in cfg_ports:
        if p_str in mapping:
            continue
        raw_ts = str(get_attr(p_obj, "typestate", None) or "").strip().lower()
        bare_ts = raw_ts.split("::", 1)[-1] if "::" in raw_ts else raw_ts
        if not bare_ts or bare_ts in TOP_STATES:
            continue
        for pk, pv in eligible.items():
            if pk in used_pks:
                continue
            if pk.strip().lower() == bare_ts and _value_compatible_with_port(get_attr(pv, "value", None), p_obj):
                mapping[p_str] = pk
                used_pks.add(pk)
                break

    # Pass 4: token-subset match against port name or bare typestate
    for p_str, p_obj in cfg_ports:
        if p_str in mapping:
            continue
        raw_ts = str(get_attr(p_obj, "typestate", None) or "").strip().lower()
        bare_ts = raw_ts.split("::", 1)[-1] if "::" in raw_ts else raw_ts
        port_toks = _param_tokens(p_str)
        if bare_ts and bare_ts not in TOP_STATES:
            port_toks = port_toks | _param_tokens(bare_ts)
        if not port_toks:
            continue
        for pk, pv in eligible.items():
            if pk in used_pks:
                continue
            pk_toks = _param_tokens(pk)
            if not pk_toks:
                continue
            if (pk_toks <= port_toks or port_toks <= pk_toks) and _value_compatible_with_port(
                get_attr(pv, "value", None), p_obj
            ):
                mapping[p_str] = pk
                used_pks.add(pk)
                break

    # Pass 5: type-unique bijective matching (strict carrier first, then relaxed coercion)
    if allow_single_fallback:
        _run_bijective_cfg_pass(req_cfg, eligible, mapping, used_pks, _strict_carrier_match, raw_prompt)
        _run_bijective_cfg_pass(opt_cfg, eligible, mapping, used_pks, _strict_carrier_match, raw_prompt)
        _run_bijective_cfg_pass(req_cfg, eligible, mapping, used_pks, _value_compatible_with_port, raw_prompt)
        _run_bijective_cfg_pass(opt_cfg, eligible, mapping, used_pks, _value_compatible_with_port, raw_prompt)

    return mapping


def coerce_literal_for_port(val: Any, p_obj: Any) -> Any:
    """Perform schema-guided literal coercion (list->tuple, scalar->uniform N-tuple, int<->float)."""
    if val is None or p_obj is None:
        return val
    car = str(get_attr(p_obj, "carrier", None) or "").strip()
    base = car.split("[", 1)[0].strip().lower()
    if base == "tuple":
        if isinstance(val, list):
            return tuple(val)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            layout = get_attr(p_obj, "layout", None)
            len_v = get_attr(layout, "length", None)
            shape_v = get_attr(layout, "shape", None)
            arity: Optional[int] = None
            if isinstance(len_v, int) and not isinstance(len_v, bool) and len_v > 0:
                arity = len_v
            elif (
                isinstance(shape_v, (list, tuple))
                and len(shape_v) == 1
                and isinstance(shape_v[0], int)
                and not isinstance(shape_v[0], bool)
                and shape_v[0] > 0
            ):
                arity = shape_v[0]
            elif "[" in car and car.endswith("]") and "..." not in car:
                parts = [pt.strip() for pt in car[car.find("[") + 1 : -1].split(",") if pt.strip()]
                if parts:
                    arity = len(parts)
            if arity is not None and 2 <= arity <= 4:
                elem = (
                    int(val)
                    if (not isinstance(val, int) and float(val).is_integer() and "float" not in car.lower())
                    else val
                )
                return tuple(elem for _ in range(arity))
    if base == "float" and isinstance(val, int) and not isinstance(val, bool):
        return float(val)
    if base == "int" and isinstance(val, float) and val.is_integer():
        return int(val)
    return val


def is_egress_asset_port(port: Any) -> bool:
    """Return True iff an external_asset port designates a write/output destination path."""
    if derive_value_class(port) != "external_asset":
        return False
    role = str(get_attr(port, "port_role", None) or "").strip().lower()
    if role in ("destination", "egress", "output_path", "output_file"):
        return True
    ts = str(get_attr(port, "typestate", None) or "").strip().lower()
    bare_ts = ts.split("::", 1)[-1] if "::" in ts else ts
    return bare_ts == "destination_path"


def get_effect_kind(eff: Any) -> str:
    """Return canonical normalized effect_kind from a ConditionPredicate, object, or dict."""
    if isinstance(eff, dict):
        raw = eff.get("effect_kind", "")
        if raw:
            return str(raw).strip().lower()
        if eff.get("property") == "effect_kind" and eff.get("value"):
            return str(eff.get("value")).strip().lower()
        return ""
    raw = getattr(eff, "effect_kind", "")
    if raw:
        return str(raw).strip().lower()
    if getattr(eff, "property", None) == "effect_kind" and getattr(eff, "value", None):
        return str(getattr(eff, "value")).strip().lower()
    return ""


def cell_has_effect_kind(cell: Any, kinds: Collection[str]) -> bool:
    """Return True iff any declared effect on `cell` has `effect_kind` in `kinds`."""
    return any(get_effect_kind(e) in kinds for e in (get_attr(cell, "effects", None) or ()))


def is_bridge_cell(cell: Any) -> bool:
    """Return True iff `cell` carries an explicit bridge_profile declaration."""
    if cell is None:
        return False
    return get_attr(cell, "bridge_profile", None) is not None


class LanguageSpec(BaseModel):
    """Target-language syntax and runtime specification declared on a core domain tree."""
    model_config = ConfigDict(extra="forbid")

    name: str
    file_extension: str = ""
    ast_backend: str = "lexical"
    runtime: Optional[str] = None
    assignment_operator: str = "="
    statement_terminator: str = ""
    line_comment: str = "#"
    dependency_prefixes: List[str] = Field(default_factory=list)
    declaration_prefixes: List[str] = Field(default_factory=list)
    function_keyword: str = ""
    catch_keyword: str = ""
    subscript_open: str = "["
    subscript_close: str = "]"
    attr_separator: str = "."
    block_keywords: List[str] = Field(default_factory=list)
    sandbox_driver: Optional[str] = None
    forbidden_constructs: List[str] = Field(default_factory=list)
    forbidden_identifiers: List[str] = Field(default_factory=list)
    permitted_builtins: List[str] = Field(default_factory=list)
    variadic_markers: List[str] = Field(default_factory=list)
    null_literal: str = "None"
    bool_literals: List[str] = Field(default_factory=lambda: ["True", "False"])
    empty_collection_literals: List[str] = Field(default_factory=lambda: ["[]", "()"])
    method_receiver_names: List[str] = Field(default_factory=lambda: ["self", "cls"])
    debug_builtins: List[str] = Field(default_factory=list)
    union_type_constructors: List[str] = Field(default_factory=lambda: ["Union"])
    noop_statement: str = "pass"
    return_keyword: str = "return"
    loop_keyword: str = "for"
    loop_in_keyword: str = "in"
    block_suffix: str = ":"
    container_constructors: Dict[str, str] = Field(default_factory=dict)
    collection_append_templates: Dict[str, str] = Field(default_factory=dict)


class ConditionPredicate(BaseModel):
    """Structured relational effect or postcondition predicate on a cell."""
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    property: Optional[str] = None
    operator: Optional[str] = "=="
    value: Optional[Any] = None
    expression: Optional[str] = None
    target: Optional[str] = None
    description: Optional[str] = None
    effect_kind: Optional[str] = None
    from_port: Optional[str] = None
    variable: Optional[str] = None
    state: Optional[str] = None
    predicate: Optional[str] = None


class TypestateDefinition(BaseModel):
    """Declared typestate entry inside a domain typestate vocabulary."""
    model_config = ConfigDict(extra="forbid")

    name: str
    parent_state: Optional[str] = None
    carrier_type: Optional[str] = None
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class LayoutContract(BaseModel):
    """Axis 2: Dimensional, structural, and element-dtype contract."""
    model_config = ConfigDict(extra="forbid")

    rank: Optional[Union[int, str]] = None
    shape: Optional[List[Union[int, str, None]]] = None
    dtype: Optional[str] = None
    rank_polymorphic: bool = False
    channel_count: Optional[int] = None
    channel_axis: Optional[int] = None
    length: Optional[Union[int, str]] = None


class BridgeProfileSchema(BaseModel):
    """Declared cost and information-loss properties for explicit bridge cells."""
    model_config = ConfigDict(extra="forbid")

    is_bridge: bool = False
    delta_axes: List[str] = Field(default_factory=list)
    cost_tier: Optional[float] = None
    lossy: Optional[bool] = None
    requires_intent: bool = False
    notes: Optional[str] = None


class PolymorphicPropagationSchema(BaseModel):
    """Axis preservation annotations across polymorphic operations."""
    model_config = ConfigDict(extra="forbid")

    preserves_layout: bool = False
    preserves_dtype: bool = False
    preserves_typestate: bool = False
    preserves_semantics: bool = False
    preserves_nominal: bool = False
    source_port: Optional[str] = None


class TypestateVocabularySchema(BaseModel):
    """Domain-level typestate vocabulary and state hierarchy."""
    model_config = ConfigDict(extra="forbid")

    domain: str
    states: List[Union[str, TypestateDefinition]] = Field(default_factory=list)
    transitions: List[Dict[str, Any]] = Field(default_factory=list)
    initial_state: Optional[str] = None
    terminal_states: List[str] = Field(default_factory=list)


class RuntimeSetupSchema(BaseModel):
    """Declarative sandbox runtime preparation declared by a domain tree."""
    model_config = ConfigDict(extra="forbid")

    env_vars: Dict[str, str] = Field(default_factory=dict)
    pre_exec_hooks: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class PortSchema(BaseModel):
    """Canonical 5-axis port specification."""
    model_config = ConfigDict(extra="forbid")

    carrier: str
    typestate: str = "any"
    accepted_states: List[str] = Field(default_factory=list)
    layout: Optional[LayoutContract] = None
    invariants: List[str] = Field(default_factory=list)
    value_class: Literal["data", "literal", "external_asset", "identifier", "enum"] = "data"
    resource_mode: Literal["value", "acquire", "borrow", "release"] = "value"
    required: bool = True
    default_value: Optional[Any] = None
    enum_values: Optional[List[Any]] = None
    description: Optional[str] = None
    port_role: Optional[str] = None
    domain: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        had_legacy_shape = any(k in d for k in _LEGACY_PORT_SHORTHAND_KEYS)
        if not d.get("carrier"):
            raw_c = d.get("type_name") or d.get("type")
            if raw_c:
                d["carrier"] = raw_c
        if not d.get("typestate"):
            raw_ts = d.get("state")
            if raw_ts is not None:
                d["typestate"] = str(raw_ts)
        if "default_value" not in d and "default" in d:
            d["default_value"] = d.get("default")
        if "port_role" not in d and d.get("role"):
            d["port_role"] = d.get("role")
        if d.get("layout") is None:
            sc = d.get("shape_contract")
            sc_d = sc if isinstance(sc, dict) else {}
            rank_v = sc_d.get("ndim", d.get("rank"))
            dtype_v = d.get("element_type")
            rp_v = bool(sc_d.get("rank_polymorphic", False))
            cc_v = sc_d.get("channel_count")
            ca_v = sc_d.get("channel_axis")
            len_v = sc_d.get("len")
            if any(v is not None for v in (rank_v, dtype_v, cc_v, ca_v, len_v)) or rp_v:
                d["layout"] = {
                    "rank": rank_v,
                    "shape": sc_d.get("shape"),
                    "dtype": dtype_v,
                    "rank_polymorphic": rp_v,
                    "channel_count": cc_v,
                    "channel_axis": ca_v,
                    "length": len_v,
                }
        if not d.get("value_class"):
            d["value_class"] = derive_value_class(d)
        if had_legacy_shape:
            for lk in _LEGACY_PORT_SHORTHAND_KEYS:
                d.pop(lk, None)
        if d.get("accepted_states") is None:
            d["accepted_states"] = []
        if d.get("invariants") is None:
            d["invariants"] = []
        return d

    def canonical_signature(self, default_domain: str = "") -> Dict[str, Any]:
        """Return canonical domain-namespaced signature dict for this port."""
        dom = str(self.domain or default_domain or "").strip().lower()
        ts = normalize_typestate(self.typestate, dom)
        norm_acc = [
            normalize_typestate(s, dom)
            for s in (self.accepted_states or ())
            if str(s or "").strip()
        ]
        return {
            "carrier": str(self.carrier or "any").strip(),
            "layout": self.layout.model_dump() if self.layout is not None else None,
            "typestate": ts,
            "accepted_states": norm_acc,
            "invariants": list(self.invariants or ()),
            "resource_mode": self.resource_mode,
            "value_class": self.value_class,
            "domain": dom,
        }


class CellSchema(BaseModel):
    """Canonical domain-agnostic operation specification."""
    model_config = ConfigDict(extra="forbid")

    cell_id: str
    domain_name: str = ""
    docstring: str = ""
    code_template: str = ""
    code_templates: Dict[str, str] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    inputs: Dict[str, PortSchema] = Field(default_factory=dict)
    outputs: Dict[str, PortSchema] = Field(default_factory=dict)
    primary_in: Optional[str] = None
    primary_out: Optional[str] = None
    effects: List[Union[ConditionPredicate, str, Dict[str, Any]]] = Field(default_factory=list)
    mutation_type: str = "pure"
    bridge_profile: Optional[BridgeProfileSchema] = None
    polymorphic_propagation: Optional[PolymorphicPropagationSchema] = None
    is_context_manager: bool = False

    @field_validator("code_template", mode="before")
    @classmethod
    def validate_template_syntax(cls, v: Any) -> str:
        if v is None:
            return ""
        v_str = str(v)
        if not v_str.strip():
            return v_str
        try:
            from .language_backend import validate_placeholder_braces
        except ImportError:
            from language_backend import validate_placeholder_braces
        if not validate_placeholder_braces(v_str):
            raise ValueError(f"Unbalanced placeholder braces in code_template: {v_str}")
        return v_str

    def model_post_init(self, __context: Any) -> None:
        if not self.code_template and self.code_templates:
            try:
                from .language_backend import get_active_language_backend
            except ImportError:
                from language_backend import get_active_language_backend
            self.code_template = get_active_language_backend().select_cell_template(self)

    @property
    def primary_input(self) -> Optional[PortSchema]:
        if not self.inputs:
            return None
        if self.primary_in and self.primary_in in self.inputs:
            return self.inputs[self.primary_in]
        required = [p for p in self.inputs.values() if p.required]
        if required:
            data_ports = [p for p in required if p.value_class == "data"]
            return data_ports[0] if data_ports else required[0]
        return next(iter(self.inputs.values()))

    @property
    def primary_output(self) -> Optional[PortSchema]:
        if not self.outputs:
            return None
        if self.primary_out and self.primary_out in self.outputs:
            return self.outputs[self.primary_out]
        return next(iter(self.outputs.values()))


class TreeSchema(BaseModel):
    """Canonical single-file STRATA domain manifest schema."""
    model_config = ConfigDict(extra="forbid")

    domain: str
    version: str = "2.0.0"
    description: str = ""
    is_core: bool = False
    aliases: List[str] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    types: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(default_factory=dict)
    top_types: List[str] = Field(default_factory=list)
    typestates: Optional[TypestateVocabularySchema] = None
    runtime_setup: Optional[RuntimeSetupSchema] = None
    language_spec: Optional[LanguageSpec] = None
    cells: List[CellSchema] = Field(default_factory=list)

    @field_validator("types", mode="before")
    @classmethod
    def _normalize_types(cls, v: Any) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if v is None:
            return {}
        if isinstance(v, (dict, list)):
            return v
        return {}

    @field_validator("typestates", mode="before")
    @classmethod
    def _normalize_typestates(cls, v: Any) -> Optional[TypestateVocabularySchema]:
        if v is None or isinstance(v, TypestateVocabularySchema):
            return v
        if isinstance(v, dict):
            return TypestateVocabularySchema(**v)
        if isinstance(v, (list, tuple)):
            return TypestateVocabularySchema(domain="generic", states=list(v))
        return None

    def model_post_init(self, __context: Any) -> None:
        if self.language_spec is not None:
            try:
                from .language_backend import get_language_backend
            except ImportError:
                from language_backend import get_language_backend
            backend = get_language_backend(self.language_spec)
            for cell in self.cells:
                tmpl = backend.select_cell_template(cell)
                if tmpl:
                    ok, err = backend.validate_syntax(tmpl, allow_placeholders=True)
                    if not ok:
                        raise ValueError(
                            f"Invalid {self.language_spec.name} template in {cell.cell_id}: {tmpl}. Error: {err}"
                        )
