# src/schema.py
from typing import AbstractSet, Collection, Dict, List, Optional, Literal, Any, Union, Set, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

try:
    from language_backend import (
        DEFAULT_STATE_SYMBOLS,
        LexicalLanguageBackend,
        UNIVERSAL_BOTTOM_SYMBOLS,
        UNIVERSAL_TOP_SYMBOLS,
        get_active_language_backend,
        get_language_backend,
        validate_placeholder_braces,
    )
except ImportError:
    from .language_backend import (
        DEFAULT_STATE_SYMBOLS,
        LexicalLanguageBackend,
        UNIVERSAL_BOTTOM_SYMBOLS,
        UNIVERSAL_TOP_SYMBOLS,
        get_active_language_backend,
        get_language_backend,
        validate_placeholder_braces,
    )

_SCHEMA_LEXER = LexicalLanguageBackend(None)


class LanguageSpec(BaseModel):
    """
    Target-language syntax, comment, assignment, and runtime specification
    declared on a core tree (e.g. python_core.json, rust_core.json).
    Allows `src/` to stay 100% target-language-agnostic.
    """
    model_config = ConfigDict(extra="allow")

    name: str
    file_extension: str = ""
    ast_backend: str = "lexical"  # "python_ast" | "lexical"
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


# Canonical domain-agnostic structural schema tuples (from ConditionPredicate, Cell, PortSignature)
EXTENT_EFFECT_KINDS: Tuple[str, ...] = ("extent_mutation", "dedup", "schema_removal")
COMPONENT_CLEAN_EFFECT_KINDS: Tuple[str, ...] = ("clean", "component_transform")
DESTRUCTIVE_OR_CLEAN_EFFECT_KINDS: Tuple[str, ...] = ("clean", "dedup", "extent_mutation", "schema_removal")
STRUCTURAL_COMPOUND_KINDS: Tuple[str, ...] = ("composite", "collection", "tensor", "mapping")
STRUCTURAL_PATH_COMPAT_KINDS: Tuple[str, ...] = ("path_spec", "scalar", "string")
ATOMIC_TYPE_KINDS: Tuple[str, ...] = ("scalar", "numeric", "boolean", "string", "path", "void", "bottom")
SCALAR_PRIMITIVE_KINDS: Tuple[str, ...] = ("scalar", "numeric", "boolean", "string")
COMPOSITE_TYPE_KINDS: Tuple[str, ...] = ("composite", "collection", "array", "mapping")
NON_HANDLE_DATA_KINDS: Tuple[str, ...] = ("tensor", "scalar", "collection", "mapping", "path_spec", "composite")
VALUE_LEAF_KINDS: Tuple[str, ...] = ("scalar", "handle", "path_spec")
UNRESOLVED_SENTINELS: Tuple[str, ...] = ("<UNRESOLVED_PORT>", "<UNRESOLVED>", "<unbound>")


def is_unresolved_value(val: Any) -> bool:
    """True when `val` is an unresolved-port sentinel object or marker string."""
    if val is None:
        return False
    if type(val).__name__ == "_UnresolvedPortSentinel":
        return True
    return isinstance(val, str) and val in UNRESOLVED_SENTINELS


def get_effect_kind(eff: Any) -> str:
    """Returns canonical normalized effect_kind from a ConditionPredicate or dict."""
    if isinstance(eff, dict):
        return str(eff.get("effect_kind", "") or "").strip().lower()
    return str(getattr(eff, "effect_kind", "") or "").strip().lower()


def cell_has_effect_kind(cell: Any, kinds: Collection[str]) -> bool:
    """True if any declared effect on `cell` has `effect_kind` in `kinds`."""
    return any(get_effect_kind(e) in kinds for e in (getattr(cell, "effects", None) or ()))


def get_optic_role(cell: Any) -> str:
    """Returns normalized optic_role ('getter' | 'setter' | 'complement' | 'lens' | '')."""
    return str(getattr(cell, "optic_role", "") or "").strip().lower()


def is_getter_cell(cell: Any) -> bool:
    return get_optic_role(cell) == "getter"


def is_setter_cell(cell: Any) -> bool:
    return get_optic_role(cell) == "setter"


def is_complement_cell(cell: Any) -> bool:
    return get_optic_role(cell) == "complement"


def is_combinator_cell(cell: Any, reg: Optional[Any] = None) -> bool:
    """True iff `cell` is a combinator / macro / higher-order node."""
    if cell is None:
        return False
    if getattr(cell, "is_combinator", False):
        return True
    comb_roles: Collection[str] = reg.get_roles_for_category("combinator") if reg is not None else ("combinator", "macro", "higher_order")
    nr = str(getattr(cell, "node_role", "") or "").strip().lower()
    rl = str(getattr(cell, "role", "") or "").strip().lower()
    nt = str(getattr(cell, "node_type", "") or "").strip().lower()
    ct = str(getattr(cell, "cell_type", "") or "").strip().lower()
    top = str(getattr(cell, "topology_type", "") or "").strip().lower()
    return (
        nr in comb_roles
        or rl in comb_roles
        or nt in comb_roles
        or ct in comb_roles
        or (bool(top) and top in comb_roles)
    )


def is_bridge_cell(cell: Any, reg: Optional[Any] = None) -> bool:
    """True iff `cell` is declared as a bridge / tunnel / adapter node."""
    if cell is None:
        return False
    nr = str(getattr(cell, "node_role", "") or "").strip().lower()
    nt = str(getattr(cell, "node_type", "") or "").strip().lower()
    bridge_roles: Collection[str] = reg.get_bridge_roles_set() if reg is not None else ("bridge", "tunnel", "adapter")
    return nr in bridge_roles or nt in bridge_roles or nt == "tunnel"


def is_macro_cell_dict(raw: Dict[str, Any], reg: Optional[Any] = None, check_slots: bool = False) -> bool:
    """True iff a raw cell dictionary represents a MacroCell / higher-order combinator."""
    nt = str(raw.get("node_type", "") or "").strip().lower()
    nr = str(raw.get("node_role", "") or "").strip().lower()
    top = str(raw.get("topology_type", "") or "").strip().lower()
    comb_roles: Collection[str] = reg.get_roles_for_category("combinator") if reg is not None else ("combinator", "macro", "higher_order")
    if nt in comb_roles or nr in comb_roles or nt.startswith("macro_") or nr.startswith("macro_"):
        return True
    if top and top in comb_roles:
        return True
    if raw.get("sub_cells"):
        return True
    if check_slots and raw.get("slots"):
        return True
    return False


class ConditionPredicate(BaseModel):
    """
    Formal predicate for preconditions and postconditions/effects.
    Domain-agnostic: supports arbitrary properties, operators, and target ports/variables.
    Examples: channels==1, dtype=='binary', is_fitted==True, shape[-1]==3.
    """
    model_config = ConfigDict(extra="ignore")

    target: Optional[str] = None       # Port name or variable identifier
    property: Optional[str] = None     # Property being evaluated
    operator: str = "=="               # "==", "!=", ">", ">=", "<", "<=", "in", "is", "matches"
    value: Any = None                  # Target value
    expression: Optional[str] = None   # Raw string expression
    description: Optional[str] = None  # Human-readable explanation
    name: Optional[str] = None         # Structured predicate / effect name
    effect_kind: Optional[str] = None  # Canonical structural effect kind (e.g. extent_mutation, component_transform, schema_removal, state_mutation)
    from_port: Optional[str] = None    # Port specifying affected items (e.g. "port_1")

    @classmethod
    def from_any(cls, v: Any) -> "ConditionPredicate":
        if isinstance(v, ConditionPredicate):
            return v

        if isinstance(v, str):
            expr = v.strip()
            if not expr:
                return cls(expression=expr)

            parsed = _SCHEMA_LEXER.parse_comparison(expr)
            if parsed is not None:
                return cls(**parsed)

            # Compound / complex expressions are retained intact.
            return cls(expression=expr)

        if isinstance(v, dict):
            if "name" in v or "property" in v or "operator" in v or "expression" in v or "target" in v:
                return cls(**v)
            items = list(v.items())
            if len(items) == 1:
                return cls(property=items[0][0], operator="==", value=items[0][1])
            return cls(
                property=items[0][0],
                operator="==",
                value=items[0][1],
                description=str(v),
            )

        return cls(description=str(v))


class EdgeSchema(BaseModel):
    """
    Explicit transition edge between lattice cells.
    Captures learned/curated transition affinities and bridging contracts.
    """
    model_config = ConfigDict(extra="ignore")

    target_cell_id: str
    affinity_score: float = Field(default=1.0, ge=0.0)
    bridging_precondition: Optional[Union[ConditionPredicate, str, Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("bridging_precondition", mode="before")
    @classmethod
    def normalize_bridging_precondition(cls, v: Any) -> Optional[Any]:
        if v is None:
            return None
        if isinstance(v, (str, dict)):
            return ConditionPredicate.from_any(v)
        return v


class TypestateDefinition(BaseModel):
    """Declarative typestate specification within a domain."""
    model_config = ConfigDict(extra="ignore")

    name: str
    parent_state: Optional[str] = None
    carrier_type: Optional[str] = None
    description: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class ShapeContractSchema(BaseModel):
    """
    Unified formal contract for port tensor shapes and rank constraints.
    Supports rank polymorphism where operations accept arbitrary/variable ndims.
    """
    model_config = ConfigDict(extra="ignore")

    ndim: Optional[int] = None
    shape: Optional[List[Union[int, str]]] = None
    rank_polymorphic: bool = False
    channel_count: Optional[int] = None
    channel_axis: Optional[int] = None
    len: Optional[int] = None

    @classmethod
    def from_any(cls, v: Any) -> Optional["ShapeContractSchema"]:
        if v is None:
            return None
        if isinstance(v, ShapeContractSchema):
            return v
        if isinstance(v, dict):
            ndim = v.get("ndim")
            shape = v.get("shape")
            poly = bool(v.get("rank_polymorphic", False))
            ch = v.get("channel_count")
            ch_ax = v.get("channel_axis")
            l = v.get("len")
            if shape is not None and ndim is None and not poly:
                ndim = len(shape)
            if l is not None and ndim is None:
                ndim = 1
                shape = [l]
            if shape is not None:
                shape = list(shape)
            return cls(
                ndim=ndim,
                shape=shape,
                rank_polymorphic=poly,
                channel_count=ch,
                channel_axis=ch_ax,
                len=l,
            )
        if isinstance(v, str):
            s = v.strip()
            s_low = s.lower()
            if not s or s_low in UNIVERSAL_TOP_SYMBOLS or s_low in UNIVERSAL_BOTTOM_SYMBOLS:
                return cls(
                    ndim=None,
                    shape=None,
                    rank_polymorphic=True,
                    channel_count=None,
                    channel_axis=None,
                    len=None,
                )
            if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
                s = s[1:-1]
            parts = [p.strip() for p in s.split(",") if p.strip()]
            shape_elements: List[Union[int, str]] = []
            for p in parts:
                if p.isdigit():
                    shape_elements.append(int(p))
                else:
                    shape_elements.append(p)
            ndim = len(shape_elements)
            poly = any(p in ("*", "...", "any") for p in shape_elements)
            ch = None
            ch_ax = None
            l = None
            if (
                ndim == 3
                and isinstance(shape_elements[0], str)
                and isinstance(shape_elements[1], str)
                and isinstance(shape_elements[2], int)
            ):
                ch = shape_elements[2]
                ch_ax = -1
            elif ndim == 1 and isinstance(shape_elements[0], int):
                l = shape_elements[0]
            return cls(
                ndim=None if poly else ndim,
                shape=shape_elements,
                rank_polymorphic=poly,
                channel_count=ch,
                channel_axis=ch_ax,
                len=l,
            )
        return None


class LayoutContract(BaseModel):
    """
    Axis 2 (M): Algebraic Memory Layout for a port.
    Separates physical tensor/container rank, symbolic dimensions, dtype, and order
    from semantic interpretations and value invariants.
    Supports polymorphic TypeVar symbols (e.g. rank='R', dtype='D').
    """
    model_config = ConfigDict(extra="ignore")

    rank: Optional[Union[int, str]] = None
    shape: Optional[List[Union[int, str]]] = None
    dtype: Optional[str] = None
    memory_order: Optional[Literal["C", "F", "any"]] = None
    rank_polymorphic: bool = False
    channel_count: Optional[int] = None
    channel_axis: Optional[int] = None
    length: Optional[Union[int, str]] = None


class RelationConstraintSchema(BaseModel):
    """
    Cell-level multi-port coupling constraint beyond single-port 5-axis types.
    Examples: complement_of, disjoint_from, key_of, same_provenance, same_nominal.
    """
    model_config = ConfigDict(extra="ignore")

    kind: Literal["complement_of", "disjoint_from", "key_of", "same_provenance", "same_nominal", "same_length", "custom"]
    source_port: str
    target_port: str
    role: Optional[str] = None
    description: Optional[str] = None


class BridgeProfileSchema(BaseModel):
    """
    Declarative morphism-bridge contract indicating which 5-axis delta(s) a conversion
    cell resolves and whether the conversion is zero-copy, copy, or lossy.
    """
    model_config = ConfigDict(extra="ignore")

    is_bridge: bool = True
    delta_axes: List[Literal["carrier", "layout", "semantics", "typestate", "invariants", "nominal"]] = Field(default_factory=list)
    cost_tier: float = 1.0  # 0.0=subtype/identity, 0.01=view, 1.0=copy/permute, 10.0=lossy projection
    lossy: bool = False
    requires_intent: bool = False


class PolymorphicPropagationSchema(BaseModel):
    """
    Declares whether an operation acts as a structure-preserving functor across
    input ports to output ports (e.g. CLIP preserves BGR semantics; MEAN collapses it).
    """
    model_config = ConfigDict(extra="ignore")

    preserves_semantics: bool = False
    preserves_layout: bool = False
    preserves_typestate: bool = False
    preserves_nominal: bool = False
    source_port: Optional[str] = None


class TypestateVocabularySchema(BaseModel):
    """Domain-level typestate vocabulary and state transitions."""
    model_config = ConfigDict(extra="ignore")

    domain: str
    states: List[Union[str, TypestateDefinition]] = Field(default_factory=list)
    transitions: List[Dict[str, Any]] = Field(default_factory=list)
    initial_state: Optional[str] = None
    terminal_states: List[str] = Field(default_factory=list)


class PortSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    type_name: str
    state: str = "default"
    parent_state: Optional[str] = None
    accepted_states: List[str] = Field(default_factory=list)
    qualifiers: List[Any] = Field(default_factory=list)
    rejected_qualifiers: List[Any] = Field(default_factory=list)
    default_value: Optional[Any] = None
    description: Optional[str] = None
    domain: Optional[str] = None
    required: bool = True
    abstract_type: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    param_kind: Optional[str] = "standard"
    value_constraints: Optional[Dict[str, Any]] = None
    shape_contract: Optional[Union[ShapeContractSchema, Dict[str, Any]]] = None
    port_role: Optional[str] = None
    role: Optional[str] = None
    polarity: Optional[str] = None
    binds: Optional[str] = None
    structure_kind: Optional[str] = None
    rank: Optional[int] = None
    element_type: Optional[str] = None
    supported_formats: List[str] = Field(default_factory=list)
    complement_of_port: Optional[str] = None
    disjoint_from_port: Optional[str] = None
    key_from_upstream_role: Optional[str] = None
    mutates: Optional[str] = None
    key_of: Optional[str] = None
    key_effect: Optional[str] = None
    partition_index: Optional[int] = None
    partition: Optional[Any] = None
    name: Optional[str] = None
    type: Optional[str] = None
    default: Optional[Any] = None
    # --- 5-Axis Product Fields (C, M, Sigma, S, Phi) ---
    carrier: Optional[str] = None
    nominal_type: Optional[str] = None
    layout: Optional[Union[LayoutContract, Dict[str, Any]]] = None
    semantics: Optional[str] = None
    typestate: Optional[str] = None
    invariants: List[str] = Field(default_factory=list)
    effect_suspended: bool = False

    def model_post_init(self, __context: Any) -> None:
        if self.role and not self.port_role:
            self.port_role = self.role
        elif self.port_role and not self.role:
            self.role = self.port_role
        # Populate 5-axis normalized projections cleanly if omitted on legacy JSONs
        if self.carrier is None and self.type_name:
            self.carrier = self.type_name
        if self.layout is None and (self.shape_contract is not None or self.rank is not None or self.element_type is not None):
            sc = self.shape_contract if isinstance(self.shape_contract, dict) else {}
            self.layout = LayoutContract(
                rank=sc.get("ndim", self.rank),
                shape=sc.get("shape"),
                dtype=self.element_type,
                rank_polymorphic=bool(sc.get("rank_polymorphic", False)),
                channel_count=sc.get("channel_count"),
                channel_axis=sc.get("channel_axis"),
                length=sc.get("len"),
            )
        if self.typestate is None and self.state:
            dom = str(self.domain or "").strip().lower()
            st = str(self.state).strip().lower()
            is_top_d = st in DEFAULT_STATE_SYMBOLS
            self.typestate = f"{dom}::{st}" if (dom and "::" not in st and not is_top_d) else st
        if self.semantics is None and self.state:
            dom = str(self.domain or "").strip().lower()
            st = str(self.state).strip().lower()
            is_top_d = st in DEFAULT_STATE_SYMBOLS
            self.semantics = "any" if is_top_d else (f"{dom}.{st}" if (dom and "." not in st and "::" not in st) else st)

    @field_validator("shape_contract", mode="before")
    @classmethod
    def normalize_shape_contract(cls, v: Any) -> Optional[Dict[str, Any]]:
        if v is None:
            return None
        res = ShapeContractSchema.from_any(v)
        return res.model_dump() if res else None

    @field_validator("accepted_states", mode="before")
    @classmethod
    def clean_accepted_states(cls, v: Any) -> List[str]:
        if v is None:
            return []
        return v

    @field_validator("abstract_type", mode="before")
    @classmethod
    def clean_abstract_type(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        v_str = str(v).strip()
        if v_str.lower() in ("", "none", "null"):
            return None
        return v_str


class CellSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cell_id: str

    stage: int = 2

    inputs: Dict[str, PortSchema] = Field(default_factory=dict)
    outputs: Dict[str, PortSchema] = Field(default_factory=dict)
    topology_type: str = "sequential"
    slots: Union[Dict[str, Any], List[str]] = Field(default_factory=list)
    feedback_state_type: Optional[str] = None
    bound_slots: Dict[str, Any] = Field(default_factory=dict)
    code_template: str = ""
    code_templates: Dict[str, str] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)

    # Canonical semantic fields.
    semantic_tags: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)  # alias for semantic_tags
    docstring: Optional[str] = ""
    doc: Optional[str] = None  # alias for docstring

    enrichment_source: Optional[str] = None
    enriched_at: Optional[str] = None
    domain_name: Optional[str] = None
    node_type: Optional[str] = "function"
    node_role: Optional[str] = "function"
    role: Optional[str] = None  # alias for node_role
    verified: bool = True
    source_priority: int = 100
    is_public: bool = True
    mutation_type: str = "pure"
    is_context_manager: bool = False
    raises: List[str] = Field(default_factory=list)
    type_vars: List[str] = Field(default_factory=list)
    endable: Optional[bool] = None
    primary_in: Optional[str] = None
    primary_out: Optional[str] = None
    projection: Optional[str] = None
    optic_role: Optional[Literal["getter", "setter", "complement", "lens"]] = None
    container_port: Optional[str] = None
    key_port: Optional[str] = None
    focus_port: Optional[str] = None

    # --- Macro Node Pipeline ---
    sub_cells: List[str] = Field(default_factory=list)
    internal_topology: Dict[str, List[str]] = Field(default_factory=dict)
    algorithmic_steps: List[str] = Field(default_factory=list)

    # --- Semantic IR & 5-Axis Relational Extensions ---
    preconditions: List[Union[ConditionPredicate, str, Dict[str, Any]]] = Field(default_factory=list)
    postconditions: List[Union[ConditionPredicate, str, Dict[str, Any]]] = Field(default_factory=list)
    effects: List[Union[ConditionPredicate, str, Dict[str, Any]]] = Field(default_factory=list)  # alias for postconditions
    edges: List[Union[EdgeSchema, Dict[str, Any]]] = Field(default_factory=list)
    relations: List[Union[RelationConstraintSchema, Dict[str, Any]]] = Field(default_factory=list)
    bridge_profile: Optional[Union[BridgeProfileSchema, Dict[str, Any]]] = None
    polymorphic_propagation: Optional[Union[PolymorphicPropagationSchema, Dict[str, Any]]] = None

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases_and_stage(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # doc -> docstring
        if "doc" in d:
            if not d.get("docstring"):
                d["docstring"] = d.pop("doc")
            else:
                d.pop("doc", None)

        # keywords -> semantic_tags
        if "keywords" in d:
            kw = d.pop("keywords") or []
            st = d.get("semantic_tags") or []
            if not st:
                d["semantic_tags"] = kw
            else:
                d["semantic_tags"] = list(dict.fromkeys(list(st) + list(kw)))

        # role -> node_role
        if "role" in d:
            if not d.get("node_role"):
                d["node_role"] = d.pop("role")
            else:
                d.pop("role", None)

        # effects -> postconditions
        if "effects" in d:
            ef = d.pop("effects") or []
            pc = d.get("postconditions") or []
            if not pc:
                d["postconditions"] = ef
            elif isinstance(pc, list) and isinstance(ef, list):
                d["postconditions"] = list(pc) + list(ef)

        if "stage" in d:
            v = d["stage"]
            if v is None:
                d["stage"] = 2
            elif isinstance(v, str) and v.isdigit():
                d["stage"] = int(v)
            if d.get("stage") is not None and d.get("stage") < 1:
                d["stage"] = 1

        return d

    @field_validator("inputs", mode="before")
    @classmethod
    def clean_inputs(cls, v: Any) -> Any:
        if isinstance(v, dict):
            cleaned = {}
            for k, val in v.items():
                if k == "dependencies" and isinstance(val, (list, tuple)):
                    continue
                cleaned[k] = val
            return cleaned
        return v

    def model_post_init(self, __context: Any) -> None:
        if not self.code_template and self.code_templates:
            self.code_template = get_active_language_backend().select_cell_template(self)
        if self.doc and not self.docstring:
            self.docstring = self.doc
        if self.docstring and not self.doc:
            self.doc = self.docstring
        if self.role and not self.node_role:
            self.node_role = self.role
        if self.node_role and not self.role:
            self.role = self.node_role
        if self.sub_cells and (not self.node_type or self.node_type == "function"):
            self.node_type = "macro"
            self.node_role = "macro"
            self.role = "macro"
        if self.postconditions and not self.effects:
            self.effects = list(self.postconditions)
        elif self.effects and not self.postconditions:
            self.postconditions = list(self.effects)

    @field_validator("preconditions", mode="before")
    @classmethod
    def normalize_preconditions(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, dict):
            if "property" in v or "expression" in v or "target" in v:
                return [ConditionPredicate.from_any(v)]
            return [ConditionPredicate(property=k, operator="==", value=val) for k, val in v.items()]
        if isinstance(v, (list, tuple)):
            res = []
            for item in v:
                if isinstance(item, dict) and not any(k in item for k in ("property", "expression", "target", "operator")):
                    for k, val in item.items():
                        res.append(ConditionPredicate(property=k, operator="==", value=val))
                elif isinstance(item, (str, dict, ConditionPredicate)):
                    res.append(
                        ConditionPredicate.from_any(item)
                        if not isinstance(item, ConditionPredicate)
                        else item
                    )
                else:
                    res.append(item)
            return res
        return [ConditionPredicate.from_any(v)]

    @field_validator("postconditions", mode="before")
    @classmethod
    def normalize_conditions_list(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, dict):
            if "property" in v or "expression" in v or "target" in v:
                return [ConditionPredicate.from_any(v)]
            return [ConditionPredicate(property=k, operator="==", value=val) for k, val in v.items()]
        if isinstance(v, (list, tuple)):
            res = []
            for item in v:
                if isinstance(item, dict) and not any(k in item for k in ("property", "expression", "target", "operator")):
                    for k, val in item.items():
                        res.append(ConditionPredicate(property=k, operator="==", value=val))
                elif isinstance(item, (str, dict, ConditionPredicate)):
                    res.append(
                        ConditionPredicate.from_any(item)
                        if not isinstance(item, ConditionPredicate)
                        else item
                    )
                else:
                    res.append(item)
            return res
        return [ConditionPredicate.from_any(v)]

    @field_validator("edges", mode="before")
    @classmethod
    def normalize_edges(cls, v: Any) -> List[Any]:
        if v is None:
            return []
        if isinstance(v, dict):
            if "target_cell_id" in v:
                return [EdgeSchema(**v)]
            res = []
            for target_id, edge_info in v.items():
                if isinstance(edge_info, dict):
                    res.append(EdgeSchema(target_cell_id=target_id, **edge_info))
                elif isinstance(edge_info, (int, float)):
                    res.append(EdgeSchema(target_cell_id=target_id, affinity_score=float(edge_info)))
                else:
                    res.append(EdgeSchema(target_cell_id=target_id))
            return res
        if isinstance(v, (list, tuple)):
            res = []
            for item in v:
                if isinstance(item, dict):
                    res.append(EdgeSchema(**item))
                elif isinstance(item, EdgeSchema):
                    res.append(item)
                else:
                    res.append(item)
            return res
        return []

    @field_validator("topology_type")
    @classmethod
    def validate_topology(cls, v: str) -> str:
        if not v or not str(v).strip():
            return "sequential"
        return str(v).strip().lower()

    @property
    def primary_input(self) -> Optional[PortSchema]:
        if not self.inputs:
            return None
        if self.primary_in and self.primary_in in self.inputs:
            return self.inputs[self.primary_in]
        required = [p for p in self.inputs.values() if p.required]
        if required:
            non_scalar = [
                p
                for p in required
                if (p.structure_kind or "").lower() != "scalar"
                and (p.abstract_type or "").lower() != "scalar"
            ]
            return non_scalar[0] if non_scalar else required[0]
        return next(iter(self.inputs.values()))

    @property
    def primary_output(self) -> Optional[PortSchema]:
        if not self.outputs:
            return None
        if self.primary_out and self.primary_out in self.outputs:
            return self.outputs[self.primary_out]
        return next(iter(self.outputs.values()))

    @field_validator("code_template", mode="before")
    @classmethod
    def validate_template_syntax(cls, v: Any) -> str:
        if v is None:
            return ""
        v_str = str(v)
        if not v_str.strip():
            return v_str
        if not validate_placeholder_braces(v_str):
            raise ValueError(f"Unbalanced placeholder braces in code_template: {v_str}")
        return v_str


class TreeSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str
    version: str = "1.0.0"
    description: Optional[str] = None
    language_spec: Optional[LanguageSpec] = None
    target_language: Optional[str] = None
    cells: List[CellSchema]
    types: Optional[Union[Dict[str, Any], List[Any]]] = Field(default_factory=dict)
    typestates: Optional[Union[TypestateVocabularySchema, Dict[str, Any], List[str]]] = None
    type_vars: List[str] = Field(default_factory=list)
    aliases: Dict[str, str] = Field(default_factory=dict)
    top_types: List[str] = Field(default_factory=list)
    top: Optional[Union[bool, List[str]]] = None
    product_constructors: List[str] = Field(default_factory=list)
    artifact_readers: Dict[str, List[str]] = Field(default_factory=dict)
    advisory_qualifiers: List[Union[str, List[str]]] = Field(default_factory=list)
    exclusive_qualifiers: List[List[str]] = Field(default_factory=list)
    exclusive_roles: List[List[str]] = Field(default_factory=list)
    dependent_ports: Dict[str, List[str]] = Field(default_factory=dict)
    forbidden_side_effects: List[str] = Field(default_factory=list)
    abstract_carriers: List[str] = Field(default_factory=list)
    abstract_carrier_mapping: Dict[str, str] = Field(default_factory=dict)
    data_bearing_roles: List[str] = Field(default_factory=list)
    pointed_defaults: Dict[str, Any] = Field(default_factory=dict)
    supported_formats: List[str] = Field(default_factory=list)
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    carrier_roles: Dict[str, Any] = Field(default_factory=dict)
    is_core: Optional[bool] = None
    top_level_carriers: List[str] = Field(default_factory=list)
    top_states: List[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if self.language_spec is not None:
            backend = get_language_backend(self.language_spec)
            for cell in self.cells:
                tmpl = backend.select_cell_template(cell)
                if tmpl:
                    ok, err = backend.validate_syntax(tmpl, allow_placeholders=True)
                    if not ok:
                        raise ValueError(
                            f"Invalid {self.language_spec.name} template in {cell.cell_id}: {tmpl}. Error: {err}"
                        )

    @field_validator("types", mode="before")
    @classmethod
    def normalize_types(cls, v: Any) -> Optional[Union[Dict[str, Any], List[Any]]]:
        if v is None:
            return {}
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("typestates", mode="before")
    @classmethod
    def normalize_typestates(cls, v: Any) -> Optional[Any]:
        if v is None:
            return None
        if isinstance(v, TypestateVocabularySchema):
            return v
        if isinstance(v, dict):
            if "domain" in v:
                return TypestateVocabularySchema(**v)
            return v
        if isinstance(v, (list, tuple)):
            return TypestateVocabularySchema(domain="generic", states=list(v))
        return v
