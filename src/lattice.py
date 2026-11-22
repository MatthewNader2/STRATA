"""
src/lattice.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Domain-Agnostic Typed Lattice, Algebraic Signatures, and Modular Graph Topology.

Conforms strictly to Sections 3.1-3.2 of the STRATA paper:
  Every primitive is represented as a node v = (f, tau_in, tau_out).
  Nodes are organized into a lattice (directed graph) where an edge from
  node u to node v exists iff unify(tau_out(u), tau_in(v)) != bottom.
"""

from __future__ import annotations
import ast
import copy
import functools
import json
import contextlib
import os
import sqlite3
import sys
import threading
try:
    from utils import tokenize_alphanumeric, extract_template_placeholders
except ImportError:
    from .utils import tokenize_alphanumeric, extract_template_placeholders
try:
    from .language_backend import (
        DEFAULT_STATE_SYMBOLS,
        UNIVERSAL_BOTTOM_SYMBOLS,
        UNIVERSAL_TOP_SYMBOLS,
        _mask_placeholders,
        get_active_language_backend,
        set_active_language_spec,
    )
except (ImportError, ValueError):
    from language_backend import (
        DEFAULT_STATE_SYMBOLS,
        UNIVERSAL_BOTTOM_SYMBOLS,
        UNIVERSAL_TOP_SYMBOLS,
        _mask_placeholders,
        get_active_language_backend,
        set_active_language_spec,
    )
from abc import ABC
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Union, Collection, Sequence
import collections.abc

from log_config import get_logger
try:
    from .tokenizer import CellTokenizer
except (ImportError, ValueError):
    from tokenizer import CellTokenizer
try:
    from .config import settings
except (ImportError, ValueError):
    from config import settings
try:
    from .schema import (
        ATOMIC_TYPE_KINDS,
        COMPOSITE_TYPE_KINDS,
        NON_HANDLE_DATA_KINDS,
        SCALAR_PRIMITIVE_KINDS,
        STRUCTURAL_COMPOUND_KINDS,
        STRUCTURAL_PATH_COMPAT_KINDS,
        UNRESOLVED_SENTINELS,
        is_bridge_cell as schema_is_bridge_cell,
        is_macro_cell_dict,
    )
except (ImportError, ValueError):
    from schema import (
        ATOMIC_TYPE_KINDS,
        COMPOSITE_TYPE_KINDS,
        NON_HANDLE_DATA_KINDS,
        SCALAR_PRIMITIVE_KINDS,
        STRUCTURAL_COMPOUND_KINDS,
        STRUCTURAL_PATH_COMPAT_KINDS,
        UNRESOLVED_SENTINELS,
        is_bridge_cell as schema_is_bridge_cell,
        is_macro_cell_dict,
    )

logger = get_logger('lattice')

TREE_SEARCH_DIRS: Tuple[str, ...] = ("trees",)


# Type variables are data-driven: populated from tree JSON `type_vars` fields
# into TypeRegistry._type_vars. The mathematical fallback (single uppercase
# letter) is handled by TypeRegistry.is_type_variable().



# Advisory qualifiers and abstract carriers are data-driven: populated from
# tree JSON declarations (`advisory_qualifiers`, `abstract_carriers` fields)
# into TypeRegistry at ingest time. Access via:
#   TypeRegistry.get_instance().get_advisory_qualifiers()
#   TypeRegistry.get_instance().get_abstract_carriers()

# Backward-compatible module-level proxies for importers that reference
# ABSTRACT_CARRIERS or GENERIC_TYPE_VARIABLE_NAMES. These delegate to
# TypeRegistry at call time — zero hardcoded vocabulary.
class _DynamicAbstractCarriers(collections.abc.Set):
    """Read-only live set view that delegates to TypeRegistry."""
    _from_iterable = staticmethod(set)
    def __contains__(self, item):
        return str(item).strip().lower() in TypeRegistry.get_instance()._abstract_carriers
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_abstract_carriers())
    def __len__(self):
        return len(TypeRegistry.get_instance()._abstract_carriers)

class _DynamicGenericTypeVarNames(collections.abc.Set):
    """Read-only live set view that delegates to TypeRegistry.is_type_variable()."""
    _from_iterable = staticmethod(set)
    def __contains__(self, item):
        return TypeRegistry.get_instance().is_type_variable(str(item))
    def __iter__(self):
        return iter(TypeRegistry.get_instance()._type_vars)
    def __len__(self):
        return len(TypeRegistry.get_instance()._type_vars)

class _DynamicAdvisoryQualifiers(collections.abc.Set):
    """Read-only live set view that delegates to TypeRegistry.get_advisory_qualifiers()."""
    _from_iterable = staticmethod(set)
    def __contains__(self, item):
        return item in TypeRegistry.get_instance().get_advisory_qualifiers()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_advisory_qualifiers())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_advisory_qualifiers())

# These module-level names are imported by unification.py and others
ABSTRACT_CARRIERS = _DynamicAbstractCarriers()
GENERIC_TYPE_VARIABLE_NAMES = _DynamicGenericTypeVarNames()
ADVISORY_QUALIFIERS = _DynamicAdvisoryQualifiers()


class _UnresolvedPortSentinel:
    """
    Singleton sentinel indicating a port could not be resolved during binding.
    Structurally prevented from being treated as a bound variable or literal value.
    """
    __slots__ = ()
    _instance: Optional["_UnresolvedPortSentinel"] = None

    def __new__(cls) -> "_UnresolvedPortSentinel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNRESOLVED_PORT>"

    def __str__(self) -> str:
        return "<UNRESOLVED_PORT>"

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: Any) -> bool:
        return other is self or (isinstance(other, str) and other in UNRESOLVED_SENTINELS)

    def __hash__(self) -> int:
        return hash("<UNRESOLVED_PORT>")


UNRESOLVED_PORT = _UnresolvedPortSentinel()


@dataclass(frozen=True)
class LatticeType:
    """
    Formal Algebraic Free Term for Higher-Kinded Carriers:
      tau ::= B | F[tau_1, ..., tau_n] | Refined(B, phi)
    """
    constructor: str
    parameters: Tuple["LatticeType", ...] = field(default_factory=tuple)
    qualifiers: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_str(cls, s: str) -> "LatticeType":
        if not s:
            return cls(constructor="None")
        s = s.strip()
        quals = ()
        if "{" in s and s.endswith("}"):
            base_part, q_part = s[:-1].split("{", 1)
            s = base_part.strip()
            quals = tuple(q.strip() for q in q_part.split(",") if q.strip())

        if "[" not in s:
            return cls(constructor=s.strip(), qualifiers=quals)

        head, rest = s.split("[", 1)
        inner = rest.rsplit("]", 1)[0]
        parts, depth, cur = [], 0, []
        for ch in inner:
            if ch in "[({":
                depth += 1
            elif ch in "])}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append("".join(cur).strip())
                cur = []
            else:
                cur.append(ch)
        if cur:
            parts.append("".join(cur).strip())
        return cls(
            constructor=head.strip(),
            parameters=tuple(cls.from_str(p) for p in parts),
            qualifiers=quals
        )

    def is_bottom(self) -> bool:
        low = self.constructor.strip().lower()
        return bool(low) and (
            low in _TRUE_BOTTOM_SYMBOLS
            or TypeRegistry.get_instance().is_uninhabited_bottom(low)
        )

    def is_void(self) -> bool:
        low = self.constructor.strip().lower()
        return bool(low) and low in _VOID_UNIT_SYMBOLS

    def is_top(self, is_top_fn: Optional[Callable[[str], bool]] = None) -> bool:
        if is_top_fn is not None:
            return bool(is_top_fn(self.constructor))
        return TypeRegistry.get_instance().is_declared_top(self.constructor)

    def is_subtype_of(
        self,
        other: "LatticeType",
        poset_lookup: Callable[[str, str], bool],
        is_top_fn: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        # 1. Initial bottom axiom: _|_ <= tau for all tau (including _|_)
        if self.is_bottom():
            return True
        if other.is_bottom():
            return False

        # 2. Terminal unit/void isolation: void/None only matches void/None
        if self.is_void() or other.is_void():
            return self.is_void() and other.is_void()

        # 3. Terminal top axiom: tau <= Top for all non-void tau
        if other.is_top(is_top_fn):
            return True
        if self.is_top(is_top_fn):
            return False
        if self == other:
            return True

        # Constructor poset subtyping
        if not poset_lookup(self.constructor, other.constructor):
            return False

        # Qualifiers MUST be checked before raw-type fallback F[T] <= F{q}
        if other.qualifiers:
            if not set(other.qualifiers).issubset(set(self.qualifiers)):
                return False

        # Raw type fallback: F[T1, ..., Tn] <: F (once required qualifiers match)
        if len(other.parameters) == 0:
            return True

        c_low = other.constructor.strip().lower()

        # Variadic tuple subtyping: Tuple[S1, ..., Sn] <: Tuple[T, ...]
        if c_low == "tuple" and len(other.parameters) == 2 and other.parameters[1].constructor in ("...", "Ellipsis"):
            elem_t = other.parameters[0]
            if len(self.parameters) == 2 and self.parameters[1].constructor in ("...", "Ellipsis"):
                return self.parameters[0].is_subtype_of(elem_t, poset_lookup, is_top_fn=is_top_fn)
            if len(self.parameters) > 0:
                return all(p.is_subtype_of(elem_t, poset_lookup, is_top_fn=is_top_fn) for p in self.parameters)
            return False

        # Callable contravariance in parameters, covariance in return type
        if c_low == "callable":
            if len(self.parameters) == 0:
                o_args_ok = len(other.parameters) >= 1 and other.parameters[0].constructor in ("...", "Ellipsis")
                o_ret_ok = len(other.parameters) < 2 or other.parameters[1].is_top(is_top_fn)
                return o_args_ok and o_ret_ok
            if len(self.parameters) == 2 and len(other.parameters) == 2:
                s_args, s_ret = self.parameters
                o_args, o_ret = other.parameters
                if not s_ret.is_subtype_of(o_ret, poset_lookup, is_top_fn=is_top_fn):
                    return False
                if s_args.constructor in ("...", "Ellipsis"):
                    return True
                if o_args.constructor in ("...", "Ellipsis"):
                    return False
                s_list = s_args.parameters if s_args.constructor == "" else (s_args,)
                o_list = o_args.parameters if o_args.constructor == "" else (o_args,)
                if len(s_list) != len(o_list):
                    return False
                return all(
                    oa.is_subtype_of(sa, poset_lookup, is_top_fn=is_top_fn)
                    for sa, oa in zip(s_list, o_list)
                )

        # Sound parameter demand: F </: F[T] if producer is unparameterized
        if len(self.parameters) == 0 and len(other.parameters) > 0:
            return all(p.is_top(is_top_fn) for p in other.parameters)

        if len(self.parameters) != len(other.parameters):
            return False

        var_spec = _CONSTRUCTOR_VARIANCE.get(c_low, "covariant")
        for idx, (p1, p2) in enumerate(zip(self.parameters, other.parameters)):
            pos_var = var_spec[idx] if isinstance(var_spec, tuple) and idx < len(var_spec) else (
                var_spec if isinstance(var_spec, str) else "covariant"
            )
            if pos_var == "invariant":
                if not (
                    p2.is_top(is_top_fn)
                    or (
                        p1.is_subtype_of(p2, poset_lookup, is_top_fn=is_top_fn)
                        and p2.is_subtype_of(p1, poset_lookup, is_top_fn=is_top_fn)
                    )
                ):
                    return False
            elif pos_var == "contravariant":
                if not p2.is_subtype_of(p1, poset_lookup, is_top_fn=is_top_fn):
                    return False
            else:
                if not p1.is_subtype_of(p2, poset_lookup, is_top_fn=is_top_fn):
                    return False
        return True

    def meet(self, other: "LatticeType", reg: Optional["TypeRegistry"] = None) -> "LatticeType":
        """Greatest lower bound (meet: self |\\ other) in the parameterized carrier lattice."""
        if self.is_bottom() or other.is_bottom():
            return LatticeType(constructor="bottom")
        if self.is_void() and other.is_void():
            return self
        if self.is_void() or other.is_void():
            return LatticeType(constructor="bottom")
        r = reg or TypeRegistry.get_instance()
        poset_fn = r._atomic_is_subtype
        top_fn = r.is_declared_top
        if self == other:
            return self
        if other.is_top(top_fn):
            return self
        if self.is_top(top_fn):
            return other
        if self.is_subtype_of(other, poset_fn, is_top_fn=top_fn):
            return self
        if other.is_subtype_of(self, poset_fn, is_top_fn=top_fn):
            return other

        c_meet = r.meet_types(self.constructor, other.constructor)
        if not c_meet or r.is_uninhabited_bottom(c_meet):
            return LatticeType(constructor="bottom")
        quals = tuple(sorted(set(self.qualifiers) | set(other.qualifiers)))

        if not self.parameters and not other.parameters:
            return LatticeType(constructor=c_meet, qualifiers=quals)
        if not self.parameters:
            cand = LatticeType(constructor=c_meet, parameters=other.parameters, qualifiers=quals)
            return cand if (cand.is_subtype_of(self, poset_fn, is_top_fn=top_fn) and cand.is_subtype_of(other, poset_fn, is_top_fn=top_fn)) else LatticeType(constructor="bottom")
        if not other.parameters:
            cand = LatticeType(constructor=c_meet, parameters=self.parameters, qualifiers=quals)
            return cand if (cand.is_subtype_of(self, poset_fn, is_top_fn=top_fn) and cand.is_subtype_of(other, poset_fn, is_top_fn=top_fn)) else LatticeType(constructor="bottom")

        if len(self.parameters) != len(other.parameters):
            return LatticeType(constructor="bottom")

        c_low = c_meet.strip().lower()
        if c_low == "callable" and len(self.parameters) == 2:
            s_args, s_ret = self.parameters
            o_args, o_ret = other.parameters
            m_ret = s_ret.meet(o_ret, r)
            if m_ret.is_bottom():
                return LatticeType(constructor="bottom")
            if s_args.constructor in ("...", "Ellipsis"):
                j_args = o_args
            elif o_args.constructor in ("...", "Ellipsis"):
                j_args = s_args
            elif s_args.constructor == "" and o_args.constructor == "" and len(s_args.parameters) == len(o_args.parameters):
                j_args = LatticeType(
                    constructor="",
                    parameters=tuple(sa.join(oa, r) for sa, oa in zip(s_args.parameters, o_args.parameters)),
                )
            elif s_args.constructor != "" and o_args.constructor != "":
                j_args = s_args.join(o_args, r)
            else:
                return LatticeType(constructor="bottom")
            cand = LatticeType(constructor=c_meet, parameters=(j_args, m_ret), qualifiers=quals)
            return cand if (cand.is_subtype_of(self, poset_fn, is_top_fn=top_fn) and cand.is_subtype_of(other, poset_fn, is_top_fn=top_fn)) else LatticeType(constructor="bottom")

        var_spec = _CONSTRUCTOR_VARIANCE.get(c_low, "covariant")
        merged_params: List[LatticeType] = []
        for idx, (p1, p2) in enumerate(zip(self.parameters, other.parameters)):
            pos_var = var_spec[idx] if isinstance(var_spec, tuple) and idx < len(var_spec) else (
                var_spec if isinstance(var_spec, str) else "covariant"
            )
            if pos_var == "invariant":
                if p1.is_subtype_of(p2, poset_fn, is_top_fn=top_fn) and p2.is_subtype_of(p1, poset_fn, is_top_fn=top_fn):
                    merged_params.append(p1)
                elif p2.is_top(top_fn):
                    merged_params.append(p1)
                elif p1.is_top(top_fn):
                    merged_params.append(p2)
                else:
                    return LatticeType(constructor="bottom")
            elif pos_var == "contravariant":
                merged_params.append(p1.join(p2, r))
            else:
                mp = p1.meet(p2, r)
                if mp.is_bottom():
                    return LatticeType(constructor="bottom")
                merged_params.append(mp)

        cand = LatticeType(constructor=c_meet, parameters=tuple(merged_params), qualifiers=quals)
        if cand.is_subtype_of(self, poset_fn, is_top_fn=top_fn) and cand.is_subtype_of(other, poset_fn, is_top_fn=top_fn):
            return cand
        return LatticeType(constructor="bottom")

    def join(self, other: "LatticeType", reg: Optional["TypeRegistry"] = None) -> "LatticeType":
        """Least upper bound (join: self |_| other) in the parameterized carrier lattice."""
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        if self.is_void() and other.is_void():
            return self
        if self.is_void() or other.is_void():
            return LatticeType(constructor="any")
        r = reg or TypeRegistry.get_instance()
        poset_fn = r._atomic_is_subtype
        top_fn = r.is_declared_top
        if self == other:
            return self
        if self.is_top(top_fn) or other.is_top(top_fn):
            return LatticeType(constructor="any")
        if self.is_subtype_of(other, poset_fn, is_top_fn=top_fn):
            return other
        if other.is_subtype_of(self, poset_fn, is_top_fn=top_fn):
            return self

        c_join = r.join_types(self.constructor, other.constructor)
        if not c_join or r.is_declared_top(c_join):
            return LatticeType(constructor="any")
        quals = tuple(sorted(set(self.qualifiers) & set(other.qualifiers)))

        if not self.parameters or not other.parameters or len(self.parameters) != len(other.parameters):
            return LatticeType(constructor=c_join, qualifiers=quals)

        c_low = c_join.strip().lower()
        if c_low == "callable" and len(self.parameters) == 2:
            s_args, s_ret = self.parameters
            o_args, o_ret = other.parameters
            j_ret = s_ret.join(o_ret, r)
            if s_args.constructor in ("...", "Ellipsis") or o_args.constructor in ("...", "Ellipsis"):
                m_args = LatticeType(constructor="...")
            elif s_args.constructor == "" and o_args.constructor == "" and len(s_args.parameters) == len(o_args.parameters):
                args_meets = tuple(sa.meet(oa, r) for sa, oa in zip(s_args.parameters, o_args.parameters))
                if any(a.is_bottom() for a in args_meets):
                    return LatticeType(constructor=c_join, qualifiers=quals)
                m_args = LatticeType(constructor="", parameters=args_meets)
            elif s_args.constructor != "" and o_args.constructor != "":
                m_args = s_args.meet(o_args, r)
                if m_args.is_bottom():
                    return LatticeType(constructor=c_join, qualifiers=quals)
            else:
                return LatticeType(constructor=c_join, qualifiers=quals)
            cand = LatticeType(constructor=c_join, parameters=(m_args, j_ret), qualifiers=quals)
            return cand if (self.is_subtype_of(cand, poset_fn, is_top_fn=top_fn) and other.is_subtype_of(cand, poset_fn, is_top_fn=top_fn)) else LatticeType(constructor=c_join, qualifiers=quals)

        var_spec = _CONSTRUCTOR_VARIANCE.get(c_low, "covariant")
        merged_params: List[LatticeType] = []
        for idx, (p1, p2) in enumerate(zip(self.parameters, other.parameters)):
            pos_var = var_spec[idx] if isinstance(var_spec, tuple) and idx < len(var_spec) else (
                var_spec if isinstance(var_spec, str) else "covariant"
            )
            if pos_var == "invariant":
                if p1.is_subtype_of(p2, poset_fn, is_top_fn=top_fn) and p2.is_subtype_of(p1, poset_fn, is_top_fn=top_fn):
                    merged_params.append(p1)
                else:
                    return LatticeType(constructor=c_join, qualifiers=quals)
            elif pos_var == "contravariant":
                mp = p1.meet(p2, r)
                if mp.is_bottom():
                    return LatticeType(constructor=c_join, qualifiers=quals)
                merged_params.append(mp)
            else:
                merged_params.append(p1.join(p2, r))

        cand = LatticeType(constructor=c_join, parameters=tuple(merged_params), qualifiers=quals)
        if self.is_subtype_of(cand, poset_fn, is_top_fn=top_fn) and other.is_subtype_of(cand, poset_fn, is_top_fn=top_fn):
            return cand
        return LatticeType(constructor=c_join, qualifiers=quals)

    def __str__(self) -> str:
        s = self.constructor
        if self.parameters:
            s += f"[{', '.join(str(p) for p in self.parameters)}]"
        if self.qualifiers:
            s += f"{{{', '.join(self.qualifiers)}}}"
        return s

_SUBTYPE_CACHE_MAX = 16384
_EMPTY_TUPLE: Tuple[str, ...] = ()
_TRUE_BOTTOM_SYMBOLS: Tuple[str, ...] = ("bottom", "never", "\u22a5")
_VOID_UNIT_SYMBOLS: Tuple[str, ...] = ("void", "unit", "none", "nonetype", "null", "nil", "undefined")
_CONSTRUCTOR_VARIANCE: Dict[str, Any] = {
    "callable": "callable",
    "list": "invariant",
    "dict": "invariant",
    "set": "invariant",
    "mutablemapping": "invariant",
    "mutablesequence": "invariant",
    "mutableset": "invariant",
    "deque": "invariant",
    "mapping": ("invariant", "covariant"),
    "tuple": "covariant",
    "sequence": "covariant",
    "iterable": "covariant",
    "iterator": "covariant",
    "optional": "covariant",
    "union": "covariant",
}
_TOP_STATE_LITERALS: Set[str] = set(DEFAULT_STATE_SYMBOLS) | {"unregistered"}
_TOP_TYPE_FAST: Set[str] = set(UNIVERSAL_TOP_SYMBOLS)
_BOTTOM_LITERALS: Set[str] = set(UNIVERSAL_BOTTOM_SYMBOLS)


def is_macro_cell(cell: Any) -> bool:
    """True iff `cell` is a composite MacroCell (structural attribute check, zero ID-prefix guessing)."""
    if cell is None:
        return False
    return getattr(cell, "cell_type", None) == "macro" or bool(getattr(cell, "sub_cells", None))


def _locked(fn):
    """Run a TypeRegistry method while holding the registry's re-entrant lock."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)
    return wrapper


class TypeRegistry:
    """
    Dynamic Poset Type Hierarchy (T, <=).
    Maintains a directed acyclic graph of subtype relationships.
    Types are registered dynamically from trees without hardcoded domain dependencies.
    Subtyping is verified dynamically using Python's MRO, standard protocols, and poset reachability.
    """
    _instance: Optional[TypeRegistry] = None
    _lock = threading.RLock()

    def __init__(self):
        self._parents: Dict[str, Set[str]] = {}
        self._aliases: Dict[str, str] = {}
        self._state_parents: Dict[str, str] = {}
        self._state_carriers: Dict[str, str] = {}
        self._state_properties: Dict[str, Dict[str, Any]] = {}
        self._domain_state_parents: Dict[Tuple[str, str], str] = {}
        self._domain_state_carriers: Dict[Tuple[str, str], str] = {}
        self._domain_state_properties: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._state_domain_defs: Dict[str, Dict[str, Tuple[Optional[str], Optional[str]]]] = {}
        self._colliding_states: Set[str] = set()
        self._owner_quals: Set[Tuple[str, ...]] = set()
        self._type_vars: Set[str] = set()
        self._declared_top: Set[str] = set()
        self._declared_top_states: Set[str] = set()
        self._product_constructors: Set[str] = set()
        self._function_words: Optional[Tuple[str, ...]] = None
        self._egress_tokens: Set[str] = set()
        self._terminal_states: Set[str] = set()
        self._role_carriers: Set[str] = set()
        self._polarity_hints: Dict[str, Set[str]] = {}
        self._artifact_readers: Dict[str, List[Tuple[str, str]]] = {}
        self._advisory_qualifiers: Set[Tuple[str, ...]] = set()
        self._abstract_carriers: Set[str] = set()
        self._abstract_carrier_mapping: Dict[str, str] = {}
        self._data_bearing_roles: Set[str] = set()
        self._estimator_verbs: Set[str] = set()
        self._projection_tokens: Set[str] = set()
        self._column_projection_tokens = self._projection_tokens
        self._preposition_triggers: Set[str] = set()
        self._generic_prepositions: Set[str] = set()
        self._coordinating_conjunctions: Set[str] = set()
        self._sequencing_connectives: Set[str] = set()
        self._filter_context_tokens: Set[str] = set()
        self._prompt_filler_tokens: Set[str] = set()
        self._asset_placeholders: Dict[str, str] = {}
        self._output_asset_placeholders: Dict[str, str] = {}
        self._operation_tokens: Set[str] = set()
        self._ancestry_cache: Dict[Tuple[str, Tuple[str, ...]], bool] = {}
        # Data-driven extension registries (declared ONLY in tree JSON; empty by
        # default so the engine core never carries domain information):
        #   semantic_flag_patterns  — keyword->flag observers for slot extraction
        #   port_name_type_hints    — declared port-name/affix -> type_name hints
        self._semantic_flag_patterns: List[Dict[str, Any]] = []
        self._port_name_type_hints: Dict[str, str] = {}
        self._port_name_morphology_hints: List[Dict[str, str]] = []
        # Data-driven structural vocabulary (declared in tree JSON; empty by default)
        self._top_level_carriers: Set[str] = set()
        self._registered_supported_formats: Set[str] = set()
        self._state_role_ancestors: Dict[str, Set[str]] = {}
        self._state_role_markers: Dict[str, Tuple[str, ...]] = {}
        self._carrier_roles: Dict[str, Tuple[str, ...]] = {}
        self._identifier_root_states: Set[str] = set()
        self._egress_exclusion_tokens: Set[str] = set()
        self._type_defaults: Dict[str, Any] = {}
        self._capabilities: Dict[str, Any] = {}
        self._language_specs: Dict[str, Any] = {}
        self._active_language_spec: Optional[Any] = None
        self._type_kinds: Dict[str, str] = {}
        self._type_properties: Dict[str, Dict[str, Any]] = {}
        self._type_empty_exprs: Dict[str, str] = {}
        self._exclusive_qualifiers: List[Tuple[str, ...]] = []
        self._exclusive_roles: List[Tuple[str, ...]] = []
        self._dependent_ports: Dict[str, Set[str]] = {}
        self._forbidden_side_effects: Set[str] = set()
        self._control_flow_domains: Set[str] = set()
        self._combinator_topologies: Set[str] = set()
        # Subtype memo: per-instance, invalidated by bumping a generation counter
        # so a result computed against a stale poset can never be cached.
        self._subtype_cache: Dict[Tuple[Any, Any], bool] = {}
        self._meet_cache: Dict[Tuple[str, str], str] = {}
        self._join_cache: Dict[Tuple[str, str], str] = {}
        self._state_raw_cache: Dict[Any, bool] = {}
        self._registered_types_cache: Optional[Set[str]] = None
        self._composite_carrier_cache: Dict[str, bool] = {}
        self._state_properties_cache: Dict[str, Dict[str, Any]] = {}
        self._states_by_property_cache: Dict[Tuple[str, Any], Set[str]] = {}
        self._state_carrier_cache: Dict[str, Optional[str]] = {}
        self._state_compatible_cache: Dict[Tuple[Any, ...], bool] = {}
        self._terminal_sink_states_cache: Optional[Set[str]] = None
        self._path_port_sig_cache: Dict[Any, bool] = {}
        self._unifies_cache: Dict[Tuple[Any, Any], bool] = {}
        self._top_cache: Dict[str, bool] = {}
        self._top_state_cache: Dict[str, bool] = {}
        self._core_domains: Set[str] = set()
        self._sym_to_id: Dict[str, int] = {"": 0}
        self._id_to_sym: List[str] = [""]
        self._type_ancestors_by_id: Optional[Dict[int, Set[int]]] = None
        self._cached_cell_init_roles: Optional[Tuple[Set[str], ...]] = None
        self._cached_abstract_carriers: Optional[Set[str]] = None
        self._cached_top_level_carriers: Optional[Set[str]] = None
        self._cached_terminal_states: Optional[Set[str]] = None
        self._cached_advisory_qualifiers: Optional[Set[Any]] = None
        self._cached_projection_tokens: Optional[Set[str]] = None
        self._cached_edge_sem_bundle: Optional[Tuple[Set[Any], ...]] = None
        self._cached_bridge_cand_bundle: Optional[Tuple[Set[str], ...]] = None
        self._cached_config_roles: Optional[Set[str]] = None
        self._cached_hof_roles: Optional[Set[str]] = None
        self._cached_terminal_roles: Optional[Set[str]] = None
        self._cached_ingress_roles: Optional[Set[str]] = None
        self._cached_bridge_roles: Optional[Set[str]] = None
        self._cached_void_types: Optional[Set[str]] = None
        self._cached_bottom_types: Optional[Set[str]] = None
        self._cached_scalar_types: Optional[Set[str]] = None
        self._cached_top_types_set: Optional[Set[str]] = None
        self._cached_types_by_kind: Dict[str, Set[str]] = {}
        self._cached_type_props: Dict[str, Dict[str, Any]] = {}
        self._cached_types_with_prop: Dict[Tuple[str, Any], Set[str]] = {}
        self._cached_roles_for_cat: Dict[str, Set[str]] = {}
        self._cached_atomic_carrier_roots: Optional[Set[str]] = None
        self._cached_composite_carrier_roots: Optional[Set[str]] = None
        self._generation: int = 0
        self._batch_depth: int = 0
        self._batch_dirty: bool = False
        self._reflected_python_classes: Set[int] = set()
        with self.batch_mutations():
            self._bootstrap_carrier_hierarchy()
            self._bootstrap_state_hierarchy()
            self._bootstrap_plugin_types()

    @property
    def _parents_lower(self) -> Dict[str, Set[str]]:
        return self._parents

    @_locked
    def register_type_default(self, type_name: str, default_value: Any, domain: Optional[str] = None) -> None:
        """Registers a Pointed Monoidal Default eta_tau : 1 -> tau for a type, scoped by domain."""
        k = str(type_name or "").strip().lower()
        if not k:
            return
        d = str(domain).strip().lower() if domain else ""
        if d:
            self._type_defaults[f"{d}:{k}"] = default_value
        import builtins
        if not d or d in self._core_domains:
            self._type_defaults[k] = default_value
        else:
            is_builtin_t = isinstance(getattr(builtins, k, None), type)
            if not is_builtin_t and k not in self._type_defaults:
                self._type_defaults[k] = default_value

    def get_type_default(self, type_name: str, domain: Optional[str] = None) -> Any:
        """Resolves the Pointed Monoidal Default eta_tau : 1 -> tau via poset reachability."""
        k = str(type_name or "").strip().lower()
        if not k:
            return None
        d = str(domain or "").strip().lower() if domain else ""
        if d and f"{d}:{k}" in self._type_defaults:
            return self._type_defaults[f"{d}:{k}"]
        if k in self._type_defaults:
            return self._type_defaults[k]
        # Walk supertype poset via BFS
        visited = {k}
        queue = deque(self._parents.get(k, ()))
        while queue:
            curr = queue.popleft()
            if curr in visited:
                continue
            visited.add(curr)
            if d and f"{d}:{curr}" in self._type_defaults:
                return self._type_defaults[f"{d}:{curr}"]
            if curr in self._type_defaults:
                return self._type_defaults[curr]
            for p in self._parents.get(curr, ()):
                if p not in visited:
                    queue.append(p)
        return None

    @_locked
    def register_capabilities(self, caps: Dict[str, Any]) -> None:
        if isinstance(caps, dict):
            for k, v in caps.items():
                if isinstance(v, list) and isinstance(self._capabilities.get(k), list):
                    self._capabilities[k] = list(dict.fromkeys(self._capabilities[k] + v))
                elif isinstance(v, dict) and isinstance(self._capabilities.get(k), dict):
                    merged = dict(self._capabilities[k])
                    merged.update(v)
                    self._capabilities[k] = merged
                else:
                    self._capabilities[k] = v

    def get_capabilities(self) -> Dict[str, Any]:
        return dict(self._capabilities)

    # --- Language spec & first-class schema metadata -------------------------
    @_locked
    def register_language_spec(self, spec: Any) -> None:
        if spec is None:
            return
        if isinstance(spec, dict):
            try:
                from .schema import LanguageSpec
            except (ImportError, ValueError):
                from schema import LanguageSpec
            spec_obj = LanguageSpec(**spec)
        else:
            spec_obj = spec
        key = str(getattr(spec_obj, "name", "") or "").strip().lower()
        if key:
            self._language_specs[key] = spec_obj
        self._active_language_spec = spec_obj
        set_active_language_spec(spec_obj)
        if str(getattr(spec_obj, "runtime", "") or "").strip().lower() == "python":
            try:
                from .runtime_poset import bootstrap_python_runtime_poset
            except (ImportError, ValueError):
                from runtime_poset import bootstrap_python_runtime_poset
            with self.batch_mutations():
                bootstrap_python_runtime_poset(self)
        self._invalidate_caches()

    def get_language_spec(self, name: Optional[str] = None) -> Optional[Any]:
        if name:
            return self._language_specs.get(str(name).strip().lower())
        return self._active_language_spec

    @_locked
    def register_type_kind(self, type_name: str, kind: str) -> None:
        t = str(type_name or "").strip().lower()
        k = str(kind or "").strip().lower()
        if t and k and self._type_kinds.get(t) != k:
            self._type_kinds[t] = k
            self._invalidate_caches()

    def get_types_by_kind(self, kind: str) -> Set[str]:
        k = str(kind or "").strip().lower()
        if not k:
            return set()
        cached = self._cached_types_by_kind.get(k)
        if cached is not None:
            return cached
        gen = self._generation
        direct = {t for t, tk in self._type_kinds.items() if tk == k}
        if k in ("top", "bottom"):
            # Top/bottom symbols do not expand downward via poset closure on mutation checks.
            res = set(direct)
            with self._lock:
                if gen == self._generation:
                    self._cached_types_by_kind[k] = res
            return res
        anc_map = self._get_type_ancestors_by_id()
        kind_ids = {self.intern_symbol(t) for t in direct}
        kind_sym = self.intern_symbol(k)
        if kind_sym:
            kind_ids.add(kind_sym)
        res = set(direct)
        for tid, ancs in anc_map.items():
            if ancs & kind_ids:
                sym = self._id_to_sym[tid]
                if sym:
                    res.add(sym)
        with self._lock:
            if gen == self._generation:
                self._cached_types_by_kind[k] = res
        return res

    def has_type_kind(self, type_name: Optional[str], kind: str) -> bool:
        t = str(type_name or "").strip().lower()
        return bool(t) and t in self.get_types_by_kind(kind)

    @_locked
    def register_type_properties(self, type_name: str, props: Dict[str, Any]) -> None:
        t = str(type_name or "").strip().lower()
        if not t or not isinstance(props, dict):
            return
        cur = self._type_properties.setdefault(t, {})
        changed = False
        for pk, pv in props.items():
            sk = str(pk).strip()
            if sk and cur.get(sk) != pv:
                cur[sk] = pv
                changed = True
        if changed:
            self._invalidate_caches()

    def get_type_properties(self, type_name: Optional[str]) -> Dict[str, Any]:
        t = str(type_name or "").strip().lower()
        if not t:
            return {}
        cached = self._cached_type_props.get(t)
        if cached is not None:
            return cached
        gen = self._generation
        merged: Dict[str, Any] = {}
        # Walk BFS ancestors root-down (or child overrides ancestors)
        order: List[str] = []
        seen: Set[str] = {t}
        q: deque = deque([t])
        while q:
            cur = q.popleft()
            order.append(cur)
            for par in self._parents.get(cur, ()):
                pl = str(par).strip().lower()
                if pl and pl not in seen:
                    seen.add(pl)
                    q.append(pl)
        for anc in reversed(order):
            merged.update(self._type_properties.get(anc, {}))
        with self._lock:
            if gen == self._generation:
                self._cached_type_props[t] = merged
        return merged

    def get_types_with_property(self, prop_key: str, expected: Any = True) -> Set[str]:
        cache_key = (prop_key, expected)
        try:
            cached = self._cached_types_with_prop.get(cache_key)
            if cached is not None:
                return set(cached)
        except TypeError:
            pass
        gen = self._generation
        res: Set[str] = set()
        candidates = set(self._type_properties.keys()) | set(self._parents.keys())
        for t in candidates:
            props = self.get_type_properties(t)
            if props.get(prop_key) == expected:
                res.add(t)
        try:
            with self._lock:
                if gen == self._generation:
                    self._cached_types_with_prop[cache_key] = set(res)
        except TypeError:
            pass
        return res

    def get_type_property_values(self, prop_key: str) -> Set[str]:
        out: Set[str] = set()
        for props in self._type_properties.values():
            v = props.get(prop_key)
            if isinstance(v, (list, tuple, set)):
                for item in v:
                    s = str(item).strip()
                    if s:
                        out.add(s)
            elif isinstance(v, str) and v.strip():
                out.add(v.strip())
        return out

    @_locked
    def register_type_empty_expr(self, type_name: str, expr: Any, domain: Optional[str] = None) -> None:
        t = str(type_name or "").strip().lower()
        if not t or expr is None:
            return
        d = str(domain or "").strip().lower()
        if d:
            self._type_empty_exprs[f"{d}:{t}"] = str(expr)
        if not d or d in self._core_domains or t not in self._type_empty_exprs:
            self._type_empty_exprs[t] = str(expr)
        self.register_type_default(type_name, expr, domain=domain)

    def get_type_empty_expr(self, type_name: str, domain: Optional[str] = None) -> Optional[str]:
        t = str(type_name or "").strip().lower()
        d = str(domain or "").strip().lower()
        if d and f"{d}:{t}" in self._type_empty_exprs:
            return self._type_empty_exprs[f"{d}:{t}"]
        if t in self._type_empty_exprs:
            return self._type_empty_exprs[t]
        val = self.get_type_default(type_name, domain=domain)
        return str(val) if val is not None else None

    def get_states_by_property(self, prop_key: str, expected: Any = True) -> Set[str]:
        key = str(prop_key).strip().lower()
        cache_key = (key, expected)
        try:
            cached = self._states_by_property_cache.get(cache_key)
            if cached is not None:
                return set(cached)
        except TypeError:
            pass
        gen = self._generation
        out: Set[str] = set()
        all_states = set(self._state_properties.keys()) | set(self._state_parents.keys())
        for st in all_states:
            props = self._get_state_properties_ref(st)
            if any(str(k).strip().lower() == key and v == expected for k, v in props.items()):
                out.add(st)
        for (_, st), d_props in self._domain_state_properties.items():
            if any(str(k).strip().lower() == key and v == expected for k, v in d_props.items()):
                out.add(st)
        try:
            with self._lock:
                if gen == self._generation:
                    self._states_by_property_cache[cache_key] = set(out)
        except TypeError:
            pass
        return out

    def get_declared_state_property_keys(self) -> Set[str]:
        keys: Set[str] = set()
        for props in self._state_properties.values():
            for k in props:
                keys.add(str(k).strip().lower())
        for d_props in self._domain_state_properties.values():
            for k in d_props:
                keys.add(str(k).strip().lower())
        return keys

    @_locked
    def register_exclusive_qualifiers(self, groups: Any) -> None:
        changed = False
        for grp in groups or ():
            tup = tuple(str(x).strip().lower() for x in (grp if isinstance(grp, (list, tuple, set)) else (grp,)) if str(x).strip())
            if tup and tup not in self._exclusive_qualifiers:
                self._exclusive_qualifiers.append(tup)
                changed = True
        if changed:
            self._invalidate_caches()

    def get_exclusive_qualifiers(self) -> List[Tuple[str, ...]]:
        return list(self._exclusive_qualifiers)

    @_locked
    def register_exclusive_roles(self, groups: Any) -> None:
        changed = False
        for grp in groups or ():
            tup = tuple(str(x).strip().lower() for x in (grp if isinstance(grp, (list, tuple, set)) else (grp,)) if str(x).strip())
            if tup and tup not in self._exclusive_roles:
                self._exclusive_roles.append(tup)
                changed = True
        if changed:
            self._invalidate_caches()

    def get_exclusive_roles(self) -> List[Tuple[str, ...]]:
        return list(self._exclusive_roles)

    @_locked
    def register_dependent_ports(self, mapping: Dict[str, Any]) -> None:
        changed = False
        for k, v in (mapping or {}).items():
            kl = str(k).strip().lower()
            vals = {str(x).strip().lower() for x in (v if isinstance(v, (list, tuple, set)) else (v,)) if str(x).strip()}
            if kl and vals:
                cur = self._dependent_ports.setdefault(kl, set())
                if not vals.issubset(cur):
                    cur.update(vals)
                    changed = True
        if changed:
            self._invalidate_caches()

    def get_dependent_ports(self) -> Dict[str, Set[str]]:
        return {k: set(v) for k, v in self._dependent_ports.items()}

    @_locked
    def register_forbidden_side_effects(self, methods: Any) -> None:
        for m in methods or ():
            s = str(m).strip()
            if s:
                self._forbidden_side_effects.add(s)

    def get_forbidden_side_effects(self) -> Set[str]:
        res = set(self._forbidden_side_effects)
        if self._active_language_spec is not None:
            res.update(str(x).strip() for x in (getattr(self._active_language_spec, "forbidden_side_effects", None) or ()) if x)
        return res

    def get_roles_for_category(self, category: str) -> Set[str]:
        cat = str(category or "").strip().lower()
        cached = self._cached_roles_for_cat.get(cat)
        if cached is not None:
            return set(cached)
        gen = self._generation
        res: Set[str] = {cat} if cat else set()
        if cat in self._carrier_roles:
            res.update(self._carrier_roles[cat])
        with self._lock:
            if gen == self._generation:
                self._cached_roles_for_cat[cat] = set(res)
        return res

    def get_atomic_carrier_roots(self) -> Set[str]:
        cached = self._cached_atomic_carrier_roots
        if cached is not None:
            return set(cached)
        gen = self._generation
        roots: Set[str] = set()
        for k in ATOMIC_TYPE_KINDS:
            roots.add(k)
            roots.update(self.get_types_by_kind(k))
        roots.update(self.get_types_with_property("is_atomic", True))
        with self._lock:
            if gen == self._generation:
                self._cached_atomic_carrier_roots = set(roots)
        return roots

    def get_composite_carrier_roots(self) -> Set[str]:
        cached = self._cached_composite_carrier_roots
        if cached is not None:
            return set(cached)
        gen = self._generation
        roots: Set[str] = set(COMPOSITE_TYPE_KINDS)
        for k in COMPOSITE_TYPE_KINDS:
            roots.update(self.get_types_by_kind(k))
        roots.update(self.get_types_with_property("is_composite", True))
        with self._lock:
            if gen == self._generation:
                self._cached_composite_carrier_roots = set(roots)
        return roots

    def _bootstrap_carrier_hierarchy(self):
        """Track 10: Dynamically bootstraps language-level poset & POS categories via runtime reflection."""
        try:
            from .runtime_poset import bootstrap_runtime_poset
        except (ImportError, ValueError):
            from runtime_poset import bootstrap_runtime_poset
        bootstrap_runtime_poset(self)

    def _bootstrap_state_hierarchy(self):
        """Bootstraps universal typestates from first principles."""
        for s in _TOP_STATE_LITERALS:
            self.register_top_state(s)
            if s:
                self.register_state(s, None)


    @_locked
    def register_artifact_reader(self, category: str, module_name: str, function_name: str) -> None:
        cat = category.strip().lower()
        if cat not in self._artifact_readers:
            self._artifact_readers[cat] = []
        entry = (module_name.strip(), function_name.strip())
        if entry not in self._artifact_readers[cat]:
            self._artifact_readers[cat].append(entry)

    @_locked
    def _bootstrap_plugin_types(self):
        """Dynamically ingests domain plugin types and structural poset declarations from trees/*.json."""
        import glob, json
        active_orch = None
        try:
            active_orch = LatticeOrchestrator.get_active_instance()
        except Exception:
            active_orch = None

        search_dirs: List[str] = []
        active_dir = getattr(active_orch, "trees_directory", None) if active_orch is not None else None
        if active_dir:
            search_dirs.append(str(active_dir))
            if os.path.basename(os.path.normpath(str(active_dir))) in TREE_SEARCH_DIRS:
                search_dirs.extend(TREE_SEARCH_DIRS)
                try:
                    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    for rel in TREE_SEARCH_DIRS:
                        search_dirs.append(os.path.join(root_dir, rel))
                except Exception:
                    pass
        else:
            search_dirs = list(TREE_SEARCH_DIRS)
            try:
                root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                for rel in TREE_SEARCH_DIRS:
                    search_dirs.append(os.path.join(root_dir, rel))
            except Exception:
                pass
            try:
                try:
                    from .config import settings as _settings
                except (ImportError, ValueError):
                    from config import settings as _settings
                if _settings.trees_dir:
                    search_dirs.append(str(_settings.trees_dir))
            except Exception:
                pass

        seen_dirs: Set[str] = set()
        for s_dir in search_dirs:
            if not s_dir:
                continue
            canon_dir = os.path.realpath(s_dir)
            if canon_dir in seen_dirs or not os.path.isdir(canon_dir):
                continue
            seen_dirs.add(canon_dir)
            for p in sorted(glob.glob(os.path.join(canon_dir, "*.json"))):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        dom = str(data.get("domain") or "").strip().lower() or None
                        if dom and data.get("is_core"):
                            self._core_domains.add(dom)
                        if isinstance(data.get("language_spec"), dict):
                            self.register_language_spec(data["language_spec"])
                        if isinstance(data.get("exclusive_qualifiers"), list):
                            self.register_exclusive_qualifiers(data["exclusive_qualifiers"])
                        if isinstance(data.get("exclusive_roles"), list):
                            self.register_exclusive_roles(data["exclusive_roles"])
                        if isinstance(data.get("dependent_ports"), dict):
                            self.register_dependent_ports(data["dependent_ports"])
                        if isinstance(data.get("forbidden_side_effects"), list):
                            self.register_forbidden_side_effects(data["forbidden_side_effects"])
                        if isinstance(data.get("top_level_carriers"), list):
                            self.register_top_level_carriers(data["top_level_carriers"])
                        if isinstance(data.get("state_role_ancestors"), dict):
                            self.register_state_role_ancestors(data["state_role_ancestors"])
                        if isinstance(data.get("state_role_markers"), dict):
                            self.register_state_role_markers(data["state_role_markers"])
                        if isinstance(data.get("carrier_roles"), dict):
                            self.register_carrier_roles(data["carrier_roles"])
                        if isinstance(data.get("identifier_root_states"), list):
                            self.register_identifier_root_states(data["identifier_root_states"])
                        if isinstance(data.get("supported_formats"), list):
                            self.register_supported_formats(data["supported_formats"])
                        if isinstance(data.get("capabilities"), dict):
                            self.register_capabilities(data["capabilities"])
                        if isinstance(data.get("pointed_defaults"), dict):
                            for td_k, td_v in data["pointed_defaults"].items():
                                self.register_type_default(td_k, td_v, domain=dom)
                        raw_types = data.get("types")
                        if isinstance(raw_types, dict):
                            for t_name, t_meta in raw_types.items():
                                if isinstance(t_meta, dict):
                                    parents = t_meta.get("parents") or ([t_meta["parent"]] if t_meta.get("parent") else [])
                                    if t_meta.get("kind"):
                                        self.register_type_kind(t_name, t_meta["kind"])
                                    if isinstance(t_meta.get("properties"), dict):
                                        self.register_type_properties(t_name, t_meta["properties"])
                                    if "empty_expr" in t_meta:
                                        self.register_type_empty_expr(t_name, t_meta["empty_expr"], domain=dom)
                                    elif "default_value" in t_meta:
                                        self.register_type_default(t_name, t_meta["default_value"], domain=dom)
                                    elif "pointed_default" in t_meta:
                                        self.register_type_default(t_name, t_meta["pointed_default"], domain=dom)
                                else:
                                    parents = [str(t_meta)]
                                if not parents:
                                    self.register_type(t_name, None)
                                for parent in parents:
                                    self.register_type(t_name, parent or None)
                        elif isinstance(raw_types, list):
                            for item in raw_types:
                                if isinstance(item, str):
                                    self.register_type(item, None)
                                elif isinstance(item, dict):
                                    t_name = item.get("name") or item.get("type_name")
                                    if not t_name:
                                        continue
                                    if item.get("kind"):
                                        self.register_type_kind(t_name, item["kind"])
                                    if isinstance(item.get("properties"), dict):
                                        self.register_type_properties(t_name, item["properties"])
                                    if "empty_expr" in item:
                                        self.register_type_empty_expr(t_name, item["empty_expr"], domain=dom)
                                    elif "default_value" in item:
                                        self.register_type_default(t_name, item["default_value"], domain=dom)
                                    elif "pointed_default" in item:
                                        self.register_type_default(t_name, item["pointed_default"], domain=dom)
                                    parents = item.get("parents") or ([item.get("parent") or item.get("parent_type")] if (item.get("parent") or item.get("parent_type")) else [])
                                    if not parents:
                                        self.register_type(t_name, None)
                                    for parent in parents:
                                        self.register_type(t_name, parent or None)
                        if "type_vars" in data and isinstance(data["type_vars"], list):
                            self.register_type_vars(data["type_vars"])
                        if "aliases" in data and isinstance(data["aliases"], dict):
                            for a_k, a_v in data["aliases"].items():
                                self.register_alias(a_k, a_v)
                        if "top_types" in data and isinstance(data["top_types"], list):
                            for top_t in data["top_types"]:
                                self.register_top(top_t)
                        if "top_states" in data and isinstance(data["top_states"], list):
                            for top_s in data["top_states"]:
                                self.register_top_state(top_s)
                        if "product_constructors" in data and isinstance(data["product_constructors"], list):
                            for p_ctor in data["product_constructors"]:
                                self.register_product_constructor(p_ctor)
                        if "artifact_readers" in data and isinstance(data["artifact_readers"], dict):
                            for cat, readers in data["artifact_readers"].items():
                                if isinstance(readers, list):
                                    for r in readers:
                                        parts = str(r).replace(":", ".").rsplit(".", 1)
                                        if len(parts) == 2:
                                            self.register_artifact_reader(cat, parts[0], parts[1])
                        if "cells" in data and isinstance(data["cells"], list):
                            for c in data["cells"]:
                                if isinstance(c, dict):
                                    if "type_vars" in c:
                                        self.register_type_vars(c["type_vars"])
                                    for p in (c.get("inputs") or {}).values() if isinstance(c.get("inputs"), dict) else []:
                                        role = p.get("port_role") or p.get("role")
                                        if role:
                                            self.register_role_carrier(role)
                        if "typestates" in data:
                            ts_block = data["typestates"]
                            if isinstance(ts_block, dict) and isinstance(ts_block.get("terminal_states"), list):
                                self.register_terminal_states(ts_block["terminal_states"])
                            states = ts_block.get("states", []) if isinstance(ts_block, dict) else (ts_block if isinstance(ts_block, list) else [])
                            dom_l = str(data.get("domain") or "").strip().lower() or None
                            for s_entry in states:
                                if isinstance(s_entry, dict) and "name" in s_entry:
                                    s_name = s_entry["name"]
                                    s_parent = s_entry.get("parent_state")
                                    s_carrier = s_entry.get("carrier_type")
                                    props = s_entry.get("properties") or {}
                                    self.register_state(s_name, s_parent, carrier_type=s_carrier, properties=props, domain=dom_l)
                        if "advisory_qualifiers" in data and isinstance(data["advisory_qualifiers"], list):
                            self.register_advisory_qualifiers(data["advisory_qualifiers"])
                        if "abstract_carriers" in data and isinstance(data["abstract_carriers"], list):
                            self.register_abstract_carriers(data["abstract_carriers"])
                        if "abstract_carrier_mapping" in data and isinstance(data["abstract_carrier_mapping"], dict):
                            self.register_abstract_carrier_mapping(data["abstract_carrier_mapping"])
                        if "data_bearing_roles" in data and isinstance(data["data_bearing_roles"], list):
                            self.register_data_bearing_roles(data["data_bearing_roles"])
                except Exception as e:
                    logger.debug("suppressed: %s", e, exc_info=False)

    # --- Cache invalidation -------------------------------------------------
    @contextlib.contextmanager
    def batch_mutations(self):
        """Defers cache invalidation until a batch of poset registrations finishes."""
        with self._lock:
            self._batch_depth += 1
        try:
            yield
        finally:
            should_flush = False
            with self._lock:
                self._batch_depth = max(0, self._batch_depth - 1)
                if self._batch_depth == 0 and self._batch_dirty:
                    self._batch_dirty = False
                    should_flush = True
            if should_flush:
                self._invalidate_caches()

    @_locked
    def _invalidate_caches(self) -> None:
        """Drop every derived cache. Bumping the generation makes any in-flight
        computation refuse to store its (possibly stale) result."""
        with self._lock:
            if getattr(self, "_batch_depth", 0) > 0:
                self._batch_dirty = True
                return
            self._generation += 1
            self._subtype_cache.clear()
            self._meet_cache.clear()
            self._join_cache.clear()
            self._ancestry_cache.clear()
            self._registered_types_cache = None
            self._composite_carrier_cache.clear()
            self._state_properties_cache.clear()
            self._states_by_property_cache.clear()
            self._state_carrier_cache.clear()
            self._state_compatible_cache.clear()
            if hasattr(self, "_state_raw_cache"):
                self._state_raw_cache.clear()
            self._terminal_sink_states_cache = None
            self._path_port_sig_cache.clear()
            self._unifies_cache.clear()
            self._top_cache.clear()
            self._top_state_cache.clear()
            self._type_ancestors_by_id = None
            self._cached_cell_init_roles = None
            self._cached_abstract_carriers = None
            self._cached_top_level_carriers = None
            self._cached_terminal_states = None
            self._cached_advisory_qualifiers = None
            self._cached_projection_tokens = None
            self._cached_edge_sem_bundle = None
            self._cached_bridge_cand_bundle = None
            self._cached_config_roles = None
            self._cached_hof_roles = None
            self._cached_terminal_roles = None
            self._cached_ingress_roles = None
            self._cached_bridge_roles = None
            self._cached_void_types = None
            self._cached_bottom_types = None
            self._cached_scalar_types = None
            self._cached_top_types_set = None
            self._cached_types_by_kind = {}
            self._cached_type_props = {}
            self._cached_types_with_prop.clear()
            self._cached_roles_for_cat.clear()
            self._cached_atomic_carrier_roots = None
            self._cached_composite_carrier_roots = None

    # --- Data-driven structural vocabulary ----------------------------------
    @_locked
    def register_top_level_carriers(self, names: Any) -> None:
        """Declare mutually-disjoint root carriers (distinct roots never subsume each other)."""
        mutated = False
        for n in names or ():
            s_ = str(n).strip().lower()
            if s_ and s_ not in self._top_level_carriers:
                self._top_level_carriers.add(s_)
                mutated = True
        if mutated:
            self._invalidate_caches()

    def get_top_level_carriers(self) -> Set[str]:
        cached = self._cached_top_level_carriers
        if cached is not None:
            return cached
        res = set(self._top_level_carriers)
        self._cached_top_level_carriers = res
        return res

    def get_bridge_candidate_bundle(self) -> Tuple[Set[str], Set[str], Set[str], Set[str], Set[str]]:
        cached = self._cached_bridge_cand_bundle
        if cached is not None:
            return cached
        gen = self._generation
        cf_domains = {"control_flow"} | set(self._control_flow_domains)
        cf_topologies = set(self.get_roles_for_category("combinator")) | set(self._combinator_topologies)
        ingress_stages = {"1"}
        terminal_stages = {"3"}
        cfg_roles = set(self.get_config_roles_set())
        bundle = (cf_domains, cf_topologies, ingress_stages, terminal_stages, cfg_roles)
        with self._lock:
            if gen == self._generation:
                self._cached_bridge_cand_bundle = bundle
        return bundle

    @_locked
    def register_state_role_ancestors(self, mapping: Dict[str, Any]) -> None:
        """role -> typestate ancestors whose descendants take that role (declaration order = priority)."""
        mutated = False
        for role, ancestors in (mapping or {}).items():
            r = str(role).strip().lower()
            if not r:
                continue
            anc = {str(a).strip().lower() for a in (ancestors or ()) if str(a).strip()}
            if anc:
                old = self._state_role_ancestors.get(r, set())
                new_val = old | anc
                if new_val != old:
                    self._state_role_ancestors[r] = new_val
                    mutated = True
        if mutated:
            self._invalidate_caches()

    def get_state_role_ancestors(self) -> Dict[str, Set[str]]:
        return {k: set(v) for k, v in self._state_role_ancestors.items()}

    @_locked
    def register_state_role_markers(self, mapping: Dict[str, Any]) -> None:
        """role -> substrings that, when present in a state name, assign that role."""
        mutated = False
        for role, markers in (mapping or {}).items():
            r = str(role).strip().lower()
            ms = tuple(str(m).strip().lower() for m in (markers or ()) if str(m).strip())
            if r and ms:
                old = self._state_role_markers.get(r, ())
                merged = tuple(dict.fromkeys(old + ms))
                if merged != old:
                    self._state_role_markers[r] = merged
                    mutated = True
        if mutated:
            self._invalidate_caches()

    def get_state_role_markers(self) -> Dict[str, Tuple[str, ...]]:
        return dict(self._state_role_markers)

    @_locked
    def register_carrier_roles(self, mapping: Dict[str, Any]) -> None:
        """role -> carrier types (poset ancestors) that imply that role."""
        mutated = False
        for role, carriers in (mapping or {}).items():
            r = str(role).strip().lower()
            cs = tuple(str(c).strip().lower() for c in (carriers or ()) if str(c).strip())
            if r and cs:
                old = self._carrier_roles.get(r, ())
                merged = tuple(dict.fromkeys(old + cs))
                if merged != old:
                    self._carrier_roles[r] = merged
                    mutated = True
        if mutated:
            self._invalidate_caches()

    def get_carrier_roles(self) -> Dict[str, Tuple[str, ...]]:
        return dict(self._carrier_roles)

    @_locked
    def register_identifier_root_states(self, states: Any) -> None:
        """Root typestates whose descendants are mutually compatible (asset-identifier families)."""
        mutated = False
        for st in states or ():
            s_ = str(st).strip().lower()
            if s_ and s_ not in self._identifier_root_states:
                self._identifier_root_states.add(s_)
                mutated = True
        if mutated:
            self._invalidate_caches()

    @_locked
    def register_egress_exclusion_tokens(self, tokens: Any) -> None:
        for t in tokens or ():
            s_ = str(t).strip().lower()
            if s_:
                self._egress_exclusion_tokens.add(s_)

    def get_egress_exclusion_tokens(self) -> Set[str]:
        return set(self._egress_exclusion_tokens)

    @_locked
    def register_type_vars(self, names: Any) -> None:
        mutated = False
        for n in names or ():
            n = str(n).strip()
            if n and n not in self._type_vars:
                self._type_vars.add(n)
                mutated = True
        if mutated:
            self._invalidate_caches()

    def is_type_variable(self, name: str) -> bool:
        n = str(name).strip()
        if n in self._type_vars:
            return True
        return len(n) > 0 and n[0].isupper() and n[0].isalpha() and (len(n) == 1 or n[1:].isdigit())

    @_locked
    def register_top(self, name: str) -> None:
        n = str(name).strip().lower()
        if n and n not in self._declared_top:
            self._declared_top.add(n)
            self._invalidate_caches()

    def is_declared_top(self, name: str) -> bool:
        if isinstance(name, str):
            hit = self._top_cache.get(name)
            if hit is not None:
                return hit
            s = name.strip()
        else:
            s = str(name).strip()
        gen = self._generation
        if s in _TOP_TYPE_FAST:
            res = True
        else:
            s_lower = s.lower()
            if s_lower in self._declared_top:
                res = True
            else:
                res = self.canonical_name(s).lower() in self._declared_top
        if isinstance(name, str):
            with self._lock:
                if gen == self._generation:
                    if len(self._top_cache) >= _SUBTYPE_CACHE_MAX:
                        self._top_cache.clear()
                    self._top_cache[name] = res
        return res

    def is_uninhabited_bottom(self, name: Optional[str]) -> bool:
        """True iff `name` denotes true uninhabited bottom (_|_: bottom/never)."""
        if name is None:
            return False
        s = name.strip().lower() if isinstance(name, str) else str(name).strip().lower()
        return s in _TRUE_BOTTOM_SYMBOLS or (s in self.get_types_by_kind("bottom") and s not in _VOID_UNIT_SYMBOLS)

    def is_declared_bottom(self, name: Optional[str]) -> bool:
        if name is None:
            return True
        s = name.strip().lower() if isinstance(name, str) else str(name).strip().lower()
        return s in self.get_bottom_types_set()

    def intern_symbol(self, name: Optional[str]) -> int:
        """Maps a canonicalized lowercase symbol string to an integer ID for O(1) equality and graph indexing."""
        if not name:
            return 0
        s = name.strip().lower() if isinstance(name, str) else str(name).strip().lower()
        if not s:
            return 0
        hit = self._sym_to_id.get(s)
        if hit is not None:
            return hit
        with self._lock:
            hit = self._sym_to_id.get(s)
            if hit is not None:
                return hit
            idx = len(self._id_to_sym)
            self._sym_to_id[s] = idx
            self._id_to_sym.append(s)
            return idx

    def _get_type_ancestors_by_id(self) -> Dict[int, Set[int]]:
        """Precomputes transitive supertype ID closure per generation for O(1) poset queries."""
        cached = self._type_ancestors_by_id
        if cached is not None:
            return cached
        gen = self._generation
        closure: Dict[int, Set[int]] = {}
        parents_snapshot = dict(self._parents)
        for t_name in list(parents_snapshot.keys()):
            t_id = self.intern_symbol(t_name)
            anc: Set[int] = {t_id}
            seen = {t_name}
            q = deque(parents_snapshot.get(t_name, ()))
            while q:
                par = q.popleft()
                pl = par.lower() if isinstance(par, str) else str(par).lower()
                if not pl or pl in seen:
                    continue
                seen.add(pl)
                anc.add(self.intern_symbol(pl))
                for gp in parents_snapshot.get(pl, ()):
                    if gp.lower() not in seen:
                        q.append(gp)
            closure[t_id] = anc
        with self._lock:
            if gen == self._generation:
                self._type_ancestors_by_id = closure
        return closure

    def get_bottom_types_set(self) -> Set[str]:
        cached = self._cached_bottom_types
        if cached is not None:
            return cached
        gen = self._generation
        res = set(_BOTTOM_LITERALS) | set(self.get_types_by_kind("bottom"))
        with self._lock:
            if gen == self._generation:
                self._cached_bottom_types = res
        return res

    def get_void_types_set(self) -> Set[str]:
        cached = self._cached_void_types
        if cached is not None:
            return cached
        gen = self._generation
        res = set(_BOTTOM_LITERALS) | set(self.get_types_by_kind("void")) | set(self.get_types_by_kind("bottom"))
        with self._lock:
            if gen == self._generation:
                self._cached_void_types = res
        return res

    def get_top_types_set(self) -> Set[str]:
        cached = self._cached_top_types_set
        if cached is not None:
            return cached
        gen = self._generation
        res = set(_TOP_TYPE_FAST) | set(self._declared_top) | set(self.get_types_by_kind("top"))
        with self._lock:
            if gen == self._generation:
                self._cached_top_types_set = res
        return res

    def get_config_roles_set(self) -> Set[str]:
        cached = self._cached_config_roles
        if cached is not None:
            return cached
        gen = self._generation
        res = (
            set(self.get_projection_tokens() or ())
            | self.get_roles_for_category("config")
            | self.get_roles_for_category("configuration")
            | self.get_roles_for_category("parameter")
            | self.get_roles_for_category("projection")
        )
        with self._lock:
            if gen == self._generation:
                self._cached_config_roles = res
        return res

    def get_hof_roles_set(self) -> Set[str]:
        cached = self._cached_hof_roles
        if cached is not None:
            return cached
        gen = self._generation
        res = self.get_roles_for_category("hof")
        with self._lock:
            if gen == self._generation:
                self._cached_hof_roles = res
        return res

    def get_terminal_roles_set(self) -> Set[str]:
        cached = self._cached_terminal_roles
        if cached is not None:
            return cached
        gen = self._generation
        res = (
            self.get_roles_for_category("terminal")
            | self.get_roles_for_category("sink")
            | self.get_roles_for_category("evaluator")
        )
        with self._lock:
            if gen == self._generation:
                self._cached_terminal_roles = res
        return res

    def get_ingress_roles_set(self) -> Set[str]:
        cached = self._cached_ingress_roles
        if cached is not None:
            return cached
        gen = self._generation
        res = (
            self.get_roles_for_category("ingress")
            | self.get_roles_for_category("source")
        )
        with self._lock:
            if gen == self._generation:
                self._cached_ingress_roles = res
        return res

    def get_bridge_roles_set(self) -> Set[str]:
        cached = self._cached_bridge_roles
        if cached is not None:
            return cached
        gen = self._generation
        res = self.get_roles_for_category("bridge")
        with self._lock:
            if gen == self._generation:
                self._cached_bridge_roles = res
        return res

    @_locked
    def is_builtin_scalar_primitive(self, type_name: Optional[str]) -> bool:
        """True iff `type_name` is a declared scalar/numeric/boolean/string primitive in the active tree poset."""
        t = str(type_name or "").strip().lower()
        if not t:
            return False
        return t in self.get_scalar_primitives_set()

    def get_scalar_primitives_set(self) -> Set[str]:
        cached = self._cached_scalar_types
        if cached is not None:
            return cached
        gen = self._generation
        res: Set[str] = {
            t for t, tk in self._type_kinds.items()
            if tk in SCALAR_PRIMITIVE_KINDS
        }
        res -= set(self.get_types_by_kind("path") or ())
        with self._lock:
            if gen == self._generation:
                self._cached_scalar_types = res
        return res

    def get_lifecycle_roles_set(self) -> Set[str]:
        res: Set[str] = set()
        for cat in ("lifecycle", "model", "estimator", "receiver", "model_input"):
            res |= self.get_roles_for_category(cat)
        return res

    def get_dataflow_roles_set(self) -> Set[str]:
        res: Set[str] = set(self.get_data_bearing_roles())
        for cat in ("data", "receiver", "strict", "data_input"):
            res |= self.get_roles_for_category(cat)
        return res

    def get_config_or_slot_roles_set(self) -> Set[str]:
        res: Set[str] = set(self.get_config_roles_set())
        for cat in ("literal", "key", "projection", "config", "parameter"):
            res |= self.get_roles_for_category(cat)
        return res

    def are_structure_kinds_compatible(
        self,
        out_sk: str,
        in_sk: str,
        out_abs: str = "",
        in_abs: str = "",
    ) -> bool:
        out_sk = str(out_sk or "").strip().lower()
        in_sk = str(in_sk or "").strip().lower()
        out_abs = str(out_abs or "").strip().lower()
        in_abs = str(in_abs or "").strip().lower()
        if out_sk == "handle" and in_sk and in_sk != "handle":
            return False
        if in_sk == "handle" and out_sk and out_sk != "handle":
            return False
        if out_sk == "path_spec" and in_sk and in_sk not in STRUCTURAL_PATH_COMPAT_KINDS:
            return False
        if in_sk == "path_spec" and out_sk and out_sk not in STRUCTURAL_PATH_COMPAT_KINDS:
            return False
        if out_sk or in_sk or out_abs or in_abs:
            scalar_abs = self.get_atomic_carrier_roots()
            is_out_compound = (out_sk in STRUCTURAL_COMPOUND_KINDS) or (
                not out_sk
                and bool(out_abs)
                and (self.is_composite_carrier(out_abs) or self.is_abstract_carrier(out_abs))
            )
            is_in_compound = (in_sk in STRUCTURAL_COMPOUND_KINDS) or (
                not in_sk
                and bool(in_abs)
                and (self.is_composite_carrier(in_abs) or self.is_abstract_carrier(in_abs))
            )
            is_out_scalar = (not is_out_compound) and (
                (out_sk == "scalar") or (not out_sk and out_abs in scalar_abs)
            )
            is_in_scalar = (not is_in_compound) and (
                (in_sk == "scalar") or (not in_sk and in_abs in scalar_abs)
            )
            if (is_out_scalar and is_in_compound) or (is_in_scalar and is_out_compound):
                return False
        return True

    def is_format_extension_compatible(self, ext: str, port: Any, check_concrete_prefix: bool = False) -> bool:
        """Checks whether a single path port is compatible with lowercase extension `ext`."""
        port_fmts = getattr(port, "supported_formats", None)
        if port_fmts:
            clean_fmts = {str(f).strip().lower() for f in port_fmts if str(f).strip()}
            if clean_fmts:
                return "*" in clean_fmts or "any" in clean_fmts or ext in clean_fmts
        tn = str(getattr(port, "type_name", "") or "").strip()
        if not ("[" in tn and tn.endswith("]")):
            return True
        ctor, _, inner = tn[:-1].partition("[")
        loc_ctors = set(self.get_types_by_kind("path") or ()) | {"path", "file"}
        if ctor.strip().lower() not in loc_ctors:
            return True
        parts = [part.strip().lower() for part in inner.split(",") if part.strip()]
        if not parts:
            return True
        fmt_groups = self.get_format_groups()
        default_ph = self._asset_placeholders.get("default", "")
        default_ext = default_ph.rpartition(".")[2].strip().lower() if "." in default_ph else ""

        def _match_fmt(target_fmt: str, allow_prefix: bool = False, allow_substr: bool = False) -> bool:
            if (
                ext == target_fmt
                or (allow_prefix and len(ext) >= 2 and target_fmt.startswith(ext))
                or (allow_substr and (target_fmt in ext or ext in target_fmt))
                or self.is_subtype(ext, target_fmt)
                or self.is_subtype(target_fmt, ext)
            ):
                return True
            for anc in fmt_groups:
                if self.is_subtype(ext, anc) and (
                    self.is_subtype(target_fmt, anc) or target_fmt == anc or (allow_substr and target_fmt in anc)
                ):
                    return True
            return False

        def _match_placeholder_ext(carrier_key: str) -> Optional[bool]:
            try:
                placeholder = self.get_asset_placeholder(carrier_key, "")
                if placeholder and "." in placeholder:
                    c_ext = placeholder.rpartition(".")[2].strip().lower()
                    if c_ext and not self.is_declared_top(c_ext) and (not default_ext or c_ext != default_ext):
                        return _match_fmt(c_ext, allow_prefix=False, allow_substr=False)
            except Exception:
                pass
            return None

        if len(parts) >= 2:
            fmt = parts[1]
            if not fmt or self.is_declared_top(fmt):
                res = _match_placeholder_ext(parts[0])
                return True if res is None else res
            if check_concrete_prefix and fmt not in fmt_groups:
                return _match_fmt(fmt, allow_prefix=True, allow_substr=False)
            if check_concrete_prefix and (self.is_subtype(ext, fmt) or ext == fmt):
                return True
            return _match_fmt(fmt, allow_prefix=False, allow_substr=(not check_concrete_prefix))
        target = parts[0]
        if not target or self.is_declared_top(target):
            return True
        if _match_fmt(target, allow_prefix=False, allow_substr=True):
            return True
        res = _match_placeholder_ext(target)
        return False if res is None else res

    def is_core_domain(self, domain: Optional[str]) -> bool:
        d = (domain or "").strip().lower()
        return not d or d in self._core_domains

    def is_config_role(self, role: Optional[str]) -> bool:
        return bool(role) and str(role).strip().lower() in self.get_config_roles_set()

    def is_hof_role(self, role: Optional[str]) -> bool:
        return bool(role) and str(role).strip().lower() in self.get_hof_roles_set()

    def is_terminal_role(self, role: Optional[str]) -> bool:
        return bool(role) and str(role).strip().lower() in self.get_terminal_roles_set()

    def is_ingress_role(self, role: Optional[str]) -> bool:
        return bool(role) and str(role).strip().lower() in self.get_ingress_roles_set()

    def is_bridge_role(self, role: Optional[str]) -> bool:
        return bool(role) and str(role).strip().lower() in self.get_bridge_roles_set()

    def is_void_type(self, type_name: Optional[str]) -> bool:
        if type_name is None:
            return True
        return str(type_name).strip().lower() in self.get_void_types_set()

    def is_scalar_primitive(self, type_name: Optional[str]) -> bool:
        return bool(type_name) and str(type_name).strip().lower() in self.get_scalar_primitives_set()

    @_locked
    def register_top_state(self, name: str) -> None:
        n = str(name).strip().lower()
        if n and n not in self._declared_top_states:
            self._declared_top_states.add(n)
            self._invalidate_caches()

    def is_declared_top_state(self, name: Optional[str]) -> bool:
        if name is None:
            return True
        if isinstance(name, str):
            hit = self._top_state_cache.get(name)
            if hit is not None:
                return hit
            sl = name.strip().lower()
        else:
            sl = str(name).strip().lower()
        gen = self._generation
        res = (
            sl in _TOP_STATE_LITERALS
            or sl in self._declared_top_states
            or sl in self.get_states_by_property("is_top", True)
        )
        if isinstance(name, str):
            with self._lock:
                if gen == self._generation:
                    if len(self._top_state_cache) >= _SUBTYPE_CACHE_MAX:
                        self._top_state_cache.clear()
                    self._top_state_cache[name] = res
        return res

    is_top_state = is_declared_top_state

    def get_declared_top_states(self) -> Set[str]:
        return (
            set(_TOP_STATE_LITERALS)
            | set(self._declared_top_states)
            | set(self.get_states_by_property("is_top", True))
        )

    @_locked
    def register_product_constructor(self, ctor: str) -> None:
        c = str(ctor).strip().lower()
        if c and c not in self._product_constructors:
            self._product_constructors.add(c)
            self._invalidate_caches()

    def is_product_constructor(self, ctor: str) -> bool:
        c = str(ctor).strip().lower()
        return c in self._product_constructors

    @_locked
    def derive_function_words(self, corpus_docs: List[str], cutoff: Optional[float] = None) -> Set[str]:
        if cutoff is None:
            cutoff = settings.function_words_cutoff
        from collections import Counter
        df = Counter()
        docs = [set(CellTokenizer.tokenize_prompt(d)) for d in corpus_docs if d]
        n = max(len(docs), 1)
        for toks in docs:
            df.update(toks)
        non_ast_types = {
            t for t in self.get_registered_types()
            if "ast" not in {str(p).lower() for p in self._parents.get(t, ())}
        }
        words = tuple(
            sorted(
                w for w, c in df.items()
                if c / n >= cutoff and len(w) >= 2 and w not in non_ast_types
            )
        )
        self._function_words = words
        return set(words)

    def get_function_words(self) -> Set[str]:
        """Return corpus-derived function words or empty set without static fallbacks."""
        fw = getattr(self, "_function_words", None)
        if fw:
            return set(fw)
        return set()

    def is_informative_token(self, token: str) -> bool:
        """Data-driven token informativeness based on corpus statistics; zero hardcoded lists."""
        t = str(token).strip().lower()
        if not t or len(t) < 2:
            return False
        fw = getattr(self, "_function_words", None)
        if fw and t in fw:
            return False
        return True

    @_locked
    def register_egress_tokens(self, tokens: Any) -> None:
        fw = getattr(self, "_function_words", None) or set()
        for t in tokens or ():
            s = str(t).strip().lower()
            if s and len(s) >= 2 and s not in fw:
                self._egress_tokens.add(s)

    def get_egress_tokens(self) -> Set[str]:
        return set(self._egress_tokens)

    @_locked
    def register_terminal_states(self, states: Any) -> None:
        if not hasattr(self, "_terminal_states"):
            self._terminal_states = set()
        mutated = False
        for st in states or ():
            s = str(st).strip().lower()
            if s and s not in self._terminal_states:
                self._terminal_states.add(s)
                mutated = True
        if mutated:
            self._invalidate_caches()

    def get_terminal_states(self) -> Set[str]:
        cached = self._cached_terminal_states
        if cached is not None:
            return cached
        res = set(getattr(self, "_terminal_states", set()))
        self._cached_terminal_states = res
        return res

    @_locked
    def register_polarity_hints(self, direction: str, hints: Any) -> None:
        if hints:
            d = str(direction).strip().lower()
            if d not in self._polarity_hints:
                self._polarity_hints[d] = set()
            for h in hints:
                self._polarity_hints[d].add(str(h).strip().lower())

    def get_polarity_hints(self, direction: str) -> Set[str]:
        return set(self._polarity_hints.get(str(direction).strip().lower(), set()))

    def get_declared_role_carriers(self) -> Set[str]:
        # Roles are declared on ports in trees; collect any that have been seen
        # during load. Until trees register them, return empty.
        return set(getattr(self, "_role_carriers", set()))

    @_locked
    def register_role_carrier(self, role: str) -> None:
        if not hasattr(self, "_role_carriers"):
            self._role_carriers = set()
        s = str(role or "").strip().lower()
        if s and s not in self._role_carriers:
            self._role_carriers.add(s)
            self._invalidate_caches()

    def get_all_aliases(self) -> Dict[str, str]:
        return dict(self._aliases)

    # --- Advisory qualifiers (data-driven from tree declarations) ---
    @_locked
    def register_advisory_qualifiers(self, qualifiers: Any) -> None:
        before = len(self._advisory_qualifiers)
        for q in qualifiers or ():
            if isinstance(q, (list, tuple)):
                tup = tuple(str(x).strip() for x in q)
                self._advisory_qualifiers.add(tup)
                if len(tup) == 1:
                    self._advisory_qualifiers.add(tup[0])
            elif isinstance(q, str):
                self._advisory_qualifiers.add((q.strip(),))
                self._advisory_qualifiers.add(q.strip())
        if len(self._advisory_qualifiers) != before:
            self._invalidate_caches()

    def get_advisory_qualifiers(self) -> Set[Any]:
        cached = self._cached_advisory_qualifiers
        if cached is not None:
            return cached
        res = set(self._advisory_qualifiers)
        self._cached_advisory_qualifiers = res
        return res

    def has_conflicting_qualifiers(self, producer_quals: Any, consumer_quals: Any) -> bool:
        """True if producer and consumer declare mutually exclusive qualifiers from the same declared group."""
        if not producer_quals or not consumer_quals:
            return False
        groups = self._exclusive_qualifiers
        if not groups:
            return False
        p_flat = {
            str(x).strip().lower()
            for q in producer_quals
            for x in (q if isinstance(q, (tuple, list)) else (q,))
            if str(x).strip()
        }
        c_flat = {
            str(x).strip().lower()
            for q in consumer_quals
            for x in (q if isinstance(q, (tuple, list)) else (q,))
            if str(x).strip()
        }
        if not p_flat or not c_flat:
            return False
        for grp in groups:
            g_set = {
                str(x).strip().lower()
                for x in (grp if isinstance(grp, (tuple, list)) else (grp,))
                if str(x).strip()
            }
            p_hits = p_flat & g_set
            c_hits = c_flat & g_set
            if p_hits and c_hits and not (c_hits & p_hits):
                return True
        return False

    # --- Abstract carriers (data-driven from tree declarations) ---
    @_locked
    def register_abstract_carriers(self, carriers: Any) -> None:
        mutated = False
        for c in carriers or ():
            s = str(c).strip().lower()
            if s and s not in self._abstract_carriers:
                self._abstract_carriers.add(s)
                mutated = True
        if mutated:
            self._invalidate_caches()

    def get_abstract_carriers(self) -> Set[str]:
        cached = self._cached_abstract_carriers
        if cached is not None:
            return cached
        res = set(self._abstract_carriers)
        self._cached_abstract_carriers = res
        return res

    def is_abstract_carrier(self, name: str) -> bool:
        return str(name).strip().lower() in self._abstract_carriers

    def is_carrier(self, name: str) -> bool:
        """Determines if a type or abstract type represents a carrier."""
        if not name:
            return False
        s = str(name).strip().lower()
        return (
            self.is_abstract_carrier(s)
            or self.is_composite_carrier(s)
            or s in self.get_top_level_carriers()
            or bool(self.get_abstract_carrier(s))
        )

    def is_composite_carrier(self, name: str) -> bool:
        """Determines if a type represents a composite, stateful, or non-scalar carrier."""
        if not name:
            return False
        raw_key = str(name).strip()
        if not raw_key:
            return False
        hit = self._composite_carrier_cache.get(raw_key)
        if hit is not None:
            return hit
        gen = self._generation
        s = raw_key.lower()
        universal_top = self.get_top_types_set()
        if (
            s in universal_top
            or self.is_declared_top(s)
            or self.is_declared_bottom(s)
            or self.is_type_variable(raw_key)
        ):
            res = False
        else:
            atomic_roots = self.get_atomic_carrier_roots()
            if any(self.is_subtype(s, root) for root in atomic_roots):
                res = False
            else:
                composite_roots = self.get_composite_carrier_roots()
                if any(self.is_subtype(s, root) for root in composite_roots):
                    res = True
                elif any(
                    not any(self.is_subtype(ac, root) for root in atomic_roots) and self.is_subtype(s, ac)
                    for ac in self.get_abstract_carriers()
                ):
                    res = True
                elif any(
                    not any(self.is_subtype(tc, root) for root in atomic_roots) and self.is_subtype(s, tc)
                    for tc in self._top_level_carriers
                ):
                    res = True
                elif s not in self.get_registered_types():
                    res = False
                else:
                    res = True
        with self._lock:
            if gen == self._generation:
                if len(self._composite_carrier_cache) >= _SUBTYPE_CACHE_MAX:
                    self._composite_carrier_cache.clear()
                self._composite_carrier_cache[raw_key] = res
        return res

    # --- Abstract carrier mapping (type name -> abstract category) ---
    @_locked
    def register_abstract_carrier_mapping(self, mapping: Dict[str, str]) -> None:
        mutated = False
        for k, v in (mapping or {}).items():
            kl = str(k).strip().lower()
            vl = str(v).strip().lower()
            if self._abstract_carrier_mapping.get(kl) != vl:
                self._abstract_carrier_mapping[kl] = vl
                mutated = True
        if mutated:
            self._invalidate_caches()

    def get_registered_types(self) -> Set[str]:
        """Returns all dynamically registered type names, aliases, typestates, and abstract carriers."""
        cached = self._registered_types_cache
        if cached is not None:
            return cached
        gen = self._generation
        res = (
            {k.lower() for k in self._parents.keys()}
            | {k.lower() for k in self._aliases.keys()}
            | {k.lower() for k in self._abstract_carriers}
            | {k.lower() for k in self._state_parents.keys()}
            | {k.lower() for k in self._declared_top}
            | {k.lower() for k in self._declared_top_states}
        )
        with self._lock:
            if gen == self._generation:
                self._registered_types_cache = res
        return res

    def get_abstract_carrier(self, type_name: str) -> Optional[str]:
        """Returns the abstract carrier category for a type name, or None."""
        key = str(type_name).strip().lower()
        result = self._abstract_carrier_mapping.get(key)
        if result:
            return result
        # Composite names ("dataframe_cleaned", "sequence of int"): match on any declared
        # mapping key at a word/underscore boundary, longest key first. No built-in vocabulary.
        mapping = dict(self._abstract_carrier_mapping)
        for prefix in sorted(mapping, key=len, reverse=True):
            if prefix and key.startswith(prefix):
                rest = key[len(prefix):]
                if not rest or not rest[0].isalnum():
                    return mapping[prefix]
        return None

    def is_literal_groundable(self, type_name: str, has_enum: bool = False) -> bool:
        """Domain-agnostic check whether a carrier type can be synthesized as a prompt literal expression."""
        if not type_name:
            return False
        t = str(type_name).strip().lower()
        if self.is_declared_top(t) or has_enum:
            return True
        if self.is_builtin_scalar_primitive(t) or self.has_type_kind(t, "path"):
            return True
        literal_roots = self.get_types_with_property("literal_compatible", True)
        return any(self.is_subtype(t, root) for root in literal_roots)

    # --- Data-bearing roles (data-driven from tree declarations) ---
    @_locked
    def register_data_bearing_roles(self, roles: Any) -> None:
        mutated = False
        for r in roles or ():
            s = str(r).strip().lower()
            if s and s not in self._data_bearing_roles:
                self._data_bearing_roles.add(s)
                mutated = True
        if mutated:
            self._invalidate_caches()

    def get_data_bearing_roles(self) -> Set[str]:
        return set(self._data_bearing_roles)

    # --- Estimator verbs (data-driven from tree declarations) ---
    @_locked
    def register_estimator_verbs(self, verbs: Any) -> None:
        if verbs:
            for v in verbs:
                self._estimator_verbs.add(str(v).strip().lower())

    def get_estimator_verbs(self) -> Set[str]:
        return set(self._estimator_verbs)

    # --- Projection tokens (data-driven from tree declarations) ---
    @_locked
    def register_projection_tokens(self, tokens: Any) -> None:
        if tokens:
            mutated = False
            for t in tokens:
                s = str(t).strip().lower()
                if s not in self._projection_tokens:
                    self._projection_tokens.add(s)
                    mutated = True
            if mutated:
                self._invalidate_caches()

    def get_projection_tokens(self) -> Set[str]:
        cached = self._cached_projection_tokens
        if cached is not None:
            return cached
        gen = self._generation
        res = set(self._projection_tokens) | self.get_roles_for_category("projection")
        with self._lock:
            if gen == self._generation:
                self._cached_projection_tokens = res
        return res

    register_column_projection_tokens = register_projection_tokens
    get_column_projection_tokens = get_projection_tokens

    def get_format_groups(self) -> Set[str]:
        """Returns registered format categories (subtypes of 'format' registered in poset)."""
        cached = getattr(self, "_cached_format_groups", None)
        if cached is not None and cached[0] == self._generation:
            return cached[1]
        groups = {
            t for t in self.get_registered_types()
            if t != "format" and self.is_subtype(t, "format") and (
                "format" in self._parents.get(t, ())
                or any(t in parents for parents in self._parents.values())
                or self.has_type_kind(t, "format_group")
                or bool(self._type_properties.get(t, {}).get("is_format_group"))
            )
        }
        self._cached_format_groups = (self._generation, groups)
        return groups

    # --- Preposition triggers (data-driven from tree declarations) ---
    @_locked
    def register_preposition_triggers(self, triggers: Any) -> None:
        if triggers:
            for t in triggers:
                self._preposition_triggers.add(str(t).strip().lower())

    def get_preposition_triggers(self) -> Set[str]:
        return set(self._preposition_triggers)

    # --- Generic prepositions (data-driven from tree declarations) ---
    @_locked
    def register_generic_prepositions(self, preps: Any) -> None:
        if preps:
            for p in preps:
                self._generic_prepositions.add(str(p).strip().lower())

    def get_generic_prepositions(self) -> Set[str]:
        return set(self._generic_prepositions)

    # --- Coordinating conjunctions (data-driven from tree declarations) ---
    @_locked
    def register_coordinating_conjunctions(self, words: Any) -> None:
        if words:
            for w in words:
                self._coordinating_conjunctions.add(str(w).strip().lower())

    def get_coordinating_conjunctions(self) -> Set[str]:
        return set(self._coordinating_conjunctions)

    # --- Sequencing connectives (data-driven from tree declarations) ---
    @_locked
    def register_sequencing_connectives(self, words: Any) -> None:
        if words:
            for w in words:
                self._sequencing_connectives.add(str(w).strip().lower())

    def get_sequencing_connectives(self) -> Set[str]:
        return set(self._sequencing_connectives)

    # --- Filter-context tokens (data-driven from tree declarations) ---
    @_locked
    def register_filter_context_tokens(self, words: Any) -> None:
        if words:
            for w in words:
                self._filter_context_tokens.add(str(w).strip().lower())

    def get_filter_context_tokens(self) -> Set[str]:
        return set(self._filter_context_tokens)

    @_locked
    def register_prompt_filler_tokens(self, words: Any) -> None:
        if words:
            for w in words:
                self._prompt_filler_tokens.add(str(w).strip().lower())

    def get_prompt_filler_tokens(self) -> Set[str]:
        return set(self._prompt_filler_tokens)

    def _get_edge_sem_bundle(self) -> Tuple[Set[Any], ...]:
        """Cached 5-tuple of schema-derived role/state sets used on every edge check."""
        cached = self._cached_edge_sem_bundle
        if cached is not None:
            return cached
        gen = self._generation
        bundle = (
            self.get_roles_for_category("prediction_in"),
            self.get_roles_for_category("prediction_out"),
            self.get_states_by_property("is_prediction", True),
            self.get_states_by_property("is_invalid", True) | self.get_states_by_property("is_terminal_invalid", True),
            {str(x).strip().lower() for x in self.get_type_property_values("protocols")},
        )
        with self._lock:
            if gen == self._generation:
                self._cached_edge_sem_bundle = bundle
        return bundle

    def _get_cell_init_role_bundle(self) -> Tuple[Any, ...]:
        """Cached bundle of schema-derived role sets read by Cell.__init__."""
        cached = self._cached_cell_init_roles
        if cached is not None:
            return cached
        gen = self._generation
        key_roles = self.get_roles_for_category("key") | self.get_roles_for_category("projection")
        key_port_names = set(self.get_projection_tokens())
        val_roles = self.get_roles_for_category("assigned_value")
        val_port_names = self.get_roles_for_category("assigned_value")
        single_proj_kinds = self.get_roles_for_category("single_projection")
        multi_proj_kinds = self.get_roles_for_category("multi_projection")
        feature_roles = self.get_roles_for_category("feature")
        target_roles = self.get_roles_for_category("target")
        paired_ranks = self._capabilities.get("paired_morphism_ranks") or {}
        bundle = (
            key_roles, key_port_names, val_roles, val_port_names,
            single_proj_kinds, multi_proj_kinds, feature_roles, target_roles,
            paired_ranks,
        )
        with self._lock:
            if gen == self._generation:
                self._cached_cell_init_roles = bundle
        return bundle

    def get_slot_aliases(self, role: str) -> List[str]:
        """Returns registered slot aliases for an operational role from carrier_roles."""
        k = str(role).strip().lower()
        return sorted(self.get_roles_for_category(k))

    # --- Operation / algorithm tokens (data-driven from tree cell declarations) ---
    @_locked
    def register_operation_tokens(self, tokens: Any) -> None:
        for t in tokens or ():
            s = str(t).strip().lower()
            if s and len(s) >= 2:
                self._operation_tokens.add(s)

    def get_operation_tokens(self) -> Set[str]:
        return set(self._operation_tokens)

    def is_operation_token(self, token: str) -> bool:
        return str(token).strip().lower() in self._operation_tokens

    # --- Default asset placeholders (data-driven from tree declarations) ---
    @_locked
    def register_asset_placeholders(self, placeholders: Dict[str, str]) -> None:
        for k, v in (placeholders or {}).items():
            self._asset_placeholders[str(k).strip().lower()] = str(v).strip()

    def get_asset_placeholder(self, modality: str, default: Optional[str] = None) -> Optional[str]:
        k = str(modality).strip().lower()
        if k in self._asset_placeholders:
            return self._asset_placeholders[k]
        return self._asset_placeholders.get("default", default)

    @_locked
    def register_output_asset_placeholders(self, placeholders: Dict[str, str]) -> None:
        for k, v in (placeholders or {}).items():
            self._output_asset_placeholders[str(k).strip().lower()] = str(v).strip()

    def get_output_asset_placeholder(self, modality: str, default: Optional[str] = None) -> Optional[str]:
        k = str(modality).strip().lower()
        if k in self._output_asset_placeholders:
            return self._output_asset_placeholders[k]
        return self._output_asset_placeholders.get("default", default)

    @_locked
    def register_supported_formats(self, formats: Any) -> None:
        if not formats:
            return
        if isinstance(formats, (str, bytes)):
            formats = [formats]
        mutated = False
        for f in formats:
            clean = str(f).strip().lower().lstrip(".")
            if clean and clean != "*" and clean not in self._registered_supported_formats:
                self._registered_supported_formats.add(clean)
                mutated = True
        if mutated:
            self._invalidate_caches()

    def get_registered_supported_formats(self) -> Set[str]:
        if not self._registered_supported_formats:
            try:
                orch = LatticeOrchestrator.get_active_instance()
                if orch:
                    for c in getattr(orch, "cells", []) or ():
                        for p in list(getattr(c, "inputs", {}).values()) + list(getattr(c, "outputs", {}).values()):
                            if getattr(p, "supported_formats", None):
                                self.register_supported_formats(p.supported_formats)
            except Exception:
                pass
        return set(self._registered_supported_formats)

    @classmethod
    def get_instance(cls) -> TypeRegistry:
        inst = cls._instance
        if inst is not None:
            return inst
        with cls._lock:
            if cls._instance is None:
                inst = cls.__new__(cls)
                cls._instance = inst
                inst.__init__()
            return cls._instance

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    @property
    def poset(self) -> Dict[str, Set[str]]:
        return self._parents

    def _would_create_poset_cycle(self, name: str, super_name: str) -> bool:
        """Checks whether adding edge name -> super_name violates Top/Bottom or creates a poset cycle in O(deg * depth)."""
        bottom_set = set(_BOTTOM_LITERALS) | {t for t, tk in self._type_kinds.items() if tk == "bottom"}
        uninhabited_bottom = {t for t in bottom_set if t not in _VOID_UNIT_SYMBOLS} | set(_TRUE_BOTTOM_SYMBOLS)
        top_set = set(_TOP_TYPE_FAST) | set(self._declared_top) | {t for t, tk in self._type_kinds.items() if tk == "top"}
        parents = self._parents

        def _is_top_sym(t: str) -> bool:
            tl = str(t or "").strip().lower()
            if not tl:
                return False
            if tl in top_set:
                return True
            return self.canonical_name(tl).lower() in top_set

        def _live_atomic_is_subtype(sub: str, sup: str) -> bool:
            sub_l = str(sub or "").strip().lower()
            sup_l = str(sup or "").strip().lower()
            if not sub_l or not sup_l:
                return False
            sub_c = self.canonical_name(sub_l).lower()
            sup_c = self.canonical_name(sup_l).lower()

            if sub_l in uninhabited_bottom or sub_c in uninhabited_bottom:
                return True
            if sup_l in uninhabited_bottom or sup_c in uninhabited_bottom:
                return False

            if (
                sub_l in bottom_set or sub_c in bottom_set
                or sup_l in bottom_set or sup_c in bottom_set
            ):
                return sub_c == sup_c or (
                    sub_c in _VOID_UNIT_SYMBOLS and sup_c in _VOID_UNIT_SYMBOLS
                )

            if sup_c in top_set or sup_l in top_set:
                return True
            if sub_c in top_set or sub_l in top_set:
                return False

            if "_or_" in sub_c:
                parts = [p for p in sub_c.split("_or_") if p]
                if parts and all(_live_atomic_is_subtype(part, sup_c) for part in parts):
                    return True
            if "_or_" in sup_c:
                parts = [p for p in sup_c.split("_or_") if p]
                if parts and any(_live_atomic_is_subtype(sub_c, part) for part in parts):
                    return True

            sup_targets = {sup_l, sup_c}
            if "." in sup_l:
                sup_targets.add(sup_l.rsplit(".", 1)[-1])
            if "." in sup_c:
                sup_targets.add(sup_c.rsplit(".", 1)[-1])
            if sub_l in sup_targets or sub_c in sup_targets:
                return True

            visited: Set[str] = set()
            queue = deque([sub_c, sub_l])
            if "." in sub_c and sub_c not in parents:
                queue.append(sub_c.rsplit(".", 1)[-1])
            if "." in sub_l and sub_l not in parents:
                queue.append(sub_l.rsplit(".", 1)[-1])

            while queue:
                cur = queue.popleft()
                if cur in visited:
                    continue
                visited.add(cur)
                for p in parents.get(cur, ()):
                    if p in sup_targets:
                        return True
                    if p not in visited:
                        queue.append(p)
            return False

        if "[" in name or "[" in super_name or "{" in name or "{" in super_name:
            if _live_atomic_is_subtype(super_name, name):
                return True
            return LatticeType.from_str(super_name).is_subtype_of(
                LatticeType.from_str(name),
                _live_atomic_is_subtype,
                is_top_fn=_is_top_sym,
            )
        return _live_atomic_is_subtype(super_name, name)

    @_locked
    def register_type(self, type_name: str, super_type: Optional[str] = None):
        """Registers a type and optionally declares its supertype in the poset."""
        name = str(type_name).strip().lower()
        if not name or name in ("none", "null"):
            return
        mutated = False
        try:
            if name not in self._parents:
                self._parents[name] = set()
                mutated = True
            if super_type:
                super_name = str(super_type).strip().lower()
                if super_name and super_name != name and super_name not in ("none", "null"):
                    # Poset invariant: declared distinct top-level carriers never subsume each other
                    if name in self._top_level_carriers and super_name in self._top_level_carriers:
                        return
                    # Prevent cycles in the poset without triggering O(N^2) full-closure rebuilds mid-mutation
                    if self._would_create_poset_cycle(name, super_name):
                        return
                    if super_name not in self._parents:
                        self._parents[super_name] = set()
                        mutated = True
                    if super_name not in self._parents[name]:
                        self._parents[name].add(super_name)
                        mutated = True
        finally:
            if mutated:
                self._invalidate_caches()

    # ---------------------------------------------------------------------
    # Data-driven extension registries (all populated from trees/*.json)
    # ---------------------------------------------------------------------

    @_locked
    def register_semantic_flag_patterns(self, patterns: Any) -> None:
        """Registers declared keyword->flag observation patterns (data, not code)."""
        for p in patterns or ():
            if isinstance(p, dict) and p.get("keywords") and p.get("flag"):
                self._semantic_flag_patterns.append({
                    "keywords": set(str(k).lower() for k in p["keywords"]),
                    "flag": str(p["flag"]),
                    "value": p.get("value", True),
                    "extras": dict(p.get("extras") or {}),
                })

    def get_semantic_flag_patterns(self) -> List[Dict[str, Any]]:
        return list(self._semantic_flag_patterns)

    @_locked
    def register_port_name_type_hints(self, hints: Any) -> None:
        """Registers declared exact port-name -> type_name hints (data, not code)."""
        if isinstance(hints, dict):
            for name, t_name in hints.items():
                n = str(name).strip().lower()
                if n and t_name:
                    self._port_name_type_hints[n] = str(t_name)

    @_locked
    def register_port_name_morphology_hints(self, hints: Any) -> None:
        """Registers declared affix (prefix/suffix) -> type_name hints."""
        for h in hints or ():
            if isinstance(h, dict) and h.get("token") and h.get("type_name"):
                self._port_name_morphology_hints.append({
                    "affix": str(h.get("affix", "suffix")).lower(),
                    "token": str(h["token"]).lower(),
                    "type_name": str(h["type_name"]),
                })

    def lookup_port_name_type(self, param_name: str) -> Optional[str]:
        """Resolves a port name against DECLARED hints from trees. Returns None
        when no tree declared anything for this name — callers then fall back to
        the structural default ("any"), never to engine-side guessing."""
        name = str(param_name or "").strip().lower()
        if not name:
            return None
        exact = self._port_name_type_hints.get(name)
        if exact:
            return exact
        for hint in self._port_name_morphology_hints:
            token = hint["token"]
            if hint["affix"] == "prefix" and name.startswith(token) and len(name) > len(token):
                return hint["type_name"]
            if hint["affix"] == "suffix" and name.endswith(token) and len(name) > len(token):
                return hint["type_name"]
        return None

    def is_known_type_name(self, name: str) -> bool:
        """
        Dynamically checks if a type name is known in the Poset ontology,
        carrier registries, or Python runtime built-ins.
        Zero hardcoded type string sets.
        """
        if not name:
            return False
        n_low = str(name).strip().lower()
        if n_low in self._parents or n_low in self._aliases:
            return True
        if n_low in self._top_level_carriers or n_low in self._state_carriers:
            return True
        if hasattr(self, "_abstract_carriers") and n_low in self._abstract_carriers:
            return True
        if hasattr(self, "_role_carriers") and n_low in self._role_carriers:
            return True
        # Dynamic Python built-in type reflection
        import builtins
        builtin_val = getattr(builtins, n_low, None) or getattr(builtins, str(name).strip(), None)
        if builtin_val is not None and isinstance(builtin_val, type):
            return True
        return False

    is_registered = is_known_type_name

    @_locked
    def register_alias(self, alias: str, canonical: str):
        k = alias.strip().lower()
        v = canonical.strip()
        if self._aliases.get(k) == v:
            return
        self._aliases[k] = v
        self._invalidate_caches()  # canonical_name() feeds every subtype query
        # (apply_fixes_v7) propagate to tokenizer
        try:
            from .tokenizer import register_aliases as _ra
        except ImportError:
            try: from tokenizer import register_aliases as _ra
            except ImportError: _ra = None
        if _ra is not None:
            try: _ra(dict(self._aliases))  # snapshot, never the live dict
            except Exception: pass

    def canonical_name(self, type_name: str) -> str:
        if not type_name:
            return "any"
        clean = str(type_name).strip()
        return self._aliases.get(clean.lower(), clean)

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def _resolve_runtime_class(type_name: str) -> Optional[type]:
        """Dynamically locates a runtime class via Python introspection without hardcoding."""
        if not type_name or not isinstance(type_name, str):
            return None
        import builtins
        import numbers
        import warnings
        clean = type_name.strip()
        clean_lower = clean.lower()
        if clean_lower in ("numeric", "number"):
            return numbers.Number
        if hasattr(builtins, clean):
            obj = getattr(builtins, clean)
            if isinstance(obj, type):
                return obj
        if hasattr(builtins, clean_lower):
            obj = getattr(builtins, clean_lower)
            if isinstance(obj, type):
                return obj
        if "." in clean:
            parts = clean.rsplit(".", 1)
            mod_name, cls_name = parts[0], parts[1]
            mod = sys.modules.get(mod_name)
            if mod and hasattr(mod, cls_name):
                obj = getattr(mod, cls_name)
                if isinstance(obj, type):
                    return obj

        # For unqualified type names, restrict lookup to top-level canonical root modules
        # and verify the class actually belongs to the declaring package namespace.
        # Lowercase builtins and qualified dotted paths are already handled above;
        # unqualified symbolic state/role/tag tokens start lowercase and are never
        # Python runtime classes in top-level modules.
        if clean and clean[0].isupper():
            known_roots = [
                mod for name, mod in list(sys.modules.items())
                if mod and "." not in name and not name.startswith("_")
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for mod in known_roots:
                    try:
                        if hasattr(mod, clean):
                            obj = getattr(mod, clean)
                            if isinstance(obj, type) and getattr(obj, "__module__", "").split(".")[0] == getattr(mod, "__name__", ""):
                                return obj
                    except Exception:
                        continue
        return None

    def generation(self) -> int:
        """Public read of the poset mutation counter, for callers that memoize
        their own derived predicates and need to invalidate them in step with
        the registry (mirrors the guard used by ``is_subtype``'s own cache)."""
        return self._generation

    def is_subtype(self, sub: str, super_: str) -> bool:
        """Memoised poset query. The cache is per-instance and generation-guarded:
        a result computed while a mutation was in flight is returned but never stored."""
        key = (sub, super_)
        hit = self._subtype_cache.get(key)
        if hit is not None:
            return hit
        gen = self._generation
        result = self._compute_is_subtype(sub, super_)
        with self._lock:
            if gen == self._generation:
                if len(self._subtype_cache) >= _SUBTYPE_CACHE_MAX:
                    self._subtype_cache.clear()
                self._subtype_cache[key] = result
        return result

    def _collect_all_supertypes(self, typ: str) -> Set[str]:
        canon = self.canonical_name(typ).strip().lower()
        if not canon:
            return set()
        out: Set[str] = {canon}
        canon_key = canon.rsplit(".", 1)[-1] if canon not in self._parents and "." in canon else canon
        out.add(canon_key)
        q: deque[str] = deque([canon, canon_key])
        seen: Set[str] = set()
        while q:
            curr = q.popleft()
            if not curr or curr in seen:
                continue
            seen.add(curr)
            out.add(curr)
            for par in self._parents.get(curr, ()):
                pl = str(par).strip().lower()
                if pl and pl not in seen:
                    out.add(pl)
                    q.append(pl)
        return out

    def join_types(self, t1: str, t2: str) -> str:
        """Least upper bound (join: t1 |_| t2) over the dynamic type poset."""
        a = str(t1 or "").strip()
        b = str(t2 or "").strip()
        if not a:
            return b.lower() if b else "any"
        if not b:
            return a.lower()
        hit = self._join_cache.get((a, b))
        if hit is not None:
            return hit
        gen = self._generation

        if self.is_uninhabited_bottom(a):
            res = b.lower() if ("[" not in b and "{" not in b) else str(LatticeType.from_str(b))
        elif self.is_uninhabited_bottom(b):
            res = a.lower() if ("[" not in a and "{" not in a) else str(LatticeType.from_str(a))
        elif a.lower() in _VOID_UNIT_SYMBOLS and b.lower() in _VOID_UNIT_SYMBOLS:
            res = self.canonical_name(a).lower()
        elif a.lower() in self.get_bottom_types_set() or b.lower() in self.get_bottom_types_set():
            res = "any"
        elif self.is_declared_top(a) or self.is_declared_top(b):
            res = "any"
        elif "[" in a or "[" in b or "{" in a or "{" in b:
            res = str(LatticeType.from_str(a).join(LatticeType.from_str(b), self))
        elif self.is_subtype(a, b) and self.is_subtype(b, a):
            ca, cb = self.canonical_name(a).lower(), self.canonical_name(b).lower()
            res = min(ca, cb)
        elif self.is_subtype(a, b):
            res = self.canonical_name(b).lower()
        elif self.is_subtype(b, a):
            res = self.canonical_name(a).lower()
        else:
            top_exclude = self.get_top_types_set() | self.get_void_types_set() | {"object"}
            common = (self._collect_all_supertypes(a) & self._collect_all_supertypes(b)) - top_exclude
            if not common:
                res = "any"
            else:
                minimal = [
                    x for x in common
                    if not any(y != x and self.is_subtype(y, x) for y in common)
                ]
                pool = minimal if minimal else list(common)
                # Prefer the candidate with the largest ancestor chain (deepest/most specific), then lexical
                res = sorted(pool, key=lambda x: (-len(self._collect_all_supertypes(x)), x))[0]

        with self._lock:
            if gen == self._generation:
                if len(self._join_cache) >= _SUBTYPE_CACHE_MAX:
                    self._join_cache.clear()
                self._join_cache[(a, b)] = res
                self._join_cache[(b, a)] = res
        return res

    def meet_types(self, t1: str, t2: str) -> str:
        """Greatest lower bound (meet: t1 |\\ t2) over the dynamic type poset."""
        a = str(t1 or "").strip()
        b = str(t2 or "").strip()
        if not a or not b:
            return "bottom"
        hit = self._meet_cache.get((a, b))
        if hit is not None:
            return hit
        gen = self._generation

        if self.is_uninhabited_bottom(a) or self.is_uninhabited_bottom(b):
            res = "bottom"
        elif a.lower() in _VOID_UNIT_SYMBOLS and b.lower() in _VOID_UNIT_SYMBOLS:
            res = self.canonical_name(a).lower()
        elif a.lower() in self.get_bottom_types_set() or b.lower() in self.get_bottom_types_set():
            res = "bottom"
        elif self.is_declared_top(a):
            res = self.canonical_name(b).lower() if ("[" not in b and "{" not in b) else str(LatticeType.from_str(b))
        elif self.is_declared_top(b):
            res = self.canonical_name(a).lower() if ("[" not in a and "{" not in a) else str(LatticeType.from_str(a))
        elif "[" in a or "[" in b or "{" in a or "{" in b:
            res = str(LatticeType.from_str(a).meet(LatticeType.from_str(b), self))
        elif self.is_subtype(a, b) and self.is_subtype(b, a):
            ca, cb = self.canonical_name(a).lower(), self.canonical_name(b).lower()
            res = min(ca, cb)
        elif self.is_subtype(a, b):
            res = self.canonical_name(a).lower()
        elif self.is_subtype(b, a):
            res = self.canonical_name(b).lower()
        else:
            excluded = self.get_void_types_set() | self.get_top_types_set() | set(_TRUE_BOTTOM_SYMBOLS) | {"object"}
            common_lb = [
                x for x in self._parents.keys()
                if x not in excluded and self.is_subtype(x, a) and self.is_subtype(x, b)
            ]
            if not common_lb:
                res = "bottom"
            else:
                maximal = [
                    x for x in common_lb
                    if not any(y != x and self.is_subtype(x, y) for y in common_lb)
                ]
                if len(maximal) == 1:
                    res = maximal[0]
                elif maximal:
                    # Prefer shallowest (closest to join / fewest ancestors -> most general common lower bound), then lexical
                    res = sorted(maximal, key=lambda x: (len(self._collect_all_supertypes(x)), x))[0]
                else:
                    res = "bottom"

        with self._lock:
            if gen == self._generation:
                if len(self._meet_cache) >= _SUBTYPE_CACHE_MAX:
                    self._meet_cache.clear()
                self._meet_cache[(a, b)] = res
                self._meet_cache[(b, a)] = res
        return res

    def _compute_is_subtype(self, sub: str, super_: str) -> bool:
        """
        Computes poset partial order: returns True iff sub <= super_.
        Supports recursive structural generic subtyping for terms F[tau_1, ..., tau_n].
        Wildcards ('any', 'object', '*', 'top') are Top types that subsume all types.
        Uses formal MRO and poset reachability dynamically.
        """
        if not sub or not super_:
            return False
        sub_str = str(sub).strip()
        super_str = str(super_).strip()

        sub_low = sub_str.lower()
        super_low = super_str.lower()
        if sub_low in _TRUE_BOTTOM_SYMBOLS or self.is_uninhabited_bottom(sub_low):
            return True
        if super_low in _TRUE_BOTTOM_SYMBOLS or self.is_uninhabited_bottom(super_low):
            return False

        bottom_set = self.get_bottom_types_set()
        if sub_low in bottom_set or super_low in bottom_set:
            return sub_low == super_low or (sub_low in _VOID_UNIT_SYMBOLS and super_low in _VOID_UNIT_SYMBOLS)

        if "[" in sub_str or "[" in super_str or "{" in sub_str or "{" in super_str:
            if self._atomic_is_subtype(sub_str, super_str):
                return True
            t1 = LatticeType.from_str(sub_str)
            t2 = LatticeType.from_str(super_str)
            return t1.is_subtype_of(t2, self._atomic_is_subtype, is_top_fn=self.is_declared_top)

        return self._atomic_is_subtype(sub_str, super_str)

    def _atomic_is_subtype(self, sub: str, super_: str) -> bool:
        if not sub or not super_:
            return False
        sub_c = self.canonical_name(sub)
        super_c = self.canonical_name(super_)

        sub_l = sub_c.lower()
        super_l = super_c.lower()

        if sub_l in _TRUE_BOTTOM_SYMBOLS or self.is_uninhabited_bottom(sub_l):
            return True
        if super_l in _TRUE_BOTTOM_SYMBOLS or self.is_uninhabited_bottom(super_l):
            return False

        bottom_set = self.get_bottom_types_set()
        if sub_l in bottom_set or super_l in bottom_set:
            return sub_l == super_l or (sub_l in _VOID_UNIT_SYMBOLS and super_l in _VOID_UNIT_SYMBOLS)

        top_set = self.get_top_types_set()
        if super_l in top_set:
            return True
        if sub_l in top_set:
            return False
        if sub_l == super_l:
            return True

        if "_or_" in sub_l:
            parts = [p for p in sub_l.split("_or_") if p]
            if parts and all(self._atomic_is_subtype(part, super_c) for part in parts):
                return True

        if "_or_" in super_l:
            parts = [p for p in super_l.split("_or_") if p]
            if parts and any(self._atomic_is_subtype(sub_c, part) for part in parts):
                return True

        sub_key = sub_l.rsplit(".", 1)[-1] if sub_l not in self._parents and "." in sub_l else sub_l
        super_key = super_l.rsplit(".", 1)[-1] if super_l not in self._parents and "." in super_l else super_l
        if sub_key in self._parents:
            anc_map = self._get_type_ancestors_by_id()
            sub_id = self.intern_symbol(sub_key)
            anc = anc_map.get(sub_id)
            if anc is not None:
                sup_id = self._sym_to_id.get(super_key)
                sup_full_id = self._sym_to_id.get(super_l)
                if (sup_id and sup_id in anc) or (sup_full_id and sup_full_id in anc):
                    return True

        sub_cls = self._resolve_runtime_class(sub_c)
        if sub_cls is not None:
            super_cls = self._resolve_runtime_class(super_c)
            if super_cls is not None:
                try:
                    if issubclass(sub_cls, super_cls):
                        return True
                except TypeError:
                    pass

        return False

    @_locked
    def register_state(
        self,
        state: str,
        parent_state: Optional[str] = None,
        carrier_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None,
    ):
        """Registers a typestate with its optional parent_state, declared
        carrier_type, and verifiable properties (all declared tree data).
        Domain-scoped definitions isolate colliding names across domains."""
        s = str(state).strip().lower()
        if "::" in s:
            d_pfx, _, s_bare = s.partition("::")
            if not domain:
                domain = d_pfx
            s = s_bare
        if not s:
            return
        dom = str(domain).strip().lower() if domain else ""
        mutated = False
        p = str(parent_state).strip().lower() if parent_state else ""
        if "::" in p:
            p = p.partition("::")[2]
        c_str = str(carrier_type) if carrier_type else ""

        if dom:
            defs_map = self._state_domain_defs.setdefault(s, {})
            new_sig = (p or None, c_str.lower() or None)
            prev_sig = defs_map.get(dom)
            if prev_sig != new_sig:
                defs_map[dom] = new_sig
                if len(set(defs_map.values())) > 1:
                    self._colliding_states.add(s)
                mutated = True
            if p and p != s and self._domain_state_parents.get((dom, s)) != p:
                self._domain_state_parents[(dom, s)] = p
                mutated = True
            if c_str and self._domain_state_carriers.get((dom, s)) != c_str:
                self._domain_state_carriers[(dom, s)] = c_str
                mutated = True
            if properties:
                d_pmap = self._domain_state_properties.setdefault((dom, s), {})
                for pk, pv in properties.items():
                    sk = str(pk)
                    if sk not in d_pmap or d_pmap[sk] != pv:
                        d_pmap[sk] = pv
                        mutated = True

        # Legacy global fallback maps (do not clobber colliding cross-domain carriers/parents)
        is_colliding = s in self._colliding_states
        if p and p != s and not is_colliding:
            existing_p = self._state_parents.get(s)
            if not existing_p:
                self._state_parents[s] = p
                mutated = True
            elif existing_p != p:
                # If p reaches existing_p, p is tighter/more specific, upgrade to p
                if self.state_ancestry_reaches(p, {existing_p}):
                    self._state_parents[s] = p
                    mutated = True
                # If existing_p already reaches p, existing_p is tighter, keep existing_p
        if c_str and not is_colliding:
            if s not in self._state_carriers:
                self._state_carriers[s] = c_str
                mutated = True
        if properties and not is_colliding:
            prop_map = self._state_properties.setdefault(s, {})
            for pk, pv in properties.items():
                sk = str(pk)
                if sk not in prop_map or prop_map[sk] != pv:
                    prop_map[sk] = pv
                    mutated = True
        if mutated:
            self._invalidate_caches()

    def get_state_parent(self, state: str) -> Optional[str]:
        """Returns the declared parent_state of a typestate, if any."""
        return self._state_parents.get(str(state).strip().lower())

    @staticmethod
    def _normalize_prop_key(k: str) -> str:
        return str(k).strip().lower()

    def get_state_carrier(self, state: str) -> Optional[str]:
        """Returns declared or inherited carrier_type for a typestate."""
        s = str(state).strip().lower()
        if not s:
            return None
        if s in self._state_carrier_cache:
            return self._state_carrier_cache[s]
        gen = self._generation
        seen: Set[str] = set()
        curr = s
        res: Optional[str] = None
        while curr and curr not in seen:
            seen.add(curr)
            if curr in self._state_carriers:
                res = self._state_carriers[curr]
                break
            curr = self._state_parents.get(curr, "")
        with self._lock:
            if gen == self._generation:
                if len(self._state_carrier_cache) >= _SUBTYPE_CACHE_MAX:
                    self._state_carrier_cache.clear()
                self._state_carrier_cache[s] = res
        return res

    def _get_state_properties_ref(self, state: str) -> Dict[str, Any]:
        """Internal read-only cached view of resolved typestate properties."""
        if not state:
            return {}
        key = str(state).strip().lower()
        if not key:
            return {}
        cached = self._state_properties_cache.get(key)
        if cached is not None:
            return cached
        gen = self._generation
        chain = []
        seen: Set[str] = set()
        curr = key
        while curr and curr not in seen:
            seen.add(curr)
            chain.append(curr)
            curr = self._state_parents.get(curr, "")
        result: Dict[str, Any] = {}
        for s in reversed(chain):
            result.update(self._state_properties.get(s, {}))
        with self._lock:
            if gen == self._generation:
                if len(self._state_properties_cache) >= _SUBTYPE_CACHE_MAX:
                    self._state_properties_cache.clear()
                self._state_properties_cache[key] = result
        return result

    def get_state_properties(self, state: str) -> Dict[str, Any]:
        """Returns the declared verifiable properties of a typestate.
        Walks the parent chain from root down to child so refinements inherit
        base properties while child properties override parent properties."""
        ref = self._get_state_properties_ref(state)
        return ref.copy() if ref else {}

    def has_property_conflict(self, producer_state: str, consumer_state: str) -> bool:
        """True iff consumer_state requires a non-None property that conflicts
        with a property declared on producer_state (on the same carrier family)."""
        p_s = str(producer_state or "").strip().lower()
        c_s = str(consumer_state or "").strip().lower()
        if not p_s or not c_s or p_s == c_s:
            return False
        p_props = self._get_state_properties_ref(p_s)
        c_props = self._get_state_properties_ref(c_s)
        if not p_props or not c_props:
            return False

        # Carrier family check: only compare properties if on the same carrier family
        p_car = self.get_state_carrier(p_s)
        c_car = self.get_state_carrier(c_s)
        if p_car and c_car:
            p_car_l, c_car_l = p_car.lower(), c_car.lower()
            if p_car_l != c_car_l and not self.is_subtype(p_car_l, c_car_l) and not self.is_subtype(c_car_l, p_car_l):
                return False

        norm_p = {self._normalize_prop_key(k): v for k, v in p_props.items()}
        norm_c = {self._normalize_prop_key(k): v for k, v in c_props.items()}

        for k, v_c in norm_c.items():
            if v_c is None:
                continue
            if k in norm_p:
                v_p = norm_p[k]
                if v_p is None:
                    continue
                if isinstance(v_c, str) and isinstance(v_p, str):
                    if v_c.strip().lower() != v_p.strip().lower():
                        return True
                elif v_c != v_p:
                    return True
        return False

    def state_ancestry_reaches(self, state: str, ancestors: Union[Set[str], Tuple[str, ...]]) -> bool:
        """True iff the state's declared parent chain reaches any of `ancestors`."""
        curr = state.strip().lower() if isinstance(state, str) else str(state).strip().lower()
        if not curr:
            return False
        if isinstance(ancestors, tuple):
            anc_key = ancestors
        else:
            anc_key = tuple(sorted({a.strip().lower() for a in ancestors}))
        cache_key = (curr, anc_key)
        cached = self._ancestry_cache.get(cache_key)
        if cached is not None:
            return cached
        anc_set = set(anc_key)

        gen = self._generation
        res = False
        seen: Set[str] = set()
        temp_curr = curr
        parents_map = self._state_parents
        while temp_curr and temp_curr not in seen:
            if temp_curr in anc_set:
                res = True
                break
            seen.add(temp_curr)
            temp_curr = parents_map.get(temp_curr, "")

        with self._lock:
            if gen == self._generation:
                if len(self._ancestry_cache) >= _SUBTYPE_CACHE_MAX * 2:
                    self._ancestry_cache.clear()
                self._ancestry_cache[cache_key] = res
        return res

    def _get_terminal_sink_states(self) -> Tuple[Set[str], Tuple[str, ...]]:
        cached = self._terminal_sink_states_cache
        if cached is not None:
            return cached
        gen = self._generation
        s = (
            set(self.get_terminal_states())
            | self.get_states_by_property("is_terminal", True)
            | self.get_states_by_property("is_invalid", True)
            | self.get_states_by_property("is_terminal_invalid", True)
        )
        suffixes = tuple("_" + ts for ts in s if ts)
        res = (s, suffixes)
        with self._lock:
            if gen == self._generation:
                self._terminal_sink_states_cache = res
        return res

    def is_state_compatible(
        self,
        producer_state: Optional[str],
        consumer_state: Optional[str],
        producer_accepted: Union[Set[str], Tuple[str, ...], List[str]] = (),
        consumer_accepted: Union[Set[str], Tuple[str, ...], List[str]] = (),
        consumer_parent: Optional[str] = None,
        producer_parent: Optional[str] = None,
        producer_domain: Optional[str] = None,
        consumer_domain: Optional[str] = None,
    ) -> bool:
        """
        Dual-layer typestate compatibility evaluation between producer output and consumer input.
        - Top/unconstrained states (None, '', 'any', '*', 'default', 'unregistered', 'unbound', 'top')
          always accept any producer state.
        - Concrete consumer states require proof: an unconstrained Top producer cannot satisfy them.
        - Exact match and consumer accepted_states (with cross-domain collision guard).
        - Covariant substate compatibility: walks state ancestry.
        - Property-bundle subsumption (Phi_prod |= Phi_cons): verifies no conflicting properties.
        - Ingress/egress asset identifier family equivalence.
        """
        if not consumer_accepted:
            if consumer_state is None or (isinstance(consumer_state, str) and consumer_state in _TOP_STATE_LITERALS):
                return True
        raw_cache = getattr(self, "_state_raw_cache", None)
        raw_key: Optional[Tuple[Any, ...]] = None
        if raw_cache is not None and (
            (producer_state is None or isinstance(producer_state, str))
            and (consumer_state is None or isinstance(consumer_state, str))
        ):
            p_acc_k = (
                () if not producer_accepted else (
                    producer_accepted if isinstance(producer_accepted, tuple) else (
                        tuple(producer_accepted) if isinstance(producer_accepted, list) else (
                            tuple(sorted(str(x) for x in producer_accepted)) if isinstance(producer_accepted, set) else None
                        )
                    )
                )
            )
            c_acc_k = (
                () if not consumer_accepted else (
                    consumer_accepted if isinstance(consumer_accepted, tuple) else (
                        tuple(consumer_accepted) if isinstance(consumer_accepted, list) else (
                            tuple(sorted(str(x) for x in consumer_accepted)) if isinstance(consumer_accepted, set) else None
                        )
                    )
                )
            )
            if p_acc_k is not None and c_acc_k is not None:
                raw_key = (
                    producer_state, consumer_state, p_acc_k, c_acc_k,
                    consumer_parent, producer_parent, producer_domain, consumer_domain,
                )
                raw_hit = raw_cache.get(raw_key)
                if raw_hit is not None:
                    return raw_hit
        p_state = producer_state.strip().lower() if isinstance(producer_state, str) else str(producer_state or "").strip().lower()
        c_state = consumer_state.strip().lower() if isinstance(consumer_state, str) else str(consumer_state or "").strip().lower()
        p_dom = producer_domain.strip().lower() if isinstance(producer_domain, str) else ""
        c_dom = consumer_domain.strip().lower() if isinstance(consumer_domain, str) else ""
        if "::" in p_state:
            p_dom_p, _, p_state = p_state.partition("::")
            p_dom = p_dom or p_dom_p
        if "::" in c_state:
            c_dom_c, _, c_state = c_state.partition("::")
            c_dom = c_dom or c_dom_c
        if not consumer_accepted and c_state in _TOP_STATE_LITERALS:
            return True
        p_acc = tuple(sorted({str(s).strip().lower() for s in producer_accepted if str(s).strip()})) if producer_accepted else _EMPTY_TUPLE
        c_acc = tuple(sorted({str(s).strip().lower() for s in consumer_accepted if str(s).strip()})) if consumer_accepted else _EMPTY_TUPLE
        c_par = consumer_parent.strip().lower() if isinstance(consumer_parent, str) else (str(consumer_parent).strip().lower() if consumer_parent else "")
        p_par = producer_parent.strip().lower() if isinstance(producer_parent, str) else (str(producer_parent).strip().lower() if producer_parent else "")

        cache_key = (p_state, c_state, p_acc, c_acc, c_par, p_par, p_dom, c_dom)
        hit = self._state_compatible_cache.get(cache_key)
        if hit is not None:
            if raw_key is not None and len(raw_cache) < _SUBTYPE_CACHE_MAX * 2:
                raw_cache[raw_key] = hit
            return hit

        gen = self._generation
        res = self._compute_is_state_compatible(p_state, c_state, p_acc, c_acc, c_par, p_par, p_dom, c_dom)
        with self._lock:
            if gen == self._generation:
                if len(self._state_compatible_cache) >= _SUBTYPE_CACHE_MAX * 2:
                    self._state_compatible_cache.clear()
                self._state_compatible_cache[cache_key] = res
                if raw_cache is None:
                    self._state_raw_cache = {}
                    raw_cache = self._state_raw_cache
                if raw_key is not None:
                    if len(raw_cache) >= _SUBTYPE_CACHE_MAX * 2:
                        raw_cache.clear()
                    raw_cache[raw_key] = res
        return res

    def _compute_is_state_compatible(
        self,
        p_state: str,
        c_state: str,
        p_acc: Tuple[str, ...],
        c_acc: Tuple[str, ...],
        c_par: str,
        p_par: str,
        p_dom: str = "",
        c_dom: str = "",
    ) -> bool:
        # 0. Cross-domain collision guard: if bare state is declared with conflicting
        # carriers across domains (e.g. audio::filtered vs pandas::filtered), require
        # compatible domain carriers rather than conflating identical strings.
        def _cross_domain_carrier_conflict(ps: str, cs: str) -> bool:
            if not (p_dom and c_dom and p_dom != c_dom):
                return False
            if ps not in self._colliding_states and cs not in self._colliding_states:
                return False
            p_car = self._domain_state_carriers.get((p_dom, ps))
            c_car = self._domain_state_carriers.get((c_dom, cs))
            if p_car and c_car:
                pl, cl = p_car.strip().lower(), c_car.strip().lower()
                if pl != cl and not self.is_subtype(pl, cl) and not self.is_subtype(cl, pl):
                    return True
            elif (p_car and not c_car) or (c_car and not p_car):
                if ps == cs:
                    return True
            return False

        # 1. Unconstrained consumer typestate (Phi = Top)
        if any(self.is_declared_top_state(s) for s in (c_acc or (c_state,))) or (
            self.is_declared_top_state(c_state) and not (set(c_acc) - self._declared_top_states)
        ):
            # A terminal sink state (e.g. evaluated_metrics, predicted_values, written_to_disk)
            # must NOT bypass a concrete non-terminal consumer state via a generic carrier
            # wildcard in c_acc; it requires exact or explicitly declared compatibility.
            if (
                p_state
                and c_state
                and not self.is_declared_top_state(c_state)
                and p_state != c_state
                and p_state not in c_acc
            ):
                terminal_sink_states, terminal_sink_suffixes = self._get_terminal_sink_states()
                pred_states = self.get_states_by_property("is_prediction", True)
                is_p_terminal = p_state in terminal_sink_states or (
                    bool(terminal_sink_suffixes) and p_state.endswith(terminal_sink_suffixes)
                )
                is_c_terminal = c_state in terminal_sink_states or (
                    bool(terminal_sink_suffixes) and c_state.endswith(terminal_sink_suffixes)
                )
                is_p_pred = p_state in pred_states
                is_c_pred = c_state in pred_states
                if not ((is_p_terminal and not is_c_terminal) or (is_p_pred and not is_c_terminal and not is_c_pred)):
                    return True
            else:
                return True

        # 2. Unconstrained producer state cannot satisfy constrained consumer
        if self.is_declared_top_state(p_state) and not self.is_declared_top_state(c_state):
            if p_state and p_state in c_acc:
                return True
            return False

        # 3. Exact match
        if p_state and p_state == c_state:
            if _cross_domain_carrier_conflict(p_state, c_state):
                return False
            return not self.has_property_conflict(p_state, c_state)

        # 4. Accepted states match (consumer declares accepted producer states)
        if p_state and p_state in c_acc:
            return True

        # 5. Covariant substate compatibility: walks declared state ancestry
        target_ancestors: Set[str] = set()
        if c_state and not self.is_declared_top_state(c_state):
            target_ancestors.add(c_state)
        target_ancestors.update(s for s in c_acc if not self.is_declared_top_state(s))

        if p_state and target_ancestors:
            tgt_tuple = tuple(sorted(set(target_ancestors)))
            if self.state_ancestry_reaches(p_state, tgt_tuple):
                for tgt in target_ancestors:
                    if self.state_ancestry_reaches(p_state, (tgt,)):
                        if not self.has_property_conflict(p_state, tgt):
                            return True
            curr_p = (
                p_par
                or (self._domain_state_parents.get((p_dom, p_state)) if p_dom else None)
                or self.get_state_parent(p_state)
            )
            seen_dp: Set[str] = {p_state}
            while curr_p and curr_p not in seen_dp:
                seen_dp.add(curr_p)
                if curr_p in target_ancestors and not self.has_property_conflict(p_state, curr_p):
                    return True
                if self.state_ancestry_reaches(curr_p, tgt_tuple):
                    for tgt in target_ancestors:
                        if self.state_ancestry_reaches(curr_p, (tgt,)):
                            if not self.has_property_conflict(curr_p, tgt):
                                return True
                curr_p = (
                    (self._domain_state_parents.get((p_dom, curr_p)) if p_dom else None)
                    or self.get_state_parent(curr_p)
                )

        # 6. Ingress/egress asset identifier state equivalence (e.g. source_identifier <-> valid_path / file_path)
        def _get_root_state(s: str) -> str:
            visited = set()
            curr = s
            while curr and curr not in visited:
                visited.add(curr)
                parent = self._state_parents.get(curr)
                if not parent:
                    return curr
                curr = parent
            return curr

        if p_state and c_state:
            root_p = _get_root_state(p_state)
            root_c = _get_root_state(c_state)
            if root_p and root_p == root_c and root_p in self._identifier_root_states:
                return True

        # 7. Positive Cross-Tree Property-Bundle Entailment (Phi_prod |= Phi_cons)
        if p_state and c_state:
            p_props = self._get_state_properties_ref(p_state)
            c_props = self._get_state_properties_ref(c_state)
            if p_props and c_props:
                terminal_sink_states, terminal_sink_suffixes = self._get_terminal_sink_states()
                is_p_terminal = p_state in terminal_sink_states or (bool(terminal_sink_suffixes) and p_state.endswith(terminal_sink_suffixes))
                is_c_terminal = c_state in terminal_sink_states or (bool(terminal_sink_suffixes) and c_state.endswith(terminal_sink_suffixes))
                if not (is_p_terminal or is_c_terminal):
                    p_car = self.get_state_carrier(p_state)
                    c_car = self.get_state_carrier(c_state)
                    carrier_compat = True
                    if p_car and c_car:
                        p_car_l, c_car_l = p_car.lower(), c_car.lower()
                        carrier_compat = (
                            p_car_l == c_car_l
                            or self.is_subtype(p_car_l, c_car_l)
                            or self.is_subtype(c_car_l, p_car_l)
                        )
                    if carrier_compat:
                        norm_p = {self._normalize_prop_key(k): v for k, v in p_props.items()}
                        norm_c = {self._normalize_prop_key(k): v for k, v in c_props.items()}
                        non_none_c = {k: v for k, v in norm_c.items() if v is not None}
                        if non_none_c:
                            entailed = True
                            for k, v_c in non_none_c.items():
                                if k not in norm_p:
                                    entailed = False
                                    break
                                v_p = norm_p[k]
                                if v_p is None:
                                    entailed = False
                                    break
                                if isinstance(v_c, str) and isinstance(v_p, str):
                                    if v_c.strip().lower() != v_p.strip().lower():
                                        entailed = False
                                        break
                                elif v_c != v_p:
                                    entailed = False
                                    break
                            if entailed and not self.has_property_conflict(p_state, c_state):
                                return True

        return False



def _legacy_cache_clear() -> None:
    TypeRegistry.get_instance()._invalidate_caches()

# Back-compat: callers used to do `registry.is_subtype.cache_clear()` (functools.lru_cache API).
TypeRegistry.is_subtype.cache_clear = _legacy_cache_clear


def is_subtype(sub: str, parent: str) -> bool:
    return TypeRegistry.get_instance().is_subtype(sub, parent)


def canonical_type_name(type_name: str) -> str:
    return TypeRegistry.get_instance().canonical_name(type_name)


def is_path_port(port: Any, registry: Optional[TypeRegistry] = None) -> bool:
    """
    Single shared predicate for "this port carries a filesystem path asset".
    Type-driven (declared poset + abstract_type + declared role) with a documented naming
    fallback for trees whose ports are string-typed path parameters.
    """
    if port is None:
        return False
    if registry is None:
        registry = TypeRegistry.get_instance()
    gen = registry._generation
    if isinstance(port, AlgebraicSignature):
        hit = registry._path_port_sig_cache.get(port)
        if hit is not None:
            return hit
        res = _compute_is_path_port(port, registry)
        with registry._lock:
            if gen == registry._generation:
                if len(registry._path_port_sig_cache) >= _SUBTYPE_CACHE_MAX:
                    registry._path_port_sig_cache.clear()
                registry._path_port_sig_cache[port] = res
        return res

    cached = getattr(port, "_cached_is_path_port", None)
    if isinstance(cached, tuple) and len(cached) == 2 and cached[0] == gen:
        return cached[1]

    res = _compute_is_path_port(port, registry)
    try:
        port._cached_is_path_port = (gen, res)
    except (AttributeError, TypeError):
        pass
    return res


def _compute_is_path_port(port: Any, registry: Optional[TypeRegistry] = None) -> bool:
    if registry is None:
        registry = TypeRegistry.get_instance()
    if str(getattr(port, "structure_kind", "") or "").strip().lower() == "path_spec":
        return True
    sig = getattr(port, "signature", port)

    def _is_path_type(candidate: str) -> bool:
        c = str(candidate or "").strip().lower()
        if not c:
            return False
        base = c.partition("[")[0].strip() if ("[" in c and c.endswith("]")) else c
        path_types = set(registry.get_types_by_kind("path") or ()) | {"path", "file"}
        for tok in (c, base):
            if not tok:
                continue
            if (
                tok in path_types
                or registry.has_type_kind(tok, "path")
                or registry.is_subtype(tok, "path")
                or registry.is_subtype(tok, "file")
            ):
                return True
            aliased = registry._aliases.get(tok)
            if aliased:
                al = str(aliased).strip().lower()
                if (
                    al in path_types
                    or registry.has_type_kind(al, "path")
                    or registry.is_subtype(al, "path")
                    or registry.is_subtype(al, "file")
                ):
                    return True
        return False

    t_name = str(getattr(sig, "type_name", "") or "")
    if _is_path_type(t_name):
        return True
    abstract_t = str(
        getattr(port, "abstract_type", "") or getattr(sig, "abstract_type", "") or ""
    )
    if _is_path_type(abstract_t):
        return True
    c_type = str(
        getattr(sig, "carrier_type", "")
        or getattr(port, "carrier_type", "")
        or getattr(port, "carrier", "")
        or ""
    )
    if _is_path_type(c_type):
        return True
    return False


def _clean_abs_carrier(val: Any) -> str:
    if not val:
        return ""
    s = str(val).strip()
    if s.lower() in ("none", "null", ""):
        return ""
    return s


def _normalize_qualifiers(raw_qualifiers: Any) -> Tuple[Tuple[str, ...], ...]:
    if not raw_qualifiers:
        return ()
    # A bare string is ONE qualifier, not a sequence of characters
    if isinstance(raw_qualifiers, str):
        return ((raw_qualifiers.strip(),),) if raw_qualifiers.strip() else ()
    result = []
    for q in raw_qualifiers:
        if isinstance(q, (list, tuple)):
            result.append(tuple(str(item) for item in q))
        elif isinstance(q, str):
            result.append((q,))
        else:
            result.append((str(q),))
    return tuple(sorted(set(result)))


@dataclass(frozen=True, slots=True)
class AlgebraicSignature:
    """
    Formal typestate signature: tau = (type_name, state, qualifiers, abstract_type, accepted_states, parent_state, domain).
    Conforms to Section 3.1 of the STRATA paper.
    """
    type_name: str = "any"
    state: str = "any"
    qualifiers: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)
    abstract_type: str = ""
    accepted_states: Tuple[str, ...] = field(default_factory=tuple)
    parent_state: Optional[str] = None
    domain: str = ""
    _hash_val: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self):
        cleaned = _clean_abs_carrier(self.abstract_type)
        if cleaned != self.abstract_type:
            object.__setattr__(self, "abstract_type", cleaned)
        if self.accepted_states:
            acc_norm = tuple(sorted({str(s).strip().lower() for s in self.accepted_states if str(s).strip()}))
            if acc_norm != self.accepted_states:
                object.__setattr__(self, "accepted_states", acc_norm)
        elif not isinstance(self.accepted_states, tuple):
            object.__setattr__(self, "accepted_states", ())
        if self.parent_state:
            object.__setattr__(self, "parent_state", str(self.parent_state).strip().lower())
        dom_norm = str(self.domain or "").strip().lower()
        if dom_norm != self.domain:
            object.__setattr__(self, "domain", dom_norm)
        if self.qualifiers is not None:
            norm_q = _normalize_qualifiers(self.qualifiers)
            if norm_q != self.qualifiers:
                object.__setattr__(self, "qualifiers", norm_q)
        else:
            object.__setattr__(self, "qualifiers", ())
        object.__setattr__(
            self,
            "_hash_val",
            hash((self.type_name, self.state, self.qualifiers, self.abstract_type, self.accepted_states, self.parent_state, self.domain)),
        )

    def __hash__(self) -> int:
        return self._hash_val

    @classmethod
    def from_string(cls, type_name: str, state: str = "any", abstract_type: str = "", accepted_states: Optional[Any] = None, parent_state: Optional[str] = None, domain: str = "") -> "AlgebraicSignature":
        acc = tuple(sorted({str(s).strip().lower() for s in (accepted_states or []) if str(s).strip()})) if accepted_states else ()
        p = str(parent_state).strip().lower() if parent_state else None
        return cls(type_name=type_name, state=state, abstract_type=_clean_abs_carrier(abstract_type), accepted_states=acc, parent_state=p, domain=domain)

    def is_top(self) -> bool:
        return TypeRegistry.get_instance().is_declared_top(self.type_name)

    def unifies_with(self, other: Any, registry: Optional[TypeRegistry] = None) -> bool:
        """
        Evaluates whether producer output `self` can satisfy consumer input `other`.
        Rules:
          1. Consumer Top accepts any producer type.
          2. Non-top consumer rejects top producer.
          3. State compatibility check:
             - matches if either is 'any' or '*'
             - matches if exact match
             - (a) accepts if other_sig.state is in producer's accepted_states (or vice versa)
             - (b) walks parent_state up consumer's declared chain before rejecting
          4. Producer type must be a subtype of consumer type in the Type Poset,
             or unify via compatible abstract categorical carriers.
          5. Consumer qualifiers must be a subset of producer qualifiers.
        """
        if isinstance(other, AlgebraicSignature):
            other_sig = other
        elif isinstance(other, PortSignature):
            other_sig = other.signature
        elif hasattr(other, "signature") and isinstance(other.signature, AlgebraicSignature):
            other_sig = other.signature
        else:
            return False

        if registry is None:
            registry = TypeRegistry.get_instance()
        cache_key = (self, other_sig)
        hit = registry._unifies_cache.get(cache_key)
        if hit is not None:
            return hit

        gen = registry._generation
        res = self._compute_unifies_with(other_sig, registry)
        with registry._lock:
            if gen == registry._generation:
                if len(registry._unifies_cache) >= _SUBTYPE_CACHE_MAX * 4:
                    registry._unifies_cache.clear()
                registry._unifies_cache[cache_key] = res
        return res

    def _compute_unifies_with(self, other_sig: "AlgebraicSignature", registry: TypeRegistry) -> bool:
        # State compatibility check
        is_both_path = is_path_port(self, registry) and is_path_port(other_sig, registry)
        if is_both_path:
            p_st = str(self.state or "").strip().lower()
            inv_states = registry.get_states_by_property("is_invalid", True) | registry.get_states_by_property("is_terminal_invalid", True)
            if p_st in inv_states:
                return False
        elif not registry.is_state_compatible(
            producer_state=self.state,
            consumer_state=other_sig.state,
            producer_accepted=self.accepted_states,
            consumer_accepted=other_sig.accepted_states,
            consumer_parent=other_sig.parent_state,
            producer_parent=self.parent_state,
            producer_domain=self.domain or None,
            consumer_domain=other_sig.domain or None,
        ):
            return False

        # Consumer accepts anything
        if registry.is_declared_top(other_sig.type_name):
            return True
        # Producer is untyped wildcard, cannot guarantee concrete type requirement
        if registry.is_declared_top(self.type_name):
            return False

        # Category-Theoretic Generic Functor & Type Variable Unification
        p_tn = (self.type_name or "").strip()
        c_tn = (other_sig.type_name or "").strip()
        is_generic_match = False
        m_c = None
        m_p = None

        # 1. Declared type variables (dynamic registry vocabulary + syntactic fallback) unify universally.
        if registry.is_type_variable(c_tn) or registry.is_type_variable(p_tn):
            is_generic_match = True
        else:
            # 2. Parametric Container / Functor unification (e.g. List[Contour] <-> List[T]) without regex
            def _parse_container(tn: str) -> Optional[Tuple[str, str]]:
                if tn.endswith("]") and "[" in tn:
                    ctor, _, inner = tn[:-1].partition("[")
                    if ctor.isidentifier():
                        return ctor.lower(), inner.strip()
                return None
            m_c = _parse_container(c_tn)
            m_p = _parse_container(p_tn)
            if m_c and m_p:
                c_ctor, c_inner = m_c
                p_ctor, p_inner = m_p
                if c_ctor == p_ctor:
                    c_args = [a.strip() for a in c_inner.split(",") if a.strip()]
                    p_args = [a.strip() for a in p_inner.split(",") if a.strip()]
                    if len(c_args) == len(p_args) and len(c_args) > 1:
                        if all(
                            registry.is_type_variable(ca) or registry.is_declared_top(ca)
                            or registry.is_type_variable(pa) or registry.is_declared_top(pa)
                            or registry.is_subtype(pa, ca)
                            for ca, pa in zip(c_args, p_args)
                        ):
                            is_generic_match = True
                    elif (
                        registry.is_type_variable(c_inner) or registry.is_declared_top(c_inner)
                        or registry.is_type_variable(p_inner) or registry.is_declared_top(p_inner)
                    ):
                        is_generic_match = True
            elif m_c and not m_p:
                # Raw producer C into parameterized consumer F[args]: sound only when every
                # demanded argument is a type variable / Top (no element proof required).
                c_ctor, c_inner = m_c
                c_args = [a.strip() for a in c_inner.split(",") if a.strip()]
                if registry.is_subtype(p_tn, c_ctor) and all(
                    registry.is_type_variable(a) or registry.is_declared_top(a) for a in c_args
                ):
                    is_generic_match = True
            elif m_p and not m_c:
                # Parameterized producer F[args] into raw consumer C: F ⊑ C suffices (forgetful map).
                p_ctor, _ = m_p
                if registry.is_subtype(p_ctor, c_tn):
                    is_generic_match = True

        # Poset subtyping
        if not is_generic_match and not registry.is_subtype(self.type_name, other_sig.type_name):
            # Poset abstract carrier compatibility fallback:
            # Fallback ONLY applies when consumer target is expecting an abstract carrier/interface,
            # NOT when consumer target is a concrete class receiver (e.g. MultiIndex, LinearRegression).
            other_tn = (other_sig.type_name or "").strip().lower()
            other_abs = (other_sig.abstract_type or "").strip().lower()
            is_consumer_abstract = (registry.is_abstract_carrier(other_tn) or other_tn == other_abs)
            if (
                is_consumer_abstract
                and self.abstract_type
                and other_sig.abstract_type
                and registry.is_subtype(self.abstract_type, other_sig.abstract_type)
            ):
                pass
            elif is_both_path:
                if m_c and m_p and not is_generic_match:
                    return False
            else:
                return False

        # Qualifier satisfaction & mutual-exclusion check
        self_quals = set(self.qualifiers or ())
        other_quals = set(other_sig.qualifiers or ())
        if registry.has_conflicting_qualifiers(self_quals, other_quals):
            return False
        if other_quals and not other_quals.issubset(self_quals):
            # Advisory qualifiers denote shape/role REFINEMENTS a producer may
            # legitimately omit (harvesters do not always propagate them);
            # they are documented contract-relaxation data, not silent demands.
            adv_quals = registry.get_advisory_qualifiers()
            req = {q for q in other_quals if q not in adv_quals}
            if req and not req.issubset(self_quals):
                return False

        return True

    def matches(self, other: Any) -> bool:
        return self.unifies_with(other)


class CaseInsensitiveDict(dict):
    """
    Case-insensitive dictionary for cell lookups across lowercase/uppercase identifiers.
    Preserves exact keys while allowing case-insensitive retrieval and domain alias resolution.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lower_map: Dict[str, Any] = {str(k).lower(): k for k in self.keys()}

    def _resolve_key(self, key: Any) -> Optional[str]:
        if super().__contains__(key):
            return key
        lower = str(key).lower()
        if lower in self._lower_map:
            return self._lower_map[lower]
        # Check domain prefix aliases registered in TypeRegistry (e.g. pd <-> pandas)
        try:
            reg = TypeRegistry.get_instance()
            aliases = reg.get_all_aliases()
            for alias, canonical in aliases.items():
                a_pre = f"{alias.lower()}_"
                c_pre = f"{canonical.lower()}_"
                if lower.startswith(c_pre):
                    candidate = a_pre + lower[len(c_pre):]
                    if candidate in self._lower_map:
                        return self._lower_map[candidate]
                elif lower.startswith(a_pre):
                    candidate = c_pre + lower[len(a_pre):]
                    if candidate in self._lower_map:
                        return self._lower_map[candidate]
        except Exception:
            pass
        return None

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._lower_map[str(key).lower()] = key

    def __delitem__(self, key):
        actual = self._resolve_key(key)
        if actual is not None:
            super().__delitem__(actual)
            self._lower_map.pop(str(actual).lower(), None)
        else:
            super().__delitem__(key)

    def __getitem__(self, key):
        actual = self._resolve_key(key)
        if actual is not None:
            return super().__getitem__(actual)
        raise KeyError(key)

    def __contains__(self, key):
        return self._resolve_key(key) is not None

    def get(self, key, default=None):
        actual = self._resolve_key(key)
        if actual is not None:
            return super().__getitem__(actual)
        return default

    def pop(self, key, *args):
        actual = self._resolve_key(key)
        if actual is None:
            if args:
                return args[0]
            raise KeyError(key)
        self._lower_map.pop(str(actual).lower(), None)
        return super().pop(actual)

    def setdefault(self, key, default=None):
        actual = self._resolve_key(key)
        if actual is not None:
            return super().__getitem__(actual)
        self[key] = default
        return default

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def clear(self):
        super().clear()
        self._lower_map.clear()


class PortMapping(dict):
    """
    Port dictionary supporting canonical port names as primary keys,
    while permitting alias access (e.g. 'port_0', 'port_1') for backward compatibility.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases: Dict[str, str] = {}

    def add_alias(self, alias: str, canonical: str):
        self._aliases[alias] = canonical

    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if key in self._aliases:
            return super().__getitem__(self._aliases[key])
        raise KeyError(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        return super().__contains__(key) or key in self._aliases


class PortSignature:
    """Named port carrying an AlgebraicSignature typestate."""
    __slots__ = [
        "name", "signature", "required", "default_value", "doc", "domain",
        "abstract_type", "enum_values", "param_kind", "value_constraints", "shape_contract",
        "rejected_qualifiers",
        "accepted_states", "parent_state", "port_role", "_cached_term", "_cached_col_proj",
        "_cached_derived_role", "_cached_is_path_port", "_cached_ndim", "_cached_rank_cache", "binds",
        "structure_kind", "rank", "element_type", "supported_formats", "owner_cell",
        "complement_of_port", "disjoint_from_port", "key_from_upstream_role",
        "carrier", "nominal_type", "layout", "semantics", "typestate", "invariants", "effect_suspended",
        "mutates", "key_of", "key_effect", "partition_index", "partition"
    ]

    def __init__(
        self,
        name: str = "",
        signature: Union[AlgebraicSignature, str, Any] = "any",
        required: bool = True,
        default_value: Optional[Any] = None,
        doc: str = "",
        domain: str = "",
        abstract_type: str = "",
        enum_values: Optional[List[Any]] = None,
        param_kind: str = "standard",
        value_constraints: Optional[Dict[str, Any]] = None,
        shape_contract: Optional[Dict[str, Any]] = None,
        rejected_qualifiers: Optional[Union[List[str], Set[str], Tuple[str, ...]]] = None,
        accepted_states: Optional[Union[List[str], Set[str], Tuple[str, ...]]] = None,
        parent_state: Optional[str] = None,
        port_role: Optional[str] = None,
        binds: Optional[str] = None,
        structure_kind: Optional[str] = None,
        rank: Optional[int] = None,
        element_type: Optional[str] = None,
        supported_formats: Optional[List[str]] = None,
        complement_of_port: Optional[str] = None,
        disjoint_from_port: Optional[str] = None,
        key_from_upstream_role: Optional[str] = None,
        carrier: Optional[str] = None,
        nominal_type: Optional[str] = None,
        layout: Optional[Dict[str, Any]] = None,
        semantics: Optional[str] = None,
        typestate: Optional[str] = None,
        invariants: Optional[Union[List[str], Tuple[str, ...]]] = None,
        effect_suspended: bool = False,
        mutates: Optional[str] = None,
        key_of: Optional[str] = None,
        key_effect: Optional[str] = None,
        partition_index: Optional[int] = None,
        partition: Optional[Any] = None,
        **kwargs
    ):
        self.name = str(name)
        self.domain = str(domain or kwargs.get("domain", ""))
        self.abstract_type = _clean_abs_carrier(abstract_type or kwargs.get("abstract_type", ""))
        self.enum_values = enum_values if enum_values is not None else kwargs.get("enum_values")
        self.param_kind = str(param_kind or kwargs.get("param_kind", "standard"))
        self.value_constraints = value_constraints if value_constraints is not None else kwargs.get("value_constraints")
        self.shape_contract = shape_contract if shape_contract is not None else kwargs.get("shape_contract")
        raw_binds = binds if binds is not None else kwargs.get("binds")
        self.binds = str(raw_binds).strip().lower() if raw_binds else None
        self.structure_kind = structure_kind or kwargs.get("structure_kind") or None
        self.rank = rank if rank is not None else kwargs.get("rank")
        self.element_type = element_type or kwargs.get("element_type") or None
        self.supported_formats = list(supported_formats or kwargs.get("supported_formats") or [])
        self.complement_of_port = complement_of_port or kwargs.get("complement_of_port") or None
        self.disjoint_from_port = disjoint_from_port or kwargs.get("disjoint_from_port") or None
        self.key_from_upstream_role = key_from_upstream_role or kwargs.get("key_from_upstream_role") or None
        self.nominal_type = str(nominal_type or kwargs.get("nominal_type") or "").strip().lower() or None
        self.carrier = str(carrier or kwargs.get("carrier") or "").strip() or None
        raw_layout = layout if layout is not None else kwargs.get("layout")
        if raw_layout is None and (self.shape_contract is not None or self.rank is not None or self.element_type is not None):
            sc_d = self.shape_contract if isinstance(self.shape_contract, dict) else {}
            raw_layout = {
                "rank": sc_d.get("ndim", self.rank),
                "shape": sc_d.get("shape"),
                "dtype": self.element_type,
                "rank_polymorphic": bool(sc_d.get("rank_polymorphic", False)),
                "channel_count": sc_d.get("channel_count"),
                "channel_axis": sc_d.get("channel_axis"),
            }
        self.layout = dict(raw_layout) if isinstance(raw_layout, dict) else raw_layout
        self.semantics = str(semantics or kwargs.get("semantics") or "").strip().lower() or None
        self.typestate = str(typestate or kwargs.get("typestate") or "").strip().lower() or None
        raw_inv = invariants if invariants is not None else kwargs.get("invariants", [])
        self.invariants = tuple(str(x).strip().lower() for x in raw_inv if str(x).strip()) if isinstance(raw_inv, (list, tuple, set)) else ()
        self.effect_suspended = bool(effect_suspended or kwargs.get("effect_suspended", False))
        self.mutates = mutates if mutates is not None else kwargs.get("mutates")
        self.key_of = key_of if key_of is not None else kwargs.get("key_of")
        self.key_effect = key_effect if key_effect is not None else kwargs.get("key_effect")
        raw_p_idx = partition_index if partition_index is not None else kwargs.get("partition_index")
        self.partition_index = int(raw_p_idx) if raw_p_idx is not None else None
        self.partition = partition if partition is not None else kwargs.get("partition")
        if self.supported_formats:
            try:
                TypeRegistry.get_instance().register_supported_formats(self.supported_formats)
            except Exception:
                pass
        acc_rej = rejected_qualifiers if rejected_qualifiers is not None else kwargs.get("rejected_qualifiers", [])
        if isinstance(acc_rej, (set, list, tuple)):
            self.rejected_qualifiers = tuple(sorted({str(s).strip().lower() for s in acc_rej if str(s).strip()}))
        else:
            self.rejected_qualifiers = ()
        self.port_role = str(port_role).strip().lower() if port_role else (str(kwargs.get("port_role")).strip().lower() if kwargs.get("port_role") else None)

        acc = accepted_states if accepted_states is not None else kwargs.get("accepted_states", [])
        if isinstance(acc, (set, list, tuple)):
            self.accepted_states = tuple(sorted({str(s).strip().lower() for s in acc if str(s).strip()}))
        else:
            self.accepted_states = ()

        raw_parent = parent_state or kwargs.get("parent_state")
        self.parent_state = str(raw_parent).strip().lower() if raw_parent else None
        self.owner_cell = kwargs.get("owner_cell", None)

        eff_dom = str(self.domain or "").strip().lower()
        if not eff_dom and self.typestate and "::" in self.typestate:
            eff_dom = self.typestate.partition("::")[0].strip().lower()
        if isinstance(signature, AlgebraicSignature):
            acc_combined = self.accepted_states or signature.accepted_states
            p_combined = self.parent_state or signature.parent_state
            dom_combined = eff_dom or signature.domain
            if (self.abstract_type and not signature.abstract_type) or (acc_combined != signature.accepted_states) or (p_combined != signature.parent_state) or (dom_combined != signature.domain):
                self.signature = AlgebraicSignature(
                    type_name=signature.type_name,
                    state=signature.state,
                    qualifiers=signature.qualifiers,
                    abstract_type=self.abstract_type or signature.abstract_type,
                    accepted_states=acc_combined,
                    parent_state=p_combined,
                    domain=dom_combined,
                )
            else:
                self.signature = signature
            if signature.abstract_type:
                self.abstract_type = signature.abstract_type
            if signature.accepted_states and not self.accepted_states:
                self.accepted_states = signature.accepted_states
            if signature.parent_state and not self.parent_state:
                self.parent_state = signature.parent_state
        elif hasattr(signature, "signature") and isinstance(signature.signature, AlgebraicSignature):
            sig = signature.signature
            self.signature = AlgebraicSignature(
                type_name=sig.type_name,
                state=sig.state,
                qualifiers=sig.qualifiers,
                abstract_type=self.abstract_type or sig.abstract_type,
                accepted_states=self.accepted_states or sig.accepted_states,
                parent_state=self.parent_state or sig.parent_state,
                domain=eff_dom or sig.domain,
            )
            if self.signature.abstract_type:
                self.abstract_type = self.signature.abstract_type
        elif isinstance(signature, str):
            if "name" in kwargs:
                self.name = kwargs["name"]
                self.signature = AlgebraicSignature(
                    type_name=signature,
                    state=kwargs.get("state", "any"),
                    abstract_type=self.abstract_type,
                    accepted_states=self.accepted_states,
                    parent_state=self.parent_state,
                    domain=eff_dom,
                )
            elif signature != "any":
                self.name = kwargs.get("name", f"port_{name}")
                self.signature = AlgebraicSignature(
                    type_name=name,
                    state=signature,
                    abstract_type=self.abstract_type,
                    accepted_states=self.accepted_states,
                    parent_state=self.parent_state,
                    domain=eff_dom,
                )
            else:
                self.signature = AlgebraicSignature(
                    type_name=name if name else "any",
                    state="any",
                    abstract_type=self.abstract_type,
                    accepted_states=self.accepted_states,
                    parent_state=self.parent_state,
                    domain=eff_dom,
                )
        else:
            self.signature = AlgebraicSignature(
                "any", "any",
                abstract_type=self.abstract_type,
                accepted_states=self.accepted_states,
                parent_state=self.parent_state,
                domain=eff_dom,
            )

        self.required = bool(required)
        if self.required and default_value == "":
            # An empty string cannot act as a default value for a required data_input or
            # non-string carrier (e.g. malformed tree schemas declaring default_value: "" on bytes).
            _is_data_in = str(self.port_role or "").strip().lower() == "data_input"
            _is_str = TypeRegistry.get_instance().has_type_kind(self.signature.type_name, "string")
            if _is_data_in or not _is_str:
                default_value = None
        self.default_value = default_value
        self.doc = str(doc or "")
        self._cached_term = None
        self._cached_col_proj = None
        self._cached_derived_role = None
        self._cached_is_path_port = None
        self._cached_ndim = None
        self._cached_rank_cache = None
        if self.carrier is None and self.signature.type_name:
            self.carrier = self.signature.type_name
        dom_low = str(self.domain or "").strip().lower()
        st_low = str(self.signature.state or "").strip().lower()
        if self.typestate is None and st_low:
            self.typestate = f"{dom_low}::{st_low}" if (dom_low and "::" not in st_low and st_low not in ("any", "*", "default")) else st_low
        if self.semantics is None and st_low:
            self.semantics = "any" if st_low in ("any", "*", "default") else (f"{dom_low}.{st_low}" if (dom_low and "." not in st_low and "::" not in st_low) else st_low)

    def with_signature(self, new_signature: AlgebraicSignature) -> "PortSignature":
        """Return a copy of this PortSignature with `new_signature`, preserving all 5-axis and effect fields."""
        return PortSignature(
            name=self.name,
            signature=new_signature,
            required=self.required,
            default_value=self.default_value,
            doc=self.doc,
            domain=self.domain,
            abstract_type=self.abstract_type,
            enum_values=self.enum_values,
            param_kind=self.param_kind,
            value_constraints=self.value_constraints,
            shape_contract=self.shape_contract,
            rejected_qualifiers=self.rejected_qualifiers,
            accepted_states=self.accepted_states,
            parent_state=self.parent_state,
            port_role=self.port_role,
            binds=self.binds,
            structure_kind=self.structure_kind,
            rank=self.rank,
            element_type=self.element_type,
            supported_formats=self.supported_formats,
            complement_of_port=self.complement_of_port,
            disjoint_from_port=self.disjoint_from_port,
            key_from_upstream_role=self.key_from_upstream_role,
            carrier=self.carrier,
            nominal_type=self.nominal_type,
            layout=(dict(self.layout) if isinstance(self.layout, dict) else self.layout),
            semantics=self.semantics,
            typestate=self.typestate,
            invariants=self.invariants,
            effect_suspended=self.effect_suspended,
            mutates=self.mutates,
            key_of=self.key_of,
            key_effect=self.key_effect,
            partition_index=self.partition_index,
            partition=self.partition,
            owner_cell=self.owner_cell,
        )

    @property
    def type_name(self) -> str:
        return self.signature.type_name

    @property
    def state(self) -> str:
        return self.signature.state

    @property
    def description(self) -> str:
        return self.doc or ""

    _DERIVED_ROLE_MEMO: Dict[Tuple[int, str, str], str] = {}

    @property
    def derived_role(self) -> str:
        """
        Role resolution is DECLARED-DATA-FIRST:
          1. the port's declared `port_role` (tree JSON field),
          2. the declared typestate ancestry of its state, per the tree-declared
             `state_role_ancestors` map (role -> ancestor states),
          3. the declared carrier type, per the tree-declared `carrier_roles` map.
        No port-name conventions: naming is data, typing is semantics.
        """
        if self._cached_derived_role is not None:
            return self._cached_derived_role

        if self.port_role:
            self._cached_derived_role = self.port_role
            return self.port_role

        state_lower = str(getattr(self.signature, "state", "")).lower()
        t_name = self.type_name.lower()
        registry = TypeRegistry.get_instance()
        memo_key = (registry._generation, state_lower, t_name)
        memo_hit = PortSignature._DERIVED_ROLE_MEMO.get(memo_key)
        if memo_hit is not None:
            self._cached_derived_role = memo_hit
            return memo_hit

        # Declared state ancestry -> role families (tree JSON: `state_role_ancestors`,
        # `state_role_markers`, `carrier_roles`; earlier declarations win).
        role = "standard"
        if state_lower and state_lower not in ("any", "*"):
            for role_name, ancestors in registry._state_role_ancestors.items():
                if registry.state_ancestry_reaches(state_lower, ancestors):
                    role = role_name
                    break
            else:
                for role_name, markers in registry._state_role_markers.items():
                    if any(m in state_lower for m in markers):
                        role = role_name
                        break

        if role == "standard":
            for role_name, carriers in registry._carrier_roles.items():
                if any(registry.is_subtype(t_name, c) for c in carriers):
                    role = role_name
                    break

        if len(PortSignature._DERIVED_ROLE_MEMO) >= 16384:
            PortSignature._DERIVED_ROLE_MEMO.clear()
        PortSignature._DERIVED_ROLE_MEMO[memo_key] = role
        self._cached_derived_role = role
        return role

    @property
    def canonical_rank(self) -> Optional[int]:
        if self._cached_ndim is not None:
            return self._cached_ndim

        reg = TypeRegistry.get_instance()
        gen = reg._generation
        sc = self.shape_contract
        rk = self.rank
        rc = self._cached_rank_cache
        if rc is not None and rc[0] == gen and rc[1] is sc and rc[2] == rk:
            return rc[3]

        if isinstance(sc, dict):
            for k in ("ndim", "rank", "dimensions"):
                if k in sc:
                    try:
                        r = int(sc[k])
                        self._cached_ndim = r
                        self._cached_rank_cache = (gen, sc, rk, r)
                        return r
                    except (ValueError, TypeError):
                        pass
        elif isinstance(sc, str):
            s_sc = sc.strip()
            if (s_sc.startswith("(") and s_sc.endswith(")")) or (s_sc.startswith("[") and s_sc.endswith("]")):
                inner_sc = s_sc[1:-1].strip()
                if not inner_sc:
                    self._cached_ndim = 0
                    self._cached_rank_cache = (gen, sc, rk, 0)
                    return 0
                parts_sc = [p.strip() for p in inner_sc.split(",") if p.strip()]
                if not any(p in ("*", "...", "any") for p in parts_sc):
                    r = len(parts_sc)
                    self._cached_ndim = r
                    self._cached_rank_cache = (gen, sc, rk, r)
                    return r

        if rk is not None:
            try:
                r = int(rk)
                self._cached_ndim = r
                self._cached_rank_cache = (gen, sc, rk, r)
                return r
            except (ValueError, TypeError):
                pass

        if reg:
            t_name = str(self.type_name).lower()
            # Atomic scalar: rank 0
            if reg.is_subtype(t_name, "scalar"):
                self._cached_ndim = 0
                self._cached_rank_cache = (gen, sc, rk, 0)
                return 0

        # If state or accepted_states is rank-polymorphic / generic, rank is polymorphic (None)
        sig = self.signature
        state_str = str(getattr(sig, "state", "") or "").strip().lower()
        if reg:
            if reg.is_declared_top_state(state_str) or bool(reg._get_state_properties_ref(state_str).get("is_rank_polymorphic")):
                self._cached_rank_cache = (gen, sc, rk, None)
                return None

        acc_states = getattr(sig, "accepted_states", None) or self.accepted_states
        if reg and acc_states and any(reg.is_declared_top_state(s) or bool(reg._get_state_properties_ref(s).get("is_rank_polymorphic")) for s in acc_states):
            self._cached_rank_cache = (gen, sc, rk, None)
            return None

        self._cached_rank_cache = (gen, sc, rk, None)
        return None

    @canonical_rank.setter
    def canonical_rank(self, val: Optional[int]) -> None:
        self._cached_ndim = val
        self._cached_rank_cache = None
        if val is not None:
            if not isinstance(self.shape_contract, dict):
                self.shape_contract = {}
            self.shape_contract["ndim"] = val
            self.shape_contract["rank"] = val

    def is_top(self) -> bool:
        return self.signature.is_top()

    def unifies_with(self, other: Any) -> bool:
        if isinstance(other, PortSignature):
            p_nom = str(getattr(self, "nominal_type", "") or "").strip().lower()
            c_nom = str(getattr(other, "nominal_type", "") or "").strip().lower()
            if p_nom and c_nom and p_nom != c_nom:
                if not TypeRegistry.get_instance().is_subtype(p_nom, c_nom):
                    return False
            return self.signature.unifies_with(other.signature)
        if isinstance(other, AlgebraicSignature):
            return self.signature.unifies_with(other)
        return False

    def __repr__(self) -> str:
        abs_info = f", abs={self.abstract_type}" if self.abstract_type else ""
        acc_info = f", acc={list(self.accepted_states)}" if self.accepted_states else ""
        return f"Port({self.name}: {self.signature.type_name}[{self.signature.state}]{abs_info}{acc_info}, req={self.required})"


class InvariantEntailer:
    """
    Domain-agnostic decidable QF_LIA (interval + excluded values) + equality/set
    invariant entailment engine for Axis 5 (Phi_out |= Phi_in). Uses stdlib AST
    parsing and string methods (zero regular expressions).
    """

    @staticmethod
    def _is_num(val: Any) -> bool:
        return isinstance(val, (int, float)) and not isinstance(val, bool)

    @classmethod
    def _norm_val(cls, val: Any) -> Any:
        if cls._is_num(val):
            return float(val)
        if isinstance(val, str):
            return val.strip().lower()
        return val

    @classmethod
    def _parse_scalar(cls, raw: str) -> Any:
        s = raw.strip()
        if not s:
            return ""
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            sl = s.lower()
            if sl == "true":
                return True
            if sl == "false":
                return False
            if sl == "none":
                return None
            return s

    @classmethod
    def _ast_lit(cls, node: ast.AST) -> Tuple[bool, Any]:
        if isinstance(node, ast.Constant):
            return True, node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            ok, inner = cls._ast_lit(node.operand)
            if ok and cls._is_num(inner):
                return True, (-inner if isinstance(node.op, ast.USub) else inner)
            return False, None
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            items: List[Any] = []
            for elt in node.elts:
                ok, iv = cls._ast_lit(elt)
                if not ok:
                    return False, None
                items.append(iv)
            if isinstance(node, ast.Set):
                return True, set(items)
            return True, tuple(items)
        if isinstance(node, ast.Name):
            low = node.id.lower()
            if low == "true":
                return True, True
            if low == "false":
                return True, False
            if low == "none":
                return True, None
            return True, node.id
        return False, None

    @staticmethod
    def _ast_prop_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id.strip().lower()
        try:
            return ast.unparse(node).strip().lower()
        except Exception:
            return str(node).strip().lower()

    @staticmethod
    def _cmpop_str(op_node: ast.cmpop) -> Optional[str]:
        if isinstance(op_node, (ast.Eq, ast.Is)):
            return "=="
        if isinstance(op_node, (ast.NotEq, ast.IsNot)):
            return "!="
        if isinstance(op_node, ast.Lt):
            return "<"
        if isinstance(op_node, ast.LtE):
            return "<="
        if isinstance(op_node, ast.Gt):
            return ">"
        if isinstance(op_node, ast.GtE):
            return ">="
        if isinstance(op_node, ast.In):
            return "in"
        if isinstance(op_node, ast.NotIn):
            return "not in"
        return None

    @staticmethod
    def _flip_op(op: str) -> str:
        if op == "<":
            return ">"
        if op == "<=":
            return ">="
        if op == ">":
            return "<"
        if op == ">=":
            return "<="
        return op

    @classmethod
    def extract_value_constraints(cls, vc: Any) -> List[Tuple[str, str, Any]]:
        if not vc:
            return []
        if isinstance(vc, dict):
            if "property" in vc:
                return cls.parse_predicate(vc)
            out: List[Tuple[str, str, Any]] = []
            for k, v in vc.items():
                kl = str(k).strip().lower()
                if kl in ("description", "note", "notes", "reason", "optional"):
                    continue
                if kl in ("min", "gte") and cls._is_num(v):
                    out.append(("value", ">=", v))
                elif kl in ("max", "lte") and cls._is_num(v):
                    out.append(("value", "<=", v))
                elif kl in ("gt", "exclusive_min") and cls._is_num(v):
                    out.append(("value", ">", v))
                elif kl in ("lt", "exclusive_max") and cls._is_num(v):
                    out.append(("value", "<", v))
                elif kl in ("neq", "not_equal"):
                    out.append(("value", "!=", v))
                elif kl in ("in", "allowed", "enum") and isinstance(v, (list, tuple, set)):
                    out.append(("value", "in", tuple(v)))
                else:
                    out.append((kl, "==", v))
            return out
        if isinstance(vc, (list, tuple)):
            out = []
            for item in vc:
                out.extend(cls.parse_predicate(item))
            return out
        return cls.parse_predicate(vc)

    @classmethod
    def parse_predicate(cls, raw: Any) -> List[Tuple[str, str, Any]]:
        """Parses structured or string invariants/conditions into (prop, op, val) triples."""
        if raw is None:
            return []
        if isinstance(raw, tuple) and len(raw) == 3:
            prop, op, val = raw
            op_map = {"is": "==", "is not": "!=", "eq": "==", "ne": "!=", "gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
            op_s = op_map.get(str(op).strip().lower(), str(op).strip().lower())
            return [(str(prop).strip().lower(), op_s, val)]
        if isinstance(raw, dict):
            prop = raw.get("property")
            if prop:
                op = str(raw.get("operator", "==")).strip().lower()
                return [(str(prop).strip().lower(), op, raw.get("value"))]
            if raw.get("expression"):
                return cls.parse_predicate(raw["expression"])
            return cls.extract_value_constraints(raw)
        if hasattr(raw, "property"):
            prop = getattr(raw, "property", None)
            if prop:
                op = str(getattr(raw, "operator", "==") or "==").strip().lower()
                return [(str(prop).strip().lower(), op, getattr(raw, "value", None))]
            expr = getattr(raw, "expression", None)
            if expr:
                return cls.parse_predicate(expr)
            return []

        s = str(raw).strip()
        if not s:
            return []

        # Single '=' tree shorthand: min=0, max=1.0, gt=0.0, dtype=uint8, nonzero=True
        if (
            "=" in s
            and s.count("=") == 1
            and "==" not in s
            and "!=" not in s
            and "<=" not in s
            and ">=" not in s
        ):
            lhs, rhs = s.split("=", 1)
            lk = lhs.strip().lower()
            if lk and all(ch.isalnum() or ch in "._" for ch in lk):
                rv = cls._parse_scalar(rhs)
                if lk in ("min", "gte") and cls._is_num(rv):
                    return [("value", ">=", rv)]
                if lk in ("max", "lte") and cls._is_num(rv):
                    return [("value", "<=", rv)]
                if lk == "gt" and cls._is_num(rv):
                    return [("value", ">", rv)]
                if lk == "lt" and cls._is_num(rv):
                    return [("value", "<", rv)]
                return [(lk, "==", rv)]

        has_cmp = any(tok in s for tok in ("==", "!=", "<=", ">=", "<", ">", " in ", " not in ", " is "))
        if has_cmp:
            try:
                tree = ast.parse(s, mode="eval").body
                parsed: List[Tuple[str, str, Any]] = []

                def _walk_expr(node: ast.AST) -> bool:
                    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
                        return all(_walk_expr(v) for v in node.values)
                    if isinstance(node, ast.Compare):
                        left = node.left
                        for op_node, right in zip(node.ops, node.comparators):
                            op_s = cls._cmpop_str(op_node)
                            if op_s is None:
                                return False
                            r_ok, r_val = cls._ast_lit(right)
                            l_ok, l_val = cls._ast_lit(left)
                            if isinstance(left, (ast.Name, ast.Attribute, ast.Subscript)) and r_ok and not isinstance(right, ast.Name):
                                parsed.append((cls._ast_prop_name(left), op_s, r_val))
                            elif isinstance(left, ast.Name) and isinstance(right, ast.Name):
                                parsed.append((left.id.strip().lower(), op_s, right.id))
                            elif isinstance(left, (ast.Attribute, ast.Subscript)) and r_ok:
                                parsed.append((cls._ast_prop_name(left), op_s, r_val))
                            elif l_ok and isinstance(right, (ast.Name, ast.Attribute, ast.Subscript)):
                                parsed.append((cls._ast_prop_name(right), cls._flip_op(op_s), l_val))
                            else:
                                return False
                            left = right
                        return True
                    return False

                if _walk_expr(tree) and parsed:
                    return parsed
            except SyntaxError:
                pass

        return [(s.lower(), "tag", True)]

    @classmethod
    def _compile_domains(cls, preds: Sequence[Any]) -> Tuple[bool, Dict[str, Dict[str, Any]], Set[str]]:
        domains: Dict[str, Dict[str, Any]] = {}
        tags: Set[str] = set()
        contradictory = False

        def _dom(prop: str) -> Dict[str, Any]:
            d = domains.get(prop)
            if d is None:
                d = {
                    "lo": None,
                    "lo_inc": True,
                    "hi": None,
                    "hi_inc": True,
                    "excluded_nums": set(),
                    "eq_set": False,
                    "eq_val": None,
                    "neq_vals": set(),
                    "allowed_set": None,
                    "contradictory": False,
                }
                domains[prop] = d
            return d

        for raw in preds or ():
            for prop, op, val in cls.parse_predicate(raw):
                if op == "tag":
                    tags.add(prop)
                    continue
                d = _dom(prop)
                if op == "==":
                    if cls._is_num(val):
                        fv = float(val)
                        if d["lo"] is None or fv > d["lo"]:
                            d["lo"], d["lo_inc"] = fv, True
                        if d["hi"] is None or fv < d["hi"]:
                            d["hi"], d["hi_inc"] = fv, True
                        if d["eq_set"] and cls._norm_val(d["eq_val"]) != fv:
                            d["contradictory"] = True
                        d["eq_set"] = True
                        d["eq_val"] = fv
                    else:
                        nv = cls._norm_val(val)
                        if d["eq_set"] and d["eq_val"] != nv:
                            d["contradictory"] = True
                        d["eq_set"] = True
                        d["eq_val"] = nv
                elif op == "!=":
                    if cls._is_num(val):
                        d["excluded_nums"].add(float(val))
                    else:
                        d["neq_vals"].add(cls._norm_val(val))
                elif op == ">" and cls._is_num(val):
                    fv = float(val)
                    if d["lo"] is None or fv > d["lo"]:
                        d["lo"], d["lo_inc"] = fv, False
                    elif fv == d["lo"]:
                        d["lo_inc"] = False
                elif op == ">=" and cls._is_num(val):
                    fv = float(val)
                    if d["lo"] is None or fv > d["lo"]:
                        d["lo"], d["lo_inc"] = fv, True
                elif op == "<" and cls._is_num(val):
                    fv = float(val)
                    if d["hi"] is None or fv < d["hi"]:
                        d["hi"], d["hi_inc"] = fv, False
                    elif fv == d["hi"]:
                        d["hi_inc"] = False
                elif op == "<=" and cls._is_num(val):
                    fv = float(val)
                    if d["hi"] is None or fv < d["hi"]:
                        d["hi"], d["hi_inc"] = fv, True
                elif op == "in" and isinstance(val, (list, tuple, set)):
                    norm_s = {cls._norm_val(x) for x in val}
                    d["allowed_set"] = norm_s if d["allowed_set"] is None else (d["allowed_set"] & norm_s)
                elif op == "not in" and isinstance(val, (list, tuple, set)):
                    for x in val:
                        if cls._is_num(x):
                            d["excluded_nums"].add(float(x))
                        else:
                            d["neq_vals"].add(cls._norm_val(x))

        for d in domains.values():
            lo, hi = d["lo"], d["hi"]
            if lo is not None and hi is not None:
                if lo > hi:
                    d["contradictory"] = True
                elif lo == hi and (not d["lo_inc"] or not d["hi_inc"] or lo in d["excluded_nums"]):
                    d["contradictory"] = True
            if d["eq_set"]:
                ev = d["eq_val"]
                if cls._is_num(ev):
                    fev = float(ev)
                    if fev in d["excluded_nums"]:
                        d["contradictory"] = True
                    if d["allowed_set"] is not None and fev not in d["allowed_set"]:
                        d["contradictory"] = True
                else:
                    if ev in d["neq_vals"]:
                        d["contradictory"] = True
                    if d["allowed_set"] is not None and ev not in d["allowed_set"]:
                        d["contradictory"] = True
            if d["allowed_set"] is not None and len(d["allowed_set"]) == 0:
                d["contradictory"] = True
            if d["contradictory"]:
                contradictory = True

        return contradictory, domains, tags

    @staticmethod
    def _num_satisfies(x: float, op: str, c: float) -> bool:
        if op == ">":
            return x > c
        if op == ">=":
            return x >= c
        if op == "<":
            return x < c
        if op == "<=":
            return x <= c
        if op == "==":
            return x == c
        if op == "!=":
            return x != c
        return False

    @classmethod
    def entails(
        cls,
        producer_preds: Sequence[Any],
        consumer_preds: Sequence[Any],
        registry: Optional["TypeRegistry"] = None,
    ) -> bool:
        """
        True iff producer_preds logically entail every requirement in consumer_preds
        (without contradictory producer assumptions).
        """
        reqs: List[Tuple[str, str, Any]] = []
        for raw in consumer_preds or ():
            reqs.extend(cls.parse_predicate(raw))
        if not reqs:
            return True

        contradictory, domains, tags = cls._compile_domains(producer_preds or ())
        if contradictory:
            return False

        for prop, op, val in reqs:
            if op == "tag":
                if prop in tags:
                    continue
                dom = domains.get(prop)
                if dom is not None and dom["eq_set"] and dom["eq_val"] is True:
                    continue
                if registry is not None and any(registry.state_ancestry_reaches(t, (prop,)) for t in tags):
                    continue
                return False

            dom = domains.get(prop)
            if dom is None:
                return False

            if op in (">", ">=", "<", "<=", "==", "!=") and cls._is_num(val):
                c = float(val)
                if dom["allowed_set"] is not None and dom["allowed_set"] and all(cls._is_num(x) for x in dom["allowed_set"]):
                    if all(cls._num_satisfies(float(x), op, c) for x in dom["allowed_set"]):
                        continue
                    # Also allow interval bounds below if tighter than allowed_set alone
                lo, lo_inc = dom["lo"], dom["lo_inc"]
                hi, hi_inc = dom["hi"], dom["hi_inc"]
                excl = dom["excluded_nums"]
                if op == ">":
                    ok = lo is not None and (lo > c or (lo == c and (not lo_inc or c in excl)))
                elif op == ">=":
                    ok = lo is not None and lo >= c
                elif op == "<":
                    ok = hi is not None and (hi < c or (hi == c and (not hi_inc or c in excl)))
                elif op == "<=":
                    ok = hi is not None and hi <= c
                elif op == "==":
                    ok = lo is not None and hi is not None and lo == c and hi == c and lo_inc and hi_inc and c not in excl
                else:  # !=
                    ok = (
                        c in excl
                        or (lo is not None and (lo > c or (lo == c and (not lo_inc or c in excl))))
                        or (hi is not None and (hi < c or (hi == c and (not hi_inc or c in excl))))
                    )
                if not ok:
                    return False
                continue

            if op == "==":
                nv = cls._norm_val(val)
                ok = (
                    (dom["eq_set"] and dom["eq_val"] == nv)
                    or (dom["allowed_set"] is not None and dom["allowed_set"] == {nv})
                )
                if not ok:
                    return False
            elif op == "!=":
                nv = cls._norm_val(val)
                ok = (
                    nv in dom["neq_vals"]
                    or (dom["eq_set"] and dom["eq_val"] != nv)
                    or (dom["allowed_set"] is not None and nv not in dom["allowed_set"])
                )
                if not ok:
                    return False
            elif op == "in" and isinstance(val, (list, tuple, set)):
                s_vals = {cls._norm_val(x) for x in val}
                ok = (
                    (dom["eq_set"] and dom["eq_val"] in s_vals)
                    or (dom["allowed_set"] is not None and dom["allowed_set"].issubset(s_vals))
                )
                if not ok:
                    return False
            elif op == "not in" and isinstance(val, (list, tuple, set)):
                s_vals = {cls._norm_val(x) for x in val}
                ok = (
                    (dom["eq_set"] and dom["eq_val"] not in s_vals)
                    or (dom["allowed_set"] is not None and dom["allowed_set"].isdisjoint(s_vals))
                    or s_vals.issubset(dom["neq_vals"])
                )
                if not ok:
                    return False
            else:
                return False

        return True


@dataclass(frozen=True, slots=True)
class AxisResidual:
    """
    5-Axis Residual Vector Delta(P_out, P_in) = (dC, dM, dSigma, dS, dPhi, nominal_conflict).
    Zero residual means direct zero-cost morphism is valid without bridges.
    Non-zero residual pinpoints the exact axes a bridge chain must resolve.
    """
    carrier_delta: bool = False
    layout_delta: bool = False
    semantics_delta: bool = False
    typestate_delta: bool = False
    invariants_delta: bool = False
    nominal_conflict: bool = False

    @property
    def is_zero(self) -> bool:
        return not (
            self.carrier_delta
            or self.layout_delta
            or self.semantics_delta
            or self.typestate_delta
            or self.invariants_delta
            or self.nominal_conflict
        )

    @property
    def failing_axes(self) -> Tuple[str, ...]:
        axes: List[str] = []
        if self.nominal_conflict:
            axes.append("nominal")
        if self.carrier_delta:
            axes.append("carrier")
        if self.layout_delta:
            axes.append("layout")
        if self.semantics_delta:
            axes.append("semantics")
        if self.typestate_delta:
            axes.append("typestate")
        if self.invariants_delta:
            axes.append("invariants")
        return tuple(axes)


def compute_port_residual(
    p_out: Any,
    p_in: Any,
    registry: Optional[TypeRegistry] = None,
    producer_cell: Optional[Any] = None,
    consumer_cell: Optional[Any] = None,
) -> AxisResidual:
    """
    Computes the 5-Axis Product Lattice residual Delta(P_out, P_in) between a producer
    output port and a consumer input port, orthogonalizing:
      C (carrier + nominal identity), M (layout rank/dtype/shape), Sigma (domain semantics),
      S (typestate/lifecycle), and Phi (invariants/value constraints).
    """
    reg = registry or TypeRegistry.get_instance()
    p_sig = getattr(p_out, "signature", p_out)
    c_sig = getattr(p_in, "signature", p_in)

    # Axis 1a: Nominal class identity conflict (e.g. Lasso vs Ridge on method receiver/sink)
    p_nom = str(getattr(p_out, "nominal_type", "") or "").strip().lower()
    c_nom = str(getattr(p_in, "nominal_type", "") or "").strip().lower()
    nominal_conflict = False
    if p_nom and c_nom and p_nom != c_nom:
        if not reg.is_subtype(p_nom, c_nom):
            nominal_conflict = True

    # Axis 1b: Carrier subtyping (C_out <= C_in) with full recursive variance
    p_car = str(getattr(p_out, "carrier", None) or getattr(p_sig, "type_name", "any") or "any").strip()
    c_car = str(getattr(p_in, "carrier", None) or getattr(c_sig, "type_name", "any") or "any").strip()
    both_path = is_path_port(p_out, reg) and is_path_port(p_in, reg)
    carrier_delta = False
    if not (both_path or reg.is_declared_top(c_car) or reg.is_type_variable(c_car)):
        if reg.is_declared_top(p_car):
            carrier_delta = True
        elif not LatticeType.from_str(p_car).is_subtype_of(
            LatticeType.from_str(c_car), reg._atomic_is_subtype, is_top_fn=reg.is_declared_top
        ):
            p_abs = str(getattr(p_out, "abstract_type", "") or getattr(p_sig, "abstract_type", "") or "").strip().lower()
            c_abs = str(getattr(p_in, "abstract_type", "") or getattr(c_sig, "abstract_type", "") or "").strip().lower()
            c_is_abs = reg.is_abstract_carrier(c_car.lower()) or (c_car.lower() == c_abs)
            if not (c_is_abs and p_abs and c_abs and reg.is_subtype(p_abs, c_abs)):
                carrier_delta = True

    # Axis 2: Algebraic Memory Layout (M_out <= M_in)
    def _is_poly_dtype(raw_dt: Any) -> bool:
        if raw_dt is None:
            return True
        raw_s = str(raw_dt).strip()
        dt = raw_s.lower()
        if not dt or reg.is_declared_top(dt) or reg.is_type_variable(raw_s):
            return True
        if dt in ("opaque", "composite_record", "object") or dt in reg.get_types_by_kind("opaque"):
            return True
        return False

    layout_delta = False
    c_lay = getattr(p_in, "layout", None)
    p_lay = getattr(p_out, "layout", None)
    if isinstance(c_lay, dict) and isinstance(p_lay, dict):
        c_rk = c_lay.get("rank")
        p_rk = p_lay.get("rank")
        c_poly = bool(c_lay.get("rank_polymorphic")) or (isinstance(c_rk, str) and not c_rk.isdigit())
        if not c_poly and isinstance(c_rk, int) and not isinstance(c_rk, bool) and isinstance(p_rk, int) and not isinstance(p_rk, bool) and p_rk != c_rk:
            layout_delta = True

        c_ch = c_lay.get("channel_count")
        p_ch = p_lay.get("channel_count")
        if c_ch is not None and p_ch is not None and c_ch != p_ch:
            layout_delta = True

        c_ax = c_lay.get("channel_axis")
        p_ax = p_lay.get("channel_axis")
        if isinstance(c_ax, int) and not isinstance(c_ax, bool) and isinstance(p_ax, int) and not isinstance(p_ax, bool) and p_ax != c_ax:
            layout_delta = True

        raw_p_dt = p_lay.get("dtype")
        raw_c_dt = c_lay.get("dtype")
        p_dt = str(raw_p_dt or "").strip().lower()
        c_dt = str(raw_c_dt or "").strip().lower()
        c_dt_poly = _is_poly_dtype(raw_c_dt)
        p_dt_poly = _is_poly_dtype(raw_p_dt)
        if not c_dt_poly and not p_dt_poly and p_dt != c_dt:
            if not reg.is_subtype(p_dt, c_dt):
                layout_delta = True

        p_ord = str(p_lay.get("memory_order") or "any").strip().lower()
        c_ord = str(c_lay.get("memory_order") or "any").strip().lower()
        if p_ord != "any" and c_ord != "any" and p_ord != c_ord:
            layout_delta = True

        p_len = p_lay.get("length")
        c_len = c_lay.get("length")
        if isinstance(p_len, int) and not isinstance(p_len, bool) and isinstance(c_len, int) and not isinstance(c_len, bool) and p_len != c_len:
            layout_delta = True

        p_sh = p_lay.get("shape")
        c_sh = c_lay.get("shape")
        if isinstance(p_sh, (list, tuple)) and isinstance(c_sh, (list, tuple)) and len(p_sh) == len(c_sh):
            for pd, cd in zip(p_sh, c_sh):
                if isinstance(pd, int) and not isinstance(pd, bool) and isinstance(cd, int) and not isinstance(cd, bool) and pd != cd:
                    layout_delta = True
                    break

    # Axis 3 & 4: Domain-scoped Typestate (S) & Semantics (Sigma)
    p_dom = str(getattr(producer_cell, "domain_name", "") or getattr(p_out, "domain", "") or "").strip().lower() or None
    c_dom = str(getattr(consumer_cell, "domain_name", "") or getattr(p_in, "domain", "") or "").strip().lower() or None
    c_sem = str(getattr(p_in, "semantics", "") or "").strip().lower()
    p_sem = str(getattr(p_out, "semantics", "") or "").strip().lower()
    st_ok = True if both_path else reg.is_state_compatible(
        producer_state=getattr(p_sig, "state", "any"),
        consumer_state=getattr(c_sig, "state", "any"),
        producer_accepted=getattr(p_out, "accepted_states", ()) or getattr(p_sig, "accepted_states", ()),
        consumer_accepted=getattr(p_in, "accepted_states", ()) or getattr(c_sig, "accepted_states", ()),
        consumer_parent=getattr(p_in, "parent_state", None) or getattr(c_sig, "parent_state", None),
        producer_parent=getattr(p_out, "parent_state", None) or getattr(p_sig, "parent_state", None),
        producer_domain=p_dom,
        consumer_domain=c_dom,
    )
    typestate_delta = not st_ok

    def _is_top_sem(sem_val: str) -> bool:
        return (not sem_val) or reg.is_declared_top(sem_val)

    def _is_universal_top_state(st_val: str) -> bool:
        return (not st_val) or (st_val in _TOP_STATE_LITERALS) or reg.is_declared_top(st_val)

    p_state_l = str(getattr(p_sig, "state", "") or "").strip().lower()
    c_state_l = str(getattr(c_sig, "state", "") or "").strip().lower()
    c_acc = tuple(getattr(p_in, "accepted_states", ()) or getattr(c_sig, "accepted_states", ()) or ())
    c_sem_is_state_mirror = (
        not _is_universal_top_state(c_state_l)
        and (c_sem == c_state_l or (bool(c_dom) and c_sem == f"{c_dom}.{c_state_l}"))
    )
    p_sem_is_state_mirror = (
        not _is_universal_top_state(p_state_l)
        and (p_sem == p_state_l or (bool(p_dom) and p_sem == f"{p_dom}.{p_state_l}"))
    )

    if both_path or _is_top_sem(c_sem):
        semantics_delta = False
    elif (not _is_top_sem(p_sem)) and (
        p_sem == c_sem
        or p_sem.startswith(c_sem + ".")
        or reg.is_subtype(p_sem, c_sem)
        or reg.state_ancestry_reaches(p_sem, (c_sem,))
    ):
        # Rule A: Direct semantic subsumption
        semantics_delta = False
    else:
        p_non_univ = not _is_universal_top_state(p_state_l)
        c_non_univ = not _is_universal_top_state(c_state_l)
        p_reaches_acc = (
            bool(c_acc)
            and p_non_univ
            and (p_state_l in c_acc or reg.state_ancestry_reaches(p_state_l, c_acc))
        )
        p_reaches_c_state = (
            p_non_univ
            and c_non_univ
            and (p_state_l == c_state_l or reg.state_ancestry_reaches(p_state_l, (c_state_l,)))
        )
        if st_ok and p_reaches_acc:
            # Rule B: Consumer explicitly widened port acceptance via `accepted_states`
            semantics_delta = False
        elif st_ok and p_reaches_c_state and (
            p_sem_is_state_mirror or (_is_top_sem(p_sem) and c_sem_is_state_mirror)
        ):
            # Rule C: Proven typestate subsumption (`s_p <= s_c`) when producer carries
            # state-mirrored (or top-with-mirrored-consumer) semantics
            semantics_delta = False
        elif (
            st_ok
            and c_sem_is_state_mirror
            and p_sem_is_state_mirror
            and p_dom
            and c_dom
            and p_dom == c_dom
        ):
            # Rule D: Intra-domain property-subsumed state mirrors (must hold WITHOUT
            # wildcard top states in accepted_states so terminal sinks cannot bypass
            # semantics via generic carrier wildcards such as ndarray_generic).
            c_acc_non_top = tuple(s for s in c_acc if not reg.is_declared_top_state(s))
            st_ok_non_top = (
                st_ok
                if len(c_acc_non_top) == len(c_acc)
                else reg.is_state_compatible(
                    p_state_l,
                    c_state_l,
                    producer_accepted=getattr(p_out, "accepted_states", ()) or getattr(p_sig, "accepted_states", ()) or (),
                    consumer_accepted=c_acc_non_top,
                    consumer_parent=getattr(p_in, "parent_state", None) or getattr(c_sig, "parent_state", None),
                    producer_parent=getattr(p_out, "parent_state", None) or getattr(p_sig, "parent_state", None),
                    producer_domain=p_dom,
                    consumer_domain=c_dom,
                )
            )
            semantics_delta = not st_ok_non_top
        else:
            semantics_delta = True

    # Axis 5: Mathematical Invariants / Refinements (Phi_out |= Phi_in)
    if both_path:
        invariants_delta = False
    else:
        adv_flat = {
            q.strip().lower() if isinstance(q, str) else ".".join(str(x).strip().lower() for x in q)
            for q in getattr(reg, "_advisory_qualifiers", set())
        }
        p_inv_raw = getattr(p_out, "invariants", ()) or ()
        c_inv_raw = getattr(p_in, "invariants", ()) or ()
        p_inv_list: List[Any] = [
            i for i in p_inv_raw
            if str(i).strip().lower() not in adv_flat and not reg.is_abstract_carrier(str(i))
        ]
        c_inv_list: List[Any] = [
            i for i in c_inv_raw
            if str(i).strip().lower() not in adv_flat and not reg.is_abstract_carrier(str(i))
        ]
        p_vc = getattr(p_out, "value_constraints", None)
        c_vc = getattr(p_in, "value_constraints", None)
        if p_vc:
            p_inv_list.extend(InvariantEntailer.extract_value_constraints(p_vc))
        if c_vc:
            c_inv_list.extend(InvariantEntailer.extract_value_constraints(c_vc))
        if isinstance(p_lay, dict):
            raw_lay_dt = p_lay.get("dtype")
            lay_dt = str(raw_lay_dt or "").strip().lower()
            if lay_dt and not _is_poly_dtype(raw_lay_dt):
                p_inv_list.append(("dtype", "==", lay_dt))
            lay_ch = p_lay.get("channel_count")
            if isinstance(lay_ch, int) and not isinstance(lay_ch, bool):
                p_inv_list.append(("channels", "==", lay_ch))
        invariants_delta = bool(c_inv_list) and not InvariantEntailer.entails(
            p_inv_list, c_inv_list, registry=reg
        )

    return AxisResidual(
        carrier_delta=carrier_delta,
        layout_delta=layout_delta,
        semantics_delta=semantics_delta,
        typestate_delta=typestate_delta,
        invariants_delta=invariants_delta,
        nominal_conflict=nominal_conflict,
    )


def apply_polymorphic_propagation(
    out_port: Any,
    src_port: Any,
    pp_spec: Optional[Dict[str, Any]],
    registry: Optional[TypeRegistry] = None,
) -> Any:
    """Propagate concrete typestate, semantics, and unspecified layout slots across a polymorphic functor port."""
    if src_port is None or not isinstance(pp_spec, dict):
        return out_port
    reg = registry or TypeRegistry.get_instance()
    res = out_port
    copied = False

    if pp_spec.get("preserves_typestate"):
        src_sig_inner = getattr(src_port, "signature", src_port)
        src_state = getattr(src_sig_inner, "state", None)
        src_state_l = str(src_state or "").strip().lower()
        cur_sig_inner = getattr(res, "signature", res)
        cur_state_l = str(getattr(cur_sig_inner, "state", "") or "").strip().lower()
        term_states = set(reg.get_terminal_states() or ()) | reg.get_states_by_property("is_terminal", True)
        if (
            src_state
            and src_state_l not in ("", "any", "default")
            and not reg.is_declared_top_state(src_state_l)
            and cur_state_l != src_state_l
            and cur_state_l not in term_states
        ):
            new_parent_st = getattr(cur_sig_inner, "parent_state", None)
            if not new_parent_st and getattr(src_sig_inner, "parent_state", None):
                new_parent_st = src_sig_inner.parent_state
            new_alg = AlgebraicSignature(
                type_name=getattr(cur_sig_inner, "type_name", "any"),
                state=src_state,
                qualifiers=getattr(cur_sig_inner, "qualifiers", ()),
                abstract_type=getattr(cur_sig_inner, "abstract_type", ""),
                accepted_states=getattr(cur_sig_inner, "accepted_states", ()),
                parent_state=new_parent_st,
                domain=getattr(cur_sig_inner, "domain", ""),
            )
            src_ts = getattr(src_port, "typestate", None)
            src_sem_pt = getattr(src_port, "semantics", None)
            cur_dom_l = str(getattr(res, "domain", "") or getattr(cur_sig_inner, "domain", "") or "").strip().lower()
            cur_sem_l = str(getattr(res, "semantics", "") or "").strip().lower()
            cur_sem_is_state_derived = (
                not cur_sem_l
                or cur_sem_l == "any"
                or reg.is_declared_top(cur_sem_l)
                or cur_sem_l == cur_state_l
                or (cur_dom_l and cur_sem_l == f"{cur_dom_l}.{cur_state_l}")
            )
            if isinstance(res, PortSignature):
                res = res.with_signature(new_alg)
                copied = True
                if src_ts:
                    res.typestate = src_ts
                if src_sem_pt and cur_sem_is_state_derived:
                    res.semantics = src_sem_pt
            elif hasattr(res, "signature") and res.signature is not res:
                if not copied:
                    res = copy.copy(res)
                    copied = True
                res.signature = new_alg
                if hasattr(res, "_cached_term"):
                    res._cached_term = None
                if hasattr(res, "_cached_derived_role"):
                    res._cached_derived_role = None
                if src_ts and hasattr(res, "typestate"):
                    res.typestate = src_ts
                if src_sem_pt and cur_sem_is_state_derived and hasattr(res, "semantics"):
                    res.semantics = src_sem_pt
            else:
                res = new_alg
                copied = True

    if pp_spec.get("preserves_semantics"):
        cur_sem = str(getattr(res, "semantics", "") or "").strip().lower()
        if not cur_sem or cur_sem == "any" or reg.is_declared_top(cur_sem):
            src_sem = getattr(src_port, "semantics", None)
            src_sem_l = str(src_sem or "").strip().lower()
            if src_sem and src_sem_l != "any" and not reg.is_declared_top(src_sem_l):
                if not copied:
                    res = copy.copy(res)
                    copied = True
                res.semantics = src_sem

    if pp_spec.get("preserves_layout") and isinstance(getattr(src_port, "layout", None), dict):
        src_lay = src_port.layout
        cur_lay = getattr(res, "layout", None)

        def _is_uninst_dtype(dt: Any) -> bool:
            if dt is None:
                return True
            dt_s = str(dt).strip()
            return dt_s.lower() in ("", "any") or reg.is_type_variable(dt_s)

        if not isinstance(cur_lay, dict):
            if not copied:
                res = copy.copy(res)
                copied = True
            res.layout = dict(src_lay)
            if isinstance(src_lay.get("rank"), int) and not isinstance(src_lay.get("rank"), bool) and getattr(res, "rank", None) is None:
                res.rank = src_lay["rank"]
                for _c_attr in ("_cached_rank_cache", "_cached_ndim"):
                    if hasattr(res, _c_attr):
                        setattr(res, _c_attr, None)
        else:
            new_lay = dict(cur_lay)
            if cur_lay.get("rank") is None or isinstance(cur_lay.get("rank"), str) or bool(cur_lay.get("rank_polymorphic")):
                if isinstance(src_lay.get("rank"), int) and not isinstance(src_lay.get("rank"), bool):
                    new_lay["rank"] = src_lay["rank"]
                    new_lay["rank_polymorphic"] = False
            if _is_uninst_dtype(cur_lay.get("dtype")) and not _is_uninst_dtype(src_lay.get("dtype")):
                new_lay["dtype"] = src_lay["dtype"]
            if cur_lay.get("shape") is None and src_lay.get("shape") is not None:
                new_lay["shape"] = list(src_lay["shape"]) if isinstance(src_lay["shape"], (list, tuple)) else src_lay["shape"]
            for _lk in ("channel_count", "channel_axis"):
                if cur_lay.get(_lk) is None and src_lay.get(_lk) is not None:
                    new_lay[_lk] = src_lay[_lk]
            if cur_lay.get("shape") is None and cur_lay.get("length") is None and src_lay.get("length") is not None:
                new_lay["length"] = src_lay["length"]
            if str(cur_lay.get("memory_order") or "any").strip().lower() == "any" and str(src_lay.get("memory_order") or "any").strip().lower() != "any":
                new_lay["memory_order"] = src_lay["memory_order"]
            if new_lay != cur_lay:
                if not copied:
                    res = copy.copy(res)
                res.layout = new_lay
                if isinstance(new_lay.get("rank"), int) and not isinstance(new_lay.get("rank"), bool) and getattr(res, "rank", None) is None:
                    res.rank = new_lay["rank"]
                    for _c_attr in ("_cached_rank_cache", "_cached_ndim"):
                        if hasattr(res, _c_attr):
                            setattr(res, _c_attr, None)

    return res


class Cell(ABC):
    """
    Mathematical Lattice Node: v = (f, tau_in, tau_out).
    Independent of domain, language, or execution runtime.
    """
    __slots__ = [
        "cell_id", "stage", "keywords", "cell_type",
        "inputs", "outputs", "slots", "domain_name", "node_type", "node_role",
        "dependencies", "code_template", "metadata_tags",
        "configuration_schema", "verified", "semantic_tags",
        "docstring", "enrichment_source", "enriched_at",
        "source_priority", "source_provenance", "is_public",
        "topology_type", "feedback_state_type", "bound_slots",
        "replica_of", "replica_role",
        "mutation_type", "is_context_manager", "raises", "type_vars",
        "preconditions", "postconditions", "effects", "edges", "endable",
        "primary_in", "primary_out",
        "relations", "bridge_profile", "polymorphic_propagation",
        "_primary_input", "_primary_output", "_token_set", "_token_count",
        "_identity_tokens", "bound_parent_ids", "matched_clause_idx",
        "clause_literals", "fixture_needs", "projection", "typed_bindings", "_propagated_outputs",
        "planned_port_wires", "is_typed_hole", "task_description",
        "optic_role", "container_port", "key_port", "focus_port",
        "op_name", "_canonical_op_names", "_formal_identifiers",
        "explicit_port_bindings", "formal_program_step", "_auto_inserted_bridge",
        "_is_prerequisite_producer",
    ]

    def __init__(
        self,
        cell_id: str,
        stage: int = 2,
        keywords: Optional[Union[Set[str], List[str]]] = None,
        cell_type: str = "micro",
        inputs: Optional[Dict[str, Union[PortSignature, AlgebraicSignature, dict]]] = None,
        outputs: Optional[Dict[str, Union[PortSignature, AlgebraicSignature, dict]]] = None,
        slots: Optional[Dict[str, Any]] = None,
        domain_name: str = "generic",
        node_type: str = "function",
        node_role: str = "function",
        dependencies: Optional[List[str]] = None,
        code_template: str = "",
        metadata_tags: Optional[Dict[str, Any]] = None,
        configuration_schema: Optional[Dict[str, Any]] = None,
        verified: bool = False,
        semantic_tags: Optional[List[str]] = None,
        docstring: str = "",
        enrichment_source: Optional[str] = None,
        enriched_at: Optional[str] = None,
        source_priority: int = 100,
        source_provenance: Optional[str] = "unknown",
        topology_type: str = "sequential",
        feedback_state_type: Optional[str] = None,
        bound_slots: Optional[Dict[str, Any]] = None,
        is_public: bool = True,
        mutation_type: str = "pure",
        is_context_manager: bool = False,
        raises: Optional[List[str]] = None,
        type_vars: Optional[List[str]] = None,
        preconditions: Optional[List[Any]] = None,
        postconditions: Optional[List[Any]] = None,
        effects: Optional[List[Any]] = None,
        edges: Optional[List[Any]] = None,
        endable: Optional[bool] = None,
        primary_in: Optional[str] = None,
        primary_out: Optional[str] = None,
        fixture_needs: Optional[List[str]] = None,
        projection: Optional[str] = None,
        optic_role: Optional[str] = None,
        container_port: Optional[str] = None,
        key_port: Optional[str] = None,
        focus_port: Optional[str] = None,
        relations: Optional[List[Dict[str, Any]]] = None,
        bridge_profile: Optional[Dict[str, Any]] = None,
        polymorphic_propagation: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.cell_id = cell_id
        self.stage = stage
        self.keywords = {t for t in (str(k).strip().lower() for k in keywords) if t} if keywords else set()
        self.cell_type = cell_type
        self.domain_name = domain_name
        self.node_type = node_type
        self.node_role = str(node_role).lower() if node_role else "function"
        if isinstance(slots, (list, set, tuple)):
            self.slots = {s: {} for s in slots}
        else:
            self.slots = slots or {}
        self.dependencies = dependencies or []
        self.code_template = code_template
        self.metadata_tags = metadata_tags or {}
        self.configuration_schema = configuration_schema or {}
        self.source_priority = int(source_priority) if source_priority is not None else 100
        self.source_provenance = str(source_provenance) if source_provenance else "unknown"
        self.verified = bool(verified)
        self.semantic_tags = list(semantic_tags) if semantic_tags else list(self.keywords)
        self.docstring = docstring or ""
        self.enrichment_source = enrichment_source
        self.enriched_at = enriched_at
        self.is_public = bool(is_public)
        self.mutation_type = str(mutation_type or kwargs.get("mutation_type", "pure"))
        self.is_context_manager = bool(is_context_manager or kwargs.get("is_context_manager", False))
        self.raises = list(raises or kwargs.get("raises", []))
        self.type_vars = list(type_vars or kwargs.get("type_vars", []))

        # Infer topology type from feedback carrier if sequential
        if topology_type == "sequential" and feedback_state_type:
            topology_type = "traced_loop"

        self.topology_type = topology_type
        self.feedback_state_type = feedback_state_type
        self.bound_slots = dict(bound_slots) if bound_slots else {}

        self.preconditions = list(preconditions or kwargs.get("preconditions", []))
        # Postconditions and effects are separate declared slots. Declared
        # postconditions WIN; effects may fill in only when postconditions are
        # absent (some trees declare predicate-style effects only); effects
        # fall back to postconditions so single-declaration trees populate both.
        declared_posts = postconditions if postconditions is not None else kwargs.get("postconditions")
        declared_effects = effects if effects is not None else kwargs.get("effects")
        self.postconditions = list(declared_posts or declared_effects or [])
        self.effects = list(declared_effects if declared_effects is not None else self.postconditions)
        self.edges = list(edges or kwargs.get("edges", []))
        self.endable = endable if endable is not None else kwargs.get("endable")
        self.primary_in = primary_in if primary_in is not None else kwargs.get("primary_in")
        self.primary_out = primary_out if primary_out is not None else kwargs.get("primary_out")
        self.fixture_needs = list(fixture_needs or kwargs.get("fixture_needs", []))
        self.projection = str(projection).strip().lower() if projection else (str(kwargs.get("projection")).strip().lower() if kwargs.get("projection") else None)
        raw_optic = optic_role if optic_role is not None else kwargs.get("optic_role")
        self.optic_role = str(raw_optic).strip().lower() if raw_optic else None
        self.container_port = container_port if container_port is not None else kwargs.get("container_port")
        self.key_port = key_port if key_port is not None else kwargs.get("key_port")
        self.focus_port = focus_port if focus_port is not None else kwargs.get("focus_port")
        self.relations: List[Dict[str, Any]] = list(relations or kwargs.get("relations") or [])
        bp_val = bridge_profile if bridge_profile is not None else kwargs.get("bridge_profile")
        self.bridge_profile: Optional[Dict[str, Any]] = dict(bp_val) if isinstance(bp_val, dict) else None
        pp_val = polymorphic_propagation if polymorphic_propagation is not None else kwargs.get("polymorphic_propagation")
        self.polymorphic_propagation: Optional[Dict[str, Any]] = dict(pp_val) if isinstance(pp_val, dict) else None

        # For-each multiplicity expansion: a replica is a runtime copy of a
        # planned cell that re-consumes its receiver from the environment and
        # binds the NEXT member of an identifier role group (X, Y, Z). None on
        # original cells.
        self.replica_of: Optional[str] = None
        self.replica_role: Optional[str] = None
        self.bound_parent_ids: Optional[Set[str]] = None
        self.planned_port_wires: Optional[Dict[str, Tuple[Any, str]]] = None
        self.matched_clause_idx: Optional[int] = None
        self._auto_inserted_bridge: bool = False
        self._is_prerequisite_producer: bool = False
        self.typed_bindings: Dict[str, Any] = {}
        self._propagated_outputs: Dict[str, Any] = {}

        self._primary_input = None
        self._primary_output = None

        # Extract declared slot names from slots or code_template ONLY when positional 'port_*' keys need resolution
        has_positional_ports = (
            isinstance(inputs, list)
            or isinstance(outputs, list)
            or (isinstance(inputs, dict) and any(isinstance(k, str) and k.startswith("port_") for k in inputs))
            or (isinstance(outputs, dict) and any(isinstance(k, str) and k.startswith("port_") for k in outputs))
        )
        lhs_matches: List[str] = []
        in_slots_derived: List[str] = []
        if has_positional_ports:
            slot_names_list = []
            if isinstance(slots, (list, tuple, set)):
                slot_names_list = list(slots)
            elif isinstance(slots, dict):
                slot_names_list = list(slots.keys())

            lhs_outs = {"output_var"}
            all_placeholders = list(dict.fromkeys(extract_template_placeholders(code_template)))
            if code_template:
                try:
                    masked_tpl, ph_map = _mask_placeholders(code_template)
                    inv_ph = {v: k[1:-1] for k, v in ph_map.items()}
                    for tgt in get_active_language_backend().extract_assignment_targets(masked_tpl):
                        if tgt in inv_ph:
                            lhs_outs.add(inv_ph[tgt])
                except Exception:
                    eq_idx = code_template.find("=")
                    if eq_idx != -1:
                        lhs = code_template[:eq_idx].split(";")[-1]
                        lhs_outs.update(extract_template_placeholders(lhs))

            lhs_matches = [p for p in all_placeholders if p in lhs_outs]
            rhs_placeholders = [p for p in all_placeholders if p not in lhs_outs]
            in_slots_derived = [s for s in slot_names_list if s not in lhs_outs]
            if not in_slots_derived and code_template:
                seen_m = set()
                in_slots_derived = [m for m in rhs_placeholders if not (m in seen_m or seen_m.add(m))]

        # Normalize inputs into Dict[str, PortSignature]
        if isinstance(inputs, list):
            inputs_dict = {}
            for idx, item in enumerate(inputs):
                if isinstance(item, dict):
                    p_name = item.get("name") or item.get("port_name") or f"port_{idx}"
                    inputs_dict[p_name] = item
                elif isinstance(item, (PortSignature, AlgebraicSignature)):
                    inputs_dict[getattr(item, "name", f"port_{idx}")] = item
                else:
                    inputs_dict[f"port_{idx}"] = item
            inputs = inputs_dict

        self.inputs: Dict[str, PortSignature] = PortMapping()
        for k, v in (inputs or {}).items():
            orig_k = k
            if k.startswith("port_"):
                try:
                    p_idx = int(k.split("_")[1])
                    if p_idx < len(in_slots_derived):
                        k = in_slots_derived[p_idx]
                except (ValueError, IndexError):
                    pass

            if isinstance(v, PortSignature):
                port_sig = v
            elif isinstance(v, AlgebraicSignature):
                port_sig = PortSignature(name=k, signature=v, accepted_states=v.accepted_states, parent_state=v.parent_state)
            elif isinstance(v, dict):
                abs_t = _clean_abs_carrier(v.get("abstract_type"))
                acc_s = tuple(sorted({str(s).strip().lower() for s in (v.get("accepted_states") or []) if str(s).strip()}))
                p_s = str(v.get("parent_state") or "").strip().lower() or None
                p_dom_in = str(v.get("domain") or self.domain_name or "").strip().lower()
                sig = AlgebraicSignature(
                    type_name=v.get("type_name", "any"),
                    state=v.get("state", "any"),
                    qualifiers=_normalize_qualifiers(v.get("qualifiers", [])),
                    abstract_type=abs_t,
                    accepted_states=acc_s,
                    parent_state=p_s,
                    domain=p_dom_in,
                )
                port_sig = PortSignature(
                    name=k,
                    signature=sig,
                    required=v.get("required", True),
                    default_value=v.get("default_value"),
                    doc=v.get("doc", v.get("description", "")),
                    domain=v.get("domain") or self.domain_name or "",
                    abstract_type=abs_t,
                    enum_values=v.get("enum_values"),
                    param_kind=v.get("param_kind", "standard"),
                    value_constraints=v.get("value_constraints"),
                    shape_contract=v.get("shape_contract"),
                    rejected_qualifiers=v.get("rejected_qualifiers", []),
                    accepted_states=acc_s,
                    parent_state=p_s,
                    port_role=v.get("port_role") or v.get("role"),
                    binds=v.get("binds"),
                    structure_kind=v.get("structure_kind"),
                    rank=v.get("rank"),
                    element_type=v.get("element_type"),
                    supported_formats=v.get("supported_formats"),
                    complement_of_port=v.get("complement_of_port"),
                    disjoint_from_port=v.get("disjoint_from_port"),
                    key_from_upstream_role=v.get("key_from_upstream_role"),
                    carrier=v.get("carrier"),
                    nominal_type=v.get("nominal_type"),
                    layout=v.get("layout"),
                    semantics=v.get("semantics"),
                    typestate=v.get("typestate"),
                    invariants=v.get("invariants"),
                    effect_suspended=bool(v.get("effect_suspended", False)),
                )
            else:
                port_sig = PortSignature(name=k, signature=AlgebraicSignature("any", "any"))

            port_sig.name = k
            self.inputs[k] = port_sig
            if orig_k != k:
                self.inputs.add_alias(orig_k, k)

        # Normalize outputs into Dict[str, PortSignature]
        if isinstance(outputs, list):
            outputs_dict = {}
            for idx, item in enumerate(outputs):
                if isinstance(item, dict):
                    p_name = item.get("name") or item.get("port_name") or f"port_{idx}"
                    outputs_dict[p_name] = item
                elif isinstance(item, (PortSignature, AlgebraicSignature)):
                    outputs_dict[getattr(item, "name", f"port_{idx}")] = item
                else:
                    outputs_dict[f"port_{idx}"] = item
            outputs = outputs_dict

        self.outputs: Dict[str, PortSignature] = PortMapping()
        for k, v in (outputs or {}).items():
            orig_k = k
            if k.startswith("port_"):
                try:
                    p_idx = int(k.split("_")[1])
                    if p_idx < len(lhs_matches):
                        k = lhs_matches[p_idx]
                    elif len(lhs_matches) == 1:
                        k = lhs_matches[0]
                except (ValueError, IndexError):
                    pass

            if isinstance(v, PortSignature):
                v.name = k
                self.outputs[k] = v
            elif isinstance(v, AlgebraicSignature):
                self.outputs[k] = PortSignature(name=k, signature=v, accepted_states=v.accepted_states, parent_state=v.parent_state)
            elif isinstance(v, dict):
                abs_t = _clean_abs_carrier(v.get("abstract_type"))
                acc_s = tuple(sorted({str(s).strip().lower() for s in (v.get("accepted_states") or []) if str(s).strip()}))
                p_s = str(v.get("parent_state") or "").strip().lower() or None
                p_dom_in = str(v.get("domain") or self.domain_name or "").strip().lower()
                sig = AlgebraicSignature(
                    type_name=v.get("type_name", "any"),
                    state=v.get("state", "any"),
                    qualifiers=_normalize_qualifiers(v.get("qualifiers", [])),
                    abstract_type=abs_t,
                    accepted_states=acc_s,
                    parent_state=p_s,
                    domain=p_dom_in,
                )
                self.outputs[k] = PortSignature(
                    name=k,
                    signature=sig,
                    required=v.get("required", True),
                    default_value=v.get("default_value"),
                    doc=v.get("doc", v.get("description", "")),
                    domain=v.get("domain") or self.domain_name or "",
                    abstract_type=abs_t,
                    enum_values=v.get("enum_values"),
                    param_kind=v.get("param_kind", "standard"),
                    value_constraints=v.get("value_constraints"),
                    shape_contract=v.get("shape_contract"),
                    accepted_states=acc_s,
                    parent_state=p_s,
                    port_role=v.get("port_role") or v.get("role"),
                    binds=v.get("binds"),
                    structure_kind=v.get("structure_kind"),
                    rank=v.get("rank"),
                    element_type=v.get("element_type"),
                    supported_formats=v.get("supported_formats"),
                    complement_of_port=v.get("complement_of_port"),
                    disjoint_from_port=v.get("disjoint_from_port"),
                    key_from_upstream_role=v.get("key_from_upstream_role"),
                    carrier=v.get("carrier"),
                    nominal_type=v.get("nominal_type"),
                    layout=v.get("layout"),
                    semantics=v.get("semantics"),
                    typestate=v.get("typestate"),
                    invariants=v.get("invariants"),
                    effect_suspended=bool(v.get("effect_suspended", False)),
                )
            else:
                self.outputs[k] = PortSignature(name=k, signature=AlgebraicSignature("any", "any"))

            if orig_k != k:
                self.outputs.add_alias(orig_k, k)

        self._token_set: Optional[Set[str]] = None
        self._token_count: int = 0
        self._identity_tokens: Optional[Set[str]] = None
        self.op_name: Optional[str] = kwargs.get("op_name")
        self._canonical_op_names: Optional[Set[str]] = None
        self._formal_identifiers: Optional[Set[str]] = None
        self.explicit_port_bindings: Dict[str, Any] = {}
        self.formal_program_step: bool = False

        # Register types automatically in TypeRegistry
        registry = TypeRegistry.get_instance()
        parents_map = registry._parents
        for p in self.inputs.values():
            p.owner_cell = self
            tn_l = p.type_name.strip().lower()
            if tn_l and tn_l not in ("none", "null") and tn_l not in parents_map:
                registry.register_type(p.type_name)
        for p in self.outputs.values():
            p.owner_cell = self
            tn_l = p.type_name.strip().lower()
            if tn_l and tn_l not in ("none", "null") and tn_l not in parents_map:
                registry.register_type(p.type_name)

        # Mathematical structural rank derivation for projections, setters, and paired morphisms
        is_setter = False
        key_input_is_collection = False
        key_input_is_scalar = False
        val_port = None
        key_port = None

        (
            key_roles, key_port_names, val_roles, val_port_names,
            single_proj_kinds, multi_proj_kinds, feature_roles, target_roles,
            paired_ranks,
        ) = registry._get_cell_init_role_bundle()

        for p_name, p_sig in self.inputs.items():
            bind = p_sig.binds or ""
            role = p_sig.port_role or ""
            p_tn = p_sig.type_name.lower()
            if bind in key_roles or role in key_roles or p_name in key_port_names:
                key_port = p_sig
                is_explicit_scalar = (
                    p_sig.structure_kind == "scalar"
                    or p_sig.rank == 0
                    or registry.is_subtype(p_tn, "scalar")
                    or p_name in single_proj_kinds
                )
                if not is_explicit_scalar and (registry.is_subtype(p_tn, "collection") or p_name in multi_proj_kinds):
                    key_input_is_collection = True
                else:
                    key_input_is_scalar = True
            elif bind in val_roles or role in val_roles or p_name in val_port_names:
                val_port = p_sig
                is_setter = True

        prim_out = self.primary_output if self.outputs else None
        prim_in = self.primary_input if self.inputs else None

        # Projections
        if self.projection:
            if self.projection in multi_proj_kinds or (key_input_is_collection and self.projection not in single_proj_kinds):
                if prim_out and prim_out.canonical_rank is None:
                    prim_out.canonical_rank = 2
            elif self.projection in single_proj_kinds or key_input_is_scalar:
                if prim_out and prim_out.canonical_rank is None:
                    prim_out.canonical_rank = 1
        elif key_input_is_collection and not is_setter:
            if prim_out and prim_out.canonical_rank is None:
                prim_out.canonical_rank = 2
        elif key_input_is_scalar and not is_setter:
            if prim_out and prim_out.canonical_rank is None:
                prim_out.canonical_rank = 1

        # Unary endomorphisms F -> F where input has rank 1
        if prim_in and prim_out and len(self.inputs) == 1 and len(self.outputs) == 1:
            in_r = prim_in.canonical_rank
            if in_r == 1 and prim_out.canonical_rank is None:
                prim_out.canonical_rank = 1

        # Lens setter: port V demands rank 1 only for higher-rank composite containers
        if is_setter and val_port and key_input_is_scalar:
            val_sk = (val_port.structure_kind or "").lower()
            prim_r = prim_in.canonical_rank if prim_in else None
            if val_sk != "scalar" and (prim_r is None or prim_r >= 2) and val_port.canonical_rank is None:
                val_port.canonical_rank = 1

        # Dual-input paired morphism (g: X x Y -> M)
        if paired_ranks and (isinstance(paired_ranks, dict) or hasattr(paired_ranks, "get")):
            feat_port = None
            tgt_port = None
            for p_name, p_sig in self.inputs.items():
                bind = p_sig.binds or ""
                role = p_sig.port_role or ""
                if role in feature_roles or bind in feature_roles:
                    feat_port = p_sig
                elif role in target_roles or bind in target_roles:
                    tgt_port = p_sig

            if feat_port and tgt_port:
                f_r = paired_ranks.get("feature")
                t_r = paired_ranks.get("target")
                if f_r is not None and feat_port.canonical_rank is None:
                    feat_port.canonical_rank = int(f_r)
                if t_r is not None and tgt_port.canonical_rank is None:
                    tgt_port.canonical_rank = int(t_r)

    @property
    def token_set(self) -> Set[str]:
        if self._token_set is None:
            try:
                from .tokenizer import CellTokenizer
            except (ImportError, ValueError):
                from tokenizer import CellTokenizer
            kws = list(self.keywords or ())
            for tag in (self.semantic_tags or ()):
                if "_" not in str(tag) and tag not in kws:
                    kws.append(tag)
            toks = CellTokenizer.tokenize_cell(self.cell_id, kws)
            if getattr(self, "domain_name", None) and self.domain_name != "generic":
                toks.update(CellTokenizer.tokenize_identifier(self.domain_name))
            if self.docstring:
                toks.update(CellTokenizer.tokenize_prompt(self.docstring))
            for p in self.inputs:
                toks.update(CellTokenizer.tokenize_identifier(p))
            for p in self.outputs:
                toks.update(CellTokenizer.tokenize_identifier(p))
            self._token_set = toks
            self._token_count = len(toks)
        return self._token_set

    @property
    def canonical_op_names(self) -> Set[str]:
        """Derives formal operation names deterministically from cell schema metadata."""
        cached = getattr(self, "_canonical_op_names", None)
        if cached is not None:
            return cached
        ops: Set[str] = set()
        explicit = getattr(self, "op_name", None) or getattr(self, "operation", None) or getattr(self, "callable_name", None)
        if explicit and str(explicit).strip():
            ops.add(str(explicit).strip().lower())

        # Extract callable name from code_template via active language backend (zero regex)
        active_be = get_active_language_backend()
        tpl = str(getattr(self, "code_template", "") or "").strip()
        if tpl:
            try:
                skip_calls = set(active_be.debug_builtins)
                calls = active_be.extract_call_names(tpl)
                for cname in calls:
                    if cname and cname not in skip_calls:
                        ops.add(cname)
                        if len(calls) > 1:
                            break
            except Exception:
                pass

        # Strip declared domain_name or declared dependency aliases from cell_id
        cid_low = str(getattr(self, "cell_id", "") or "").strip().lower()
        prefixes: Set[str] = set()
        dom = str(getattr(self, "domain_name", "") or "").strip().lower()
        if dom and dom != "generic":
            prefixes.add(dom)
        for dep in (getattr(self, "dependencies", None) or ()):
            for ident in active_be.extract_dependency_identifiers(str(dep or "")):
                prefixes.add(ident.lower())
        for pref in prefixes:
            if cid_low.startswith(pref + "_") and len(cid_low) > len(pref) + 1:
                ops.add(cid_low[len(pref) + 1:])

        res = set(ops)
        self._canonical_op_names = res
        return res

    @property
    def formal_identifiers(self) -> Set[str]:
        """Exact formal identifier set: {cell_id} union canonical_op_names (never synonyms/keywords)."""
        cached = getattr(self, "_formal_identifiers", None)
        if cached is not None:
            return cached
        cid = str(getattr(self, "cell_id", "") or "").strip().lower()
        res = ({cid} if cid else set()) | set(self.canonical_op_names)
        self._formal_identifiers = res
        return res

    @property
    def identity_tokens(self) -> Set[str]:
        try:
            from .inference import ModelManager
            is_p0 = ModelManager.get_instance().current_profile_name == "0"
        except Exception:
            try:
                from inference import ModelManager
                is_p0 = ModelManager.get_instance().current_profile_name == "0"
            except Exception:
                is_p0 = False
        if is_p0:
            return set(self.formal_identifiers)

        if self._identity_tokens is None:
            kws = self.keywords if self.keywords else (self.semantic_tags or ())
            id_toks = CellTokenizer.tokenize_identifier(self.cell_id)
            kw_toks: Set[str] = set()
            for kw in kws:
                kw_toks.update(CellTokenizer.tokenize_identifier(kw))
            # Strip library/domain namespace from cell_id only when at least 2 atomic
            # operation tokens remain (e.g. preserve 'pygame' on PYGAME_INIT / PYGAME_QUIT)
            # and never strip tokens explicitly declared in keywords / semantic_tags.
            if getattr(self, "domain_name", None) and self.domain_name != "generic":
                dom_toks = CellTokenizer.tokenize_identifier(self.domain_name)
                rem_atomic = {t for t in (id_toks - dom_toks) if "_" not in t}
                if len(rem_atomic) >= 2:
                    id_toks = id_toks - dom_toks
            try:
                _fw = TypeRegistry.get_instance().get_function_words()
            except Exception:
                _fw = set()
            if _fw:
                rem_non_fw = {t for t in (id_toks - _fw) if "_" not in t}
                if len(rem_non_fw) >= 2:
                    id_toks = id_toks - _fw
                _strip_fw = _fw - id_toks
                if _strip_fw and (kw_toks - _strip_fw):
                    kw_toks = kw_toks - _strip_fw
            toks = id_toks | kw_toks
            try:
                aliases = TypeRegistry.get_instance().get_all_aliases()
            except Exception:
                aliases = {}
            if aliases:
                toks = toks - {t for t in toks if t in aliases and t not in kw_toks}
            # Identity is the OPERATION a cell performs (id + declared keywords/semantic tags).
            # Port/parameter names are deliberately NOT identity: a parameter such
            # as `drop_first` says nothing about what the cell does, and adding
            # its tokens made unrelated prompt words ("drop null values") claim
            # the cell. Port vocabulary stays available through `token_set`
            # (weak evidence) only.
            self._identity_tokens = toks
        return self._identity_tokens

    @property
    def is_endable(self) -> bool:
        """
        Determines whether this node can validly terminate an execution pipeline
        purely from its structural morphism signature f : A -> 1 / terminal sink:
        - Explicit node-level endable override if declared.
        - Stage 3 or declared terminal/egress/sink/evaluator node_role.
        - Empty outputs or codomain tau_out in {1, None, NoneType, void, unit, Effect[IO]}.
        - Output state in declared terminal_states or materialization_states.
        """
        if self.endable is not None:
            return bool(self.endable)
        if self.stage == 3:
            return True
        reg = TypeRegistry.get_instance()
        if reg.is_terminal_role(self.node_role):
            return True
        if not self.outputs:
            return True
        term_states = reg.get_terminal_states() | reg.get_states_by_property("is_materialized", True) | reg.get_states_by_property("is_terminal", True)
        for out_p in self.outputs.values():
            out_sig = getattr(out_p, "signature", out_p)
            out_tn = getattr(out_sig, "type_name", "")
            out_st = str(getattr(out_sig, "state", "") or "").strip().lower()
            if reg.is_void_type(out_tn):
                return True
            if out_st and out_st in term_states:
                return True
        try:
            from .runtime_poset import classify_cell_structural_role
        except (ImportError, ValueError):
            from runtime_poset import classify_cell_structural_role
        if classify_cell_structural_role(self, reg) == "sink":
            return True
        return False

    @property
    def is_egress(self) -> bool:
        return self.is_endable

    @property
    def is_ingress(self) -> bool:
        """
        Determines whether this node is an ingress/source morphism f : 1 -> A
        purely from its structural signature and port types.
        """
        if self.stage == 1:
            return True
        reg = TypeRegistry.get_instance()
        if self.stage == 3 or reg.is_terminal_role(self.node_role) or getattr(self, "is_sink", False):
            return False
        if reg.is_ingress_role(self.node_role):
            return True
        try:
            from .runtime_poset import classify_cell_structural_role
        except (ImportError, ValueError):
            from runtime_poset import classify_cell_structural_role
        if classify_cell_structural_role(self, reg) == "source":
            return True
        param_roles = reg.get_roles_for_category("parameter")
        ins = self.inputs or {}
        req_data = []
        for p in ins.values():
            if not p.required or is_path_port(p, reg):
                continue
            sk = str(getattr(p, "structure_kind", "") or "").strip().lower()
            if sk != "scalar":
                req_data.append(p)
                continue
            p_t = str(getattr(getattr(p, "signature", p), "type_name", "") or "").strip().lower()
            p_role = str(getattr(p, "port_role", "") or getattr(p, "role", "") or "").strip().lower()
            if not reg.is_config_role(p_role) and p_role not in param_roles:
                if not reg.is_scalar_primitive(p_t) or reg.is_subtype(p_t, "bytes") or p_t == "bytes":
                    req_data.append(p)
        return bool(self.outputs) and not req_data

    @property
    def role(self) -> str:
        return self.node_role

    @property
    def is_combinator(self) -> bool:
        return str(self.node_role).lower() == "combinator" or str(self.node_type).lower() == "combinator"

    @property
    def token_count(self) -> int:
        if self._token_set is None:
            _ = self.token_set
        return self._token_count

    @property
    def primary_input(self) -> PortSignature:
        """
        Identifies the primary data-bearing input port.
        Categorically stage-aware: an ingestion morphism consumes an
        environmental asset, so its textual/path-typed carrier port is primary;
        otherwise the first required non-scalar data port wins, falling back to
        the first declared non-scalar port.
        """
        if self._primary_input is not None:
            return self._primary_input

        if not self.inputs:
            res = PortSignature("input_data", AlgebraicSignature("any", "any"))
            self._primary_input = res
            return res

        if self.primary_in and self.primary_in in self.inputs:
            res = self.inputs[self.primary_in]
            self._primary_input = res
            return res

        registry = TypeRegistry.get_instance()

        def _is_data_carrier(p: PortSignature) -> bool:
            return (
                not registry.is_builtin_scalar_primitive(p.type_name)
                and not registry.has_type_kind(p.type_name, "scalar")
            )

        def _is_asset_carrier(p: PortSignature) -> bool:
            tn = p.type_name.lower()
            path_roots = set(registry.get_types_by_kind("path") or ()) | {"path", "file"}
            return (
                registry.has_type_kind(tn, "path")
                or tn in path_roots
                or any(registry.is_subtype(tn, pt) for pt in path_roots)
            )

        res = None
        if self.is_ingress:
            # Ingestion morphism (Env -> C): the asset carrier is the categorical input.
            asset_ports = [p for p in self.inputs.values() if _is_asset_carrier(p)]
            if asset_ports:
                res = asset_ports[0]
        if res is None:
            required_ports = [p for p in self.inputs.values() if p.required]
            if required_ports:
                required_data = [p for p in required_ports if _is_data_carrier(p)]
                res = required_data[0] if required_data else required_ports[0]
            else:
                data_ports = [p for p in self.inputs.values() if _is_data_carrier(p)]
                res = data_ports[0] if data_ports else next(iter(self.inputs.values()))
        self._primary_input = res
        return res

    @property
    def primary_output(self) -> PortSignature:
        """Identifies the primary data-bearing output port."""
        if self._primary_output is not None:
            return self._primary_output

        if not self.outputs:
            res = PortSignature("output_data", AlgebraicSignature("any", "any"))
            self._primary_output = res
            return res

        if self.primary_out and self.primary_out in self.outputs:
            res = self.outputs[self.primary_out]
            self._primary_output = res
            return res

        res = next(iter(self.outputs.values()))
        self._primary_output = res
        return res

    @property
    def is_public_morphism(self) -> bool:
        """
        Evaluates whether the cell represents a public morphism in domain category C.
        Excludes private/internal implementation artifacts (leading underscores, dunder methods).
        """
        if "__" in self.cell_id:
            return False
        for dep in self.dependencies:
            if "._" in dep or "import _" in dep:
                return False
        return True

    def can_accept(self, sig: Union[AlgebraicSignature, PortSignature]) -> bool:
        """True iff this cell has an input port that unifies with `sig`."""
        target_sig = sig.signature if hasattr(sig, "signature") else sig
        for p in self.inputs.values():
            if target_sig.unifies_with(p.signature):
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "cell_id": self.cell_id,
            "stage": self.stage,
            "keywords": list(self.keywords) if isinstance(self.keywords, (set, list)) else [],
            "node_type": self.node_type,
            "node_role": self.node_role,
            "domain_name": self.domain_name,
            "dependencies": list(self.dependencies),
            "code_template": self.code_template,
            "verified": self.verified,
            "semantic_tags": list(self.semantic_tags),
            "docstring": self.docstring,
            "source_priority": getattr(self, "source_priority", 100),
            "is_public": getattr(self, "is_public", True),
            "mutation_type": getattr(self, "mutation_type", "pure"),
            "is_context_manager": getattr(self, "is_context_manager", False),
            "raises": list(getattr(self, "raises", []) or []),
            "type_vars": list(getattr(self, "type_vars", []) or []),
            "preconditions": [dict(x) if isinstance(x, dict) else x for x in (getattr(self, "preconditions", []) or [])],
            "postconditions": [dict(x) if isinstance(x, dict) else x for x in (getattr(self, "postconditions", []) or [])],
            "effects": list(getattr(self, "effects", []) or []),
            "edges": list(getattr(self, "edges", []) or []),
            "endable": getattr(self, "endable", None),
            "primary_in": getattr(self, "primary_in", None),
            "primary_out": getattr(self, "primary_out", None),
            "fixture_needs": list(getattr(self, "fixture_needs", []) or []),
            "projection": getattr(self, "projection", None),
            "optic_role": getattr(self, "optic_role", None),
            "container_port": getattr(self, "container_port", None),
            "key_port": getattr(self, "key_port", None),
            "focus_port": getattr(self, "focus_port", None),
            "topology_type": self.topology_type,
            "feedback_state_type": getattr(self, "feedback_state_type", None),
            "slots": self.slots,
            "bound_slots": getattr(self, "bound_slots", {}),
            "inputs": {
                name: {
                    "type_name": p.type_name,
                    "state": p.state,
                    "required": p.required,
                    "default_value": p.default_value,
                    "role": getattr(p, "port_role", None) or getattr(p, "role", None),
                    "port_role": getattr(p, "port_role", None) or getattr(p, "role", None),
                    "binds": getattr(p, "binds", None),
                    "qualifiers": [list(q) if isinstance(q, (tuple, set, list)) else q for q in (getattr(getattr(p, "signature", None), "qualifiers", []) or getattr(p, "qualifiers", []))],
                    "rejected_qualifiers": list(getattr(p, "rejected_qualifiers", []) or []),
                    "doc": getattr(p, "doc", ""),
                    "domain": getattr(p, "domain", ""),
                    "abstract_type": getattr(p, "abstract_type", None),
                    "enum_values": getattr(p, "enum_values", None),
                    "param_kind": getattr(p, "param_kind", "standard"),
                    "value_constraints": getattr(p, "value_constraints", None),
                    "shape_contract": getattr(p, "shape_contract", None),
                    "accepted_states": list(getattr(p, "accepted_states", []) or []),
                    "parent_state": getattr(p, "parent_state", None),
                    "structure_kind": getattr(p, "structure_kind", None),
                    "rank": getattr(p, "rank", None),
                    "element_type": getattr(p, "element_type", None),
                    "supported_formats": getattr(p, "supported_formats", None),
                    "complement_of_port": getattr(p, "complement_of_port", None),
                    "disjoint_from_port": getattr(p, "disjoint_from_port", None),
                    "key_from_upstream_role": getattr(p, "key_from_upstream_role", None),
                }
                for name, p in self.inputs.items()
            },
            "outputs": {
                name: {
                    "type_name": p.type_name,
                    "state": p.state,
                    "required": p.required,
                    "role": getattr(p, "port_role", None) or getattr(p, "role", None),
                    "port_role": getattr(p, "port_role", None) or getattr(p, "role", None),
                    "binds": getattr(p, "binds", None),
                    "qualifiers": [list(q) if isinstance(q, (tuple, set, list)) else q for q in (getattr(getattr(p, "signature", None), "qualifiers", []) or getattr(p, "qualifiers", []))],
                    "doc": getattr(p, "doc", ""),
                    "domain": getattr(p, "domain", ""),
                    "abstract_type": getattr(p, "abstract_type", None),
                    "enum_values": getattr(p, "enum_values", None),
                    "param_kind": getattr(p, "param_kind", "standard"),
                    "value_constraints": getattr(p, "value_constraints", None),
                    "shape_contract": getattr(p, "shape_contract", None),
                    "accepted_states": list(getattr(p, "accepted_states", []) or []),
                    "parent_state": getattr(p, "parent_state", None),
                    "structure_kind": getattr(p, "structure_kind", None),
                    "rank": getattr(p, "rank", None),
                    "element_type": getattr(p, "element_type", None),
                    "supported_formats": getattr(p, "supported_formats", None),
                    "complement_of_port": getattr(p, "complement_of_port", None),
                    "disjoint_from_port": getattr(p, "disjoint_from_port", None),
                    "key_from_upstream_role": getattr(p, "key_from_upstream_role", None),
                }
                for name, p in self.outputs.items()
            },
        }
        if hasattr(self, "sub_cells"):
            d["sub_cells"] = list(self.sub_cells)
        if hasattr(self, "algorithmic_steps"):
            d["algorithmic_steps"] = list(self.algorithmic_steps)
        if hasattr(self, "internal_topology"):
            d["internal_topology"] = dict(self.internal_topology)
        if getattr(self, "bound_parent_ids", None):
            d["bound_parent_ids"] = list(self.bound_parent_ids)
        if getattr(self, "planned_port_wires", None):
            d["planned_port_wires"] = dict(self.planned_port_wires)
        if getattr(self, "matched_clause_idx", None) is not None:
            d["matched_clause_idx"] = self.matched_clause_idx
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cell":
        return build_cell_from_raw(data)

    def clone(self) -> "Cell":
        c = copy.copy(self)
        if self.bound_slots:
            c.bound_slots = {k: list(v) for k, v in self.bound_slots.items()}
        if self.metadata_tags:
            c.metadata_tags = dict(self.metadata_tags)
        if self.bound_parent_ids:
            c.bound_parent_ids = set(self.bound_parent_ids)
        if getattr(self, "planned_port_wires", None):
            c.planned_port_wires = dict(self.planned_port_wires)
        if getattr(self, "explicit_port_bindings", None):
            c.explicit_port_bindings = dict(self.explicit_port_bindings)
        if getattr(self, "formal_program_step", False):
            c.formal_program_step = True
        if getattr(self, "clause_literals", None):
            c.clause_literals = list(self.clause_literals)
        if getattr(self, "preconditions", None):
            c.preconditions = [dict(x) if isinstance(x, dict) else x for x in self.preconditions]
        if getattr(self, "postconditions", None):
            c.postconditions = [dict(x) if isinstance(x, dict) else x for x in self.postconditions]
        if getattr(self, "effects", None):
            c.effects = list(self.effects)
        c.typed_bindings = dict(getattr(self, "typed_bindings", {}) or {})
        c._propagated_outputs = dict(getattr(self, "_propagated_outputs", {}) or {})
        return c

    def effect_names(self) -> Set[str]:
        """Extracts normalized effect names whether self.effects elements are str, dict, or objects."""
        out: Set[str] = set()
        for eff in getattr(self, "effects", None) or ():
            if isinstance(eff, dict):
                for k in ("name", "kind", "type", "effect", "property"):
                    v = eff.get(k)
                    if v:
                        out.add(str(v).strip().lower())
            elif isinstance(eff, str):
                s = eff.strip()
                if s:
                    out.add(s.lower())
            elif eff is not None:
                for attr in ("name", "property", "kind", "type", "effect"):
                    v = getattr(eff, attr, None)
                    if v and isinstance(v, (str, int)):
                        out.add(str(v).strip().lower())
                s = str(eff).strip()
                if s and not s.startswith("{") and not s.startswith("<"):
                    out.add(s.lower())
        return out

    def __repr__(self) -> str:
        in_str = f"{self.primary_input.type_name}[{self.primary_input.state}]"
        out_str = f"{self.primary_output.type_name}[{self.primary_output.state}]"
        return f"<{self.__class__.__name__} {self.cell_id} ({self.domain_name}) {in_str} -> {out_str}>"


def extract_effect_names(cell: Any) -> Set[str]:
    """Extracts normalized effect names from any cell object."""
    if cell is None:
        return set()
    if hasattr(cell, "effect_names") and callable(cell.effect_names):
        return cell.effect_names()
    out: Set[str] = set()
    for eff in getattr(cell, "effects", None) or ():
        if isinstance(eff, dict):
            for k in ("name", "kind", "type", "effect", "property"):
                v = eff.get(k)
                if v:
                    out.add(str(v).strip().lower())
        elif isinstance(eff, str):
            s = eff.strip()
            if s:
                out.add(s.lower())
        elif eff is not None:
            for attr in ("name", "property", "kind", "type", "effect"):
                v = getattr(eff, attr, None)
                if v and isinstance(v, (str, int)):
                    out.add(str(v).strip().lower())
            s = str(eff).strip()
            if s and not s.startswith("{") and not s.startswith("<"):
                out.add(s.lower())
    return out


class MicroCell(Cell):
    """Primitive node in the lattice."""
    __slots__ = ()

    def __init__(self, **kwargs):
        kwargs["cell_type"] = "micro"
        super().__init__(**kwargs)


class MacroCell(Cell):
    """Higher-level hierarchical composite node in the lattice (Section 3.1)."""
    __slots__ = ("sub_cells", "algorithmic_steps", "_resolved_sub_cells", "internal_topology", "__dict__")

    def __init__(
        self,
        sub_cells: Optional[List[str]] = None,
        algorithmic_steps: Optional[List[str]] = None,
        internal_topology: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        kwargs["cell_type"] = "macro"
        super().__init__(**kwargs)
        self.sub_cells = sub_cells or []
        self.algorithmic_steps = algorithmic_steps or []
        self.internal_topology = internal_topology or {}
        self._resolved_sub_cells: Dict[str, Cell] = {}


def _apply_method_owner_qualification(cell: Any) -> None:
    """
    Universal OOP Method-Owner Qualification.

    In any object-oriented library, methods and constructors associated with an
    OwnerClass (e.g. OwnerClass.method or OwnerClass.__init__) that produce or
    consume a stateful instance of OwnerClass must unify on OwnerClass.

    Grounded exclusively in declared structural fields:
    - For any cell whose cell_id has a qualified class prefix Owner.method
      (where node_type == "method" or Owner is a registered class/type or shares
      a prefix with constructor / method cells in the domain):
      1. Producer side: If an output port produces an abstract/role carrier
         (its type_name is an abstract role/supertype or matches the receiver type
         of sibling methods on Owner, or port_role is a stateful instance sink),
         stamp (owner_qual,) onto that output port's signature.qualifiers AND
         register reg.register_type(owner_qual, out_type_name) in TypeRegistry.
      2. Consumer side: If an input port on Owner.method is the instance receiver
         (port_role in consumer roles or has a precondition on that port),
         stamp (owner_qual,) onto that input port's signature.qualifiers.

    Idempotent; applied to every loaded cell on topology rebuild.
    """
    cid = str(getattr(cell, "cell_id", "") or "")
    if not cid or "." not in cid or str(getattr(cell, "node_type", "") or "").lower() != "method":
        return

    owner = cid.rsplit(".", 1)[0]
    owner_qual = owner.strip().lower()
    if not owner_qual:
        return

    def _qual_tuple_set(sig: Any) -> Set[Tuple[str, ...]]:
        return {tuple(q) for q in (getattr(sig, "qualifiers", None) or ())}

    def _stamp(port_sig: Any, owner_qual: str) -> None:
        try:
            if not getattr(port_sig, "nominal_type", None):
                port_sig.nominal_type = owner_qual
        except Exception:
            pass
        sig = getattr(port_sig, "signature", None)
        if sig is None or not hasattr(sig, "qualifiers"):
            return
        quals = _qual_tuple_set(sig)
        if (owner_qual,) in quals:
            return
        quals.add((owner_qual,))
        try:
            new_sig = replace(sig, qualifiers=tuple(sorted(quals)))
            port_sig.signature = new_sig
        except Exception:
            return

    def _cond_field(obj: Any, field_name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(field_name, default)
        return getattr(obj, field_name, default)

    reg = TypeRegistry.get_instance()
    try:
        reg._owner_quals.add((owner_qual,))
    except Exception:
        pass

    # 1. Producer side: stamp (owner_qual,) on output ports where port_role in instance_sink_roles
    # or where type_name is a subtype of stateful_instance_roots in the poset (returning fitted/initialized self instance)
    owner_sink_roles = reg.get_roles_for_category("instance_sink") | reg.get_roles_for_category("owner_sink") | {"model_sink"}
    stateful_roots = (
        reg.get_types_with_property("is_stateful_instance", True)
        | reg.get_types_by_kind("model")
        | {"model"}
    )
    for out_p in (getattr(cell, "outputs", {}) or {}).values():
        port_role = str(getattr(out_p, "port_role", "") or "").lower()
        out_sig = getattr(out_p, "signature", None)
        out_tn = str(getattr(out_sig, "type_name", "") or getattr(out_p, "type_name", "") or "").lower()
        is_owner_output = (
            port_role in owner_sink_roles
            or (bool(out_tn) and any(reg.is_subtype(out_tn, root) for root in stateful_roots))
        )
        if is_owner_output:
            _stamp(out_p, owner_qual)
            if out_tn:
                try:
                    reg.register_type(owner_qual, out_tn)
                except Exception:
                    pass

    # 2. Consumer side: stamp (owner_qual,) ONLY on instance-receiver input ports
    # (never on data/tensor/scalar arguments that merely carry data preconditions like ndim or is_complex).
    consumer_roles = (
        reg.get_roles_for_category("strict")
        | reg.get_roles_for_category("receiver")
        | reg.get_roles_for_category("instance_consumer")
        | reg.get_roles_for_category("model")
    )
    data_arg_roles = (
        reg.get_roles_for_category("feature")
        | reg.get_roles_for_category("target")
        | reg.get_roles_for_category("data")
    ) - consumer_roles
    state_precondition_props = reg.get_declared_state_property_keys()
    stateful_precondition_ports = set()
    for pre in getattr(cell, "preconditions", []) or []:
        prop = str(_cond_field(pre, "property", "") or "").strip().lower()
        target = str(_cond_field(pre, "target", "") or "").strip()
        if prop in state_precondition_props and target:
            stateful_precondition_ports.add(target)

    non_handle_data_kinds = NON_HANDLE_DATA_KINDS
    for in_name, in_p in (getattr(cell, "inputs", {}) or {}).items():
        in_role = str(getattr(in_p, "port_role", "") or "").lower()
        in_sk = str(getattr(in_p, "structure_kind", "") or "").lower()
        in_sig = getattr(in_p, "signature", None)
        in_tn = str(getattr(in_sig, "type_name", "") or getattr(in_p, "type_name", "") or "").lower()
        is_stateful_type = (in_sk == "handle") or (
            bool(in_tn) and any(reg.is_subtype(in_tn, root) for root in stateful_roots)
        )
        if not is_stateful_type and (in_role in data_arg_roles or in_sk in non_handle_data_kinds):
            continue
        is_receiver_input = (
            in_role in consumer_roles
            or in_name in stateful_precondition_ports
        )
        if is_receiver_input:
            _stamp(in_p, owner_qual)


_apply_declared_estimator_identity = _apply_method_owner_qualification



def build_cell_from_raw(c_dict: Dict[str, Any], domain: str = "generic", default_verified: bool = True) -> "Cell":
    """
    The ONE place a tree entry (raw dict) becomes a Cell.

    The JSON loader, register_cell() and the compiled-database loader all call this, so the
    CLI (which runs from lattice.db) and the tests (which load JSON) can never again disagree
    about what a cell is. Previously three hand-copied constructors had drifted: the DB path
    dropped every port's `binds`, merged semantic_tags into keywords (changing retrieval
    identity), and lost `raises`, `fixture_needs`, `mutation_type` and `topology_type`.
    """
    _reg_m = TypeRegistry.get_instance()
    is_macro = is_macro_cell_dict(c_dict, reg=_reg_m)
    cell_cls = MacroCell if is_macro else MicroCell
    cell_instance = cell_cls(
        cell_id=c_dict.get("cell_id"),
        stage=c_dict.get("stage", 2),
        keywords=c_dict.get("keywords", []),
        inputs=c_dict.get("inputs", {}),
        outputs=c_dict.get("outputs", {}),
        domain_name=c_dict.get("domain_name") or domain or "generic",
        node_type=c_dict.get("node_type", "function"),
        node_role=c_dict.get("node_role", "function"),
        slots=c_dict.get("slots", {}),
        dependencies=c_dict.get("dependencies", []),
        code_template=c_dict.get("code_template", ""),
        verified=c_dict.get("verified", default_verified),
        semantic_tags=c_dict.get("semantic_tags", []),
        docstring=c_dict.get("docstring", ""),
        source_priority=c_dict.get("source_priority", 100),
        is_public=bool(c_dict.get("is_public", True)),
        mutation_type=c_dict.get("mutation_type", "pure"),
        is_context_manager=bool(c_dict.get("is_context_manager", False)),
        raises=c_dict.get("raises", []),
        type_vars=c_dict.get("type_vars", []),
        preconditions=c_dict.get("preconditions", []),
        postconditions=c_dict.get("postconditions", []),
        effects=c_dict.get("effects", []),
        edges=c_dict.get("edges", []),
        endable=c_dict.get("endable"),
        primary_in=c_dict.get("primary_in"),
        primary_out=c_dict.get("primary_out"),
        fixture_needs=c_dict.get("fixture_needs", []),
        projection=c_dict.get("projection"),
        optic_role=c_dict.get("optic_role"),
        container_port=c_dict.get("container_port"),
        key_port=c_dict.get("key_port"),
        focus_port=c_dict.get("focus_port"),
        relations=c_dict.get("relations", []),
        bridge_profile=c_dict.get("bridge_profile"),
        polymorphic_propagation=c_dict.get("polymorphic_propagation"),
        op_name=c_dict.get("op_name") or c_dict.get("operation") or c_dict.get("callable_name"),
        sub_cells=c_dict.get("sub_cells", []),
        algorithmic_steps=c_dict.get("algorithmic_steps", []),
        internal_topology=c_dict.get("internal_topology", {}),
        topology_type=c_dict.get("topology_type", "sequential"),
        feedback_state_type=c_dict.get("feedback_state_type"),
        bound_slots=c_dict.get("bound_slots", {}),
    )
    if "bound_parent_ids" in c_dict and c_dict["bound_parent_ids"]:
        cell_instance.bound_parent_ids = set(c_dict["bound_parent_ids"])
    if "planned_port_wires" in c_dict and c_dict["planned_port_wires"]:
        cell_instance.planned_port_wires = dict(c_dict["planned_port_wires"])
    if "matched_clause_idx" in c_dict and c_dict["matched_clause_idx"] is not None:
        cell_instance.matched_clause_idx = c_dict["matched_clause_idx"]
    return cell_instance


class LatticeOrchestrator:
    """
    Mathematical Lattice Topology G = (V, E).
    Maintains nodes V and allows loading/unloading modular knowledge trees.
    An edge (u, v) exists iff u.primary_output unifies with v's accepting input port.
    """
    _active_instance: Optional["LatticeOrchestrator"] = None
    _instance_lock = threading.RLock()

    @classmethod
    def get_active_instance(cls) -> Optional["LatticeOrchestrator"]:
        with cls._instance_lock:
            return cls._active_instance

    def __init__(self, trees_directory: str = "trees", active_domain: str = "all", db_path: Optional[str] = None, auto_load: bool = True):
        with LatticeOrchestrator._instance_lock:
            LatticeOrchestrator._active_instance = self
        self.trees_directory = trees_directory
        self.db_path = db_path if db_path is not None else os.path.join(trees_directory, "lattice.db")
        self.active_domain = active_domain
        self.loaded_cells: Dict[str, Cell] = CaseInsensitiveDict()
        self.typestate_vocabularies: Dict[str, Any] = {}
        self._adjacency: Dict[str, List[str]] = {}
        self._reverse_adjacency: Dict[str, List[str]] = {}
        self._token_index: Dict[str, List[Cell]] = {}
        self._bridge_cells: List[Cell] = []
        self.dynamic_edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._bridge_path_cache: Dict[Tuple[str, str, int], List[Cell]] = {}
        self._directly_unify_cache: Dict[Tuple[str, str], bool] = {}
        self._valid_bridge_cache: Dict[str, bool] = {}
        self._macro_edges: Optional[Set[Tuple[str, str]]] = None
        self._generic_port_cache: Optional[Any] = None
        self._version: int = 0
        self.edge_mode: Optional[str] = None
        self._lock = threading.RLock()

        if auto_load:
            if os.path.exists(self.db_path):
                try:
                    try:
                        from .lattice_db import ensure_lattice_compiled
                    except (ImportError, ValueError):
                        from lattice_db import ensure_lattice_compiled
                    ensure_lattice_compiled(self.trees_directory, self.db_path)
                except Exception:
                    pass
                self.load_from_database(self.db_path)
            else:
                self.load_all_json_trees()
                self.build_topology()

    @property
    def cells(self) -> List[Cell]:
        return list(self.loaded_cells.values())

    @property
    def macro_edges(self) -> Set[Tuple[str, str]]:
        if self._macro_edges is None:
            with self._lock:
                if self._macro_edges is None:
                    macro_set: Set[Tuple[str, str]] = set()
                    for m in self.loaded_cells.values():
                        if getattr(m, "cell_type", "") == "macro" or getattr(m, "sub_cells", None):
                            topo = getattr(m, "internal_topology", {}) or {}
                            for s_id, targets in topo.items():
                                for t_id in targets:
                                    macro_set.add((s_id, t_id))
                            subs = getattr(m, "sub_cells", None)
                            if subs:
                                for i in range(len(subs) - 1):
                                    macro_set.add((subs[i], subs[i + 1]))
                    self._macro_edges = macro_set
        return self._macro_edges

    @property
    def _cells_by_input(self) -> Dict[Tuple[str, str], List[Cell]]:
        res: Dict[Tuple[str, str], List[Cell]] = {}
        for c in self.loaded_cells.values():
            for p in c.inputs.values():
                res.setdefault((p.type_name, p.state), []).append(c)
        return res

    @property
    def _cells_by_output(self) -> Dict[Tuple[str, str], List[Cell]]:
        res: Dict[Tuple[str, str], List[Cell]] = {}
        for c in self.loaded_cells.values():
            for p in c.outputs.values():
                res.setdefault((p.type_name, p.state), []).append(c)
        return res

    def load_tree_file(self, json_path: str):
        """Loads an arbitrary domain tree from JSON without engine hardcodes."""
        with self._lock:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                raw_cells = data
                domain = Path(json_path).stem
            else:
                domain = data.get("domain", "generic")
                dom = str(domain or "").strip().lower() or None
                raw_cells = data.get("cells", [])
                reg = TypeRegistry.get_instance()
                if dom and data.get("is_core"):
                    reg._core_domains.add(dom)
                if isinstance(data.get("language_spec"), dict):
                    reg.register_language_spec(data["language_spec"])
                if isinstance(data.get("exclusive_qualifiers"), list):
                    reg.register_exclusive_qualifiers(data["exclusive_qualifiers"])
                if isinstance(data.get("exclusive_roles"), list):
                    reg.register_exclusive_roles(data["exclusive_roles"])
                if isinstance(data.get("dependent_ports"), dict):
                    reg.register_dependent_ports(data["dependent_ports"])
                if isinstance(data.get("forbidden_side_effects"), list):
                    reg.register_forbidden_side_effects(data["forbidden_side_effects"])
                if isinstance(data.get("top_level_carriers"), list):
                    reg.register_top_level_carriers(data["top_level_carriers"])
                if isinstance(data.get("state_role_ancestors"), dict):
                    reg.register_state_role_ancestors(data["state_role_ancestors"])
                if isinstance(data.get("state_role_markers"), dict):
                    reg.register_state_role_markers(data["state_role_markers"])
                if isinstance(data.get("carrier_roles"), dict):
                    reg.register_carrier_roles(data["carrier_roles"])
                if isinstance(data.get("identifier_root_states"), list):
                    reg.register_identifier_root_states(data["identifier_root_states"])
                if isinstance(data.get("supported_formats"), list):
                    reg.register_supported_formats(data["supported_formats"])
                if isinstance(data.get("capabilities"), dict):
                    reg.register_capabilities(data["capabilities"])
                if isinstance(data.get("pointed_defaults"), dict):
                    for td_k, td_v in data["pointed_defaults"].items():
                        reg.register_type_default(td_k, td_v, domain=dom)
                if "type_vars" in data and isinstance(data["type_vars"], list):
                    reg.register_type_vars(data["type_vars"])
                if "aliases" in data and isinstance(data["aliases"], dict):
                    for a_k, a_v in data["aliases"].items():
                        reg.register_alias(a_k, a_v)
                if "top_types" in data and isinstance(data["top_types"], list):
                    for top_t in data["top_types"]:
                        reg.register_top(top_t)
                if "top_states" in data and isinstance(data["top_states"], list):
                    for top_s in data["top_states"]:
                        reg.register_top_state(top_s)
                if "product_constructors" in data and isinstance(data["product_constructors"], list):
                    for p_ctor in data["product_constructors"]:
                        reg.register_product_constructor(p_ctor)
                if "artifact_readers" in data and isinstance(data["artifact_readers"], dict):
                    for cat, readers in data["artifact_readers"].items():
                        if isinstance(readers, list):
                            for r in readers:
                                parts = str(r).replace(":", ".").rsplit(".", 1)
                                if len(parts) == 2:
                                    reg.register_artifact_reader(cat, parts[0], parts[1])
                raw_types = data.get("types")
                if isinstance(raw_types, dict):
                    for t_name, t_meta in raw_types.items():
                        if isinstance(t_meta, dict):
                            parents = t_meta.get("parents") or ([t_meta["parent"]] if t_meta.get("parent") else [])
                            if t_meta.get("kind"):
                                reg.register_type_kind(t_name, t_meta["kind"])
                            if isinstance(t_meta.get("properties"), dict):
                                reg.register_type_properties(t_name, t_meta["properties"])
                            if "empty_expr" in t_meta:
                                reg.register_type_empty_expr(t_name, t_meta["empty_expr"], domain=dom)
                            elif "default_value" in t_meta:
                                reg.register_type_default(t_name, t_meta["default_value"], domain=dom)
                            elif "pointed_default" in t_meta:
                                reg.register_type_default(t_name, t_meta["pointed_default"], domain=dom)
                        else:
                            parents = [str(t_meta)]
                        if not parents:
                            reg.register_type(t_name, None)
                        for parent in parents:
                            reg.register_type(t_name, parent or None)
                elif isinstance(raw_types, list):
                    for item in raw_types:
                        if isinstance(item, str):
                            reg.register_type(item, None)
                        elif isinstance(item, dict):
                            t_name = item.get("name") or item.get("type_name")
                            if not t_name:
                                continue
                            if item.get("kind"):
                                reg.register_type_kind(t_name, item["kind"])
                            if isinstance(item.get("properties"), dict):
                                reg.register_type_properties(t_name, item["properties"])
                            if "empty_expr" in item:
                                reg.register_type_empty_expr(t_name, item["empty_expr"], domain=dom)
                            elif "default_value" in item:
                                reg.register_type_default(t_name, item["default_value"], domain=dom)
                            elif "pointed_default" in item:
                                reg.register_type_default(t_name, item["pointed_default"], domain=dom)
                            parents = item.get("parents") or ([item.get("parent") or item.get("parent_type")] if (item.get("parent") or item.get("parent_type")) else [])
                            if not parents:
                                reg.register_type(t_name, None)
                            for parent in parents:
                                reg.register_type(t_name, parent or None)
                if "typestates" in data and data["typestates"]:
                    self.typestate_vocabularies[domain] = data["typestates"]
                    ts_info = data["typestates"]
                    if isinstance(ts_info, dict) and isinstance(ts_info.get("terminal_states"), list):
                        reg.register_terminal_states(ts_info["terminal_states"])
                    states_list = ts_info.get("states", []) if isinstance(ts_info, dict) else (ts_info if isinstance(ts_info, list) else [])
                    dom_l = str(domain or "").strip().lower() or None
                    for s in states_list:
                        if isinstance(s, dict):
                            s_name = s.get("name")
                            s_parent = s.get("parent_state")
                            if s_name:
                                reg.register_state(
                                    s_name,
                                    s_parent,
                                    carrier_type=s.get("carrier_type"),
                                    properties=s.get("properties"),
                                    domain=dom_l,
                                )
                        elif isinstance(s, str):
                            reg.register_state(s, domain=dom_l)
                if "advisory_qualifiers" in data and isinstance(data["advisory_qualifiers"], list):
                    reg.register_advisory_qualifiers(data["advisory_qualifiers"])
                if "abstract_carriers" in data and isinstance(data["abstract_carriers"], list):
                    reg.register_abstract_carriers(data["abstract_carriers"])
                if "abstract_carrier_mapping" in data and isinstance(data["abstract_carrier_mapping"], dict):
                    reg.register_abstract_carrier_mapping(data["abstract_carrier_mapping"])
                if "data_bearing_roles" in data and isinstance(data["data_bearing_roles"], list):
                    reg.register_data_bearing_roles(data["data_bearing_roles"])

            reg = TypeRegistry.get_instance()
            for c_dict in raw_cells:
                if isinstance(c_dict, dict):
                    if "type_vars" in c_dict and c_dict["type_vars"]:
                        reg.register_type_vars(c_dict["type_vars"])
                    for p in (c_dict.get("inputs") or {}).values() if isinstance(c_dict.get("inputs"), dict) else ():
                        if isinstance(p, dict):
                            role = p.get("port_role") or p.get("role")
                            if role:
                                reg.register_role_carrier(role)
                cell = build_cell_from_raw(c_dict, domain, default_verified=True)
                self.loaded_cells[cell.cell_id] = cell
            logger.info(f"[LATTICE] Loaded {len(raw_cells)} nodes from tree: {json_path} (domain: {domain})")

    def unload_tree(self, domain: str):
        """Removes all nodes belonging to a tree domain."""
        with self._lock:
            to_remove = [cid for cid, c in self.loaded_cells.items() if c.domain_name == domain]
            for cid in to_remove:
                del self.loaded_cells[cid]
            logger.info(f"[LATTICE] Unloaded {len(to_remove)} nodes for domain: {domain}")

    def register_cell(self, cell_or_dict: Union[Cell, Dict[str, Any]], domain: Optional[str] = None) -> Cell:
        """
        Dynamically registers a new Cell or cell dictionary into the active orchestrator,
        updates the type registry, and rebuilds topology so the node can participate in routing.
        """
        with self._lock:
            if isinstance(cell_or_dict, Cell):
                cell = cell_or_dict
            else:
                c_dict = cell_or_dict
                if isinstance(c_dict, dict) and "type_vars" in c_dict and c_dict["type_vars"]:
                    TypeRegistry.get_instance().register_type_vars(c_dict["type_vars"])
                cell = build_cell_from_raw(c_dict, domain or "generic", default_verified=False)
            self.loaded_cells[cell.cell_id] = cell
            self.build_topology()
            logger.info(f"[LATTICE] Dynamically registered cell: {cell.cell_id} (domain: {cell.domain_name})")
            return cell

    def load_all_json_trees(self):
        """Loads all JSON trees located in trees_directory and supplemental trees."""
        with self._lock:
            self.loaded_cells.clear()
            self._bridge_path_cache.clear()
            self._directly_unify_cache.clear()
            self._valid_bridge_cache.clear()
            self._macro_edges = None
            self._generic_port_cache = None
            self._version += 1
            target_dir = self.trees_directory
        search_dirs = [target_dir]
        if os.path.basename(os.path.normpath(target_dir)) in TREE_SEARCH_DIRS:
            for candidate in TREE_SEARCH_DIRS:
                cand_path = os.path.normpath(candidate)
                if os.path.exists(cand_path) and cand_path not in [os.path.normpath(d) for d in search_dirs]:
                    search_dirs.append(cand_path)

        loaded_domains = set()
        for s_dir in search_dirs:
            if not os.path.exists(s_dir):
                continue
            all_fnames = [f for f in os.listdir(s_dir) if f.endswith(".json")]
            normalized_fnames = [f for f in all_fnames if f.endswith("_normalized.json")]
            target_fnames = normalized_fnames if (normalized_fnames and s_dir == target_dir) else all_fnames
            for fname in sorted(target_fnames):
                base_domain = fname.split("_v1.")[0].replace("_normalized.json", "").replace(".json", "").split("#")[0].replace("_raw", "").replace("_merged", "")
                if base_domain in loaded_domains:
                    continue
                fpath = os.path.join(s_dir, fname)
                self.load_tree_file(fpath)
                loaded_domains.add(base_domain)

    def load_from_database(self, db_path: Optional[str] = None):
        """Loads nodes from the compiled SQLite database.

        Falls back loudly to the JSON trees when the database is empty or
        unreadable — an empty/aborted compile must never silently produce a
        zero-cell lattice.
        """
        with self._lock:
            if db_path is not None:
                self.db_path = db_path
            if not os.path.exists(self.db_path):
                logger.warning(f"[LATTICE] Database not found at {self.db_path}")
                return

            self.loaded_cells.clear()
            self._bridge_path_cache.clear()
            self._directly_unify_cache.clear()
            self._valid_bridge_cache.clear()
            self._macro_edges = None
            self._generic_port_cache = None
            self._version += 1
            try:
                with contextlib.closing(sqlite3.connect(self.db_path, check_same_thread=False)) as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='types'")
                    if cursor.fetchone():
                        cursor.execute("SELECT type_name, parent_type FROM types")
                        reg = TypeRegistry.get_instance()
                        for t_name, p_type in cursor.fetchall():
                            if t_name:
                                reg.register_type(t_name, p_type or None)

                    # Declared typestate vocabulary (state hierarchy + verifiable
                    # properties) — same data the JSON trees carry in their
                    # `typestates` blocks; persisted here by tools/compile_trees.py.
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='typestates'")
                    if cursor.fetchone():
                        reg = TypeRegistry.get_instance()
                        cursor.execute("PRAGMA table_info(typestates)")
                        ts_cols = {r[1] for r in cursor.fetchall()}
                        ts_dom_sel = "domain" if "domain" in ts_cols else ("domain_name" if "domain_name" in ts_cols else "NULL AS domain")
                        cursor.execute(
                            f"SELECT state_name, parent_state, carrier_type, properties, {ts_dom_sel} FROM typestates"
                        )
                        for s_name, s_parent, s_carrier, s_props_json, s_dom in cursor.fetchall():
                            if not s_name:
                                continue
                            props: Optional[Dict[str, Any]] = None
                            if s_props_json:
                                try:
                                    parsed = json.loads(s_props_json)
                                    props = parsed if isinstance(parsed, dict) else None
                                except Exception as e:
                                    props = None
                            reg.register_state(
                                s_name,
                                s_parent,
                                carrier_type=s_carrier,
                                properties=props,
                                domain=str(s_dom).strip().lower() if s_dom else None,
                            )

                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='aliases'")
                    if cursor.fetchone():
                        reg = TypeRegistry.get_instance()
                        cursor.execute("SELECT alias, canonical FROM aliases")
                        for a_k, a_v in cursor.fetchall():
                            if a_k and a_v:
                                reg.register_alias(a_k, a_v)

                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_readers'")
                    if cursor.fetchone():
                        reg = TypeRegistry.get_instance()
                        cursor.execute("SELECT category, module_name, function_name FROM artifact_readers")
                        for cat, m_name, f_name in cursor.fetchall():
                            if cat and m_name and f_name:
                                reg.register_artifact_reader(cat, m_name, f_name)

                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='structural_metadata'")
                    if cursor.fetchone():
                        reg = TypeRegistry.get_instance()
                        cursor.execute("SELECT category, item, extra FROM structural_metadata")
                        for cat, item, extra in cursor.fetchall():
                            if not item:
                                continue
                            if cat == 'advisory_qualifier':
                                try:
                                    reg.register_advisory_qualifiers([json.loads(item)])
                                except Exception:
                                    pass
                            elif cat == 'abstract_carrier':
                                reg.register_abstract_carriers([item])
                            elif cat == 'abstract_carrier_mapping':
                                reg.register_abstract_carrier_mapping({item: extra})
                            elif cat == 'data_bearing_role':
                                reg.register_data_bearing_roles([item])
                            elif cat == 'type_var':
                                reg.register_type_vars([item])
                            elif cat == 'top_type':
                                reg.register_top(item)
                            elif cat == 'top_state':
                                reg.register_top_state(item)
                            elif cat == 'product_constructor':
                                reg.register_product_constructor(item)
                            elif cat == 'terminal_state':
                                reg.register_terminal_states([item])
                            elif cat == 'role_carrier':
                                reg.register_role_carrier(item)
                            elif cat == 'top_level_carrier':
                                reg.register_top_level_carriers([item])
                            elif cat == 'identifier_root_state':
                                reg.register_identifier_root_states([item])
                            elif cat == 'supported_format':
                                reg.register_supported_formats([item])
                            elif cat == 'state_role_ancestor':
                                try:
                                    reg.register_state_role_ancestors({item: json.loads(extra)})
                                except Exception:
                                    pass
                            elif cat == 'state_role_marker':
                                try:
                                    reg.register_state_role_markers({item: json.loads(extra)})
                                except Exception:
                                    pass
                            elif cat == 'carrier_role':
                                try:
                                    reg.register_carrier_roles({item: json.loads(extra)})
                                except Exception:
                                    pass
                            elif cat == 'type_default':
                                try:
                                    if ":" in item:
                                        td_dom, td_key = item.split(":", 1)
                                    else:
                                        td_dom, td_key = None, item
                                    reg.register_type_default(td_key, json.loads(extra), domain=td_dom)
                                except Exception:
                                    pass
                            elif cat == 'capability':
                                try:
                                    reg.register_capabilities({item: json.loads(extra)})
                                except Exception:
                                    pass
                            elif cat == 'language_spec':
                                try:
                                    reg.register_language_spec(json.loads(extra))
                                except Exception:
                                    pass
                            elif cat == 'core_domain':
                                reg._core_domains.add(str(item).strip().lower())
                            elif cat == 'exclusive_qualifier':
                                try:
                                    reg.register_exclusive_qualifiers([json.loads(extra if extra is not None else item)])
                                except Exception:
                                    pass
                            elif cat == 'exclusive_role':
                                try:
                                    reg.register_exclusive_roles([json.loads(extra if extra is not None else item)])
                                except Exception:
                                    pass
                            elif cat == 'dependent_port':
                                try:
                                    reg.register_dependent_ports({item: json.loads(extra)})
                                except Exception:
                                    pass
                            elif cat == 'forbidden_side_effect':
                                reg.register_forbidden_side_effects([item])
                            elif cat == 'type_kind':
                                reg.register_type_kind(item, extra or "")
                            elif cat == 'type_property':
                                try:
                                    reg.register_type_properties(item, json.loads(extra))
                                except Exception:
                                    pass
                            elif cat == 'type_empty_expr':
                                try:
                                    if ":" in item:
                                        te_dom, te_key = item.split(":", 1)
                                    else:
                                        te_dom, te_key = None, item
                                    reg.register_type_empty_expr(te_key, json.loads(extra), domain=te_dom)
                                except Exception:
                                    pass

                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('nodes', 'cells')")
                    tables = [row[0] for row in cursor.fetchall()]

                    if "nodes" in tables:
                        cursor.execute("PRAGMA table_info(nodes)")
                        col_names = {row[1] for row in cursor.fetchall()}

                        doc_sel = "docstring" if "docstring" in col_names else "'' AS docstring"
                        prio_sel = "source_priority" if "source_priority" in col_names else "100 AS source_priority"
                        role_sel = "node_role" if "node_role" in col_names else "'function' AS node_role"
                        type_sel = "node_type" if "node_type" in col_names else "'function' AS node_type"
                        ver_sel = "verified" if "verified" in col_names else "1 AS verified"
                        dom_sel = "domain_name" if "domain_name" in col_names else "'' AS domain_name"
                        deps_sel = "dependencies" if "dependencies" in col_names else "'' AS dependencies"
                        cfg_sel = "configuration_schema" if "configuration_schema" in col_names else "'' AS configuration_schema"
                        slots_sel = "slots" if "slots" in col_names else "'' AS slots"
                        raw_sel = "raw_cell" if "raw_cell" in col_names else "NULL AS raw_cell"
                        if "raw_cell" not in col_names:
                            logger.warning(
                                "[LATTICE] %s predates the raw_cell column: cells are rebuilt from lossy columns "
                                "(port `binds`, semantic tags, raises, ... are NOT preserved). "
                                "Recompile with `compile --clean`.", self.db_path)

                        cursor.execute(f"""
                            SELECT cell_id, {dom_sel}, {type_sel}, {role_sel}, stage,
                                   keywords, input_type, input_state, output_type, output_state,
                                   code, {deps_sel}, {cfg_sel}, {slots_sel}, {ver_sel}, {doc_sel}, {prio_sel}, {raw_sel}
                            FROM nodes
                        """)
                        for row in cursor.fetchall():
                            (cell_id, domain_name, node_type, node_role, stage,
                             keywords_json, in_type, in_state, out_type, out_state,
                             code, deps_json, config_json, slots_json, verified, doc_str, source_priority, raw_json) = row

                            if raw_json:
                                # Faithful path: identical construction to the JSON loader.
                                try:
                                    cell = build_cell_from_raw(json.loads(raw_json), domain_name or "generic", default_verified=True)
                                    cell.source_provenance = f"sqlite:{Path(self.db_path).name}"
                                    self.loaded_cells[cell.cell_id] = cell
                                    continue
                                except Exception as _raw_err:
                                    logger.warning("[LATTICE] raw_cell for %s unusable (%s); falling back to lossy columns",
                                                   cell_id, _raw_err)

                            try:
                                keywords = set(json.loads(keywords_json)) if keywords_json else set()
                            except Exception as e:
                                keywords = set()
                            try:
                                deps = json.loads(deps_json) if deps_json else []
                            except Exception as e:
                                deps = []
                            try:
                                cfg = json.loads(config_json) if config_json else {}
                            except Exception as e:
                                cfg = {}

                            slots_val = cfg.get("slots", {})
                            if not slots_val and slots_json:
                                try:
                                    slots_val = json.loads(slots_json)
                                except Exception as e:
                                    slots_val = {}
                            sub_cells_val = cfg.get("sub_cells", [])

                            in_sig = AlgebraicSignature(type_name=in_type or "any", state=in_state or "any")
                            out_sig = AlgebraicSignature(type_name=out_type or "None", state=out_state or "any")

                            inputs: Dict[str, PortSignature] = {}
                            outputs: Dict[str, PortSignature] = {}

                            if isinstance(cfg, dict) and ("inputs" in cfg or "outputs" in cfg):
                                dom_l = str(domain_name or "generic").strip().lower()
                                for p_name, p_val in cfg.get("inputs", {}).items():
                                    if isinstance(p_val, dict):
                                        abs_t = _clean_abs_carrier(p_val.get("abstract_type"))
                                        acc_s = tuple(sorted({str(s).strip().lower() for s in (p_val.get("accepted_states") or []) if str(s).strip()}))
                                        p_s = str(p_val.get("parent_state") or "").strip().lower() or None
                                        p_dom = str(p_val.get("domain") or dom_l or "").strip().lower() or None
                                        if p_s and p_val.get("state"):
                                            TypeRegistry.get_instance().register_state(p_val.get("state"), p_s, domain=p_dom)
                                        inputs[p_name] = PortSignature(
                                            name=p_name,
                                            signature=AlgebraicSignature(
                                                type_name=str(p_val.get("type_name", in_type or "any")),
                                                state=str(p_val.get("state", in_state or "any")),
                                                qualifiers=_normalize_qualifiers(p_val.get("qualifiers", [])),
                                                abstract_type=abs_t,
                                                accepted_states=acc_s,
                                                parent_state=p_s,
                                            ),
                                            required=p_val.get("required", True),
                                            default_value=p_val.get("default_value"),
                                            doc=p_val.get("doc", p_val.get("description", "")),
                                            domain=p_dom or "",
                                            abstract_type=abs_t,
                                            enum_values=p_val.get("enum_values"),
                                            param_kind=p_val.get("param_kind", "standard"),
                                            value_constraints=p_val.get("value_constraints"),
                                            shape_contract=p_val.get("shape_contract"),
                                            rejected_qualifiers=p_val.get("rejected_qualifiers", []),
                                            accepted_states=acc_s,
                                            parent_state=p_s,
                                            port_role=p_val.get("port_role"),
                                            binds=p_val.get("binds"),
                                            structure_kind=p_val.get("structure_kind"),
                                            rank=p_val.get("rank"),
                                            element_type=p_val.get("element_type"),
                                            supported_formats=p_val.get("supported_formats"),
                                            complement_of_port=p_val.get("complement_of_port"),
                                            disjoint_from_port=p_val.get("disjoint_from_port"),
                                            key_from_upstream_role=p_val.get("key_from_upstream_role"),
                                            carrier=p_val.get("carrier"),
                                            nominal_type=p_val.get("nominal_type"),
                                            layout=p_val.get("layout"),
                                            semantics=p_val.get("semantics"),
                                            typestate=p_val.get("typestate"),
                                            invariants=p_val.get("invariants"),
                                            effect_suspended=bool(p_val.get("effect_suspended", False)),
                                        )
                                for p_name, p_val in cfg.get("outputs", {}).items():
                                    if isinstance(p_val, dict):
                                        abs_t = _clean_abs_carrier(p_val.get("abstract_type"))
                                        acc_s = tuple(sorted({str(s).strip().lower() for s in (p_val.get("accepted_states") or []) if str(s).strip()}))
                                        p_s = str(p_val.get("parent_state") or "").strip().lower() or None
                                        p_dom = str(p_val.get("domain") or dom_l or "").strip().lower() or None
                                        if p_s and p_val.get("state"):
                                            TypeRegistry.get_instance().register_state(p_val.get("state"), p_s, domain=p_dom)
                                        outputs[p_name] = PortSignature(
                                            name=p_name,
                                            signature=AlgebraicSignature(
                                                type_name=str(p_val.get("type_name", out_type or "any")),
                                                state=str(p_val.get("state", out_state or "any")),
                                                qualifiers=_normalize_qualifiers(p_val.get("qualifiers", [])),
                                                abstract_type=abs_t,
                                                accepted_states=acc_s,
                                                parent_state=p_s,
                                            ),
                                            required=p_val.get("required", True),
                                            default_value=p_val.get("default_value"),
                                            doc=p_val.get("doc", p_val.get("description", "")),
                                            domain=p_dom or "",
                                            abstract_type=abs_t,
                                            enum_values=p_val.get("enum_values"),
                                            param_kind=p_val.get("param_kind", "standard"),
                                            value_constraints=p_val.get("value_constraints"),
                                            shape_contract=p_val.get("shape_contract"),
                                            accepted_states=acc_s,
                                            parent_state=p_s,
                                            port_role=p_val.get("port_role"),
                                            binds=p_val.get("binds"),
                                            structure_kind=p_val.get("structure_kind"),
                                            rank=p_val.get("rank"),
                                            element_type=p_val.get("element_type"),
                                            supported_formats=p_val.get("supported_formats"),
                                            complement_of_port=p_val.get("complement_of_port"),
                                            disjoint_from_port=p_val.get("disjoint_from_port"),
                                            key_from_upstream_role=p_val.get("key_from_upstream_role"),
                                            carrier=p_val.get("carrier"),
                                            nominal_type=p_val.get("nominal_type"),
                                            layout=p_val.get("layout"),
                                            semantics=p_val.get("semantics"),
                                            typestate=p_val.get("typestate"),
                                            invariants=p_val.get("invariants"),
                                            effect_suspended=bool(p_val.get("effect_suspended", False)),
                                        )

                            if not inputs and (not isinstance(cfg, dict) or "inputs" not in cfg):
                                inputs = {"input_data": PortSignature("input_data", in_sig)}
                            if not outputs and (not isinstance(cfg, dict) or "outputs" not in cfg):
                                outputs = {"output_data": PortSignature("output_data", out_sig)}

                            top_val = str(cfg.get("topology_type", "") if isinstance(cfg, dict) else "").lower()
                            is_macro = is_macro_cell_dict(
                                {
                                    "node_type": node_type,
                                    "node_role": node_role,
                                    "topology_type": top_val,
                                    "sub_cells": sub_cells_val,
                                    "slots": slots_val,
                                },
                                reg=TypeRegistry.get_instance(),
                                check_slots=True,
                            )
                            cls = MacroCell if is_macro else MicroCell
                            cell = cls(
                                cell_id=cell_id,
                                stage=stage or 2,
                                keywords=keywords,
                                inputs=inputs,
                                outputs=outputs,
                                domain_name=domain_name or "generic",
                                node_type=node_type or "function",
                                node_role=node_role or "function",
                                dependencies=deps,
                                code_template=code or "",
                                metadata_tags=cfg.get("metadata_tags", {}),
                                configuration_schema=cfg,
                                slots=slots_val,
                                verified=bool(verified),
                                docstring=doc_str or "",
                                source_priority=int(source_priority) if source_priority is not None else 100,
                                source_provenance=f"sqlite:{Path(self.db_path).name}",
                                topology_type=cfg.get("topology_type", "sequential"),
                                feedback_state_type=cfg.get("feedback_state_type"),
                                bound_slots=cfg.get("bound_slots", {}),
                                is_public=bool(cfg.get("is_public", True)),
                                mutation_type=cfg.get("mutation_type", "pure"),
                                is_context_manager=cfg.get("is_context_manager", False),
                                raises=cfg.get("raises", []),
                                type_vars=cfg.get("type_vars", []),
                                preconditions=cfg.get("preconditions", []),
                                postconditions=cfg.get("postconditions", []),
                                effects=cfg.get("effects", []),
                                edges=cfg.get("edges", []),
                                endable=cfg.get("endable"),
                                primary_in=cfg.get("primary_in"),
                                primary_out=cfg.get("primary_out"),
                                sub_cells=cfg.get("sub_cells", []),
                                algorithmic_steps=cfg.get("algorithmic_steps", []),
                                internal_topology=cfg.get("internal_topology", {}),
                                projection=cfg.get("projection"),
                                optic_role=cfg.get("optic_role"),
                                container_port=cfg.get("container_port"),
                                key_port=cfg.get("key_port"),
                                focus_port=cfg.get("focus_port"),
                            )
                            self.loaded_cells[cell.cell_id] = cell

                    elif "cells" in tables:
                        domain_name = Path(self.db_path).stem.lower()
                        cursor.execute("""
                            SELECT cell_id, stage, input_type, input_state, output_type, output_state,
                                   code_template, configuration_schema, dependencies
                            FROM cells
                        """)
                        for row in cursor.fetchall():
                            (cell_id, stage, in_type, in_state, out_type, out_state,
                             code, config_json, deps_json) = row

                            try:
                                deps = json.loads(deps_json) if deps_json else []
                            except Exception as e:
                                deps = []
                            try:
                                cfg = json.loads(config_json) if config_json else {}
                            except Exception as e:
                                cfg = {}

                            in_sig = AlgebraicSignature(type_name=in_type or "any", state=in_state or "any")
                            out_sig = AlgebraicSignature(type_name=out_type or "None", state=out_state or "any")

                            inputs: Dict[str, PortSignature] = {}
                            outputs: Dict[str, PortSignature] = {}

                            if isinstance(cfg, dict) and ("inputs" in cfg or "outputs" in cfg):
                                for p_name, p_val in cfg.get("inputs", {}).items():
                                    if isinstance(p_val, dict):
                                        inputs[p_name] = PortSignature(
                                            name=p_name,
                                            signature=AlgebraicSignature(
                                                type_name=str(p_val.get("type_name", in_type or "any")),
                                                state=str(p_val.get("state", in_state or "any")),
                                                qualifiers=_normalize_qualifiers(p_val.get("qualifiers", []))
                                            ),
                                            required=p_val.get("required", True),
                                            default_value=p_val.get("default_value")
                                        )
                                for p_name, p_val in cfg.get("outputs", {}).items():
                                    if isinstance(p_val, dict):
                                        outputs[p_name] = PortSignature(
                                            name=p_name,
                                            signature=AlgebraicSignature(
                                                type_name=str(p_val.get("type_name", out_type or "any")),
                                                state=str(p_val.get("state", out_state or "any")),
                                                qualifiers=_normalize_qualifiers(p_val.get("qualifiers", []))
                                            )
                                        )

                            if not inputs:
                                inputs = {"input_data": PortSignature("input_data", in_sig)}
                            if not outputs:
                                outputs = {"output_data": PortSignature("output_data", out_sig)}

                            keywords = CellTokenizer.tokenize_identifier(cell_id)

                            cell = MicroCell(
                                cell_id=cell_id,
                                stage=stage or 2,
                                keywords=keywords,
                                inputs=inputs,
                                outputs=outputs,
                                domain_name=domain_name or Path(self.db_path).stem.lower(),
                                dependencies=deps,
                                code_template=code or "",
                                verified=True,
                                endable=cfg.get("endable"),
                                primary_in=cfg.get("primary_in"),
                                primary_out=cfg.get("primary_out"),
                                projection=cfg.get("projection")
                            )
                            if cell.is_public_morphism:
                                self.loaded_cells[cell.cell_id] = cell

                logger.info(f"[LATTICE] Loaded {len(self.loaded_cells)} nodes from database.")
                self.build_topology()
            except Exception as e:
                logger.error(f"[LATTICE] Failed loading database: {e}")

            if not self.loaded_cells:
                # Empty or failed database compile: NEVER ship a silently-empty
                # lattice — rebuild from the JSON trees instead.
                logger.warning(
                    f"[LATTICE] Database '{self.db_path}' yielded 0 nodes; "
                    f"falling back to JSON trees in '{self.trees_directory}'."
                )
                self.loaded_cells.clear()
                self._bridge_path_cache.clear()
                self._directly_unify_cache.clear()
                self._macro_edges = None
                self.load_all_json_trees()

    load_from_db = load_from_database

    def build_topology(self, edge_mode: Optional[str] = None):
        """
        Builds the lattice directed edges based strictly on declared edges and 5-axis compatibility:
          (u, v) in E <=> v.cell_id in u.edges or is_edge_valid(u.primary_output, v.primary_input)
        Supports both 'static' (eager precomputed 5-axis adjacency) and 'live' (lazy + live 5-axis).
        """
        with self._lock:
            self._adjacency.clear()
            self._reverse_adjacency.clear()
            self._token_index.clear()
            self._bridge_cells.clear()
            self.dynamic_edges.clear()
            self._bridge_path_cache.clear()
            self._directly_unify_cache.clear()
            self._valid_bridge_cache.clear()
            self._macro_edges = None

            all_cells = list(self.loaded_cells.values())
            for cell in all_cells:
                _apply_declared_estimator_identity(cell)
                _ = cell.token_set  # Warm up cached token set
                for tok in cell.token_set:
                    self._token_index.setdefault(tok, []).append(cell)
                if (
                    getattr(cell, "node_role", "") == "bridge"
                    or getattr(cell, "node_type", "") == "tunnel"
                    or self._is_valid_bridge_cached(cell)
                ):
                    self._bridge_cells.append(cell)
            try:
                from .experiments import build_indexed_topology, ACTIVE_EXPERIMENTS
            except (ImportError, ValueError):
                from experiments import build_indexed_topology, ACTIVE_EXPERIMENTS

            eff_mode = str(
                edge_mode
                or self.edge_mode
                or getattr(ACTIVE_EXPERIMENTS, "edge_mode", None)
                or getattr(settings, "edge_mode", "live")
            ).strip().lower()
            if eff_mode not in ("static", "live"):
                eff_mode = "live"
            self.edge_mode = eff_mode

            self._adjacency, self._reverse_adjacency = build_indexed_topology(
                all_cells,
                self.loaded_cells,
                TypeRegistry.get_instance(),
                edge_mode=eff_mode,
            )


            # 3. Macro internal topology synaptic edges (synapses connecting constituent micro-cells)
            # In live mode, append macro seeds without triggering LazyAdjacencyMap._resolve upfront.
            def _append_macro_edge(adj_map: Any, src_cid: str, dst_cid: str) -> None:
                seeds = getattr(adj_map, "_seed_lists", None)
                cache = getattr(adj_map, "_cache", None)
                if isinstance(seeds, dict) and isinstance(cache, dict) and src_cid not in cache:
                    seed_bucket = seeds.setdefault(src_cid, [])
                    if dst_cid not in seed_bucket:
                        seed_bucket.append(dst_cid)
                    return
                if dst_cid not in adj_map[src_cid]:
                    adj_map[src_cid].append(dst_cid)

            for m in all_cells:
                if getattr(m, "cell_type", "") == "macro" or getattr(m, "sub_cells", None):
                    topo = getattr(m, "internal_topology", {}) or {}
                    for src_id, targets in topo.items():
                        if src_id in self.loaded_cells:
                            tgt_list = targets if isinstance(targets, list) else [targets]
                            for tgt_id in tgt_list:
                                if tgt_id in self.loaded_cells:
                                    _append_macro_edge(self._adjacency, src_id, tgt_id)
                                    _append_macro_edge(self._reverse_adjacency, tgt_id, src_id)
                                    self.dynamic_edges[(src_id, tgt_id)] = {
                                        "affinity_score": 0.9,
                                        "score_provenance": "macro_synapse",
                                        "macro_id": m.cell_id
                                    }
                    subs = getattr(m, "sub_cells", []) or []
                    for i in range(len(subs) - 1):
                        s_u, s_v = subs[i], subs[i + 1]
                        if s_u in self.loaded_cells and s_v in self.loaded_cells:
                            _append_macro_edge(self._adjacency, s_u, s_v)
                            _append_macro_edge(self._reverse_adjacency, s_v, s_u)
                            if (s_u, s_v) not in self.dynamic_edges:
                                self.dynamic_edges[(s_u, s_v)] = {
                                    "affinity_score": 0.9,
                                    "score_provenance": "macro_synapse",
                                    "macro_id": m.cell_id
                                }
                    if hasattr(m, "_resolved_sub_cells"):
                        m._resolved_sub_cells = {
                            sid: self.loaded_cells[sid]
                            for sid in (getattr(m, "sub_cells", []) or [])
                            if sid in self.loaded_cells
                        }

            reg = TypeRegistry.get_instance()

            # Register formal operation tokens derived strictly from canonical cell_id identifiers
            op_tokens_acc: Set[str] = set()
            mat_states_acc: Set[str] = set()
            src_tokens_acc: Set[str] = set()
            id_roots_acc: Set[str] = set()
            role_anc_acc: Dict[str, Set[str]] = {}
            data_roles_acc: Set[str] = set()
            dest_roles_acc: Set[str] = set()
            egress_tokens_acc: Set[str] = set()
            in_port_names_acc: Set[str] = set()
            ing_kw_acc: Set[str] = set()
            sup_fmts_lower = {str(f).strip().lower() for f in (reg.get_registered_supported_formats() or ())}

            corpus_docs = [c.docstring for c in all_cells if getattr(c, "docstring", None)]
            if corpus_docs:
                reg.derive_function_words(corpus_docs)
                for _c_fw in all_cells:
                    _c_fw._identity_tokens = None
            func_words_lower = {str(w).strip().lower() for w in (reg.get_function_words() or ())}

            all_domain_toks: Set[str] = set()
            dom_non_sink_toks: Dict[str, Set[str]] = {}
            dom_port_toks: Dict[str, Set[str]] = {}
            path_port_toks_global: Set[str] = set()
            for c_pre in all_cells:
                pre_dom = str(getattr(c_pre, "domain_name", "") or "").strip().lower()
                if pre_dom:
                    all_domain_toks.update(t for t in CellTokenizer.tokenize_identifier(pre_dom) if "_" not in t)
                pre_in = getattr(c_pre, "inputs", {}) or {}
                pre_out = getattr(c_pre, "outputs", {}) or {}
                dp_set = dom_port_toks.setdefault(pre_dom, set())
                for pp in list(pre_in.values()) + list(pre_out.values()):
                    for fmt in (getattr(pp, "supported_formats", None) or ()):
                        sup_fmts_lower.add(str(fmt).strip().lower())
                    p_desc = str(getattr(pp, "description", "") or "").lower()
                    for dw in p_desc.split():
                        dw_cl = dw.strip(".,;:()[]{}'\"")
                        if dw_cl.startswith(".") and len(dw_cl) > 1 and dw_cl[1:].isalnum():
                            sup_fmts_lower.add(dw_cl[1:])
                    p_st_s = str(getattr(getattr(pp, "signature", pp), "state", "") or "").lower()
                    p_nm_s = str(getattr(pp, "name", "") or "").lower()
                    p_sub = {
                        t for s in (p_st_s, p_nm_s) if s
                        for t in CellTokenizer.tokenize_identifier(s) if "_" not in t
                    }
                    dp_set.update(p_sub)
                    if is_path_port(pp, reg) or str(getattr(pp, "port_role", "") or "").lower() in (
                        reg.get_roles_for_category("destination") | reg.get_terminal_roles_set()
                    ):
                        path_port_toks_global.update(p_sub)
                pre_is_sink = (
                    getattr(c_pre, "stage", None) == 3
                    or reg.is_terminal_role(getattr(c_pre, "node_role", ""))
                    or reg.is_terminal_role(getattr(c_pre, "mutation_type", ""))
                )
                if not pre_is_sink:
                    ns_set = dom_non_sink_toks.setdefault(pre_dom, set())
                    for raw_s in (
                        [c_pre.cell_id]
                        + [str(k) for k in (getattr(c_pre, "keywords", None) or ())]
                        + [str(t) for t in (getattr(c_pre, "semantic_tags", None) or ())]
                    ):
                        ns_set.update(t for t in CellTokenizer.tokenize_identifier(raw_s) if "_" not in t)

            non_data_roles = (
                reg.get_config_roles_set()
                | reg.get_roles_for_category("key")
                | reg.get_roles_for_category("projection")
            )
            declared_dataflow = (
                reg.get_roles_for_category("data")
                | reg.get_roles_for_category("dataflow")
                | reg.get_roles_for_category("strict")
                | reg.get_roles_for_category("model")
            )

            # Harvest structural materialization states, terminal states, and port roles from cell signatures
            for c in all_cells:
                op_tokens_acc.update(CellTokenizer.tokenize_identifier_frozen(c.cell_id))
                c_inputs = getattr(c, "inputs", {})
                c_outputs = getattr(c, "outputs", {})
                c_role_low = str(getattr(c, "node_role", "")).lower()
                c_mut_low = str(getattr(c, "mutation_type", "")).lower()
                term_roles = reg.get_terminal_roles_set()
                sink_only_roles = reg.get_roles_for_category("sink")
                non_egress_roles = (
                    reg.get_ingress_roles_set()
                    | reg.get_roles_for_category("transform")
                    | reg.get_roles_for_category("estimator")
                    | reg.get_roles_for_category("evaluator")
                    | reg.get_roles_for_category("clean")
                )
                is_egress = (
                    c_role_low in sink_only_roles
                    or c_mut_low in sink_only_roles
                    or (
                        getattr(c, "stage", None) == 3
                        and c_role_low not in non_egress_roles
                        and c_mut_low != "pure"
                    )
                )
                if is_egress:
                    for p in c_outputs.values():
                        p_st = getattr(getattr(p, "signature", p), "state", None)
                        if p_st and not reg.is_declared_top_state(p_st):
                            mat_states_acc.add(str(p_st))

                is_ingress = (
                    getattr(c, "stage", None) == 1
                    or reg.is_ingress_role(c_role_low)
                    or c_mut_low in reg.get_ingress_roles_set()
                )
                if is_ingress:
                    for p in c_inputs.values():
                        if is_path_port(p, reg) or getattr(p, "abstract_type", "") == "path":
                            p_name = getattr(p, "name", "")
                            if p_name:
                                src_tokens_acc.add(p_name)
                            p_st = getattr(getattr(p, "signature", p), "state", None)
                            if p_st and not reg.is_declared_top(str(p_st)):
                                id_roots_acc.add(str(p_st))

                for p in c_inputs.values():
                    role = getattr(p, "port_role", None) or getattr(p, "role", None)
                    p_st = getattr(getattr(p, "signature", p), "state", None)
                    if role and p_st and not reg.is_declared_top(str(p_st)):
                        role_anc_acc.setdefault(str(role), set()).add(str(p_st))
                    r_low = str(role or "").strip().lower()
                    if r_low and r_low not in non_data_roles:
                        p_tn = str(getattr(getattr(p, "signature", p), "type_name", "") or "").strip().lower()
                        if r_low in declared_dataflow or reg.is_composite_carrier(p_tn) or reg.is_abstract_carrier(p_tn):
                            data_roles_acc.add(r_low)

                is_ing_cell = getattr(c, "stage", None) == 1 or reg.is_ingress_role(getattr(c, "node_role", ""))
                if is_ing_cell:
                    ing_kws = getattr(c, "keywords", None) or ()
                    for kw in ing_kws:
                        k_str = str(kw).strip().lower()
                        if k_str and len(k_str) >= 2:
                            ing_kw_acc.add(k_str)
                    if not ing_kws:
                        c_dom_ing = str(getattr(c, "domain_name", "") or "").strip().lower()
                        c_dom_ing_toks = set(CellTokenizer.tokenize_identifier(c_dom_ing)) if c_dom_ing else set()
                        id_parts = [p for p in str(c.cell_id).lower().split("_") if p and len(p) >= 2 and p not in c_dom_ing_toks]
                        if id_parts and id_parts[0] not in func_words_lower and id_parts[0] not in all_domain_toks:
                            ing_kw_acc.add(id_parts[0])
                is_sink_cell = getattr(c, "stage", None) == 3 or reg.is_terminal_role(getattr(c, "node_role", "")) or reg.is_terminal_role(getattr(c, "mutation_type", ""))
                for pn_in in c_inputs.keys():
                    pn_low = str(pn_in).strip().lower()
                    if pn_low:
                        in_port_names_acc.add(pn_low)
                dest_roles_s = reg.get_roles_for_category("destination") | reg.get_terminal_roles_set()
                if is_sink_cell and any(
                    is_path_port(p, reg) or str(getattr(p, "port_role", "") or "").lower() in dest_roles_s
                    for p in list(c_inputs.values()) + list(c_outputs.values())
                ):
                    c_dom = str(getattr(c, "domain_name", "") or "").strip().lower()
                    c_dom_toks = set(CellTokenizer.tokenize_identifier(c_dom)) if c_dom else set()
                    c_fmts = set(sup_fmts_lower)
                    for p in list(c_inputs.values()) + list(c_outputs.values()):
                        for fmt in (getattr(p, "supported_formats", None) or ()):
                            c_fmts.add(str(fmt).strip().lower())
                    c_kws = getattr(c, "keywords", None) or ()
                    for kw in c_kws:
                        k_str = str(kw).strip().lower()
                        if k_str and len(k_str) >= 2 and k_str not in c_dom_toks and k_str not in c_fmts:
                            egress_tokens_acc.add(k_str)
                    # When a sink omits explicit keywords, harvest action verbs from cell_id and semantic_tags
                    # for dataflow-to-destination sinks, excluding domain/port/format/function/non-sink nouns.
                    has_data_in = any(
                        not is_path_port(p, reg) and str(getattr(p, "port_role", "") or "").lower() not in (non_data_roles | dest_roles_s)
                        for p in c_inputs.values()
                    )
                    if not c_kws and has_data_in:
                        c_excl = (
                            c_dom_toks
                            | all_domain_toks
                            | c_fmts
                            | func_words_lower
                            | dom_non_sink_toks.get(c_dom, set())
                            | dom_port_toks.get(c_dom, set())
                        )
                        id_parts = [p for p in str(c.cell_id).lower().split("_") if p and len(p) >= 2]
                        h_idx = 0
                        while h_idx < len(id_parts) and (id_parts[h_idx] in c_dom_toks or id_parts[h_idx] in all_domain_toks):
                            h_idx += 1
                        if h_idx < len(id_parts) and id_parts[h_idx] not in c_excl:
                            egress_tokens_acc.add(id_parts[h_idx])
                        tail_target_toks = path_port_toks_global | c_fmts | dom_port_toks.get(c_dom, set())
                        for tag in (getattr(c, "semantic_tags", None) or ()):
                            t_parts = [p for p in str(tag).lower().split("_") if p and len(p) >= 2]
                            if len(t_parts) >= 2 and t_parts[0] not in c_excl and any(tp in tail_target_toks for tp in t_parts[1:]):
                                egress_tokens_acc.add(t_parts[0])
                for p in c_outputs.values():
                    role = getattr(p, "port_role", None) or getattr(p, "role", None)
                    p_st = getattr(getattr(p, "signature", p), "state", None)
                    if role and p_st and not reg.is_declared_top(str(p_st)):
                        role_anc_acc.setdefault(str(role), set()).add(str(p_st))
                    r_low = str(role or "").strip().lower()
                    if r_low and is_sink_cell and (is_path_port(p, reg) or getattr(p, "abstract_type", "") == "path"):
                        dest_roles_acc.add(r_low)

            if op_tokens_acc:
                reg.register_operation_tokens(op_tokens_acc)
            if mat_states_acc:
                reg.register_terminal_states(mat_states_acc)
            if id_roots_acc:
                reg.register_identifier_root_states(id_roots_acc)
            if role_anc_acc:
                reg.register_state_role_ancestors(role_anc_acc)
            if data_roles_acc:
                reg.register_data_bearing_roles(data_roles_acc)
            if egress_tokens_acc:
                param_or_src = (in_port_names_acc | src_tokens_acc | ing_kw_acc) - dest_roles_acc
                egress_tokens_acc = {t for t in egress_tokens_acc if t not in param_or_src}
                if egress_tokens_acc:
                    reg.register_egress_tokens(egress_tokens_acc)
                reg_types = {
                    t for t in reg.get_registered_types()
                    if "ast" not in {str(p).lower() for p in reg._parents.get(t, ())}
                }
                fw_cur = getattr(reg, "_function_words", None)
                if fw_cur:
                    reg._function_words = tuple(w for w in fw_cur if w not in reg_types)

            # Dynamic Advisory Qualifier Harvesting (gated behind allow_advisory_qualifier_waiver
            # so undeclared input qualifiers are enforced strictly by default unless explicitly waived).
            try:
                from config import settings as _adv_settings
                _allow_adv_harvest = bool(getattr(_adv_settings, "allow_advisory_qualifier_waiver", False))
            except Exception:
                _allow_adv_harvest = False

            if _allow_adv_harvest:
                all_rejected_quals = set()
                for c in all_cells:
                    for p in getattr(c, "inputs", {}).values():
                        p_rej = getattr(p, "rejected_qualifiers", None)
                        if p_rej:
                            for r in p_rej:
                                all_rejected_quals.add(str(r).strip().lower())

                owner_quals = getattr(reg, "_owner_quals", set())
                advisory_quals_to_register = set()
                for c in all_cells:
                    for p in getattr(c, "inputs", {}).values():
                        p_sig = getattr(p, "signature", p)
                        quals = getattr(p_sig, "qualifiers", None)
                        if quals:
                            for q in quals:
                                q_norm = tuple(str(item).strip().lower() for item in q) if isinstance(q, (tuple, list, set)) else (str(q).strip().lower(),)
                                if q_norm in owner_quals or q in owner_quals:
                                    continue
                                q_tokens = set(q_norm)
                                if q_tokens.isdisjoint(all_rejected_quals):
                                    advisory_quals_to_register.add(q)
                if advisory_quals_to_register:
                    reg.register_advisory_qualifiers(list(advisory_quals_to_register))

    def audit_topology(self) -> Any:
        try:
            from lattice_auditor import LatticeAuditor
            return LatticeAuditor(self).audit()
        except Exception as e:
            logger.debug(f"[LATTICE] Auditor hook: {e}")
            return None

    @property
    def token_index(self) -> Dict[str, List[Cell]]:
        return self._token_index

    @property
    def bridge_cells(self) -> List[Cell]:
        return self._bridge_cells

    def get_successors(self, cell: Cell, candidate_pool: Optional[List[Cell]] = None) -> List[Cell]:
        """
        Returns all nodes in candidate_pool whose input can be validly chained from cell's output.
        Computed on-demand via monadic unification check.
        """
        pool = candidate_pool if candidate_pool is not None else self.cells
        out_sig = cell.primary_output
        successors = []
        for cand in pool:
            if cand.cell_id == cell.cell_id:
                continue
            if cand.can_accept(out_sig):
                successors.append(cand)
        return successors

    def get_successors_for_sig(self, sig: AlgebraicSignature) -> List[Cell]:
        """Returns cells whose input unifies with the given output signature."""
        results = []
        for c in self.loaded_cells.values():
            if c.can_accept(sig):
                results.append(c)
        return results

    def cells_directly_unify(self, src: Cell, dst: Cell) -> bool:
        """True iff any output port of src can validly unify with an input port of dst."""
        if not src or not dst:
            return False
        cache_key = (src.cell_id, dst.cell_id)
        if cache_key in self._directly_unify_cache:
            return self._directly_unify_cache[cache_key]
        # Semantic port unification check (including typestate qualifiers and states)
        res = False
        try:
            try:
                from .unification import is_edge_valid
            except (ImportError, ValueError):
                from unification import is_edge_valid
            for out_p in src.outputs.values():
                for in_p in dst.inputs.values():
                    if is_edge_valid(out_p, in_p, producer_cell=src, consumer_cell=dst):
                        res = True
                        break
                if res:
                    break
        except Exception:
            pass
        self._directly_unify_cache[cache_key] = res
        return res

    def _is_valid_bridge_cached(self, cand: Cell) -> bool:
        cid = getattr(cand, "cell_id", None)
        if not cid:
            return False
        hit = self._valid_bridge_cache.get(cid)
        if hit is not None:
            return hit
        res = self._is_valid_bridge_candidate(cand)
        self._valid_bridge_cache[cid] = res
        return res

    @staticmethod
    def _is_valid_bridge_candidate(cand: Cell) -> bool:
        """Determines if a cell is suitable to act as a 1-to-1 dataflow precondition bridge.
        A cell is a valid Refinement/Coercion Bridge if it is a deterministic unary dataflow morphism
        (has exactly 1 required data input port without default, >= 1 output port, is not an ingress/stage-1
        or egress/stage-3 sink, and is not a higher-order control-flow combinator) that either:
          a) Carrier Coercion: changes physical type_name (tin -> tout, e.g. Image -> ndarray), OR
          b) State Refinement: preserves physical type_name (tin == tout) while transitioning semantic state (sin -> sout).
        """
        if not cand or not getattr(cand, "outputs", None):
            return False

        reg = TypeRegistry.get_instance()
        cf_domains, cf_topologies, ingress_stages, terminal_stages, cfg_roles = reg.get_bridge_candidate_bundle()
        if (
            getattr(cand, "domain_name", "") in cf_domains
            or getattr(cand, "node_type", "") in cf_topologies
            or getattr(cand, "topology_type", "") in cf_topologies
            or (getattr(cand, "cell_type", "") == "macro" and getattr(cand, "node_role", "") != "bridge")
        ):
            return False

        cand_stage = getattr(cand, "stage", None)
        if (
            (cand_stage is not None and str(cand_stage) in ingress_stages)
            or (cand_stage is not None and str(cand_stage) in terminal_stages)
        ):
            return False

        dataflow_inputs = []
        for p_name, p_sig in (getattr(cand, "inputs", {}) or {}).items():
            if getattr(p_sig, "default_value", None) is not None:
                continue
            p_role = str(getattr(p_sig, "port_role", None) or getattr(p_sig, "role", None) or getattr(p_sig, "derived_role", "") or "").lower()
            if p_role in cfg_roles or str(getattr(p_sig, "param_kind", "")).lower() in cfg_roles:
                continue
            t_str = str(getattr(getattr(p_sig, "signature", p_sig), "type_name", ""))
            if "Callable" in t_str or "function" in t_str.lower():
                return False
            dataflow_inputs.append(p_sig)
        if len(dataflow_inputs) != 1:
            return False

        in_port = dataflow_inputs[0]
        out_port = getattr(cand, "primary_output", None) or next(iter(cand.outputs.values()))
        tin = str(getattr(getattr(in_port, "signature", in_port), "type_name", "")).strip().lower()
        tout = str(getattr(getattr(out_port, "signature", out_port), "type_name", "")).strip().lower()
        sin = str(getattr(getattr(in_port, "signature", in_port), "state", "") or "").strip().lower()
        sout = str(getattr(getattr(out_port, "signature", out_port), "state", "") or "").strip().lower()

        if (
            not tin
            or not tout
            or reg.is_declared_top(tin)
            or reg.is_declared_top(tout)
        ):
            return False

        is_carrier_coercion = (tin != tout)
        is_state_refinement = (
            (tin == tout or (reg.is_subtype(tin, tout) and reg.is_subtype(tout, tin)))
            and (sin != sout)
            and not reg.is_declared_top_state(sout)
        )
        cand_role = str(getattr(cand, "node_role", "") or getattr(cand, "role", "") or "").lower()
        cand_ntype = str(getattr(cand, "node_type", "") or "").lower()
        cand_tags = {str(t).lower() for t in (getattr(cand, "semantic_tags", None) or ())}
        bridge_roles = reg.get_bridge_roles_set()
        bridge_tokens = bridge_roles
        bridge_prof = getattr(cand, "bridge_profile", None)
        has_shape_coercion = bool(getattr(cand, "postconditions", None)) and (
            "shape" in cand_tags or "reshape" in cand_tags
        )
        is_declared = (
            schema_is_bridge_cell(cand, reg)
            or (isinstance(bridge_prof, dict) and bool(bridge_prof.get("is_bridge")))
            or bool(getattr(bridge_prof, "is_bridge", False))
        )
        is_structural_bridge = is_declared or bool(bridge_tokens & cand_tags) or has_shape_coercion
        return is_declared or (is_structural_bridge and (is_carrier_coercion or is_state_refinement))

    def find_shortest_bridge_path(
        self,
        src: Cell,
        dst: Cell,
        max_hops: int = 4,
        candidate_pool: Optional[Collection[Cell]] = None,
    ) -> List[Cell]:
        """
        Unified Shortest-Path Precondition Bridge Closure:
        Computes the minimal bridge morphism chain:
            src -> b_1 -> ... -> b_k -> dst
        over the precomputed directed type-transition adjacency graph.
        - Distance = 0: if src directly unifies with dst, returns [].
        - Distance > 0: finds minimal intermediate bridge cells [b_1, ..., b_k].
        - If no valid chain exists within max_hops: returns [].
        """
        if not src or not dst or src.cell_id == dst.cell_id:
            return []

        # 1. Direct connection check (Distance = 0 intermediate bridges)
        if self.cells_directly_unify(src, dst):
            return []

        cache_key = (src.cell_id, dst.cell_id, max_hops) if candidate_pool is None else None
        if cache_key is not None and cache_key in self._bridge_path_cache:
            return list(self._bridge_path_cache[cache_key])

        # 2. Shortest-path BFS over the transition graph
        if candidate_pool is not None:
            pool = {c.cell_id: c for c in candidate_pool}
            pool.update({c.cell_id: c for c in self._bridge_cells})
        else:
            pool = self.loaded_cells
        queue: deque = deque()
        visited: Set[str] = {src.cell_id}

        # Seed BFS from src
        next_ids = self._adjacency.get(src.cell_id, [])
        for n_id in next_ids:
            if n_id == dst.cell_id:
                return []
            if n_id in self.loaded_cells and (pool is self.loaded_cells or n_id in pool):
                cand = self.loaded_cells[n_id]
                if not self._is_valid_bridge_cached(cand):
                    continue
                visited.add(n_id)
                queue.append((cand, [cand]))

        # Seed only pre-filtered bridge candidates (or restricted candidate_pool) that accept src output
        seed_iterable = self._bridge_cells if candidate_pool is None else pool.values()
        for cand in seed_iterable:
            cand_id = cand.cell_id
            if cand_id not in visited and cand_id != dst.cell_id:
                if not self._is_valid_bridge_cached(cand):
                    continue
                if self.cells_directly_unify(src, cand):
                    visited.add(cand_id)
                    queue.append((cand, [cand]))

        shortest_len: Optional[int] = None
        best_path: List[Cell] = []
        best_score: float = -1.0

        def _score_path(p: List[Cell]) -> float:
            if not p:
                return 0.0
            last_cell = p[-1]
            last_out = last_cell.primary_output or (next(iter(last_cell.outputs.values())) if last_cell.outputs else None)
            last_state = str(getattr(getattr(last_out, "signature", last_out), "state", "") or "").strip().lower()
            dst_in = dst.primary_input or (next(iter(dst.inputs.values())) if dst.inputs else None)
            dst_state = str(getattr(getattr(dst_in, "signature", dst_in), "state", "") or "").strip().lower()
            src_out = src.primary_output or (next(iter(src.outputs.values())) if src.outputs else None)
            src_state = str(getattr(getattr(src_out, "signature", src_out), "state", "") or "").strip().lower()

            reg = TypeRegistry.get_instance()
            target_match_w = float(getattr(settings, "bridge_target_match_weight", 100.0))
            prop_match_w = float(getattr(settings, "bridge_property_match_weight", 10.0))
            bridge_bonus = float(getattr(settings, "bridge_bonus_weight", 20.0))
            reduction_penalty = float(getattr(settings, "bridge_reduction_penalty", -20.0))

            score = 0.0
            # 1. Direct match to target's primary state
            if last_state and dst_state and last_state == dst_state:
                score += target_match_w
            # 2. Property preservation from source
            p_props = reg.get_state_properties(src_state)
            c_props = reg.get_state_properties(last_state)
            for k, v in c_props.items():
                if k in p_props and p_props[k] == v:
                    score += prop_match_w
            # 3. Categorical morphism role preference: prioritize declared bridge morphisms
            for b_c in p:
                b_role = getattr(b_c, "node_role", "")
                if schema_is_bridge_cell(b_c, reg):
                    score += bridge_bonus
                elif b_role == "reduction":
                    score += reduction_penalty
                else:
                    # Rank-collapsing bridges are penalized unless needed
                    b_in = getattr(b_c, "primary_input", None)
                    b_out = getattr(b_c, "primary_output", None)
                    in_r = getattr(b_in, "canonical_rank", None)
                    out_r = getattr(b_out, "canonical_rank", None)
                    if in_r is not None and out_r is not None and in_r > out_r:
                        score += reduction_penalty
            return score

        expansions = 0
        max_expansions = settings.bridge_bfs_max_expansions
        work_cut = False
        while queue:
            if expansions >= max_expansions:
                if shortest_len is None and len(queue) > 0:
                    work_cut = True
                break
            expansions += 1
            curr_cell, path = queue.popleft()
            if shortest_len is not None and len(path) > shortest_len:
                break

            # If curr_cell directly unifies with dst, candidate found
            if self.cells_directly_unify(curr_cell, dst):
                sc = _score_path(path)
                if shortest_len is None:
                    shortest_len = len(path)
                    best_path = path
                    best_score = sc
                elif len(path) == shortest_len and sc > best_score:
                    best_path = path
                    best_score = sc
                continue

            if shortest_len is None and len(path) < max_hops:
                succ_ids = self._adjacency.get(curr_cell.cell_id, [])
                for n_id in succ_ids:
                    if n_id == dst.cell_id:
                        sc = _score_path(path)
                        if shortest_len is None:
                            shortest_len = len(path)
                            best_path = path
                            best_score = sc
                        elif len(path) == shortest_len and sc > best_score:
                            best_path = path
                            best_score = sc
                        continue
                    if n_id not in visited and n_id in self.loaded_cells:
                        if pool is self.loaded_cells or n_id in pool:
                            nxt_c = self.loaded_cells[n_id]
                            if not self._is_valid_bridge_cached(nxt_c):
                                continue
                            visited.add(n_id)
                            queue.append((nxt_c, path + [nxt_c]))

                # Also search valid bridge candidates in bridge pool
                for b_cand in self._bridge_cells:
                    b_id = b_cand.cell_id
                    if b_id not in visited and b_id != dst.cell_id and (pool is self.loaded_cells or b_id in pool):
                        if self._is_valid_bridge_cached(b_cand) and self.cells_directly_unify(curr_cell, b_cand):
                            visited.add(b_id)
                            queue.append((b_cand, path + [b_cand]))

        if work_cut:
            if not hasattr(self, "trace_events"):
                self.trace_events = []
            self.trace_events.append({
                "event": "bridge_bfs_truncated",
                "src": getattr(src, "cell_id", str(src)),
                "dst": getattr(dst, "cell_id", str(dst)),
                "expansions": expansions,
                "max_expansions": max_expansions,
            })
            logger.debug(
                "[LATTICE] bridge_bfs_truncated: %s -> %s (expansions=%s, cap=%s)",
                getattr(src, "cell_id", str(src)),
                getattr(dst, "cell_id", str(dst)),
                expansions,
                max_expansions,
            )

        if best_path:
            if cache_key is not None:
                self._bridge_path_cache[cache_key] = list(best_path)
            return best_path
        if cache_key is not None:
            self._bridge_path_cache[cache_key] = []
        return []

    def register_dynamic_edge(
        self,
        src_cell_id: str,
        dst_cell_id: str,
        affinity_score: float = 0.8,
        provenance: str = "runtime_verified"
    ) -> bool:
        """
        Dynamically promotes a verified cross-tree or within-tree bridge morphism edge at runtime.
        Updates in-memory adjacency, adds to source cell edges, and tracks in dynamic_edges.
        """
        with self._lock:
            src_cell = self.loaded_cells.get(src_cell_id)
            dst_cell = self.loaded_cells.get(dst_cell_id)
            if not src_cell or not dst_cell:
                return False

            edge_data = {
                "target_cell_id": dst_cell.cell_id,
                "affinity_score": float(affinity_score),
                "score_provenance": provenance,
                "is_cross_tree": bool(src_cell.domain_name != dst_cell.domain_name)
            }
            self.dynamic_edges[(src_cell.cell_id, dst_cell.cell_id)] = edge_data

            # Add to cell edges if not already present
            existing_targets = {
                e.get("target_cell_id") if isinstance(e, dict) else getattr(e, "target_cell_id", None)
                for e in getattr(src_cell, "edges", [])
            }
            if dst_cell.cell_id not in existing_targets:
                if not hasattr(src_cell, "edges") or src_cell.edges is None:
                    src_cell.edges = []
                src_cell.edges.append(edge_data)

            # Add to directed adjacency
            if src_cell.cell_id not in self._adjacency:
                self._adjacency[src_cell.cell_id] = []
            if dst_cell.cell_id not in self._adjacency[src_cell.cell_id]:
                self._adjacency[src_cell.cell_id].append(dst_cell.cell_id)

            if dst_cell.cell_id not in self._reverse_adjacency:
                self._reverse_adjacency[dst_cell.cell_id] = []
            if src_cell.cell_id not in self._reverse_adjacency[dst_cell.cell_id]:
                self._reverse_adjacency[dst_cell.cell_id].append(src_cell.cell_id)

            self._bridge_path_cache.clear()
            self._directly_unify_cache.clear()
            self._valid_bridge_cache.clear()
            self._generic_port_cache = None
            self._version += 1

            return True

    def get_promoted_bridges(self) -> List[Dict[str, Any]]:
        """Returns all dynamically promoted bridge edges."""
        with self._lock:
            return [
                {"source": s, "target": t, **data}
                for (s, t), data in self.dynamic_edges.items()
            ]
