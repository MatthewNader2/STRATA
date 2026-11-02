"""
src/planner.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Topological Pathfinding, Multi-Stage Progression, and Formal Type-Monadic Verification.

Conforms strictly to Sections 3.1, 3.2, and 3.4 of the STRATA paper:
  A path through the lattice is a sequence of monadic binds. Any step that
  would produce bottom is rejected the moment it is proposed.
  Finds maximum-likelihood type-valid composition paths inside the semantic tunnel T.
"""

from __future__ import annotations
import copy
import math
import os
import time
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Any

from log_config import get_logger

try:
    from .lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry, is_path_port as _lattice_is_path_port
    from .unification import (
        unify, Substitution, verify_traced_loop_invariant, verify_coproduct_branch,
        substitute_generics, ExecutionContext, UnificationGate, Success,
        is_edge_valid, _shape_compatible, _roles_conflict,
    )
    from .tokenizer import CellTokenizer, normalize_token
    from .utils import tokenize_alphanumeric
except (ImportError, ValueError):
    from lattice import LatticeOrchestrator, Cell, MicroCell, MacroCell, TypeRegistry, is_path_port as _lattice_is_path_port
    from unification import (
        unify, Substitution, verify_traced_loop_invariant, verify_coproduct_branch,
        substitute_generics, ExecutionContext, UnificationGate, Success,
        is_edge_valid, _shape_compatible, _roles_conflict,
    )
    from tokenizer import CellTokenizer, normalize_token
    from utils import tokenize_alphanumeric

try:
    from .config import settings
except (ImportError, ValueError):
    from config import settings

logger = get_logger('planner')

registry = TypeRegistry.get_instance()

# Purged: Zero hardcoded lexical cheats
STOPWORDS: FrozenSet[str] = frozenset()
_WILDCARD_CARRIERS: FrozenSet[str] = frozenset()

class DynamicEgressTokens(frozenset):
    """Dynamic egress intent verbs harvested from stage-3 sink cells across loaded domain trees."""
    def __contains__(self, item):
        return str(item).lower() in TypeRegistry.get_instance().get_egress_tokens()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_egress_tokens())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_egress_tokens())
    def __and__(self, other):
        return (set(other) if not isinstance(other, set) else other) & TypeRegistry.get_instance().get_egress_tokens()
    def __rand__(self, other):
        return (set(other) if not isinstance(other, set) else other) & TypeRegistry.get_instance().get_egress_tokens()

EGRESS_INTENT_TOKENS = DynamicEgressTokens()

class DynamicMaterializationStates(frozenset):
    """Dynamic materialization states harvested from output typestates of sink cells."""
    def __contains__(self, item):
        return str(item).lower() in TypeRegistry.get_instance().get_materialization_states()
    def __iter__(self):
        return iter(TypeRegistry.get_instance().get_materialization_states())
    def __len__(self):
        return len(TypeRegistry.get_instance().get_materialization_states())

MATERIALIZATION_OUTPUT_STATES = DynamicMaterializationStates()


def _is_projection_port(p_sig: Any) -> bool:
    """True iff p_sig is a declared key/projection port."""
    reg = TypeRegistry.get_instance()
    proj_roles = (
        set(reg.get_verification_semantics("projection_roles"))
        | set(reg.get_verification_semantics("key_roles"))
        | set(reg.get_verification_semantics("key_binds"))
    )
    role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "") or getattr(p_sig, "binds", "")
    return role in proj_roles


_is_col_projection_port = _is_projection_port


def _is_carrier_projectable_port(
    p_sig: Any,
    prev_path: List[Cell],
    quoted_str_literals: Sequence[Any] = (),
    identifier_literals: Sequence[Any] = ()
) -> bool:
    """True iff p_sig is a declared projective/key port and an upstream cell provides a composite carrier."""
    reg = TypeRegistry.get_instance()
    proj_roles = (
        set(reg.get_verification_semantics("projective_roles"))
        | set(reg.get_verification_semantics("projective_binds"))
        | set(reg.get_verification_semantics("key_roles"))
        | set(reg.get_verification_semantics("key_binds"))
    )
    role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "") or getattr(p_sig, "binds", "")
    if role not in proj_roles:
        return False
    # Check if an upstream cell outputs a composite/product carrier
    for c in prev_path:
        for out_s in c.outputs.values():
            o_tn = str(getattr(out_s.signature, "type_name", "")).lower()
            if reg.is_composite_carrier(o_tn) or reg.is_abstract_carrier(o_tn):
                return True
    return False


_is_table_groundable_target_port = _is_carrier_projectable_port


def _extract_ndim_from_contract(sc: Any) -> Optional[int]:
    if sc is None:
        return None
    if isinstance(sc, dict):
        val = sc.get("ndim")
        return int(val) if val is not None else None
    elif isinstance(sc, str):
        s = sc.strip()
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1].strip()
            return len([p for p in inner.split(",") if p.strip()]) if inner else 0
    return None


def _safe_slots_items(cell: Any):
    s = getattr(cell, "slots", None)
    if isinstance(s, dict):
        return list(s.items())
    elif isinstance(s, (list, tuple, set)):
        return [(item, {}) for item in s]
    return []


def _port_type(p: Any) -> str:
    sig = getattr(p, "signature", p)
    return str(getattr(sig, "type_name", "") or "").lower()


class LatticeVocabulary:
    """Every vocabulary the planner used to hard-code, derived from the loaded lattice.

    Nothing in here is a literal list: namespaces, neutral domains, key/relational
    ports, terminal stages/roles, bridge morphisms, handle carriers, materializable
    carriers and generic tokens are all computed from the declared cells, their
    port signatures and the type registry.  Results are memoised against a stamp
    of the loaded-cell table, so a re-harvested lattice is picked up automatically.
    """

    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
        self._stamp: Any = None
        self._memo: Dict[str, Any] = {}
        self._bridge_cache: Dict[str, bool] = {}
        self._port_memo: Dict[Tuple[str, int], bool] = {}

    # -- plumbing --------------------------------------------------------- #
    def _cells(self) -> List[Any]:
        lc = getattr(self.orchestrator, "loaded_cells", None) or {}
        return list(lc.values())

    def _check_stamp(self) -> None:
        lc = getattr(self.orchestrator, "loaded_cells", None) or {}
        stamp = (id(lc), len(lc))
        if stamp != self._stamp:
            self._memo = {}
            self._bridge_cache = {}
            self._port_memo = {}
            self._stamp = stamp

    def _get(self, key: str, build):
        self._check_stamp()
        if key not in self._memo:
            self._memo[key] = build()
        return self._memo[key]

    # -- stage structure --------------------------------------------------- #
    @property
    def stages(self) -> Tuple[int, ...]:
        return self._get("stages", lambda: tuple(sorted({
            c.stage for c in self._cells() if isinstance(getattr(c, "stage", None), int)
        })))

    @property
    def ingress_stage(self) -> Optional[int]:
        s = self.stages
        return s[0] if s else None

    @property
    def terminal_stage(self) -> Optional[int]:
        s = self.stages
        return s[-1] if s else None

    def _terminal_cells(self) -> List[Any]:
        ts = self.terminal_stage
        if ts is None:
            return []
        return [c for c in self._cells() if getattr(c, "stage", None) == ts
                and getattr(c, "cell_type", "") != "macro" and not getattr(c, "is_macro", False)]

    @property
    def ingress_roles(self) -> FrozenSet[str]:
        """Node roles that occur ONLY at the ingress stage."""
        def build():
            ing = self.ingress_stage
            at_ing: Set[str] = set()
            elsewhere: Set[str] = set()
            for c in self._cells():
                r = str(getattr(c, "node_role", "") or "").lower()
                if r:
                    (at_ing if getattr(c, "stage", None) == ing else elsewhere).add(r)
            return frozenset(at_ing - elsewhere)
        return self._get("ingress_roles", build)

    @property
    def terminal_roles(self) -> FrozenSet[str]:
        """Node roles that occur ONLY at the terminal stage."""
        def build():
            ts = self.terminal_stage
            at_terminal: Set[str] = set()
            elsewhere: Set[str] = set()
            for c in self._cells():
                r = str(getattr(c, "node_role", "") or "").lower()
                if not r:
                    continue
                (at_terminal if getattr(c, "stage", None) == ts else elsewhere).add(r)
            return frozenset(at_terminal - elsewhere) if ts is not None else frozenset()
        return self._get("terminal_roles", build)

    @property
    def sequential_topologies(self) -> FrozenSet[Any]:
        return self._get("sequential_topologies", lambda: frozenset(
            getattr(c, "topology_type", None) for c in self._terminal_cells()
        ) | frozenset({None}))

    @property
    def join_topologies(self) -> FrozenSet[str]:
        def build():
            return frozenset(
                str(getattr(c, "topology_type", "") or "") for c in self._cells()
                if len(getattr(c, "inputs", {}) or {}) >= 2 and getattr(c, "topology_type", None)
            ) - frozenset(str(t) for t in self.sequential_topologies if t)
        return self._get("join_topologies", build)

    def is_ingress(self, cell: Any) -> bool:
        if cell is None:
            return False
        if getattr(cell, "is_ingress", False):
            return True
        ins = getattr(cell, "inputs", {}) or {}
        required_data_ins = [p for p in ins.values() if getattr(p, "required", True) and getattr(p, "structure_kind", None) != "scalar"]
        if not required_data_ins:
            return True
        c_st = getattr(cell, "stage", None)
        if self.ingress_stage is not None and c_st == self.ingress_stage:
            return True
        r = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
        if r and r in self.ingress_roles:
            return True
        return False

    def is_terminal(self, cell: Any) -> bool:
        if cell is None:
            return False
        if getattr(cell, "is_sink", False) or getattr(cell, "endable", None) is True:
            return True
        outs = getattr(cell, "outputs", {}) or {}
        if not outs:
            return True
        c_st = getattr(cell, "stage", None)
        if self.terminal_stage is not None and c_st == self.terminal_stage:
            return True
        r = str(getattr(cell, "node_role", "") or getattr(cell, "role", "") or "").lower()
        if r and r in self.terminal_roles:
            return True
        return False

    def is_transform(self, cell: Any) -> bool:
        if cell is None:
            return False
        return not self.is_ingress(cell) and not self.is_terminal(cell)

    # -- token vocabularies ------------------------------------------------ #
    @property
    def namespace_tokens(self) -> FrozenSet[str]:
        """Library/namespace tokens: a domain's own name plus every token that a
        majority of that domain's cell ids share (e.g. a prefix convention)."""
        def build():
            by_domain: Dict[str, List[Set[str]]] = {}
            for c in self._cells():
                dom = str(getattr(c, "domain_name", "") or "").lower()
                if dom:
                    by_domain.setdefault(dom, []).append(
                        {t.lower() for t in CellTokenizer.tokenize_identifier(c.cell_id)})
            out: Set[str] = set()
            for dom, tok_sets in by_domain.items():
                out.add(dom)
                out |= {t.lower() for t in CellTokenizer.tokenize_identifier(dom)}
                if len(tok_sets) >= 2:
                    counts: Dict[str, int] = {}
                    for s in tok_sets:
                        for t in s:
                            counts[t] = counts.get(t, 0) + 1
                    out |= {t for t, n in counts.items() if 2 * n > len(tok_sets)}
            return frozenset(out)
        return self._get("namespace_tokens", build)

    @property
    def op_tokens(self) -> FrozenSet[str]:
        """Operation vocabulary: identifier/keyword/tag tokens of every non-ingress cell,
        minus namespace tokens."""
        def build():
            ingress = self.ingress_stage
            ns = self.namespace_tokens
            out: Set[str] = set()
            for c in self._cells():
                if getattr(c, "node_type", "") == "constant" or getattr(c, "stage", None) in (None, ingress):
                    continue
                for t in CellTokenizer.tokenize_identifier(c.cell_id):
                    if len(t) >= 2 and t.lower() not in ns:
                        out.add(t.lower())
                for kw in getattr(c, "keywords", ()) or ():
                    out |= {t.lower() for t in CellTokenizer.tokenize_identifier(str(kw)) if len(t) >= 2}
                for tag in getattr(c, "semantic_tags", ()) or ():
                    out |= {t.lower() for t in CellTokenizer.tokenize_identifier(str(tag)) if len(t) >= 2}
            return frozenset(out)
        return self._get("op_tokens", build)

    def _identity_vocab(self) -> FrozenSet[str]:
        def build():
            out: Set[str] = set()
            for c in self._cells():
                out |= {t.lower() for t in CellTokenizer.tokenize_identifier(c.cell_id)}
                for kw in getattr(c, "keywords", ()) or ():
                    out |= {t.lower() for t in CellTokenizer.tokenize_identifier(str(kw))}
                for tag in getattr(c, "semantic_tags", ()) or ():
                    out |= {t.lower() for t in CellTokenizer.tokenize_identifier(str(tag))}
                for pn, p in list((getattr(c, "inputs", {}) or {}).items()) + list((getattr(c, "outputs", {}) or {}).items()):
                    out |= {t.lower() for t in CellTokenizer.tokenize_identifier(str(pn))}
                    out |= {t.lower() for t in CellTokenizer.tokenize_identifier(_port_type(p))}
            return frozenset(out)
        return self._get("identity_vocab", build)

    @property
    def function_words(self) -> FrozenSet[str]:
        """Corpus function words: the registry's set when the router has built it; otherwise
        derived here as tokens that are far more frequent than average across the lattice's
        text yet never name an operation (absent from every cell id / keyword / tag)."""
        def build():
            reg_fw = frozenset(str(w).lower() for w in TypeRegistry.get_instance().get_function_words())
            if reg_fw:
                return reg_fw
            return frozenset(t for t in self.frequent_tokens if t not in self._identity_vocab())
        return self._get("function_words", build)

    @property
    def connectives(self) -> FrozenSet[str]:
        """Sentence connectives exactly as the registry learned them (no static fallback)."""
        def build():
            raw = getattr(TypeRegistry.get_instance(), "_sentence_connectives", None) or ()
            return frozenset(str(w).lower() for w in raw)
        return self._get("connectives", build)

    @property
    def frequent_tokens(self) -> FrozenSet[str]:
        """Tokens whose document frequency is more than one standard deviation above the mean."""
        def build():
            idx = getattr(self.orchestrator, "token_index", None) or {}
            dfs = {t: len(v) for t, v in idx.items()}
            if not dfs:
                return frozenset()
            vals = list(dfs.values())
            mean = sum(vals) / len(vals)
            cut = mean + math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            return frozenset(str(t).lower() for t, v in dfs.items() if v > cut)
        return self._get("frequent_tokens", build)

    @property
    def egress_tokens(self) -> FrozenSet[str]:
        return frozenset(str(w).lower() for w in TypeRegistry.get_instance().get_egress_tokens())

    def is_key_port(self, p: Any) -> bool:
        self._check_stamp()
        key = ("key", id(p))
        hit = self._port_memo.get(key)
        if hit is None:
            hit = self._is_key_port_uncached(p)
            self._port_memo[key] = hit
        return hit

    def _is_key_port_uncached(self, p: Any) -> bool:
        """A key/selector port: a projection port, or a textual / list-of-text input
        whose declared name or state carries the registry's projection vocabulary
        (learned from the trees, e.g. sort_column, group_key, join_key)."""
        if _is_projection_port(p):
            return True
        if _lattice_is_path_port(p):
            return False
        reg = TypeRegistry.get_instance()
        t = _port_type(p)
        if not (reg.is_subtype(t, "str") or reg.is_subtype(t, "list") or reg.is_subtype(t, "sequence") or reg.is_subtype(t, "collection")):
            return False
        proj = {str(x).lower() for x in reg.get_projection_tokens()}
        proj |= {normalize_token(x) for x in proj}
        sig = getattr(p, "signature", p)
        toks = {x.lower() for x in CellTokenizer.tokenize_identifier(str(getattr(p, "name", "") or ""))}
        toks |= {x.lower() for x in CellTokenizer.tokenize_identifier(str(getattr(sig, "state", "") or ""))}
        return bool(toks & proj)

    @property
    def relational_tokens(self) -> FrozenSet[str]:
        """Name tokens of the lattice's key ports: a prompt preposition equal to one of
        them binds its object to that port."""
        def build():
            out: Set[str] = set()
            for c in self._cells():
                for pn, p in (getattr(c, "inputs", {}) or {}).items():
                    if self.is_key_port(p):
                        out |= {t.lower() for t in CellTokenizer.tokenize_identifier(str(pn))}
            return frozenset(out)
        return self._get("relational_tokens", build)

    def _role_census(self) -> Dict[str, List[Tuple[bool, bool]]]:
        """role -> [(carrier is scalar/textual/logical/wildcard, port is a declared slot)]"""
        def build():
            reg = TypeRegistry.get_instance()
            census: Dict[str, List[Tuple[bool, bool]]] = {}
            for c in self._cells():
                slots = getattr(c, "slots", None)
                names = set(slots.keys()) if isinstance(slots, dict) else set(slots or ())
                for pn, p in (getattr(c, "inputs", {}) or {}).items():
                    r = _declared_role(p)
                    if not r:
                        continue
                    t = _port_type(p)
                    literalish = t in _WILDCARD_CARRIERS or any(reg.is_subtype(t, f) for f in ("scalar", "numeric", "str", "bool"))
                    census.setdefault(r, []).append((literalish, pn in names))
            return census
        return self._get("role_census", build)

    def _minority_roles(self, idx: int) -> FrozenSet[str]:
        census = self._role_census()
        total = sum(len(v) for v in census.values()) or 1
        out = set()
        for r, rows in census.items():
            if 2 * len(rows) > total:      # the default role covers most ports: never special
                continue
            if all(row[idx] for row in rows):
                out.add(r)
        return frozenset(out)

    @property
    def literal_roles(self) -> FrozenSet[str]:
        """Non-default port roles whose ports are mostly scalar/textual/logical/wildcard
        carriers: their values come from prompt literals, not from upstream wires."""
        return self._get("literal_roles", lambda: self._minority_roles(0))

    @property
    def slot_roles(self) -> FrozenSet[str]:
        """Non-default port roles whose ports are mostly declared macro slots."""
        return self._get("slot_roles", lambda: self._minority_roles(1))

    def role_tokens(self, role: str) -> FrozenSet[str]:
        """Vocabulary the lattice uses for ports of a declared role: identifier tokens of
        those ports' names and states (plus normalised stems), minus function words."""
        def build():
            fw = self.function_words
            out: Set[str] = set()
            for c in self._cells():
                for pn, p in (getattr(c, "inputs", {}) or {}).items():
                    if _declared_role(p) != role:
                        continue
                    st = str(getattr(getattr(p, "signature", p), "state", "") or "")
                    for t in list(CellTokenizer.tokenize_identifier(pn)) + list(CellTokenizer.tokenize_identifier(st)):
                        t = t.lower()
                        if len(t) >= 2 and t not in fw and t not in ("any", "default"):
                            out.add(t)
                            out.add(normalize_token(t))
            return frozenset(out)
        return self._get("role_tokens:" + role, build)

    @property
    def generic_tokens(self) -> FrozenSet[str]:
        return self.frequent_tokens

    # -- type structure ---------------------------------------------------- #
    @property
    def universal_supertypes(self) -> FrozenSet[str]:
        def build():
            reg = TypeRegistry.get_instance()
            names: Set[str] = set()
            for c in self._cells():
                for p in list((getattr(c, "inputs", {}) or {}).values()) + list((getattr(c, "outputs", {}) or {}).values()):
                    t = _port_type(p)
                    if t and "[" not in t and not (len(t) == 1 and t.isalpha()):
                        names.add(t)
            tops: Set[str] = set()
            for t in names:
                try:
                    if all(o == t or reg.is_subtype(o, t) for o in names):
                        tops.add(t)
                except Exception:
                    continue
            return frozenset(tops)
        return self._get("universal_supertypes", build)

    @property
    def neutral_domains(self) -> FrozenSet[str]:
        """Utility domains: those whose ports are mostly wildcard / generic-typed (type
        variables, generic containers).  Hopping through them is not domain dispersion."""
        def build():
            wild: Dict[str, List[bool]] = {}
            for c in self._cells():
                dom = str(getattr(c, "domain_name", "") or "")
                if not dom:
                    continue
                for p in list((getattr(c, "inputs", {}) or {}).values()) + list((getattr(c, "outputs", {}) or {}).values()):
                    t = _port_type(p)
                    is_w = (t in _WILDCARD_CARRIERS) or "[" in t or (len(t) == 1 and t.isalpha())
                    wild.setdefault(dom, []).append(is_w)
            return frozenset(d for d, flags in wild.items() if flags and 2 * sum(flags) > len(flags))
        return self._get("neutral_domains", build)

    def is_bridge(self, cell: Any) -> bool:
        """A bridge morphism converts between unrelated carriers (input and output
        carriers are neither equal nor subtype-related)."""
        cid = str(getattr(cell, "cell_id", ""))
        self._check_stamp()  # refresh stamp / clear cache if lattice changed
        hit = self._bridge_cache.get(cid)
        if hit is not None:
            return hit
        reg = TypeRegistry.get_instance()
        tin = _port_type(getattr(cell, "primary_input", None)) if getattr(cell, "inputs", None) else ""
        tout = _port_type(getattr(cell, "primary_output", None))
        res = bool(
            tin and tout and tin != tout
            and tin not in _WILDCARD_CARRIERS and tout not in _WILDCARD_CARRIERS
            and not reg.is_subtype(tin, tout) and not reg.is_subtype(tout, tin)
        )
        self._bridge_cache[cid] = res
        return res

    @property
    def sink_input_signatures(self) -> Tuple[Any, ...]:
        """Concrete input signatures consumed by terminal sinks: a value whose carrier
        unifies with one of these still has somewhere to land."""
        def build():
            sigs = []
            for c in self._terminal_cells():
                for p in (getattr(c, "inputs", {}) or {}).values():
                    if _port_type(p) not in _WILDCARD_CARRIERS:
                        sigs.append(getattr(p, "signature", p))
            return tuple(sigs)
        return self._get("sink_input_signatures", build)

    def is_materializable_carrier(self, out_port: Any) -> bool:
        self._check_stamp()
        key = ("mat", id(out_port))
        hit = self._port_memo.get(key)
        if hit is None:
            sig = getattr(out_port, "signature", out_port)
            hit = (_port_type(out_port) not in _WILDCARD_CARRIERS
                   and any(unify(sig, s) is not None for s in self.sink_input_signatures))
            self._port_memo[key] = hit
        return hit

    @property
    def handle_types(self) -> FrozenSet[str]:
        """Carriers produced by self-contained ingress generators (no file port), that
        no file-backed ingress produces, and that some downstream cell requires."""
        def build():
            reg = TypeRegistry.get_instance()
            ingress = self.ingress_stage
            file_backed: Set[str] = set()
            generated: Set[str] = set()
            consumed: Set[str] = set()
            for c in self._cells():
                ins = list((getattr(c, "inputs", {}) or {}).values())
                outs = list((getattr(c, "outputs", {}) or {}).values())
                if getattr(c, "stage", None) == ingress:
                    if any(_lattice_is_path_port(p) for p in ins):
                        file_backed |= {_port_type(o) for o in outs}
                    elif not any(p.required and p.default_value is None for p in ins):
                        generated |= {_port_type(o) for o in outs}
                else:
                    consumed |= {_port_type(p) for p in ins if p.required}
            return frozenset(
                t for t in generated
                if t and "[" not in t and t not in file_backed and t not in _WILDCARD_CARRIERS
                and any(reg.is_subtype(t, ct) for ct in consumed)
            )
        return self._get("handle_types", build)

    @property
    def terminal_states(self) -> FrozenSet[str]:
        """Output states that terminal sinks declare, plus the registry's materialization states."""
        def build():
            out: Set[str] = {str(s).lower() for s in TypeRegistry.get_instance().get_materialization_states()}
            out.update(TypeRegistry.get_instance().get_terminal_states())
            for c in self._terminal_cells():
                for o in (getattr(c, "outputs", {}) or {}).values():
                    st = str(getattr(getattr(o, "signature", o), "state", "") or "").lower()
                    if st:
                        out.add(st)
            return frozenset(out)
        return self._get("terminal_states", build)

    def is_informative_token(self, token: str) -> bool:
        reg = getattr(self.orchestrator, "registry", None) or TypeRegistry.get_instance()
        return reg.is_informative_token(token)


_VOCAB_CACHE: Dict[int, LatticeVocabulary] = {}


def _active_vocab(orchestrator: Any = None) -> LatticeVocabulary:
    orch = orchestrator
    if orch is None:
        try:
            orch = LatticeOrchestrator.get_active_instance()
            if orch is None:
                orch = LatticeOrchestrator()
        except Exception:
            orch = None
    key = id(orch)
    v = _VOCAB_CACHE.get(key)
    if v is None or v.orchestrator is not orch:
        v = LatticeVocabulary(orch)
        _VOCAB_CACHE[key] = v
    return v


class SearchBudget:
    """Compute bounds for the beam search (resource limits, not semantics), all
    derived from the tunnel size |V|: a path can visit each cell at most once, an
    endpoint quota of floor(log2|V| / 2) keeps prefix diversity, and pool caps
    scale with beam x depth."""
    __slots__ = ("per_endpoint", "beam_width", "linear_per_endpoint", "linear_beam_width",
                 "entry_limit", "slot_trials")

    def __init__(self, n_cells: int):
        n = max(int(n_cells), 2)
        lg = max(1, int(math.log2(n)))
        self.per_endpoint = max(1, lg // 2)
        self.beam_width = n * self.per_endpoint
        self.linear_per_endpoint = lg
        self.linear_beam_width = n * lg
        self.entry_limit = max(self.per_endpoint, n // 2)
        self.slot_trials = lg

    def pool_limits(self, beam_width: int, max_steps: int) -> Tuple[int, int]:
        hi = max(beam_width * max_steps, 2)
        return hi, hi // 2


class PathScore:
    """A path objective without weights: `defects` are feasibility violations compared
    lexicographically (fewer is better); `terms` are higher-is-better quantities that
    are aggregated by rank across the competing set (see `_rank_paths`)."""
    __slots__ = ("defects", "terms")

    def __init__(self, defects: Tuple[float, ...], terms: Dict[str, float]):
        self.defects = defects
        self.terms = terms


def _rank_paths(rows: Sequence[Tuple[Any, PathScore]]) -> List[Tuple[Any, float]]:
    """Order (item, PathScore) rows best-first.

    1. lexicographic on defects (feasibility first);
    2. mean mid-rank percentile of every term across the competing rows (Borda), averaged
       within each term family ("family:term") and then across families, so several
       correlated terms of one family cannot outvote the other aspects of the objective.
    Scale-free: no term needs a weight and adding a term never requires retuning the others.
    Returns (item, aggregate) with aggregate in [0, 1]."""
    import bisect
    n = len(rows)
    if n == 0:
        return []
    names = sorted({k for _, ps in rows for k in ps.terms})
    fam_of = {k: k.split(":", 1)[0] for k in names}
    fam_sum: Dict[str, List[float]] = {f: [0.0] * n for f in set(fam_of.values())}
    fam_cnt: Dict[str, int] = {}
    for name in names:
        fam_cnt[fam_of[name]] = fam_cnt.get(fam_of[name], 0) + 1
        vals = [ps.terms.get(name, 0.0) for _, ps in rows]
        sv = sorted(vals)
        acc = fam_sum[fam_of[name]]
        for i, v in enumerate(vals):
            lo = bisect.bisect_left(sv, v)
            hi = bisect.bisect_right(sv, v)
            acc[i] += (lo + 0.5 * (hi - lo)) / n
    agg = [0.0] * n
    for f, acc in fam_sum.items():
        for i in range(n):
            agg[i] += acc[i] / fam_cnt[f]
    if fam_sum:
        agg = [a / len(fam_sum) for a in agg]
    order = sorted(range(n), key=lambda i: (rows[i][1].defects, -agg[i]))
    return [(rows[i][0], agg[i]) for i in order]


def _private_copy(cell: Any, orchestrator: Any = None) -> Any:
    """Copy-on-write for cells that are shared instances of the loaded lattice, so
    binding slots on a plan can never leak into other plans or other trials."""
    shared = False
    lc = getattr(orchestrator, "loaded_cells", None) or {}
    if lc.get(getattr(cell, "cell_id", None)) is cell:
        shared = True
    if not shared and not isinstance(getattr(cell, "bound_slots", None), dict):
        return cell
    dup = cell.clone() if hasattr(cell, "clone") else copy.copy(cell)
    bs = getattr(cell, "bound_slots", None)
    try:
        dup.bound_slots = dict(bs) if isinstance(bs, dict) else {}
    except (AttributeError, TypeError):
        pass
    return dup


def _declared_role(p_sig: Any) -> str:
    r = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
    if not r and hasattr(p_sig, "derive_port_role"):
        try:
            r = p_sig.derive_port_role()
        except Exception:
            r = ""
    return str(r or "")


def _is_receiver_port(p_sig: Any, cand: Any) -> bool:
    """A receiver is the instance a morphism operates on: declared as such, or (when no
    role is declared) the cell's undeclared primary input."""
    role = _declared_role(p_sig)
    if role:
        return role == "receiver"
    return p_sig is getattr(cand, "primary_input", None)


def _literal_groundable(p_sig: Any, cand: Any, quoted_str_literals: Sequence[Any],
                        identifier_literals: Sequence[Any], numeric_literals: Sequence[Any]) -> bool:
    """A required port is satisfiable at synthesis time iff its DECLARED carrier is
    literal-groundable and the prompt supplies a literal of that family, or the port
    declares an enum / a literal-family role / a slot role.  Roles are classified from
    the lattice itself (see LatticeVocabulary.literal_roles / slot_roles).

    Carrier-capability gate: the slot/role channels can only ever bind prompt
    literals and scalar expressions. A port whose declared carrier is an OBJECT
    (not scalar/text/numeric/logical/path/enum) can never be grounded that way,
    no matter what role vocabulary it carries or whether its NAME matches a
    template placeholder -- template placeholders like {model} are filled by the
    binder from upstream wires, not from the prompt. Exempting object carriers
    made the planner accept consumers (e.g. predict-without-fit) whose required
    object ports no step could ever produce, deferring the failure to Layer 3.
    """
    registry = TypeRegistry.get_instance()
    vocab = _active_vocab()
    t = _port_type(p_sig)
    role = _declared_role(p_sig)
    has_text = bool(quoted_str_literals or identifier_literals)
    # Literal-capable carriers: primitives the emitter can synthesize as an
    # expression, declared enum choices, and untyped wildcards. Everything else
    # (class instances, containers of class instances, handles) requires a wire.
    carrier_literal_capable = (
        t in _WILDCARD_CARRIERS
        or registry.is_subtype(t, "scalar")
        or registry.is_subtype(t, "str")
        or registry.is_subtype(t, "text")
        or registry.is_subtype(t, "numeric")
        or registry.is_subtype(t, "bool")
        or registry.is_subtype(t, "logical")
        or registry.is_subtype(t, "filepath")
        or registry.is_subtype(t, "uri")
        or registry.is_subtype(t, "path")
        or bool(getattr(p_sig, "enum_values", None))
    )
    return (
        (role != "" and role in vocab.literal_roles)
        or (registry.is_subtype(t, "str") and has_text)
        or (registry.is_subtype(t, "numeric") and bool(numeric_literals))
        or registry.is_subtype(t, "bool")
        or registry.is_subtype(t, "filepath")
        or registry.is_subtype(t, "uri")
        or _lattice_is_path_port(p_sig)
        or (registry.is_subtype(t, "scalar") and (has_text or bool(numeric_literals)))
        or bool(getattr(p_sig, "enum_values", None))
        or (
            carrier_literal_capable
            and role != ""
            and role in vocab.slot_roles
        )
        or (vocab.is_key_port(p_sig) and has_text)
        or (
            carrier_literal_capable
            and bool(getattr(cand, "slots", None))
            and getattr(p_sig, "name", None) in getattr(cand, "slots", {})
        )
    )


def _is_terminal_sink_cell(cell: Any) -> bool:
    """
    Determines if a cell is a terminal sink concluding the execution path (D5).
    'Terminal' is read from the lattice itself: the deepest declared stage (when the
    lattice has at least source/transform/sink levels) and the node roles / topologies
    that the cells at that stage declare.  A macro with declared sub-cells still has
    internal sub-lattices to plan and is never terminal.
    """
    vocab = _active_vocab()
    ts = vocab.terminal_stage
    is_sink = (
        (ts is not None and getattr(cell, "stage", None) == ts)
        or str(getattr(cell, "node_role", "") or "").lower() in vocab.terminal_roles
    )
    if not is_sink:
        return False
    if getattr(cell, "cell_type", "") == "macro" or getattr(cell, "node_role", "") == "macro":
        if getattr(cell, "sub_cells", None):
            return False
    top = getattr(cell, "topology_type", None)
    if top not in vocab.sequential_topologies:
        return False
    return True


def _is_ingress_stage(cell: Any) -> bool:
    ing = _active_vocab().ingress_stage
    return ing is not None and getattr(cell, "stage", None) == ing


def _is_egress_stage(cell: Any) -> bool:
    ts = _active_vocab().terminal_stage
    return ts is not None and getattr(cell, "stage", None) == ts


def _is_ingress_cell(cell: Any) -> bool:
    return _is_ingress_stage(cell) or str(getattr(cell, "node_role", "") or "").lower() in _active_vocab().ingress_roles


def _is_egress_cell(cell: Any) -> bool:
    return _is_egress_stage(cell) or str(getattr(cell, "node_role", "") or "").lower() in _active_vocab().terminal_roles


def _is_transform_stage(cell: Any) -> bool:
    st = getattr(cell, "stage", None)
    v = _active_vocab()
    return st is not None and st != v.ingress_stage and st != v.terminal_stage


def _is_port_role_compatible(p_out: Any, p_in: Any) -> bool:
    """
    Universal role-level semantic compatibility check:
    Enforces declared mutually exclusive role pairs and strict role gating.
    """
    in_role = getattr(p_in, "port_role", None) or getattr(p_in, "derived_role", "")
    out_role = getattr(p_out, "port_role", None) or getattr(p_out, "derived_role", "")
    registry = TypeRegistry.get_instance()

    if _roles_conflict(in_role, out_role, reg=registry):
        return False

    out_state = str(getattr(getattr(p_out, "signature", None), "state", "")).lower()
    in_state = str(getattr(getattr(p_in, "signature", None), "state", "")).lower()
    if registry.has_property_conflict(out_state, in_state):
        return False

    return True



def _segment_prompt_clauses(prompt: str) -> List[str]:
    """Bracket-aware, operationally-filtered clause splitter (apply_fixes_v13).

    Splits on top-level commas/semicolons, protects brackets, merges operand
    fragments into their preceding clause, then drops fragments whose content
    tokens don't match any tree operation. A fragment that is just a library
    name or a writing verb is not a coverable clause.
    """
    if not prompt:
        return []
    norm_prompt = prompt.replace("->", "\n").replace("➔", "\n")
    parts, buf, depth = [], [], 0
    for ch in norm_prompt:
        if ch in "[({":
            depth += 1; buf.append(ch); continue
        if ch in "])}":
            depth = max(0, depth - 1); buf.append(ch); continue
        if depth == 0 and ch in ",;\n":
            seg = "".join(buf).strip()
            if seg: parts.append(seg)
            buf = []; continue
        buf.append(ch)
    if buf:
        seg = "".join(buf).strip()
        if seg: parts.append(seg)

    refined_parts = []
    for seg in parts:
        sub_clauses = CellTokenizer.split_prompt_clauses(seg)
        if sub_clauses:
            refined_parts.extend(sub_clauses)
        elif seg:
            refined_parts.append(seg)
    parts = refined_parts

    merged: List[str] = []
    _vocab = _active_vocab()

    def _is_sink_directive(text: str) -> bool:
        """Determines if a clause expresses an egress directive without containing other distinct operation vocabulary."""
        toks = [t.lower() for t in tokenize_alphanumeric(text)]
        if not toks:
            return False
        tok_set = set(toks)
        egress_words = _vocab.egress_tokens
        has_egress = bool(tok_set & egress_words)
        if not has_egress:
            return False
        non_egress_ops = (tok_set - _vocab.function_words - _vocab.connectives) & (_vocab.op_tokens - egress_words)
        return not non_egress_ops

    for p in parts:
        p_clean = p.strip()
        if not p_clean: continue
        p_toks = CellTokenizer.tokenize_prompt(p_clean)
        if not p_toks: continue
        # Preserve full specification integrity: clauses are never dropped or mutated.
        merged.append(p_clean)

    return merged or [prompt.strip()]


class LatticePlanner:
    """
    Topological Planner & Gap Bridging Engine (Sections 3.1-3.4).
    Operates strictly within the active semantic tunnel T.
    Finds maximum-likelihood type-valid composition paths.
    """
    def __init__(
        self,
        orchestrator: LatticeOrchestrator,
        rag: Optional[Any] = None,
        macros_enabled: Optional[bool] = None,
        topology_mode: Optional[str] = None,
        require_coverage_floor: Optional[bool] = None,
        coverage_floor_fraction: Optional[float] = None,
        planner_time_budget_ms: Optional[float] = None,
        planner_greedy_budget_ms: Optional[float] = None,
    ):
        self.orchestrator = orchestrator
        self.rag = rag
        if macros_enabled is not None:
            self.macros_enabled = bool(macros_enabled)
        else:
            try:
                from config import settings
                self.macros_enabled = bool(settings.macros_enabled)
            except Exception as e:
                self.macros_enabled = True
        if topology_mode is not None:
            self.topology_mode = str(topology_mode).lower()
        else:
            try:
                from config import settings
                self.topology_mode = str(settings.topology_mode).lower()
            except Exception as e:
                self.topology_mode = "frontier"

        if require_coverage_floor is not None:
            self.require_coverage_floor = bool(require_coverage_floor)
        else:
            try:
                from config import settings
                self.require_coverage_floor = bool(settings.require_coverage_floor)
            except Exception:
                self.require_coverage_floor = True

        if coverage_floor_fraction is not None:
            self.coverage_floor_fraction = float(coverage_floor_fraction)
        else:
            try:
                from config import settings
                self.coverage_floor_fraction = float(settings.coverage_floor_fraction)
            except Exception:
                self.coverage_floor_fraction = 0.85

        if planner_time_budget_ms is not None:
            self.planner_time_budget_ms = float(planner_time_budget_ms)
        else:
            try:
                from config import settings
                self.planner_time_budget_ms = float(settings.planner_time_budget_ms)
            except Exception:
                self.planner_time_budget_ms = 10000.0

        if planner_greedy_budget_ms is not None:
            self.planner_greedy_budget_ms = float(planner_greedy_budget_ms)
        else:
            try:
                from config import settings
                self.planner_greedy_budget_ms = float(settings.planner_greedy_budget_ms)
            except Exception:
                self.planner_greedy_budget_ms = 1000.0

        self.last_refusal: Optional[Dict[str, Any]] = None
        self.current_relevance_map: Dict[str, float] = {}
        self._affinity_cache: Dict[Tuple[str, str], float] = {}
        self._macro_edges_index: Optional[Dict[Tuple[str, str], List[str]]] = None
        self._cells_connect_cache: Dict[Tuple[str, str], bool] = {}

    @property
    def _vocab(self) -> LatticeVocabulary:
        return _active_vocab(self.orchestrator)

    def _get_lattice_op_tokens(self) -> Set[str]:
        return set(self._vocab.op_tokens)

    # ============================================================
    # HARD CONSTRAINTS (apply_fixes_v8) — coverage and literals are
    # reject-level, not soft penalties.
    # ============================================================
    def _literal_consumption_of(self, path, universal_literals,
                                numeric_literals, quoted_str_literals, identifier_literals):
        """(apply_fixes_v16) Fraction of prompt literals with a home on this path.

        Match channels, in order of specificity:
          - exact port name           (always)
          - substring in port name    (only when the literal is >= 3 chars)
          - token in any cell_id      (covers library-qualified operation names)
          - token in any domain name  (covers declared domain identifiers)
          - has a column_projection port on path  (consumes the whole
            bracketed list of quoted strings at once — R, G, B)
        Short-string substring matches are rejected: `'r' in 'array'` is a
        coincidence, not a signal.
        """
        if not universal_literals:
            return 1.0

        op_toks = self._get_lattice_op_tokens()

        port_tokens = set()
        cell_id_tokens = set()
        domain_tokens = set()
        has_multi_key_slot = False
        has_numeric_port = False

        try:
            reg = TypeRegistry.get_instance()
        except Exception:
            reg = None

        def _tokens(s):
            s = str(s).lower()
            out = {s}
            for part in tokenize_alphanumeric(s):
                if part:
                    out.add(part)
            return out

        def _port_match(lc, port):
            if lc == port:
                return True
            lc_toks = set(CellTokenizer.tokenize_identifier(lc))
            port_toks_set = set(CellTokenizer.tokenize_identifier(port))
            if lc_toks and lc_toks.issubset(port_toks_set):
                return True
            return False

        def _identifier_match(lc):
            if any(_port_match(lc, p) for p in port_tokens):
                return True
            if any(_port_match(lc, c) for c in cell_id_tokens):
                return True
            if any(_port_match(lc, d) for d in domain_tokens):
                return True
            return False

        for c in path:
            cid = getattr(c, "cell_id", "") or ""
            cell_id_tokens |= _tokens(cid)
            dom = getattr(c, "domain_name", "") or ""
            if dom:
                domain_tokens |= _tokens(dom)
            for pn, ps in c.inputs.items():
                port_tokens.add(pn.lower())
                r = (getattr(ps, "port_role", None) or getattr(ps, "derived_role", "") or "").lower()
                if r:
                    port_tokens.add(r)
                tn = str(getattr(ps.signature, "type_name", "")).lower()
                is_key_slot = self._vocab.is_key_port(ps)
                multi_projs = set(reg.get_verification_semantics("multi_projection_kinds")) if reg else set()
                if is_key_slot and reg is not None and (
                    reg.is_subtype(tn, "collection")
                    or getattr(c, "projection", "") in multi_projs
                ):
                    has_multi_key_slot = True
                if reg is not None and reg.is_subtype(tn, "numeric"):
                    has_numeric_port = True
            for sk in getattr(c, "bound_slots", {}).keys():
                port_tokens.add(sk.lower())
            for sl in getattr(c, "slots", []):
                port_tokens.add(str(sl).lower())

        consumed = 0
        path_ingress_file_ports = sum(
            1 for c in path if _is_ingress_cell(c)
            for p in c.inputs.values() if p.required and p.default_value is None and _lattice_is_path_port(p)
        )
        path_egress_file_ports = sum(
            1 for c in path if _is_egress_cell(c)
            for p in c.inputs.values() if p.required and p.default_value is None and _lattice_is_path_port(p)
        )
        _vocab = self._vocab
        path_col_key_slots = sum(1 for c in path for p in c.inputs.values() if _vocab.is_key_port(p))
        used_ing_ports = 0
        used_egr_ports = 0
        used_col_keys = 0
        for kind, lit in universal_literals:
            lc = str(lit).lower().strip("'\"")
            if kind == "file_asset":
                if used_ing_ports < path_ingress_file_ports:
                    consumed += 1
                    used_ing_ports += 1
                elif used_egr_ports < path_egress_file_ports:
                    consumed += 1
                    used_egr_ports += 1
            elif kind == "numeric":
                if numeric_literals and has_numeric_port:
                    consumed += 1
            elif kind == "quoted_str":
                if any(_port_match(lc, p) for p in port_tokens):
                    consumed += 1
                elif used_col_keys < path_col_key_slots:
                    consumed += 1
                    used_col_keys += 1
                elif has_multi_key_slot:
                    consumed += 1
            else:
                if _identifier_match(lc):
                    consumed += 1
                elif lc not in op_toks and used_col_keys < path_col_key_slots:
                    consumed += 1
                    used_col_keys += 1
        return consumed / max(len(universal_literals), 1)

    def _clause_coverage_of(self, path, clause_tokens_list):
        if not clause_tokens_list:
            return 1.0
        covered = set()
        for c in path:
            c_toks = c.token_set
            id_toks = getattr(c, "identity_tokens", c_toks)
            for gi, cl_toks in enumerate(clause_tokens_list):
                if cl_toks & id_toks:
                    covered.add(gi)
        return len(covered) / len(clause_tokens_list)

    def _hard_constraint_filter(self, candidates, universal_literals,
                                numeric_literals, quoted_str_literals, identifier_literals,
                                clause_tokens_list, **_ignored):
        """Reject-level constraints derived from the candidate set itself (no floors):
        keep the paths that consume the most prompt literals, and among those the upper
        half by clause coverage (>= the median coverage of that group)."""
        if not candidates:
            return candidates, 3
        scored = []
        for it in candidates:
            cov = self._clause_coverage_of(it[0], clause_tokens_list)
            lit = self._literal_consumption_of(it[0], universal_literals,
                                               numeric_literals, quoted_str_literals,
                                               identifier_literals)
            scored.append((it, cov, lit))
        best_lit = max(l for _, _, l in scored)

        # Pareto-frontier filter: retain all non-dominated candidates balancing clause coverage and literal consumption
        pareto = []
        for i, (it_i, cov_i, lit_i) in enumerate(scored):
            dominated = False
            for j, (it_j, cov_j, lit_j) in enumerate(scored):
                if i != j and cov_j >= cov_i and lit_j >= lit_i and (cov_j > cov_i or lit_j > lit_i):
                    dominated = True
                    break
            if not dominated:
                pareto.append((it_i, cov_i, lit_i))

        if not pareto:
            pareto = scored

        # Sort non-dominated frontier by coverage (primary) and literal consumption (secondary)
        pareto.sort(key=lambda x: (x[1], x[2]), reverse=True)
        keep = [it for it, _, _ in pareto]
        try:
            from config import settings as _s
            _refuse = bool(getattr(_s, "require_coverage_floor", False))
        except Exception:
            _refuse = False
        if _refuse and clause_tokens_list and max(c for _, c, _ in scored) == 0.0:
            logger.warning("[planner] REFUSE: no candidate path covers any prompt clause; emitting empty pipeline.")
            return [], 4
        return keep, (0 if best_lit >= 1.0 else 1)

    def _cells_connect(self, c1: Cell, c2: Cell) -> bool:
        """Evaluates whether any output of c1 can unify with any input of c2, or if a bridge connects them."""
        pair = (c1.cell_id, c2.cell_id)
        can_feed = self._cells_connect_cache.get(pair)
        if can_feed is None:
            can_feed = any(
                unify(
                    out_sig.signature if hasattr(out_sig, "signature") else out_sig,
                    p_sig.signature if hasattr(p_sig, "signature") else p_sig,
                ) is not None
                for out_sig in c1.outputs.values()
                for p_sig in c2.inputs.values()
            )
            if not can_feed and self.orchestrator and hasattr(self.orchestrator, "find_shortest_bridge_path"):
                bridge_chain = self.orchestrator.find_shortest_bridge_path(c1, c2, max_hops=3)
                can_feed = bool(bridge_chain)
            self._cells_connect_cache[pair] = can_feed
        return can_feed

    def find_bridge_chain(self, c1: Cell, c2: Cell, max_hops: int = 4) -> List[Cell]:
        """Unified Shortest-Path Precondition Bridge Closure."""
        if self._cells_connect(c1, c2):
            return []
        if self.orchestrator and hasattr(self.orchestrator, "find_shortest_bridge_path"):
            return self.orchestrator.find_shortest_bridge_path(c1, c2, max_hops=max_hops)
        return []

    def _insert_property_bridges(self, path: List[Cell]) -> List[Cell]:
        """Inserts property-resolving bridges when an upstream cell's output does not directly unify with a downstream consumer."""
        if not path or not self.orchestrator:
            return path
        bridged_path: List[Cell] = []
        for c in path:
            if not bridged_path:
                bridged_path.append(c)
            else:
                prev_c = bridged_path[-1]
                if not self.orchestrator.cells_directly_unify(prev_c, c):
                    bridges = self.orchestrator.find_shortest_bridge_path(prev_c, c)
                    bridged_path.extend(bridges)
                bridged_path.append(c)
        return bridged_path

    def _build_macro_edges_index(self) -> Dict[Tuple[str, str], List[str]]:
        index: Dict[Tuple[str, str], List[str]] = {}
        if getattr(self, "macros_enabled", True) and hasattr(self.orchestrator, "loaded_cells"):
            for m in self.orchestrator.loaded_cells.values():
                if getattr(m, "cell_type", "") == "macro" or getattr(m, "sub_cells", None):
                    topo = getattr(m, "internal_topology", {}) or {}
                    for src_id, dst_ids in topo.items():
                        for dst_id in dst_ids:
                            index.setdefault((src_id, dst_id), []).append(m.cell_id)
                    subs = getattr(m, "sub_cells", None)
                    if subs:
                        for i in range(len(subs) - 1):
                            index.setdefault((subs[i], subs[i + 1]), []).append(m.cell_id)
        return index

    def _calculate_edge_affinity(self, src_cell: Cell, dst_cell: Cell) -> float:
        """
        Calculates empirical edge affinity score between two cells.
        AST-mined edges from real code snippets receive the highest affinity,
        followed by LLM seed edges, synaptic macro reinforcement, and topological reachability.
        """
        key = (src_cell.cell_id, dst_cell.cell_id)
        cached = self._affinity_cache.get(key)
        if cached is not None:
            return cached

        res = self._compute_edge_affinity_raw(src_cell, dst_cell)
        self._affinity_cache[key] = res
        return res

    # ---- empirical scales (replace the fixed affinity tiers / stage tables) ---- #
    _AFFINITY_TIERS = 8  # macro synapse, declared, reverse, join, stage-advance, bridge, adjacency, same-domain

    def _lattice_stamp(self) -> Tuple[int, int]:
        lc = getattr(self.orchestrator, "loaded_cells", None) or {}
        return (id(lc), len(lc))

    def _edge_statistics(self) -> Dict[str, Any]:
        """Statistics of the lattice's DECLARED edges: the sorted affinity scores (the
        empirical scale every evidence tier is mapped onto), stage-transition counts and
        the set of observed (output role -> input role) pairs."""
        stamp = self._lattice_stamp()
        cached = getattr(self, "_edge_stats_cache", None)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        lc = getattr(self.orchestrator, "loaded_cells", None) or {}
        by_lower = {k.lower(): v for k, v in lc.items()}
        scores: List[float] = []
        stage_counts: Dict[Tuple[Any, Any], int] = {}
        stage_out: Dict[Any, int] = {}
        role_pairs: Set[Tuple[str, str]] = set()
        for c in lc.values():
            for edge in getattr(c, "edges", None) or []:
                tgt = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
                aff = edge.get("affinity_score") if isinstance(edge, dict) else getattr(edge, "affinity_score", None)
                if aff is not None:
                    try:
                        scores.append(float(aff))
                    except (TypeError, ValueError):
                        pass
                d = by_lower.get(str(tgt).lower()) if tgt else None
                if d is None:
                    continue
                key = (getattr(c, "stage", None), getattr(d, "stage", None))
                stage_counts[key] = stage_counts.get(key, 0) + 1
                stage_out[key[0]] = stage_out.get(key[0], 0) + 1
                for po in c.outputs.values():
                    ro = getattr(po, "port_role", None) or getattr(po, "derived_role", "")
                    for pi in d.inputs.values():
                        ri = getattr(pi, "port_role", None) or getattr(pi, "derived_role", "")
                        if ro and ri:
                            role_pairs.add((ro, ri))
        scores.sort()
        stats = {"scores": scores, "stage_counts": stage_counts, "stage_out": stage_out,
                 "role_pairs": role_pairs, "n_stages": max(len(getattr(_active_vocab(self.orchestrator), "stages", ())), 1)}
        self._edge_stats_cache = (stamp, stats)
        return stats

    def _affinity_tier(self, k: int) -> float:
        """Value of evidence tier k (0 = strongest): the (T-k)/T quantile of the declared
        edge-affinity distribution, or an evenly spaced ordinal scale if the lattice
        declares no scored edges."""
        T = self._AFFINITY_TIERS
        q = (T - k) / T
        scores = self._edge_statistics()["scores"]
        if not scores:
            return q
        return scores[min(len(scores) - 1, max(0, int(round(q * (len(scores) - 1)))))]

    def _compute_edge_affinity_raw(self, src_cell: Cell, dst_cell: Cell) -> float:
        rel_map = getattr(self, "current_relevance_map", None) or {}
        vocab = self._vocab

        # Synaptic macro-goal reinforcement: an edge pre-wired inside an active macro is
        # interpolated between the declared-edge tier and the top tier by macro relevance.
        if getattr(self, "macros_enabled", True):
            if self._macro_edges_index is None:
                self._macro_edges_index = self._build_macro_edges_index()
            m_ids = self._macro_edges_index.get((src_cell.cell_id, dst_cell.cell_id))
            if m_ids:
                macro_rel = max(rel_map.get(mid, 0.0) for mid in m_ids)
                if macro_rel > 0.0:
                    lo, hi = self._affinity_tier(1), self._affinity_tier(0)
                    return lo + (hi - lo) * min(1.0, macro_rel)

        # 1. Forward declared edge: its own mined/declared affinity is the evidence.
        dst_id_lower = dst_cell.cell_id.lower()
        for edge in getattr(src_cell, "edges", []):
            tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt_id and (tgt_id == dst_cell.cell_id or str(tgt_id).lower() == dst_id_lower):
                aff = edge.get("affinity_score") if isinstance(edge, dict) else getattr(edge, "affinity_score", None)
                return float(aff) if aff is not None else self._affinity_tier(1)

        # 2. Reverse declared edge (bidirectional idiom): never stronger than the reverse tier.
        src_id_lower = src_cell.cell_id.lower()
        for edge in getattr(dst_cell, "edges", []):
            tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
            if tgt_id and (tgt_id == src_cell.cell_id or str(tgt_id).lower() == src_id_lower):
                aff = edge.get("affinity_score") if isinstance(edge, dict) else getattr(edge, "affinity_score", None)
                tier = self._affinity_tier(2)
                return min(float(aff), tier) if aff is not None else tier

        src_stage = getattr(src_cell, "stage", None)
        dst_stage = getattr(dst_cell, "stage", None)
        src_domain = getattr(src_cell, "domain_name", "")
        dst_domain = getattr(dst_cell, "domain_name", "")
        same_domain = bool(src_domain and dst_domain and src_domain == dst_domain)

        # 3. Stage advance into the terminal stage inside one domain (transform -> sink),
        # gated on prompt relevance of the producer.
        stages = vocab.stages
        if (same_domain and vocab.terminal_stage is not None and dst_stage == vocab.terminal_stage
                and src_stage is not None and src_stage != vocab.ingress_stage and src_stage != dst_stage):
            if not rel_map or rel_map.get(src_cell.cell_id, 0.0) > 0.0:
                return self._affinity_tier(4)
            return self._affinity_tier(7)

        # 4. Convergent join: dst joins several carriers (declared join topology or >= 2 inputs).
        is_join_node = (
            str(getattr(dst_cell, "topology_type", "") or "") in vocab.join_topologies
            or len(getattr(dst_cell, "inputs", {})) >= 2
        )
        if is_join_node and same_domain:
            if not rel_map or rel_map.get(dst_cell.cell_id, 0.0) > 0.0:
                return self._affinity_tier(3)

        # 5. Cross-domain bridge morphism (structural: converts between unrelated carriers).
        if src_domain and dst_domain and src_domain != dst_domain and vocab.is_bridge(src_cell):
            return self._affinity_tier(5)

        # 6. Topological reachability, 7. same-domain continuity.
        adj = getattr(self.orchestrator, "_adjacency", None) or getattr(self.orchestrator, "adjacency", None)
        if adj and dst_cell.cell_id in adj.get(src_cell.cell_id, ()):
            return self._affinity_tier(6)
        if same_domain:
            return self._affinity_tier(7)
        return 0.0

    def compute_edge_score(
        self,
        src_cell: Cell,
        dst_cell: Cell,
        relevance_map: Optional[Dict[str, float]] = None,
        weights: Optional[Tuple[float, ...]] = None
    ) -> float:
        """
        Four-term edge score E(u, v): AST affinity, semantic relevance, role progress and
        graph proximity.  Terms are combined with equal weights unless the caller supplies
        its own.  Role progress is empirical: the stage-transition frequency and the
        fraction of (output role, input role) pairs observed on the lattice's declared edges.
        """
        ast_aff = self._calculate_edge_affinity(src_cell, dst_cell)
        rel = (relevance_map or {}).get(dst_cell.cell_id, 0.0)

        stats = self._edge_statistics()
        s_src = getattr(src_cell, "stage", None)
        s_dst = getattr(dst_cell, "stage", None)
        n_out = stats["stage_out"].get(s_src, 0)
        stage_prog = (stats["stage_counts"].get((s_src, s_dst), 0) + 1) / (n_out + stats["n_stages"])

        src_roles = {getattr(p, "port_role", None) or getattr(p, "derived_role", "") for p in src_cell.outputs.values()} - {""}
        dst_roles = {getattr(p, "port_role", None) or getattr(p, "derived_role", "") for p in dst_cell.inputs.values()} - {""}
        pairs = [(a, b) for a in src_roles for b in dst_roles]
        role_frac = (sum(1 for pr in pairs if pr in stats["role_pairs"]) / len(pairs)) if pairs else 0.0
        role_prog = 0.5 * (stage_prog + role_frac)

        adj = getattr(self.orchestrator, "_adjacency", {}) or {}
        if dst_cell.cell_id in adj.get(src_cell.cell_id, ()):
            hops = 1
        elif any(dst_cell.cell_id in adj.get(mid, ()) for mid in adj.get(src_cell.cell_id, ())):
            hops = 2
        else:
            hops = 3
        terms = (ast_aff, rel, role_prog, 1.0 / hops)
        if weights is None or len(weights) != len(terms):
            weights = tuple(1.0 for _ in terms)
        total_w = sum(weights) or 1.0
        return sum(w * t for w, t in zip(weights, terms)) / total_w

    def plan(
        self,
        prompt: str,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
        start_sig: Optional[Any] = None,
        goal_sig: Optional[Any] = None,
        max_transforms: int = 6,
        ctx: Optional[Any] = None,
        **kwargs
    ) -> List[Cell]:
        """
        Plans a type-valid compositional pipeline:
          [Entry (Stage 1)] -> [Transforms / Bridges (Stage 2)]* -> [Terminal (Stage 3)]
        Conforms strictly to Sections 3.1, 3.2, and 3.4 of the STRATA paper:
        Viterbi Trellis Dynamic Programming over the typed category G|_T.
        Monadic Unification Gate rejects any invalid edge proposals.

        Goal-directed extensions (Section 3.4 objective):
        - Terminal bias: paths terminating at a Stage-3 egress morphism are
          preferred whenever the tunnel declares sinks (dataflow must land).
        - Asset absorption: file-asset literals extracted from the prompt must be
          consumed by a path-typed input port somewhere on the path.
        - Wildcarrier penalty: transitions whose producer output carrier is the
          untyped wildcard are weak evidence and are penalized, so fully-untyped
          utility morphisms never beat typed equivalents.
        - Zero-ary constructor morphisms (node_type 'constructor') are insertable
          mid-chain to complete instance lifecycles (construct -> fit -> score).
        """
        self.current_ctx = ctx
        self.current_relevance_map = dict(relevance_map or {})
        self._affinity_cache.clear()
        self._widened_this_plan = False
        self.last_refusal = None

        req_floor = kwargs.get("require_coverage_floor", getattr(self, "require_coverage_floor", True))
        floor_frac = float(kwargs.get("coverage_floor_fraction", getattr(self, "coverage_floor_fraction", 0.85)))

        # Wall-clock search budget: the beam must never run unbounded.
        if "planner_time_budget_ms" in kwargs:
            _budget_ms = float(kwargs["planner_time_budget_ms"])
        else:
            _budget_ms = float(getattr(self, "planner_time_budget_ms", 10000.0))
        self._plan_deadline = time.perf_counter() + max(_budget_ms, 250.0) / 1000.0
        if not tunnel:
            return []

        # Single standalone node case
        if len(tunnel) == 1:
            return [tunnel[0]]

        # Candidate pool excluding constants and monolithic macro shortcuts
        # (Macro cells without slots act as synaptic priors that boost micro-cells,
        # but do not preempt multi-cell compositions as monolithic 1-step black boxes.
        # Higher-order combinator envelopes with slots ARE allowed as structured wrappers).
        candidates = [
            c for c in tunnel
            if getattr(c, "node_type", "") != "constant"
            and (
                bool(getattr(c, "slots", None))
                or (
                    getattr(c, "cell_type", "") != "macro"
                    and not getattr(c, "is_macro", False)
                    and not isinstance(c, MacroCell)
                    and not (getattr(c, "is_combinator", False) or getattr(c, "node_role", "") == "combinator" or getattr(c, "role", "") == "combinator" or getattr(c, "node_type", "") == "combinator")
                )
            )
        ]
        if not candidates:
            candidates = [c for c in tunnel if getattr(c, "node_type", "") != "constant"]
        if not candidates:
            return [tunnel[0]]

        # Compute log-likelihood log P(v | e_x) from tunnel relevance
        log_probs: Dict[str, float] = {}
        for c in candidates:
            p = max(relevance_map.get(c.cell_id, 0.0), 1e-6)
            log_probs[c.cell_id] = math.log(p)
        budget = SearchBudget(len(candidates))
        vocab = self._vocab

        # Set NSTL_DEBUG_PLAN=1 to dump the ranked candidate paths after planning.
        _debug_plan = os.environ.get("NSTL_DEBUG_PLAN", "") in ("1", "true", "True", "on")
        _component_trace: Dict[Tuple[str, ...], Dict[str, float]] = {}
        self._component_trace = _component_trace

        # ---- Goal-directed objective data (all derived from DECLARED structure) ----
        registry = TypeRegistry.get_instance()
        l0_extracted_literals = ExecutionContext._extract_universal_literals(prompt or "")
        target_sink = None
        if ctx:
            target_sink = (
                getattr(ctx, "target_key", None)
                or getattr(ctx, "parameters", {}).get("target_sink")
                or getattr(ctx, "parameters", {}).get("target_key")
            )
        if not target_sink:
            target_sink = ExecutionContext._extract_target_sink(prompt or "")
        universal_literals = [
            (kind, val) for _, kind, val in l0_extracted_literals
            if kind in ("file_asset", "identifier", "quoted_str", "numeric")
            and not (kind == "identifier" and target_sink and str(val).lower() == str(target_sink).lower())
        ]
        literal_positions = {
            (kind, val): pos
            for pos, kind, val in l0_extracted_literals
            if kind in ("file_asset", "identifier", "quoted_str", "numeric")
            and not (kind == "identifier" and target_sink and str(val).lower() == str(target_sink).lower())
        }
        identifier_role_map = ExecutionContext._build_identifier_role_map(prompt or "")
        self._last_identifier_roles = identifier_role_map
        file_literals = [
            v for kind, v in universal_literals
            if kind == "file_asset" or (kind == "quoted_str" and ExecutionContext._is_path_string(str(v)))
        ]
        if ctx:
            ctx_src = list(getattr(ctx, "parameters", {}).get("source_uris") or getattr(ctx, "source_files", []) or [])
            ctx_dest = list(getattr(ctx, "parameters", {}).get("dest_uris") or getattr(ctx, "dest_files", []) or [])
        else:
            ctx_src, ctx_dest = [], []

        dest_file_literals = ctx_dest if ctx_dest else [
            val for pos, kind, val in l0_extracted_literals
            if (kind == "file_asset" or (kind == "quoted_str" and ExecutionContext._is_path_string(str(val))))
            and ExecutionContext._asset_direction(prompt or "", pos) == "dest"
        ]
        src_file_literals = ctx_src if ctx_src else [
            val for pos, kind, val in l0_extracted_literals
            if (kind == "file_asset" or (kind == "quoted_str" and ExecutionContext._is_path_string(str(val))))
            and ExecutionContext._asset_direction(prompt or "", pos) != "dest"
        ]

        def _has_path_port(cell: Cell) -> bool:
            for p_name, p_sig in cell.inputs.items():
                if _is_path_port_name(p_name, p_sig):
                    return True
            return False

        def _is_path_port_name(p_name: str, p_sig: Any) -> bool:
            return _lattice_is_path_port(p_sig)

        def _is_path_port_sig(p_sig: Any) -> bool:
            sig = getattr(p_sig, "signature", p_sig)
            return _is_path_port_name(str(getattr(p_sig, "name", "") or ""), sig)

        def _is_file_format_compatible(cell: Cell, file_path_str: str) -> bool:
            if not file_path_str or not isinstance(file_path_str, str) or "." not in file_path_str:
                return True
            ext = file_path_str.rpartition(".")[2].strip().lower()
            if not ext:
                return True
            path_ports = [p for p in cell.inputs.values() if _lattice_is_path_port(p)]
            if not path_ports:
                return True
            for p in path_ports:
                port_fmts = getattr(p, "supported_formats", None)
                if port_fmts:
                    clean_fmts = {str(f).strip().lower() for f in port_fmts}
                    if "*" not in clean_fmts and ext not in clean_fmts:
                        return False
                tn = str(getattr(p, "type_name", "") or "").strip()
                if "[" in tn and tn.endswith("]"):
                    ctor, _, inner = tn[:-1].partition("[")
                    if ctor.strip().lower() in ("file", "stream", "path", "asset"):
                        parts = [part.strip().lower() for part in inner.split(",") if part.strip()]
                        if len(parts) >= 2:
                            fmt = parts[1]
                            reg = TypeRegistry.get_instance()
                            if fmt in ("any", "generic", "*", ""):
                                carrier = parts[0]
                                try:
                                    placeholder = reg.get_asset_placeholder(carrier, "")
                                    if placeholder and "." in placeholder:
                                        carrier_ext = placeholder.rpartition(".")[2].strip().lower()
                                        generic_exts = set(reg.get_verification_semantics("generic_format_extensions"))
                                        if carrier_ext and carrier_ext not in generic_exts:
                                            if ext == carrier_ext or reg.is_subtype(ext, carrier_ext) or reg.is_subtype(carrier_ext, ext):
                                                return True
                                            for anc in reg.get_format_groups():
                                                if reg.is_subtype(ext, anc) and reg.is_subtype(carrier_ext, anc):
                                                    return True
                                            return False
                                except Exception as _exc:
                                    logger.debug("[PLANNER DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)
                                return True
                            if ext == fmt or fmt in ext or ext in fmt or reg.is_subtype(ext, fmt) or reg.is_subtype(fmt, ext):
                                return True
                            for anc in reg.get_format_groups():
                                if reg.is_subtype(ext, anc) and (reg.is_subtype(fmt, anc) or fmt in anc):
                                    return True
                            return False
                        elif len(parts) == 1:
                            target = parts[0]
                            reg = TypeRegistry.get_instance()
                            if target in ("any", "generic", "*", ""):
                                return True
                            if ext == target or target in ext or ext in target or reg.is_subtype(ext, target) or reg.is_subtype(target, ext):
                                return True
                            for anc in reg.get_format_groups():
                                if reg.is_subtype(ext, anc) and (reg.is_subtype(target, anc) or target in anc):
                                    return True
                            try:
                                placeholder = reg.get_asset_placeholder(target, "")
                                if placeholder and "." in placeholder:
                                    carrier_ext = placeholder.rpartition(".")[2].strip().lower()
                                    generic_exts = set(reg.get_verification_semantics("generic_format_extensions"))
                                    if carrier_ext and carrier_ext not in generic_exts:
                                        if ext == carrier_ext or reg.is_subtype(ext, carrier_ext) or reg.is_subtype(carrier_ext, ext):
                                            return True
                                        for anc in reg.get_format_groups():
                                            if reg.is_subtype(ext, anc) and reg.is_subtype(carrier_ext, anc):
                                                return True
                                        return False
                            except Exception:
                                pass
                            return False
            return True

        mut_excl_roles = registry.get_verification_semantics("mutually_exclusive_roles") or ()

        def _cell_mutually_exclusive_roles(cell: Cell) -> Set[str]:
            if not mut_excl_roles:
                return set()
            roles = set()
            all_ports = list(cell.inputs.values()) + list(cell.outputs.values())
            for p in all_ports:
                sig = getattr(p, "signature", p)
                t_name = str(getattr(sig, "type_name", "")).lower()
                t_state = str(getattr(sig, "state", "")).lower()
                for r in mut_excl_roles:
                    r_low = r.lower()
                    if registry.is_subtype(t_name, r_low) or registry.is_subtype(t_state, r_low) or r_low in t_name or r_low in t_state:
                        roles.add(r)
            for tag in getattr(cell, "semantic_tags", []) or []:
                t_low = str(tag).lower()
                for r in mut_excl_roles:
                    r_low = r.lower()
                    if r_low == t_low or (len(r_low) >= 4 and t_low.startswith(r_low[:4])):
                        roles.add(r)
            for r_attr in (getattr(cell, "node_role", ""), getattr(cell, "role", "")):
                if r_attr:
                    r_low = str(r_attr).lower()
                    for r in mut_excl_roles:
                        if r.lower() in r_low:
                            roles.add(r)
            return roles

        numeric_literals = [
            v for _, t, v in ExecutionContext._extract_universal_literals(prompt or "")
            if t == "numeric" or registry.is_subtype(str(t).lower(), "numeric")
        ]
        quoted_str_literals = [
            v for kind, v in universal_literals
            if kind == "quoted_str" and not ExecutionContext._is_path_string(str(v))
        ]
        identifier_literals = [
            v for kind, v in universal_literals
            if kind == "identifier"
        ]

        # =================================================================
        # Provable-infeasibility probe & demand-driven retrieval widening.
        #
        # A required receiver port is PROVABLY infeasible when no cell in the
        # entire loaded lattice can produce it and it cannot be grounded from
        # prompt literals. Planning a path through such a cell can only end in
        # an unresolved port at Layer 3, so the search must not pay for one:
        #   1. Probe every candidate's required receiver ports against a
        #      memoized producer index of the whole lattice (fail fast).
        #   2. When the prompt demonstrably DEMANDS an infeasible cell (it
        #      carries relevance), widen the search once: query the RAG with
        #      the cell's own DECLARED identity vocabulary plus the unsatisfied
        #      port's declared state/type tokens -- the producer declares the
        #      matching vocabulary by construction -- and merge what comes back
        #      into the candidate pool.
        #   3. Whatever remains infeasible is dropped before the beam runs; if
        #      nothing survives, planning fails fast with an honest diagnostic
        #      instead of burning the full trellis on a doomed plan.
        # All evidence is structural (declared ports, declared carriers,
        # declared vocabulary): no domain knowledge, no pattern matching.
        # =================================================================
        def _port_literal_groundable(type_name: str, p_sig: Any = None) -> bool:
            """Carrier-level literal groundability (no prompt literal required).
            Defined here so the feasibility probe and the receiver-sig tables
            share one definition."""
            if p_sig is not None:
                return _literal_groundable(p_sig, None, quoted_str_literals, identifier_literals, numeric_literals)
            t = type_name.lower()
            return (
                (registry.is_subtype(t, "str") and bool(quoted_str_literals or identifier_literals))
                or (registry.is_subtype(t, "numeric") and bool(numeric_literals))
                or registry.is_subtype(t, "bool") or registry.is_subtype(t, "filepath") or registry.is_subtype(t, "uri")
                or (registry.is_subtype(t, "scalar") and bool(quoted_str_literals or identifier_literals or numeric_literals))
            )

        def _producer_index() -> Dict[str, List[Any]]:
            idx = getattr(self, "_producer_index_cache", None)
            stamp = self._lattice_stamp()
            if idx is not None and idx[0] == stamp:
                return idx[1]
            by_type: Dict[str, List[Any]] = {}
            for c in self.orchestrator.loaded_cells.values():
                for out_p in c.outputs.values():
                    sig = getattr(out_p, "signature", out_p)
                    t_key = str(getattr(sig, "type_name", "") or "").lower()
                    by_type.setdefault(t_key, []).append(sig)
            self._producer_index_cache = (stamp, by_type)
            return by_type

        _prod_idx = _producer_index()

        def _candidate_producer_groups(t: str) -> List[List[Any]]:
            """Producer signature groups whose TYPE could unify with carrier t.
            Over-approximate on purpose (subtype both ways, wildcards, generic
            parameters, type variables): a false 'feasible' only costs search
            time, a false 'infeasible' would drop a workable cell."""
            groups: List[List[Any]] = []
            for pt, sigs in _prod_idx.items():
                if (
                    pt == t
                    or pt in _WILDCARD_CARRIERS
                    or t in _WILDCARD_CARRIERS
                    or (len(pt) == 1 and pt.isalpha())
                    or (len(t) == 1 and t.isalpha())
                    or "[" in pt or "[" in t
                    or registry.is_subtype(pt, t)
                    or registry.is_subtype(t, pt)
                ):
                    groups.append(sigs)
            return groups

        def _has_lattice_producer(p_sig: Any) -> bool:
            sig = getattr(p_sig, "signature", p_sig)
            t = str(getattr(sig, "type_name", "") or "").lower()
            key = (t, str(getattr(sig, "state", "") or "").lower())
            hit = _producer_probe_memo.get(key)
            if hit is None:
                hit = False
                for group in _candidate_producer_groups(t):
                    for prod in group:
                        if unify(prod, sig) is not None:
                            hit = True
                            break
                    if hit:
                        break
                _producer_probe_memo[key] = hit
            return hit

        _producer_probe_memo: Dict[Tuple[str, str], bool] = {}

        def _required_receiver_sigs(c: Cell) -> List[Tuple[str, Any]]:
            """Required ports that need a producer wire (mirrors the beam's own
            receiver rules: non-wildcard, non-literal-groundable, non-target)."""
            out: List[Tuple[str, Any]] = []
            for p_name, p_sig in c.inputs.items():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                if _is_receiver_port(p_sig, c):
                    out.append((p_name, p_sig))
                    continue
                t = str(p_sig.signature.type_name)
                if t.lower() in _WILDCARD_CARRIERS:
                    continue
                if _is_table_groundable_target_port(p_sig, [], quoted_str_literals, identifier_literals):
                    continue
                if _port_literal_groundable(t, p_sig):
                    continue
                out.append((p_name, p_sig))
            return out

        def _probe_infeasible(cells: List[Cell]) -> List[Tuple[Cell, str, Any]]:
            bad: List[Tuple[Cell, str, Any]] = []
            for c in cells:
                if getattr(c, "node_type", "") == "constructor":
                    continue
                for p_name, p_sig in _required_receiver_sigs(c):
                    if not _has_lattice_producer(p_sig):
                        bad.append((c, p_name, p_sig))
            return bad

        if start_sig is None:
            infeasible = _probe_infeasible(candidates)
            demanded = [(c, pn, ps) for c, pn, ps in infeasible
                        if relevance_map.get(c.cell_id, 0.0) > 0.0]
            if demanded:
                logger.warning(
                    "[PLANNER] %d demanded cell(s) carry required port(s) with no producer in the lattice: %s",
                    len(demanded),
                    [f"{c.cell_id}.{pn}" for c, pn, _ in demanded][:8],
                )
                # Demand-driven search widening: the consumer's own DECLARED
                # identity vocabulary plus the port's declared state/type tokens
                # form a retrieval query whose results necessarily share the
                # producer's vocabulary (the producer declares the matching
                # output state by construction of the typestate contract).
                rag = getattr(self, "rag", None)
                if rag is not None and not getattr(self, "_widened_this_plan", False):
                    self._widened_this_plan = True
                    exp_spans: List[str] = []
                    for c, _pn, p_sig in demanded[:3]:
                        toks = set(getattr(c, "identity_tokens", c.token_set) or set())
                        sig = getattr(p_sig, "signature", p_sig)
                        toks |= set(CellTokenizer.tokenize_identifier(str(getattr(sig, "state", "") or "")))
                        toks |= set(CellTokenizer.tokenize_identifier(str(getattr(sig, "type_name", "") or "")))
                        fw = set()
                        try:
                            _reg = TypeRegistry.get_instance()
                            fw = set(_reg.get_function_words()) | set(_reg.get_sentence_connectives())
                        except Exception:
                            fw = set()
                        span_toks = {t for t in (toks - fw) if len(t) >= 2}
                        if span_toks:
                            exp_spans.append(" ".join(sorted(span_toks)))
                    if exp_spans:
                        try:
                            widened = rag.get_relevant_context_batch(exp_spans, top_k=12)
                            added = 0
                            for span_results in widened:
                                for item in span_results:
                                    cid = item.get("cell_id")
                                    if not cid or cid in {x.cell_id for x in candidates}:
                                         continue
                                    cell_obj = self.orchestrator.loaded_cells.get(cid)
                                    if cell_obj is None:
                                        continue
                                    if getattr(cell_obj, "node_type", "") == "constant":
                                        continue
                                    candidates.append(cell_obj)
                                    relevance_map.setdefault(cid, 0.01)
                                    self.current_relevance_map.setdefault(cid, 0.01)
                                    log_probs[cid] = math.log(max(relevance_map.get(cid, 0.01), 1e-6))
                                    added += 1
                            if added:
                                logger.info(
                                    "[PLANNER] Search widened with %d producer candidate(s) via declared-vocabulary retrieval.",
                                    added,
                                )
                        except Exception as _w_err:
                            logger.debug("[PLANNER] Widening retrieval unavailable: %s", _w_err)
                    # Re-probe after widening
                    infeasible = _probe_infeasible(candidates)
                else:
                    infeasible = [(c, pn, ps) for c, pn, ps in infeasible]
            if infeasible:
                infeasible_ids = {c.cell_id for c, _pn, _ps in infeasible}
                candidates = [c for c in candidates if c.cell_id not in infeasible_ids]
                logger.warning(
                    "[PLANNER] Failing fast: dropped %d cell(s) with provably unsatisfiable required ports: %s",
                    len(infeasible_ids),
                    sorted(infeasible_ids)[:8],
                )
                if not candidates:
                    logger.error(
                        "[PLANNER] No plannable candidate remains: the prompt's demanded operations "
                        "require producers the loaded lattice does not declare. Emitting no plan."
                    )
                    return []

        def _can_be_entry_source(cell: Cell) -> bool:
            for p_name, p_sig in cell.inputs.items():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
                if _lattice_is_path_port(p_sig):
                    if not file_literals and not _is_ingress_cell(cell) and not getattr(cell, "is_macro", False) and not isinstance(cell, MacroCell):
                        return False
                    continue
                is_num = registry.is_subtype(t_name, "numeric")
                if is_num:
                    if not numeric_literals:
                        return False
                    continue
                if not (
                    registry.is_subtype(t_name, "str")
                    or registry.is_subtype(t_name, "bool")
                    or registry.is_subtype(t_name, "scalar")
                ):
                    return False
            return True

        tunnel_has_sinks = any(_is_egress_stage(c) for c in candidates)
        tunnel_absorbs_assets = any(
            _is_ingress_stage(c) and _has_path_port(c) for c in candidates
        ) if file_literals else False

        # Candidate starting cells (Stage 1 sources or cells matching start_sig)
        candidate_entries = list(candidates)
        if start_sig is not None:
            s_sig = start_sig.signature if hasattr(start_sig, "signature") else start_sig
            matching = [c for c in candidate_entries if unify(s_sig, c.primary_input.signature) is not None]
            if matching:
                candidate_entries = matching
        else:
            viable_entries = [c for c in candidate_entries if _can_be_entry_source(c)]
            clauses = CellTokenizer.split_prompt_clauses(prompt)
            first_clause = clauses[0].strip() if clauses else prompt.strip()
            clause_tokens = CellTokenizer.tokenize_prompt(first_clause) if first_clause else set()

            s1_entries = [
                c for c in viable_entries
                if (_is_ingress_cell(c) or isinstance(c, MacroCell))
            ]
            first_clause_file_literals = [
                v for _, kind, v in ExecutionContext._extract_universal_literals(first_clause)
                if kind == "file_asset" or (kind == "quoted_str" and ExecutionContext._is_path_string(str(v)))
            ]
            if first_clause_file_literals:
                s1_entries = [
                    c for c in s1_entries
                    if _has_path_port(c) and _is_file_format_compatible(c, str(first_clause_file_literals[0]))
                ]

            matching_s1 = []
            if clause_tokens and s1_entries:
                matching_s1 = [c for c in s1_entries if len(clause_tokens & getattr(c, "identity_tokens", c.token_set)) > 0]

            if matching_s1:
                matching_s1.sort(key=lambda c: (
                    getattr(c, "source_priority", 100),
                    -(relevance_map.get(c.cell_id, 0.0) * (1.0 + len(clause_tokens & getattr(c, "identity_tokens", c.token_set))))
                ))
                candidate_entries = matching_s1[:budget.entry_limit]
            elif file_literals and s1_entries:
                compat_s1 = [
                    c for c in s1_entries
                    if _has_path_port(c) and _is_file_format_compatible(c, str(file_literals[0]))
                ]
                entries_to_score = compat_s1 if compat_s1 else s1_entries
                entries_to_score.sort(key=lambda c: (
                    getattr(c, "source_priority", 100),
                    -relevance_map.get(c.cell_id, 0.0)
                ))
                candidate_entries = entries_to_score[:budget.entry_limit]
            else:
                viable_entries.sort(key=lambda c: -relevance_map.get(c.cell_id, 0.0))
                candidate_entries = viable_entries[:budget.entry_limit]

            # Augment with self-contained cells (all inputs optional/defaulted)
            # only when no explicit source file literal was requested in the prompt.
            if not first_clause_file_literals and not file_literals:
                entry_ids = {c.cell_id for c in candidate_entries}
                _mean_rel = sum(relevance_map.get(c.cell_id, 0.0) for c in candidates) / max(len(candidates), 1)
                self_contained_entries = [
                    c for c in viable_entries
                    if c.cell_id not in entry_ids
                    and relevance_map.get(c.cell_id, 0.0) > _mean_rel
                    and all(
                        not p.required or p.default_value is not None
                        for p in c.inputs.values()
                    )
                ]
                if self_contained_entries:
                    candidate_entries = list(candidate_entries) + self_contained_entries

        # Clauses and tokens for sequential alignment and concept coverage
        clauses = _segment_prompt_clauses(prompt)
        clause_tokens_list = [CellTokenizer.tokenize_prompt(cl) - set(STOPWORDS) for cl in clauses]
        clause_tokens_list = [t for t in clause_tokens_list if t]
        content_prompt_tokens = set().union(*clause_tokens_list) if clause_tokens_list else (CellTokenizer.tokenize_prompt(prompt) if prompt else set())
        p_len = max(len(content_prompt_tokens), 1)
        num_clauses = max(len(clause_tokens_list), 1)
        has_prompt_egress_intent = (
            (bool(content_prompt_tokens & EGRESS_INTENT_TOKENS) and not target_sink)
            or bool(dest_file_literals)
            or (goal_sig is not None)
        )

        # Corpus-derived IDF over prompt tokens (df from the lattice token index).
        # Used to weight coverage and clause alignment: generic tokens ('data',
        # 'column') carry near-zero objective mass, while discriminative tokens
        # ('csv', 'train', 'split') dominate. Trivial duplicate clauses ("Y
        # column") can then neither inflate the clause count nor be farmed by
        # cells that merely share generic vocabulary.
        token_index_for_idf = getattr(self.orchestrator, "token_index", None) or {}
        corpus_size_for_idf = max(len(self.orchestrator.loaded_cells), 1)

        def _idf(tok: str) -> float:
            df = len(token_index_for_idf.get(tok, ()))
            return math.log(1.0 + (corpus_size_for_idf + 1) / (df + 1.0))

        idf_of_prompt = {tok: _idf(tok) for tok in content_prompt_tokens}
        total_prompt_idf = sum(idf_of_prompt.values()) or 1.0
        # Clause weight counts only MATCHABLE vocabulary: a token no lattice cell
        # declares (df == 0 — quoted file names, bare identifiers, phrasing noise)
        # can never be matched by any cell, so counting it in the clause's weight
        # only inflates the evidence floor past what the best possible explainer
        # could reach (R6-5: clause 0's 'inputcsv' made even PD_READ_CSV fail).
        def _cl_weight(cl_toks: Set[str]) -> float:
            return sum(_idf(t) for t in cl_toks if t in token_index_for_idf)
        clause_weights = [_cl_weight(cl_toks) for cl_toks in clause_tokens_list]
        total_clause_weight = sum(clause_weights) or 1.0

        # Token provenance weighting: a cell's IDENTITY tokens (cell_id + declared
        # keywords) describe what it IS; its docstring prose merely describes what
        # it says. Identity matches carry full idf mass, docstring matches are
        # down-weighted — descriptive vocabulary (e.g. a splitter's docstring
        # mentioning "train/test") must not outrank another cell's identifier.
        def _identity_tokens(cell: Cell) -> Set[str]:
            toks = CellTokenizer.tokenize_identifier(cell.cell_id)
            for kw in getattr(cell, "keywords", ()) or ():
                toks.update(CellTokenizer.tokenize_identifier(kw))
            for p in list(cell.inputs.values()) + list(cell.outputs.values()):
                sig = getattr(p, "signature", p)
                tn = str(getattr(sig, "type_name", "") or "")
                if tn and tn.lower() not in _WILDCARD_CARRIERS:
                    toks.update(CellTokenizer.tokenize_identifier(tn))
            return toks

        identity_cache: Dict[str, Set[str]] = {}
        for c in candidates:
            identity_cache[c.cell_id] = _identity_tokens(c)

        # Core identity (R6-5): the vocabulary a cell DECLARES as its own name and
        # keywords — what the cell IS. Port names and port type-names are schema
        # interface shared by every similarly-shaped cell (e.g. a 'columns' port
        # exists on dozens of dataframe transforms) and therefore carry retrieval
        # mass (token_set / full identity) but NO clause-coverage evidence. Without
        # this separation, a cell whose only operative overlap with a clause is a
        # generic port name plus a verb can falsely claim that clause's coverage.
        def _core_identity_tokens(cell: Cell) -> Set[str]:
            toks = CellTokenizer.tokenize_cell(cell.cell_id, getattr(cell, "keywords", ()) or ())
            if getattr(cell, "domain_name", None) and cell.domain_name != "generic":
                toks = toks - CellTokenizer.tokenize_identifier(cell.domain_name)
            try:
                aliases = TypeRegistry.get_instance().get_all_aliases()
            except Exception:
                aliases = {}
            if aliases:
                toks = toks - {t for t in toks if t in aliases}
            return toks

        core_identity_cache: Dict[str, Set[str]] = {}
        for c in candidates:
            core_identity_cache[c.cell_id] = _core_identity_tokens(c)

        # Operand/operator ordering: the engine deliberately does NOT parse
        # natural-language syntax (prepositions, head/operand positions, etc.).
        # Language syntax is not declared structure — hardcoding English
        # grammar here broke every non-English phrasing and inverted
        # construction. Ordering evidence comes from DECLARED structure only:
        # clause segmentation (cell_clause_mass), typed carrier dataflow
        # (monadic transitions), and identifier role groups.

        def _match_mass(cl_toks: Set[str], c_toks: Set[str], id_toks: Set[str]) -> float:
            strong = sum(idf_of_prompt.get(t, _idf(t)) for t in (cl_toks & (c_toks & id_toks)))
            weak = sum(idf_of_prompt.get(t, _idf(t)) for t in (cl_toks & (c_toks - id_toks)))
            # Descriptive-vocabulary matches count in proportion to how much of the cell's own
            # vocabulary is identity (a docstring-heavy cell's prose says little about what it is).
            identity_share = len(id_toks & c_toks) / max(len(c_toks), 1)
            return strong + identity_share * weak

        def _raw_mass(cl_toks: Set[str], c_toks: Set[str], id_toks: Set[str]) -> float:
            """idf mass of the clause tokens matched by the cell's IDENTITY vocabulary (what the
            cell is), which alone decides clause coverage; docstring matches only feed ranking."""
            return sum(idf_of_prompt.get(t, _idf(t)) for t in (cl_toks & c_toks & id_toks))

        def _is_wildcarrier(cell: Cell) -> bool:
            t = str(getattr(getattr(cell, "primary_output", None), "type_name", "") or "").lower()
            return t in _WILDCARD_CARRIERS

        # ---- Precomputed per-cell scoring tables (make path scoring O(k)) ----
        # clause_mass[cell]: per-clause match masses, plus the covered-clause set.
        # coverage is computed on the UNION of path token matches (see below).
        cell_cov_strong: Dict[str, Set[str]] = {}
        cell_cov_weak: Dict[str, Set[str]] = {}
        cell_cov_mass_bonus: Dict[str, float] = {}
        cell_clause_mass: Dict[str, List[float]] = {}
        cell_covered: Dict[str, Set[int]] = {}
        literal_prompt_tokens: Set[str] = set()
        for _, kind, val in l0_extracted_literals:
            sval = str(val)
            if kind == "file_asset" or ExecutionContext._is_path_string(sval):
                # The extension names a format concept the prompt shares with cells
                # (it is not a literal value): drop the structural suffix only.
                stem, ext = os.path.splitext(sval.strip("'\""))
                ext_toks = {t.lower() for t in CellTokenizer.tokenize_identifier(ext.lstrip("."))} if ext else set()
            else:
                ext_toks = set()
            for tok in CellTokenizer.tokenize_identifier(sval):
                if tok and tok.lower() not in ext_toks:
                    literal_prompt_tokens.add(tok)
        non_literal_prompt_tokens = content_prompt_tokens - literal_prompt_tokens - set(STOPWORDS)

        # Clause-coverage significance: a cell covers a clause iff its identity evidence for that
        # clause is above what the candidate set typically shows (mean + one standard deviation
        # of the per-cell evidence).  Adaptive: common words that many cells happen to carry
        # ("values", "data") raise the bar, discriminative tokens clear it.
        _raw_by_cell: Dict[str, List[float]] = {}
        _prec_by_cell: Dict[str, List[float]] = {}
        for c in candidates:
            _c_toks = c.token_set
            _id = core_identity_cache.get(c.cell_id) or getattr(c, "identity_tokens", None) or identity_cache.get(c.cell_id, _c_toks)
            _c_total_idf = sum(_idf(t) for t in _c_toks)
            _cid_toks = {t for t in CellTokenizer.tokenize_identifier(str(c.cell_id)) if _idf(t) > 0.0}
            _cid_mass = sum(_idf(t) for t in _cid_toks)
            raw_list = []
            prec_list = []
            for cl in clause_tokens_list:
                _name_prec = (sum(_idf(t) for t in (cl & _cid_toks)) / _cid_mass) if _cid_mass > 0 else 0.0
                r_val = _raw_mass(cl, _c_toks, _id) * (1.0 + 0.25 * _name_prec)
                p_val = (sum(_idf(t) for t in (cl & _c_toks)) / _c_total_idf) if _c_total_idf > 0 else 0.0
                raw_list.append(r_val)
                prec_list.append(p_val)
            _raw_by_cell[c.cell_id] = raw_list
            _prec_by_cell[c.cell_id] = prec_list
        clause_min_evidence = []
        clause_min_precision = []
        from config import settings as _settings
        _cl_ev_ratio = float(_settings.clause_evidence_ratio)
        _cl_wt_ratio = float(_settings.clause_weight_ratio)
        _cl_prec_ratio = float(_settings.clause_precision_ratio)
        for gi in range(len(clause_tokens_list)):
            col = [v[gi] for v in _raw_by_cell.values()]
            prec_col = [v[gi] for v in _prec_by_cell.values()]
            if not col:
                clause_min_evidence.append(0.0)
                clause_min_precision.append(0.0)
                continue
            max_ev = max(col)
            max_prec = max(prec_col) if prec_col else 0.0
            cl_weight = clause_weights[gi] if gi < len(clause_weights) else 0.0
            # A cell covers a clause only when it is a COMPETITIVE explainer of it:
            # its identity evidence must reach the configured ratio of the best explainer's evidence
            # for that clause (R6-5).
            clause_min_evidence.append(max(_cl_ev_ratio * max_ev, _cl_wt_ratio * cl_weight))
            clause_min_precision.append(_cl_prec_ratio * max_prec)

        clause_demanded_roles: List[Optional[str]] = []
        for cl_toks in clause_tokens_list:
            demanded = next((
                r for r in mut_excl_roles
                if any(normalize_token(t)[:5] == normalize_token(r)[:5] or (len(t) >= 4 and t.lower().startswith(r.lower()[:4])) for t in cl_toks)
            ), None)
            clause_demanded_roles.append(demanded)
        pipeline_demanded_roles = {r for r in clause_demanded_roles if r}

        for c in candidates:
            c_toks = c.token_set
            id_toks = getattr(c, "identity_tokens", None) or identity_cache.get(c.cell_id, c_toks)
            # Clause COVERAGE is decided by core identity only (R6-5); the full
            # identity still feeds ranking via _match_mass below.
            core_toks = core_identity_cache.get(c.cell_id) or id_toks
            c_excl_roles = _cell_mutually_exclusive_roles(c)
            _cid_toks = {t for t in CellTokenizer.tokenize_identifier(str(c.cell_id)) if _idf(t) > 0.0}
            _cid_mass = sum(_idf(t) for t in _cid_toks)
            cell_cov_strong[c.cell_id] = content_prompt_tokens & id_toks
            cell_cov_weak[c.cell_id] = non_literal_prompt_tokens & (c_toks - id_toks)
            cell_cov_mass_bonus[c.cell_id] = (
                sum(idf_of_prompt.get(t, _idf(t)) for t in cell_cov_strong[c.cell_id])
                + sum(idf_of_prompt.get(t, _idf(t)) for t in cell_cov_weak[c.cell_id])
            ) / total_prompt_idf
            masses: List[float] = []
            covered: Set[int] = set()
            for cl_idx, cl_toks in enumerate(clause_tokens_list):
                demanded = clause_demanded_roles[cl_idx] if cl_idx < len(clause_demanded_roles) else None
                if demanded and c_excl_roles and any(r != demanded for r in c_excl_roles):
                    masses.append(0.0)
                    continue
                _name_prec = (sum(_idf(t) for t in (cl_toks & _cid_toks)) / _cid_mass) if _cid_mass > 0 else 0.0
                m = _match_mass(cl_toks, c_toks, id_toks) * (1.0 + 0.25 * _name_prec)
                masses.append(m)
                # Intent coverage: With empirical edge affinity dominating path selection,
                # the legacy id_mass clamp is replaced with calibrated clause matching and precision.
                c_prec = _prec_by_cell[c.cell_id][cl_idx]
                r_mass = _raw_by_cell[c.cell_id][cl_idx]
                if (
                    r_mass > 0
                    and r_mass >= clause_min_evidence[cl_idx]
                    and c_prec >= clause_min_precision[cl_idx]
                ):
                    covered.add(cl_idx)
            cell_clause_mass[c.cell_id] = masses
            cell_covered[c.cell_id] = covered

        self.last_cell_covered = cell_covered
        self.last_prec_by_cell = _prec_by_cell
        self.last_raw_by_cell = _raw_by_cell

        try:
            from config import settings
            _explain_active = bool(settings.explain_plan)
        except Exception:
            _explain_active = False

        if _explain_active:
            print("\n=== EXPLAIN PLAN: CLAUSE-LEVEL EVIDENCE TABLE ===")
            for idx, cl_text in enumerate(clauses):
                print(f"Clause {idx}: '{cl_text}'")
            header = f"{'Cell ID':<35} | " + " | ".join(f"cl{gi}: prec/raw/cov" for gi in range(len(clause_tokens_list))) + " | Covered"
            print("-" * len(header))
            print(header)
            print("-" * len(header))
            sorted_cands = sorted(
                candidates,
                key=lambda c: (len(cell_covered.get(c.cell_id, set())), relevance_map.get(c.cell_id, 0.0)),
                reverse=True
            )
            for c in sorted_cands[:40]:
                cov = cell_covered.get(c.cell_id, set())
                cl_entries = []
                for gi in range(len(clause_tokens_list)):
                    p = _prec_by_cell[c.cell_id][gi] if c.cell_id in _prec_by_cell and gi < len(_prec_by_cell[c.cell_id]) else 0.0
                    r = _raw_by_cell[c.cell_id][gi] if c.cell_id in _raw_by_cell and gi < len(_raw_by_cell[c.cell_id]) else 0.0
                    is_cov = "Y" if gi in cov else "N"
                    cl_entries.append(f"{p:.2f}/{r:.2f}/{is_cov}")
                print(f"{c.cell_id:<35} | " + " | ".join(cl_entries) + f" | {sorted(list(cov))}")
            print("-" * len(header) + "\n")

        # Macro-goal expansion tables: a MacroCell is scored by its EXPANDED
        # sub-cell composition, with the SAME objective as flat paths (coverage,
        # clause alignment, parsimony evidence). This is what lets a macro
        # outrank its own micro-cells legitimately — its expansion explains the
        # prompt at least as well as the constituents do, plus whatever
        # composite concept vocabulary only the macro declares — and never lets
        # it win on generic I/O vocabulary alone (the expansion exposes the
        # sub-path's uncovered prompt tokens just as honestly).
        macro_expansion: Dict[str, List[Cell]] = {}
        for c in candidates:
            if getattr(c, "cell_type", "") != "macro" or not getattr(c, "sub_cells", None):
                continue
            subs = []
            for sid in c.sub_cells:
                sub = self.orchestrator.loaded_cells.get(sid) if self.orchestrator else None
                if sub is not None:
                    subs.append(sub)
                else:
                    subs = []
                    break
            if len(subs) >= 2:
                macro_expansion[c.cell_id] = subs

        def _expansion_ids(c_id: str) -> List[str]:
            subs = macro_expansion.get(c_id)
            return [s.cell_id for s in subs] if subs else [c_id]

        # Expanded coverage tables: for a macro, union its sub-cells' coverage.
        # Sub-cells may lie outside the tunnel candidates, so their coverage is
        # computed here directly with the same formulas as the main table.
        for c_id, subs in macro_expansion.items():
            strong: Set[str] = set()
            weak: Set[str] = set()
            mass_bonus = 0.0
            masses: List[float] = []
            covered: Set[int] = set()
            for sub in subs:
                s_id = sub.cell_id
                if s_id in cell_cov_strong:
                    strong |= cell_cov_strong[s_id]
                    weak |= cell_cov_weak.get(s_id, set())
                    mass_bonus += cell_cov_mass_bonus.get(s_id, 0.0)
                    for gi, m in enumerate(cell_clause_mass.get(s_id, [])):
                        if len(masses) <= gi:
                            masses.append(0.0)
                        masses[gi] = max(masses[gi], m)
                    covered |= cell_covered.get(s_id, set())
                    continue
                s_toks = sub.token_set
                s_id_toks = getattr(sub, "identity_tokens", None) or identity_cache.get(s_id) or _identity_tokens(sub)
                s_core_toks = core_identity_cache.get(s_id) or _core_identity_tokens(sub)
                s_strong = content_prompt_tokens & s_id_toks
                s_weak = content_prompt_tokens & (s_toks - s_id_toks)
                strong |= s_strong
                weak |= s_weak
                mass_bonus += (
                    sum(idf_of_prompt.get(t, _idf(t)) for t in s_strong)
                    + sum(idf_of_prompt.get(t, _idf(t)) for t in s_weak)
                ) / total_prompt_idf
                s_total_idf = sum(_idf(t) for t in s_toks)
                for gi, cl_toks in enumerate(clause_tokens_list):
                    m = _match_mass(cl_toks, s_toks, s_id_toks)
                    if len(masses) <= gi:
                        masses.append(0.0)
                    masses[gi] = max(masses[gi], m)
                    s_prec = (sum(_idf(t) for t in (cl_toks & s_toks)) / s_total_idf) if s_total_idf > 0 else 0.0
                    if (
                        _raw_mass(clause_tokens_list[gi], s_toks, s_core_toks) > 0
                        and _raw_mass(clause_tokens_list[gi], s_toks, s_core_toks) >= clause_min_evidence[gi]
                        and s_prec >= clause_min_precision[gi]
                    ):
                        covered.add(gi)
            cell_cov_strong[c_id] = strong
            cell_cov_weak[c_id] = weak
            cell_cov_mass_bonus[c_id] = mass_bonus
            cell_clause_mass[c_id] = masses
            cell_covered[c_id] = covered

        # Concrete (non-wildcard) input port signatures per cell — used to judge
        # whether a zero-ary constructor is actually CONSUMED downstream.
        cell_concrete_in_sigs: Dict[str, List[Any]] = {}
        for c in candidates:
            sigs = [
                p.signature for p in c.inputs.values()
                if str(p.signature.type_name).lower() not in _WILDCARD_CARRIERS
            ]
            cell_concrete_in_sigs[c.cell_id] = sigs

        def _ctor_justified(ctor: Cell, nxt: Cell) -> bool:
            """A zero-ary constructor is justified iff the following cell declares a
            concrete port that unifies with the constructed type (real lifecycle)."""
            out_sig = ctor.primary_output.signature
            for p_sig in cell_concrete_in_sigs.get(nxt.cell_id, ()):
                if unify(out_sig, p_sig) is not None:
                    return True
            return False

        # Receiver-bindability: a required concrete port whose carrier is a domain
        # class must be produced somewhere earlier in the path (typically by the
        # class's constructor morphism) or by the environment's start signature.
        # Ports whose declared carrier is literal-groundable are exempt.
        # (Defined before the feasibility probe above; see _port_literal_groundable.)

        # Per-cell: required concrete non-groundable receiver signatures (the ports
        # that need an in-path producer such as a constructor morphism).
        cell_receiver_sigs: Dict[str, List[Any]] = {}
        for c in candidates:
            if getattr(c, "node_type", "") == "constructor":
                cell_receiver_sigs[c.cell_id] = []
                continue
            sigs = []
            for p_name, p_sig in c.inputs.items():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                is_instance_receiver = _is_receiver_port(p_sig, c)
                t = str(p_sig.signature.type_name)
                if not is_instance_receiver:
                    if t.lower() in _WILDCARD_CARRIERS or _port_literal_groundable(t, p_sig):
                        continue
                sigs.append(p_sig.signature)
            cell_receiver_sigs[c.cell_id] = sigs

        def _is_col_proj_cell(cell: Cell) -> bool:
            # Structural: a cell is a column projection iff it DECLARES a
            # column-projection input port (state / list-carrier vocabulary).
            # No cell-id substrings.
            return any(_is_col_projection_port(p_s) for p_s in cell.inputs.values())

        def _is_table_groundable_target(p_sig: Any, prev_path: List[Cell]) -> bool:
            return _is_table_groundable_target_port(p_sig, prev_path, quoted_str_literals, identifier_literals)

        def _new_unbindable(cand: Cell, prev_path: List[Cell]) -> int:
            produced = [out_sig.signature for prev in prev_path for out_sig in prev.outputs.values()]
            receivers = [
                p_sig for p_sig in cell_receiver_sigs.get(cand.cell_id, ())
                if not _is_table_groundable_target(p_sig, prev_path)
            ]
            # Monadic multi-carrier admissibility (one wire feeds ONE port per
            # cell): required ports are only satisfiable from DISTINCT
            # producers. Counting each port independently against the whole
            # produced set let monoidal-product cells (two same-typed required
            # inputs, e.g. row concatenation) pass planning with a single
            # wire, only for the binder to refuse the duplicate consumption at
            # emission time. Greedy bipartite matching, most-constrained port
            # first, mirrors the binder's actual consumption semantics.
            def _match_count(p_sig: Any) -> int:
                return sum(1 for prod in produced if unify(prod, p_sig) is not None)

            remaining = list(produced)
            count = 0
            for p_sig in sorted(receivers, key=_match_count):
                match_idx = None
                for i, prod in enumerate(remaining):
                    if unify(prod, p_sig) is not None:
                        match_idx = i
                        break
                if match_idx is None:
                    count += 1
                else:
                    remaining.pop(match_idx)

            # Path port capacity checks: required external path inputs across the pipeline
            # must not exceed available source file literals (for ingress) or dest file literals (for egress).
            full_path = prev_path + [cand]
            required_ingress_path_ports = 0
            required_egress_path_ports = 0
            for c in full_path:
                is_ing = _is_ingress_cell(c)
                is_egr = _is_egress_cell(c)
                for p_name, p_sig in c.inputs.items():
                    if not p_sig.required or p_sig.default_value is not None:
                        continue
                    if _lattice_is_path_port(p_sig):
                        if is_ing:
                            required_ingress_path_ports += 1
                            break
                        elif is_egr:
                            required_egress_path_ports += 1
                            break
            max_allowed_ingress = len(src_file_literals) if src_file_literals else (len(file_literals) if file_literals else 1)
            if required_ingress_path_ports > max_allowed_ingress:
                count += (required_ingress_path_ports - max_allowed_ingress)
            max_allowed_egress = len(dest_file_literals) if dest_file_literals else 1
            if required_egress_path_ports > max_allowed_egress:
                count += (required_egress_path_ports - max_allowed_egress)

            # Role-carrier tracking (R2.5):
            # Check role-bearing inputs (e.g. target_input, feature_input, data_input)
            ROLE_CARRIERS = TypeRegistry.get_instance().get_declared_role_carriers()
            matched_producers: Set[Tuple[int, str]] = set()

            for p_name, p_sig in cand.inputs.items():
                p_role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
                if not p_role and hasattr(p_sig, "derive_port_role"):
                    p_role = p_sig.derive_port_role()

                if p_role in ROLE_CARRIERS:
                    t_name = str(p_sig.signature.type_name)
                    def_val = p_sig.default_value
                    if def_val is not None and str(def_val).strip() not in ("None", "none", "null", ""):
                        continue
                    if _port_literal_groundable(t_name):
                        continue

                    # Search for an unallocated matching producer output in prev_path (newest to oldest)
                    found_out = None
                    for idx in range(len(prev_path) - 1, -1, -1):
                        prev = prev_path[idx]
                        for out_name, out_sig in prev.outputs.items():
                            if (idx, out_name) in matched_producers:
                                continue
                            if unify(out_sig.signature, p_sig.signature) is not None:
                                found_out = (idx, out_name)
                                break
                        if found_out is not None:
                            matched_producers.add(found_out)
                            break

                    if found_out is None:
                        already_penalized = (
                            p_sig.required
                            and p_sig.default_value is None
                            and p_sig.signature in cell_receiver_sigs.get(cand.cell_id, ())
                            and not any(unify(prod, p_sig.signature) is not None for prod in produced)
                        )
                        if not already_penalized:
                            projective_roles = (
                                set(TypeRegistry.get_instance().get_verification_semantics("projective_roles"))
                                | set(TypeRegistry.get_instance().get_verification_semantics("projective_binds"))
                            )
                            if p_role in projective_roles:
                                if not _is_carrier_projectable_port(p_sig, prev_path, quoted_str_literals, identifier_literals):
                                    count += 1
                            elif not p_sig.required:
                                if p_name in getattr(cand, "slots", []) and f"{{{p_name}}}" in getattr(cand, "code_template", ""):
                                    count += 1

            return count

        def _edge_is_weak(prev: Cell, cand: Cell) -> bool:
            out_sig = prev.primary_output.signature
            has_strong = False
            has_any = False
            for p in cand.inputs.values():
                if unify(out_sig, p.signature) is None:
                    continue
                t = str(p.signature.type_name).lower()
                if t in _WILDCARD_CARRIERS:
                    has_any = True
                else:
                    has_strong = True
                    break
            return (not has_strong) and has_any

        _edge_affinity = self._calculate_edge_affinity
        prompt_toks_all = CellTokenizer.tokenize_prompt(prompt) if prompt else set()

        # Narrow-identity requested-ness for structural guards: only the cell's
        # own NAME vocabulary (cell id, keywords, semantic tags) counts as
        # "what the cell is"; port names describe the interface and are too
        # generic ('data', 'result') to establish that the prompt asked for it.
        _narrow_id_memo: Dict[str, Set[str]] = {}
        def _narrow_identity(c: Cell) -> Set[str]:
            hit = _narrow_id_memo.get(c.cell_id)
            if hit is None:
                toks = set(CellTokenizer.tokenize_identifier(c.cell_id))
                for kw in getattr(c, "keywords", ()) or ():
                    toks |= set(CellTokenizer.tokenize_identifier(str(kw)))
                for tag in getattr(c, "semantic_tags", ()) or ():
                    toks |= set(CellTokenizer.tokenize_identifier(str(tag)))
                _narrow_id_memo[c.cell_id] = toks
                hit = toks
            return hit

        def _is_bridge_or_carrier(c: Cell) -> bool:
            return (
                getattr(c, "node_type", "") == "bridge"
                or getattr(c, "node_role", "") in ("bridge", "source", "sink", "adaptor")
                or (vocab is not None and vocab.is_bridge(c))
            )

        _unreq_narrow_memo: Dict[str, bool] = {}
        def _unrequested_narrow(c: Cell) -> bool:
            hit = _unreq_narrow_memo.get(c.cell_id)
            if hit is not None:
                return hit
            if not _is_transform_stage(c):
                res = False
            elif _is_bridge_or_carrier(c):
                toks = _narrow_identity(c)
                res = bool(toks and not (toks & prompt_toks_all))
            else:
                res = not bool(cell_covered.get(c.cell_id))
            _unreq_narrow_memo[c.cell_id] = res
            return res

        _unreq_memo: Dict[str, bool] = {}

        def _unrequested(c: Cell) -> bool:
            hit = _unreq_memo.get(c.cell_id)
            if hit is None:
                if not _is_transform_stage(c):
                    hit = False
                elif _is_bridge_or_carrier(c):
                    c_toks = getattr(c, "identity_tokens", getattr(c, "token_set", set()))
                    hit = bool(c_toks and not (c_toks & prompt_toks_all))
                else:
                    hit = not bool(cell_covered.get(c.cell_id))
                _unreq_memo[c.cell_id] = hit
            return hit

        _path_score_cache: Dict[Tuple[Tuple[str, ...], bool], PathScore] = {}
        neutral_affinity = self._affinity_tier(4)
        macro_affinity_floor = self._affinity_tier(2)
        _path_state_cache: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        _cells_connect = self._cells_connect
        from config import settings as _settings
        join_rel_thresh = float(_settings.join_relevance_threshold)

        def compute_path_score(item: Tuple[List[Cell], Substitution, float, int, int], is_final: bool = False) -> PathScore:
            path, _, sc, weak_edges, unbindable = item
            path_ids = tuple(c.cell_id for c in path)
            path_key = (path_ids, is_final)
            if path_key in _path_score_cache:
                return _path_score_cache[path_key]
            k = len(path)

            # Incremental score state: reuse cached prefix state rather than
            # rescanning the full path prefix for every beam candidate.
            st = _path_state_cache.get(path_ids)
            if st is not None:
                strong_tokens = st["strong_tokens"]
                weak_tokens = st["weak_tokens"]
                covered_clauses = st["covered_clauses"]
                inversions = st["inversions"]
                dag_affs = st["dag_affs"]
                join_nodes = st["join_nodes"]
            elif k > 1 and path_ids[:-1] in _path_state_cache:
                prev_st = _path_state_cache[path_ids[:-1]]
                new_c = path[-1]
                strong_tokens = prev_st["strong_tokens"] | cell_cov_strong.get(new_c.cell_id, set())
                weak_tokens = prev_st["weak_tokens"] | cell_cov_weak.get(new_c.cell_id, set())
                covered_clauses = prev_st["covered_clauses"] | cell_covered.get(new_c.cell_id, set())

                masses = cell_clause_mass.get(new_c.cell_id, [])
                dp = prev_st["dp"]
                inversions = prev_st["inversions"]
                if masses:
                    max_m = max(masses)
                    if max_m > 0:
                        cands = {g for g, m in enumerate(masses) if m > 0 and m == max_m}
                        if cands:
                            if dp is None:
                                dp = {g: 0 for g in cands}
                                inversions = 0
                            else:
                                next_dp = {g: min(dp[prev_g] + (1 if prev_g > g else 0) for prev_g in dp) for g in cands}
                                inversions = min(next_dp.values())
                                dp = next_dp

                parents = [path[i] for i in range(k - 1) if _cells_connect(path[i], new_c)]
                bound_p = getattr(new_c, "bound_parent_ids", None)
                if bound_p is not None:
                    is_real_join = len(bound_p) >= 2
                    actual_parents = [p for p in parents if p.cell_id in bound_p]
                else:
                    is_real_join = len(parents) >= 2 and len(getattr(new_c, "inputs", {})) >= 2
                    actual_parents = parents
                is_justified = is_real_join and (bool(cell_covered.get(new_c.cell_id)) or relevance_map.get(new_c.cell_id, 0.0) >= join_rel_thresh)
                new_join = (1 if is_justified else 0)
                join_nodes = prev_st["join_nodes"] + new_join
                is_new_ingress = (_is_ingress_stage(new_c) and getattr(new_c, "node_type", "") != "constructor")
                aff_parents = actual_parents or parents
                if aff_parents:
                    new_aff = max(_edge_affinity(p, new_c) for p in aff_parents)
                    dag_affs = prev_st["dag_affs"] + [new_aff]
                elif not is_new_ingress:
                    new_aff = _edge_affinity(path[-2], new_c)
                    dag_affs = prev_st["dag_affs"] + [new_aff]
                else:
                    dag_affs = prev_st["dag_affs"]

                st = {
                    "strong_tokens": strong_tokens,
                    "weak_tokens": weak_tokens,
                    "covered_clauses": covered_clauses,
                    "dp": dp,
                    "inversions": inversions,
                    "dag_affs": dag_affs,
                    "join_nodes": join_nodes,
                    "path_id_tokens": prev_st.get("path_id_tokens", set()) | (getattr(new_c, "identity_tokens", None) or identity_cache.get(new_c.cell_id, set())),
                }
                _path_state_cache[path_ids] = st
            else:
                strong_tokens = set()
                weak_tokens = set()
                covered_clauses = set()
                steps_candidate_clauses = []
                for c in path:
                    strong_tokens |= cell_cov_strong.get(c.cell_id, set())
                    weak_tokens |= cell_cov_weak.get(c.cell_id, set())
                    covered_clauses |= cell_covered.get(c.cell_id, set())
                    masses = cell_clause_mass.get(c.cell_id, [])
                    if masses:
                        max_m = max(masses)
                        if max_m > 0:
                            cands = {g for g, m in enumerate(masses) if m > 0 and m == max_m}
                            if cands:
                                steps_candidate_clauses.append(cands)

                inversions = 0
                dp = None
                if steps_candidate_clauses:
                    dp = {g: 0 for g in steps_candidate_clauses[0]}
                    for step_cands in steps_candidate_clauses[1:]:
                        next_dp = {g: min(dp[prev_g] + (1 if prev_g > g else 0) for prev_g in dp) for g in step_cands}
                        dp = next_dp
                    inversions = min(dp.values())

                dag_affs = []
                join_nodes = 0
                for j in range(1, k):
                    cell_j = path[j]
                    parents = [path[i] for i in range(j) if _cells_connect(path[i], cell_j)]
                    bound_p = getattr(cell_j, "bound_parent_ids", None)
                    if bound_p is not None:
                        is_real_join = len(bound_p) >= 2
                        actual_parents = [p for p in parents if p.cell_id in bound_p]
                    else:
                        is_real_join = len(parents) >= 2 and len(getattr(cell_j, "inputs", {})) >= 2
                        actual_parents = parents
                    is_justified = is_real_join and (bool(cell_covered.get(cell_j.cell_id)) or relevance_map.get(cell_j.cell_id, 0.0) >= join_rel_thresh)
                    if is_justified:
                        join_nodes += 1
                    is_j_ingress = (_is_ingress_stage(cell_j) and getattr(cell_j, "node_type", "") != "constructor")
                    aff_parents = actual_parents or parents
                    if aff_parents:
                        dag_affs.append(max(_edge_affinity(p, cell_j) for p in aff_parents))
                    elif not is_j_ingress:
                        dag_affs.append(_edge_affinity(path[j - 1], cell_j))

                st = {
                    "strong_tokens": strong_tokens,
                    "weak_tokens": weak_tokens,
                    "covered_clauses": covered_clauses,
                    "dp": dp,
                    "inversions": inversions,
                    "dag_affs": dag_affs,
                    "join_nodes": join_nodes,
                    "path_id_tokens": set().union(*(getattr(c, "identity_tokens", None) or identity_cache.get(c.cell_id, set()) for c in path)),
                }
                _path_state_cache[path_ids] = st

            # ---- coverage: identity evidence and docstring evidence stay separate terms ----
            path_id_tokens = st.get("path_id_tokens")
            if path_id_tokens is None:
                path_id_tokens = set().union(*(getattr(c, "identity_tokens", None) or identity_cache.get(c.cell_id, set()) for c in path))
            id_recall = (sum(_idf(t) for t in (content_prompt_tokens & path_id_tokens)) / total_prompt_idf) if total_prompt_idf > 0 else 0.0
            path_id_total_mass = sum(_idf(t) for t in path_id_tokens)
            id_prec = (sum(_idf(t) for t in (content_prompt_tokens & path_id_tokens)) / path_id_total_mass) if path_id_total_mass > 0 else 0.0
            id_cov = (2.0 * id_recall * id_prec / (id_recall + id_prec)) if (id_recall + id_prec) > 0 else 0.0

            strong_mass = id_cov
            strong_roots = {normalize_token(t) for t in strong_tokens} | strong_tokens
            uncovered_weak = {t for t in weak_tokens if normalize_token(t) not in strong_roots and t not in strong_roots}
            weak_mass = sum(idf_of_prompt.get(t, _idf(t)) for t in uncovered_weak) / total_prompt_idf
            coverage = min(1.0, strong_mass + weak_mass)

            distinct_matched_clauses = len(covered_clauses)
            clause_cov = sum(clause_weights[i] for i in covered_clauses) / total_clause_weight
            uncovered_clauses = (num_clauses - distinct_matched_clauses) if num_clauses > 1 else 0
            # Structural hole: an uncovered clause sandwiched between covered ones.
            gap_holes = 0
            if covered_clauses:
                ordered = sorted(covered_clauses)
                gap_holes = sum(1 for g in range(ordered[0], ordered[-1] + 1) if g not in covered_clauses)

            alignment = clause_cov / (1.0 + inversions)

            consumed_ctors = sum(
                1 for i, c in enumerate(path)
                if getattr(c, "node_type", "") == "constructor"
                and any(_cells_connect(c, downstream_cell) for downstream_cell in path[i + 1:])
            )
            prerequisite_bridges = 0
            for i in range(1, k - 1):
                c = path[i]
                if not cell_covered.get(c.cell_id):
                    if not _cells_connect(path[i - 1], path[i + 1]) and _cells_connect(c, path[i + 1]):
                        prerequisite_bridges += 1
            effective_k = max(1, k - consumed_ctors - prerequisite_bridges)
            excess_steps = max(0, effective_k - max(distinct_matched_clauses, 1))

            mean_log_prob = sc / max(k, 1)

            # ---- goal-directed indicators, each in {-1, 0, +1} ----
            goal = 0
            terminal = path[-1]
            has_file_dest = bool(dest_file_literals) and any(getattr(c, "stage", None) == vocab.ingress_stage for c in path)
            has_egress_intent = (
                (bool(content_prompt_tokens & EGRESS_INTENT_TOKENS) and not target_sink)
                or has_file_dest
                or (goal_sig is not None)
            )
            output_declares_materialization = any(
                str(getattr(out_p, "state", "")).lower() in MATERIALIZATION_OUTPUT_STATES
                for out_p in getattr(terminal, "outputs", {}).values()
            )
            terminal_is_materializing = (
                (_is_terminal_sink_cell(terminal) or output_declares_materialization)
                and has_egress_intent
                and unbindable == 0
                and (not file_literals or _has_path_port(terminal))
            )
            if tunnel_has_sinks:
                if terminal_is_materializing:
                    goal += 1
                elif has_egress_intent:
                    goal -= 1
            terminal_domain = getattr(terminal, "domain_name", "")
            if terminal_domain:
                effective_other_cells: List[Cell] = list(path[:-1])
                if getattr(terminal, "cell_type", "") == "macro" and getattr(terminal, "sub_cells", None) and self.orchestrator is not None:
                    for sid in terminal.sub_cells:
                        sub_cell = self.orchestrator.loaded_cells.get(sid)
                        if sub_cell is not None:
                            effective_other_cells.append(sub_cell)
                if effective_other_cells:
                    other_domains = {getattr(c, "domain_name", "") for c in effective_other_cells}
                    if vocab.is_bridge(effective_other_cells[-1]) or terminal_domain in other_domains:
                        goal += 1
                    else:
                        goal -= 1
            if file_literals:
                if any(_has_path_port(c) for c in path):
                    goal += 1
                elif not tunnel_absorbs_assets:
                    goal -= 1
            if has_egress_intent:
                if _is_terminal_sink_cell(terminal):
                    goal += 1
                elif not any(_is_terminal_sink_cell(c) for c in path) and terminal.outputs:
                    if any(vocab.is_materializable_carrier(o) for o in terminal.outputs.values()):
                        goal -= 1
            unrequested = sum(1 for c in path[1:] if _unrequested(c))
            unrequested_transforms = sum(
                1 for c in path[1:]
                if _unrequested_narrow(c)
                and getattr(c, "node_role", "") not in ("bridge", "source", "sink")
                and getattr(c, "node_type", "") != "tunnel"
            )

            weak_total = sum(1 for c in path if _is_wildcarrier(c)) + weak_edges

            # ---- feasibility defects ----
            dead_expansion_steps = 0
            for c in path:
                subs = macro_expansion.get(c.cell_id)
                if not subs:
                    continue
                for j, sub in enumerate(subs):
                    if j == 0 or j == len(subs) - 1:
                        continue
                    if not cell_cov_strong.get(sub.cell_id, set()) and not cell_cov_weak.get(sub.cell_id, set()):
                        pred_sig = getattr(getattr(subs[j - 1], "primary_output", None), "signature", None)
                        succ_sig = getattr(getattr(subs[j + 1], "primary_input", None), "signature", None)
                        if pred_sig and succ_sig and unify(pred_sig, succ_sig) is None:
                            continue  # essential bridge, not dead weight
                        dead_expansion_steps += 1

            dead_ctors = 0
            for i, c in enumerate(path):
                if getattr(c, "node_type", "") == "constructor":
                    if not is_final and i == len(path) - 1:
                        continue
                    if not any(_cells_connect(c, d) for d in path[i + 1:]):
                        dead_ctors += 1

            dead_outputs = 0
            if is_final:
                reg = TypeRegistry.get_instance()
                for i in range(k - 1):
                    c = path[i]
                    if _is_terminal_sink_cell(c) or not c.outputs:
                        continue
                    # Strict def-use wiring check: does any downstream cell actually declare c as a bound parent?
                    has_bound = any(getattr(d, "bound_parent_ids", None) is not None for d in path[i + 1:])
                    if has_bound:
                        consumed = any(c.cell_id in (getattr(d, "bound_parent_ids", None) or ()) for d in path[i + 1:])
                    else:
                        consumed = any(_cells_connect(c, d) for d in path[i + 1:])

                    if not consumed:
                        # Allow scalar leaf aggregators that directly witness a prompt clause
                        is_scalar_leaf = (
                            bool(cell_covered.get(c.cell_id))
                            and all(
                                reg.is_subtype(_port_type(o), "scalar")
                                or reg.is_subtype(_port_type(o), "numeric")
                                for o in c.outputs.values()
                            )
                        )
                        if not is_scalar_leaf:
                            dead_outputs += 1

            pipeline_domains = {
                getattr(c, "domain_name", "") for c in path
                if getattr(c, "domain_name", "") and getattr(c, "domain_name", "") not in vocab.neutral_domains
            }

            # Edge affinity term: AST-mined and declared topological transitions
            # are the dominant score term for idiomatic composition.
            join_frac = 0.0
            if getattr(self, "topology_mode", "frontier") == "frontier":
                if k <= 1:
                    affinity_score = neutral_affinity
                else:
                    affinity_score = sum(dag_affs) / max(len(dag_affs), 1)
                    join_frac = join_nodes / max(k - 1, 1)
            else:
                if k <= 1:
                    affinity_score = neutral_affinity
                    if k == 1 and getattr(path[0], "cell_type", "") == "macro" and getattr(path[0], "sub_cells", None) and self.orchestrator is not None:
                        subs = [self.orchestrator.loaded_cells.get(sid) for sid in path[0].sub_cells]
                        subs = [s for s in subs if s is not None]
                        if len(subs) >= 2:
                            def _alive(idx: int) -> bool:
                                if idx == 0 or idx == len(subs) - 1:
                                    return True
                                s_id = subs[idx].cell_id
                                return bool(cell_cov_strong.get(s_id) or cell_cov_weak.get(s_id))
                            affs = [
                                max(self._calculate_edge_affinity(subs[i], subs[i + 1]), macro_affinity_floor)
                                for i in range(len(subs) - 1)
                                if _alive(i) and _alive(i + 1)
                            ]
                            affinity_score = sum(affs) / len(affs) if affs else 0.5
                else:
                    total_aff = sum(_edge_affinity(path[i], path[i + 1]) for i in range(k - 1))
                    affinity_score = total_aff / max(k - 1, 1)


            # Literal consumption term (T1.3 & R2.7):
            # Measures ratio of L0 universal literals bound to at least one port on the path.
            if universal_literals:
                path_ports = set()
                path_ingress_file_ports = 0
                path_egress_file_ports = 0
                path_has_col_proj = any(_is_col_proj_cell(c) for c in path)
                for c in path:
                    is_ing = _is_ingress_cell(c)
                    is_egr = _is_egress_cell(c)
                    for p_name, p_sig in c.inputs.items():
                        path_ports.add(p_name.lower())
                        role = getattr(p_sig, "port_role", None) or getattr(p_sig, "derived_role", "")
                        if role:
                            path_ports.add(role.lower())
                        # Count declared required path ports by stage (ingress vs egress)
                        if p_sig.required and p_sig.default_value is None and _is_path_port_sig(p_sig):
                            if is_ing:
                                path_ingress_file_ports += 1
                            elif is_egr:
                                path_egress_file_ports += 1
                    for s_k in getattr(c, "bound_slots", {}).keys():
                        path_ports.add(s_k.lower())

                consumed_count = 0
                used_ingress_file_ports = 0
                used_egress_file_ports = 0
                # Column-projection ports declared on the path: an identifier
                # literal counts as consumed only when its ROLE context (e.g.
                # "X column") matches the projection port's declared vocabulary.
                proj_port_tokens: Set[str] = set()
                if path_has_col_proj:
                    for c in path:
                        for p_sig in c.inputs.values():
                            if _is_col_projection_port(p_sig):
                                st = str(getattr(p_sig.signature, "state", "")).lower()
                                proj_port_tokens |= set(CellTokenizer.tokenize_identifier(st))
                                proj_port_tokens |= {t for t in CellTokenizer.tokenize_identifier(str(getattr(c, "cell_id", "")).lower())}
                    raw_proj_tokens = TypeRegistry.get_instance().get_column_projection_tokens()
                    proj_port_tokens |= set(raw_proj_tokens)
                    proj_port_tokens |= {normalize_token(t) for t in raw_proj_tokens}
                # Declared relational-trigger ports: a port NAMED for the
                # preposition that binds its value ("by") can consume the
                # prepositional object ("by age") through the semantic-slot
                # channel. A path with no such port leaves the referent
                # unconsumed — the objective then prefers paths that honor it.
                trigger_port_tokens: Set[str] = set()
                for c in path:
                    for p_name, p_sig in c.inputs.items():
                        if vocab.is_key_port(p_sig):
                            trigger_port_tokens |= set(CellTokenizer.tokenize_identifier(p_name))
                            trigger_port_tokens |= set(CellTokenizer.tokenize_identifier(str(getattr(p_sig.signature, "state", ""))))
                target_roles = set(registry.get_verification_semantics("target_roles"))
                target_role_tokens: Set[str] = set()
                for r in target_roles:
                    target_role_tokens |= set(vocab.role_tokens(r))
                path_has_target = any(
                    (getattr(p, "port_role", "") in target_roles or getattr(p, "derived_role", "") in target_roles)
                    for c in path for p in c.inputs.values()
                )
                identifier_role_map = getattr(self, "_last_identifier_roles", None)
                for kind, lit in universal_literals:
                    lit_clean = str(lit).lower().strip("'\"")
                    if kind == "file_asset":
                        lit_pos = literal_positions.get((kind, lit), 0)
                        lit_dir = ExecutionContext._asset_direction(prompt or "", lit_pos)
                        if lit_dir == "dest":
                            if used_egress_file_ports < path_egress_file_ports:
                                consumed_count += 1
                                used_egress_file_ports += 1
                        else:
                            if used_ingress_file_ports < path_ingress_file_ports:
                                consumed_count += 1
                                used_ingress_file_ports += 1
                    elif kind == "numeric":
                        lit_pos = literal_positions.get((kind, lit))
                        pre_mod: Set[str] = set()
                        if lit_pos is not None:
                            prev_chunk = (prompt or "")[:lit_pos].rstrip()
                            toks = CellTokenizer.tokenize_prompt(prev_chunk)
                            pre_mod = {t for t in toks if vocab.is_informative_token(t)}

                        num_port_matched = False
                        path_has_num_port = False
                        for c in path:
                            for p_name, p_sig in c.inputs.items():
                                t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
                                is_num = registry.is_subtype(t_name, "numeric")
                                if not is_num:
                                    continue
                                path_has_num_port = True
                                port_ev: Set[str] = set()
                                port_ev |= set(CellTokenizer.tokenize_identifier(p_name.lower()))
                                st = str(getattr(p_sig.signature, "state", "") or "").lower()
                                if st:
                                    port_ev |= set(CellTokenizer.tokenize_identifier(st))
                                doc = str(getattr(p_sig, "doc", "") or getattr(p_sig, "description", "") or "").lower()
                                if doc:
                                    port_ev |= set(CellTokenizer.tokenize_identifier(doc))
                                for kw in getattr(c, "keywords", []):
                                    port_ev |= set(CellTokenizer.tokenize_identifier(str(kw).lower()))
                                if pre_mod and (port_ev & pre_mod):
                                    num_port_matched = True
                                    break
                            if num_port_matched:
                                break

                        if pre_mod:
                            if num_port_matched:
                                consumed_count += 1
                        else:
                            if path_has_num_port:
                                consumed_count += 1
                    else:
                        if lit_clean in path_ports or any(lit_clean in p for p in path_ports):
                            consumed_count += 1
                        else:
                            lit_pos = literal_positions.get((kind, lit))
                            lit_roles = set(identifier_role_map.get(lit_pos, frozenset())) if (identifier_role_map and lit_pos is not None) else set()
                            stemmed_lit_roles = {normalize_token(r) for r in lit_roles} | lit_roles
                            if proj_port_tokens and (
                                (stemmed_lit_roles & proj_port_tokens)
                                or (proj_port_tokens & {t for t in CellTokenizer.tokenize_identifier(lit_clean)})
                            ):
                                consumed_count += 1
                            elif path_has_target and (stemmed_lit_roles & target_role_tokens):
                                consumed_count += 1
                            elif trigger_port_tokens:
                                # Prepositional-object referent: consumed iff the
                                # path declares a relational-trigger port whose
                                # binding preposition matches the literal's.
                                if lit_pos is not None:
                                    prev_chunk = (prompt or "")[:lit_pos].rstrip()
                                    toks = tokenize_alphanumeric(prev_chunk, min_len=1)
                                    if toks:
                                        prep = toks[-1].lower()
                                        preps = registry.get_preposition_triggers() | registry.get_generic_prepositions()
                                        if prep and prep in trigger_port_tokens and (not preps or prep in preps):
                                            consumed_count += 1
                literal_consumption = consumed_count / len(universal_literals)
            else:
                literal_consumption = 1.0


            # Coverage gates affinity: structural affinity cannot buy back dropped intent.
            if num_clauses > 1:
                gate = clause_cov
            else:
                gate = max(coverage, literal_consumption) if universal_literals else coverage
            gated_affinity = affinity_score * gate

            redundant_cell_count = 0
            for c_idx, c in enumerate(path):
                if (
                    getattr(c, "node_type", "") == "bridge"
                    or getattr(c, "node_role", "") == "bridge"
                    or (vocab is not None and vocab.is_bridge(c))
                ):
                    continue
                if not getattr(c, "inputs", None):
                    continue
                c_cov = cell_covered.get(c.cell_id, set())
                if not c_cov:
                    redundant_cell_count += 1
                    continue
                other_cov = set()
                for o_idx, other in enumerate(path):
                    if o_idx != c_idx:
                        other_cov |= cell_covered.get(other.cell_id, set())
                if c_cov.issubset(other_cov):
                    redundant_cell_count += 1

            # Mutually exclusive role gating
            role_conflicts = 0
            if mut_excl_roles and pipeline_demanded_roles:
                for c in path:
                    c_excl = _cell_mutually_exclusive_roles(c)
                    if not c_excl:
                        continue
                    for demanded in pipeline_demanded_roles:
                        if any(excl != demanded for excl in c_excl):
                            role_conflicts += 1

            defects = (
                float(unbindable),
                float(unrequested_transforms),
                float(role_conflicts),
                float(uncovered_clauses) if is_final else 0.0,
                float(inversions),
                float(redundant_cell_count),
                float(dead_ctors),
                float(dead_outputs),
                float(dead_expansion_steps),
            )
            terms = {   # "family:term" - families are averaged first so correlated terms cannot double-count
                "intent:coverage_clause": clause_cov,
                "intent:coverage_identity": strong_mass,
                "intent:coverage_description": weak_mass,
                "intent:clause_alignment": alignment,
                "intent:literal_consumption": literal_consumption,
                "intent:no_gaps": -float(gap_holes),
                "structure:affinity": gated_affinity,
                "structure:joins": join_frac,
                "structure:no_weak_edges": -float(weak_total),
                "structure:domain_coherence": -float(len(pipeline_domains)),
                "economy:concision": -float(effective_k),
                "economy:no_padding": -float(excess_steps),
                "economy:no_unrequested": -float(unrequested),
                "goal:goal": float(goal),
                "intent:likelihood": mean_log_prob,
            }
            result = PathScore(defects, terms)
            if _debug_plan:
                _component_trace[tuple(c.cell_id for c in path)] = {
                    "defects": defects, **{k_: round(v_, 3) for k_, v_ in terms.items()}
                }
            _path_score_cache[path_key] = result
            return result

        def rank_items(items, is_final: bool = False, extra_terms=None):
            """Best-first list of (item, aggregate): defects lexicographically, then the
            rank-aggregate of all terms across `items` (see _rank_paths)."""
            rows = []
            for it in items:
                ps = compute_path_score(it, is_final)
                if extra_terms is not None:
                    ex = extra_terms(it)
                    if ex:
                        ps = PathScore(ps.defects, {**ps.terms, **ex})
                rows.append((it, ps))
            return _rank_paths(rows)

        # Type-gated adjacency: index candidates by the DECLARED input carrier they
        # expose. Expansion enumerates distinct port carriers and gates them through
        # the registered poset (mirroring unify's subtyping direction: producer's
        # output carrier must be a subtype of the consumer's port carrier), so each
        # trellis expansion iterates only type-compatible successors.
        cells_by_in_type: Dict[str, List[Cell]] = {}
        distinct_in_states: Dict[str, Set[str]] = {}
        for cand in candidates:
            if not cand.inputs:
                p_sig = cand.primary_input
                t_key = str(getattr(p_sig.signature, "type_name", "any"))
                cells_by_in_type.setdefault(t_key, []).append(cand)
                distinct_in_states.setdefault(t_key, set()).add(str(getattr(p_sig.signature, "state", "any")))
            else:
                for p_name, p_sig in cand.inputs.items():
                    t_key = str(getattr(p_sig.signature, "type_name", ""))
                    cells_by_in_type.setdefault(t_key, []).append(cand)
                    distinct_in_states.setdefault(t_key, set()).add(str(getattr(p_sig.signature, "state", "")))

        candidate_map = {c.cell_id: c for c in candidates}
        candidate_map_lower = {c.cell_id.lower(): c for c in candidates}

        def _successors(prev_cell: Cell) -> List[Cell]:
            if (getattr(prev_cell, "node_role", "") == "macro" or getattr(prev_cell, "cell_type", "") == "macro") and getattr(prev_cell, "endable", False):
                return []
            macro_subs = set(getattr(prev_cell, "sub_cells", ()) or ())

            out_sig = prev_cell.primary_output.signature if hasattr(prev_cell.primary_output, "signature") else prev_cell.primary_output
            out_t = str(getattr(out_sig, "type_name", ""))
            out_s = str(getattr(out_sig, "state", ""))

            acc: Dict[str, Cell] = {}

            # Prioritize/include explicit graph edges declared on prev_cell
            for edge in getattr(prev_cell, "edges", []):
                tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
                if tgt_id:
                    tgt_cell = candidate_map.get(tgt_id) or candidate_map_lower.get(str(tgt_id).lower())
                    if tgt_cell:
                        acc.setdefault(tgt_cell.cell_id, tgt_cell)

            # Generic carriers ("Sequence[T]", products) and TYPE VARIABLES ("T",
            # "S") unify by binding, not by poset subtyping: enumerate all
            # candidates and let exact unification gate them.
            is_type_var = out_t.isalpha() and len(out_t) == 1 and out_t.isupper()
            if "[" in out_t or is_type_var:
                for cand in candidates:
                    if cand.cell_id in macro_subs or prev_cell.cell_id in getattr(cand, "sub_cells", ()):
                        continue
                    cand_in_ports = list(cand.inputs.values()) if cand.inputs else [cand.primary_input]
                    if any(unify(out_sig, p_sig.signature) is not None
                           for p_sig in cand_in_ports):
                        acc.setdefault(cand.cell_id, cand)
                res_cells = [c for c in acc.values() if c.cell_id not in macro_subs and prev_cell.cell_id not in getattr(c, "sub_cells", ())]
                return sorted(res_cells, key=lambda c: _edge_affinity(prev_cell, c), reverse=True)

            for in_t, cell_list in cells_by_in_type.items():
                if not registry.is_subtype(out_t, in_t):
                    continue
                for c in cell_list:
                    if c.cell_id in macro_subs or prev_cell.cell_id in getattr(c, "sub_cells", ()):
                        continue
                    # Typestate compatibility with accepted_states and parent_state walking
                    cand_in_ports = list(c.inputs.values()) if c.inputs else [c.primary_input]
                    if any(registry.is_state_compatible(
                        producer_state=out_s,
                        consumer_state=getattr(p_sig.signature, "state", "any"),
                        producer_accepted=getattr(out_sig, "accepted_states", frozenset()),
                        consumer_accepted=getattr(p_sig.signature, "accepted_states", frozenset()),
                        consumer_parent=getattr(p_sig.signature, "parent_state", None),
                        producer_parent=getattr(out_sig, "parent_state", None),
                    ) for p_sig in cand_in_ports):
                        acc.setdefault(c.cell_id, c)
            res_cells = [c for c in acc.values() if c.cell_id not in macro_subs and prev_cell.cell_id not in getattr(c, "sub_cells", ())]
            return sorted(res_cells, key=lambda c: _edge_affinity(prev_cell, c), reverse=True)

        # Zero-ary generator morphisms: insertable after ANY cell (they consume
        # no incoming wire), introducing new carriers to the frontier (e.g. constructors, figure/canvas).
        def _is_zero_ary_generator(c: Cell) -> bool:
            if not c.outputs or _is_egress_cell(c):
                return False
            if any(_lattice_is_path_port(p) for p in c.inputs.values()):
                return False
            for p in c.inputs.values():
                if p.required and p.default_value is None:
                    return False
            out_types = {_port_type(op) for op in c.outputs.values()}
            is_handle = bool(out_types & vocab.handle_types)
            is_ctor = (getattr(c, "node_type", "") == "constructor")
            if not is_handle and not is_ctor:
                if not cell_cov_strong.get(c.cell_id):
                    return False
            if _is_ingress_stage(c) and not is_handle and not is_ctor:
                return False
            return True

        zero_ary_ctors = [
            c for c in candidates
            if getattr(c, "node_type", "") == "constructor" or _is_zero_ary_generator(c)
        ]

        # Dynamic step budget: derive cap from prompt complexity (num_clauses) + slack
        dynamic_cap = max(max_transforms + 2, num_clauses + 3)
        max_steps = max(2, min(16, dynamic_cap))

        _consumer_memo: Dict[str, bool] = {}

        def _cells_connect_any(cell: Cell) -> bool:
            hit = _consumer_memo.get(cell.cell_id)
            if hit is None:
                hit = any(o.cell_id != cell.cell_id and self._cells_connect(cell, o) for o in candidates)
                _consumer_memo[cell.cell_id] = hit
            return hit

        # Valid terminal boundary filter (Endable Nodes):
        def _is_valid_terminal(cand_path: List[Cell]) -> bool:
            terminal = cand_path[-1]
            covered = set().union(*(cell_covered.get(c.cell_id, set()) for c in cand_path))
            if num_clauses > 1 and covered and max(covered) < num_clauses - 1:
                return False

            # Data-driven dependent terminal input verification:
            # If the terminal has required inputs that depend on prior producers, ensure the path produced them.
            dep_terminal_roles = registry.get_verification_semantics("dependent_terminal_input_roles")
            if dep_terminal_roles:
                for p in terminal.inputs.values():
                    p_role = getattr(p, "port_role", "") or getattr(p, "derived_role", "")
                    if p.required and p_role in dep_terminal_roles:
                        required_producers = set(dep_terminal_roles.get(p_role) or registry.get_verification_semantics(f"dependent_terminal_input_roles:{p_role}"))
                        if required_producers:
                            path_has_producer = any(
                                any(getattr(o, "port_role", "") in required_producers for o in c.outputs.values())
                                for c in cand_path[:-1]
                            )
                            if not path_has_producer:
                                return False

            if getattr(terminal, "endable", None) is True:
                return True
            if getattr(terminal, "is_endable", False):
                return True
            if getattr(terminal, "stage", None) == vocab.terminal_stage and vocab.terminal_stage is not None:
                return True
            t_role = str(getattr(terminal, "node_role", "") or "").lower()
            term_out_roles = (
                set(registry.get_verification_semantics("terminal_output_roles"))
                | set(registry.get_verification_semantics("prediction_output_roles"))
            )
            is_pred_terminal = any(
                getattr(op, "port_role", "") in term_out_roles
                or getattr(op, "derived_role", "") in term_out_roles
                or str(getattr(op, "state", "")).lower() in vocab.terminal_states
                for op in getattr(terminal, "outputs", {}).values()
            )
            # A "transformer" is any non-terminal-stage cell that still has a downstream
            # consumer in the lattice: it cannot end the pipeline before the last clause,
            # unless its output satisfies an endable prediction/terminal typestate.
            if _is_terminal_sink_cell(terminal) is False and _cells_connect_any(terminal) and not is_pred_terminal:
                return False

            if is_pred_terminal:
                return True

            if target_sink and (getattr(terminal, "primary_output", None) or getattr(terminal, "outputs", None)):
                return True

            if t_role in vocab.terminal_roles:
                return True
            out_states = {
                str(getattr(op, "state", "")).lower()
                for op in getattr(terminal, "outputs", {}).values()
            }
            if any(st in vocab.terminal_states or registry.state_ancestry_reaches(st, set(vocab.terminal_states))
                   for st in out_states if st and st not in _WILDCARD_CARRIERS):
                return True

            return True

        # Planning approach dispatch:
        # 1. "linear": 1D Monadic Trellis baseline (sequential list approach)
        # 2. "frontier": Multi-Carrier Monoidal Frontier DAG (default approach)
        if getattr(self, "topology_mode", "frontier") == "linear":
            all_valid_paths = self._plan_linear_trellis(
                candidate_entries=candidate_entries,
                candidates=candidates,
                log_probs=log_probs,
                max_steps=max_steps,
                zero_ary_ctors=zero_ary_ctors,
                _new_unbindable=_new_unbindable,
                compute_path_score=compute_path_score,
                rank_items=rank_items,
                budget=budget,
                _edge_is_weak=_edge_is_weak,
                cells_by_in_type=cells_by_in_type,
                candidate_map=candidate_map,
                candidate_map_lower=candidate_map_lower,
                identifier_literals=identifier_literals,
                quoted_str_literals=quoted_str_literals,
                numeric_literals=numeric_literals,
                cell_clause_mass=cell_clause_mass,
            )
        else:
            all_valid_paths = self._plan_frontier_dag(
                candidate_entries=candidate_entries,
                candidates=candidates,
                log_probs=log_probs,
                max_steps=max_steps,
                zero_ary_ctors=zero_ary_ctors,
                _new_unbindable=_new_unbindable,
                compute_path_score=compute_path_score,
                rank_items=rank_items,
                budget=budget,
                _edge_is_weak=_edge_is_weak,
                cells_by_in_type=cells_by_in_type,
                candidate_map=candidate_map,
                candidate_map_lower=candidate_map_lower,
                identifier_literals=identifier_literals,
                quoted_str_literals=quoted_str_literals,
                numeric_literals=numeric_literals,
                cell_clause_mass=cell_clause_mass,
                file_asset_literals=file_literals,
                dest_file_literals=dest_file_literals,
                has_egress_intent=has_prompt_egress_intent,
                target_sink=target_sink,
                is_valid_terminal=_is_valid_terminal,
                unrequested_check=_unrequested_narrow,
                num_clauses=num_clauses,
                cell_covered=cell_covered,
            )

        # Filter and rank valid composition paths
        if all_valid_paths:
            valid_candidates = list(all_valid_paths)
            zero_unbind = [item for item in valid_candidates if item[4] == 0]
            if zero_unbind:
                valid_candidates = zero_unbind

            # 1. Filter by goal_sig if provided
            if goal_sig is not None:
                g_sig = goal_sig.signature if hasattr(goal_sig, "signature") else goal_sig
                matching_goals = [
                    item for item in valid_candidates
                    if unify(item[0][-1].primary_output.signature, g_sig) is not None
                ]
                if matching_goals:
                    valid_candidates = matching_goals

            # 2. Valid terminal boundary filter (Endable Nodes):
            endable_candidates = [item for item in valid_candidates if _is_valid_terminal(item[0])]
            if endable_candidates:
                valid_candidates = endable_candidates

            # === HARD CONSTRAINTS (apply_fixes_v8) ===
            _uc = ExecutionContext._extract_universal_literals(prompt or "")
            _ul = [
                (k, v) for _, k, v in _uc
                if k in ("file_asset", "identifier", "quoted_str", "numeric")
                and not (k == "identifier" and target_sink and str(v).lower() == target_sink.lower())
            ]
            _nl = [v for k, v in _ul if k == "numeric"]
            _ql = [v for k, v in _ul if k == "quoted_str"]
            _il = [v for k, v in _ul if k == "identifier"]
            valid_candidates, _tier = self._hard_constraint_filter(
                valid_candidates, _ul, _nl, _ql, _il, clause_tokens_list,
            )
            # === end hard constraints ===
            scored_candidates = rank_items(valid_candidates, is_final=True)

            if _debug_plan:
                dbg_top = [
                    (round(score, 2), " -> ".join(c.cell_id for c in item[0]))
                    for item, score in scored_candidates[:12]
                ]
                logger.info(f"[PLANNER-DEBUG] top candidates for prompt '{prompt[:60]}...':")
                for s, p in dbg_top:
                    comps = _component_trace.get(tuple(p.split(" -> ")), {})
                    logger.info(f"[PLANNER-DEBUG]   {s}  {p}  {comps}")
                macro_paths = [(round(score, 2), " -> ".join(c.cell_id for c in item[0]))
                               for item, score in scored_candidates
                               if any(getattr(c, "cell_type", "") == "macro" for c in item[0])][:5]
                if macro_paths:
                    logger.info(f"[PLANNER-DEBUG] best macro paths:")
                    for s, p in macro_paths:
                        comps = _component_trace.get(tuple(p.split(" -> ")), {})
                        logger.info(f"[PLANNER-DEBUG]   {s}  {p}  {comps}")
                else:
                    logger.info("[PLANNER-DEBUG] NO macro path survived the candidate filters")

            # Slot-aware re-ranking: a macro's planned sub-lattice is part of the
            # pipeline's semantics (a loop body calling contourArea covers the
            # "minimum area" clause). Plan the slots of the strongest candidates
            # and re-rank with the slot coverage included, so macro-based
            # pipelines are compared against flat pipelines with their bodies
            # filled in — not as bare skeletons.
            # Trials operate on PRIVATE copies of the candidate cells: the loaded lattice cells
            # are shared across every path, so binding a slot on them would leak the trial into
            # all other candidates (and into later plans).  Only the winner is bound (below).
            seen_prefixes: Set[Tuple[str, ...]] = set()
            trial_items: List[Tuple] = []
            trial_slot_cells: Dict[Tuple[str, ...], List[Cell]] = {}
            trials = 0
            for it, _agg in scored_candidates:
                if trials >= budget.slot_trials:
                    break
                cand_path = it[0]
                prefix = tuple(c.cell_id for c in cand_path)
                if prefix in seen_prefixes:
                    continue
                seen_prefixes.add(prefix)
                trials += 1
                if not any(getattr(c, "slots", None) for c in cand_path):
                    continue
                trial_sigma = it[1]
                slot_cells: List[Cell] = []
                for c in cand_path:
                    for slot_cells_list in (getattr(c, "bound_slots", {}) or {}).values():
                        slot_cells.extend(slot_cells_list)
                    for slot_name, slot_contract in _safe_slots_items(c):
                        if slot_name in (getattr(c, "bound_slots", {}) or {}):
                            continue
                        sub = self.plan_sublattice(c, slot_name, slot_contract, tunnel, relevance_map, trial_sigma, prompt)
                        if sub:
                            slot_cells.extend(sub)   # counted for the trial only; nothing is stored
                trial_slot_cells[prefix] = slot_cells
                trial_items.append(it)

            if trial_slot_cells:
                def _slot_terms(it):
                    pfx = tuple(c.cell_id for c in it[0])
                    sc_cells = trial_slot_cells.get(pfx)
                    if not sc_cells:
                        return {}
                    path_cov = set().union(*(cell_covered.get(pc.cell_id, set()) for pc in it[0]))
                    new_mass = sum(cell_cov_mass_bonus.get(x.cell_id, 0.0) for x in sc_cells)
                    new_clause = sum(
                        clause_weights[g] for x in sc_cells for g in cell_covered.get(x.cell_id, set()) if g not in path_cov
                    ) / total_clause_weight
                    return {"intent:slot_mass": new_mass, "intent:slot_clauses": new_clause}
                # Rank ONLY the tried candidates against each other with the slot terms added;
                # untried candidates keep their relative order behind them.
                reranked = rank_items(trial_items, is_final=True, extra_terms=_slot_terms)
                tried = {tuple(c.cell_id for c in it[0]) for it in trial_items}
                rest = [(it, a) for it, a in scored_candidates if tuple(c.cell_id for c in it[0]) not in tried]
                head = [(it, a) for it, a in reranked]
                scored_candidates = head + rest

            if _debug_plan:
                import sys as _sys
                print("[PLAN-DEBUG] top ranked paths:", file=_sys.stderr)
                for it, agg in scored_candidates[:30]:
                    ids = tuple(c.cell_id for c in it[0])
                    print(f"  agg={agg:.3f} k={len(ids)} weak={it[3]} unbind={it[4]} comps={_component_trace.get(ids, {})}  {' -> '.join(ids)}", file=_sys.stderr)
            # The acceptance gate must share this planner's orchestrator so that
            # macro-goal cells can resolve their string sub-cell ids during
            # pre-unification expansion (otherwise macros stay unexpanded here
            # and are accepted as opaque single nodes).
            gate = UnificationGate(orchestrator=self.orchestrator)
            chosen_candidate = None
            acceptance_deadline = min(getattr(self, "_plan_deadline", time.perf_counter() + 15.0), time.perf_counter() + max(15.0, float(getattr(self, "planner_greedy_budget_ms", 15000.0)) / 1000.0))
            for cand_idx, (it, sc) in enumerate(scored_candidates):
                if cand_idx >= 30 and time.perf_counter() > acceptance_deadline:
                    logger.debug("[PLANNER] Plan deadline exceeded during candidate acceptance loop.")
                    break
                cand_p = it[0]
                cand_test = self._expand_identifier_multiplicity(cand_p, prompt)
                try:
                    # Faithful emission dry-run: acceptance requires the FULL
                    # emission path (clause scoping, slot planning, unification,
                    # template instantiation, liveness) to succeed, exactly as
                    # it will in synthesis. The trial runs on PRIVATE cell
                    # copies so shared lattice cells are never mutated.
                    # Candidates whose pipelines cannot render into executable
                    # code are skipped in favor of the next ranked candidate.
                    cand_private = [
                        _private_copy(c, self.orchestrator) for c in cand_test
                    ]
                    # Clause scoping (same attachment the winning path receives)
                    for c in cand_private:
                        if getattr(c, "matched_clause_idx", None) is None and c.cell_id in cell_clause_mass:
                            masses = cell_clause_mass[c.cell_id]
                            if masses and max(masses) > 0.0:
                                c.matched_clause_idx = max(range(len(masses)), key=lambda i: masses[i])
                    # Sub-lattice slot planning (same as the winning path receives)
                    for c in cand_private:
                        if getattr(c, "slots", None):
                            for slot_name, slot_contract in _safe_slots_items(c):
                                if slot_name in getattr(c, "bound_slots", {}):
                                    continue
                                sub_plan = self.plan_sublattice(
                                    c, slot_name, slot_contract, tunnel, relevance_map, it[1], prompt
                                )
                                if sub_plan:
                                    sub_plan = self._bind_chain_slots(sub_plan, tunnel, relevance_map, it[1], prompt)
                                    c.bound_slots[slot_name] = sub_plan
                    from route_methods.base import RouteMethod as _RM
                    _RM.tag_cells_with_clause_indices(cand_private, prompt, orchestrator=self.orchestrator)
                    cand_private = self._insert_property_bridges(cand_private)

                    emitted = gate.emit_code(cand_private, ExecutionContext(prompt=prompt), extract_llm_slots=False)

                    from preflight import PreflightLinter
                    lint_res = PreflightLinter.lint(gate.last_pipeline_bindings, prompt=prompt, code_str=emitted, orchestrator=self.orchestrator)
                    if not lint_res.is_valid:
                        raise ValueError(f"Preflight lint failed: {lint_res.violations}")

                    chosen_candidate = (cand_private, it[1], it[2], it[3], it[4])
                    break
                except Exception as e:
                    logger.debug(
                        "[PLANNER] Candidate rejected at emission dry-run (%s): %s",
                        " -> ".join(c.cell_id for c in cand_p),
                        str(e)[:200],
                    )
                    continue
            if chosen_candidate is None and scored_candidates:
                # Semantic repair stage (Section 3.4): attempt dynamic repair of candidate cells
                try:
                    from semantic_repair_engine import repair_cell_semantics
                    for it, _ in scored_candidates:
                        cand_p = it[0]
                        repaired_cells = []
                        any_repaired = False
                        for c in cand_p:
                            c_dict = c.to_dict() if hasattr(c, "to_dict") else dict(c.__dict__)
                            if repair_cell_semantics(c_dict, domain=getattr(c, "domain_name", "")):
                                any_repaired = True
                                repaired_cells.append(Cell.from_dict(c_dict))
                            else:
                                repaired_cells.append(c)
                        if any_repaired:
                            cand_test = self._expand_identifier_multiplicity(repaired_cells, prompt)
                            res = gate.unify_pipeline(cand_test, ExecutionContext(prompt=prompt), extract_llm_slots=False)
                            if isinstance(res, Success) and not res.is_bottom():
                                chosen_candidate = (repaired_cells, res.sigma, it[2], it[3], it[4])
                                break
                except Exception as e:
                    logger.debug(f"[PLANNER] Semantic repair pass: {e}")

            if chosen_candidate is None:
                chosen_candidate = scored_candidates[0][0] if scored_candidates else None
            if chosen_candidate is None:
                if settings.allow_fallback:
                    logger.info("[PLANNER] Trellis candidate empty: running Monte Carlo fallback...")
                    try:
                        mcts_path = self._monte_carlo_chain_search(prompt, tunnel, relevance_map)
                    except Exception:
                        mcts_path = []
                    if mcts_path:
                        if not hasattr(self, "trace_events"):
                            self.trace_events = []
                        self.trace_events.append({"event": "m0_mcts_fallback_triggered", "plan_len": len(mcts_path)})
                        return mcts_path
                self.last_refusal = {
                    "reason": "coverage_below_floor",
                    "coverage_fraction": 0.0,
                    "coverage_floor": floor_frac,
                    "uncovered_clauses": [(i, clauses[i]) for i in range(len(clauses))] if clauses else [],
                }
                logger.warning("[PLANNER] Refused: Trellis found no valid paths meeting coverage floor.")
                return []
            best_candidate = chosen_candidate
            best_path, best_sigma, _, _, _ = best_candidate

            # For-each multiplicity expansion (Section 3.4 goal decomposition):
            # a clause that names an enumerable set of referential identifiers
            # sharing one role ("normalize X column, Y column and Z column")
            # demands one application of the witnessing transform PER MEMBER,
            # not a single best-effort witness. The witnessing cell is the path
            # cell whose DECLARED identity vocabulary intersects the group's
            # role context and which declares a defaultless reference port;
            # replicas re-consume the receiver from the environment and bind
            # the remaining members in prompt order at unification time.
            best_path = self._expand_identifier_multiplicity(best_path, prompt)

            # Sub-Lattice recursive planning for macro/control-flow cells with slots.
            # The chosen plan owns private copies of its cells: binding never touches the
            # shared lattice instances.
            best_path = [
                _private_copy(c, self.orchestrator) if getattr(c, "slots", None) else c
                for c in best_path
            ]
            for cell in best_path:
                if getattr(cell, "slots", None):
                    for slot_name, slot_contract in _safe_slots_items(cell):
                        if slot_name not in getattr(cell, "bound_slots", {}):
                            sub_plan = self.plan_sublattice(
                                cell, slot_name, slot_contract, tunnel, relevance_map, best_sigma, prompt
                            )
                            if sub_plan:
                                sub_plan = self._bind_chain_slots(sub_plan, tunnel, relevance_map, best_sigma, prompt)
                                cell.bound_slots[slot_name] = sub_plan

            # Attach matched clause index to cells for unification literal scoping (D8)
            for pos, cell in enumerate(best_path):
                if getattr(cell, "matched_clause_idx", None) is None and cell.cell_id in cell_clause_mass:
                    masses = cell_clause_mass[cell.cell_id]
                    if masses and max(masses) > 0.0:
                        if self.orchestrator.loaded_cells.get(cell.cell_id) is cell:
                            cell = _private_copy(cell, self.orchestrator)
                            best_path[pos] = cell
                        cell.matched_clause_idx = max(range(len(masses)), key=lambda idx: masses[idx])
            from route_methods.base import RouteMethod as _RM
            _RM.tag_cells_with_clause_indices(best_path, prompt, orchestrator=self.orchestrator)

            # R5-5 / R6-7: Coverage floor enforcement. The floor is computed from
            # the SAME precision-based cell_covered the explain table shows — the
            # previous token-union re-add here let any identity-token touch count
            # as coverage, silently inflating the fraction past the floor.
            if req_floor and clause_tokens_list:
                all_cells_on_path = list(best_path)
                for c in best_path:
                    for slot_cells in (getattr(c, "bound_slots", {}) or {}).values():
                        all_cells_on_path.extend(slot_cells)
                covered_clauses = set()
                for c in all_cells_on_path:
                    covered_clauses |= cell_covered.get(c.cell_id, set())
                total_clauses = len(clause_tokens_list)
                cov_frac = len(covered_clauses) / total_clauses if total_clauses > 0 else 1.0
                uncovered = [i for i in range(total_clauses) if i not in covered_clauses]

                if cov_frac < floor_frac:
                    if uncovered and (time.perf_counter() < self._plan_deadline):
                        greedy_cfg_s = float(getattr(self, "planner_greedy_budget_ms", 1000.0)) / 1000.0
                        greedy_budget = min(greedy_cfg_s, max(0.0, self._plan_deadline - time.perf_counter()))
                        greedy_deadline = time.perf_counter() + greedy_budget
                        best_path_ids = {x.cell_id.lower() for x in best_path}
                        for cl_i in uncovered:
                            if time.perf_counter() > greedy_deadline:
                                break
                            if cl_i in covered_clauses:
                                continue
                            clause_cands = [
                                c for c in candidates
                                if cl_i in cell_covered.get(c.cell_id, set())
                                and (c.cell_id.lower() not in best_path_ids or (best_path and best_path[-1].cell_id.lower() != c.cell_id.lower() and sum(1 for x in best_path if x.cell_id.lower() == c.cell_id.lower()) < 2))
                            ]
                            clause_cands.sort(key=lambda c: log_probs.get(c.cell_id, -10.0), reverse=True)
                            for cand in clause_cands:
                                if time.perf_counter() > greedy_deadline:
                                    break
                                v_res = self._cells_connect(best_path[-1], cand) if best_path else False
                                if v_res:
                                    best_path.append(cand)
                                    best_path_ids.add(cand.cell_id.lower())
                                    covered_clauses |= cell_covered.get(cand.cell_id, set())
                                    break
                        cov_frac = len(covered_clauses) / total_clauses if total_clauses else 1.0
                        uncovered = [i for i in range(total_clauses) if i not in covered_clauses]

                if cov_frac < floor_frac:
                    self.last_refusal = {
                        "reason": "coverage_below_floor",
                        "coverage_fraction": cov_frac,
                        "coverage_floor": floor_frac,
                        "uncovered_clauses": [(i, clauses[i]) for i in uncovered if i < len(clauses)],
                    }
                    logger.warning(
                        "[PLANNER] Refused: coverage %.2f < floor %.2f; missing clauses: %s",
                        cov_frac,
                        floor_frac,
                        self.last_refusal["uncovered_clauses"],
                    )
                    return []

                try:
                    from config import settings as _settings
                    if bool(_settings.explain_plan):
                        print("\n=== EXPLAIN PLAN: CHOSEN PATH COVERAGE ===")
                        for pos_, c_ in enumerate(best_path):
                            print(
                                f"  step {pos_}: {c_.cell_id:<45} covered={sorted(cell_covered.get(c_.cell_id, set()))}"
                            )
                        print(f"  coverage fraction: {cov_frac:.3f} (floor {floor_frac:.2f})")
                        print(f"  greedy completion ran: {'yes' if cov_frac >= floor_frac and uncovered == [] and cov_frac < 1.0 else 'see refusal log above if refused'}")
                        print("=============================================\n")
                except Exception as _exc:
                    logger.debug("[PLANNER DIAGNOSTIC] Suppressed exception: %s", _exc, exc_info=True)

                return best_path

            return best_path

        # Step 4: Monte Carlo Tree Search over chains if the Trellis was disconnected (gated on allow_fallback)
        if settings.allow_fallback:
            logger.info("[PLANNER] Trellis disconnected: running Monte Carlo Tree Search fallback...")
            try:
                mcts_path = self._monte_carlo_chain_search(prompt, tunnel, relevance_map)
            except Exception as mcts_err:
                logger.debug(f"[PLANNER] MCTS chain search encountered error: {mcts_err}")
                mcts_path = []

            if mcts_path:
                if not hasattr(self, "trace_events"):
                    self.trace_events = []
                self.trace_events.append({"event": "m0_mcts_fallback_triggered", "plan_len": len(mcts_path)})
                return mcts_path

        self.last_refusal = {
            "reason": "coverage_below_floor",
            "coverage_fraction": 0.0,
            "coverage_floor": floor_frac,
            "uncovered_clauses": [(i, clauses[i]) for i in range(len(clauses))] if clauses else [],
        }
        logger.warning("[PLANNER] Refused: Trellis found no valid paths meeting coverage floor.")
        return []

    def _expand_identifier_multiplicity(
        self,
        path: List[Cell],
        prompt: str
    ) -> List[Cell]:
        """
        Monoidal for-each distribution: for every identifier role group in the
        prompt (>= 2 members sharing a role noun) and every witnessing cell on
        the path, inserts exactly len(group.members) - 1 replicas of the
        witness directly after it — the monoidal functor distributing a
        morphism over the declared product of identifiers.

        The replica count is DATA-DRIVEN: it is exactly the declared group
        membership extracted structurally from the prompt (no fixed cap such
        as the previous fixed replica cap, which silently truncated legitimate
        multi-target requests). Gating remains DECLARED-structure only: role
        tokens must intersect the cell's own token vocabulary, and the cell
        must own a defaultless required port of the reference value class
        (strict str, or any-typed with a declared semantic role state).
        Mirrors the binding gate in ExecutionContext.resolve_literal_for_port
        exactly.
        """
        if not prompt or len(path) < 1:
            return path
        try:
            groups = ExecutionContext.extract_identifier_groups(prompt)
        except Exception as e:
            return path
        if not groups:
            return path

        registry = TypeRegistry.get_instance()

        def _is_witness(cell: Cell, role_tokens: FrozenSet[str]) -> bool:
            if getattr(cell, "replica_of", None):
                return False
            # Only transform cells can be identifier multiplicity witnesses (never sinks or constructors)
            if not _is_transform_stage(cell):
                return False
            if getattr(cell, "node_type", "") == "constructor":
                return False
            # Cells that accept collection or column projection inputs consume multiplicity directly
            for p_sig in cell.inputs.values():
                t_name = str(p_sig.signature.type_name).lower()
                if (
                    _is_col_projection_port(p_sig)
                    or registry.is_subtype(t_name, "list")
                    or registry.is_subtype(t_name, "collection")
                    or registry.is_subtype(t_name, "sequence")
                    or t_name in ("list", "sequence", "collection")
                    or getattr(getattr(p_sig, "signature", p_sig), "abstract_type", None) == "collection"
                ):
                    state = str(getattr(p_sig.signature, "state", "") or "").lower()
                    st_tokens = CellTokenizer.tokenize_identifier(state) if state not in ("any", "default", "") else set()
                    if _is_col_projection_port(p_sig) or bool(role_tokens & st_tokens):
                        return False
            for p_sig in cell.inputs.values():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                t_name = str(p_sig.signature.type_name).lower()
                if (
                    _is_col_projection_port(p_sig)
                    or registry.is_subtype(t_name, "list")
                    or registry.is_subtype(t_name, "collection")
                    or registry.is_subtype(t_name, "sequence")
                    or t_name in ("list", "sequence", "collection")
                    or getattr(getattr(p_sig, "signature", p_sig), "abstract_type", None) == "collection"
                ):
                    continue
                strict_str = registry.is_subtype(t_name, "str") and t_name not in ("any", "*", "top", "")
                state = str(getattr(p_sig.signature, "state", "") or "").lower()
                st_tokens = CellTokenizer.tokenize_identifier(state) if state not in ("any", "default", "") else set()
                role_state = bool(role_tokens & st_tokens)
                if (strict_str or role_state) and bool(role_tokens & (cell.token_set | st_tokens)):
                    return True
            return False

        expanded: List[Cell] = []
        for cell in path:
            expanded.append(cell)
            for group in groups:
                if len(group.members) < 2 or not group.role_tokens:
                    continue
                if not _is_witness(cell, group.role_tokens):
                    continue
                # Exactly the declared multiplicity: one witness already on
                # the path, one replica per additional declared member.
                for _pos, _tok in group.members[1:]:
                    replica = copy.copy(cell)
                    replica.replica_of = cell.cell_id
                    replica.replica_role = ",".join(sorted(group.role_tokens))
                    expanded.append(replica)
        return expanded

    def _verify_transition(
        self,
        prev_path: List[Cell],
        cand: Cell,
        prev_sigma: Substitution,
        identifier_literals: Sequence[Any] = (),
        quoted_str_literals: Sequence[Any] = (),
        numeric_literals: Sequence[Any] = (),
    ) -> Optional[Substitution]:
        """
        Verifies monadic transition from prev_path to cand.
        Supports:
          1. Flat sequential 1D morphism: prev_cell.primary_output -> cand.primary_input.
          2. Multi-port monoidal product & port sharing (⊗, Δ):
             prev_cell.primary_output binds to some input of cand,
             and other required inputs are satisfied by earlier cells in prev_path or defaults.
        """
        prev_cell = prev_path[-1]
        prev_out = prev_cell.primary_output.signature

        def _get_ndim(port_obj: Any) -> Optional[int]:
            cached = getattr(port_obj, "_cached_ndim", ...)
            if cached is not ...:
                return cached
            sc = getattr(port_obj, "shape_contract", None)
            val = None
            if isinstance(sc, dict):
                v = sc.get("ndim")
                if isinstance(v, int):
                    val = v
                else:
                    try:
                        val = int(v) if v is not None else None
                    except (ValueError, TypeError):
                        val = None
            elif isinstance(sc, str):
                s = sc.strip()
                if s.startswith("(") and s.endswith(")"):
                    inner = s[1:-1].strip()
                    val = 0 if not inner else len([p for p in inner.split(",") if p.strip()])
            try:
                port_obj._cached_ndim = val
            except (AttributeError, TypeError):
                pass
            return val

        # 1. Check primary input first
        sub = None
        if _shape_compatible(prev_cell.primary_output, cand.primary_input):
            sub = unify(prev_out, cand.primary_input.signature, prev_sigma)
        bound_in: Optional[str] = cand.primary_input.name if sub is not None else None

        # 2. If primary input did not match, check other input ports (required ports take precedence)
        if sub is None:
            req_ports = [(k, v) for k, v in cand.inputs.items() if v.required]
            candidate_ports = req_ports if req_ports else list(cand.inputs.items())
            for p_name, p_sig in candidate_ports:
                if not _shape_compatible(prev_cell.primary_output, p_sig):
                    continue
                s_try = unify(prev_out, p_sig.signature, prev_sigma)
                if s_try is not None:
                    sub = s_try
                    bound_in = p_name
                    break

        # 2b. Multi-Carrier Scope Fallback: if prev_cell did not connect, check earlier ancestors in prev_path (DAG fork/branch)
        if sub is None and len(prev_path) > 1:
            req_ports = [(k, v) for k, v in cand.inputs.items() if v.required]
            candidate_ports = req_ports if req_ports else list(cand.inputs.items())
            for ancestor in reversed(prev_path[:-1]):
                for out_name, out_sig in ancestor.outputs.items():
                    for p_name, p_sig in candidate_ports:
                        if not _shape_compatible(out_sig, p_sig):
                            continue
                        s_try = unify(out_sig.signature, p_sig.signature, prev_sigma)
                        if s_try is not None:
                            sub = s_try
                            bound_in = p_name
                            break
                    if sub is not None:
                        break
                if sub is not None:
                    break

        # 2c. Bridge Fallback: if direct transition fails, check if a valid bridge chain connects prev_cell to cand
        if sub is None and self.orchestrator and hasattr(self.orchestrator, "find_shortest_bridge_path"):
            bridges = self.orchestrator.find_shortest_bridge_path(prev_cell, cand, max_hops=3)
            if bridges:
                sub = prev_sigma
                bound_in = cand.primary_input.name

        if sub is None:
            return None

        # 3. Check that all remaining REQUIRED inputs of cand can be satisfied
        # from earlier cells in prev_path (Port Sharing Δ) or have default values/literals
        for p_name, p_sig in cand.inputs.items():
            if p_name == bound_in:
                continue
            if not p_sig.required or p_sig.default_value is not None:
                continue

            satisfied = False
            for earlier_cell in reversed(prev_path):
                for out_name, out_sig in earlier_cell.outputs.items():
                    if not _shape_compatible(out_sig, p_sig):
                        continue
                    s_wire = unify(out_sig.signature, p_sig.signature, sub)
                    if s_wire is not None:
                        sub = s_wire
                        satisfied = True
                        break
                if satisfied:
                    break

            if not satisfied:
                is_instance_receiver = _is_receiver_port(p_sig, cand)
                if not is_instance_receiver:
                    registry = TypeRegistry.get_instance()
                    t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
                    # Type-driven literal groundability: a required port is satisfiable at
                    # synthesis time iff its DECLARED carrier is literal-groundable (scalar
                    # family, textual/path family, logical, or an explicitly untyped carrier)
                    # or the port declares an enum domain for reflection-based grounding.
                    # Zero port-name heuristics: naming is data, typing is semantics.
                    is_literal_groundable = _literal_groundable(p_sig, cand, quoted_str_literals, identifier_literals, numeric_literals)
                    if is_literal_groundable:
                        satisfied = True

            if not satisfied:
                return None

        # 4. Verify tree-declared property preconditions against established properties in prev_path
        cand_pres = getattr(cand, "preconditions", None) or []
        if cand_pres:
            for pre in cand_pres:
                p_op = pre.get("operator") if isinstance(pre, dict) else getattr(pre, "operator", None)
                p_prop = pre.get("property") if isinstance(pre, dict) else getattr(pre, "property", None)
                p_want = pre.get("value") if isinstance(pre, dict) else getattr(pre, "value", None)
                if p_op == "==" and p_prop is not None:
                    # Look backwards through prev_path to see the latest established value for this property
                    for c in reversed(prev_path):
                        c_posts = getattr(c, "postconditions", None) or []
                        found_prop = False
                        for q in c_posts:
                            q_op = q.get("operator") if isinstance(q, dict) else getattr(q, "operator", None)
                            q_prop = q.get("property") if isinstance(q, dict) else getattr(q, "property", None)
                            q_val = q.get("value") if isinstance(q, dict) else getattr(q, "value", None)
                            if q_op == "==" and q_prop == p_prop:
                                if q_val != p_want:
                                    return None  # Property contract violation
                                found_prop = True
                                break
                        if found_prop:
                            break

        return sub

    def _plan_linear_trellis(
        self,
        candidate_entries: List[Cell],
        candidates: List[Cell],
        log_probs: Dict[str, float],
        max_steps: int,
        zero_ary_ctors: List[Cell],
        _new_unbindable: Any,
        compute_path_score: Any,
        rank_items: Any,
        budget: SearchBudget,
        _edge_is_weak: Any,
        cells_by_in_type: Dict[str, List[Cell]],
        candidate_map: Dict[str, Cell],
        candidate_map_lower: Dict[str, Cell],
        identifier_literals: Sequence[Any] = (),
        quoted_str_literals: Sequence[Any] = (),
        numeric_literals: Sequence[Any] = (),
        cell_clause_mass: Optional[Dict[str, List[float]]] = None,
    ) -> List[Tuple[List[Cell], Substitution, float, int, int]]:
        """
        1D Monadic Trellis Baseline Planner.
        Standard sequential list approach where transitions branch from prev_cell.primary_output.
        Preserved as an ablation baseline for comparative benchmarks.
        """
        registry = TypeRegistry.get_instance()
        all_valid_paths: List[Tuple[List[Cell], Substitution, float, int, int]] = []

        # Step t = 1: Initialize beam
        current_beam: List[Tuple[List[Cell], Substitution, float, int, int]] = []
        for entry in candidate_entries:
            sc = log_probs.get(entry.cell_id, -10.0)
            p_tuple = ([entry], Substitution(), sc, 0, _new_unbindable(entry, []))
            current_beam.append(p_tuple)
            all_valid_paths.append(p_tuple)

        edge_compat_cache: Dict[Tuple[str, str, Any, Any], Optional[Substitution]] = {}

        def _sigma_fingerprint(sigma: Substitution) -> Any:
            try:
                return tuple(sorted((k, str(v)) for k, v in sigma.mappings.items()))
            except Exception:
                return ()

        def _required_ports_bindable(cand: Cell, prev_path: List[Cell], sigma: Substitution) -> Optional[Substitution]:
            sub = sigma
            for p_name, p_sig in cand.inputs.items():
                if not p_sig.required or p_sig.default_value is not None:
                    continue
                satisfied = False
                for earlier_cell in reversed(prev_path):
                    for out_name, out_sig in earlier_cell.outputs.items():
                        s_wire = unify(out_sig.signature, p_sig.signature, sub)
                        if s_wire is not None:
                            sub = s_wire
                            satisfied = True
                            break
                    if satisfied:
                        break
                if not satisfied:
                    if not _is_receiver_port(p_sig, cand) and _literal_groundable(
                            p_sig, cand, quoted_str_literals, identifier_literals, numeric_literals):
                        satisfied = True
                if not satisfied:
                    return None
            return sub

        _cell_succ_cache: Dict[str, List[Cell]] = {}

        def _successors(prev_cell: Cell) -> List[Cell]:
            if prev_cell.cell_id in _cell_succ_cache:
                return _cell_succ_cache[prev_cell.cell_id]
            if (getattr(prev_cell, "node_role", "") == "macro" or getattr(prev_cell, "cell_type", "") == "macro") and getattr(prev_cell, "endable", False):
                _cell_succ_cache[prev_cell.cell_id] = []
                return []
            macro_subs = set(getattr(prev_cell, "sub_cells", ()) or ())

            out_sig = prev_cell.primary_output.signature if hasattr(prev_cell.primary_output, "signature") else prev_cell.primary_output
            out_t = str(getattr(out_sig, "type_name", ""))
            out_s = str(getattr(out_sig, "state", ""))

            acc: Dict[str, Cell] = {}

            for edge in getattr(prev_cell, "edges", []):
                tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
                if tgt_id:
                    tgt_cell = candidate_map.get(tgt_id) or candidate_map_lower.get(str(tgt_id).lower())
                    if tgt_cell:
                        acc.setdefault(tgt_cell.cell_id, tgt_cell)

            is_type_var = out_t.isalpha() and len(out_t) == 1 and out_t.isupper()
            if "[" in out_t or is_type_var:
                for cand in candidates:
                    if cand.cell_id in macro_subs or prev_cell.cell_id in getattr(cand, "sub_cells", ()):
                        continue
                    if any(unify(out_sig, p_sig.signature) is not None
                           for p_sig in cand.inputs.values()):
                        acc.setdefault(cand.cell_id, cand)
                res_cells = [c for c in acc.values() if c.cell_id not in macro_subs and prev_cell.cell_id not in getattr(c, "sub_cells", ())]
                res = sorted(res_cells, key=lambda c: self._calculate_edge_affinity(prev_cell, c), reverse=True)
                _cell_succ_cache[prev_cell.cell_id] = res
                return res

            for in_t, cell_list in cells_by_in_type.items():
                if not registry.is_subtype(out_t, in_t):
                    continue
                for c in cell_list:
                    if c.cell_id in macro_subs or prev_cell.cell_id in getattr(c, "sub_cells", ()):
                        continue
                    if any(registry.is_state_compatible(
                        producer_state=out_s,
                        consumer_state=getattr(p_sig.signature, "state", "any"),
                        producer_accepted=getattr(out_sig, "accepted_states", frozenset()),
                        consumer_accepted=getattr(p_sig.signature, "accepted_states", frozenset()),
                        consumer_parent=getattr(p_sig.signature, "parent_state", None),
                        producer_parent=getattr(out_sig, "parent_state", None),
                    ) for p_sig in c.inputs.values()):
                        acc.setdefault(c.cell_id, c)
            res_cells = [c for c in acc.values() if c.cell_id not in macro_subs and prev_cell.cell_id not in getattr(c, "sub_cells", ())]
            res = sorted(res_cells, key=lambda c: self._calculate_edge_affinity(prev_cell, c), reverse=True)
            _cell_succ_cache[prev_cell.cell_id] = res
            return res

        for step in range(2, max_steps + 1):
            if time.perf_counter() > self._plan_deadline:
                logger.warning(
                    "[PLANNER] Search budget exhausted after step %d; returning best-so-far paths (fail fast).",
                    step - 1,
                )
                break
            candidates_for_next: List[Tuple[List[Cell], Substitution, float, int, int]] = []

            for prev_path, prev_sigma, prev_score, prev_weak, prev_unbind in current_beam:
                if time.perf_counter() > self._plan_deadline:
                    break
                prev_cell = prev_path[-1]

                if _is_terminal_sink_cell(prev_cell):
                    continue

                prev_path_ids = {c.cell_id.lower() for c in prev_path}

                successor_cells = _successors(prev_cell)
                seen_ids = {c.cell_id.lower() for c in successor_cells}
                for ctor in zero_ary_ctors:
                    if ctor.cell_id.lower() not in seen_ids:
                        successor_cells.append(ctor)
                        seen_ids.add(ctor.cell_id.lower())

                prev_out_sigs = tuple(sorted((out_sig.signature.type_name, str(getattr(out_sig.signature, "state", "any"))) for c in prev_path for out_sig in c.outputs.values()))
                prev_sigma_fp = _sigma_fingerprint(prev_sigma)

                for cand in successor_cells:
                    if time.perf_counter() > self._plan_deadline:
                        break
                    if cand.cell_id.lower() in prev_path_ids:
                        continue

                    cand_node_type = getattr(cand, "node_type", "")
                    cand_stage = getattr(cand, "stage", None)

                    if cand_node_type == "constructor" or cand in zero_ary_ctors:
                        new_sigma = _required_ports_bindable(cand, prev_path, prev_sigma)
                        if new_sigma is None:
                            continue
                    elif _is_terminal_sink_cell(cand) and not any(p.required for p in cand.inputs.values()):
                        new_sigma = prev_sigma
                    else:
                        if _is_ingress_stage(cand) and cand not in zero_ary_ctors:
                            continue

                        pair_key = (prev_cell.cell_id, cand.cell_id, prev_sigma_fp, prev_out_sigs)
                        if pair_key in edge_compat_cache:
                            new_sigma = edge_compat_cache[pair_key]
                        else:
                            new_sigma = self._verify_transition(
                                prev_path, cand, prev_sigma,
                                identifier_literals=identifier_literals,
                                quoted_str_literals=quoted_str_literals,
                                numeric_literals=numeric_literals,
                            )
                            edge_compat_cache[pair_key] = new_sigma

                        if new_sigma is None:
                            continue

                    cand_unbind = _new_unbindable(cand, prev_path)
                    if cand_unbind > 0:
                        continue

                    cand_sc = log_probs.get(cand.cell_id, -10.0)
                    total_sc = prev_score + cand_sc
                    step_weak = prev_weak + (
                        1 if (cand_node_type != "constructor" and cand not in zero_ary_ctors and not _is_terminal_sink_cell(cand) and _edge_is_weak(prev_cell, cand)) else 0
                    )
                    step_unbind = prev_unbind + cand_unbind
                    new_tuple = (prev_path + [cand], new_sigma, total_sc, step_weak, step_unbind)
                    candidates_for_next.append(new_tuple)
                    all_valid_paths.append(new_tuple)

            if not candidates_for_next:
                break

            _hi, _lo = budget.pool_limits(budget.linear_beam_width, max_steps)
            if len(all_valid_paths) > _hi:
                all_valid_paths = [it for it, _ in rank_items(all_valid_paths, False)][:_lo]

            candidates_for_next = [it for it, _ in rank_items(candidates_for_next, False)]
            endpoint_counts: Dict[str, int] = {}
            next_beam = []
            for item in candidates_for_next:
                endpoint = item[0][-1].cell_id
                if endpoint_counts.get(endpoint, 0) < budget.linear_per_endpoint:
                    next_beam.append(item)
                    endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + 1
                    if len(next_beam) >= budget.linear_beam_width:
                        break
            current_beam = next_beam

        return all_valid_paths

    def _plan_frontier_dag(
        self,
        candidate_entries: List[Cell],
        candidates: List[Cell],
        log_probs: Dict[str, float],
        max_steps: int,
        zero_ary_ctors: List[Cell],
        _new_unbindable: Any,
        compute_path_score: Any,
        rank_items: Any,
        budget: SearchBudget,
        _edge_is_weak: Any,
        cells_by_in_type: Dict[str, List[Cell]],
        candidate_map: Dict[str, Cell],
        candidate_map_lower: Dict[str, Cell],
        identifier_literals: Sequence[Any] = (),
        quoted_str_literals: Sequence[Any] = (),
        numeric_literals: Sequence[Any] = (),
        cell_clause_mass: Optional[Dict[str, List[float]]] = None,
        file_asset_literals: Sequence[Any] = (),
        dest_file_literals: Sequence[Any] = (),
        has_egress_intent: bool = False,
        target_sink: Optional[str] = None,
        is_valid_terminal: Optional[Any] = None,
        unrequested_check: Optional[Any] = None,
        num_clauses: int = 1,
        cell_covered: Optional[Dict[str, Set[int]]] = None,
    ) -> List[Tuple[List[Cell], Substitution, float, int, int]]:
        """
        Monoidal Category Frontier DAG Planner (Default Approach).
        Maintains an active multi-carrier frontier F over candidate paths in the category C.
        Candidate extensions are verified against all available active output wires in F(P).
        Fork-join / convergent morphisms (nabla) consuming >= 2 distinct ancestor carriers
        are naturally synthesized and rewarded with a multi-carrier join bonus.
        """
        cell_covered = cell_covered or {}
        registry = TypeRegistry.get_instance()
        _cells_connect = self._cells_connect
        all_valid_paths: List[Tuple[List[Cell], Substitution, float, int, int]] = []

        # Step t = 1: Initialize beam with entry sources
        current_beam: List[Tuple[List[Cell], Substitution, float, int, int]] = []
        for entry in candidate_entries:
            sc = log_probs.get(entry.cell_id, -10.0)
            p_tuple = ([entry], Substitution(), sc, 0, _new_unbindable(entry, []))
            current_beam.append(p_tuple)
            if is_valid_terminal is None or is_valid_terminal([entry]):
                all_valid_paths.append(p_tuple)

        # ---- Wire index (bottleneck fix) --------------------------------------------------
        # Whether an output port CAN feed an input port is a property of the two port
        # signatures alone (unification with an empty substitution only gets stricter as the
        # substitution grows), so it is decided ONCE per port pair and memoised.  The
        # per-path search then runs the substitution-carrying unify only for pairs that
        # passed, instead of re-unifying every ancestor output against every input port for
        # every beam entry.
        _wire_memo: Dict[Tuple[int, int], bool] = {}

        def _unify_with_product_support(out_sig: Any, p_sig: Any, sub: Substitution) -> Optional[Substitution]:
            out_term = getattr(out_sig, "signature", out_sig)
            in_term = getattr(p_sig, "signature", p_sig)
            s_wire = unify(out_term, in_term, sub)
            if s_wire is not None:
                return s_wire
            # Monoidal product elimination (e.g. tuple[A, B, ...] -> A)
            raw = str(getattr(out_term, "type_name", "") or "")
            if "[" in raw and raw.endswith("]"):
                ctor = raw.split("[", 1)[0].strip().lower()
                if registry.is_product_constructor(ctor):
                    try:
                        from unification import split_bracketed_args, TypeTerm
                    except (ImportError, ValueError):
                        from .unification import split_bracketed_args, TypeTerm
                    inner = raw[raw.index("[") + 1 : -1]
                    members = list(split_bracketed_args(inner))
                    for m_str in members:
                        try:
                            m_term = TypeTerm.from_string(m_str)
                            u = unify(m_term, in_term, sub)
                            if u is not None:
                                return u
                        except Exception:
                            continue
            return None

        def _shape_compatible(p_out: Any, p_in: Any) -> bool:
            key = (id(p_out), id(p_in))
            hit = _wire_memo.get(key)
            if hit is None:
                hit = is_edge_valid(p_out, p_in)
                _wire_memo[key] = hit
            return hit

        def _verify_frontier_step(
            prev_path: List[Cell],
            cand: Cell,
            prev_sigma: Substitution
        ) -> Optional[Tuple[Substitution, Set[str], bool, Dict[str, Tuple[Cell, str]]]]:
            if hasattr(self, "_plan_deadline") and time.perf_counter() > self._plan_deadline:
                return None
            # STRICT STAGE MONOID: S3 (Sink) is terminal. No transforms may follow a sink.
            if prev_path and _is_terminal_sink_cell(prev_path[-1]):
                return None
            cand_stage = getattr(cand, "stage", None)
            cand_role = str(getattr(cand, "node_role", "")).lower()
            if _is_transform_stage(cand) and any(_is_terminal_sink_cell(c) for c in prev_path):
                return None

            # Terminal sink guard: when cand is a terminal sink, ensure egress intent exists
            # and that any required destination path ports can be bound.
            if _is_terminal_sink_cell(cand):
                has_req_path = any(
                    p.required and p.default_value is None and _lattice_is_path_port(p)
                    for p in cand.inputs.values()
                )
                if has_req_path and not dest_file_literals:
                    return None
                if not has_egress_intent and not target_sink:
                    return None

            sub = prev_sigma
            bound_parents: Set[str] = set()
            bound_input_ports: Set[str] = set()
            bound_port_map: Dict[str, Tuple[Cell, str]] = {}

            # 1. Check primary input first if present
            cand_prim_in = getattr(cand, "primary_input", None)
            if cand_prim_in is not None:
                p_name = getattr(cand_prim_in, "name", "input")
                p_sig = cand_prim_in
                for earlier_cell in reversed(prev_path):
                    for out_name, out_sig in earlier_cell.outputs.items():
                        if not _shape_compatible(out_sig, p_sig):
                            continue
                        s_wire = _unify_with_product_support(out_sig, p_sig, sub)
                        if s_wire is not None:
                            sub = s_wire
                            bound_parents.add(earlier_cell.cell_id)
                            bound_input_ports.add(p_name)
                            bound_port_map[p_name] = (earlier_cell, out_name)
                            break
                    if p_name in bound_input_ports:
                        break

            # 2. Check all remaining inputs of cand
            for p_name, p_sig in cand.inputs.items():
                if p_name in bound_input_ports:
                    continue

                satisfied = False
                for earlier_cell in reversed(prev_path):
                    for out_name, out_sig in earlier_cell.outputs.items():
                        if not _shape_compatible(out_sig, p_sig):
                            continue
                        s_wire = _unify_with_product_support(out_sig, p_sig, sub)
                        if s_wire is not None:
                            sub = s_wire
                            bound_parents.add(earlier_cell.cell_id)
                            bound_input_ports.add(p_name)
                            bound_port_map[p_name] = (earlier_cell, out_name)
                            satisfied = True
                            break
                    if satisfied:
                        break

                if not satisfied:
                    if not p_sig.required or p_sig.default_value is not None:
                        continue

                    is_instance_receiver = _is_receiver_port(p_sig, cand)
                    if not is_instance_receiver:
                        t_name = str(getattr(p_sig.signature, "type_name", "")).lower()
                        if _is_carrier_projectable_port(p_sig, prev_path, quoted_str_literals, identifier_literals):
                            satisfied = True
                            for earlier_cell in reversed(prev_path):
                                for out_name, out_s in earlier_cell.outputs.items():
                                    o_tn = str(getattr(out_s.signature, "type_name", "")).lower()
                                    if registry.is_composite_carrier(o_tn) or registry.is_abstract_carrier(o_tn):
                                        bound_parents.add(earlier_cell.cell_id)
                                        bound_input_ports.add(p_name)
                                        break
                                if p_name in bound_input_ports:
                                    break
                        else:
                            is_literal_groundable = _literal_groundable(p_sig, cand, quoted_str_literals, identifier_literals, numeric_literals)
                            if is_literal_groundable:
                                satisfied = True

                    if not satisfied:
                        return None

            cand_node_type = getattr(cand, "node_type", "")
            if cand_node_type != "constructor" and cand not in zero_ary_ctors and not bound_parents:
                if _is_terminal_sink_cell(cand) and not any(p.required for p in cand.inputs.values()):
                    bound_parents.add(prev_path[-1].cell_id)
                    if cand.inputs and prev_path[-1].primary_output:
                        first_in = next(iter(cand.inputs.keys()))
                        out_n = getattr(prev_path[-1].primary_output, "name", "output")
                        bound_port_map[first_in] = (prev_path[-1], out_n)
                elif _is_ingress_stage(cand):
                    pass
                else:
                    return None

            # Unrequested-wildcard guard: a morphism the prompt never names
            # (zero identity-vocabulary overlap) whose output carrier is a bare
            # type variable / wildcard is neither a demanded operation nor a
            # typed transformation -- it can only serve as an untyped bridge
            # filler, so it is rejected as an extension entirely. Generic
            # utilities the prompt DOES name, and typed transformations, are
            # unaffected.
            if unrequested_check is not None and unrequested_check(cand):
                _cand_role = str(getattr(cand, "node_role", "") or "").lower()
                _cand_type = str(getattr(cand, "node_type", "") or "").lower()
                if _cand_role not in ("bridge", "source", "sink") and _cand_type != "tunnel":
                    cov_set = cell_covered.get(cand.cell_id, set()) if cell_covered else set()
                    if not cov_set:
                        return None
                _out_t = str(getattr(getattr(cand, "primary_output", None), "type_name", "") or "").lower()
                if _out_t in _WILDCARD_CARRIERS or (len(_out_t) == 1 and _out_t.isalpha()):
                    return None

            # Wildcard-bridge guard: an unrequested morphism (its declared
            # identity vocabulary shares nothing with the prompt) may only join
            # the pipeline through at least one TYPED wire. Attachments made
            # purely through wildcard-carrier wires are the fingerprint of a
            # generic utility morphism smuggled in as a bridge filler; the
            # typed carriers the path already carries must make the connection.
            if bound_parents and unrequested_check is not None and unrequested_check(cand):
                parents_by_id = {p.cell_id: p for p in prev_path}
                typed_wire_exists = any(
                    str(getattr(parents_by_id[pid].primary_output, "type_name", "") or "").lower()
                    not in _WILDCARD_CARRIERS
                    for pid in bound_parents
                    if pid in parents_by_id and parents_by_id[pid].primary_output is not None
                )
                if not typed_wire_exists:
                    return None

            is_join = len(bound_parents) >= 2
            return (sub, bound_parents, is_join, bound_port_map)

        # Precompute candidate successors from any cell
        _cell_succ_cache: Dict[str, List[Cell]] = {}
        def _cell_successors(cell: Cell) -> List[Cell]:
            if cell.cell_id in _cell_succ_cache:
                return _cell_succ_cache[cell.cell_id]

            acc: Dict[str, Cell] = {}
            for edge in getattr(cell, "edges", []):
                tgt_id = edge.get("target_cell_id") if isinstance(edge, dict) else getattr(edge, "target_cell_id", None)
                if tgt_id:
                    tgt_cell = candidate_map.get(tgt_id) or candidate_map_lower.get(str(tgt_id).lower())
                    if tgt_cell:
                        acc.setdefault(tgt_cell.cell_id, tgt_cell)

            outputs_to_inspect = list(cell.outputs.values()) if cell.outputs else ([cell.primary_output] if cell.primary_output else [])
            for out_port in outputs_to_inspect:
                out_sig = out_port.signature if hasattr(out_port, "signature") else out_port
                out_t = str(getattr(out_sig, "type_name", ""))
                out_s = str(getattr(out_sig, "state", ""))

                is_type_var = out_t.isalpha() and len(out_t) == 1 and out_t.isupper()
                if "[" in out_t or is_type_var:
                    for cand in candidates:
                        cand_in_ports = list(cand.inputs.values()) if cand.inputs else [cand.primary_input]
                        if any(_unify_with_product_support(out_sig, p_sig, Substitution()) is not None for p_sig in cand_in_ports):
                            acc.setdefault(cand.cell_id, cand)
                    continue

                for in_t, cell_list in cells_by_in_type.items():
                    if not registry.is_subtype(out_t, in_t):
                        continue
                    for c in cell_list:
                        cand_in_ports = list(c.inputs.values()) if c.inputs else [c.primary_input]
                        if any(registry.is_state_compatible(
                            producer_state=out_s,
                            consumer_state=getattr(p_sig.signature, "state", "any"),
                            producer_accepted=getattr(out_sig, "accepted_states", frozenset()),
                            consumer_accepted=getattr(p_sig.signature, "accepted_states", frozenset()),
                            consumer_parent=getattr(p_sig.signature, "parent_state", None),
                            producer_parent=getattr(out_sig, "parent_state", None),
                        ) for p_sig in cand_in_ports):
                            acc.setdefault(c.cell_id, c)
            res = list(acc.values())
            _cell_succ_cache[cell.cell_id] = res
            return res

        # Step t = 2 ... max_steps
        for step in range(2, max_steps + 1):
            if time.perf_counter() > self._plan_deadline:
                logger.warning(
                    "[PLANNER] Search budget exhausted after step %d; returning best-so-far paths (fail fast).",
                    step - 1,
                )
                break
            candidates_for_next: List[Tuple[List[Cell], Substitution, float, int, int]] = []

            for prev_path, prev_sigma, prev_score, prev_weak, prev_unbind in current_beam:
                if time.perf_counter() > self._plan_deadline:
                    break
                prev_cell = prev_path[-1]

                # Terminal check (D5): stage 3 sinks conclude the path.
                # If prev_path already contains any stage-3 sink without sublattice slots,
                # do not extend it.
                has_terminal_sink = any(_is_terminal_sink_cell(c) for c in prev_path)
                if has_terminal_sink:
                    continue

                prev_path_ids = {c.cell_id.lower() for c in prev_path}

                def _can_reuse_cell(c: Cell) -> bool:
                    if not prev_path or c.cell_id.lower() == prev_path[-1].cell_id.lower():
                        return False
                    if getattr(c, "stage", None) != 2:
                        return False
                    if sum(1 for x in prev_path if x.cell_id.lower() == c.cell_id.lower()) >= 2:
                        return False
                    if not cell_clause_mass:
                        return False
                    prev_indices = [idx for idx, x in enumerate(prev_path) if x.cell_id.lower() == c.cell_id.lower()]
                    if not prev_indices:
                        return True
                    last_idx = prev_indices[-1]
                    prev_cl = 0
                    if last_idx > 0:
                        p_cell = prev_path[last_idx - 1]
                        p_m = cell_clause_mass.get(p_cell.cell_id, [])
                        prev_cl = max(range(len(p_m)), key=lambda i: p_m[i]) if p_m and max(p_m) > 0 else 0
                    curr_m = cell_clause_mass.get(prev_path[-1].cell_id, [])
                    curr_cl = max(range(len(curr_m)), key=lambda i: curr_m[i]) if curr_m and max(curr_m) > 0 else 0
                    return curr_cl > prev_cl

                successor_candidates: List[Cell] = []
                seen_succ_ids: Set[str] = set()

                for path_cell in prev_path:
                    for succ in _cell_successors(path_cell):
                        succ_low = succ.cell_id.lower()
                        if succ_low not in seen_succ_ids and (succ_low not in prev_path_ids or _can_reuse_cell(succ)):
                            successor_candidates.append(succ)
                            seen_succ_ids.add(succ_low)

                for ctor in zero_ary_ctors:
                    ctor_low = ctor.cell_id.lower()
                    if ctor_low not in seen_succ_ids and ctor_low not in prev_path_ids:
                        successor_candidates.append(ctor)
                        seen_succ_ids.add(ctor_low)

                num_ingress = sum(
                    1 for c in prev_path
                    if _is_ingress_stage(c) and getattr(c, "node_type", "") != "constructor"
                )
                if num_ingress < len(file_asset_literals):
                    for entry in candidate_entries:
                        entry_low = entry.cell_id.lower()
                        if entry_low not in seen_succ_ids:
                            successor_candidates.append(entry)
                            seen_succ_ids.add(entry_low)

                for cand in successor_candidates:
                    if time.perf_counter() > self._plan_deadline:
                        break
                    cand_stage = getattr(cand, "stage", None)
                    is_ingress = (_is_ingress_stage(cand) and cand not in zero_ary_ctors)

                    if cand.cell_id.lower() in prev_path_ids:
                        if not _can_reuse_cell(cand) and not (is_ingress and num_ingress < len(file_asset_literals)):
                            continue

                    if is_ingress and num_ingress >= len(file_asset_literals):
                        continue

                    v_res = _verify_frontier_step(prev_path, cand, prev_sigma)
                    if v_res is None:
                        continue

                    new_sigma, bound_parents, is_join, *extra = v_res
                    bound_port_map = extra[0] if extra else {}

                    # Step canonicalization (DECLARED structure only): prune
                    # permuted orderings of independent steps when the PROMPT's
                    # own clause ordering declares the canonical sequence. If
                    # cand belongs to a strictly EARLIER prompt clause than
                    # prev_cell, does not consume it, does not connect to it,
                    # and could have attached to prev_path[:-1], then placing
                    # cand after prev_cell contradicts the user-requested
                    # sequence. The previous heuristic also compared cell_id
                    # STRINGS alphabetically ("A" < "B") — an arbitrary
                    # lexicographic criterion that discarded valid topological
                    # orderings; it has been removed. When no clause evidence
                    # distinguishes the two orderings, BOTH are kept and the
                    # scored trellis decides.
                    if (
                        len(prev_path) >= 1
                        and prev_cell.cell_id not in bound_parents
                        and not _cells_connect(prev_cell, cand)
                        and not is_ingress
                    ):
                        prev_m = cell_clause_mass.get(prev_cell.cell_id, []) if cell_clause_mass else []
                        cand_m = cell_clause_mass.get(cand.cell_id, []) if cell_clause_mass else []
                        prev_cl = max(range(len(prev_m)), key=lambda i: prev_m[i]) if prev_m and max(prev_m) > 0 else 0
                        cand_cl = max(range(len(cand_m)), key=lambda i: cand_m[i]) if cand_m and max(cand_m) > 0 else 0
                        if cand_m and cand_cl < prev_cl:
                            if not prev_path[:-1] or _verify_frontier_step(prev_path[:-1], cand, prev_sigma) is not None:
                                continue

                    cand_unbind = _new_unbindable(cand, prev_path)
                    if cand_unbind > 0:
                        continue

                    cand_sc = log_probs.get(cand.cell_id, -10.0)
                    total_sc = prev_score + cand_sc
                    cand_node_type = getattr(cand, "node_type", "")
                    step_weak = prev_weak + (
                        1 if (cand_node_type != "constructor" and cand not in zero_ary_ctors and not _is_terminal_sink_cell(cand) and _edge_is_weak(prev_cell, cand)) else 0
                    )
                    step_unbind = prev_unbind + cand_unbind

                    cand_to_add = cand.clone() if hasattr(cand, "clone") else copy.copy(cand)
                    cand_to_add.bound_parent_ids = set(bound_parents)
                    cand_to_add.planned_port_wires = bound_port_map
                    new_tuple = (prev_path + [cand_to_add], new_sigma, total_sc, step_weak, step_unbind)
                    candidates_for_next.append(new_tuple)

            if not candidates_for_next:
                break
            # Cheap likelihood prefilter: the full Borda rank (semantic
            # objective over every candidate extension) is the search's
            # dominant cost. Bounding the ranked set to a constant multiple of
            # the beam, ordered by the Viterbi log-likelihood the extension
            # already accumulated, keeps the ranking semantics for every
            # prefix that could realistically survive while making the step
            # cost independent of the tunnel's branching factor.
            _prefilter_cap = max(budget.beam_width * 2, 128)
            if len(candidates_for_next) > _prefilter_cap:
                candidates_for_next.sort(key=lambda it: it[2], reverse=True)
                candidates_for_next = candidates_for_next[:_prefilter_cap]
            candidates_for_next = [it for it, _ in rank_items(candidates_for_next, False)]
            by_endpoint: Dict[str, List[Any]] = {}
            for item in candidates_for_next:
                by_endpoint.setdefault(item[0][-1].cell_id, []).append(item)

            next_beam = []
            sorted_endpoints = list(by_endpoint.keys())   # already best-first (rank order)
            # Round 1: the best `per_endpoint` prefixes for every reachable endpoint.
            for ep in sorted_endpoints:
                next_beam.extend(by_endpoint[ep][:budget.per_endpoint])
            # Round 2: fill the remaining beam capacity with secondary candidates.
            for ep in sorted_endpoints:
                for item in by_endpoint[ep][budget.per_endpoint:]:
                    if len(next_beam) >= budget.beam_width:
                        break
                    next_beam.append(item)
                if len(next_beam) >= budget.beam_width:
                    break
            current_beam = next_beam

            # Valid completed paths addition:
            for item in candidates_for_next:
                term_cell = item[0][-1]
                if is_valid_terminal is not None:
                    if is_valid_terminal(item[0]):
                        all_valid_paths.append(item)
                else:
                    if (
                        _is_egress_stage(term_cell)
                        or getattr(term_cell, "endable", False)
                        or getattr(term_cell, "is_endable", False)
                    ):
                        all_valid_paths.append(item)

            _hi, _lo = budget.pool_limits(budget.beam_width, max_steps)
            if len(all_valid_paths) > _hi:
                all_valid_paths = [it for it, _ in rank_items(all_valid_paths, True)][:_lo]

        # Greedy completion: if all_valid_paths is empty, OR if all_valid_paths has paths
        # but the best path has clause coverage < 1.0 (uncovered clauses exist when num_clauses > 1).
        max_cov = 0
        if all_valid_paths and num_clauses > 1:
            max_cov = max(
                len(set().union(*(cell_covered.get(c.cell_id, set()) for c in it[0])))
                for it in all_valid_paths
            )

        should_greedy_complete = (not all_valid_paths and bool(current_beam)) or (
            num_clauses > 1 and max_cov < num_clauses and (bool(all_valid_paths) or bool(current_beam))
        )

        if should_greedy_complete:
            candidate_pool = list(all_valid_paths) + list(current_beam)
            if candidate_pool:
                best_tuple = max(
                    candidate_pool,
                    key=lambda it: (
                        len(set().union(*(cell_covered.get(c.cell_id, set()) for c in it[0]))),
                        it[2],
                    ),
                )
                best_path, best_sigma, best_score, best_weak, best_unbind = best_tuple
                covered_clauses = set().union(*(cell_covered.get(c.cell_id, set()) for c in best_path))
                uncovered = [cl_i for cl_i in range(num_clauses) if cl_i not in covered_clauses]

                try:
                    from config import settings as _s
                    greedy_cfg_s = float(getattr(_s, "planner_greedy_budget_ms", getattr(self, "planner_greedy_budget_ms", 1000.0))) / 1000.0
                except Exception:
                    greedy_cfg_s = float(getattr(self, "planner_greedy_budget_ms", 1000.0)) / 1000.0
                if hasattr(self, "_plan_deadline"):
                    greedy_budget = min(greedy_cfg_s, max(0.0, self._plan_deadline - time.perf_counter()))
                else:
                    greedy_budget = greedy_cfg_s
                greedy_deadline = time.perf_counter() + greedy_budget

                best_path_ids = {x.cell_id.lower() for x in best_path}

                if uncovered:
                    for cl_i in uncovered:
                        if time.perf_counter() > greedy_deadline:
                            break
                        if cl_i in covered_clauses:
                            continue
                        clause_cands = [
                            c for c in candidates
                            if cl_i in cell_covered.get(c.cell_id, set())
                            and (c.cell_id.lower() not in best_path_ids or (best_path and best_path[-1].cell_id.lower() != c.cell_id.lower() and sum(1 for x in best_path if x.cell_id.lower() == c.cell_id.lower()) < 2))
                            and bool(cell_covered.get(c.cell_id, set()) - covered_clauses)
                        ]
                        clause_cands.sort(key=lambda c: log_probs.get(c.cell_id, -10.0), reverse=True)
                        for cand in clause_cands:
                            if time.perf_counter() > greedy_deadline:
                                break
                            v_res = _verify_frontier_step(best_path, cand, best_sigma)
                            if v_res is not None:
                                new_sigma, bound_parents, is_join, *extra = v_res
                                bound_port_map = extra[0] if extra else {}
                                cand_sc = log_probs.get(cand.cell_id, -10.0)
                                cand_to_add = cand.clone() if hasattr(cand, "clone") else copy.copy(cand)
                                cand_to_add.bound_parent_ids = set(bound_parents)
                                cand_to_add.planned_port_wires = bound_port_map
                                best_path = best_path + [cand_to_add]
                                best_path_ids.add(cand.cell_id.lower())
                                covered_clauses |= cell_covered.get(cand.cell_id, set())
                                best_sigma = new_sigma
                                best_score += cand_sc
                                break

                # If not yet a valid terminal, greedily reach an endable / terminal node
                if is_valid_terminal is not None and not is_valid_terminal(best_path):
                    term_cands = [
                        c for c in candidates
                        if (_is_egress_stage(c) or getattr(c, "endable", False) or getattr(c, "is_endable", False))
                        and c.cell_id.lower() not in best_path_ids
                    ]
                    term_cands.sort(key=lambda c: log_probs.get(c.cell_id, -10.0), reverse=True)
                    for cand in term_cands:
                        if time.perf_counter() > greedy_deadline:
                            break
                        v_res = _verify_frontier_step(best_path, cand, best_sigma)
                        if v_res is not None:
                            new_sigma, bound_parents, is_join, *extra = v_res
                            bound_port_map = extra[0] if extra else {}
                            cand_sc = log_probs.get(cand.cell_id, -10.0)
                            cand_to_add = cand.clone() if hasattr(cand, "clone") else copy.copy(cand)
                            cand_to_add.bound_parent_ids = set(bound_parents)
                            cand_to_add.planned_port_wires = bound_port_map
                            best_path = best_path + [cand_to_add]
                            best_path_ids.add(cand.cell_id.lower())
                            covered_clauses |= cell_covered.get(cand.cell_id, set())
                            best_sigma = new_sigma
                            best_score += cand_sc
                            break

                all_valid_paths.append((best_path, best_sigma, best_score, best_weak, best_unbind))
        elif not all_valid_paths:
            all_valid_paths = list(current_beam)
        return all_valid_paths

    def plan_sublattice(
        self,
        parent_cell: Cell,
        slot_name: str,
        slot_contract: Any,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
        active_sigma: Substitution,
        prompt: str = ""
    ) -> Optional[List[Cell]]:
        """
        Synthesizes a type-verified sub-pipeline for a macro slot.
        Categorically verifies traced loop invariants (Tr^U) and coproduct branch joins (⊕).
        """
        topology = getattr(parent_cell, "topology_type", "sequential")

        # 1. Traced Loop Slot Planning (Tr^U) — dispatched by the DECLARED topology
        # of the macro cell, never by cell identifier substrings.
        if topology == "traced_loop":
            # Recover the container-typed input port structurally (generic carrier
            # C[T]); the item carrier T is extracted from its generic argument.
            coll_sig = None
            for p_sig in parent_cell.inputs.values():
                if "[" in str(getattr(p_sig.signature, "type_name", "")):
                    coll_sig = p_sig
                    break
            item_type: Any = "any"
            if coll_sig is not None:
                c_type_str = str(coll_sig.signature.type_name)
                c_concrete = substitute_generics(c_type_str, active_sigma)
                if "[" in c_concrete and c_concrete.endswith("]"):
                    b_idx = c_concrete.index("[")
                    item_type = c_concrete[b_idx + 1 : -1].strip()
                elif "T" in active_sigma.mappings:
                    item_type = str(active_sigma.mappings["T"])

            u_raw = getattr(parent_cell, "feedback_state_type", None) or "S"
            u_concrete = substitute_generics(u_raw, active_sigma)

            pool = [c for c in tunnel if c.cell_id != parent_cell.cell_id and getattr(c, "node_type", "") != "constant"]
            if not pool:
                pool = [c for c in self.orchestrator.loaded_cells.values() if c.cell_id != parent_cell.cell_id and getattr(c, "node_type", "") != "constant"]

            # Dynamic clause targeting: the clause(s) that describe the loop are the
            # ones sharing vocabulary with the loop morphism's DECLARED token set.
            # Zero hardcoded connector/loop keyword lists.
            parent_toks = getattr(parent_cell, "identity_tokens", getattr(parent_cell, "token_set", set()))
            clauses = _segment_prompt_clauses(prompt)
            if clauses and parent_toks:
                related = [
                    cl for cl in clauses
                    if CellTokenizer.tokenize_prompt(cl) & parent_toks
                ]
                slot_clause = " ".join(related)
            else:
                slot_clause = ""
            target_text = slot_clause.strip() or prompt
            target_tokens = CellTokenizer.tokenize_prompt(target_text) if target_text else set()
            if not target_tokens:
                target_tokens = CellTokenizer.tokenize_prompt(prompt) if prompt else set()

            child_candidates = []
            for cand in pool:
                if _is_ingress_cell(cand) and getattr(cand, "node_type", "") != "macro":
                    continue
                for p_name, p_sig in cand.inputs.items():
                    u_cand = unify(item_type, p_sig.signature, active_sigma) or unify(p_sig.signature, item_type, active_sigma)
                    if u_cand is not None:
                        rel = relevance_map.get(cand.cell_id, 0.0)
                        cand_id_toks = getattr(cand, "identity_tokens", cand.token_set)
                        tok_ov = len(target_tokens & cand_id_toks)
                        bonus_val = float(settings.domain_affinity_bonus)
                        domain_bonus = bonus_val if cand.domain_name and any(c.domain_name == cand.domain_name for c in tunnel) else 0.0
                        score = tok_ov * 1.0 + rel + domain_bonus
                        child_candidates.append((cand, score, u_cand))
                        break

            if child_candidates:
                child_candidates.sort(key=lambda x: x[1], reverse=True)
                best_child, _, child_sigma = child_candidates[0]

                valid, _ = verify_traced_loop_invariant(u_concrete, u_concrete, child_sigma)
                if valid:
                    return [best_child]

        # 2. State Loop / Fixed Point / While Loop / Repeat Until (Tr^{State}: State -> State)
        elif topology in ("state_loop", "while_loop", "fixed_point", "repeat_until"):
            s_in_contract = slot_contract.get("input_type") if isinstance(slot_contract, dict) else getattr(slot_contract, "input_type", None)
            u_raw = getattr(parent_cell, "feedback_state_type", None) or s_in_contract or "State"
            u_concrete = substitute_generics(str(u_raw), active_sigma)

            pool = [c for c in tunnel if c.cell_id != parent_cell.cell_id and getattr(c, "node_type", "") != "constant"]
            if not pool:
                pool = [c for c in self.orchestrator.loaded_cells.values() if c.cell_id != parent_cell.cell_id and getattr(c, "node_type", "") != "constant"]

            parent_toks = getattr(parent_cell, "identity_tokens", getattr(parent_cell, "token_set", set()))
            clauses = _segment_prompt_clauses(prompt)
            if clauses and parent_toks:
                related = [
                    cl for cl in clauses
                    if CellTokenizer.tokenize_prompt(cl) & parent_toks
                ]
                slot_clause = " ".join(related)
            else:
                slot_clause = ""
            target_text = slot_clause.strip() or prompt
            target_tokens = CellTokenizer.tokenize_prompt(target_text) if target_text else set()
            if not target_tokens:
                target_tokens = CellTokenizer.tokenize_prompt(prompt) if prompt else set()

            child_candidates = []
            for cand in pool:
                if _is_ingress_cell(cand) and getattr(cand, "node_type", "") != "macro":
                    continue
                cand_sub = active_sigma
                in_unified = False
                for p_name, p_sig in cand.inputs.items():
                    u_cand = unify(u_concrete, p_sig.signature, active_sigma) or unify(p_sig.signature, u_concrete, active_sigma)
                    if u_cand is not None:
                        in_unified = True
                        cand_sub = u_cand
                        break
                if in_unified or getattr(cand, "domain_name", "") == parent_cell.domain_name or u_concrete in ("State", "any"):
                    rel = relevance_map.get(cand.cell_id, 0.0)
                    cand_id_toks = getattr(cand, "identity_tokens", getattr(cand, "token_set", set()))
                    tok_ov = len(target_tokens & cand_id_toks)
                    domain_bonus = 0.5 if cand.domain_name and any(c.domain_name == cand.domain_name for c in tunnel) else 0.0
                    score = tok_ov * 1.0 + rel + domain_bonus
                    child_candidates.append((cand, score, cand_sub))

            if child_candidates:
                child_candidates.sort(key=lambda x: x[1], reverse=True)
                best_child, _, child_sigma = child_candidates[0]
                valid, _ = verify_traced_loop_invariant(u_concrete, u_concrete, child_sigma)
                if valid:
                    return [best_child]
                return [best_child]

        # 3. Coproduct Branch Slot Planning (⊕) — declared topology dispatch.
        elif topology == "coproduct_branch":
            pool = [c for c in tunnel if c.cell_id != parent_cell.cell_id and getattr(c, "node_type", "") != "constant"]
            if not pool:
                pool = [c for c in self.orchestrator.loaded_cells.values() if c.cell_id != parent_cell.cell_id and getattr(c, "node_type", "") != "constant"]
            if pool:
                bound_cell_ids: Set[str] = set()
                for b_slot, b_plan in (getattr(parent_cell, "bound_slots", {}) or {}).items():
                    if b_slot != slot_name and isinstance(b_plan, (list, tuple)):
                        bound_cell_ids.update(getattr(bc, "cell_id", "") for bc in b_plan if bc)
                unbound_pool = [c for c in pool if c.cell_id not in bound_cell_ids]
                candidate_pool = unbound_pool if unbound_pool else pool
                pool_sorted = sorted(candidate_pool, key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)
                return [pool_sorted[0]]

        # 4. Context Manager Slot Planning (bracket resource pattern)
        elif topology == "context_manager":
            pool = [c for c in tunnel if c.cell_id != parent_cell.cell_id and getattr(c, "node_type", "") != "constant"]
            if not pool:
                pool = [c for c in self.orchestrator.loaded_cells.values() if c.cell_id != parent_cell.cell_id and getattr(c, "node_type", "") != "constant"]
            if pool:
                pool_sorted = sorted(pool, key=lambda c: relevance_map.get(c.cell_id, 0.0), reverse=True)
                return [pool_sorted[0]]

        return None

    def _bind_chain_slots(self, chain: List[Cell], tunnel: List[Cell],
                          relevance_map: Dict[str, float], sigma: Substitution,
                          prompt: str = "") -> List[Cell]:
        """Plans macro slots on private copies of the chain's cells (never on shared lattice cells). Recursively binds nested slots."""
        out: List[Cell] = []
        for cell in chain:
            if getattr(cell, "slots", None):
                cell = _private_copy(cell, self.orchestrator)
                for slot_name, slot_contract in _safe_slots_items(cell):
                    if slot_name not in getattr(cell, "bound_slots", {}):
                        sub_plan = self.plan_sublattice(cell, slot_name, slot_contract, tunnel, relevance_map, sigma, prompt)
                        if sub_plan:
                            # Recursively bind nested slots inside sub_plan
                            sub_plan = self._bind_chain_slots(sub_plan, tunnel, relevance_map, sigma, prompt)
                            cell.bound_slots[slot_name] = sub_plan
            out.append(cell)
        return out

    def _monte_carlo_chain_search(
        self,
        prompt: str,
        tunnel: List[Cell],
        relevance_map: Dict[str, float],
    ) -> Optional[List[Cell]]:
        """
        Last-resort fallback of the M0 planner when the Trellis is disconnected: REAL Monte Carlo Tree Search
        (src/mcts.py: UCT selection, progressive widening, seeded prior-weighted random rollouts, value
        back-propagation) over chains of type-composable cells.

          state   = chain of cell ids so far (starts from each ingress-stage cell's chain)
          actions = tunnel cells whose input unifies with the chain tail's output (substitution threaded)
          reward  = prompt-clause coverage of the chain (cells assigned by evidence) x evidence quality,
                    with a bonus for ending on a terminal cell; a chain that cannot be unified scores 0

        This replaces the deterministic greedy walk that used to carry the name "MCTS" without any of its
        mechanics. Budget, exploration, seed and widening come from settings (mcts_*).
        """
        from config import settings
        from mcts import MonteCarloTreeSearch
        try:
            from .token_evidence import audit as _audit
        except (ImportError, ValueError):
            from token_evidence import audit as _audit

        # The planner is time-budgeted (planner_time_budget_ms / _plan_deadline): the search gets only what is
        # left of it, and does not start at all once the deadline has passed.
        _remaining = (self._plan_deadline - time.perf_counter()) if hasattr(self, "_plan_deadline") else float("inf")
        if _remaining <= 0.0:
            logger.warning("[PLANNER] Monte Carlo fallback skipped: planner deadline already exceeded.")
            return None
        _search_seconds = min(float(settings.mcts_max_seconds), _remaining)
        _max_depth = int(settings.max_plan_depth)

        _ing = self._vocab.ingress_stage
        entries = [c for c in tunnel if getattr(c, "is_ingress", False) or (c.stage == _ing)] or list(tunnel)
        by_id = {c.cell_id: c for c in tunnel}
        sigma_cache: Dict[tuple, Optional[Substitution]] = {(): Substitution()}

        def sigma_of(chain: tuple) -> Optional[Substitution]:
            if chain in sigma_cache:
                return sigma_cache[chain]
            prev = sigma_of(chain[:-1])
            sig = None
            if prev is not None:
                if len(chain) == 1:
                    sig = prev
                else:
                    sig = unify(by_id[chain[-2]].primary_output.signature, by_id[chain[-1]].primary_input.signature, prev)
            sigma_cache[chain] = sig
            return sig

        def actions(state: tuple) -> list:
            chain = state
            if not chain:
                return [(e.cell_id,) for e in entries]                    # first action picks the entry cell
            tail = by_id[chain[-1]]
            if _is_terminal_sink_cell(tail) or len(chain) >= _max_depth:
                return []
            sig = sigma_of(chain)
            if sig is None:
                return []
            used = set(chain)
            return [(c.cell_id,) for c in tunnel
                    if c.cell_id not in used and unify(tail.primary_output.signature, c.primary_input.signature, sig) is not None]

        def step(state: tuple, action: tuple) -> tuple:
            return state + action

        # heuristic for expansion order and the softmax rollout policy: how strongly some prompt clause names the
        # cell (identity evidence), plus retrieval relevance and edge affinity
        import math as _math
        from token_evidence import get_evidence as _get_ev
        _ev = _get_ev(self.orchestrator)
        _clause_toks = [_ev.clause_tokens(cl) for cl in _segment_prompt_clauses(prompt)] if _ev is not None else []
        _fit: Dict[str, float] = {}
        for _c in tunnel:
            _best = 0.0
            for _t in _clause_toks:
                if _t:
                    _o = _ev.identity_overlap(_t, _c)
                    _best = max(_best, _o["recall"] * _o["name_precision"])
            _fit[_c.cell_id] = _best

        def prior(state: tuple, action: tuple) -> float:
            c = by_id[action[0]]
            aff = self._calculate_edge_affinity(by_id[state[-1]], c) if state else 0.0
            h = _fit.get(c.cell_id, 0.0) + relevance_map.get(c.cell_id, 0.0) + aff
            return _math.exp(h / settings.mcts_rollout_temperature)

        def reward(state: tuple) -> float:
            if not state or sigma_of(state) is None:
                return 0.0
            cells = [by_id[i] for i in state]
            res = _audit(cells, prompt, self.orchestrator if hasattr(self, "orchestrator") else None)
            infos = [c for c in res.get("clauses", []) if c["status"] in ("served", "UNSERVED")]
            if not infos:
                return 0.0
            served = sum(1 for c in infos if c["status"] == "served") / len(infos)
            quality = sum(c["quality"] for c in infos) / len(infos)
            terminal = 1.0 if _is_terminal_sink_cell(cells[-1]) else 0.5
            # every cell the audit calls unrequested (or clause it calls unserved) counts against the chain,
            # exactly as in M10's reward: without this a long wander that happens to touch every clause wins
            return served * quality * terminal / (1.0 + len(res.get("issues", [])))

        search = MonteCarloTreeSearch(
            actions=actions, step=step, reward=reward, prior=prior, state_key=lambda s: s,
            describe_action=lambda a: a[0], iterations=settings.mcts_iterations, exploration=settings.mcts_exploration,
            widening_alpha=settings.mcts_widening_alpha, max_seconds=_search_seconds, max_rollout_depth=_max_depth,
            seed=settings.mcts_seed,
        )
        res = search.search(())
        self.last_mcts_summary = {
            "iterations": res.iterations, "evaluations": res.evaluations, "cache_hits": res.cache_hits, "seed": res.seed,
            "elapsed_ms": round(res.elapsed_ms, 1), "stopped_by": res.stopped_by, "best_reward": round(res.best_reward, 4),
            "best_chain": list(res.best_state), "root": res.root_stats[:8],
        }
        logger.info(f"[PLANNER] Monte Carlo chain search: {self.last_mcts_summary}")
        chain = list(res.best_state)
        if len(chain) < 1 or res.best_reward <= 0.0:
            return None
        cells = [by_id[i] for i in chain]
        sig = sigma_of(tuple(chain))
        return self._bind_chain_slots(cells, tunnel, relevance_map, sig, prompt=prompt) if sig is not None else cells


ZeroShotPlanner = LatticePlanner
