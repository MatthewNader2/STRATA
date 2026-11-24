"""
src/type_system.py — Pure Algebraic Type, Shape, Typestate, and Refinement System for STRATA.

Implements the formal 4-component graded port type T = <carrier, layout, (typestate, mode), refinement>:
  - `CarrierType` & `parse_carrier`: parametric type AST supporting variance on containers,
    tuples, Optionals, Unions, and contravariant/covariant Callables.
  - `BoundedPoset`: acyclic multi-parent DAG partial order over carriers and domain-namespaced
    typestates with BFS subtyping, cycle validation, LUB (`join_*`), and GLB (`meet_*`).
  - `RefinementSet`: decidable real-interval and side-predicate lattice with entailment,
    lattice join, and literal refinement checking.
  - `unify_shapes` / `join_shapes`: symbolic and concrete dimensional shape unification.

Zero external dependencies, zero regex, zero domain/library hardcodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

try:
    from .schema import TOP_CARRIERS, UNIVERSAL_BOTTOM_SYMBOLS, normalize_typestate
except (ImportError, ValueError):
    from schema import TOP_CARRIERS, UNIVERSAL_BOTTOM_SYMBOLS, normalize_typestate


_CANONICAL_DISPLAY_HEADS: Dict[str, str] = {
    "list": "List",
    "tuple": "Tuple",
    "dict": "Dict",
    "set": "Set",
    "optional": "Optional",
    "union": "Union",
    "callable": "Callable",
    "iterable": "Iterable",
    "iterator": "Iterator",
    "sequence": "Sequence",
    "file": "File",
    "contextmanager": "ContextManager",
    "type": "Type",
}

_VARIANCE_CONTAINER_HEADS: Set[str] = {
    "list",
    "tuple",
    "set",
    "sequence",
    "iterable",
    "iterator",
    "collection",
    "file",
    "contextmanager",
    "type",
    "dict",
    "mapping",
}

_SEQUENCE_LIKE_HEADS: Set[str] = {"sequence", "iterable", "collection"}


def _split_top_level_commas(text: str) -> List[str]:
    """Split `text` on commas at bracket depth 0."""
    parts: List[str] = []
    cur: List[str] = []
    depth = 0
    for ch in text:
        if ch in ("[", "("):
            depth += 1
        elif ch in ("]", ")") and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            item = "".join(cur).strip()
            if item:
                parts.append(item)
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _find_matching_bracket(text: str, start_idx: int) -> int:
    depth = 0
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return idx
    return -1


@dataclass(frozen=True)
class CarrierType:
    """Structured parametric type expression AST."""

    head: str
    args: Tuple["CarrierType", ...] = ()
    is_var: bool = False
    is_ellipsis: bool = False
    is_arg_list: bool = False


def parse_carrier(raw: Optional[str]) -> CarrierType:
    """Parse a carrier type string into a normalized `CarrierType` AST without regex."""
    s = str(raw or "").strip()
    if not s:
        return CarrierType(head="any")
    if s == "...":
        return CarrierType(head="...", is_ellipsis=True)

    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return CarrierType(head="__arglist__", args=(), is_arg_list=True)
        items = _split_top_level_commas(inner)
        return CarrierType(
            head="__arglist__",
            args=tuple(parse_carrier(it) for it in items),
            is_arg_list=True,
        )

    br_idx = s.find("[")
    if br_idx != -1:
        close_idx = _find_matching_bracket(s, br_idx)
        if close_idx == len(s) - 1 and close_idx > br_idx:
            head_raw = s[:br_idx].strip()
            inner = s[br_idx + 1 : close_idx].strip()
            head_low = head_raw.lower() or "any"
            if not inner:
                return CarrierType(head=head_low)
            arg_strs = _split_top_level_commas(inner)
            return CarrierType(
                head=head_low,
                args=tuple(parse_carrier(a) for a in arg_strs),
            )

    low = s.lower()
    is_var = (len(s) == 1 and s.isalpha() and s.isupper()) or (s == "State")
    return CarrierType(head=low, is_var=is_var)


def format_carrier(ct: CarrierType) -> str:
    """Render a `CarrierType` AST back into canonical display notation."""
    if ct.is_ellipsis:
        return "..."
    if ct.is_arg_list:
        return "[" + ", ".join(format_carrier(a) for a in ct.args) + "]"
    head_str = _CANONICAL_DISPLAY_HEADS.get(ct.head, ct.head.upper() if ct.is_var and len(ct.head) == 1 else ct.head)
    if not ct.args:
        return head_str
    return f"{head_str}[{', '.join(format_carrier(a) for a in ct.args)}]"


def _normalize_parent_graph(raw: Optional[Dict[str, Any]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for k, v in (raw or {}).items():
        k_l = str(k or "").strip().lower()
        if not k_l:
            continue
        parents: List[str]
        if isinstance(v, (list, tuple, set)):
            parents = [str(p).strip().lower() for p in v if str(p or "").strip()]
        else:
            p_s = str(v or "").strip().lower()
            parents = [p_s] if p_s else []
        for p_l in parents:
            if p_l and p_l != k_l:
                out.setdefault(k_l, set()).add(p_l)
    return out


def _detect_graph_cycles(graph: Dict[str, Set[str]], label: str) -> List[str]:
    """3-color DFS detecting any directed cycle in a hierarchy graph."""
    state: Dict[str, int] = {}  # 0/absent=unvisited, 1=visiting, 2=done
    cycles: List[str] = []

    def _dfs(node: str) -> None:
        state[node] = 1
        for nxt in sorted(graph.get(node, ())):
            st = state.get(nxt, 0)
            if st == 1:
                cycles.append(f"Cycle in {label} hierarchy: '{node}' -> '{nxt}'")
            elif st == 0 and nxt in graph:
                _dfs(nxt)
        state[node] = 2

    for start in sorted(graph.keys()):
        if state.get(start, 0) == 0:
            _dfs(start)
    return cycles


class BoundedPoset:
    """
    Bounded Multi-Parent DAG Partial Order over parametric carriers and domain typestates.

    Supports multiple inheritance, structural parametric subtyping, cycle validation,
    and lattice join (LUB) / meet (GLB) operations.
    """

    def __init__(
        self,
        type_parents: Optional[Dict[str, Union[str, Iterable[str]]]] = None,
        state_parents: Optional[Dict[str, Union[str, Iterable[str]]]] = None,
        top_types: Optional[Iterable[str]] = None,
    ) -> None:
        self.type_parents: Dict[str, Set[str]] = _normalize_parent_graph(type_parents)
        self.state_parents: Dict[str, Set[str]] = {}
        for k, v in (state_parents or {}).items():
            k_n = normalize_typestate(str(k or ""))
            if k_n == "any":
                continue
            vals = v if isinstance(v, (list, tuple, set)) else (v,)
            for pv in vals:
                p_n = normalize_typestate(str(pv or ""))
                if p_n != "any" and p_n != k_n:
                    self.state_parents.setdefault(k_n, set()).add(p_n)

        self.top_types: Set[str] = set(TOP_CARRIERS)
        for tt in (top_types or ()):
            tl = str(tt or "").strip().lower()
            if tl:
                self.top_types.add(tl)

        # Reverse type index built lazily for GLB meet queries
        self._type_children: Optional[Dict[str, Set[str]]] = None

    def validate_acyclic(self) -> List[str]:
        """Verify both type and typestate hierarchies form valid DAGs without cycles."""
        return (
            _detect_graph_cycles(self.type_parents, "type")
            + _detect_graph_cycles(self.state_parents, "typestate")
        )

    def type_ancestors(self, head_or_type: str) -> Dict[str, int]:
        """Return `{ancestor_head: shortest_hop_distance}` via BFS over `type_parents`."""
        start = str(head_or_type or "").strip().lower()
        if not start:
            return {}
        dists: Dict[str, int] = {start: 0}
        queue: List[Tuple[str, int]] = [(start, 0)]
        idx = 0
        while idx < len(queue):
            curr, d = queue[idx]
            idx += 1
            for par in self.type_parents.get(curr, ()):
                if par not in dists:
                    dists[par] = d + 1
                    queue.append((par, d + 1))
                if "::" not in par:
                    continue
                bare = par.split("::", 1)[1]
                if bare and bare not in dists:
                    dists[bare] = d + 1
                    queue.append((bare, d + 1))
        return dists

    def _type_descendants(self, head: str) -> Dict[str, int]:
        if self._type_children is None:
            children: Dict[str, Set[str]] = {}
            for child, pars in self.type_parents.items():
                for par in pars:
                    children.setdefault(par, set()).add(child)
            self._type_children = children
        start = str(head or "").strip().lower()
        dists: Dict[str, int] = {start: 0}
        queue: List[Tuple[str, int]] = [(start, 0)]
        idx = 0
        while idx < len(queue):
            curr, d = queue[idx]
            idx += 1
            for ch in self._type_children.get(curr, ()):
                if ch not in dists:
                    dists[ch] = d + 1
                    queue.append((ch, d + 1))
        return dists

    def _is_head_subtype(self, sub_head: str, sup_head: str) -> bool:
        s_l = str(sub_head or "").strip().lower()
        p_l = str(sup_head or "").strip().lower()
        if p_l in self.top_types:
            return True
        if not s_l:
            return False
        if s_l in UNIVERSAL_BOTTOM_SYMBOLS:
            return True
        if s_l == p_l:
            return True
        if s_l in self.top_types:
            return False

        ancs = self.type_ancestors(s_l)
        if p_l in ancs:
            return True
        if "::" not in p_l:
            return any("::" in anc and anc.split("::", 1)[1] == p_l for anc in ancs)
        return False

    def _is_carrier_subtype(self, s: CarrierType, p: CarrierType) -> bool:
        if p.head in self.top_types or p.is_var:
            return True
        if s.head in UNIVERSAL_BOTTOM_SYMBOLS:
            return True
        if s.is_var:
            return True
        if s.head in self.top_types:
            return False

        if p.head == "union" and p.args:
            return any(self._is_carrier_subtype(s, pa) for pa in p.args)
        if s.head == "union" and s.args:
            return all(self._is_carrier_subtype(sa, p) for sa in s.args)

        if p.head == "optional" and len(p.args) == 1:
            if s.head in ("none", "nonetype", "null", "nil"):
                return True
            if s.head == "optional" and len(s.args) == 1:
                return self._is_carrier_subtype(s.args[0], p.args[0])
            return self._is_carrier_subtype(s, p.args[0])
        if s.head == "optional" and len(s.args) == 1:
            return self._is_head_subtype("optional", p.head) and not p.args

        if p.head == "callable":
            if s.head != "callable":
                return self._is_head_subtype(s.head, "callable") if not p.args else False
            if not p.args:
                return True
            if not s.args:
                return len(p.args) == 2 and (p.args[0].is_ellipsis or (p.args[0].is_arg_list and all(a.is_var or a.head in self.top_types for a in p.args[0].args))) and (p.args[1].is_var or p.args[1].head in self.top_types)
            if len(s.args) == 2 and len(p.args) == 2:
                s_params, s_ret = s.args
                p_params, p_ret = p.args
                if not self._is_carrier_subtype(s_ret, p_ret):
                    return False
                if s_params.is_ellipsis or p_params.is_ellipsis:
                    return True
                if s_params.is_arg_list and p_params.is_arg_list and len(s_params.args) == len(p_params.args):
                    # Contravariant on parameter types
                    return all(
                        self._is_carrier_subtype(pa, sa)
                        for sa, pa in zip(s_params.args, p_params.args)
                    )
                return False
            return False

        if s.head == "callable" and p.head != "callable":
            return self._is_head_subtype("callable", p.head) and not p.args

        if p.head == "tuple":
            if not self._is_head_subtype(s.head, "tuple"):
                return False
            if not p.args:
                return True
            if not s.args:
                return all(a.is_ellipsis or a.is_var or a.head in self.top_types for a in p.args)
            if len(p.args) == 2 and p.args[1].is_ellipsis:
                if len(s.args) == 2 and s.args[1].is_ellipsis:
                    return self._is_carrier_subtype(s.args[0], p.args[0])
                return all(self._is_carrier_subtype(sa, p.args[0]) for sa in s.args)
            if len(s.args) == len(p.args):
                return all(self._is_carrier_subtype(sa, pa) for sa, pa in zip(s.args, p.args))
            return False

        if s.args or p.args:
            if not self._is_head_subtype(s.head, p.head):
                return False
            if not p.args:
                return True
            if not s.args:
                return all(a.is_ellipsis or a.is_var or a.head in self.top_types for a in p.args)
            if s.head == "file" and p.head == "file" and len(s.args) != len(p.args):
                return self._is_carrier_subtype(s.args[-1], p.args[-1])
            if p.head in _SEQUENCE_LIKE_HEADS and len(p.args) == 1:
                elem_sup = p.args[0]
                elem_subs = [sa for sa in s.args if not sa.is_ellipsis]
                return bool(elem_subs) and all(self._is_carrier_subtype(sa, elem_sup) for sa in elem_subs)
            if len(s.args) == len(p.args):
                return all(self._is_carrier_subtype(sa, pa) for sa, pa in zip(s.args, p.args))
            return False

        return self._is_head_subtype(s.head, p.head)

    def is_subtype(self, sub: str, sup: str) -> bool:
        """Decide `sub <= sup` over multi-parent carrier DAG and parametric variance."""
        return self._is_carrier_subtype(parse_carrier(sub), parse_carrier(sup))

    def join_carriers(self, a: str, b: str) -> str:
        """Compute least upper bound (LUB) `a \\sqcup b` in the bounded carrier poset."""
        a_s = str(a or "").strip() or "any"
        b_s = str(b or "").strip() or "any"
        if self.is_subtype(a_s, b_s):
            return b_s
        if self.is_subtype(b_s, a_s):
            return a_s

        ca = parse_carrier(a_s)
        cb = parse_carrier(b_s)
        if (
            ca.args
            and cb.args
            and ca.head == cb.head
            and ca.head != "callable"
            and len(ca.args) == len(cb.args)
        ):
            joined_args = tuple(
                parse_carrier(self.join_carriers(format_carrier(xa), format_carrier(xb)))
                for xa, xb in zip(ca.args, cb.args)
            )
            return format_carrier(CarrierType(head=ca.head, args=joined_args))

        ancs_a = self.type_ancestors(ca.head)
        ancs_b = self.type_ancestors(cb.head)
        common_non_top = [
            anc
            for anc in ancs_a
            if anc in ancs_b and anc not in self.top_types
        ]
        bare_non_top = [c for c in common_non_top if "::" not in c]
        pool = bare_non_top if bare_non_top else common_non_top
        if not pool:
            return "any"

        minimal = [
            c
            for c in pool
            if not any(other != c and self._is_head_subtype(other, c) for other in pool)
        ]
        candidates = minimal if minimal else pool
        best = min(candidates, key=lambda c: (ancs_a[c] + ancs_b[c], c))
        return best

    def meet_carriers(self, a: str, b: str) -> str:
        """Compute greatest lower bound (GLB) `a \\sqcap b`, falling back to `'void'`."""
        a_s = str(a or "").strip() or "any"
        b_s = str(b or "").strip() or "any"
        if self.is_subtype(a_s, b_s):
            return a_s
        if self.is_subtype(b_s, a_s):
            return b_s

        ca = parse_carrier(a_s)
        cb = parse_carrier(b_s)
        desc_a = self._type_descendants(ca.head)
        desc_b = self._type_descendants(cb.head)
        common = [d for d in desc_a if d in desc_b and d not in UNIVERSAL_BOTTOM_SYMBOLS]
        if not common:
            return "void"
        maximal = [
            c
            for c in common
            if not any(other != c and self._is_head_subtype(c, other) for other in common)
        ]
        candidates = maximal if maximal else common
        return min(candidates, key=lambda c: (desc_a[c] + desc_b[c], c))

    def state_ancestors(self, state: str) -> Dict[str, int]:
        """Return `{ancestor_state: shortest_hop_distance}` via BFS over `state_parents`."""
        start = normalize_typestate(state)
        if start == "any":
            return {"any": 0}
        dists: Dict[str, int] = {start: 0}
        queue: List[Tuple[str, int]] = [(start, 0)]
        idx = 0
        while idx < len(queue):
            curr, d = queue[idx]
            idx += 1
            for par in self.state_parents.get(curr, ()):
                if par not in dists:
                    dists[par] = d + 1
                    queue.append((par, d + 1))
        return dists

    @staticmethod
    def _split_ns(token: str) -> Tuple[str, str]:
        if "::" in token:
            d, n = token.split("::", 1)
            return d.strip().lower(), n.strip().lower()
        return "", token.strip().lower()

    def is_state_compatible(
        self,
        prod_state: Optional[str],
        cons_state: Optional[str],
        cons_accepted: Iterable[str] = (),
        prod_domain: str = "",
        cons_domain: str = "",
    ) -> bool:
        """Multi-parent typestate compatibility over DAG ancestors and disjunctive `accepted_states`."""
        p_st = normalize_typestate(prod_state, prod_domain)
        c_st = normalize_typestate(cons_state, cons_domain)

        targets: List[str] = []
        for s in (cons_accepted or ()):
            sn = normalize_typestate(s, cons_domain)
            if sn != "any":
                targets.append(sn)
        if c_st != "any" and c_st not in targets:
            targets.append(c_st)

        if not targets:
            return True
        if p_st == "any":
            return False

        ancs = self.state_ancestors(p_st)
        for anc in ancs:
            anc_dom, anc_name = self._split_ns(anc)
            for tgt in targets:
                if anc == tgt:
                    return True
                tgt_dom, tgt_name = self._split_ns(tgt)
                if (not anc_dom or not tgt_dom or anc_dom == tgt_dom) and anc_name == tgt_name:
                    return True
        return False

    def join_states(self, a: Optional[str], b: Optional[str], dom_a: str = "", dom_b: str = "") -> str:
        """Compute LUB typestate `sa \\sqcup sb`, returning `'any'` when states are incomparable."""
        sa = normalize_typestate(a, dom_a)
        sb = normalize_typestate(b, dom_b)
        if sa == "any" or sb == "any":
            return "any"
        if self.is_state_compatible(sa, sb):
            return sb
        if self.is_state_compatible(sb, sa):
            return sa

        ancs_a = self.state_ancestors(sa)
        ancs_b = self.state_ancestors(sb)
        common = [s for s in ancs_a if s in ancs_b and s != "any"]
        if not common:
            return "any"
        minimal = [
            c
            for c in common
            if not any(other != c and self.is_state_compatible(other, c) for other in common)
        ]
        candidates = minimal if minimal else common
        return min(candidates, key=lambda c: (ancs_a[c] + ancs_b[c], c))


def _parse_float_token(raw: str) -> Optional[float]:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def _split_kv(item: str) -> Optional[Tuple[str, str]]:
    for sep in ("=", ":"):
        if sep in item:
            k, v = item.split(sep, 1)
            return k.strip().lower(), v.strip()
    return None


@dataclass(frozen=True)
class RefinementSet:
    """
    Decidable numeric interval + boolean-property + predicate refinement lattice.

    Supports sound entailment (`min=1 |= min=0`, `min=1 |= gt=0`, `min=1 |= nonzero=True`),
    lattice join (interval hull), and concrete literal bound verification.
    """

    lo: Optional[float] = None
    lo_strict: bool = False
    hi: Optional[float] = None
    hi_strict: bool = False
    nonzero: bool = False
    non_empty: bool = False
    dtype: Optional[str] = None
    predicates: Tuple[str, ...] = ()

    @property
    def effective_nonzero(self) -> bool:
        if self.nonzero:
            return True
        if self.lo is not None and (self.lo > 0.0 or (self.lo == 0.0 and self.lo_strict)):
            return True
        if self.hi is not None and (self.hi < 0.0 or (self.hi == 0.0 and self.hi_strict)):
            return True
        return False

    @classmethod
    def from_invariants(cls, raw_invs: Optional[Iterable[str]]) -> "RefinementSet":
        lo: Optional[float] = None
        lo_strict = False
        hi: Optional[float] = None
        hi_strict = False
        nonzero = False
        non_empty = False
        dtype: Optional[str] = None
        preds: Set[str] = set()

        for raw in (raw_invs or ()):
            s = str(raw or "").strip()
            if not s:
                continue
            low = s.lower()
            if low == "non_empty":
                non_empty = True
                continue
            if low == "nonzero":
                nonzero = True
                continue

            kv = _split_kv(low)
            if kv is not None:
                k, v = kv
                if k in ("min", "gte"):
                    num = _parse_float_token(v)
                    if num is not None:
                        if lo is None or num > lo:
                            lo, lo_strict = num, False
                        continue
                elif k == "gt":
                    num = _parse_float_token(v)
                    if num is not None:
                        if lo is None or num > lo or (num == lo and not lo_strict):
                            lo, lo_strict = num, True
                        continue
                elif k in ("max", "lte"):
                    num = _parse_float_token(v)
                    if num is not None:
                        if hi is None or num < hi:
                            hi, hi_strict = num, False
                        continue
                elif k == "lt":
                    num = _parse_float_token(v)
                    if num is not None:
                        if hi is None or num < hi or (num == hi and not hi_strict):
                            hi, hi_strict = num, True
                        continue
                elif k == "nonzero":
                    if v in ("true", "1", "yes"):
                        nonzero = True
                    continue
                elif k == "non_empty":
                    if v in ("true", "1", "yes"):
                        non_empty = True
                    continue
                elif k == "dtype":
                    if v and v not in ("any", "*"):
                        dtype = v
                    continue

            preds.add(low)

        return cls(
            lo=lo,
            lo_strict=lo_strict,
            hi=hi,
            hi_strict=hi_strict,
            nonzero=nonzero,
            non_empty=non_empty,
            dtype=dtype,
            predicates=tuple(sorted(preds)),
        )

    def entails(self, other: "RefinementSet", *, prod_dtype: Optional[str] = None) -> bool:
        """Return True iff every value satisfying `self` necessarily satisfies `other`."""
        if other.lo is not None:
            if self.lo is None:
                return False
            if self.lo < other.lo:
                return False
            if self.lo == other.lo and other.lo_strict and not self.lo_strict:
                return False

        if other.hi is not None:
            if self.hi is None:
                return False
            if self.hi > other.hi:
                return False
            if self.hi == other.hi and other.hi_strict and not self.hi_strict:
                return False

        if other.effective_nonzero and not self.effective_nonzero:
            return False

        if other.non_empty and not self.non_empty:
            return False

        if other.dtype:
            eff_dt = self.dtype or (str(prod_dtype).strip().lower() if prod_dtype else None)
            if eff_dt != other.dtype:
                return False

        if other.predicates:
            p_set = set(self.predicates)
            for req_p in other.predicates:
                if req_p not in p_set:
                    return False

        return True

    def check_literal(self, val: Any) -> Tuple[bool, Optional[str]]:
        """Validate a concrete literal value against interval, nonzero, and non_empty bounds."""
        if val is None:
            return True, None

        if self.non_empty and isinstance(val, (str, list, tuple, dict, set)) and len(val) == 0:
            return False, f"expected non_empty value, got empty {type(val).__name__}"

        if isinstance(val, (int, float)) and not isinstance(val, bool):
            fval = float(val)
            if self.lo is not None:
                if fval < self.lo or (self.lo_strict and fval == self.lo):
                    op = ">" if self.lo_strict else ">="
                    return False, f"value {val} violates {op} {self.lo:g}"
            if self.hi is not None:
                if fval > self.hi or (self.hi_strict and fval == self.hi):
                    op = "<" if self.hi_strict else "<="
                    return False, f"value {val} violates {op} {self.hi:g}"
            if self.effective_nonzero and fval == 0.0:
                return False, f"value {val} violates nonzero invariant"

        return True, None

    def join(self, other: "RefinementSet") -> "RefinementSet":
        """Least upper bound: outer interval hull and intersection of guaranteed flags/predicates."""
        if self.lo is not None and other.lo is not None:
            if self.lo < other.lo:
                j_lo, j_lo_strict = self.lo, self.lo_strict
            elif other.lo < self.lo:
                j_lo, j_lo_strict = other.lo, other.lo_strict
            else:
                j_lo, j_lo_strict = self.lo, (self.lo_strict and other.lo_strict)
        else:
            j_lo, j_lo_strict = None, False

        if self.hi is not None and other.hi is not None:
            if self.hi > other.hi:
                j_hi, j_hi_strict = self.hi, self.hi_strict
            elif other.hi > self.hi:
                j_hi, j_hi_strict = other.hi, other.hi_strict
            else:
                j_hi, j_hi_strict = self.hi, (self.hi_strict and other.hi_strict)
        else:
            j_hi, j_hi_strict = None, False

        j_nz = self.effective_nonzero and other.effective_nonzero
        j_ne = self.non_empty and other.non_empty
        j_dt = self.dtype if self.dtype == other.dtype else None
        j_preds = tuple(sorted(set(self.predicates) & set(other.predicates)))

        return RefinementSet(
            lo=j_lo,
            lo_strict=j_lo_strict,
            hi=j_hi,
            hi_strict=j_hi_strict,
            nonzero=j_nz,
            non_empty=j_ne,
            dtype=j_dt,
            predicates=j_preds,
        )

    def to_invariants(self) -> Tuple[str, ...]:
        items: List[str] = []
        if self.lo is not None:
            key = "gt" if self.lo_strict else "min"
            items.append(f"{key}={self.lo:g}")
        if self.hi is not None:
            key = "lt" if self.hi_strict else "max"
            items.append(f"{key}={self.hi:g}")
        if self.nonzero:
            items.append("nonzero=true")
        if self.non_empty:
            items.append("non_empty")
        if self.dtype:
            items.append(f"dtype={self.dtype}")
        items.extend(self.predicates)
        return tuple(items)


_SHAPE_WILDCARDS: Tuple[str, ...] = ("", "*", "?", "any", "none", "_")


def _is_concrete_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def unify_shapes(
    prod_shape: Optional[Sequence[Union[int, str, None]]],
    cons_shape: Optional[Sequence[Union[int, str, None]]],
    *,
    prod_length: Optional[Union[int, str]] = None,
    cons_length: Optional[Union[int, str]] = None,
    cons_rank_poly: bool = False,
) -> bool:
    """
    Unify producer and consumer dimensional shapes and container lengths.

    Enforces concrete equality and consistent symbolic dimension variable bindings
    (e.g., `["N", "N"]` accepts `[4, 4]` and rejects `[3, 4]`), treating `*`/`?`/`_` as wildcards.
    """
    if _is_concrete_int(cons_length) and _is_concrete_int(prod_length):
        if int(prod_length) != int(cons_length):
            return False

    if cons_rank_poly or cons_shape is None or prod_shape is None:
        return True

    if len(prod_shape) != len(cons_shape):
        return False

    prod_bindings: Dict[str, int] = {}
    cons_bindings: Dict[str, int] = {}
    for p_dim, c_dim in zip(prod_shape, cons_shape):
        if p_dim is None or c_dim is None:
            continue
        if _is_concrete_int(p_dim) and _is_concrete_int(c_dim):
            if int(p_dim) != int(c_dim):
                return False
            continue
        if isinstance(c_dim, str):
            sym = c_dim.strip()
            if sym.lower() not in _SHAPE_WILDCARDS and _is_concrete_int(p_dim):
                val = int(p_dim)
                if sym in cons_bindings and cons_bindings[sym] != val:
                    return False
                cons_bindings[sym] = val
        if isinstance(p_dim, str):
            p_sym = p_dim.strip()
            if p_sym.lower() not in _SHAPE_WILDCARDS and _is_concrete_int(c_dim):
                val = int(c_dim)
                if p_sym in prod_bindings and prod_bindings[p_sym] != val:
                    return False
                prod_bindings[p_sym] = val
    return True


def join_shapes(
    s1: Optional[Sequence[Union[int, str, None]]],
    s2: Optional[Sequence[Union[int, str, None]]],
) -> Optional[Tuple[Union[int, str, None], ...]]:
    """Compute pointwise shape hull across branch alternatives."""
    if s1 is None or s2 is None or len(s1) != len(s2):
        return None
    return tuple(d1 if d1 == d2 else None for d1, d2 in zip(s1, s2))


__all__ = [
    "CarrierType",
    "parse_carrier",
    "format_carrier",
    "BoundedPoset",
    "RefinementSet",
    "unify_shapes",
    "join_shapes",
]
