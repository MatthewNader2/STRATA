"""Deterministic unit tests for STRATA's 4-component formal type system and bounded poset."""
from __future__ import annotations

import pytest

from bridge_graph import CanonicalPortType, propagate_polymorphic_signature, signatures_unify
from schema import CellSchema, LayoutContract, PolymorphicPropagationSchema, PortSchema
from strata_pipeline import StrataPipeline
from type_system import BoundedPoset, RefinementSet, join_shapes, parse_carrier, unify_shapes


def _sample_poset() -> BoundedPoset:
    return BoundedPoset(
        type_parents={
            "bool": {"int"},
            "int": {"float", "number"},
            "float": {"number"},
            "number": {"any"},
            "str": {"sequence", "any"},
            "list": {"sequence", "collection"},
            "sequence": {"collection", "any"},
            "tuple": {"sequence", "any"},
            "file": {"any"},
            "csv": {"format"},
            "png": {"imageformat", "format"},
            "imageformat": {"format"},
            "format": {"any"},
        }
    )


def test_parametric_subtyping_and_variance() -> None:
    poset = _sample_poset()

    # Covariant collections & reverse erasure guard
    assert poset.is_subtype("List[int]", "List[float]") is True
    assert poset.is_subtype("List[int]", "Sequence[number]") is True
    assert poset.is_subtype("List[float]", "List[int]") is False
    assert poset.is_subtype("List[int]", "list") is True
    assert poset.is_subtype("list", "List[Any]") is True
    assert poset.is_subtype("list", "List[int]") is False

    # Fixed vs variadic tuples
    assert poset.is_subtype("Tuple[int, str]", "Tuple[float, Any]") is True
    assert poset.is_subtype("Tuple[int, int]", "Tuple[int, ...]") is True
    assert poset.is_subtype("Tuple[int, str]", "Tuple[int, ...]") is False

    # Optional & Union
    assert poset.is_subtype("None", "Optional[int]") is True
    assert poset.is_subtype("int", "Optional[int]") is True
    assert poset.is_subtype("Optional[int]", "int") is False
    assert poset.is_subtype("bool", "Union[int, str]") is True
    assert poset.is_subtype("Union[int, str]", "Any") is True

    # Function contravariant domain / covariant codomain
    assert poset.is_subtype("Callable[[float], int]", "Callable[[int], float]") is True
    assert poset.is_subtype("Callable[[int], int]", "Callable[[float], int]") is False
    assert poset.is_subtype("Callable[[number], bool]", "Callable[[float], int]") is True

    # Arity-normalized File[Modality, Format]
    assert poset.is_subtype("File[Any, CSV]", "File[CSV]") is True
    assert poset.is_subtype("File[Image, PNG]", "File[Image, Any]") is True
    assert poset.is_subtype("File[Image, PNG]", "File[PNG]") is True


def test_single_letter_typevar_and_state_isolation() -> None:
    poset = _sample_poset()
    st_ct = parse_carrier("State")
    s_ct = parse_carrier("S")
    assert st_ct.is_var is False
    assert s_ct.is_var is True
    assert poset.is_subtype("int", "State") is False
    assert poset.is_subtype("int", "S") is True


def test_bound_tracked_type_variable_unification() -> None:
    poset = _sample_poset()
    assert poset.is_subtype("Tuple[int, str]", "Tuple[T, T]") is False
    assert poset.is_subtype("Tuple[int, int]", "Tuple[T, T]") is True
    assert poset.is_subtype("Tuple[bool, int]", "Tuple[T, T]") is True
    assert poset.is_subtype("Callable[[int], str]", "Callable[[T], T]") is False
    assert poset.is_subtype("Callable[[int], bool]", "Callable[[T], T]") is True
    assert poset.is_subtype("Union[int, str]", "T") is True


def test_symbolic_and_concrete_shape_unification() -> None:
    assert unify_shapes(["H", "W", 3], [480, 640, 3]) is True
    assert unify_shapes(["H", "W", 3], [480, 640, 1]) is False
    assert unify_shapes(["*", 3], [10, 3]) is True
    assert unify_shapes(["*", 3], [10, 4]) is False
    assert unify_shapes(["N", "N"], [4, 4]) is True
    assert unify_shapes(["N", "N"], [3, 4]) is False
    assert unify_shapes(["M", "M"], ["N", "N"]) is True
    assert unify_shapes(["M", "K"], ["N", "N"]) is False
    assert unify_shapes(["N", "N"], ["M", "K"]) is False
    assert unify_shapes(["H", "W", 3], ["H", "W", 3]) is True

    j_hw = join_shapes([480, 640, 3], [240, 320, 3])
    assert j_hw is not None and len(j_hw) == 3 and j_hw[2] == 3
    assert join_shapes([480, 640, 3], [240, 320]) is None


def test_refinement_entailment_and_strict_literal_checks() -> None:
    r_min2 = RefinementSet.from_invariants(["min=2"])
    r_min1 = RefinementSet.from_invariants(["min=1"])
    r_gt0 = RefinementSet.from_invariants(["gt=0"])
    r_min0 = RefinementSet.from_invariants(["min=0"])
    r_nz = RefinementSet.from_invariants(["nonzero"])

    assert r_min2.entails(r_min1) is True
    assert r_min1.entails(r_min2) is False
    assert r_gt0.entails(r_nz) is True
    assert r_min0.entails(r_gt0) is False
    assert r_min0.entails(r_nz) is False

    r_odd = RefinementSet.from_invariants(["min=1", "odd"])
    assert r_odd.check_literal(3)[0] is True
    assert r_odd.check_literal(2)[0] is False
    assert r_odd.check_literal((3, 5))[0] is True
    assert r_odd.check_literal((0, 0))[0] is False
    assert r_odd.check_literal(None)[0] is False
    assert r_odd.check_literal(True)[0] is False
    assert r_odd.check_literal("3")[0] is False

    r_even = RefinementSet.from_invariants(["even"])
    assert r_even.check_literal(4)[0] is True
    assert r_even.check_literal(5)[0] is False


def test_compiled_poset_laws_and_runtime_cycle_rejection(compiled_lattice) -> None:
    multi = [k for k, ps in compiled_lattice.type_parents.items() if len(ps) > 1]
    assert len(multi) >= 5
    poset = BoundedPoset(compiled_lattice.type_parents, compiled_lattice.state_parents, compiled_lattice.top_types)
    assert poset.validate_acyclic() == []

    for a, b in (("int", "float"), ("bool", "int"), ("list", "tuple"), ("str", "path")):
        assert poset.join_carriers(a, a) == a
        assert poset.meet_carriers(a, a) == a
        j_ab = poset.join_carriers(a, b)
        assert j_ab == poset.join_carriers(b, a)
        assert poset.is_subtype(a, j_ab) and poset.is_subtype(b, j_ab)
        m_ab = poset.meet_carriers(a, b)
        assert m_ab == "void" or (poset.is_subtype(m_ab, a) and poset.is_subtype(m_ab, b))

    cyc = BoundedPoset(type_parents={"a": {"b"}, "b": {"a"}})
    assert len(cyc.validate_acyclic()) >= 1

    cyc_states = BoundedPoset(state_parents={"dom::s1": {"dom::s2"}, "dom::s2": {"dom::s1"}})
    assert len(cyc_states.validate_acyclic()) >= 1

    with pytest.raises(ValueError, match="Cycle"):
        StrataPipeline([], cyc)


def test_polymorphic_propagation_preserves_shape_and_dtype() -> None:
    src_sig = CanonicalPortType(carrier="ndarray", typestate="numpy::tensor", rank=3, shape=(64, 64, 3), dtype="uint8", domain="numpy")
    raw_out = CanonicalPortType(carrier="ndarray", typestate="numpy::tensor", rank="R", dtype="D", domain="numpy")
    cell = CellSchema(
        cell_id="POLY_CLIP",
        domain_name="numpy",
        code_template="{out} = {arr}",
        inputs={"arr": PortSchema(carrier="ndarray", typestate="any", layout=LayoutContract(rank="R", dtype="D"))},
        outputs={"out": PortSchema(carrier="ndarray", typestate="numpy::tensor", layout=LayoutContract(rank="R", dtype="D"))},
        polymorphic_propagation=PolymorphicPropagationSchema(preserves_layout=True, preserves_dtype=True, source_port="arr"),
    )
    res = propagate_polymorphic_signature(src_sig, raw_out, cell)
    assert res.rank == 3 and res.shape == (64, 64, 3) and res.dtype == "uint8"
    assert signatures_unify(CanonicalPortType(carrier="tuple", length=2), CanonicalPortType(carrier="tuple", length=3), _sample_poset()) is False
