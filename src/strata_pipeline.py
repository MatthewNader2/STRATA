"""
src/strata_pipeline.py — Unified, domain-agnostic STRATA pipeline runner.

Connects:
  Layer 0: Unified TypedIR Frontend (formal syntax for model-free profiles; SLM compiler otherwise)
  Layer 1: Neutral Per-Subgoal Candidate Retrieval (zero keyword-stuffing reliance)
  Layer 2: Type-Directed Planning with Live Cross-Domain Bridges & Linear Resource Checks
  Layer 3: Pure Algebraic Port Binding & LanguageBackend Code Emission
  Layer 4/5: Declarative Domain Runtime Setups & Optional Sandboxed Execution
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

try:
    from .bridge_graph import LiveBridgeGraph, SimpleTypePoset
    from .config import settings
    from .frontends import compile_prompt_to_ir
    from .language_backend import get_active_language_spec, get_language_backend, set_active_language_spec
    from .port_binder import BoundStep, bind_and_emit
    from .schema import TreeSchema, get_attr, normalize_typestate
    from .subgoal_retrieval import NeutralSubgoalRetriever
    from .typed_ir import TypedIR
    from .typed_planner import CandidatePlan, PlannedStep, TypedIRPlanner
except (ImportError, ValueError):
    from bridge_graph import LiveBridgeGraph, SimpleTypePoset
    from config import settings
    from frontends import compile_prompt_to_ir
    from language_backend import get_active_language_spec, get_language_backend, set_active_language_spec
    from port_binder import BoundStep, bind_and_emit
    from schema import TreeSchema, get_attr, normalize_typestate
    from subgoal_retrieval import NeutralSubgoalRetriever
    from typed_ir import TypedIR
    from typed_planner import CandidatePlan, PlannedStep, TypedIRPlanner


@dataclass(frozen=True)
class PipelineRunResult:
    ok: bool
    stage_failed: Optional[str] = None
    refusal_kind: Optional[str] = None
    error: Optional[str] = None
    warnings: Tuple[str, ...] = ()
    ir: Optional[TypedIR] = None
    plan: Optional[CandidatePlan] = None
    plan_candidates: Tuple[CandidatePlan, ...] = ()
    bound_steps: Tuple[BoundStep, ...] = ()
    code: str = ""
    live_bridge_count: int = 0
    runtime_setups: Tuple[Dict[str, Any], ...] = ()
    sandbox_result: Optional[Dict[str, Any]] = None


def _count_live_bridges(steps: Iterable[PlannedStep]) -> int:
    total = 0
    for step in steps:
        if step.is_live_bridge:
            total += 1
        if step.body_steps:
            total += _count_live_bridges(step.body_steps)
        if step.else_steps:
            total += _count_live_bridges(step.else_steps)
    return total


def _iter_leaf_steps(steps: Iterable[PlannedStep]) -> Iterable[PlannedStep]:
    for step in steps:
        if step.kind == "call":
            yield step
        if step.body_steps:
            yield from _iter_leaf_steps(step.body_steps)
        if step.else_steps:
            yield from _iter_leaf_steps(step.else_steps)


class StrataPipeline:
    """Single canonical, hot-swappable pipeline orchestrator over typed lattice domains."""

    def __init__(
        self,
        cells: Iterable[Any],
        poset: Any,
        *,
        has_llm: bool = False,
        generate_fn: Optional[Callable[..., str]] = None,
        embed_fn: Optional[Callable[..., List[List[float]]]] = None,
        rerank_fn: Optional[Callable[[str, List[str]], List[float]]] = None,
        choice_scorer_fn: Optional[Callable[[str, List[str]], List[float]]] = None,
        slot_fill_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        sandbox_exec_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        domain_runtime_setups: Optional[Dict[str, Dict[str, Any]]] = None,
        domain_imports: Optional[Dict[str, List[str]]] = None,
        language_spec: Optional[Any] = None,
        max_bridge_hops: Optional[int] = None,
    ) -> None:
        self.cells: List[Any] = list(cells)
        self.poset = poset
        self.has_llm = bool(has_llm)
        self.generate_fn = generate_fn
        self.slot_fill_fn = slot_fill_fn
        self.sandbox_exec_fn = sandbox_exec_fn
        self.language_spec = language_spec
        self.language_backend = get_language_backend(language_spec) if language_spec is not None else None
        self.domain_runtime_setups: Dict[str, Dict[str, Any]] = {
            str(k).strip().lower(): dict(v)
            for k, v in (domain_runtime_setups or {}).items()
            if str(k).strip() and isinstance(v, dict)
        }
        self.domain_imports: Dict[str, List[str]] = {
            str(k).strip().lower(): [str(imp).strip().lower() for imp in v if str(imp).strip()]
            for k, v in (domain_imports or {}).items()
            if str(k).strip() and isinstance(v, (list, tuple))
        }
        self.loaded_domains: List[str] = sorted(
            {
                str(get_attr(c, "domain_name", None) or "").strip().lower()
                for c in self.cells
                if str(get_attr(c, "domain_name", None) or "").strip()
            }
            | set(self.domain_imports.keys())
        )
        loaded_set: Set[str] = set(self.loaded_domains)
        self.missing_imported_domains: Dict[str, List[str]] = {}
        self.import_warnings: Tuple[str, ...] = ()
        imp_warns: List[str] = []
        for dom, imps in sorted(self.domain_imports.items()):
            missing = [imp for imp in imps if imp not in loaded_set]
            if missing:
                self.missing_imported_domains[dom] = missing
                imp_warns.append(
                    f"Domain '{dom}' declares import(s) {missing} not present in loaded domains."
                )
        self.import_warnings = tuple(imp_warns)

        self.bridge_graph = LiveBridgeGraph.build(self.cells, poset)
        self.retriever = NeutralSubgoalRetriever(
            self.cells,
            embed_fn=embed_fn,
            rerank_fn=rerank_fn,
            domain_hint_bonus=settings.retrieval_domain_hint_bonus,
        )
        self.planner = TypedIRPlanner(
            self.bridge_graph,
            poset,
            choice_scorer_fn=choice_scorer_fn,
            max_bridge_hops=max_bridge_hops,
        )

    @classmethod
    def from_trees(cls, trees: Iterable[Any], **kwargs: Any) -> "StrataPipeline":
        """Build a live pipeline directly from any collection of TreeSchema / tree JSON dicts."""
        type_parents: Dict[str, Set[str]] = {}
        state_parents: Dict[str, Set[str]] = {}
        top_types: List[str] = []
        all_cells: List[Any] = []
        setups: Dict[str, Dict[str, Any]] = {}
        imports_map: Dict[str, List[str]] = {}
        lang_spec: Any = kwargs.pop("language_spec", None)
        core_lang_spec: Any = None
        fallback_lang_spec: Any = None

        def _add_type_parent(child: str, parent: str, dom_l: str) -> None:
            child_l = str(child or "").strip().lower()
            parent_l = str(parent or "").strip().lower()
            if not child_l or not parent_l:
                return
            type_parents.setdefault(child_l, set()).add(parent_l)
            if "::" not in child_l and dom_l:
                type_parents.setdefault(f"{dom_l}::{child_l}", set()).add(parent_l)

        for raw_tree in trees:
            tree = raw_tree if isinstance(raw_tree, TreeSchema) else TreeSchema(**raw_tree)
            dom = str(tree.domain or "").strip().lower()
            if dom:
                imports_map[dom] = [str(i).strip().lower() for i in (tree.imports or ()) if str(i).strip()]
            if tree.runtime_setup is not None and dom:
                setups[dom] = tree.runtime_setup.model_dump()
            if tree.language_spec is not None:
                if tree.is_core and core_lang_spec is None:
                    core_lang_spec = tree.language_spec
                elif fallback_lang_spec is None and dom:
                    fallback_lang_spec = tree.language_spec

            for tt in (tree.top_types or ()):
                tt_s = str(tt or "").strip().lower()
                if tt_s:
                    top_types.append(tt_s)

            raw_types = tree.types or {}
            if isinstance(raw_types, dict):
                for tname, tinfo in raw_types.items():
                    if isinstance(tinfo, dict):
                        for par in (tinfo.get("parents") or ()):
                            _add_type_parent(str(tname), str(par), dom)
                        for k_p in ("parent", "parent_type"):
                            if tinfo.get(k_p):
                                _add_type_parent(str(tname), str(tinfo[k_p]), dom)
            elif isinstance(raw_types, list):
                for tinfo in raw_types:
                    if isinstance(tinfo, dict):
                        tname = tinfo.get("name") or tinfo.get("type_name")
                        if tname:
                            for par in (tinfo.get("parents") or ()):
                                _add_type_parent(str(tname), str(par), dom)
                            for k_p in ("parent", "parent_type"):
                                if tinfo.get(k_p):
                                    _add_type_parent(str(tname), str(tinfo[k_p]), dom)

            ts_block = tree.typestates
            states_seq = list(getattr(ts_block, "states", []) or []) if ts_block is not None else []
            for st_item in states_seq:
                sname = st_item.get("name") if isinstance(st_item, dict) else getattr(st_item, "name", None)
                sparent = (
                    st_item.get("parent_state")
                    if isinstance(st_item, dict)
                    else getattr(st_item, "parent_state", None)
                )
                if sname and sparent:
                    sn = normalize_typestate(str(sname), dom)
                    sp_vals = sparent if isinstance(sparent, (list, tuple, set)) else (sparent,)
                    for sp_raw in sp_vals:
                        sp = normalize_typestate(str(sp_raw), dom)
                        if sn != "any" and sp != "any":
                            state_parents.setdefault(sn, set()).add(sp)

            for cell in (tree.cells or ()):
                c_copy = cell.model_copy(deep=True) if hasattr(cell, "model_copy") else cell
                if dom and not getattr(c_copy, "domain_name", None):
                    c_copy.domain_name = dom
                all_cells.append(c_copy)

        if lang_spec is None:
            lang_spec = core_lang_spec or fallback_lang_spec

        poset = SimpleTypePoset(
            type_parents=type_parents,
            state_parents=state_parents,
            top_types=top_types,
        )
        return cls(
            all_cells,
            poset,
            domain_runtime_setups=setups,
            domain_imports=imports_map,
            language_spec=lang_spec,
            **kwargs,
        )

    @classmethod
    def from_compiled(cls, compiled: Any, **kwargs: Any) -> "StrataPipeline":
        """Build a live pipeline from a compiled CompiledLattice snapshot."""
        cells = list(getattr(compiled, "cells", ()) or ())
        tp = getattr(compiled, "type_parents", None) or {}
        sp = getattr(compiled, "state_parents", None) or {}
        tt = getattr(compiled, "top_types", None) or ()
        setups = dict(getattr(compiled, "domain_runtime_setups", None) or {})
        imports_map = dict(getattr(compiled, "domain_imports", None) or {})
        lang_spec = kwargs.pop("language_spec", getattr(compiled, "language_spec", None))
        poset = SimpleTypePoset(type_parents=tp, state_parents=sp, top_types=tt)
        return cls(
            cells,
            poset,
            domain_runtime_setups=setups,
            domain_imports=imports_map,
            language_spec=lang_spec,
            **kwargs,
        )

    @classmethod
    def from_trees_dir(cls, trees_dir: Union[str, os.PathLike[str]], **kwargs: Any) -> "StrataPipeline":
        """Load all *.json manifests from a directory into a live hot-swappable pipeline."""
        root_dir = os.fspath(trees_dir)
        raw_trees: List[Dict[str, Any]] = []
        if os.path.isdir(root_dir):
            for fname in sorted(os.listdir(root_dir)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(root_dir, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "r", encoding="utf-8") as fh:
                        raw_trees.append(json.load(fh))
        return cls.from_trees(raw_trees, **kwargs)

    def run(self, prompt: str, *, execute: bool = False, top_k: int = 8) -> PipelineRunResult:
        if self.language_spec is None:
            return self._execute_run(prompt, execute=execute, top_k=top_k)
        prev_spec = get_active_language_spec()
        set_active_language_spec(self.language_spec)
        try:
            return self._execute_run(prompt, execute=execute, top_k=top_k)
        finally:
            set_active_language_spec(prev_spec)

    def _execute_run(self, prompt: str, *, execute: bool = False, top_k: int = 8) -> PipelineRunResult:
        active_backend = self.language_backend

        # Layer 0: Prompt -> TypedIR
        fe_res = compile_prompt_to_ir(
            prompt,
            has_llm=self.has_llm,
            generate_fn=self.generate_fn,
            loaded_domains=self.loaded_domains,
        )
        warnings_tup = tuple(dict.fromkeys(list(self.import_warnings) + list(fe_res.warnings)))
        if not fe_res.ok or fe_res.ir is None:
            return PipelineRunResult(
                ok=False,
                stage_failed="frontend",
                refusal_kind="SLM_COMPILE_ERROR" if self.has_llm else "FORMAL_SYNTAX_ERROR",
                error=fe_res.error,
                warnings=warnings_tup,
            )

        ir = fe_res.ir

        # Layer 1: Neutral Per-Subgoal Retrieval (call nodes only)
        cands_by_node = self.retriever.retrieve_for_ir(ir, top_k=top_k)
        missing_nodes = [
            f"{node.id} ({node.intent!r})"
            for node in ir.flatten_nodes()
            if node.kind == "call" and not cands_by_node.get(node.id)
        ]
        if missing_nodes:
            return PipelineRunResult(
                ok=False,
                stage_failed="retrieval",
                refusal_kind="NO_MATCHING_OPERATION",
                error=f"No matching operations retrieved for node(s): {', '.join(missing_nodes)}",
                warnings=warnings_tup,
                ir=ir,
            )

        # Layer 2: Type-Directed Planning with Live Cross-Domain Bridges & Control Flow
        plan_res = self.planner.plan(ir, cands_by_node)
        if not plan_res.ok or plan_res.chosen_plan is None:
            return PipelineRunResult(
                ok=False,
                stage_failed="planning",
                refusal_kind=plan_res.refusal_kind,
                error=plan_res.reason,
                warnings=warnings_tup,
                ir=ir,
                plan_candidates=plan_res.candidates,
            )

        chosen = plan_res.chosen_plan
        live_bridge_count = _count_live_bridges(chosen.steps)

        # Layer 3: Algebraic Port Binding & LanguageBackend Emission
        bind_res = bind_and_emit(
            chosen,
            ir,
            slot_fill_fn=self.slot_fill_fn,
            language_backend=active_backend,
        )
        if not bind_res.ok:
            return PipelineRunResult(
                ok=False,
                stage_failed="binding",
                refusal_kind=bind_res.refusal_kind,
                error=bind_res.error,
                warnings=warnings_tup,
                ir=ir,
                plan=chosen,
                plan_candidates=plan_res.candidates,
                live_bridge_count=live_bridge_count,
            )

        # Layer 4/5: Declarative Runtime Setups, Explicit Input Assets & Optional Execution
        used_doms: List[str] = []
        used_setups: List[Dict[str, Any]] = []
        leaf_steps = list(_iter_leaf_steps(chosen.steps))
        for step in leaf_steps:
            s_dom = str(get_attr(step.cell, "domain_name", None) or "").strip().lower()
            if s_dom and s_dom not in used_doms and s_dom in self.domain_runtime_setups:
                used_doms.append(s_dom)
                used_setups.append(self.domain_runtime_setups[s_dom])

        input_assets = list(bind_res.input_assets)
        egress_paths = list(bind_res.egress_paths)
        if not input_assets and not egress_paths:
            input_assets = [
                str(val) for val in ir.grounded_assets().values() if val is not None and str(val).strip()
            ]
        sb_res: Optional[Dict[str, Any]] = None
        if execute and self.sandbox_exec_fn is not None:
            sb_res = self.sandbox_exec_fn(
                bind_res.code,
                runtime_setups=list(used_setups),
                input_assets=input_assets,
                egress_paths=egress_paths,
            )
            if not isinstance(sb_res, dict) or not sb_res.get("success", False):
                err_msg = (
                    str(sb_res.get("error") or sb_res.get("stderr") or "Sandbox execution failed")
                    if isinstance(sb_res, dict)
                    else "Sandbox execution returned invalid payload"
                )
                return PipelineRunResult(
                    ok=False,
                    stage_failed="sandbox",
                    refusal_kind="SANDBOX_ERROR",
                    error=err_msg,
                    warnings=warnings_tup,
                    ir=ir,
                    plan=chosen,
                    plan_candidates=plan_res.candidates,
                    bound_steps=bind_res.steps,
                    code=bind_res.code,
                    live_bridge_count=live_bridge_count,
                    runtime_setups=tuple(used_setups),
                    sandbox_result=sb_res if isinstance(sb_res, dict) else None,
                )

            if ir.deliverable:
                sb_scope = sb_res.get("results") or {}
                planned_vars = {s.output_var for s in leaf_steps if s.output_var}
                if ir.deliverable in planned_vars and ir.deliverable in sb_scope and sb_scope[ir.deliverable] is None:
                    return PipelineRunResult(
                        ok=False,
                        stage_failed="sandbox",
                        refusal_kind="DELIVERABLE_NONE",
                        error=f"Deliverable '{ir.deliverable}' evaluated to None in sandbox execution.",
                        warnings=warnings_tup,
                        ir=ir,
                        plan=chosen,
                        plan_candidates=plan_res.candidates,
                        bound_steps=bind_res.steps,
                        code=bind_res.code,
                        live_bridge_count=live_bridge_count,
                        runtime_setups=tuple(used_setups),
                        sandbox_result=sb_res,
                    )

        return PipelineRunResult(
            ok=True,
            warnings=warnings_tup,
            ir=ir,
            plan=chosen,
            plan_candidates=plan_res.candidates,
            bound_steps=bind_res.steps,
            code=bind_res.code,
            live_bridge_count=live_bridge_count,
            runtime_setups=tuple(used_setups),
            sandbox_result=sb_res,
        )


__all__ = ["PipelineRunResult", "StrataPipeline"]
