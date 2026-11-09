# src/main.py
"""
src/main.py - STRATA (Symbolic Topological Routing Architecture for Typed Automata)
Production REST API Server and Core Engine Lifecycle Manager.
Provides real-time endpoints for health monitoring, domain metadata,
multi-profile engine initialization, program synthesis, and benchmark automation.
"""

from __future__ import annotations
import os
import asyncio
import sys
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

try:
    import resource
except ImportError:
    resource = None

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from log_config import get_logger

try:
    from .config import settings, MODELS_DIR
    from .lattice import LatticeOrchestrator
    from .router import LatticeRouter
    from .unification import UnificationGate, ExecutionContext
    from .gevr_sandbox import GEVRSandbox
    from .inference import ModelManager, select_optimal_embedder
    from .internal_rag import LocalRAG
    from .preflight import PreflightLinter
    from .utils import ensure_comprehensive_prompt, extract_code_from_llm_response
except (ImportError, ValueError):
    from config import settings, MODELS_DIR
    from lattice import LatticeOrchestrator
    from router import LatticeRouter
    from unification import UnificationGate, ExecutionContext
    from gevr_sandbox import GEVRSandbox
    from inference import ModelManager, select_optimal_embedder
    from internal_rag import LocalRAG
    from preflight import PreflightLinter
    from utils import ensure_comprehensive_prompt, extract_code_from_llm_response

logger = get_logger("main")

DB_PATH = os.environ.get("NSTL_DB_PATH", str(settings.trees_dir / "lattice.db"))

# =====================================================================
# Global Engine State
# =====================================================================
_orchestrator: Optional[LatticeOrchestrator] = None
_router: Optional[LatticeRouter] = None
_gate: Optional[UnificationGate] = None
_sandbox: Optional[GEVRSandbox] = None
_internal_rag: Optional[LocalRAG] = None
_active_profile: str = "0"
_active_embedder: str = "none"
_active_llm: str = "none"
_engine_ready: bool = False
# Reentrant: initialize_engine holds this lock across its body and calls
# _ensure_base_engine_loaded(), which re-acquires it. A plain Lock deadlocked
# Profile-0 initialization here (thread acquire-vs-itself).
_engine_lock = threading.RLock()
_benchmark_lock = threading.Lock()
_start_time = time.time()
_benchmark_active: bool = False


def _get_process_memory_mb() -> float:
    """
    Resolves current process Resident Set Size (RSS) memory in megabytes across
    platforms agnostically without hardcoded operating system multipliers.
    """
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    # Generic POSIX procfs fallback
    try:
        with open("/proc/self/statm", "r") as f:
            pages = int(f.read().split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    # resource module fallback for POSIX environments where psutil is unavailable
    if resource is not None:
        try:
            raw_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            scale = 1.0 / (1024.0 * 1024.0) if sys.platform.startswith("darwin") else 1.0 / 1024.0
            return round(raw_rss * scale, 2)
        except Exception:
            pass

    return 0.0


def _ensure_base_engine_loaded():
    """Initializes the base SQLite lattice orchestrator, gate, and sandbox with thread synchronization."""
    global _orchestrator, _gate, _sandbox, _router
    if _orchestrator is None:
        with _engine_lock:
            if _orchestrator is None:
                logger.info(f"[*] Preloading STRATA Lattice Database from '{DB_PATH}'...")
                t0 = time.perf_counter()
                orch = LatticeOrchestrator()
                orch.load_from_database(DB_PATH)
                orch.build_topology()
                _orchestrator = orch
                _gate = UnificationGate(orchestrator=_orchestrator)
                _sandbox = GEVRSandbox()
                _router = LatticeRouter(orchestrator=_orchestrator, internal_rag=None)
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info(f"[✓] Base Lattice ready with {len(_orchestrator.loaded_cells)} nodes loaded in {elapsed:.1f}ms.")


# =====================================================================
# Request / Response Schemas
# =====================================================================
def get_orchestrator() -> Optional[LatticeOrchestrator]:
    """Returns the globally loaded LatticeOrchestrator singleton."""
    _ensure_base_engine_loaded()
    return _orchestrator


class InitRequest(BaseModel):
    profile: str = Field(default="0", description="Operational profile: 0, A, C, D, or E")
    embedder_model: Optional[str] = Field(default="auto", description="Embedding model identifier or 'auto'")
    llm_model: Optional[str] = Field(default="", description="GGUF LLM model identifier or 'auto'")
    embedder_device: Optional[str] = Field(default="auto", description="Target compute device: cpu, cuda, or auto")
    llm_device: Optional[str] = Field(default="auto", description="Target compute device: cpu, cuda, or auto")


class RunRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000, description="Natural language specification")
    execute_in_sandbox: bool = Field(default=False, description="Whether to execute code in GEVR sandbox")
    timeout_seconds: float = Field(default=5.0, ge=0.5, le=60.0, description="Sandbox execution timeout")
    allowed_capabilities: Optional[List[str]] = Field(default=None, description="Optional list of capabilities to grant")


class RunResponse(BaseModel):
    status: str = "success"
    prompt: str
    path: List[str]
    code: str
    latency_ms: float = 0.0
    route_latency_ms: float = 0.0
    synthesis_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    sandbox_result: Optional[Dict[str, Any]] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)


class HealthResponse(BaseModel):
    status: str
    nodes_count: int
    uptime_seconds: float
    memory_rss_mb: float


class DomainInfo(BaseModel):
    total_nodes: int
    curated_seeds: int


class DomainsResponse(BaseModel):
    domains: Dict[str, DomainInfo]
    total_domains: int
    total_nodes: int


# =====================================================================
# Core In-Process Engine Functions
# =====================================================================
def initialize_engine(
    req_or_profile: Union[InitRequest, Dict[str, Any], str] = "0",
    embedder_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    embedder_device: str = "auto",
    llm_device: str = "auto",
    trees_storage: str = "ram"
) -> Dict[str, Any]:
    """
    Initializes or reconfigures the STRATA synthesis engine for a specific profile.
    Can be called directly in-process or via REST endpoint.
    """
    global _orchestrator, _router, _gate, _sandbox, _internal_rag
    global _active_profile, _active_embedder, _active_llm, _engine_ready

    if isinstance(req_or_profile, InitRequest):
        prof = req_or_profile.profile
        emb_m = req_or_profile.embedder_model
        llm_m = req_or_profile.llm_model
        emb_dev = req_or_profile.embedder_device or embedder_device
        llm_dev = req_or_profile.llm_device or llm_device
    elif isinstance(req_or_profile, dict):
        prof = req_or_profile.get("profile", "0")
        emb_m = req_or_profile.get("embedder_model", embedder_model)
        llm_m = req_or_profile.get("llm_model", llm_model)
        emb_dev = req_or_profile.get("embedder_device", embedder_device)
        llm_dev = req_or_profile.get("llm_device", llm_device)
    else:
        prof = str(req_or_profile)
        emb_m = embedder_model
        llm_m = llm_model
        emb_dev = embedder_device
        llm_dev = llm_device

    p_norm = prof.strip().upper()
    with _engine_lock:
        _engine_ready = False
        _ensure_base_engine_loaded()

        if p_norm in ("0", "SYMBOLIC", "ZERO", "PURE"):
            _active_profile = "0"
            _active_embedder = "none"
            _active_llm = "none"
            _internal_rag = None
            _router = LatticeRouter(orchestrator=_orchestrator, internal_rag=None)
            _gate = UnificationGate(orchestrator=_orchestrator)
            _engine_ready = True
            logger.info("[✓] STRATA Engine configured for Profile 0 (Pure Symbolic).")
            return {
                "status": "ready",
                "profile": "0",
                "embedder": "none",
                "llm": "none",
                "device": "cpu",
                "nodes": len(_orchestrator.loaded_cells)
            }

        # Neural profiles: A, B, C, D, E, S, U, AE
        try:
            from .router import HardwareProfiler
        except (ImportError, ValueError):
            from router import HardwareProfiler
        HardwareProfiler.set_config(embedder_device=emb_dev, llm_device=llm_dev)
        actual_device = HardwareProfiler.get_optimal_device()

        if p_norm == "U":
            optimal_emb = "llm_early_exit"
        elif p_norm == "AE":
            optimal_emb = "transition_autoencoder"
        else:
            optimal_emb = select_optimal_embedder(emb_m or "auto")
        _active_embedder = optimal_emb

        mm = ModelManager.get_instance()
        mm.initialize_profile(
            profile_type=p_norm,
            embedder_name=optimal_emb,
            llm_name=llm_m or ""
        )
        _active_llm = llm_m or getattr(mm.active_profile, "llm_name", "none")

        _internal_rag = LocalRAG(trees_dir=str(settings.trees_dir), orchestrator=_orchestrator)
        _router = LatticeRouter(orchestrator=_orchestrator, internal_rag=_internal_rag)
        _gate = UnificationGate(orchestrator=_orchestrator)
        _active_profile = p_norm
        _engine_ready = True

        logger.info(f"[✓] STRATA Engine configured for Profile {p_norm} (Embedder: {optimal_emb}, LLM: {_active_llm}).")
        return {
            "status": "ready",
            "profile": p_norm,
            "embedder": optimal_emb,
            "llm": _active_llm,
            "device": actual_device,
            "nodes": len(_orchestrator.loaded_cells)
        }


def run_prompt(request: Union[RunRequest, Dict[str, Any], str]) -> RunResponse:
    """
    Executes end-to-end program synthesis for a natural language prompt.
    Thread-safe: instantiates a request-scoped UnificationGate to guarantee that concurrent
    synthesis threads do not overwrite pipeline bindings, egress paths, or context.
    """
    _ensure_base_engine_loaded()
    if not _engine_ready:
        initialize_engine(_active_profile)

    if isinstance(request, RunRequest):
        prompt = request.prompt
        exec_sandbox = request.execute_in_sandbox
        timeout = request.timeout_seconds
        allowed_capabilities = request.allowed_capabilities
    elif isinstance(request, dict):
        prompt = request.get("prompt", "")
        exec_sandbox = request.get("execute_in_sandbox", False)
        timeout = float(request.get("timeout_seconds", 5.0))
        allowed_capabilities = request.get("allowed_capabilities")
    else:
        prompt = str(request)
        exec_sandbox = False
        timeout = 5.0
        allowed_capabilities = None

    if not prompt or not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    t_start = time.perf_counter()

    # Step 0: Layer 0 Ingestion & Intent Compilation (Profile S / E support)
    effective_prompt = prompt
    intent_data = None
    try:
        mm = ModelManager.get_instance()
        if mm.profile and mm.has_semantic_compiler():
            intent_data = mm.compile_semantic_intent(prompt)
            effective_prompt = ensure_comprehensive_prompt(prompt, intent_data.get("effective_prompt"))
    except Exception:
        intent_data = None

    req_ctx = ExecutionContext(prompt=effective_prompt)
    if intent_data and isinstance(intent_data, dict):
        req_ctx.intent_data = dict(intent_data)
        if hasattr(req_ctx, "blackboard") and req_ctx.blackboard is not None:
            req_ctx.blackboard.lock_from_intent(intent_data)
            if req_ctx.blackboard.locked_by_neural:
                if req_ctx.blackboard.target_sink:
                    req_ctx.parameters["target_sink"] = req_ctx.blackboard.target_sink
        if intent_data.get("operations") and isinstance(intent_data["operations"], list):
            req_ctx.operations = [str(op).replace("_", " ").strip() for op in intent_data["operations"] if str(op).strip()]
        if intent_data.get("hyperparameters") and isinstance(intent_data["hyperparameters"], dict):
            req_ctx.hyperparameters = dict(intent_data["hyperparameters"])
        if intent_data.get("slots") and isinstance(intent_data["slots"], dict):
            req_ctx.llm_slots = dict(intent_data["slots"])
        if intent_data.get("variables") and isinstance(intent_data["variables"], list):
            req_ctx.variables_spec = [str(v).strip() for v in intent_data["variables"] if str(v).strip()]
            req_ctx.llm_slots.setdefault("variables", req_ctx.variables_spec)
        if hasattr(req_ctx, "sync_llm_slots"):
            req_ctx.sync_llm_slots()
        tgt = intent_data.get("target_sink") or intent_data.get("target_key") or intent_data.get("deliverable")
        if tgt:
            req_ctx.target_key = str(tgt).strip()
            req_ctx.target_literals.add(req_ctx.target_key)
            req_ctx.parameters["target_sink"] = req_ctx.target_key
        if intent_data.get("parameters") and isinstance(intent_data["parameters"], dict):
            for p_key, p_val in intent_data["parameters"].items():
                if p_key:
                    req_ctx.parameters.setdefault(str(p_key), p_val)
        if intent_data.get("source_files"):
            req_ctx.parameters["source_uris"] = list(intent_data["source_files"])
        if intent_data.get("dest_files"):
            req_ctx.parameters["dest_uris"] = list(intent_data["dest_files"])

    # 1. Routing
    t_route_0 = time.perf_counter()
    plan_res = _router.plan_path(effective_prompt, return_tuple=False, ctx=req_ctx)
    route_dt = (time.perf_counter() - t_route_0) * 1000.0

    cells = plan_res.cells if hasattr(plan_res, "cells") else plan_res
    if not cells:
        return RunResponse(
            status="error",
            prompt=prompt,
            path=[],
            code="",
            latency_ms=round(route_dt, 2),
            route_latency_ms=round(route_dt, 2),
            synthesis_latency_ms=0.0,
            total_latency_ms=round(route_dt, 2),
            sandbox_result={"success": False, "error": f"No valid path found through lattice for prompt: {prompt}"}
        )

    # 2. Synthesis & Unification (Request-isolated gate instance eliminates race conditions)
    t_synth_0 = time.perf_counter()
    local_gate = UnificationGate(orchestrator=_orchestrator)
    # Profile S: forward structured IR literals / intent into unification context
    ir_literals = getattr(plan_res, "ir_literals", None) or getattr(_router, "last_ir_literals", None) or {}
    intent_payload = dict(intent_data) if intent_data else ({"parameters": dict(ir_literals)} if isinstance(ir_literals, dict) and ir_literals else None)
    try:
        code = local_gate.unify_and_emit(cells, prompt, intent_data=intent_payload)
    except (ValueError, Exception) as unif_err:
        synth_dt = (time.perf_counter() - t_synth_0) * 1000.0
        total_dt = (time.perf_counter() - t_start) * 1000.0
        logger.warning(f"Pipeline unification failed for prompt '{prompt}': {unif_err}")
        return RunResponse(
            status="error",
            prompt=prompt,
            path=[c.cell_id for c in cells],
            code="",
            latency_ms=round(total_dt, 2),
            route_latency_ms=round(route_dt, 2),
            synthesis_latency_ms=round(synth_dt, 2),
            total_latency_ms=round(total_dt, 2),
            sandbox_result={"success": False, "error": f"Pipeline unification failed: {unif_err}"}
        )
    synth_dt = (time.perf_counter() - t_synth_0) * 1000.0
    total_dt = (time.perf_counter() - t_start) * 1000.0

    # 2b. Static Pre-Flight Linting & Emission Gating
    lint_violations: List[str] = []
    if hasattr(local_gate, "last_pipeline_bindings") and local_gate.last_pipeline_bindings:
        try:
            lint_res = PreflightLinter.lint(local_gate.last_pipeline_bindings, prompt=prompt, code_str=code, orchestrator=_orchestrator)
            if not lint_res.is_valid:
                lint_violations = list(lint_res.violations)
                logger.warning(f"Static Pre-Flight Lint Violations: {lint_violations}")
        except Exception as _lint_err:
            logger.debug(f"Preflight linting error: {_lint_err}")

    # Deterministic Bridge Insertion for Pre-Flight Lint Violations
    if lint_violations and cells:
        try:
            from route_methods.m1_clause_anchor import M1ClauseAnchorRouteMethod
            _bridge_router = M1ClauseAnchorRouteMethod(orchestrator=_orchestrator)
            repaired_cells = _bridge_router.insert_property_bridges(cells, prompt=prompt, ctx=req_ctx, trace=False)
            if [c.cell_id for c in repaired_cells] != [c.cell_id for c in cells]:
                rep_code = local_gate.unify_and_emit(repaired_cells, prompt, intent_data=intent_payload)
                rep_lint = PreflightLinter.lint(local_gate.last_pipeline_bindings, prompt=prompt, code_str=rep_code, orchestrator=_orchestrator)
                if rep_lint.is_valid:
                    cells = repaired_cells
                    code = rep_code
                    lint_violations = []
                    logger.info("Deterministic bridge inserted successfully, preflight passed.")
        except Exception as _bridge_err:
            logger.debug(f"Deterministic bridge insertion failed: {_bridge_err}")

    # Layer 5 Self-Repair for Pre-Flight Lint Violations
    # Strictly disabled in symbolic/retrieval profiles (0, A, D) to prevent benchmark backdoor cheating.
    repaired_via_feedback = False
    if lint_violations and _active_profile not in ("0", "A", "D") and ModelManager.get_instance().can_feedback_check():
        mm = ModelManager.get_instance()
        if mm.profile and mm.can_feedback_check():
            err_msg = "Pre-flight lint validation failed:\n" + "\n".join(f"- {v}" for v in lint_violations)
            repaired_code = extract_code_from_llm_response(mm.feedback_check(code, err_msg))
            if repaired_code and repaired_code.strip() != code.strip():
                try:
                    rep_lint = PreflightLinter.lint(local_gate.last_pipeline_bindings, prompt=prompt, code_str=repaired_code, orchestrator=_orchestrator)
                    if rep_lint.is_valid:
                        code = repaired_code
                        lint_violations = []
                        repaired_via_feedback = True
                except Exception:
                    pass

    if lint_violations:
        return RunResponse(
            status="refused",
            prompt=prompt,
            path=[c.cell_id for c in cells],
            code="",
            latency_ms=round(total_dt, 2),
            route_latency_ms=round(route_dt, 2),
            synthesis_latency_ms=round(synth_dt, 2),
            total_latency_ms=round(total_dt, 2),
            sandbox_result={"success": False, "error": f"Pre-flight lint validation failed: {'; '.join(lint_violations)}", "preflight_lint_violations": lint_violations}
        )

    # 3. Optional Sandbox Verification (Egress paths read from local request-isolated gate)
    sandbox_result = None
    if exec_sandbox:
        dest_paths = local_gate.get_egress_paths() or None
        sandbox = _sandbox if _sandbox is not None else GEVRSandbox()
        sandbox_result = sandbox.execute(
            code,
            timeout=timeout,
            egress_paths=dest_paths,
            verification_spec=getattr(local_gate, "last_verification_contract", None),
            runtime_aliases=local_gate.get_runtime_aliases(),
            allowed_capabilities=allowed_capabilities,
        )

    path_ids = [c.cell_id for c in cells]
    return RunResponse(
        status="success",
        prompt=prompt,
        path=path_ids,
        code=code,
        latency_ms=round(total_dt, 2),
        route_latency_ms=round(route_dt, 2),
        synthesis_latency_ms=round(synth_dt, 2),
        total_latency_ms=round(total_dt, 2),
        sandbox_result=sandbox_result
    )


# =====================================================================
# FastAPI Application & Lifespan
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes base lattice and engine on server boot."""
    _ensure_base_engine_loaded()
    initialize_engine("0")
    yield
    logger.info("[*] Shutting down STRATA engine...")
    if _sandbox and hasattr(_sandbox, "_pool") and _sandbox._pool is not None:
        try:
            _sandbox._pool.terminate()
            _sandbox._pool.join(timeout=3)
        except Exception:
            pass
    logger.info("[✓] STRATA engine shutdown complete.")


app = FastAPI(
    title="STRATA: Symbolic Topological Routing Architecture for Typed Automata API",
    description="High-throughput REST API for real-time neuro-symbolic program synthesis.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# API Endpoints
# =====================================================================
@app.get("/api/status", tags=["Status"])
async def get_status():
    """Engine status endpoint polled by frontend and evaluation runners."""
    _ensure_base_engine_loaded()
    nodes_cnt = len(_orchestrator.loaded_cells) if _orchestrator else 0
    return {
        "status": "ready" if _engine_ready else "initializing",
        "profile": _active_profile,
        "embedder": _active_embedder,
        "llm": _active_llm,
        "nodes_count": nodes_cnt,
        "uptime": round(time.time() - _start_time, 2),
        "benchmark_active": _benchmark_active
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Monitoring"])
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """System health check reporting uptime, node count, and cross-platform memory footprint."""
    uptime = time.time() - _start_time
    nodes_count = len(_orchestrator.loaded_cells) if _orchestrator else 0
    rss_mb = _get_process_memory_mb()

    return HealthResponse(
        status="healthy",
        nodes_count=nodes_count,
        uptime_seconds=round(uptime, 2),
        memory_rss_mb=rss_mb
    )


@app.post("/api/initialize", tags=["Lifecycle"])
async def api_initialize(req: InitRequest):
    """Dynamically reconfigures the synthesis profile, embedder, and LLM."""
    result = await asyncio.to_thread(initialize_engine, req)
    return result


@app.post("/api/run", response_model=RunResponse, tags=["Synthesis"])
@app.post("/synthesize", response_model=RunResponse, tags=["Synthesis"])
async def api_run(request: RunRequest):
    """Synthesizes executable code from a natural language prompt."""
    try:
        return await asyncio.to_thread(run_prompt, request)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Backward-compatible API aliases
SynthesizeRequest = RunRequest
synthesize_pipeline = api_run


@app.get("/api/models", tags=["Metadata"])
async def get_models():
    """Scans and returns available local embedding models and GGUF LLMs."""
    emb_dir = os.path.join(MODELS_DIR, "embeddings")
    llm_dir = os.path.join(MODELS_DIR, "llms")

    embeddings = (
        sorted([d for d in os.listdir(emb_dir) if os.path.isdir(os.path.join(emb_dir, d)) and not d.endswith("-GGUF")])
        if os.path.exists(emb_dir) else []
    )
    llms = (
        sorted([d for d in os.listdir(llm_dir) if os.path.isdir(os.path.join(llm_dir, d))])
        if os.path.exists(llm_dir) else []
    )

    optimal_emb = select_optimal_embedder("auto")
    return {
        "embeddings": embeddings,
        "llms": llms,
        "optimal_embedder": optimal_emb,
        "active_profile": _active_profile,
        "active_embedder": _active_embedder,
        "active_llm": _active_llm
    }


@app.get("/api/cells", tags=["Metadata"])
async def get_cells(
    limit: Optional[int] = Query(
        default=settings.default_cells_limit,
        description="Maximum number of cells to return. If omitted or <= 0, returns all cells without arbitrary truncation."
    )
):
    """Returns cell node metadata for Three.js graph visualization without hardcoded domain limits."""
    _ensure_base_engine_loaded()
    if not _orchestrator:
        return {"cells": [], "total": 0}

    all_cells = list(_orchestrator.loaded_cells.values())
    selected_cells = all_cells[:limit] if (limit is not None and limit > 0) else all_cells

    cells_out = []
    for c in selected_cells:
        in_sig = getattr(c, "primary_input", None)
        out_sig = getattr(c, "primary_output", None)
        cells_out.append({
            "id": c.cell_id,
            "domain": c.domain_name or "generic",
            "stage": c.stage,
            "input_type": in_sig.type_name if in_sig else "any",
            "output_type": out_sig.type_name if out_sig else "any",
            "verified": getattr(c, "verified", False)
        })
    return {"cells": cells_out, "total": len(all_cells)}


@app.get("/domains", response_model=DomainsResponse, tags=["Metadata"])
async def get_domains():
    """Returns domain distribution across the active lattice."""
    _ensure_base_engine_loaded()
    if not _orchestrator:
        return DomainsResponse(domains={}, total_domains=0, total_nodes=0)

    domain_stats: Dict[str, Dict[str, int]] = {}

    for cell in _orchestrator.loaded_cells.values():
        dom = cell.domain_name or "generic"
        if dom not in domain_stats:
            domain_stats[dom] = {"total_nodes": 0, "curated_seeds": 0}
        domain_stats[dom]["total_nodes"] += 1
        if getattr(cell, "verified", False):
            domain_stats[dom]["curated_seeds"] += 1

    formatted_domains = {
        dom: DomainInfo(total_nodes=data["total_nodes"], curated_seeds=data["curated_seeds"])
        for dom, data in sorted(domain_stats.items())
    }

    return DomainsResponse(
        domains=formatted_domains,
        total_domains=len(formatted_domains),
        total_nodes=len(_orchestrator.loaded_cells)
    )


@app.post("/api/benchmark/toggle", tags=["Benchmark"])
async def toggle_benchmark():
    """Toggles active benchmark runner state with thread synchronization."""
    global _benchmark_active
    with _benchmark_lock:
        _benchmark_active = not _benchmark_active
        current_state = _benchmark_active
    return {"benchmark_active": current_state}


@app.get("/api/benchmark/status", tags=["Benchmark"])
async def get_benchmark_status():
    """Returns current benchmark execution state."""
    with _benchmark_lock:
        return {"benchmark_active": _benchmark_active}


# =====================================================================
# Server Entrypoint
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    host = settings.api_host
    port = int(os.environ.get("PORT", settings.api_port))
    logger.info(f"[*] Starting STRATA API Server on {host}:{port}...")
    uvicorn.run("main:app", host=host, port=port, reload=False)
