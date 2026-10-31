"""
src/config.py - Centralized Configuration for NSTL
Uses Pydantic BaseSettings for validated, type-safe configuration with .env file support.
"""
import json
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class NSTLSettings(BaseSettings):
    """All NSTL configuration values, resolved from environment variables with NSTL_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="NSTL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    trees_dir: Optional[Path] = None
    harvests_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    rerankers_dir: Optional[Path] = None
    reranker_model: Optional[str] = Field(default=None, description="Optional reranker model identifier.")
    use_reranker: bool = Field(default=False, description="Whether neural reranking is enabled in Layer 1.")
    allow_fallback: bool = Field(default=False, description="Whether route methods may fall back to M1 on failure.")

    # Server & Networking
    api_host: str = Field(default="127.0.0.1", description="Host interface to bind the API server.")
    api_port: int = Field(default=58102, ge=1, le=65535, description="Port to bind the API server.")
    cors_origins: List[str] = Field(
        default_factory=list,
        description="Allowed CORS origins. Configurable via NSTL_CORS_ORIGINS (comma-separated or JSON list).",
    )

    # Sandbox
    sandbox_enabled: bool = Field(default=True, description="Whether sandbox isolation is enabled.")
    sandbox_timeout: float = Field(default=30.0, gt=0, description="Sandbox timeout in seconds.")
    sandbox_workers: int = Field(default=2, ge=1, description="Number of sandbox worker processes.")
    sandbox_max_memory_mb: int = Field(default=8192, ge=64, description="Sandbox memory limit in MB.")
    sandbox_max_cpu_seconds: int = Field(default=30, ge=1, description="Sandbox CPU runtime limit in seconds.")

    # Monte Carlo Tree Search (src/mcts.py): used by the M10 route method and by the M0 planner's fallback.
    mcts_iterations: int = Field(default=64, ge=1, description="Selection/expansion/rollout/backprop iterations per search.")
    mcts_exploration: float = Field(default=1.4142135623730951, ge=0, description="UCT exploration constant C (sqrt(2) is the classical choice).")
    mcts_widening_alpha: float = Field(default=0.5, ge=0, le=1, description="Progressive widening: a node with n visits may expand ceil(n**alpha) children.")
    mcts_seed: int = Field(default=0, description="RNG seed. Searches are deterministic for a given seed; change it to sample other plans.")
    mcts_max_seconds: float = Field(default=20.0, gt=0, description="Wall-clock budget per search (the search stops early and returns its best plan).")
    mcts_max_branch: int = Field(default=4, ge=1, description="Alternative cells considered per clause by the M10 route method.")
    mcts_max_bridges: int = Field(default=2, ge=0, description="Maximum property/type bridge cells M10 may insert.")
    mcts_rollout_temperature: float = Field(default=0.1, gt=0, description="Softmax temperature of the rollout policy (lower = rollouts follow the heuristic prior more closely).")

    # Inference & Similarity
    embedder_model: str = Field(default="jina-embeddings-v5-text-nano", description="Default text embedding model identifier.")
    llm_model: str = Field(default="qwen2.5-coder-0.5b-instruct", description="Default GGUF LLM model identifier.")
    llm_context_length: int = Field(default=4096, ge=512)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    similarity_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Threshold for semantic candidate similarity matching.",
    )

    # Macro-goal routing & Topology Planning
    profile: Optional[str] = Field(default="0", description="Active profile identifier (0, A, B, C, D, E, S, U, AE).")
    macros_enabled: bool = True
    topology_mode: str = "frontier"
    use_ir_compiler: bool = False
    require_coverage_floor: bool = True
    coverage_floor_fraction: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Fraction of clauses required to be covered when require_coverage_floor is active.",
    )
    dev_mode: bool = False
    planner_config: Optional[Path] = None
    planner_time_budget_ms: float = Field(
        default=25000.0,
        ge=250.0,
        description="Wall-clock budget for a single planning search. When expired, "
        "the beam returns the best-so-far valid paths instead of running unbounded.",
    )
    planner_greedy_budget_ms: float = Field(
        default=1000.0,
        ge=100.0,
        description="Wall-clock budget for greedy completion phase in planner.",
    )
    explain_plan: bool = Field(
        default=False,
        description="When True, prints diagnostic table of cell-clause precision and coverage.",
    )
    domain_affinity_bonus: float = Field(
        default=0.5,
        ge=0.0,
        description="Score bonus for cells in the primary domain.",
    )
    max_plan_depth: int = Field(
        default=24,
        ge=1,
        description="Maximum plan depth during chain expansion.",
    )

    # Search & Routing Bounds
    bridge_bfs_max_expansions: int = Field(
        default=300,
        ge=10,
        description="Maximum BFS expansions during shortest bridge path search in LatticeOrchestrator.",
    )
    m2_max_candidate_transforms: int = Field(
        default=25,
        ge=1,
        description="Maximum transform candidates evaluated in M2 endpoint routing.",
    )
    macro_router_max_depth: int = Field(
        default=2,
        ge=1,
        description="Maximum recursion depth for macro-goal unfolding in router.",
    )
    bridge_chain_max_hops: int = Field(
        default=2,
        ge=1,
        description="Maximum intermediate hops allowed when bridging macro sub-chains.",
    )
    function_words_cutoff: float = Field(
        default=0.10,
        gt=0.0,
        le=1.0,
        description="Frequency threshold above which corpus words are classified as non-informative function words.",
    )
    secondary_anchor_min_recall: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum recall threshold for secondary clause anchors in token evidence audit.",
    )
    secondary_anchor_min_f: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Minimum F-score threshold for secondary clause anchors.",
    )
    secondary_anchor_min_name_precision: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum name-precision threshold for secondary clause anchors against residual clause tokens.",
    )
    secondary_anchor_f_ratio: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Minimum ratio of secondary to primary F-score for anchor acceptance.",
    )
    max_embedding_cache_size: int = Field(
        default=16384,
        ge=128,
        description="Maximum number of entries retained in the global embedding cache.",
    )
    prune_unrequested_max_ops: int = Field(
        default=50,
        ge=1,
        description="Operation-count budget for prune_unrequested passes before returning best-so-far pipeline.",
    )
    verified_priority_threshold: int = Field(
        default=10,
        ge=1,
        description="Priority threshold for verified edges in CLI and router.",
    )
    latent_sim_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    latent_progress_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    latent_clause_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    latent_goal_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    latent_structural_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    latent_state_momentum: float = Field(default=0.7, ge=0.0, le=1.0)
    default_cells_limit: Optional[int] = Field(
        default=None,
        description="Default cell limit for API extraction.",
    )
    clause_evidence_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    clause_weight_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    clause_precision_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    join_relevance_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: Any) -> List[str]:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                import json
                try:
                    return json.loads(stripped)
                except Exception:
                    pass
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return value or []

    def model_post_init(self, __context: Any) -> None:
        """Resolve default paths and dynamic defaults after construction."""
        if self.trees_dir is None:
            self.trees_dir = self.project_root / "trees"
        if self.harvests_dir is None:
            self.harvests_dir = self.project_root / "harvests"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        if self.models_dir is None:
            self.models_dir = self.project_root / "models"
        if self.rerankers_dir is None:
            self.rerankers_dir = (self.models_dir or (self.project_root / "models")) / "rerankers"

        if getattr(self, "planner_config", None) is None:
            for cand in (
                self.project_root / "config" / "planner.yaml",
                self.project_root / "config" / "planner.json",
            ):
                if cand.exists():
                    self.planner_config = cand
                    break

        if not self.cors_origins:
            self.cors_origins = [
                f"http://{self.api_host}:{self.api_port}",
                f"http://localhost:{self.api_port}",
                f"http://127.0.0.1:{self.api_port}",
            ]


# Singleton instance — importable as `from config import settings`
settings = NSTLSettings()

# Backward-compatible aliases (preserves existing import contracts)
PROJECT_ROOT = settings.project_root
TREES_DIR = str(settings.trees_dir)
HARVESTS_DIR = str(settings.harvests_dir)
LOGS_DIR = str(settings.logs_dir)
MODELS_DIR = str(settings.models_dir)
RERANKERS_DIR = str(settings.rerankers_dir)
USE_RERANKER = settings.use_reranker
RERANKER_MODEL = settings.reranker_model
API_HOST = settings.api_host
API_PORT = settings.api_port
CORS_ORIGINS = settings.cors_origins
SIMILARITY_THRESHOLD = settings.similarity_threshold
ALLOW_FALLBACK = settings.allow_fallback
