"""
src/config.py — Centralized, zero-hardcode configuration for STRATA
(Symbolic Topological Routing Architecture for Typed Automata).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class ProfileSpec:
    """Single source of truth for orthogonal STRATA research profile capabilities."""
    code: str
    aliases: Tuple[str, ...]
    display_name: str
    short_label: str
    description: str
    pure_symbolic: bool = False
    uses_embedder: bool = False
    uses_reranker: bool = False
    uses_llm: bool = False
    uses_slot_fill: bool = False
    uses_choice_scorer: bool = False
    quick_shortcut: Optional[str] = None


PROFILE_SPECS: Dict[str, ProfileSpec] = {
    "0": ProfileSpec(
        code="0",
        aliases=("0", "SYMBOLIC", "ZERO", "PURE"),
        display_name="Profile 0 (Pure Symbolic)",
        short_label="Symbolic",
        description="Deterministic formal-syntax parsing + exact cell lookup + live type-directed bridging (zero neural models).",
        pure_symbolic=True,
        quick_shortcut="1",
    ),
    "A": ProfileSpec(
        code="A",
        aliases=("A",),
        display_name="Profile A (Dense Embedder Only)",
        short_label="Embedder",
        description="Formal-syntax frontend with dense bi-encoder retrieval over neutral API docs (no reranker, no LLM).",
        uses_embedder=True,
        quick_shortcut="2",
    ),
    "B": ProfileSpec(
        code="B",
        aliases=("B",),
        display_name="Profile B (Dense Embedder + Neural Reranker)",
        short_label="Reranked",
        description="Formal-syntax frontend with dense bi-encoder retrieval + calibrated cross-encoder/listwise reranking (no LLM).",
        uses_embedder=True,
        uses_reranker=True,
        quick_shortcut="3",
    ),
    "C": ProfileSpec(
        code="C",
        aliases=("C",),
        display_name="Profile C (SLM Compiler + Dense Embedder)",
        short_label="SLM+Dense",
        description="Natural-language SLM TypedIR compiler + dense retrieval + type-directed bridging (no slot-fill, no reranker).",
        uses_embedder=True,
        uses_llm=True,
        quick_shortcut="4",
    ),
    "D": ProfileSpec(
        code="D",
        aliases=("D",),
        display_name="Profile D (SLM Compiler + Dense + Slot-Fill)",
        short_label="SLM+SlotFill",
        description="Natural-language SLM TypedIR compiler + dense retrieval + schema-constrained non-asset slot filling.",
        uses_embedder=True,
        uses_llm=True,
        uses_slot_fill=True,
        quick_shortcut="5",
    ),
    "E": ProfileSpec(
        code="E",
        aliases=("E",),
        display_name="Profile E (SLM Compiler + Dense + Reranker + Slot-Fill)",
        short_label="SLM+Rerank+Fill",
        description="Natural-language SLM TypedIR compiler + dense retrieval + neural reranker + constrained slot filling.",
        uses_embedder=True,
        uses_reranker=True,
        uses_llm=True,
        uses_slot_fill=True,
        quick_shortcut="6",
    ),
    "O": ProfileSpec(
        code="O",
        aliases=("O", "OMEGA", "Ω"),
        display_name="Profile Omega (Full STRATA Pipeline)",
        short_label="Omega",
        description="Full STRATA: SLM TypedIR + asymmetric dense retrieval + neural reranking + logprob choice scoring + constrained slot fill.",
        uses_embedder=True,
        uses_reranker=True,
        uses_llm=True,
        uses_slot_fill=True,
        uses_choice_scorer=True,
        quick_shortcut="7",
    ),
}

PROFILE_ALIAS_MAP: Dict[str, str] = {
    alias: spec.code for spec in PROFILE_SPECS.values() for alias in spec.aliases
}


def resolve_profile_spec(profile: Optional[str]) -> ProfileSpec:
    """Resolve a profile identifier or alias to its authoritative ProfileSpec."""
    key = str(profile or "0").strip().upper()
    canonical = PROFILE_ALIAS_MAP.get(key)
    if canonical is None:
        raise ValueError(f"Unknown profile type: {profile}")
    return PROFILE_SPECS[canonical]


def active_profile_spec() -> ProfileSpec:
    """Return the currently active ProfileSpec from environment or settings."""
    env_p = os.environ.get("NSTL_PROFILE")
    raw = env_p if env_p is not None and env_p.strip() else settings.profile
    return resolve_profile_spec(raw or "0")


class NSTLSettings(BaseSettings):
    """All STRATA configuration values, resolved from environment variables with NSTL_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="NSTL_",
        env_file=(str(Path(__file__).resolve().parent.parent / ".env"), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    trees_dir: Optional[Path] = None
    logs_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    rerankers_dir: Optional[Path] = None

    # Sandbox
    sandbox_enabled: bool = Field(default=True, description="Whether sandbox execution isolation is enabled.")
    sandbox_timeout: float = Field(default=30.0, gt=0, description="Sandbox execution timeout in seconds.")
    sandbox_max_memory_mb: int = Field(default=8192, ge=64, description="Sandbox POSIX memory limit in MB.")
    sandbox_max_cpu_seconds: int = Field(default=30, ge=1, description="Sandbox CPU runtime limit in seconds.")
    smoke_test_loop_bound: int = Field(default=2, ge=1, description="Loop unroll bound used by static sandbox smoke tests.")

    # Inference (discovered dynamically from models/ unless configured via env or CLI)
    profile: Optional[str] = Field(default="0", description="Active profile identifier (0, A, B, C, D, E, O).")
    embedder_model: Optional[str] = Field(default=None, description="Configured text embedding model identifier.")
    embedder_task: Optional[str] = Field(default=None, description="Optional SentenceTransformer default task prompt name.")
    llm_model: Optional[str] = Field(default=None, description="Configured GGUF LLM model identifier.")
    reranker_model: Optional[str] = Field(default=None, description="Configured neural reranker identifier.")
    use_reranker: bool = Field(default=False, description="Whether neural reranking is explicitly forced on/off.")
    preferred_embedders: List[str] = Field(default_factory=list, description="Optional preference order for embedder discovery.")
    preferred_llms: List[str] = Field(default_factory=list, description="Optional preference order for LLM discovery.")
    preferred_rerankers: List[str] = Field(default_factory=list, description="Optional preference order for reranker discovery.")
    llm_context_length: int = Field(default=4096, ge=512)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    llm_repeat_penalty: float = Field(default=1.1, ge=0.5, le=2.0)
    llm_max_tokens: int = Field(default=1024, ge=64)
    llm_stop_tokens: List[str] = Field(default_factory=list, description="Optional extra stop sequences for LLM generation.")
    max_embedding_cache_size: int = Field(default=16384, ge=128)

    # Retrieval & Type-Directed Planning
    retrieval_domain_hint_bonus: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Optional additive similarity tie-breaker when subgoal domain_hint matches candidate domain.",
    )
    max_bridge_hops: int = Field(default=3, ge=0, le=8)
    ambiguity_temperature: float = Field(default=0.10, gt=0.0)
    ambiguity_entropy_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    ambiguity_margin_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        if self.trees_dir is None:
            self.trees_dir = self.project_root / "trees"
        if self.logs_dir is None:
            self.logs_dir = self.project_root / "logs"
        if self.models_dir is None:
            self.models_dir = self.project_root / "models"
        if self.rerankers_dir is None:
            self.rerankers_dir = (self.models_dir or (self.project_root / "models")) / "rerankers"


settings = NSTLSettings()
STRATASettings = NSTLSettings

PROJECT_ROOT = settings.project_root
TREES_DIR = str(settings.trees_dir)
LOGS_DIR = str(settings.logs_dir)
MODELS_DIR = str(settings.models_dir)
RERANKERS_DIR = str(settings.rerankers_dir)
USE_RERANKER = settings.use_reranker
RERANKER_MODEL = settings.reranker_model
