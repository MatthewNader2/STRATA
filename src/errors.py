"""
src/errors.py - STRATA Structured Error Hierarchy
Provides typed exceptions for clear error propagation instead of silent `except: pass`.
"""
from typing import Any, Dict, Optional


class STRATAError(Exception):
    """Base exception for all STRATA errors with structured metadata support."""

    def __init__(
        self,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        *args: Any,
    ) -> None:
        self.message = message or self.__doc__ or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message, *args)

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


# Backward compatibility alias
NSTLError = STRATAError


# === Routing Errors ===
class RoutingError(STRATAError):
    """Failed to find a valid path through the lattice."""
    pass


# === Synthesis Errors ===
class SynthesisError(NSTLError):
    """Failed to synthesize code from lattice cells."""
    pass


class UnificationError(SynthesisError):
    """Typestate unification failed during code emission."""
    pass


class PlaceholderResolutionError(SynthesisError):
    """A code template placeholder could not be resolved."""
    pass


class TemplateValidationError(SynthesisError):
    """A synthesized code template failed AST validation."""
    pass


# === Execution Errors ===
class ExecutionError(NSTLError):
    """Sandbox execution failed."""
    pass


class SandboxTimeoutError(ExecutionError):
    """Execution exceeded the configured time limit."""
    pass


class SandboxSecurityError(ExecutionError):
    """Execution attempted to use a blocked resource."""
    pass


class DataflowExecutionError(ExecutionError):
    """Pipeline terminal variable is missing, None, or guarded failure occurred."""
    pass


class ArtifactMaterializationError(ExecutionError):
    """Egress destination artifact was not materialized on disk or has 0 bytes."""
    pass


class PostconditionVerificationError(ExecutionError):
    """Pipeline postcondition check failed: runtime state or artifact does not match intent."""
    pass


# === Model Errors ===
class ModelError(NSTLError):
    """Model loading or inference failed."""
    pass


class ModelNotLoadedError(ModelError):
    """No model profile is currently active."""
    pass


class EmbeddingError(ModelError):
    """Embedding generation failed."""
    pass


class LLMInferenceError(ModelError):
    """LLM text generation failed."""
    pass


# === RAG Errors ===
class RAGError(NSTLError):
    """Retrieval-Augmented Generation failed."""
    pass


# === Configuration Errors ===
class ConfigurationError(NSTLError):
    """Configuration validation failed."""
    pass


# === Lattice Errors ===
class LatticeError(NSTLError):
    """Lattice database or topology operation failed."""
    pass


# Backward compatibility mapping for disconnected/deprecated exceptions
_DEPRECATED_EXCEPTIONS: Dict[str, type] = {
    "InsufficientCandidatesError": RoutingError,
    "NoPathFoundError": RoutingError,
    "FetchError": RAGError,
    "RetrievalIndexError": RAGError,
}


def __getattr__(name: str) -> Any:
    """Resolve unraised/deprecated exceptions with a warning to preserve legacy import contracts."""
    if name in _DEPRECATED_EXCEPTIONS:
        import warnings

        target = _DEPRECATED_EXCEPTIONS[name]
        warnings.warn(
            f"'{name}' is deprecated and disconnected from the runtime pipeline; "
            f"use or catch '{target.__name__}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return type(name, (target,), {
            "__doc__": f"Deprecated alias for {target.__name__} (disconnected from runtime pipeline).",
        })
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "STRATAError",
    "NSTLError",
    "RoutingError",
    "SynthesisError",
    "UnificationError",
    "PlaceholderResolutionError",
    "TemplateValidationError",
    "ExecutionError",
    "SandboxTimeoutError",
    "SandboxSecurityError",
    "DataflowExecutionError",
    "ArtifactMaterializationError",
    "PostconditionVerificationError",
    "ModelError",
    "ModelNotLoadedError",
    "EmbeddingError",
    "LLMInferenceError",
    "RAGError",
    "ConfigurationError",
    "LatticeError",
]
