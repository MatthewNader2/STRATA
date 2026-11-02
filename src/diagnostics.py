"""
src/diagnostics.py - STRATA (Neuro-Symbolic Topological Lattice)
Track 3: Diagnostic Telemetry & Failure Surface Collector.

Records caught exceptions, unification refusals, and fallback invocations across
planning, unification, and CLI execution so zero errors are silently swallowed.
"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from log_config import get_logger

logger = get_logger("diagnostics")


@dataclass
class DiagnosticEvent:
    kind: str  # "exception" | "warning" | "fallback" | "refusal"
    site: str
    error_type: str
    message: str
    cell_id: Optional[str] = None
    port_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class DiagnosticTelemetry:
    """
    Thread-safe diagnostic collector attached to ExecutionContext (`ctx.diagnostics`).
    Replaces silent `except Exception: pass` blocks with structured telemetry.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.events: List[DiagnosticEvent] = []

    def record_exception(
        self,
        site: str,
        exc: BaseException,
        *,
        cell_id: Optional[str] = None,
        port_name: Optional[str] = None,
        level: str = "debug",
    ) -> None:
        err_type = type(exc).__name__
        msg = str(exc)
        with self._lock:
            self.events.append(
                DiagnosticEvent(
                    kind="exception",
                    site=site,
                    error_type=err_type,
                    message=msg,
                    cell_id=cell_id,
                    port_name=port_name,
                )
            )
        log_fn = getattr(logger, level, logger.debug)
        log_fn(
            "[DIAGNOSTIC:%s] %s: %s (cell=%s, port=%s)",
            site,
            err_type,
            msg,
            cell_id or "-",
            port_name or "-",
            exc_info=(level in ("error", "warning")),
        )

    def record_warning(
        self,
        site: str,
        message: str,
        *,
        cell_id: Optional[str] = None,
        port_name: Optional[str] = None,
    ) -> None:
        with self._lock:
            self.events.append(
                DiagnosticEvent(
                    kind="warning",
                    site=site,
                    error_type="Warning",
                    message=message,
                    cell_id=cell_id,
                    port_name=port_name,
                )
            )
        logger.debug("[DIAGNOSTIC:WARN:%s] %s", site, message)

    def record_fallback(
        self,
        site: str,
        from_stage: str,
        to_stage: str,
        reason: str,
    ) -> None:
        with self._lock:
            self.events.append(
                DiagnosticEvent(
                    kind="fallback",
                    site=site,
                    error_type=f"{from_stage}->{to_stage}",
                    message=reason,
                )
            )
        logger.info("[DIAGNOSTIC:FALLBACK:%s] %s -> %s (%s)", site, from_stage, to_stage, reason)

    def record_refusal(
        self,
        site: str,
        reason: str,
        *,
        cell_id: Optional[str] = None,
        port_name: Optional[str] = None,
    ) -> None:
        with self._lock:
            self.events.append(
                DiagnosticEvent(
                    kind="refusal",
                    site=site,
                    error_type="Refusal",
                    message=reason,
                    cell_id=cell_id,
                    port_name=port_name,
                )
            )

    def merge(self, other: Optional["DiagnosticTelemetry"]) -> None:
        if other is None or other is self:
            return
        with self._lock:
            self.events.extend(other.events)

    def clear(self) -> None:
        with self._lock:
            self.events.clear()

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            counts: Dict[str, int] = {"exception": 0, "warning": 0, "fallback": 0, "refusal": 0}
            by_site: Dict[str, int] = {}
            for ev in self.events:
                counts[ev.kind] = counts.get(ev.kind, 0) + 1
                by_site[ev.site] = by_site.get(ev.site, 0) + 1
            return {
                "total_events": len(self.events),
                "counts": counts,
                "by_site": by_site,
                "events": [
                    {
                        "kind": e.kind,
                        "site": e.site,
                        "error_type": e.error_type,
                        "message": e.message,
                        "cell_id": e.cell_id,
                        "port_name": e.port_name,
                    }
                    for e in self.events
                ],
            }


# Global session fallback collector when no ExecutionContext is in scope
GLOBAL_DIAGNOSTICS = DiagnosticTelemetry()


def record_diagnostic_exception(
    site: str,
    exc: BaseException,
    ctx: Optional[Any] = None,
    cell_id: Optional[str] = None,
    port_name: Optional[str] = None,
) -> None:
    diag = getattr(ctx, "diagnostics", None) if ctx is not None else None
    target = diag if isinstance(diag, DiagnosticTelemetry) else GLOBAL_DIAGNOSTICS
    target.record_exception(site, exc, cell_id=cell_id, port_name=port_name)


__all__ = [
    "DiagnosticEvent",
    "DiagnosticTelemetry",
    "GLOBAL_DIAGNOSTICS",
    "record_diagnostic_exception",
]
