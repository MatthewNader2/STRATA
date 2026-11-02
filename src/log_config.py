"""
STRATA Centralized Logging Configuration
Provides structured logging with flexible formatting, configurable file + console output,
and per-component loggers for deterministic debugging and telemetry.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Sequence, Union

_DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

DEFAULT_SILENCED_LOGGERS = (
    "uvicorn.access",
    "uvicorn.error",
    "httpx",
    "httpcore",
    "sentence_transformers",
    "transformers",
    "filelock",
    "urllib3",
)

DEFAULT_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DEFAULT_FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_INITIALIZED = False


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = "nstl.log",
    log_dir: Optional[Union[str, Path]] = None,
    console_format: Optional[str] = None,
    file_format: Optional[str] = None,
    datefmt: Optional[str] = None,
    silenced_loggers: Optional[Sequence[str]] = None,
    extra_silenced_loggers: Optional[Sequence[str]] = None,
    silence_level: int = logging.WARNING,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    force: bool = False,
) -> None:
    """
    Initialize the STRATA logging system.
    Call once at application startup.

    Resolves paths, log levels, formats, and silenced third-party modules
    via explicit arguments with fallback to environment variables.
    """
    global _INITIALIZED
    if _INITIALIZED and not force:
        return

    # Resolve log level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    env_level = os.environ.get("NSTL_LOG_LEVEL")
    if env_level:
        level = getattr(logging, env_level.upper(), level)

    # Format strings
    c_fmt = (
        console_format
        or os.environ.get("NSTL_CONSOLE_FORMAT")
        or os.environ.get("NSTL_LOG_FORMAT")
        or DEFAULT_CONSOLE_FORMAT
    )
    f_fmt = (
        file_format
        or os.environ.get("NSTL_FILE_FORMAT")
        or os.environ.get("NSTL_LOG_FORMAT")
        or DEFAULT_FILE_FORMAT
    )
    d_fmt = datefmt or os.environ.get("NSTL_DATE_FORMAT") or DEFAULT_DATE_FORMAT

    root = logging.getLogger()
    root.setLevel(level)

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(c_fmt, datefmt=d_fmt))
    root.addHandler(console)

    # Determine file logging target
    file_logging_enabled = os.environ.get("NSTL_LOG_TO_FILE", "1").lower() not in ("0", "false", "no", "off")
    resolved_log_path: Optional[str] = None

    if file_logging_enabled and log_file:
        env_log_path = os.environ.get("NSTL_LOG_PATH")
        if env_log_path:
            resolved_log_path = env_log_path
        else:
            resolved_dir = (
                str(log_dir)
                if log_dir is not None
                else os.environ.get("NSTL_LOG_DIR")
                or os.environ.get("NSTL_LOGS_DIR")
                or str(_DEFAULT_LOG_DIR)
            )
            resolved_file = os.environ.get("NSTL_LOG_FILE") or str(log_file)
            if os.path.isabs(resolved_file):
                resolved_log_path = resolved_file
            else:
                os.makedirs(resolved_dir, exist_ok=True)
                resolved_log_path = os.path.join(resolved_dir, resolved_file)

        # Rotating file handler
        file_handler = RotatingFileHandler(
            resolved_log_path,
            maxBytes=int(os.environ.get("NSTL_LOG_MAX_BYTES", max_bytes)),
            backupCount=int(os.environ.get("NSTL_LOG_BACKUP_COUNT", backup_count)),
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(f_fmt, datefmt=d_fmt))
        root.addHandler(file_handler)

    # Resolve silenced third-party loggers
    targets = set(DEFAULT_SILENCED_LOGGERS if silenced_loggers is None else silenced_loggers)
    if extra_silenced_loggers:
        targets.update(extra_silenced_loggers)

    env_silenced = os.environ.get("NSTL_SILENCED_LOGGERS")
    if env_silenced:
        targets.update([s.strip() for s in env_silenced.split(",") if s.strip()])

    for noisy in targets:
        logging.getLogger(noisy).setLevel(silence_level)

    _INITIALIZED = True

    init_msg = f"STRATA logging initialized (level={logging.getLevelName(level)})"
    if resolved_log_path:
        init_msg += f", File: {resolved_log_path}"
    logging.getLogger("strata").info(init_msg)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a namespaced STRATA logger.
    Usage: logger = get_logger("router") -> creates logger "strata.router"
    """
    if name.startswith("strata."):
        return logging.getLogger(name)
    if name.startswith("nstl."):
        return logging.getLogger(name)
    return logging.getLogger(f"strata.{name}")
