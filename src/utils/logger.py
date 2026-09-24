"""
Logging infrastructure for AI-ATS application.

Uses Loguru for powerful, easy-to-use logging with automatic rotation,
structured output, and beautiful formatting.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.config import get_settings


def setup_logging() -> None:
    """
    Configure application-wide logging.

    Sets up both console and file logging with appropriate formatting,
    rotation, and retention policies.
    """
    settings = get_settings()
    log_settings = settings.logging

    # Remove default handler
    logger.remove()

    # Console handler with colored output
    # Security: diagnose=False in production to prevent sensitive data leakage in stack traces
    enable_diagnose = settings.debug and settings.environment == "development"

    if log_settings.console_output:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>",
            level=log_settings.level,
            colorize=True,
            backtrace=True,
            diagnose=enable_diagnose,
        )

    # File handler with rotation
    log_file = log_settings.file_path
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format=log_settings.format,
        level=log_settings.level,
        rotation=log_settings.rotation,
        retention=log_settings.retention,
        compression="zip",
        backtrace=True,
        diagnose=enable_diagnose,  # Security: Only in development mode
        enqueue=True,  # Thread-safe logging
    )

    # Add audit log for ethical AI compliance
    audit_log_path = log_file.parent / "audit.log"
    logger.add(
        audit_log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra[audit_type]} | {message}",
        level="INFO",
        filter=lambda record: "audit_type" in record["extra"],
        rotation="1 week",
        retention="1 year",  # Keep audit logs longer for compliance
        compression="zip",
        enqueue=True,
    )

    logger.info(f"Logging initialized - Level: {log_settings.level}")


def get_logger(name: str) -> Any:
    """
    Get a logger instance with the specified name.

    Args:
        name: The name for the logger (typically __name__)

    Returns:
        A configured logger instance
    """
    return logger.bind(name=name)


def _sanitize_for_logging(data: Any) -> Any:
    """
    Sanitize data before logging to prevent sensitive information exposure.

    Redacts passwords, tokens, and other sensitive fields.
    """
    if isinstance(data, dict):
        sensitive_keys = {
            "password", "passwd", "pwd", "secret", "token", "api_key",
            "apikey", "auth", "credential", "private_key", "access_token",
            "refresh_token", "ssn", "social_security",
        }
        return {
            k: "***REDACTED***" if any(s in k.lower() for s in sensitive_keys) else _sanitize_for_logging(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [_sanitize_for_logging(item) for item in data]
    return data


def audit_log(
    action: str,
    details: dict[str, Any],
    audit_type: str = "DECISION",
) -> None:
    """
    Log an audit entry for ethical AI compliance.

    Args:
        action: The action being audited (e.g., "candidate_scored", "bias_detected")
        details: Dictionary of relevant details
        audit_type: Type of audit entry (DECISION, BIAS, OVERRIDE, ACCESS)
    """
    sanitized_details = _sanitize_for_logging(details)
    logger.bind(audit_type=audit_type).info(f"{action} | {sanitized_details}")


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.

    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("Doing something...")
    """

    @property
    def logger(self) -> Any:
        """Get a logger instance for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


# Module-level logger for quick access
log = logger


# Auto-setup on import if settings are available
try:
    setup_logging()
except Exception:
    # If setup fails (e.g., during import), use default logger
    pass
