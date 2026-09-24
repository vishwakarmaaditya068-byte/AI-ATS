from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    """Timezone-aware UTC now — used as Python-side default for timestamps."""
    return datetime.now(timezone.utc)


class SQLBase(DeclarativeBase):
    """Root DeclarativeBase for all Postgres-backed models."""


class TimestampMixin:
    """Adds created_at / updated_at columns managed by both DB and Python."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=utc_now,
        onupdate=utc_now,
    )
