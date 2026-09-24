from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.data.sql.base import SQLBase, TimestampMixin

if TYPE_CHECKING:
    from src.data.sql.models.job_record import JobRecord


class MatchRecord(SQLBase, TimestampMixin):
    __tablename__ = "matches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    candidate_mongo_id: Mapped[str] = mapped_column(
        String(24),
        nullable=False,
    )

    # Scoring — overall + per-component as normalized [0..1] floats.
    overall_score: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    skills_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 4), nullable=True)
    experience_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 4), nullable=True)
    education_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 4), nullable=True)
    semantic_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 4), nullable=True)
    keyword_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 4), nullable=True)

    score_level: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    # Manual override — null means no override has been applied.
    manual_score_override: Mapped[Optional[float]] = mapped_column(Numeric(5, 4), nullable=True)
    override_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    overridden_by: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Rich payloads kept as JSONB for flexibility.
    explanation: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    bias_check: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    score_breakdown: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )

    scoring_model_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="1.0", server_default="1.0"
    )

    job: Mapped["JobRecord"] = relationship("JobRecord", back_populates="matches")

    __table_args__ = (
        UniqueConstraint("job_id", "candidate_mongo_id", name="uq_match_job_candidate"),
        CheckConstraint(
            "overall_score >= 0 AND overall_score <= 1",
            name="ck_match_overall_score_range",
        ),
        CheckConstraint(
            "manual_score_override IS NULL OR "
            "(manual_score_override >= 0 AND manual_score_override <= 1)",
            name="ck_match_override_score_range",
        ),
        Index("ix_matches_job_id_overall_score", "job_id", "overall_score"),
        Index("ix_matches_candidate_mongo_id", "candidate_mongo_id"),
    )

    @property
    def effective_score(self) -> float:
        """Override takes precedence over AI-computed overall_score."""
        if self.manual_score_override is not None:
            return float(self.manual_score_override)
        return float(self.overall_score)

    def __repr__(self) -> str:
        return (
            f"MatchRecord(id={self.id}, job_id={self.job_id}, "
            f"candidate={self.candidate_mongo_id!r}, score={float(self.overall_score):.3f})"
        )
