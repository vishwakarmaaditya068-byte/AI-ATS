from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.data.sql.base import SQLBase

if TYPE_CHECKING:
    from src.data.sql.models.workspace import Workspace


class AuditRecord(SQLBase):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    workspace_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
    )

    action: Mapped[str] = mapped_column(String(64), nullable=False)
    action_description: Mapped[str] = mapped_column(Text, nullable=False)

    actor: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    resource: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    changes: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )
    ai_decision: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    bias_audit: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    context: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )

    # Cross-store references — kept flat for simple filtering.
    related_candidate_mongo_id: Mapped[Optional[str]] = mapped_column(String(24), nullable=True)
    related_job_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    related_match_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)

    compliance_relevant: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    retention_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=2555, server_default="2555"
    )

    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    workspace: Mapped[Optional["Workspace"]] = relationship(
        "Workspace", back_populates="audit_records"
    )

    __table_args__ = (
        Index("ix_audit_workspace_occurred", "workspace_id", "occurred_at"),
        Index("ix_audit_action_occurred", "action", "occurred_at"),
        Index("ix_audit_compliance", "compliance_relevant", "occurred_at"),
        Index("ix_audit_related_candidate", "related_candidate_mongo_id"),
        Index("ix_audit_related_job", "related_job_id"),
        Index("ix_audit_related_match", "related_match_id"),
    )

    def __repr__(self) -> str:
        return (
            f"AuditRecord(id={self.id}, action={self.action!r}, "
            f"workspace_id={self.workspace_id})"
        )
