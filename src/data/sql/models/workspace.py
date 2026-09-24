from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, Enum as SAEnum, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.data.sql.base import SQLBase, TimestampMixin

if TYPE_CHECKING:
    from src.data.sql.models.audit_record import AuditRecord
    from src.data.sql.models.job_record import JobRecord


class WorkspaceStatus(str, enum.Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    PURGED = "purged"


class Workspace(SQLBase, TimestampMixin):
    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[WorkspaceStatus] = mapped_column(
        SAEnum(WorkspaceStatus, name="workspace_status", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=WorkspaceStatus.ACTIVE,
        server_default=WorkspaceStatus.ACTIVE.value,
    )
    created_by: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    archived_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_opened_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    jobs: Mapped[list["JobRecord"]] = relationship(
        "JobRecord",
        back_populates="workspace",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    audit_records: Mapped[list["AuditRecord"]] = relationship(
        "AuditRecord",
        back_populates="workspace",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_workspaces_status", "status"),
        Index("ix_workspaces_last_opened_at", "last_opened_at"),
    )

    def __repr__(self) -> str:
        return f"Workspace(id={self.id}, name={self.name!r}, status={self.status.value})"
