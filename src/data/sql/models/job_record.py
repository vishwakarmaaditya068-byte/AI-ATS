from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Enum as SAEnum, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.data.sql.base import SQLBase, TimestampMixin

if TYPE_CHECKING:
    from src.data.sql.models.match_record import MatchRecord
    from src.data.sql.models.workspace import Workspace


class JobStatus(str, enum.Enum):
    DRAFT = "draft"
    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"
    FILLED = "filled"


class JobRecord(SQLBase, TimestampMixin):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )
    mongo_doc_id: Mapped[Optional[str]] = mapped_column(
        String(24),
        nullable=True,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    company_name: Mapped[Optional[str]] = mapped_column(String(300), nullable=True)
    description_snippet: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, name="job_status", values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=JobStatus.OPEN,
        server_default=JobStatus.OPEN.value,
    )

    workspace: Mapped["Workspace"] = relationship("Workspace", back_populates="jobs")
    matches: Mapped[list["MatchRecord"]] = relationship(
        "MatchRecord",
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_jobs_workspace_id_status", "workspace_id", "status"),
        Index("ix_jobs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"JobRecord(id={self.id}, title={self.title!r}, status={self.status.value})"
