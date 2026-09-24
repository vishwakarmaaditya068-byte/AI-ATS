"""
Workspace repository — CRUD plus lifecycle transitions (archive / purge).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select, update as sa_update

from src.data.sql.models.workspace import Workspace, WorkspaceStatus
from src.data.sql.repositories.base import BaseSQLRepository


class WorkspaceRepository(BaseSQLRepository[Workspace]):
    @property
    def model_class(self) -> type[Workspace]:
        return Workspace

    # Queries
    def list_recent(self, limit: int = 10) -> list[Workspace]:
        """Workspaces sorted by last-opened (most recent first).

        Used to populate the startup picker. Excludes purged workspaces.
        """
        with self._sql.sync_session() as session:
            stmt = (
                select(Workspace)
                .where(Workspace.status != WorkspaceStatus.PURGED)
                .order_by(func.coalesce(Workspace.last_opened_at, Workspace.created_at).desc())
                .limit(limit)
            )
            result = session.execute(stmt).scalars().all()
            for ws in result:
                session.expunge(ws)
            return list(result)

    def list_by_status(self, status: WorkspaceStatus) -> list[Workspace]:
        with self._sql.sync_session() as session:
            stmt = (
                select(Workspace)
                .where(Workspace.status == status)
                .order_by(Workspace.created_at.desc())
            )
            result = session.execute(stmt).scalars().all()
            for ws in result:
                session.expunge(ws)
            return list(result)

    def find_by_name(self, name: str) -> Optional[Workspace]:
        with self._sql.sync_session() as session:
            stmt = select(Workspace).where(Workspace.name == name).limit(1)
            ws: Optional[Workspace] = session.execute(stmt).scalars().first()
            if ws is not None:
                session.expunge(ws)
            return ws

    # Lifecycle transitions
    def create_workspace(
        self,
        name: str,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Workspace:
        workspace = Workspace(
            name=name,
            description=description,
            created_by=created_by,
            status=WorkspaceStatus.ACTIVE,
        )
        return self.create(workspace)

    def touch(self, id_value: uuid.UUID | str) -> Optional[Workspace]:
        """Update last_opened_at — called when the user switches into it."""
        pk = self._coerce_uuid(id_value)
        with self._sql.sync_session() as session:
            stmt = (
                sa_update(Workspace)
                .where(Workspace.id == pk)
                .values(last_opened_at=datetime.now(timezone.utc))
                .returning(Workspace)
            )
            ws: Optional[Workspace] = session.execute(stmt).scalars().first()
            if ws is not None:
                session.expunge(ws)
            return ws

    def archive(self, id_value: uuid.UUID | str) -> Optional[Workspace]:
        pk = self._coerce_uuid(id_value)
        with self._sql.sync_session() as session:
            stmt = (
                sa_update(Workspace)
                .where(Workspace.id == pk)
                .values(
                    status=WorkspaceStatus.ARCHIVED,
                    archived_at=datetime.now(timezone.utc),
                )
                .returning(Workspace)
            )
            ws: Optional[Workspace] = session.execute(stmt).scalars().first()
            if ws is not None:
                session.expunge(ws)
            return ws

    def restore(self, id_value: uuid.UUID | str) -> Optional[Workspace]:
        """Archived → Active. No-op on purged or already active workspaces."""
        pk = self._coerce_uuid(id_value)
        with self._sql.sync_session() as session:
            stmt = (
                sa_update(Workspace)
                .where(
                    Workspace.id == pk,
                    Workspace.status == WorkspaceStatus.ARCHIVED,
                )
                .values(status=WorkspaceStatus.ACTIVE, archived_at=None)
                .returning(Workspace)
            )
            ws: Optional[Workspace] = session.execute(stmt).scalars().first()
            if ws is not None:
                session.expunge(ws)
            return ws

    def mark_purged(self, id_value: uuid.UUID | str) -> Optional[Workspace]:
        """Flip status to PURGED — used by the purge service after child
        data (jobs, matches) has been hard-deleted via cascade.
        """
        pk = self._coerce_uuid(id_value)
        with self._sql.sync_session() as session:
            stmt = (
                sa_update(Workspace)
                .where(Workspace.id == pk)
                .values(status=WorkspaceStatus.PURGED)
                .returning(Workspace)
            )
            ws: Optional[Workspace] = session.execute(stmt).scalars().first()
            if ws is not None:
                session.expunge(ws)
            return ws


_workspace_repo: Optional[WorkspaceRepository] = None


def get_workspace_repository() -> WorkspaceRepository:
    global _workspace_repo
    if _workspace_repo is None:
        _workspace_repo = WorkspaceRepository()
    return _workspace_repo
