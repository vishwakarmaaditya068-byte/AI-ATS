"""AuditRecord repository — write-heavy audit trail."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select

from src.data.sql.models.audit_record import AuditRecord
from src.data.sql.repositories.base import BaseSQLRepository


class AuditRecordRepository(BaseSQLRepository[AuditRecord]):
    @property
    def model_class(self) -> type[AuditRecord]:
        return AuditRecord

    # Write
    def log(
        self,
        action: str,
        action_description: str,
        *,
        workspace_id: Optional[uuid.UUID | str] = None,
        actor: Optional[dict[str, Any]] = None,
        resource: Optional[dict[str, Any]] = None,
        changes: Optional[list[dict[str, Any]]] = None,
        ai_decision: Optional[dict[str, Any]] = None,
        bias_audit: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
        related_candidate_mongo_id: Optional[str] = None,
        related_job_id: Optional[uuid.UUID | str] = None,
        related_match_id: Optional[uuid.UUID | str] = None,
        compliance_relevant: bool = False,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AuditRecord:
        record = AuditRecord(
            workspace_id=self._coerce_uuid(workspace_id) if workspace_id else None,
            action=action,
            action_description=action_description,
            actor=actor or {"actor_type": "system"},
            resource=resource or {},
            changes=changes or [],
            ai_decision=ai_decision or {},
            bias_audit=bias_audit or {},
            context=context or {},
            related_candidate_mongo_id=related_candidate_mongo_id,
            related_job_id=self._coerce_uuid(related_job_id) if related_job_id else None,
            related_match_id=self._coerce_uuid(related_match_id) if related_match_id else None,
            compliance_relevant=compliance_relevant,
            session_id=session_id,
            request_id=request_id,
        )
        return self.create(record)

    async def log_async(
        self,
        action: str,
        action_description: str,
        *,
        workspace_id: Optional[uuid.UUID | str] = None,
        actor: Optional[dict[str, Any]] = None,
        resource: Optional[dict[str, Any]] = None,
        changes: Optional[list[dict[str, Any]]] = None,
        ai_decision: Optional[dict[str, Any]] = None,
        bias_audit: Optional[dict[str, Any]] = None,
        context: Optional[dict[str, Any]] = None,
        related_candidate_mongo_id: Optional[str] = None,
        related_job_id: Optional[uuid.UUID | str] = None,
        related_match_id: Optional[uuid.UUID | str] = None,
        compliance_relevant: bool = False,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AuditRecord:
        record = AuditRecord(
            workspace_id=self._coerce_uuid(workspace_id) if workspace_id else None,
            action=action,
            action_description=action_description,
            actor=actor or {"actor_type": "system"},
            resource=resource or {},
            changes=changes or [],
            ai_decision=ai_decision or {},
            bias_audit=bias_audit or {},
            context=context or {},
            related_candidate_mongo_id=related_candidate_mongo_id,
            related_job_id=self._coerce_uuid(related_job_id) if related_job_id else None,
            related_match_id=self._coerce_uuid(related_match_id) if related_match_id else None,
            compliance_relevant=compliance_relevant,
            session_id=session_id,
            request_id=request_id,
        )
        return await self.create_async(record)

    # Read

    def list_for_workspace(
        self,
        workspace_id: uuid.UUID | str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditRecord]:
        ws_id = self._coerce_uuid(workspace_id)
        with self._sql.sync_session() as session:
            stmt = (
                select(AuditRecord)
                .where(AuditRecord.workspace_id == ws_id)
                .order_by(AuditRecord.occurred_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = session.execute(stmt).scalars().all()
            for r in result:
                session.expunge(r)
            return list(result)

    def list_by_action(
        self,
        action: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditRecord]:
        with self._sql.sync_session() as session:
            stmt = (
                select(AuditRecord)
                .where(AuditRecord.action == action)
                .order_by(AuditRecord.occurred_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = session.execute(stmt).scalars().all()
            for r in result:
                session.expunge(r)
            return list(result)

    def list_for_match(
        self,
        match_id: uuid.UUID | str,
        limit: int = 50,
    ) -> list[AuditRecord]:
        pk = self._coerce_uuid(match_id)
        with self._sql.sync_session() as session:
            stmt = (
                select(AuditRecord)
                .where(AuditRecord.related_match_id == pk)
                .order_by(AuditRecord.occurred_at.desc())
                .limit(limit)
            )
            result = session.execute(stmt).scalars().all()
            for r in result:
                session.expunge(r)
            return list(result)


_audit_record_repo: Optional[AuditRecordRepository] = None


def get_audit_record_repository() -> AuditRecordRepository:
    global _audit_record_repo
    if _audit_record_repo is None:
        _audit_record_repo = AuditRecordRepository()
    return _audit_record_repo
