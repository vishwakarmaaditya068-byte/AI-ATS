"""JobRecord repository — workspace-scoped job queries."""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select

from src.data.sql.models.job_record import JobRecord, JobStatus
from src.data.sql.repositories.base import BaseSQLRepository


class JobRecordRepository(BaseSQLRepository[JobRecord]):
    @property
    def model_class(self) -> type[JobRecord]:
        return JobRecord

    def list_for_workspace(
        self,
        workspace_id: uuid.UUID | str,
        limit: int = 100,
        offset: int = 0,
        status: Optional[JobStatus] = None,
    ) -> list[JobRecord]:
        ws_id = self._coerce_uuid(workspace_id)
        with self._sql.sync_session() as session:
            stmt = select(JobRecord).where(JobRecord.workspace_id == ws_id)
            if status is not None:
                stmt = stmt.where(JobRecord.status == status)
            stmt = stmt.order_by(JobRecord.created_at.desc()).limit(limit).offset(offset)
            result = session.execute(stmt).scalars().all()
            for job in result:
                session.expunge(job)
            return list(result)

    def find_by_mongo_id(self, mongo_doc_id: str) -> Optional[JobRecord]:
        with self._sql.sync_session() as session:
            stmt = select(JobRecord).where(JobRecord.mongo_doc_id == mongo_doc_id).limit(1)
            job: Optional[JobRecord] = session.execute(stmt).scalars().first()
            if job is not None:
                session.expunge(job)
            return job

    def count_for_workspace(self, workspace_id: uuid.UUID | str) -> int:
        from sqlalchemy import func
        ws_id = self._coerce_uuid(workspace_id)
        with self._sql.sync_session() as session:
            stmt = select(func.count()).select_from(JobRecord).where(
                JobRecord.workspace_id == ws_id
            )
            return int(session.execute(stmt).scalar_one())


_job_record_repo: Optional[JobRecordRepository] = None


def get_job_record_repository() -> JobRecordRepository:
    global _job_record_repo
    if _job_record_repo is None:
        _job_record_repo = JobRecordRepository()
    return _job_record_repo
