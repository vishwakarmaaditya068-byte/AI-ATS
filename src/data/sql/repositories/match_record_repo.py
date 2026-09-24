"""MatchRecord repository — scoring persistence and override support."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import func, select, update as sa_update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.data.sql.models.match_record import MatchRecord
from src.data.sql.repositories.base import BaseSQLRepository


class MatchRecordRepository(BaseSQLRepository[MatchRecord]):
    @property
    def model_class(self) -> type[MatchRecord]:
        return MatchRecord

    # Persistence
    def upsert(
        self,
        job_id: uuid.UUID | str,
        candidate_mongo_id: str,
        overall_score: float,
        *,
        skills_score: Optional[float] = None,
        experience_score: Optional[float] = None,
        education_score: Optional[float] = None,
        semantic_score: Optional[float] = None,
        keyword_score: Optional[float] = None,
        score_level: Optional[str] = None,
        explanation: Optional[dict[str, Any]] = None,
        bias_check: Optional[dict[str, Any]] = None,
        score_breakdown: Optional[dict[str, Any]] = None,
        scoring_model_version: str = "1.0",
    ) -> MatchRecord:
        """Insert a match, or update overall/component scores if the
        (job_id, candidate_mongo_id) pair already exists.

        Override fields are intentionally NOT updated here — overrides
        are a separate operation (see `apply_override`) and must not be
        clobbered by re-running the matcher.
        """
        job_uuid = self._coerce_uuid(job_id)
        payload: dict[str, Any] = {
            "id": uuid.uuid4(),
            "job_id": job_uuid,
            "candidate_mongo_id": candidate_mongo_id,
            "overall_score": overall_score,
            "skills_score": skills_score,
            "experience_score": experience_score,
            "education_score": education_score,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
            "score_level": score_level,
            "explanation": explanation or {},
            "bias_check": bias_check or {},
            "score_breakdown": score_breakdown or {},
            "scoring_model_version": scoring_model_version,
        }

        with self._sql.sync_session() as session:
            stmt = pg_insert(MatchRecord).values(**payload)
            update_cols = {
                "overall_score": stmt.excluded.overall_score,
                "skills_score": stmt.excluded.skills_score,
                "experience_score": stmt.excluded.experience_score,
                "education_score": stmt.excluded.education_score,
                "semantic_score": stmt.excluded.semantic_score,
                "keyword_score": stmt.excluded.keyword_score,
                "score_level": stmt.excluded.score_level,
                "explanation": stmt.excluded.explanation,
                "bias_check": stmt.excluded.bias_check,
                "score_breakdown": stmt.excluded.score_breakdown,
                "scoring_model_version": stmt.excluded.scoring_model_version,
                "updated_at": datetime.now(timezone.utc),
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_match_job_candidate",
                set_=update_cols,
            ).returning(MatchRecord)
            match: MatchRecord = session.execute(stmt).scalars().first()
            session.flush()
            session.refresh(match)
            session.expunge(match)
            return match

    # Queries
    def list_for_job(
        self,
        job_id: uuid.UUID | str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MatchRecord]:
        job_uuid = self._coerce_uuid(job_id)
        with self._sql.sync_session() as session:
            stmt = (
                select(MatchRecord)
                .where(MatchRecord.job_id == job_uuid)
                .order_by(MatchRecord.overall_score.desc())
                .limit(limit)
                .offset(offset)
            )
            result = session.execute(stmt).scalars().all()
            for m in result:
                session.expunge(m)
            return list(result)

    def get_by_job_and_candidate(
        self,
        job_id: uuid.UUID | str,
        candidate_mongo_id: str,
    ) -> Optional[MatchRecord]:
        job_uuid = self._coerce_uuid(job_id)
        with self._sql.sync_session() as session:
            stmt = (
                select(MatchRecord)
                .where(
                    MatchRecord.job_id == job_uuid,
                    MatchRecord.candidate_mongo_id == candidate_mongo_id,
                )
                .limit(1)
            )
            m: Optional[MatchRecord] = session.execute(stmt).scalars().first()
            if m is not None:
                session.expunge(m)
            return m

    def count_for_job(self, job_id: uuid.UUID | str) -> int:
        job_uuid = self._coerce_uuid(job_id)
        with self._sql.sync_session() as session:
            stmt = (
                select(func.count()).select_from(MatchRecord).where(MatchRecord.job_id == job_uuid)
            )
            return int(session.execute(stmt).scalar_one())

    # Override
    def apply_override(
        self,
        match_id: uuid.UUID | str,
        new_score: float,
        reason: str,
        overridden_by: str,
    ) -> Optional[MatchRecord]:
        """Apply a manual score override. Does not touch overall_score."""
        if not (0.0 <= new_score <= 1.0):
            raise ValueError(f"Override score out of range [0..1]: {new_score}")
        if not reason or len(reason.strip()) < 5:
            raise ValueError("Override reason must be at least 5 characters")

        pk = self._coerce_uuid(match_id)
        with self._sql.sync_session() as session:
            stmt = (
                sa_update(MatchRecord)
                .where(MatchRecord.id == pk)
                .values(
                    manual_score_override=new_score,
                    override_reason=reason.strip(),
                    overridden_by=overridden_by,
                )
                .returning(MatchRecord)
            )
            m: Optional[MatchRecord] = session.execute(stmt).scalars().first()
            if m is not None:
                session.flush()
                session.refresh(m)
                session.expunge(m)
            return m

    def clear_override(self, match_id: uuid.UUID | str) -> Optional[MatchRecord]:
        pk = self._coerce_uuid(match_id)
        with self._sql.sync_session() as session:
            stmt = (
                sa_update(MatchRecord)
                .where(MatchRecord.id == pk)
                .values(
                    manual_score_override=None,
                    override_reason=None,
                    overridden_by=None,
                )
                .returning(MatchRecord)
            )
            m: Optional[MatchRecord] = session.execute(stmt).scalars().first()
            if m is not None:
                session.flush()
                session.refresh(m)
                session.expunge(m)
            return m


_match_record_repo: Optional[MatchRecordRepository] = None


def get_match_record_repository() -> MatchRecordRepository:
    global _match_record_repo
    if _match_record_repo is None:
        _match_record_repo = MatchRecordRepository()
    return _match_record_repo
