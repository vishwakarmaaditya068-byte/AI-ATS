from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

from src.data.sql.models.job_record import JobRecord, JobStatus
from src.data.sql.models.match_record import MatchRecord
from src.data.sql.models.workspace import Workspace
from src.data.sql.repositories.job_record_repo import (
    JobRecordRepository,
    get_job_record_repository,
)
from src.data.sql.repositories.match_record_repo import (
    MatchRecordRepository,
    get_match_record_repository,
)
from src.data.sql.repositories.workspace_repo import (
    WorkspaceRepository,
    get_workspace_repository,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


DEFAULT_WORKSPACE_NAME: str = "Default"
DEFAULT_WORKSPACE_DESCRIPTION: str = (
    "Auto-created workspace. Rename or archive from the Workspace picker "
    "once multiple workspaces are in use."
)


@dataclass
class PersistedMatch:
    """Snapshot of a persisted match used by the UI worker."""

    match_id: uuid.UUID
    job_id: uuid.UUID
    workspace_id: uuid.UUID
    candidate_mongo_id: str
    overall_score: float
    effective_score: float
    score_level: Optional[str]
    is_override: bool


class MatchPersistenceService:
    """Coordinator for resolving + persisting match rows."""

    def __init__(
        self,
        workspace_repo: Optional[WorkspaceRepository] = None,
        job_record_repo: Optional[JobRecordRepository] = None,
        match_record_repo: Optional[MatchRecordRepository] = None,
    ) -> None:
        self._workspace_repo: WorkspaceRepository = workspace_repo or get_workspace_repository()
        self._job_record_repo: JobRecordRepository = job_record_repo or get_job_record_repository()
        self._match_record_repo: MatchRecordRepository = (
            match_record_repo or get_match_record_repository()
        )

    # Workspace bootstrap
    def ensure_default_workspace(self) -> Workspace:
        """Return the Default workspace, creating it if absent.

        Used as a stable fallback during Phase 2 before the workspace
        picker UI is built.
        """
        existing = self._workspace_repo.find_by_name(DEFAULT_WORKSPACE_NAME)
        if existing is not None:
            return existing
        logger.info(f"Creating {DEFAULT_WORKSPACE_NAME!r} workspace (first use)")
        return self._workspace_repo.create_workspace(
            name=DEFAULT_WORKSPACE_NAME,
            description=DEFAULT_WORKSPACE_DESCRIPTION,
            created_by="system",
        )

    # Job mirror (Mongo job → Postgres JobRecord)
    def ensure_job_record(
        self,
        *,
        workspace_id: uuid.UUID | str,
        title: str,
        company_name: Optional[str] = None,
        description: Optional[str] = None,
        mongo_doc_id: Optional[str] = None,
    ) -> JobRecord:
        """Return a JobRecord for this job, creating or linking as needed.

        Lookup order:
          1. If `mongo_doc_id` is supplied and a JobRecord already maps
             to it → return that row.
          2. Otherwise create a fresh JobRecord scoped to the workspace.
        """
        if mongo_doc_id:
            existing = self._job_record_repo.find_by_mongo_id(mongo_doc_id)
            if existing is not None:
                return existing

        snippet: Optional[str] = None
        if description:
            snippet = description[:500]

        job_record = JobRecord(
            workspace_id=self._workspace_repo._coerce_uuid(workspace_id),
            mongo_doc_id=mongo_doc_id,
            title=title,
            company_name=company_name,
            description_snippet=snippet,
            status=JobStatus.OPEN,
        )
        return self._job_record_repo.create(job_record)

    # Match upsert
    def persist_match(
        self,
        *,
        job_id: uuid.UUID | str,
        candidate_mongo_id: str,
        overall_score: float,
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
        """Upsert a match row. Override fields are preserved on re-runs."""
        return self._match_record_repo.upsert(
            job_id=job_id,
            candidate_mongo_id=candidate_mongo_id,
            overall_score=overall_score,
            skills_score=skills_score,
            experience_score=experience_score,
            education_score=education_score,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            score_level=score_level,
            explanation=explanation,
            bias_check=bias_check,
            score_breakdown=score_breakdown,
            scoring_model_version=scoring_model_version,
        )

    # Candidate resolution (Mongo)
    def resolve_candidate_mongo_id(
        self,
        *,
        email: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> Optional[str]:
        """Best-effort candidate lookup by email or file hash.

        Returns the Mongo ObjectId hex, or None if the candidate has
        not been ingested yet. Callers should ingest first and retry.
        """
        try:
            from src.data.repositories.candidate_repository import (
                get_candidate_repository,
            )
        except Exception as exc:  # pragma: no cover — mongo optional in tests
            logger.warning(f"Candidate repo import failed: {exc}")
            return None

        repo = get_candidate_repository()

        if email:
            try:
                candidate = repo.find_one({"contact.email": email.strip().lower()})
                if candidate is not None and candidate.id is not None:
                    return str(candidate.id)
            except Exception as exc:
                logger.warning(f"Candidate lookup by email failed: {exc}")

        if file_hash:
            try:
                candidate = repo.find_one({"file_hashes": file_hash})
                if candidate is not None and candidate.id is not None:
                    return str(candidate.id)
            except Exception as exc:
                logger.warning(f"Candidate lookup by file_hash failed: {exc}")

        return None


_match_persistence_service: Optional[MatchPersistenceService] = None


def get_match_persistence_service() -> MatchPersistenceService:
    global _match_persistence_service
    if _match_persistence_service is None:
        _match_persistence_service = MatchPersistenceService()
    return _match_persistence_service
