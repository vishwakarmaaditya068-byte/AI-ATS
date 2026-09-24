"""
Unit tests for JobRepository embedding wiring.
No live DB — uses MagicMock to stub self.create().
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.data.repositories.job_repository import JobRepository
from src.data.models.job import Job, JobCreate, SkillRequirement


FAKE_JOB_ID: str = "507f1f77bcf86cd799439011"


def _make_job_create() -> JobCreate:
    return JobCreate(
        title="Backend Engineer",
        description="Build scalable services with Python and FastAPI.",
        company_name="Acme Corp",
        skill_requirements=[SkillRequirement(name="python", is_required=True)],
    )


def _make_repo() -> JobRepository:
    """Return a JobRepository with self.create() stubbed to avoid MongoDB."""
    repo: JobRepository = JobRepository.__new__(JobRepository)
    fake_job: Job = Job(
        title="Backend Engineer",
        description="Build scalable services with Python and FastAPI.",
        company_name="Acme Corp",
    )
    # Attach a fake id to the job (mimic what MongoDB sets after insert)
    object.__setattr__(fake_job, "id", FAKE_JOB_ID)
    repo.create = MagicMock(return_value=fake_job)  # type: ignore[method-assign]
    return repo


class TestJobRepositoryEmbedding:
    def test_create_from_schema_triggers_embed_job(self) -> None:
        repo: JobRepository = _make_repo()
        with patch(
            "src.ml.embeddings.embedding_service.EmbeddingService.embed_job"
        ) as mock_embed:
            mock_embed.return_value = f"job_{FAKE_JOB_ID}"
            repo.create_from_schema(_make_job_create())
        mock_embed.assert_called_once()

    def test_embed_job_receives_correct_job_id(self) -> None:
        repo: JobRepository = _make_repo()
        captured: list[str] = []
        with patch(
            "src.ml.embeddings.embedding_service.EmbeddingService.embed_job",
            side_effect=lambda jid, job: captured.append(jid) or f"job_{jid}",
        ):
            repo.create_from_schema(_make_job_create())
        assert captured[0] == FAKE_JOB_ID

    def test_embedding_failure_does_not_raise(self) -> None:
        """Embedding crash must never propagate — job creation always returns."""
        repo: JobRepository = _make_repo()
        with patch(
            "src.ml.embeddings.embedding_service.EmbeddingService.embed_job",
            side_effect=RuntimeError("ChromaDB unavailable"),
        ):
            result: Job = repo.create_from_schema(_make_job_create())
        assert result is not None

    def test_create_still_returns_job_on_embed_failure(self) -> None:
        repo: JobRepository = _make_repo()
        with patch(
            "src.ml.embeddings.embedding_service.EmbeddingService.embed_job",
            side_effect=RuntimeError("GPU error"),
        ):
            job: Job = repo.create_from_schema(_make_job_create())
        assert job.title == "Backend Engineer"
