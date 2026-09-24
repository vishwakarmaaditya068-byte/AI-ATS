"""
Unit tests for EmbeddingService.
EmbeddingModel and VectorStore are injected as fakes — no GPU, no ChromaDB needed.
"""
from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.ml.embeddings.embedding_service import EmbeddingService
from src.ml.nlp.accurate_resume_parser import (
    AccurateResumeParser,
    ParsedResume,
    ContactInfo,
    SkillCategory,
    ExperienceEntry,
    EducationEntry,
    ProjectEntry,
)
from src.data.models.job import Job, SkillRequirement


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------

class FakeEmbeddingModel:
    """Returns a deterministic 4-dim vector without loading any model."""
    def encode_resume(self, text: str) -> np.ndarray:
        return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    def encode_job_description(self, text: str) -> np.ndarray:
        return np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)


class FakeVectorStore:
    """Records upsert calls for assertion."""
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.calls.append({"ids": ids, "documents": documents, "metadatas": metadatas})


def _make_parsed(
    *,
    name: str = "Vivek Vaish",
    email: str = "v@test.com",
    summary: str = "Backend engineer",
) -> ParsedResume:
    return ParsedResume(
        contact=ContactInfo(name=name, email=email),
        skills=[SkillCategory(category="Languages", skills=["python", "javascript"])],
        experience=[ExperienceEntry(title="SWE Intern", company="Acme", bullets=["Built API"])],
        education=[EducationEntry(degree="B.Tech", institution="Galgotias University")],
        projects=[ProjectEntry(name="AutoBook", bullets=["Automated booking"])],
        summary=summary,
        raw_text="raw resume text",
    )


def _make_service(repo: Optional[Any] = None) -> EmbeddingService:
    return EmbeddingService(
        model=FakeEmbeddingModel(),
        store=FakeVectorStore(),
        job_store=FakeVectorStore(),
        repo=repo,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildText:
    def test_includes_summary(self) -> None:
        svc: EmbeddingService = _make_service()
        parsed: ParsedResume = _make_parsed(summary="Experienced backend developer")
        text: str = svc._build_text(parsed)
        assert "experienced backend developer" in text.lower()

    def test_includes_skills(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_text(_make_parsed())
        assert "python" in text.lower()

    def test_includes_experience_title(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_text(_make_parsed())
        assert "swe intern" in text.lower()

    def test_includes_education(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_text(_make_parsed())
        assert "galgotias" in text.lower()

    def test_includes_project_name(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_text(_make_parsed())
        assert "autobook" in text.lower()

    def test_returns_non_empty_string(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_text(_make_parsed())
        assert len(text) > 30


class TestEmbedCandidate:
    def test_returns_embedding_id_string(self) -> None:
        svc: EmbeddingService = _make_service()
        eid: str = svc.embed_candidate("abc123", _make_parsed())
        assert eid == "candidate_abc123"

    def test_upsert_called_with_correct_id(self) -> None:
        store: FakeVectorStore = FakeVectorStore()
        svc: EmbeddingService = EmbeddingService(model=FakeEmbeddingModel(), store=store, repo=None)
        svc.embed_candidate("xyz", _make_parsed())
        assert store.calls[0]["ids"] == ["candidate_xyz"]

    def test_upsert_called_with_document(self) -> None:
        store: FakeVectorStore = FakeVectorStore()
        svc: EmbeddingService = EmbeddingService(model=FakeEmbeddingModel(), store=store, repo=None)
        svc.embed_candidate("xyz", _make_parsed())
        assert store.calls[0]["documents"] is not None
        assert len(store.calls[0]["documents"][0]) > 0

    def test_metadata_contains_candidate_id(self) -> None:
        store: FakeVectorStore = FakeVectorStore()
        svc: EmbeddingService = EmbeddingService(model=FakeEmbeddingModel(), store=store, repo=None)
        svc.embed_candidate("xyz", _make_parsed())
        meta: dict[str, Any] = store.calls[0]["metadatas"][0]
        assert meta["candidate_id"] == "xyz"

    def test_repo_set_embedding_id_called(self) -> None:
        repo: MagicMock = MagicMock()
        svc: EmbeddingService = EmbeddingService(model=FakeEmbeddingModel(), store=FakeVectorStore(), repo=repo)
        svc.embed_candidate("xyz", _make_parsed())
        repo.set_embedding_id.assert_called_once_with("xyz", "candidate_xyz")

    def test_repo_none_does_not_raise(self) -> None:
        svc: EmbeddingService = _make_service(repo=None)
        eid: str = svc.embed_candidate("xyz", _make_parsed())
        assert eid == "candidate_xyz"


def _make_job(
    *,
    title: str = "Backend Engineer",
    description: str = "Build scalable APIs",
    responsibilities: list[str] | None = None,
    skill_names: list[str] | None = None,
) -> Job:
    reqs: list[SkillRequirement] = [
        SkillRequirement(name=s, is_required=True)
        for s in (skill_names or ["python", "fastapi"])
    ]
    return Job(
        title=title,
        description=description,
        responsibilities=responsibilities or ["Design REST APIs", "Write unit tests"],
        company_name="Acme Corp",
        skill_requirements=reqs,
    )


class TestBuildJDText:
    def test_includes_title(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_jd_text(_make_job(title="ML Engineer"))
        assert "ml engineer" in text.lower()

    def test_includes_skill_names(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_jd_text(_make_job(skill_names=["pytorch", "numpy"]))
        assert "pytorch" in text.lower()

    def test_includes_responsibilities(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_jd_text(
            _make_job(responsibilities=["Deploy models to production"])
        )
        assert "deploy models" in text.lower()

    def test_includes_description(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_jd_text(_make_job(description="Build scalable microservices"))
        assert "microservices" in text.lower()

    def test_returns_non_empty(self) -> None:
        svc: EmbeddingService = _make_service()
        text: str = svc._build_jd_text(_make_job())
        assert len(text) > 20

    def test_includes_company_description(self) -> None:
        svc: EmbeddingService = _make_service()
        job: Job = Job(
            title="Engineer",
            description="Build services",
            company_name="Acme Corp",
            company_description="We build AI-powered tools for enterprises.",
        )
        text: str = svc._build_jd_text(job)
        assert "ai-powered" in text.lower()


class TestEmbedJob:
    def test_returns_job_embedding_id(self) -> None:
        svc: EmbeddingService = _make_service()
        eid: str = svc.embed_job("job001", _make_job())
        assert eid == "job_job001"

    def test_job_store_upsert_called_with_correct_id(self) -> None:
        job_store: FakeVectorStore = FakeVectorStore()
        svc: EmbeddingService = EmbeddingService(
            model=FakeEmbeddingModel(), store=FakeVectorStore(),
            job_store=job_store, repo=None,
        )
        svc.embed_job("j42", _make_job())
        assert job_store.calls[0]["ids"] == ["job_j42"]

    def test_job_metadata_contains_job_id(self) -> None:
        job_store: FakeVectorStore = FakeVectorStore()
        svc: EmbeddingService = EmbeddingService(
            model=FakeEmbeddingModel(), store=FakeVectorStore(),
            job_store=job_store, repo=None,
        )
        svc.embed_job("j42", _make_job())
        meta: dict[str, Any] = job_store.calls[0]["metadatas"][0]
        assert meta["job_id"] == "j42"

    def test_repo_set_embedding_id_called_for_job(self) -> None:
        repo: MagicMock = MagicMock()
        svc: EmbeddingService = EmbeddingService(
            model=FakeEmbeddingModel(), store=FakeVectorStore(),
            job_store=FakeVectorStore(), repo=repo,
        )
        svc.embed_job("j42", _make_job())
        repo.set_embedding_id.assert_called_once_with("j42", "job_j42")

    def test_resume_store_not_touched_by_embed_job(self) -> None:
        resume_store: FakeVectorStore = FakeVectorStore()
        job_store: FakeVectorStore = FakeVectorStore()
        svc: EmbeddingService = EmbeddingService(
            model=FakeEmbeddingModel(), store=resume_store,
            job_store=job_store, repo=None,
        )
        svc.embed_job("j1", _make_job())
        assert len(resume_store.calls) == 0
        assert len(job_store.calls) == 1

    def test_job_store_resolved_lazily_and_cached(self) -> None:
        """When job_store=None, get_job_store() is called once and cached."""
        from unittest.mock import patch, MagicMock
        fake_store: FakeVectorStore = FakeVectorStore()
        with patch(
            "src.ml.embeddings.embedding_service.get_job_store",
            return_value=fake_store,
        ) as mock_get:
            svc: EmbeddingService = EmbeddingService(
                model=FakeEmbeddingModel(), store=FakeVectorStore(), repo=None
            )  # job_store=None intentionally
            svc.embed_job("j1", _make_job())
            svc.embed_job("j2", _make_job())
        # get_job_store() called exactly once (cached after first call)
        mock_get.assert_called_once()
        assert len(fake_store.calls) == 2
