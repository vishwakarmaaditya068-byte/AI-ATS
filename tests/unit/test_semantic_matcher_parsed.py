"""
Unit tests for SemanticMatcher.compute_similarity_from_parsed().
No real model loaded — FakeEmbeddingModel returns deterministic vectors.
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pytest

from src.ml.embeddings.semantic_similarity import SemanticMatcher
from src.ml.nlp.accurate_resume_parser import (
    ParsedResume,
    ContactInfo,
    SkillCategory,
    ExperienceEntry,
    EducationEntry,
    ProjectEntry,
)
from src.data.models.job import Job, SkillRequirement
from src.data.models import SemanticMatch


class FakeEmbeddingModel:
    """Returns constant 4-dim vector — similarity will always be 1.0."""
    def encode(self, text: Any, **kwargs: Any) -> np.ndarray:
        if isinstance(text, list):
            return np.array([[1.0, 0.0, 0.0, 0.0]] * len(text), dtype=np.float32)
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.ndim > 1:
            a = a[0]
        if b.ndim > 1:
            b = b[0]
        norm_a: float = float(np.linalg.norm(a))
        norm_b: float = float(np.linalg.norm(b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


def _make_parsed(summary: str = "Senior backend engineer") -> ParsedResume:
    return ParsedResume(
        contact=ContactInfo(name="Vivek Vaish", email="v@test.com"),
        skills=[SkillCategory(category="Languages", skills=["python", "fastapi"])],
        experience=[ExperienceEntry(title="SWE", company="Acme", bullets=["Built APIs"])],
        education=[EducationEntry(degree="B.Tech", institution="Galgotias")],
        projects=[ProjectEntry(name="AutoBook", bullets=["Booking automation"])],
        summary=summary,
        raw_text="raw resume",
    )


def _make_job(title: str = "Backend Engineer") -> Job:
    return Job(
        title=title,
        description="Build scalable Python services",
        responsibilities=["Design REST APIs", "Write tests"],
        company_name="Acme Corp",
        skill_requirements=[
            SkillRequirement(name="python", is_required=True),
            SkillRequirement(name="fastapi", is_required=False),
        ],
    )


def _make_matcher() -> SemanticMatcher:
    return SemanticMatcher(embedding_model=FakeEmbeddingModel())  # type: ignore[arg-type]


class TestComputeSimilarityFromParsed:
    def test_returns_semantic_match_instance(self) -> None:
        m: SemanticMatcher = _make_matcher()
        result: SemanticMatch = m.compute_similarity_from_parsed(_make_parsed(), _make_job())
        assert isinstance(result, SemanticMatch)

    def test_overall_similarity_in_range(self) -> None:
        m: SemanticMatcher = _make_matcher()
        result: SemanticMatch = m.compute_similarity_from_parsed(_make_parsed(), _make_job())
        assert 0.0 <= result.overall_similarity <= 1.0

    def test_skills_similarity_in_range(self) -> None:
        m: SemanticMatcher = _make_matcher()
        result: SemanticMatch = m.compute_similarity_from_parsed(_make_parsed(), _make_job())
        assert 0.0 <= result.skills_similarity <= 1.0

    def test_experience_similarity_in_range(self) -> None:
        m: SemanticMatcher = _make_matcher()
        result: SemanticMatch = m.compute_similarity_from_parsed(_make_parsed(), _make_job())
        assert 0.0 <= result.experience_similarity <= 1.0

    def test_model_name_recorded(self) -> None:
        m: SemanticMatcher = _make_matcher()
        result: SemanticMatch = m.compute_similarity_from_parsed(_make_parsed(), _make_job())
        assert result.model_used != ""

    def test_empty_resume_returns_zero_overall(self) -> None:
        """ParsedResume with no text → overall_similarity = 0.0."""
        m: SemanticMatcher = _make_matcher()
        empty: ParsedResume = ParsedResume(
            contact=ContactInfo(name="X", email="x@x.com"),
            raw_text="",
            summary="",
        )
        result: SemanticMatch = m.compute_similarity_from_parsed(empty, _make_job())
        assert result.overall_similarity == 0.0

    def test_summary_similarity_in_range(self) -> None:
        m: SemanticMatcher = _make_matcher()
        result: SemanticMatch = m.compute_similarity_from_parsed(_make_parsed(), _make_job())
        assert 0.0 <= result.summary_similarity <= 1.0
