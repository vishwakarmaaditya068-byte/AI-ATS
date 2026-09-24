"""
Unit tests for MatchingEngine.match_from_parsed().
No real model, no DB — all heavy dependencies stubbed.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from src.core.matching.matching_engine import MatchingEngine, MatchResult, _estimate_years
from src.ml.nlp.accurate_resume_parser import (
    ParsedResume,
    ContactInfo,
    SkillCategory,
    ExperienceEntry,
    EducationEntry,
    ProjectEntry,
)
from src.data.models.job import Job, SkillRequirement


def _make_parsed() -> ParsedResume:
    return ParsedResume(
        contact=ContactInfo(name="Vivek Vaish", email="v@test.com"),
        skills=[SkillCategory(category="Languages", skills=["python", "fastapi"])],
        experience=[ExperienceEntry(title="SWE Intern", company="Acme", bullets=["Built APIs"])],
        education=[EducationEntry(degree="B.Tech", institution="Galgotias University")],
        projects=[ProjectEntry(name="AutoBook", bullets=["Automated booking"])],
        summary="Backend engineer with 2 years experience",
        raw_text="raw resume text",
    )


def _make_job() -> Job:
    return Job(
        title="Backend Engineer",
        description="Build scalable Python microservices",
        responsibilities=["Design REST APIs", "Write unit tests"],
        company_name="Acme Corp",
        skill_requirements=[
            SkillRequirement(name="python", is_required=True),
            SkillRequirement(name="fastapi", is_required=False),
        ],
    )


class TestMatchFromParsed:
    def test_returns_match_result(self) -> None:
        engine: MatchingEngine = MatchingEngine(use_semantic=False)
        result: MatchResult = engine.match_from_parsed(_make_parsed(), _make_job())
        assert isinstance(result, MatchResult)

    def test_candidate_name_extracted(self) -> None:
        engine: MatchingEngine = MatchingEngine(use_semantic=False)
        result: MatchResult = engine.match_from_parsed(_make_parsed(), _make_job())
        assert "vivek" in result.candidate_name.lower()

    def test_job_title_in_result(self) -> None:
        engine: MatchingEngine = MatchingEngine(use_semantic=False)
        result: MatchResult = engine.match_from_parsed(_make_parsed(), _make_job())
        assert "backend engineer" in result.job_title.lower()

    def test_overall_score_between_0_and_1(self) -> None:
        engine: MatchingEngine = MatchingEngine(use_semantic=False)
        result: MatchResult = engine.match_from_parsed(_make_parsed(), _make_job())
        assert 0.0 <= result.overall_score <= 1.0

    def test_score_breakdown_populated(self) -> None:
        engine: MatchingEngine = MatchingEngine(use_semantic=False)
        result: MatchResult = engine.match_from_parsed(_make_parsed(), _make_job())
        assert result.score_breakdown is not None

    def test_semantic_path_uses_compute_similarity_from_parsed(self) -> None:
        """When semantic is enabled, compute_similarity_from_parsed is called."""
        engine: MatchingEngine = MatchingEngine(use_semantic=True)
        mock_matcher: MagicMock = MagicMock()
        mock_matcher.compute_similarity_from_parsed.return_value = MagicMock(
            overall_similarity=0.75,
        )
        engine._semantic_matcher = mock_matcher
        engine.match_from_parsed(_make_parsed(), _make_job())
        mock_matcher.compute_similarity_from_parsed.assert_called_once()

    def test_no_semantic_score_when_disabled(self) -> None:
        engine: MatchingEngine = MatchingEngine(use_semantic=False)
        result: MatchResult = engine.match_from_parsed(_make_parsed(), _make_job())
        assert result.semantic_score == 0.0
        assert result.semantic_match is None


class TestEstimateYears:
    def test_explicit_years(self) -> None:
        assert _estimate_years("3 years") == 3.0

    def test_months_converted(self) -> None:
        assert _estimate_years("6 months") == 0.5

    def test_year_range(self) -> None:
        assert _estimate_years("2019 - 2022") == 3.0

    def test_present_keyword(self) -> None:
        import datetime
        expected: float = float(datetime.date.today().year - 2021)
        assert _estimate_years("2021 - present") == max(expected, 1.0)

    def test_empty_returns_one(self) -> None:
        assert _estimate_years("") == 1.0
