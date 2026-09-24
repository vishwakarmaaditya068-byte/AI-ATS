"""
End-to-end ingestion pipeline tests using real PDFs from data/raw/resumes/.
MongoDB is replaced with an in-memory FakeRepo — no live DB required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pytest

from src.services.ingestion_service import IngestionService, IngestionResult
from src.ml.nlp.accurate_resume_parser import ParsedResume

RESUMES_DIR: Path = Path("data/raw/resumes")
ALL_PDFS: list[Path] = list(RESUMES_DIR.glob("*.pdf"))

# Skip the entire module gracefully when the test resume fixture is absent.
# The file is personal data and not committed to the repository.
pytestmark = pytest.mark.skipif(
    not (RESUMES_DIR / "vivek_resume.pdf").exists(),
    reason="vivek_resume.pdf not found in data/raw/resumes/ — integration tests require real resume fixtures",
)


# ---------------------------------------------------------------------------
# In-memory repository stub
# ---------------------------------------------------------------------------

@dataclass
class _FakeCandidate:
    id: str


class FakeRepo:
    """In-memory candidate store satisfying CandidateRepoProtocol."""

    def __init__(self) -> None:
        self._hashes: set[str] = set()
        self._candidates: dict[str, _FakeCandidate] = {}   # email → candidate
        self._counter: int = 0

    def hash_exists(self, file_hash: str) -> bool:
        return file_hash in self._hashes

    def upsert_by_email(
        self,
        parsed: ParsedResume,
        file_hash: str,
        filename: str,
    ) -> _FakeCandidate:
        self._hashes.add(file_hash)
        email: str = (
            parsed.contact.email.lower()
            if parsed.contact.email
            else f"_no_email_{self._counter}"
        )
        self._counter += 1
        candidate = _FakeCandidate(id=str(self._counter))
        self._candidates[email] = candidate
        return candidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo() -> FakeRepo:
    return FakeRepo()


@pytest.fixture
def svc(repo: FakeRepo) -> IngestionService:
    from src.services.file_validator import FileValidator
    from src.ml.nlp.accurate_resume_parser import AccurateResumeParser

    s: IngestionService = IngestionService.__new__(IngestionService)
    s._validator = FileValidator()
    s._parser = AccurateResumeParser()
    s._repo = repo
    return s


# ---------------------------------------------------------------------------
# Single-resume tests
# ---------------------------------------------------------------------------

class TestSingleResume:
    def test_vivek_resume_ingested_successfully(self, svc: IngestionService) -> None:
        result: IngestionResult = svc.ingest_file(RESUMES_DIR / "vivek_resume.pdf")
        assert result.status == "success"
        assert result.candidate_name
        assert result.candidate_email == "vaishvivek634@gmail.com"
        assert len(result.file_hash) == 64
        assert result.processing_time_ms > 0

    def test_same_file_twice_is_duplicate(self, svc: IngestionService) -> None:
        path: Path = RESUMES_DIR / "vivek_resume.pdf"
        first: IngestionResult = svc.ingest_file(path)
        second: IngestionResult = svc.ingest_file(path)
        assert first.status == "success"
        assert second.status == "duplicate"

    def test_nonexistent_file_returns_error(self, svc: IngestionService) -> None:
        result: IngestionResult = svc.ingest_file(Path("does_not_exist.pdf"))
        assert result.status == "error"
        assert result.error_message


# ---------------------------------------------------------------------------
# Batch tests across all PDFs
# ---------------------------------------------------------------------------

class TestBatchIngestion:
    def test_all_real_pdfs_ingest_without_exception(
        self, svc: IngestionService
    ) -> None:
        assert len(ALL_PDFS) > 0, "No PDFs found in data/raw/resumes"
        statuses: list[str] = []
        for pdf in ALL_PDFS:
            result: IngestionResult = svc.ingest_file(pdf)
            assert result.status in ("success", "duplicate", "error"), (
                f"{pdf.name} returned unexpected status: {result.status}"
            )
            statuses.append(result.status)

        success_count: int = statuses.count("success")
        assert success_count >= len(ALL_PDFS) * 0.8, (
            f"Only {success_count}/{len(ALL_PDFS)} resumes ingested successfully"
        )

    def test_no_duplicate_hashes_in_batch(self, svc: IngestionService) -> None:
        """Each unique file must yield a success exactly once."""
        seen_hashes: set[str] = set()
        for pdf in ALL_PDFS:
            result: IngestionResult = svc.ingest_file(pdf)
            if result.status == "success":
                assert result.file_hash not in seen_hashes, (
                    f"Hash collision on: {pdf.name}"
                )
                seen_hashes.add(result.file_hash)


# ---------------------------------------------------------------------------
# Embedding text-building integration
# ---------------------------------------------------------------------------

class TestEmbeddingTextBuilding:
    """Verifies _build_text against real parsed PDFs. No model needed."""

    def test_vivek_text_contains_python(self) -> None:
        from src.ml.embeddings.embedding_service import EmbeddingService
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume

        parser: AccurateResumeParser = AccurateResumeParser()
        parsed: ParsedResume = parser.parse(RESUMES_DIR / "vivek_resume.pdf")
        svc: EmbeddingService = EmbeddingService(model=None, store=None, repo=None)  # type: ignore[arg-type]
        text: str = svc._build_text(parsed)
        assert "python" in text.lower()
        assert len(text) > 100

    def test_all_pdfs_produce_non_empty_text(self) -> None:
        from src.ml.embeddings.embedding_service import EmbeddingService
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume

        parser: AccurateResumeParser = AccurateResumeParser()
        svc: EmbeddingService = EmbeddingService(model=None, store=None, repo=None)  # type: ignore[arg-type]
        for pdf in ALL_PDFS:
            parsed: ParsedResume = parser.parse(pdf)
            text: str = svc._build_text(parsed)
            assert len(text) > 0, f"{pdf.name}: _build_text returned empty string"


# ---------------------------------------------------------------------------
# Job embedding text-building integration
# ---------------------------------------------------------------------------

class TestJobEmbeddingIntegration:
    """Verifies job-side text building and end-to-end SemanticMatcher path."""

    def test_build_jd_text_non_empty_for_real_job(self) -> None:
        import numpy as np
        from unittest.mock import MagicMock
        from src.ml.embeddings.embedding_service import EmbeddingService
        from src.data.models.job import Job, SkillRequirement

        job: Job = Job(
            title="Python Backend Engineer",
            description="Build high-performance REST APIs using Python and FastAPI.",
            responsibilities=["Design scalable services", "Write unit tests"],
            company_name="Acme Corp",
            skill_requirements=[
                SkillRequirement(name="python", is_required=True),
                SkillRequirement(name="fastapi", is_required=True),
                SkillRequirement(name="docker", is_required=False),
            ],
        )
        svc: EmbeddingService = EmbeddingService(model=MagicMock(), store=MagicMock(), repo=None)  # type: ignore[arg-type]
        text: str = svc._build_jd_text(job)
        assert "python" in text.lower()
        assert "fastapi" in text.lower()
        assert len(text) > 50

    def test_vivek_resume_vs_python_job_semantic_similarity(self) -> None:
        """compute_similarity_from_parsed returns SemanticMatch for a real parsed resume."""
        import numpy as np
        from src.ml.embeddings.semantic_similarity import SemanticMatcher
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume
        from src.data.models.job import Job, SkillRequirement
        from src.data.models import SemanticMatch

        class _FakeModel:
            model_name: str = "fake"

            def encode(self, text: Any, **kw: Any) -> np.ndarray:
                if isinstance(text, list):
                    return np.ones((len(text), 4), dtype=np.float32)
                return np.ones(4, dtype=np.float32)

            def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
                return 0.85

        parser: AccurateResumeParser = AccurateResumeParser()
        parsed: ParsedResume = parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        job: Job = Job(
            title="Python Backend Engineer",
            description="Build REST APIs with Python and FastAPI.",
            responsibilities=["Design APIs", "Write tests"],
            company_name="Acme",
            skill_requirements=[SkillRequirement(name="python", is_required=True)],
        )

        matcher: SemanticMatcher = SemanticMatcher(
            embedding_model=_FakeModel()  # type: ignore[arg-type]
        )
        result: SemanticMatch = matcher.compute_similarity_from_parsed(parsed, job)
        assert result.overall_similarity == pytest.approx(0.85, abs=1e-4)
        assert 0.0 <= result.skills_similarity <= 1.0

    def test_match_from_parsed_pipeline_end_to_end(self) -> None:
        """MatchingEngine.match_from_parsed returns populated MatchResult for real PDF."""
        from src.core.matching.matching_engine import MatchingEngine, MatchResult
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume
        from src.data.models.job import Job, SkillRequirement

        parser: AccurateResumeParser = AccurateResumeParser()
        parsed: ParsedResume = parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        job: Job = Job(
            title="Python Backend Engineer",
            description="Build REST APIs with Python and FastAPI.",
            responsibilities=["Design APIs", "Write unit tests"],
            company_name="Acme",
            skill_requirements=[SkillRequirement(name="python", is_required=True)],
        )

        engine: MatchingEngine = MatchingEngine(use_semantic=False)
        result: MatchResult = engine.match_from_parsed(parsed, job)

        assert result.overall_score > 0.0
        assert "vivek" in result.candidate_name.lower()
        assert result.score_breakdown is not None


# ---------------------------------------------------------------------------
# AccurateJDParser integration
# ---------------------------------------------------------------------------

_REAL_JD_TEXT: str = """
Python Backend Engineer
TechStart Ltd | Bangalore, India | Full-time | Hybrid

About the Role
We are building an AI-powered hiring platform and need a skilled backend engineer.

Responsibilities
• Design and implement REST APIs using Python and FastAPI
• Maintain PostgreSQL and MongoDB databases
• Write comprehensive unit and integration tests

Requirements
• 4+ years of Python development experience
• Strong knowledge of SQL (PostgreSQL) and NoSQL (MongoDB) databases
• Experience with Docker, Kubernetes, and AWS
• Bachelor's degree in Computer Science or related field

Nice to Have
• Experience with machine learning frameworks (scikit-learn, PyTorch)
• Knowledge of message queues (Kafka, RabbitMQ)

Benefits
• Competitive salary
• Health insurance for family
• Flexible work hours

About Us
TechStart Ltd is an AI startup focused on transforming recruitment with machine learning.
"""


class TestAccurateJDParserIntegration:
    """Integration tests for AccurateJDParser using real JD text."""

    def test_parse_returns_non_empty_title(self) -> None:
        from src.ml.nlp.accurate_jd_parser import AccurateJDParser, ParsedJob
        parser: AccurateJDParser = AccurateJDParser()
        result: ParsedJob = parser.parse(_REAL_JD_TEXT)
        assert result.title != ""
        assert len(result.title) > 3

    def test_parse_to_job_create_produces_valid_job_create(self) -> None:
        from src.ml.nlp.accurate_jd_parser import AccurateJDParser, ParsedJob
        from src.data.models.job import JobCreate
        parser: AccurateJDParser = AccurateJDParser()
        result: ParsedJob = parser.parse(_REAL_JD_TEXT)
        jc: JobCreate = result.to_job_create()
        assert isinstance(jc, JobCreate)
        assert len(jc.title) >= 1
        assert len(jc.description) >= 10
        assert len(jc.skill_requirements) > 0

    def test_parse_jd_then_match_against_vivek_resume(self) -> None:
        """Full pipeline: AccurateJDParser → Job → AccurateResumeParser → match_from_parsed."""
        from src.ml.nlp.accurate_jd_parser import AccurateJDParser, ParsedJob
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser
        from src.data.models.job import Job, SkillRequirement
        from src.core.matching.matching_engine import MatchingEngine, MatchResult

        jd_parser: AccurateJDParser = AccurateJDParser()
        parsed_jd: ParsedJob = jd_parser.parse(_REAL_JD_TEXT)

        job: Job = Job(
            title=parsed_jd.title or "Python Backend Engineer",
            description=parsed_jd.description or _REAL_JD_TEXT[:300],
            responsibilities=parsed_jd.responsibilities,
            company_name=parsed_jd.company_name or "TechStart",
            skill_requirements=(
                [SkillRequirement(name=s, is_required=True) for s in parsed_jd.required_skills]
                + [SkillRequirement(name=s, is_required=False) for s in parsed_jd.preferred_skills]
            ),
        )

        resume_parser: AccurateResumeParser = AccurateResumeParser()
        parsed_resume = resume_parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        engine: MatchingEngine = MatchingEngine(use_semantic=False)
        result: MatchResult = engine.match_from_parsed(parsed_resume, job)

        assert result.overall_score >= 0.0
        assert result.candidate_name != ""
        assert result.score_breakdown is not None
        assert result.job_title == (parsed_jd.title or "Python Backend Engineer")


# ---------------------------------------------------------------------------
# EmbeddingSkillScorer integration
# ---------------------------------------------------------------------------

class TestEmbeddingSkillScorerIntegration:
    """Integration tests for EmbeddingSkillScorer wired into match_from_parsed."""

    def test_score_skills_returns_tuple_with_matches_and_score(self) -> None:
        """EmbeddingSkillScorer.score_skills() returns (list[SkillMatch], float)."""
        from src.core.matching.skill_scorer import EmbeddingSkillScorer
        from src.data.models import SkillMatch

        scorer: EmbeddingSkillScorer = EmbeddingSkillScorer()
        required: list[str] = ["Python", "FastAPI", "PostgreSQL"]
        preferred: list[str] = ["Docker", "Kubernetes"]
        candidate: list[str] = ["python", "fastapi", "sql", "docker"]

        matches: list[SkillMatch]
        score: float
        matches, score = scorer.score_skills(required, preferred, candidate)

        assert isinstance(matches, list)
        assert all(isinstance(m, SkillMatch) for m in matches)
        assert 0.0 <= score <= 1.0
        assert len(matches) == len(required) + len(preferred)

    def test_match_from_parsed_skill_score_with_embedding_scorer(self) -> None:
        """match_from_parsed() uses EmbeddingSkillScorer for skill matching."""
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser
        from src.data.models.job import Job, SkillRequirement
        from src.core.matching.matching_engine import MatchingEngine, MatchResult

        job: Job = Job(
            title="Python Engineer",
            description="We need a Python backend engineer with FastAPI experience.",
            responsibilities=["Build Python APIs"],
            company_name="TestCo",
            skill_requirements=[
                SkillRequirement(name="python", is_required=True),
                SkillRequirement(name="fastapi", is_required=True),
            ],
        )

        resume_parser: AccurateResumeParser = AccurateResumeParser()
        parsed_resume = resume_parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        engine: MatchingEngine = MatchingEngine(use_semantic=False)
        result: MatchResult = engine.match_from_parsed(parsed_resume, job)

        assert result.overall_score >= 0.0
        assert result.skills_score >= 0.0
        assert len(result.skill_matches) >= 2
        req_matches: list = [m for m in result.skill_matches if m.required]
        assert len(req_matches) == 2

    def test_match_from_parsed_skill_matches_have_correct_fields(self) -> None:
        """SkillMatch objects from EmbeddingSkillScorer have all expected fields."""
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser
        from src.data.models.job import Job, SkillRequirement
        from src.core.matching.matching_engine import MatchingEngine, MatchResult
        from src.data.models import SkillMatch

        job: Job = Job(
            title="ML Engineer",
            description="ML engineer role requiring Python and machine learning.",
            responsibilities=["Build ML models"],
            company_name="AI Corp",
            skill_requirements=[
                SkillRequirement(name="python", is_required=True),
                SkillRequirement(name="machine learning", is_required=False),
            ],
        )

        resume_parser: AccurateResumeParser = AccurateResumeParser()
        parsed_resume = resume_parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        engine: MatchingEngine = MatchingEngine(use_semantic=False)
        result: MatchResult = engine.match_from_parsed(parsed_resume, job)

        for sm in result.skill_matches:
            assert isinstance(sm, SkillMatch)
            assert isinstance(sm.skill_name, str)
            assert isinstance(sm.required, bool)
            assert isinstance(sm.match_score, float)
            assert 0.0 <= sm.match_score <= 1.0


# ---------------------------------------------------------------------------
# TestDomainAwareExperienceScorerIntegration
# ---------------------------------------------------------------------------

class TestDomainAwareExperienceScorerIntegration:
    """End-to-end: DomainAwareExperienceScorer inside MatchingEngine.match_from_parsed()."""

    def test_experience_match_object_populated(self) -> None:
        """match_from_parsed() returns a non-None ExperienceMatch with score in [0,1]."""
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser
        from src.data.models.job import Job, ExperienceRequirement
        from src.core.matching.matching_engine import MatchingEngine, MatchResult
        from src.data.models import ExperienceMatch

        job: Job = Job(
            title="Python Backend Engineer",
            description="Python backend role with FastAPI and PostgreSQL.",
            responsibilities=[
                "Build REST APIs with FastAPI",
                "Write and maintain Python services",
                "Code review and mentoring",
            ],
            company_name="AI Corp",
            experience_requirement=ExperienceRequirement(minimum_years=2.0),
        )

        resume_parser: AccurateResumeParser = AccurateResumeParser()
        parsed_resume: ParsedResume = resume_parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        engine: MatchingEngine = MatchingEngine(use_semantic=False, use_bias_detection=False)
        result: MatchResult = engine.match_from_parsed(parsed_resume, job)

        assert result.experience_match is not None
        exp: ExperienceMatch = result.experience_match
        assert 0.0 <= exp.score <= 1.0
        assert exp.required_years == pytest.approx(2.0)
        assert exp.candidate_years >= 0.0
        assert 0.0 <= result.experience_score <= 1.0

    def test_relevant_titles_matched_is_list(self) -> None:
        """relevant_titles_matched on ExperienceMatch is a list (may be empty for mismatched roles)."""
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser
        from src.data.models.job import Job, ExperienceRequirement
        from src.core.matching.matching_engine import MatchingEngine, MatchResult
        from src.data.models import ExperienceMatch

        job: Job = Job(
            title="Python ML Engineer",
            description="Machine learning engineering role.",
            responsibilities=[
                "Develop ML pipelines in Python",
                "Build and evaluate models",
            ],
            company_name="AI Corp",
            experience_requirement=ExperienceRequirement(minimum_years=1.0),
        )

        resume_parser: AccurateResumeParser = AccurateResumeParser()
        parsed_resume: ParsedResume = resume_parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        engine: MatchingEngine = MatchingEngine(use_semantic=False, use_bias_detection=False)
        result: MatchResult = engine.match_from_parsed(parsed_resume, job)

        assert result.experience_match is not None
        exp: ExperienceMatch = result.experience_match
        assert isinstance(exp.relevant_titles_matched, list)
        # Verify the domain-aware scorer was actually invoked (not silently bypassed)
        assert result.experience_score > 0.0
        # All entries in relevant_titles_matched must be non-empty strings
        for title in exp.relevant_titles_matched:
            assert isinstance(title, str) and len(title) > 0

    def test_overall_score_includes_experience_component(self) -> None:
        """overall_score is non-zero and incorporates the experience component."""
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume
        from src.data.models.job import Job
        from src.core.matching.matching_engine import MatchingEngine, MatchResult

        job: Job = Job(
            title="Software Engineer",
            description="General software engineering role.",
            responsibilities=["write code", "review PRs"],
            company_name="AI Corp",
        )

        resume_parser: AccurateResumeParser = AccurateResumeParser()
        parsed_resume: ParsedResume = resume_parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        engine: MatchingEngine = MatchingEngine(use_semantic=False, use_bias_detection=False)
        result: MatchResult = engine.match_from_parsed(parsed_resume, job)

        # No experience requirement → score = 1.0 (has experience) or 0.5 (no experience)
        assert 0.0 <= result.overall_score <= 1.0
        assert result.experience_score in {0.5, 1.0}  # no requirement path
        assert result.overall_score > 0.0


# ---------------------------------------------------------------------------
# TestEmbeddingEducationScorerIntegration
# ---------------------------------------------------------------------------

class TestEmbeddingEducationScorerIntegration:
    """End-to-end: EmbeddingEducationScorer inside MatchingEngine.match_from_parsed()."""

    def test_education_match_object_populated(self) -> None:
        """match_from_parsed() returns a non-None EducationMatch with score in [0,1]."""
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume
        from src.data.models.job import Job, EducationRequirement
        from src.core.matching.matching_engine import MatchingEngine, MatchResult
        from src.data.models import EducationMatch

        job: Job = Job(
            title="Python Backend Engineer",
            description="Python backend role requiring a bachelor's degree in CS or related field.",
            responsibilities=["Build APIs", "Write Python services"],
            company_name="AI Corp",
            education_requirement=EducationRequirement(minimum_degree="bachelor"),
        )

        resume_parser: AccurateResumeParser = AccurateResumeParser()
        parsed_resume: ParsedResume = resume_parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        engine: MatchingEngine = MatchingEngine(use_semantic=False, use_bias_detection=False)
        result: MatchResult = engine.match_from_parsed(parsed_resume, job)

        assert result.education_match is not None
        edu: EducationMatch = result.education_match
        assert 0.0 <= edu.score <= 1.0
        assert edu.required_degree == "bachelor"
        assert 0.0 <= result.education_score <= 1.0

    def test_field_match_flag_is_bool(self) -> None:
        """EducationMatch.field_match is a bool (True or False, never None)."""
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume
        from src.data.models.job import Job, EducationRequirement
        from src.core.matching.matching_engine import MatchingEngine, MatchResult
        from src.data.models import EducationMatch

        job: Job = Job(
            title="Machine Learning Engineer",
            description="ML engineering role with Python and data science background preferred.",
            responsibilities=["Build ML pipelines", "Train and evaluate models"],
            company_name="AI Corp",
            education_requirement=EducationRequirement(minimum_degree="bachelor"),
        )

        resume_parser: AccurateResumeParser = AccurateResumeParser()
        parsed_resume: ParsedResume = resume_parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        engine: MatchingEngine = MatchingEngine(use_semantic=False, use_bias_detection=False)
        result: MatchResult = engine.match_from_parsed(parsed_resume, job)

        assert result.education_match is not None
        edu: EducationMatch = result.education_match
        assert isinstance(edu.field_match, bool)

    def test_overall_score_includes_education_component(self) -> None:
        """overall_score is positive and the education component score is in [0,1]."""
        from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume
        from src.data.models.job import Job
        from src.core.matching.matching_engine import MatchingEngine, MatchResult

        job: Job = Job(
            title="Software Engineer",
            description="General software engineering role.",
            responsibilities=["write code", "review PRs"],
            company_name="AI Corp",
        )

        resume_parser: AccurateResumeParser = AccurateResumeParser()
        parsed_resume: ParsedResume = resume_parser.parse(RESUMES_DIR / "vivek_resume.pdf")

        engine: MatchingEngine = MatchingEngine(use_semantic=False, use_bias_detection=False)
        result: MatchResult = engine.match_from_parsed(parsed_resume, job)

        assert 0.0 <= result.overall_score <= 1.0
        assert result.overall_score > 0.0
        assert 0.0 <= result.education_score <= 1.0
        # No education requirement → score = 1.0 (has degree) or 0.7 (no degree)
        assert result.education_score in {0.7, 1.0}
