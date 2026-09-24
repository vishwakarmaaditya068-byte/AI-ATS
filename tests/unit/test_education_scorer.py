"""Unit tests for EmbeddingEducationScorer."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.matching.education_scorer import EmbeddingEducationScorer
from src.data.models import EducationMatch
from src.ml.nlp.accurate_resume_parser import EducationEntry


# ---------------------------------------------------------------------------
# Fake embedding model
# ---------------------------------------------------------------------------

class _FakeEmbeddingModel:
    """
    Deterministic stub: assigns a 4-D unit vector based on the first matching
    keyword found in the text. Only single-string encode() calls are used by
    EmbeddingEducationScorer (field text + job context).

    Vectors (all unit-norm):
      computer / python: [1.0, 0.0, 0.0, 0.0]  — CS / tech direction
      data:              [0.8, 0.0, 0.6, 0.0]  — partially tech (sim with CS=0.8)
      business:          [0.0, 1.0, 0.0, 0.0]  — business direction
      arts:              [0.0, 0.0, 1.0, 0.0]  — humanities direction
      default:           [0.0, 0.0, 0.0, 1.0]  — orthogonal to all above
    """

    _VECS: dict[str, np.ndarray] = {
        "computer": np.array([1.0, 0.0, 0.0, 0.0]),
        "python":   np.array([1.0, 0.0, 0.0, 0.0]),
        "data":     np.array([0.8, 0.0, 0.6, 0.0]),
        "business": np.array([0.0, 1.0, 0.0, 0.0]),
        "arts":     np.array([0.0, 0.0, 1.0, 0.0]),
    }

    @classmethod
    def _vec(cls, text: str) -> np.ndarray:
        # First matching keyword wins — do not put two keywords in the same test text.
        t = text.lower()
        for kw, vec in cls._VECS.items():
            if kw in t:
                v = vec.copy()
                n = float(np.linalg.norm(v))
                return v / n if n > 0 else v
        return np.array([0.0, 0.0, 0.0, 1.0])

    def encode(self, texts: str | list[str], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            return self._vec(texts)
        return np.array([self._vec(t) for t in texts])


_FAKE_MODEL = _FakeEmbeddingModel()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edu(degree: str, institution: str = "Uni") -> EducationEntry:
    return EducationEntry(degree=degree, institution=institution)


@pytest.fixture
def scorer() -> EmbeddingEducationScorer:
    return EmbeddingEducationScorer(embedding_model=_FAKE_MODEL)


# ---------------------------------------------------------------------------
# TestExtractField — static helper
# ---------------------------------------------------------------------------

class TestExtractField:
    def test_btech_field_extracted(self) -> None:
        assert EmbeddingEducationScorer._extract_field("B.Tech Computer Science") == "Computer Science"

    def test_master_of_field_extracted(self) -> None:
        result = EmbeddingEducationScorer._extract_field("Master of Science in Data Science")
        # "Master", "of", "in" are stripped; "Science" remains but "data" keyword is present
        assert "Data" in result or "data" in result.lower()

    def test_degree_only_returns_empty(self) -> None:
        assert EmbeddingEducationScorer._extract_field("Bachelor") == ""

    def test_mba_returns_empty(self) -> None:
        assert EmbeddingEducationScorer._extract_field("MBA") == ""

    def test_bachelor_of_arts_returns_arts(self) -> None:
        result = EmbeddingEducationScorer._extract_field("Bachelor of Arts")
        assert result == "Arts"


# ---------------------------------------------------------------------------
# TestNormalizeDegreeLevel — static helper
# ---------------------------------------------------------------------------

class TestNormalizeDegreeLevel:
    def test_btech_maps_to_bachelor(self) -> None:
        assert EmbeddingEducationScorer._normalize_degree_level("B.Tech Computer Science") == "bachelor"

    def test_master_maps_to_master(self) -> None:
        assert EmbeddingEducationScorer._normalize_degree_level("Master of Science in CS") == "master"

    def test_mba_maps_to_mba(self) -> None:
        assert EmbeddingEducationScorer._normalize_degree_level("MBA") == "mba"

    def test_phd_maps_to_phd(self) -> None:
        assert EmbeddingEducationScorer._normalize_degree_level("Ph.D. Computer Science") == "phd"

    def test_unknown_falls_through(self) -> None:
        result = EmbeddingEducationScorer._normalize_degree_level("Certification in AWS")
        # Should not crash; returns lowercased input
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestScoreWithEmbeddings
# ---------------------------------------------------------------------------

class TestScoreWithEmbeddings:
    def test_matching_field_and_level_scores_one(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        """Bachelor CS for Python Backend job: meets level AND field → score = 1.0."""
        entries = [_edu("Bachelor Computer Science")]
        _, score = scorer.score_education(
            entries, required_degree="bachelor",
            job_title="python backend", job_description="",
        )
        # degree_score = 1.0, field_sim = 1.0 → combined = 1.0
        assert score == 1.0

    def test_mismatched_field_reduces_score(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        """Bachelor Business for Python Backend job: level OK but wrong field → score < 1.0."""
        from src.utils.constants import EDU_FIELD_WEIGHT
        entries = [_edu("Bachelor Business Administration")]
        _, score = scorer.score_education(
            entries, required_degree="bachelor",
            job_title="python backend", job_description="",
        )
        # degree_score = 1.0, field_sim = 0.0 → combined = 1.0 * (1-0.4) + 0.0 * 0.4 = 0.6
        expected = round(1.0 * (1.0 - EDU_FIELD_WEIGHT), 3)
        assert score == expected

    def test_partial_field_relevance(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        """Bachelor Data Science for Python Backend job: partial relevance (sim=0.8)."""
        from src.utils.constants import EDU_FIELD_WEIGHT
        entries = [_edu("Bachelor Data Science")]
        _, score = scorer.score_education(
            entries, required_degree="bachelor",
            job_title="python backend", job_description="",
        )
        # degree_score = 1.0, field_sim = 0.8
        expected = round(1.0 * (1.0 - EDU_FIELD_WEIGHT) + 0.8 * EDU_FIELD_WEIGHT, 3)
        assert score == expected

    def test_lower_degree_level_with_matching_field(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        """Bachelor CS for Master-required Python job: level below but field matches."""
        from src.utils.constants import EDU_FIELD_WEIGHT
        entries = [_edu("Bachelor Computer Science")]
        _, score = scorer.score_education(
            entries, required_degree="master",
            job_title="python backend", job_description="",
        )
        # bachelor=4, master=5 → one level below → degree_score = 0.7
        # field_sim = 1.0 → combined = 0.7 * 0.6 + 1.0 * 0.4 = 0.82
        expected = round(0.7 * (1.0 - EDU_FIELD_WEIGHT) + 1.0 * EDU_FIELD_WEIGHT, 3)
        assert score == expected

    def test_no_field_in_degree_uses_degree_score_only(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        """When degree has no field text after extraction, skip embedding."""
        entries = [_edu("Bachelor")]
        _, score = scorer.score_education(
            entries, required_degree="bachelor",
            job_title="python backend", job_description="",
        )
        # No field → degree_score = 1.0 returned unchanged
        assert score == 1.0

    def test_returns_education_match_object(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        entries = [_edu("Bachelor Computer Science")]
        match, _ = scorer.score_education(
            entries, required_degree="bachelor",
            job_title="python", job_description="",
        )
        assert isinstance(match, EducationMatch)
        assert match.required_degree == "bachelor"
        assert match.candidate_degree == "Bachelor Computer Science"


# ---------------------------------------------------------------------------
# TestFieldMatchFlag
# ---------------------------------------------------------------------------

class TestFieldMatchFlag:
    def test_field_match_true_when_sim_above_threshold(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        """CS field vs python job: sim=1.0 >= EDU_FIELD_MATCH_THRESHOLD → field_match=True."""
        entries = [_edu("Bachelor Computer Science")]
        match, _ = scorer.score_education(
            entries, required_degree="bachelor",
            job_title="python backend", job_description="",
        )
        assert match.field_match is True

    def test_field_match_false_when_sim_below_threshold(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        """Business field vs python job: sim=0.0 < threshold → field_match=False."""
        entries = [_edu("Bachelor Business Administration")]
        match, _ = scorer.score_education(
            entries, required_degree="bachelor",
            job_title="python backend", job_description="",
        )
        assert match.field_match is False

    def test_field_match_false_when_no_field_extracted(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        """No field text → embedding skipped → field_match=False."""
        entries = [_edu("Bachelor")]
        match, _ = scorer.score_education(
            entries, required_degree="bachelor",
            job_title="python backend", job_description="",
        )
        assert match.field_match is False


# ---------------------------------------------------------------------------
# TestNoRequirementAndEdgeCases
# ---------------------------------------------------------------------------

class TestNoRequirementAndEdgeCases:
    def test_no_requirement_with_degree_scores_one(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        entries = [_edu("Bachelor Computer Science")]
        _, score = scorer.score_education(
            entries, required_degree="",
            job_title="any", job_description="",
        )
        assert score == 1.0

    def test_no_requirement_without_degree_scores_07(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        _, score = scorer.score_education(
            [], required_degree="",
            job_title="any", job_description="",
        )
        assert score == pytest.approx(0.7)

    def test_no_candidate_degree_scores_03(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        _, score = scorer.score_education(
            [], required_degree="bachelor",
            job_title="python", job_description="",
        )
        assert score == pytest.approx(0.3)

    def test_fallback_without_model_uses_degree_score(self) -> None:
        """Without model, only degree-level comparison runs."""
        scorer_no_model = EmbeddingEducationScorer(embedding_model=None)
        scorer_no_model._model_load_failed = True
        entries = [_edu("Bachelor Computer Science")]
        _, score = scorer_no_model.score_education(
            entries, required_degree="bachelor",
            job_title="python", job_description="",
        )
        assert score == 1.0  # bachelor == bachelor → degree_score = 1.0

    def test_meets_requirement_set_correctly(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        entries = [_edu("Bachelor Computer Science")]
        match, _ = scorer.score_education(
            entries, required_degree="bachelor",
            job_title="python", job_description="",
        )
        assert match.meets_requirement is True

    def test_below_requirement_meets_requirement_false(
        self, scorer: EmbeddingEducationScorer
    ) -> None:
        entries = [_edu("Bachelor Computer Science")]
        match, _ = scorer.score_education(
            entries, required_degree="master",
            job_title="python", job_description="",
        )
        assert match.meets_requirement is False


# ---------------------------------------------------------------------------
# TestMatchingEngineWiring  (fast: no real model, no file I/O)
# ---------------------------------------------------------------------------

class TestMatchingEngineWiring:
    """Verify MatchingEngine.match_from_parsed() uses EmbeddingEducationScorer."""

    def test_field_mismatch_reduces_education_score(self) -> None:
        """Business degree against Python job should score lower than matching field."""
        from src.core.matching.matching_engine import MatchingEngine, MatchResult
        from src.core.matching.education_scorer import EmbeddingEducationScorer
        from src.ml.nlp.accurate_resume_parser import (
            ParsedResume, ContactInfo, EducationEntry, SkillCategory,
        )
        from src.data.models.job import Job, EducationRequirement

        parsed: ParsedResume = ParsedResume(
            contact=ContactInfo(name="Test Candidate"),
            skills=[SkillCategory(category="languages", skills=["python"])],
            education=[
                EducationEntry(
                    degree="Bachelor Business Administration",
                    institution="University",
                )
            ],
        )

        job: Job = Job(
            title="Python Backend Engineer",
            description="Python-focused backend engineering role.",
            responsibilities=["build python microservices"],
            company_name="AI Corp",
            education_requirement=EducationRequirement(minimum_degree="bachelor"),
        )

        engine: MatchingEngine = MatchingEngine(use_semantic=False, use_bias_detection=False)
        engine._education_scorer = EmbeddingEducationScorer(embedding_model=_FAKE_MODEL)
        engine._use_embedding_education_scorer = True
        result: MatchResult = engine.match_from_parsed(parsed, job)

        # Without field scoring: bachelor == bachelor → education_score = 1.0
        # With field scoring: business degree vs python job → field_sim = 0.0
        # combined = 1.0 * 0.6 + 0.0 * 0.4 = 0.6
        from src.utils.constants import EDU_FIELD_WEIGHT
        expected: float = round(1.0 * (1.0 - EDU_FIELD_WEIGHT), 3)
        assert result.education_score == expected

    def test_matching_field_education_match_has_field_match_true(self) -> None:
        """CS degree vs Python job: field_match should be True on EducationMatch."""
        from src.core.matching.matching_engine import MatchingEngine, MatchResult
        from src.core.matching.education_scorer import EmbeddingEducationScorer
        from src.ml.nlp.accurate_resume_parser import (
            ParsedResume, ContactInfo, EducationEntry, SkillCategory,
        )
        from src.data.models.job import Job, EducationRequirement
        from src.data.models import EducationMatch

        parsed: ParsedResume = ParsedResume(
            contact=ContactInfo(name="Test Candidate"),
            skills=[SkillCategory(category="languages", skills=["python"])],
            education=[
                EducationEntry(
                    degree="Bachelor Computer Science",
                    institution="University",
                )
            ],
        )

        job: Job = Job(
            title="Python Backend Engineer",
            description="Python-focused backend engineering role.",
            responsibilities=["build python services"],
            company_name="AI Corp",
            education_requirement=EducationRequirement(minimum_degree="bachelor"),
        )

        engine: MatchingEngine = MatchingEngine(use_semantic=False, use_bias_detection=False)
        engine._education_scorer = EmbeddingEducationScorer(embedding_model=_FAKE_MODEL)
        engine._use_embedding_education_scorer = True
        result: MatchResult = engine.match_from_parsed(parsed, job)

        assert result.education_match is not None
        edu_match: EducationMatch = result.education_match
        assert edu_match.field_match is True
