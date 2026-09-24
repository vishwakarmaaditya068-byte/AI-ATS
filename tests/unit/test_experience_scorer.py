"""Unit tests for DomainAwareExperienceScorer."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.matching.experience_scorer import DomainAwareExperienceScorer
from src.data.models import ExperienceMatch
from src.ml.nlp.accurate_resume_parser import ExperienceEntry


# ---------------------------------------------------------------------------
# Fake embedding model
# ---------------------------------------------------------------------------

class _FakeEmbeddingModel:
    """
    Deterministic stub: assigns a 4-D unit vector based on the first matching
    keyword found in the text.

    Vectors (all already unit-norm):
      python:   [1.0, 0.0, 0.0, 0.0]  sim with python job context = 1.0
      java:     [0.0, 1.0, 0.0, 0.0]  sim with python job context = 0.0
      data:     [0.6, 0.0, 0.8, 0.0]  sim with python job context = 0.6  (0.6²+0.8²=1.0)
      frontend: [0.0, 0.0, 0.0, 1.0]  sim with python job context = 0.0
      default:  [0.0, 0.0, 0.0, 1.0]
    """

    _VECS: dict[str, np.ndarray] = {
        "python":   np.array([1.0, 0.0, 0.0, 0.0]),
        "java":     np.array([0.0, 1.0, 0.0, 0.0]),
        "data":     np.array([0.6, 0.0, 0.8, 0.0]),
        "frontend": np.array([0.0, 0.0, 0.0, 1.0]),
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

def _entry(
    title: str, duration: str, bullets: list[str] | None = None
) -> ExperienceEntry:
    return ExperienceEntry(
        title=title, company="Corp", duration=duration, bullets=bullets or []
    )


@pytest.fixture
def scorer() -> DomainAwareExperienceScorer:
    return DomainAwareExperienceScorer(embedding_model=_FAKE_MODEL)


# ---------------------------------------------------------------------------
# TestScoreFromYears — static helper, no model needed
# ---------------------------------------------------------------------------

class TestScoreFromYears:
    def test_meets_requirement(self) -> None:
        assert DomainAwareExperienceScorer._score_from_years(5.0, 3.0) == 1.0

    def test_exactly_meets_requirement(self) -> None:
        assert DomainAwareExperienceScorer._score_from_years(3.0, 3.0) == 1.0

    def test_within_70_percent(self) -> None:
        # 2.5 >= 3.0 * 0.7 = 2.1 → 0.7 + 0.3 * (2.5/3.0)
        result = DomainAwareExperienceScorer._score_from_years(2.5, 3.0)
        assert abs(result - (0.7 + 0.3 * (2.5 / 3.0))) < 1e-9

    def test_below_70_percent_with_some_years(self) -> None:
        # 1.0 < 2.1 → 0.5 * (1.0/3.0)
        result = DomainAwareExperienceScorer._score_from_years(1.0, 3.0)
        assert abs(result - 0.5 * (1.0 / 3.0)) < 1e-9

    def test_zero_years(self) -> None:
        assert DomainAwareExperienceScorer._score_from_years(0.0, 3.0) == 0.0


# ---------------------------------------------------------------------------
# TestScoreWithEmbeddings
# ---------------------------------------------------------------------------

class TestScoreWithEmbeddings:
    def test_fully_relevant_experience_scores_one(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        """5 years pure Python experience vs Python job → all relevant → score 1.0."""
        entries = [_entry("Senior Python Developer", "5 years", ["built fastapi services"])]
        _, score = scorer.score_experience(
            entries, required_years=3.0,
            job_title="python backend", responsibilities=[],
        )
        assert score == 1.0

    def test_irrelevant_experience_scores_zero(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        """Java experience vs Python job → 0 relevant years → score 0.0."""
        entries = [_entry("Java Developer", "5 years", ["built spring services"])]
        _, score = scorer.score_experience(
            entries, required_years=3.0,
            job_title="python backend", responsibilities=[],
        )
        assert score == 0.0

    def test_mixed_experience_uses_weighted_years(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        """2 years Python (sim=1.0) + 3 years Java (sim=0.0) → 2.0 relevant years.

        2.0 < 3.0*0.7=2.1 → score = 0.5 * (2.0/3.0) = 0.333
        """
        entries = [
            _entry("Python Developer", "2 years"),
            _entry("Java Developer", "3 years"),
        ]
        _, score = scorer.score_experience(
            entries, required_years=3.0,
            job_title="python backend", responsibilities=[],
        )
        assert score == round(0.5 * (2.0 / 3.0), 3)

    def test_partial_relevance_data_entry(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        """Data Scientist entry (sim=0.6) × 4 years = 2.4 relevant years.

        2.4 >= 2.1 (3*0.7) → score = 0.7 + 0.3*(2.4/3.0)
        """
        entries = [_entry("Data Scientist", "4 years")]
        _, score = scorer.score_experience(
            entries, required_years=3.0,
            job_title="python backend", responsibilities=[],
        )
        expected = round(0.7 + 0.3 * (2.4 / 3.0), 3)
        assert score == expected

    def test_returns_experience_match_object(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        entries = [_entry("Python Developer", "3 years")]
        match, _ = scorer.score_experience(
            entries, required_years=3.0,
            job_title="python", responsibilities=[],
        )
        assert isinstance(match, ExperienceMatch)
        assert match.required_years == 3.0
        assert match.candidate_years == pytest.approx(3.0)

    def test_responsibilities_included_in_job_context(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        """Responsibilities text is used as part of the job context encoding."""
        entries = [_entry("Python Developer", "3 years")]
        match, score = scorer.score_experience(
            entries, required_years=3.0,
            job_title="engineer", responsibilities=["build python services", "maintain apis"],
        )
        # job_context = "engineer build python services maintain apis"
        # _FakeEmbeddingModel finds "python" in combined string → job_emb = [1,0,0,0]
        # python entry sim = 1.0 → score = 1.0
        assert score == 1.0


# ---------------------------------------------------------------------------
# TestRelevantTitlesMatched
# ---------------------------------------------------------------------------

class TestRelevantTitlesMatched:
    def test_relevant_title_added(self, scorer: DomainAwareExperienceScorer) -> None:
        """Python entry (sim=1.0 >= EXP_RELEVANCE_TITLE_THRESHOLD=0.4) → title included."""
        entries = [_entry("Senior Python Engineer", "2 years")]
        match, _ = scorer.score_experience(
            entries, required_years=3.0,
            job_title="python backend", responsibilities=[],
        )
        assert "Senior Python Engineer" in match.relevant_titles_matched

    def test_irrelevant_title_excluded(self, scorer: DomainAwareExperienceScorer) -> None:
        """Java entry (sim=0.0 < 0.4) → title NOT included."""
        entries = [_entry("Java Backend Engineer", "2 years")]
        match, _ = scorer.score_experience(
            entries, required_years=3.0,
            job_title="python backend", responsibilities=[],
        )
        assert "Java Backend Engineer" not in match.relevant_titles_matched

    def test_partial_relevance_title_added(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        """Data entry (sim=0.6 >= 0.4) → title included."""
        entries = [_entry("Data Scientist", "2 years")]
        match, _ = scorer.score_experience(
            entries, required_years=3.0,
            job_title="python backend", responsibilities=[],
        )
        assert "Data Scientist" in match.relevant_titles_matched

    def test_multiple_entries_selective_titles(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        """Only entries meeting the threshold appear in relevant_titles_matched."""
        entries = [
            _entry("Python Developer", "2 years"),
            _entry("Java Developer", "2 years"),
            _entry("Data Scientist", "2 years"),
        ]
        match, _ = scorer.score_experience(
            entries, required_years=3.0,
            job_title="python backend", responsibilities=[],
        )
        assert "Python Developer" in match.relevant_titles_matched
        assert "Java Developer" not in match.relevant_titles_matched
        assert "Data Scientist" in match.relevant_titles_matched


# ---------------------------------------------------------------------------
# TestNoRequirementAndFallback
# ---------------------------------------------------------------------------

class TestNoRequirementAndFallback:
    def test_no_requirement_has_experience_scores_one(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        entries = [_entry("Python Developer", "3 years")]
        _, score = scorer.score_experience(
            entries, required_years=0.0, job_title="any", responsibilities=[]
        )
        assert score == 1.0

    def test_no_requirement_no_experience_scores_half(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        _, score = scorer.score_experience(
            [], required_years=0.0, job_title="any", responsibilities=[]
        )
        assert score == 0.5

    def test_fallback_without_model_uses_total_years(self) -> None:
        """Without embedding model, total years tiers apply (no domain weighting)."""
        scorer_no_model = DomainAwareExperienceScorer(embedding_model=None)
        scorer_no_model._model_load_failed = True  # prevent lazy-load attempt
        entries = [_entry("Python Developer", "5 years")]
        _, score = scorer_no_model.score_experience(
            entries, required_years=3.0, job_title="x", responsibilities=[]
        )
        assert score == 1.0  # total 5 years >= 3 → 1.0

    def test_empty_entries_with_model_skips_embedding_path(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        """Empty experience list with a requirement → score 0.0 (no years, no embedding)."""
        _, score = scorer.score_experience(
            [], required_years=3.0, job_title="python", responsibilities=[]
        )
        assert score == 0.0


# ---------------------------------------------------------------------------
# TestExperienceMatchFields
# ---------------------------------------------------------------------------

class TestExperienceMatchFields:
    def test_meets_minimum_true(self, scorer: DomainAwareExperienceScorer) -> None:
        entries = [_entry("Python Developer", "5 years")]
        match, _ = scorer.score_experience(
            entries, required_years=3.0, job_title="python", responsibilities=[]
        )
        assert match.meets_minimum is True

    def test_meets_minimum_false(self, scorer: DomainAwareExperienceScorer) -> None:
        entries = [_entry("Python Developer", "1 year")]
        match, _ = scorer.score_experience(
            entries, required_years=3.0, job_title="python", responsibilities=[]
        )
        assert match.meets_minimum is False

    def test_years_difference(self, scorer: DomainAwareExperienceScorer) -> None:
        entries = [_entry("Python Developer", "5 years")]
        match, _ = scorer.score_experience(
            entries, required_years=3.0, job_title="python", responsibilities=[]
        )
        assert match.years_difference == pytest.approx(2.0)

    def test_meets_minimum_false_for_irrelevant_experience(
        self, scorer: DomainAwareExperienceScorer
    ) -> None:
        """5 years Java vs Python job requiring 3 years → meets_minimum must be False.

        Total candidate years (5) exceeds required (3), but relevant years are 0
        because Java [0,1,0,0] has zero cosine similarity with Python job [1,0,0,0].
        meets_minimum should reflect domain-relevant years, not raw totals.
        """
        entries = [_entry("Java Developer", "5 years", ["built spring services"])]
        match, _ = scorer.score_experience(
            entries, required_years=3.0,
            job_title="python backend", responsibilities=[],
        )
        assert match.meets_minimum is False
        assert match.score == 0.0


# ---------------------------------------------------------------------------
# TestMatchingEngineWiring  (fast: no real model, no file I/O)
# ---------------------------------------------------------------------------

class TestMatchingEngineWiring:
    """Verify MatchingEngine.match_from_parsed() uses DomainAwareExperienceScorer."""

    def test_experience_score_reflects_domain_relevance(self) -> None:
        """Java experience against a Python job should score lower than total-years alone."""
        from src.core.matching.matching_engine import MatchingEngine, MatchResult
        from src.core.matching.experience_scorer import DomainAwareExperienceScorer
        from src.ml.nlp.accurate_resume_parser import (
            ParsedResume, ContactInfo, ExperienceEntry, SkillCategory,
        )
        from src.data.models.job import Job, ExperienceRequirement

        # Build a ParsedResume with 5 years of Java experience
        parsed: ParsedResume = ParsedResume(
            contact=ContactInfo(name="Test Candidate"),
            skills=[SkillCategory(category="languages", skills=["java", "spring"])],
            experience=[
                ExperienceEntry(
                    title="Java Backend Developer",
                    company="Corp",
                    duration="5 years",
                    bullets=["built spring microservices"],
                )
            ],
        )

        job: Job = Job(
            title="Python Backend Engineer",
            description="Python-focused backend role.",
            responsibilities=["build python microservices", "maintain fastapi APIs"],
            company_name="AI Corp",
            experience_requirement=ExperienceRequirement(minimum_years=3.0),
        )

        engine: MatchingEngine = MatchingEngine(use_semantic=False, use_bias_detection=False)
        engine._experience_scorer = DomainAwareExperienceScorer(embedding_model=_FAKE_MODEL)
        engine._use_domain_experience_scorer = True
        result: MatchResult = engine.match_from_parsed(parsed, job)

        # Without domain weighting: 5 years Java → experience_score = 1.0 (5 >= 3)
        # With domain weighting: java entry [0,1,0,0] vs python job [1,0,0,0] → sim=0.0 → score=0.0
        assert result.experience_score == 0.0

    def test_experience_match_has_relevant_titles(self) -> None:
        """relevant_titles_matched on ExperienceMatch is populated for domain-relevant entries."""
        from src.core.matching.matching_engine import MatchingEngine, MatchResult
        from src.core.matching.experience_scorer import DomainAwareExperienceScorer
        from src.ml.nlp.accurate_resume_parser import (
            ParsedResume, ContactInfo, ExperienceEntry, SkillCategory,
        )
        from src.data.models.job import Job, ExperienceRequirement
        from src.data.models import ExperienceMatch

        parsed: ParsedResume = ParsedResume(
            contact=ContactInfo(name="Test Candidate"),
            skills=[SkillCategory(category="languages", skills=["python"])],
            experience=[
                ExperienceEntry(
                    title="Python Backend Developer",
                    company="Corp",
                    duration="4 years",
                    bullets=["built fastapi services"],
                )
            ],
        )

        job: Job = Job(
            title="Python Engineer",
            description="Python role.",
            responsibilities=["build python services"],
            company_name="AI Corp",
            experience_requirement=ExperienceRequirement(minimum_years=3.0),
        )

        engine: MatchingEngine = MatchingEngine(use_semantic=False, use_bias_detection=False)
        engine._experience_scorer = DomainAwareExperienceScorer(embedding_model=_FAKE_MODEL)
        engine._use_domain_experience_scorer = True
        result: MatchResult = engine.match_from_parsed(parsed, job)

        assert result.experience_match is not None
        exp_match: ExperienceMatch = result.experience_match
        assert "Python Backend Developer" in exp_match.relevant_titles_matched
