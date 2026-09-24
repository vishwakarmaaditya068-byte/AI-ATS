"""
Unit tests for EmbeddingSkillScorer.

Uses _FakeEmbeddingModel with pre-computed unit vectors so that cosine
similarity values are deterministic and threshold-crossing is exact.

Vector layout (4-dimensional unit vectors):
    python      : [1.000, 0.000, 0.000, 0.000]
    python3     : [0.998, 0.063, 0.000, 0.000]  -> dot(python) = 0.998 >= 0.92 (exact)
    pytorch     : [0.780, 0.625, 0.000, 0.000]  -> dot(python) = 0.780 >= 0.75 (strong partial)
    tensorflow  : [0.630, 0.777, 0.000, 0.000]  -> dot(python) = 0.630 >= 0.60 (weak partial)
    java        : [0.000, 1.000, 0.000, 0.000]  -> dot(python) = 0.000 < 0.60 (no match)
    excel       : [0.000, 0.000, 1.000, 0.000]  -> dot(python) = 0.000 < 0.60 (no match)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.matching.skill_scorer import EmbeddingSkillScorer, get_embedding_skill_scorer
from src.data.models import SkillMatch


class _FakeEmbeddingModel:
    """Deterministic unit-vector embeddings keyed by skill name (lowercase)."""

    _VECS: dict[str, np.ndarray] = {
        "python":     np.array([1.000, 0.000, 0.000, 0.000]),
        "python3":    np.array([0.998, 0.063, 0.000, 0.000]),
        "pytorch":    np.array([0.780, 0.625, 0.000, 0.000]),
        "tensorflow": np.array([0.630, 0.777, 0.000, 0.000]),
        "java":       np.array([0.000, 1.000, 0.000, 0.000]),
        "excel":      np.array([0.000, 0.000, 1.000, 0.000]),
    }
    _DEFAULT: np.ndarray = np.array([0.000, 0.000, 0.000, 1.000])

    def encode(
        self,
        texts: str | list[str],
        normalize: bool = True,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(texts, str):
            return self._VECS.get(texts.lower(), self._DEFAULT).copy()
        return np.array([self.encode(t, normalize) for t in texts])

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))


def _scorer() -> EmbeddingSkillScorer:
    return EmbeddingSkillScorer(embedding_model=_FakeEmbeddingModel())


# ---------------------------------------------------------------------------
# TestScoreWithEmbeddings
# ---------------------------------------------------------------------------

class TestScoreWithEmbeddings:
    def test_exact_string_match_returns_score_1(self) -> None:
        """Candidate has exact skill name — fast-path exact string match."""
        scorer: EmbeddingSkillScorer = _scorer()
        matches, score = scorer.score_skills(["python"], [], ["python"])
        assert matches[0].match_score == 1.0
        assert matches[0].candidate_has_skill is True

    def test_case_insensitive_exact_match(self) -> None:
        """'Python' (title case) candidate skill matches 'python' JD skill."""
        scorer: EmbeddingSkillScorer = _scorer()
        matches, score = scorer.score_skills(["python"], [], ["Python"])
        assert matches[0].match_score == 1.0

    def test_alias_above_exact_threshold_treated_as_exact(self) -> None:
        """'python3' has sim=0.998 >= 0.92 -> treated as exact match (score 1.0)."""
        scorer: EmbeddingSkillScorer = _scorer()
        matches, score = scorer.score_skills(["python"], [], ["python3"])
        assert matches[0].match_score == 1.0
        assert matches[0].candidate_has_skill is True

    def test_strong_partial_match_score_0_8(self) -> None:
        """'pytorch' has sim=0.78 >= 0.75 -> strong partial match (score 0.8)."""
        scorer: EmbeddingSkillScorer = _scorer()
        matches, score = scorer.score_skills(["python"], [], ["pytorch"])
        assert matches[0].match_score == pytest.approx(0.8)
        assert matches[0].partial_match is True
        assert matches[0].candidate_has_skill is False

    def test_weak_partial_match_score_0_5(self) -> None:
        """'tensorflow' has sim=0.63 >= 0.60 -> weak partial match (score 0.5)."""
        scorer: EmbeddingSkillScorer = _scorer()
        matches, score = scorer.score_skills(["python"], [], ["tensorflow"])
        assert matches[0].match_score == pytest.approx(0.5)
        assert matches[0].partial_match is True

    def test_no_match_below_weak_threshold_score_0(self) -> None:
        """'java' has sim=0.0 < 0.60 -> no match (score 0.0)."""
        scorer: EmbeddingSkillScorer = _scorer()
        matches, score = scorer.score_skills(["python"], [], ["java"])
        assert matches[0].match_score == 0.0
        assert matches[0].partial_match is False
        assert matches[0].candidate_has_skill is False

    def test_empty_candidate_skills_returns_zero_score(self) -> None:
        """No candidate skills -> skills score 0.0."""
        scorer: EmbeddingSkillScorer = _scorer()
        matches, score = scorer.score_skills(["python", "java"], [], [])
        assert score == 0.0
        assert all(m.match_score == 0.0 for m in matches)


# ---------------------------------------------------------------------------
# TestOverallScore
# ---------------------------------------------------------------------------

class TestOverallScore:
    def test_all_required_matched_score_1(self) -> None:
        """All required skills matched -> overall score = 1.0."""
        scorer: EmbeddingSkillScorer = _scorer()
        _, score = scorer.score_skills(["python"], [], ["python"])
        assert score == pytest.approx(1.0)

    def test_preferred_skills_use_lower_weight(self) -> None:
        """Preferred skills carry 0.3 weight; required carry 0.7."""
        scorer: EmbeddingSkillScorer = _scorer()
        # required: python matched (1.0), preferred: java not matched (0.0)
        _, score = scorer.score_skills(["python"], ["java"], ["python"])
        # expected: (0.7 * 1.0 + 0.3 * 0.0) / 1.0 = 0.7
        assert score == pytest.approx(0.7, abs=0.01)

    def test_no_jd_skills_candidate_has_skills_returns_half(self) -> None:
        """No required or preferred JD skills -> score defaults to 0.5."""
        scorer: EmbeddingSkillScorer = _scorer()
        _, score = scorer.score_skills([], [], ["python", "java"])
        assert score == pytest.approx(0.5)

    def test_score_in_unit_range(self) -> None:
        """Overall score is always in [0.0, 1.0]."""
        scorer: EmbeddingSkillScorer = _scorer()
        _, score = scorer.score_skills(["python", "java"], ["excel"], ["pytorch"])
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestFallback
# ---------------------------------------------------------------------------

class TestFallback:
    def test_fallback_exact_string_match(self) -> None:
        """Exact fallback _score_exact_fallback matches 'python' correctly."""
        scorer: EmbeddingSkillScorer = EmbeddingSkillScorer(embedding_model=None)
        matches, score = scorer._score_exact_fallback(["python"], [], ["python"])
        assert matches[0].match_score == 1.0
        assert score == pytest.approx(1.0)

    def test_fallback_returns_zero_for_no_match(self) -> None:
        """Exact fallback returns 0.0 for skills not in candidate list."""
        scorer: EmbeddingSkillScorer = EmbeddingSkillScorer(embedding_model=None)
        matches, score = scorer._score_exact_fallback(["rust"], [], ["python"])
        assert matches[0].match_score == 0.0
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestSkillMatchFields
# ---------------------------------------------------------------------------

class TestSkillMatchFields:
    def test_required_flag_set_correctly(self) -> None:
        """Required skills have required=True; preferred have required=False."""
        scorer: EmbeddingSkillScorer = _scorer()
        matches, _ = scorer.score_skills(["python"], ["java"], ["python", "java"])
        req_match: SkillMatch = next(m for m in matches if m.skill_name == "python")
        pref_match: SkillMatch = next(m for m in matches if m.skill_name == "java")
        assert req_match.required is True
        assert pref_match.required is False

    def test_partial_match_records_closest_candidate_skill(self) -> None:
        """Strong partial match records the matched candidate skill in related_skill."""
        scorer: EmbeddingSkillScorer = _scorer()
        matches, _ = scorer.score_skills(["python"], [], ["pytorch"])
        assert matches[0].related_skill == "pytorch"

    def test_factory_returns_instance(self) -> None:
        """get_embedding_skill_scorer() returns an EmbeddingSkillScorer."""
        scorer = get_embedding_skill_scorer()
        assert isinstance(scorer, EmbeddingSkillScorer)


# ---------------------------------------------------------------------------
# TestThresholdBoundaries
# ---------------------------------------------------------------------------

class TestThresholdBoundaries:
    def test_sim_exactly_at_exact_threshold_is_exact_match(self) -> None:
        """sim == SKILL_EXACT_THRESHOLD (0.92) -> score 1.0 (boundary inclusive)."""
        from src.core.matching.skill_scorer import SKILL_EXACT_THRESHOLD

        class _BoundaryModel:
            def encode(self, texts, normalize=True, **kwargs):
                if isinstance(texts, str):
                    return np.array([1.0, 0.0, 0.0])
                # Return vectors such that dot("jd_skill", "cand") == SKILL_EXACT_THRESHOLD
                results = []
                for t in texts:
                    if t == "candidate_skill":
                        results.append(np.array([SKILL_EXACT_THRESHOLD, (1 - SKILL_EXACT_THRESHOLD**2)**0.5, 0.0]))
                    else:
                        results.append(np.array([1.0, 0.0, 0.0]))
                return np.array(results)

        scorer = EmbeddingSkillScorer(embedding_model=_BoundaryModel())
        matches, _ = scorer.score_skills(["jd_skill"], [], ["candidate_skill"])
        assert matches[0].match_score == pytest.approx(1.0)

    def test_sim_exactly_at_strong_partial_threshold_is_strong_partial(self) -> None:
        """sim == SKILL_STRONG_PARTIAL_THRESHOLD (0.75) -> score 0.8 (boundary inclusive)."""
        from src.core.matching.skill_scorer import SKILL_STRONG_PARTIAL_THRESHOLD

        class _BoundaryModel:
            def encode(self, texts, normalize=True, **kwargs):
                if isinstance(texts, str):
                    return np.array([1.0, 0.0, 0.0])
                results = []
                for t in texts:
                    if t == "candidate_skill":
                        results.append(np.array([SKILL_STRONG_PARTIAL_THRESHOLD, (1 - SKILL_STRONG_PARTIAL_THRESHOLD**2)**0.5, 0.0]))
                    else:
                        results.append(np.array([1.0, 0.0, 0.0]))
                return np.array(results)

        scorer = EmbeddingSkillScorer(embedding_model=_BoundaryModel())
        matches, _ = scorer.score_skills(["jd_skill"], [], ["candidate_skill"])
        assert matches[0].match_score == pytest.approx(0.8)

    def test_sim_just_below_weak_threshold_is_no_match(self) -> None:
        """sim just below SKILL_WEAK_PARTIAL_THRESHOLD (0.60) -> score 0.0."""
        from src.core.matching.skill_scorer import SKILL_WEAK_PARTIAL_THRESHOLD

        just_below: float = SKILL_WEAK_PARTIAL_THRESHOLD - 0.01

        class _BoundaryModel:
            def encode(self, texts, normalize=True, **kwargs):
                if isinstance(texts, str):
                    return np.array([1.0, 0.0, 0.0])
                results = []
                for t in texts:
                    if t == "candidate_skill":
                        results.append(np.array([just_below, (1 - just_below**2)**0.5, 0.0]))
                    else:
                        results.append(np.array([1.0, 0.0, 0.0]))
                return np.array(results)

        scorer = EmbeddingSkillScorer(embedding_model=_BoundaryModel())
        matches, _ = scorer.score_skills(["jd_skill"], [], ["candidate_skill"])
        assert matches[0].match_score == 0.0
