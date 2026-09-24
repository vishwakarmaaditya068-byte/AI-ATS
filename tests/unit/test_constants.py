"""
Tests for src.utils.constants — enums, scoring weights, education levels.
"""

import pytest

from src.utils.constants import (
    EDUCATION_LEVELS,
    DEFAULT_SCORING_WEIGHTS,
    SCORE_THRESHOLDS,
    SKILL_CATEGORIES,
    AuditAction,
    CandidateStatus,
    JobStatus,
    MatchScoreLevel,
)


# ── MatchScoreLevel.from_score() ────────────────────────────────────────────


class TestMatchScoreLevelFromScore:
    def test_excellent_at_threshold(self):
        assert MatchScoreLevel.from_score(0.85) == MatchScoreLevel.EXCELLENT

    def test_excellent_above_threshold(self):
        assert MatchScoreLevel.from_score(0.99) == MatchScoreLevel.EXCELLENT

    def test_excellent_at_max(self):
        assert MatchScoreLevel.from_score(1.0) == MatchScoreLevel.EXCELLENT

    def test_good_at_threshold(self):
        assert MatchScoreLevel.from_score(0.70) == MatchScoreLevel.GOOD

    def test_good_just_below_excellent(self):
        assert MatchScoreLevel.from_score(0.849) == MatchScoreLevel.GOOD

    def test_fair_at_threshold(self):
        assert MatchScoreLevel.from_score(0.50) == MatchScoreLevel.FAIR

    def test_fair_just_below_good(self):
        assert MatchScoreLevel.from_score(0.699) == MatchScoreLevel.FAIR

    def test_poor_below_fair(self):
        assert MatchScoreLevel.from_score(0.499) == MatchScoreLevel.POOR

    def test_poor_at_zero(self):
        assert MatchScoreLevel.from_score(0.0) == MatchScoreLevel.POOR

    def test_poor_low_value(self):
        assert MatchScoreLevel.from_score(0.29) == MatchScoreLevel.POOR


# ── Enum value correctness ──────────────────────────────────────────────────


class TestCandidateStatus:
    def test_all_values_present(self):
        expected = {"new", "screening", "shortlisted", "interview", "offer", "hired", "rejected", "withdrawn"}
        actual = {s.value for s in CandidateStatus}
        assert actual == expected

    def test_new_value(self):
        assert CandidateStatus.NEW.value == "new"

    def test_hired_value(self):
        assert CandidateStatus.HIRED.value == "hired"


class TestJobStatus:
    def test_all_values_present(self):
        expected = {"draft", "open", "paused", "closed", "filled"}
        actual = {s.value for s in JobStatus}
        assert actual == expected

    def test_open_value(self):
        assert JobStatus.OPEN.value == "open"


class TestAuditAction:
    def test_all_values_present(self):
        expected = {
            "candidate_added", "candidate_scored", "candidate_ranked",
            "candidate_status_changed", "job_created", "job_closed",
            "bias_detected", "manual_override", "report_generated",
        }
        actual = {a.value for a in AuditAction}
        assert actual == expected

    def test_manual_override_value(self):
        assert AuditAction.MANUAL_OVERRIDE.value == "manual_override"

    def test_bias_detected_value(self):
        assert AuditAction.BIAS_DETECTED.value == "bias_detected"


# ── DEFAULT_SCORING_WEIGHTS ─────────────────────────────────────────────────


class TestDefaultScoringWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(DEFAULT_SCORING_WEIGHTS.values()) - 1.0) < 1e-9

    def test_all_positive(self):
        for key, value in DEFAULT_SCORING_WEIGHTS.items():
            assert value > 0, f"Weight '{key}' should be positive"

    def test_expected_keys(self):
        expected_keys = {"skills_match", "experience_match", "education_match", "semantic_similarity", "keyword_match"}
        assert set(DEFAULT_SCORING_WEIGHTS.keys()) == expected_keys

    def test_skills_match_is_largest(self):
        assert DEFAULT_SCORING_WEIGHTS["skills_match"] == max(DEFAULT_SCORING_WEIGHTS.values())


# ── EDUCATION_LEVELS ────────────────────────────────────────────────────────


class TestEducationLevels:
    def test_bachelor_less_than_master(self):
        assert EDUCATION_LEVELS["bachelor"] < EDUCATION_LEVELS["master"]

    def test_master_less_than_phd(self):
        assert EDUCATION_LEVELS["master"] < EDUCATION_LEVELS["phd"]

    def test_mba_equals_master(self):
        assert EDUCATION_LEVELS["mba"] == EDUCATION_LEVELS["master"]

    def test_doctorate_equals_phd(self):
        assert EDUCATION_LEVELS["doctorate"] == EDUCATION_LEVELS["phd"]

    def test_high_school_is_lowest(self):
        assert EDUCATION_LEVELS["high school"] == min(EDUCATION_LEVELS.values())

    def test_phd_is_highest(self):
        assert EDUCATION_LEVELS["phd"] == max(EDUCATION_LEVELS.values())


# ── SCORE_THRESHOLDS ────────────────────────────────────────────────────────


class TestScoreThresholds:
    def test_excellent_matches_from_score(self):
        assert SCORE_THRESHOLDS["excellent"] == 0.85

    def test_thresholds_are_descending(self):
        assert SCORE_THRESHOLDS["excellent"] > SCORE_THRESHOLDS["good"] > SCORE_THRESHOLDS["fair"] > SCORE_THRESHOLDS["poor"]


# ── SKILL_CATEGORIES ────────────────────────────────────────────────────────


class TestSkillCategories:
    def test_has_expected_categories(self):
        expected = {"programming_languages", "frameworks", "databases", "cloud_platforms", "soft_skills"}
        assert expected.issubset(set(SKILL_CATEGORIES.keys()))

    def test_python_in_programming_languages(self):
        assert "python" in SKILL_CATEGORIES["programming_languages"]

    def test_categories_are_nonempty(self):
        for category, skills in SKILL_CATEGORIES.items():
            assert len(skills) > 0, f"Category '{category}' should not be empty"
