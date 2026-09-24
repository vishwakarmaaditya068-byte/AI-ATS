import os
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

import math

import pytest

from src.ml.ethics.fairness_metrics import FairnessCalculator, FairnessMetrics


# ── Helpers ────────────────────────────────────────────────────────────────────

def _calc(min_group_size: int = 2) -> FairnessCalculator:
    return FairnessCalculator(min_group_size=min_group_size)


# ── Equalized odds: threshold propagation ──────────────────────────────────────

def test_equalized_odds_uses_selection_threshold_not_hardcoded() -> None:
    """TPR diff must be large when threshold splits the two groups."""
    # Group A: scores 0.85 (above 0.7 threshold), all actually hired
    # Group B: scores 0.65 (below 0.7 threshold), all actually hired
    # → TPR_A = 1.0 (predicted positive AND hired), TPR_B = 0.0 (predicted negative, still hired)
    scores: list[float] = [0.85, 0.85, 0.65, 0.65]
    labels: list[str] = ["A", "A", "B", "B"]
    real_outcomes: list[bool] = [True, True, True, True]

    result = _calc().calculate(scores, labels, outcomes=real_outcomes, selection_threshold=0.7)

    assert result.true_positive_rate_difference > 0.5, (
        "TPR difference should be large when threshold separates the two groups"
    )


def test_equalized_odds_lower_threshold_gives_zero_tpr_diff() -> None:
    """With threshold=0.5 all candidates are predicted positive → TPR diff = 0."""
    scores: list[float] = [0.85, 0.85, 0.65, 0.65]
    labels: list[str] = ["A", "A", "B", "B"]
    real_outcomes: list[bool] = [True, True, True, True]

    result = _calc().calculate(scores, labels, outcomes=real_outcomes, selection_threshold=0.5)

    assert result.true_positive_rate_difference == 0.0, (
        "When all candidates exceed threshold, TPR difference should be 0"
    )


def test_equalized_odds_proper_tpr_formula_verified() -> None:
    """Verify TPR = P(predicted=1 | outcome=1) computed correctly per group."""
    # Group A (3 candidates): scores [0.8, 0.8, 0.3], outcomes [True, True, False]
    #   threshold=0.7 → predictions=[1,1,0]
    #   TPR_A = count(pred=1 & outcome=True) / count(outcome=True) = 2/2 = 1.0
    # Group B (3 candidates): scores [0.8, 0.4, 0.3], outcomes [True, True, False]
    #   threshold=0.7 → predictions=[1,0,0]
    #   TPR_B = count(pred=1 & outcome=True) / count(outcome=True) = 1/2 = 0.5
    # Expected TPR diff = 0.5
    scores: list[float] = [0.8, 0.8, 0.3,   0.8, 0.4, 0.3]
    labels: list[str]  = ["A", "A", "A",    "B", "B", "B"]
    real_outcomes: list[bool] = [True, True, False,  True, True, False]

    result = _calc().calculate(scores, labels, outcomes=real_outcomes, selection_threshold=0.7)

    assert abs(result.true_positive_rate_difference - 0.5) < 0.01, (
        f"Expected TPR diff ≈ 0.5, got {result.true_positive_rate_difference}"
    )


def test_equalized_odds_warns_when_outcomes_derived_from_scores() -> None:
    """When no real outcomes are provided, a warning about equalized odds must appear."""
    scores: list[float] = [0.8, 0.9, 0.6, 0.55]
    labels: list[str] = ["A", "A", "B", "B"]

    result = _calc().calculate(scores, labels)  # no real outcomes

    assert any("equalized odds" in w.lower() for w in result.warnings), (
        "Should warn that equalized odds is not meaningful without real hiring outcomes"
    )


def test_equalized_odds_zero_when_derived_from_scores() -> None:
    """Without real outcomes, all equalized odds values must be 0 (not fake values)."""
    scores: list[float] = [0.8, 0.9, 0.3, 0.2]
    labels: list[str] = ["A", "A", "B", "B"]

    result = _calc().calculate(scores, labels)

    assert result.equalized_odds_difference == 0.0
    assert result.true_positive_rate_difference == 0.0
    assert result.false_positive_rate_difference == 0.0


# ── Small groups ───────────────────────────────────────────────────────────────

def test_small_group_excluded_with_name_in_warning() -> None:
    """Groups below min_group_size must be named in the warnings list."""
    # 4 in A, 1 in B — B is below min_group_size=3
    scores: list[float] = [0.8, 0.9, 0.7, 0.85, 0.6]
    labels: list[str] = ["A", "A", "A", "A", "B"]

    result = FairnessCalculator(min_group_size=3).calculate(scores, labels)

    assert any("B" in w for w in result.warnings), (
        "Excluded group 'B' should be named in the warnings field, not silently dropped"
    )


# ── Score variance ratio ───────────────────────────────────────────────────────

def test_score_variance_ratio_normal_case() -> None:
    """When both groups have different non-zero std, ratio is max/min variance."""
    # Group A: tight scores (low std), Group B: spread scores (high std)
    scores: list[float] = [0.8, 0.82, 0.81,   0.2, 0.9, 0.5]
    labels: list[str]   = ["A", "A",  "A",    "B", "B", "B"]

    result = _calc().calculate(scores, labels)

    assert result.score_variance_ratio > 1.0, (
        "Group B has higher score spread than A — variance ratio should exceed 1"
    )


def test_score_variance_ratio_is_inf_when_one_group_has_zero_std() -> None:
    """When one group has identical scores (std=0), ratio must be inf, not 1.0."""
    # Group A: all same score → std=0; Group B: different scores → std>0
    scores: list[float] = [0.8, 0.8, 0.8,   0.3, 0.7, 0.9]
    labels: list[str]   = ["A", "A", "A",   "B", "B", "B"]

    result = _calc().calculate(scores, labels)

    assert math.isinf(result.score_variance_ratio), (
        "When a group has zero std, score_variance_ratio must be inf, not 1.0"
    )
    assert any("variance ratio" in w.lower() for w in result.warnings), (
        "A warning must be emitted when score variance ratio is undefined"
    )


def test_score_variance_ratio_one_when_all_same_std() -> None:
    """When all groups have the same std, ratio should be 1.0."""
    # Both groups have symmetrically spread scores
    scores: list[float] = [0.3, 0.7, 0.5,   0.3, 0.7, 0.5]
    labels: list[str]   = ["A", "A", "A",   "B", "B", "B"]

    result = _calc().calculate(scores, labels)

    assert result.score_variance_ratio == pytest.approx(1.0, abs=1e-3)


def test_small_groups_result_is_fair_when_no_eligible_groups() -> None:
    """When fewer than 2 groups meet min_group_size, is_fair=True with non-empty warnings."""
    scores: list[float] = [0.8, 0.9, 0.6]
    labels: list[str] = ["A", "A", "B"]  # both groups < min_group_size=3

    result = FairnessCalculator(min_group_size=3).calculate(scores, labels)

    assert result.is_fair is True
    assert len(result.warnings) > 0, "Warnings must be populated when groups are skipped"
