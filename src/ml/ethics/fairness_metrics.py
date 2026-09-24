"""
Fairness metrics calculation for bias detection.

Implements various fairness metrics to measure and monitor
bias in the candidate matching and ranking process.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.constants import FAIRNESS_MIN_GROUP_SIZE, FAIRNESS_THRESHOLDS
from src.utils.logger import get_logger

logger = get_logger(__name__)

# np.std() on a constant array returns machine-epsilon (~1e-16) rather than
# exactly 0.0.  Variances below this threshold are treated as zero when
# computing score_variance_ratio so that all-same-score groups are correctly
# detected as having zero spread.
_VAR_EPS: float = 1e-10


@dataclass
class GroupMetrics:
    """Metrics for a specific demographic group."""

    group_name: str
    group_size: int
    positive_count: int  # Number selected/shortlisted
    positive_rate: float  # Selection rate
    average_score: float
    score_std: float


@dataclass
class FairnessMetrics:
    """Complete fairness metrics result."""

    # Demographic Parity
    demographic_parity_difference: float = 0.0
    demographic_parity_ratio: float = 1.0

    # Equalized Odds
    equalized_odds_difference: float = 0.0
    true_positive_rate_difference: float = 0.0
    false_positive_rate_difference: float = 0.0

    # Disparate Impact
    disparate_impact_ratio: float = 1.0

    # Score Distribution
    score_gap: float = 0.0  # Gap between highest and lowest group avg scores
    score_variance_ratio: float = 1.0

    # Group-level metrics
    group_metrics: list[GroupMetrics] = field(default_factory=list)

    # Fairness assessment
    is_fair: bool = True
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # Non-fatal issues (e.g. skipped small groups)
    thresholds_used: dict[str, float] = field(default_factory=dict)


class FairnessCalculator:
    """
    Calculates fairness metrics for candidate evaluation.

    Implements standard fairness metrics including:
    - Demographic Parity
    - Equalized Odds
    - Disparate Impact
    """

    def __init__(
        self,
        thresholds: Optional[dict[str, float]] = None,
        min_group_size: int = FAIRNESS_MIN_GROUP_SIZE,
    ):
        """
        Initialize the fairness calculator.

        Args:
            thresholds: Custom fairness thresholds. Defaults to config values.
            min_group_size: Minimum number of candidates a demographic group
                must contain to be included in metric calculations.  Groups
                smaller than this are excluded because their selection rates
                are too noisy to be statistically meaningful (one additional
                selection/rejection can swing the rate by 20–50 %).
                Defaults to FAIRNESS_MIN_GROUP_SIZE (5).
        """
        self.thresholds = thresholds or FAIRNESS_THRESHOLDS
        self.min_group_size = min_group_size

    def calculate(
        self,
        scores: list[float],
        group_labels: list[str],
        outcomes: Optional[list[bool]] = None,
        selection_threshold: float = 0.7,
    ) -> FairnessMetrics:
        """
        Calculate fairness metrics for a set of candidates.

        Args:
            scores: List of match scores for each candidate.
            group_labels: List of group labels (e.g., demographic group).
            outcomes: Optional list of actual outcomes (selected/not selected).
                     If not provided, uses selection_threshold on scores.
            selection_threshold: Score threshold for considering a candidate
                                as "selected" (if outcomes not provided).

        Returns:
            FairnessMetrics with all calculated metrics.
        """
        if len(scores) != len(group_labels):
            raise ValueError("Scores and group labels must have same length")

        scores = np.array(scores)
        group_labels = np.array(group_labels)

        # Track whether outcomes are real labels or score-derived BEFORE conversion
        _real_outcomes: bool = outcomes is not None

        # Determine outcomes
        if outcomes is None:
            outcomes = scores >= selection_threshold
        else:
            outcomes = np.array(outcomes)

        # Get unique groups
        unique_groups = np.unique(group_labels)

        if len(unique_groups) < 2:
            logger.warning("Fewer than 2 groups found, cannot calculate fairness metrics")
            return FairnessMetrics(
                is_fair=True,
                warnings=["Fewer than 2 demographic groups found; fairness metrics cannot be calculated."],
            )

        # Filter out groups that are too small to yield reliable statistics.
        # A single extra selection/rejection in a group of 3 moves its rate by
        # ~33 %, making any derived fairness metric meaningless.
        calc_warnings: list[str] = []
        eligible_groups = []
        for g in unique_groups:
            size = int(np.sum(group_labels == g))
            if size < self.min_group_size:
                msg = (
                    f"Group '{g}' has only {size} sample(s) "
                    f"(minimum required: {self.min_group_size}) — excluded from "
                    "fairness calculations."
                )
                calc_warnings.append(msg)
                logger.warning(msg)
            else:
                eligible_groups.append(g)

        if len(eligible_groups) < 2:
            logger.warning(
                "Fewer than 2 groups meet the minimum size requirement "
                f"({self.min_group_size}); cannot calculate fairness metrics."
            )
            return FairnessMetrics(is_fair=True, warnings=calc_warnings)

        unique_groups = np.array(eligible_groups)

        # Calculate group-level metrics
        group_metrics = []
        group_positive_rates = {}
        group_avg_scores = {}

        for group in unique_groups:
            mask = group_labels == group
            group_scores = scores[mask]
            group_outcomes = outcomes[mask]

            positive_count = int(np.sum(group_outcomes))
            positive_rate = positive_count / len(group_outcomes) if len(group_outcomes) > 0 else 0

            metrics = GroupMetrics(
                group_name=str(group),
                group_size=int(np.sum(mask)),
                positive_count=positive_count,
                positive_rate=positive_rate,
                average_score=float(np.mean(group_scores)),
                score_std=float(np.std(group_scores)) if len(group_scores) > 1 else 0,
            )
            group_metrics.append(metrics)
            group_positive_rates[group] = positive_rate
            group_avg_scores[group] = metrics.average_score

        # Calculate Demographic Parity
        dp_diff, dp_ratio = self._calculate_demographic_parity(group_positive_rates)

        # Calculate Disparate Impact
        di_ratio = self._calculate_disparate_impact(group_positive_rates)

        eo_diff, tpr_diff, fpr_diff = self._calculate_equalized_odds(
            scores, group_labels, outcomes, unique_groups, selection_threshold,
            _real_outcomes, calc_warnings,
        )

        # Calculate score distribution metrics.
        # _VAR_EPS (module-level) guards against np.std machine-epsilon noise.
        score_gap = max(group_avg_scores.values()) - min(group_avg_scores.values())
        variances: list[float] = [m.score_std ** 2 for m in group_metrics]
        if len(variances) >= 2 and min(variances) > _VAR_EPS:
            score_var_ratio: float = max(variances) / min(variances)
        elif len(variances) >= 2 and max(variances) > _VAR_EPS:
            # One or more groups have effectively zero score variance (all candidates
            # in that group received the same score).  The ratio is mathematically ∞ —
            # using 1.0 would falsely signal "perfectly equal spread", masking the disparity.
            score_var_ratio = float("inf")
            calc_warnings.append(
                "Score variance ratio is undefined: at least one group has zero score variance."
            )
        else:
            score_var_ratio = 1.0

        # Check for violations
        violations = []
        if abs(dp_diff) > self.thresholds["demographic_parity_difference"]:
            violations.append(
                f"Demographic parity difference ({dp_diff:.3f}) exceeds threshold "
                f"({self.thresholds['demographic_parity_difference']})"
            )

        if abs(eo_diff) > self.thresholds["equalized_odds_difference"]:
            violations.append(
                f"Equalized odds difference ({eo_diff:.3f}) exceeds threshold "
                f"({self.thresholds['equalized_odds_difference']})"
            )

        if di_ratio < self.thresholds["disparate_impact_ratio"]:
            violations.append(
                f"Disparate impact ratio ({di_ratio:.3f}) below threshold "
                f"({self.thresholds['disparate_impact_ratio']})"
            )

        return FairnessMetrics(
            demographic_parity_difference=dp_diff,
            demographic_parity_ratio=dp_ratio,
            equalized_odds_difference=eo_diff,
            true_positive_rate_difference=tpr_diff,
            false_positive_rate_difference=fpr_diff,
            disparate_impact_ratio=di_ratio,
            score_gap=score_gap,
            score_variance_ratio=score_var_ratio,
            group_metrics=group_metrics,
            is_fair=len(violations) == 0,
            violations=violations,
            warnings=calc_warnings,
            thresholds_used=self.thresholds.copy(),
        )

    def _calculate_demographic_parity(
        self,
        group_rates: dict[str, float],
    ) -> tuple[float, float]:
        """
        Calculate demographic parity metrics.

        Demographic parity requires that the selection rate is equal
        across all groups.

        Returns:
            (difference, ratio) where:
            - difference: max_rate - min_rate
            - ratio: min_rate / max_rate
        """
        rates = list(group_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)

        difference = max_rate - min_rate
        ratio = min_rate / max_rate if max_rate > 0 else 1.0

        return round(difference, 4), round(ratio, 4)

    def _calculate_disparate_impact(
        self,
        group_rates: dict[str, float],
    ) -> float:
        """
        Calculate disparate impact ratio (Four-Fifths Rule).

        The four-fifths rule states that selection rate for any group
        should be at least 80% of the highest group's rate.

        Returns:
            Ratio of lowest to highest selection rate.
        """
        rates = list(group_rates.values())
        max_rate = max(rates)
        min_rate = min(rates)

        if max_rate == 0:
            return 1.0

        return round(min_rate / max_rate, 4)

    def _calculate_equalized_odds(
        self,
        scores: np.ndarray,
        group_labels: np.ndarray,
        outcomes: np.ndarray,
        groups: np.ndarray,
        selection_threshold: float,
        real_outcomes: bool,
        warnings_out: list[str] | None = None,
    ) -> tuple[float, float, float]:
        """
        Calculate equalized odds metrics.

        Equalized odds requires that both TPR and FPR are equal across groups.
        Uses ``scores >= selection_threshold`` as the model's predictions and
        ``outcomes`` as the true labels (actual hiring decisions).

        Only meaningful when *real* outcomes are provided.  When outcomes are
        score-derived the caller must pass ``real_outcomes=False`` and this
        method returns (0.0, 0.0, 0.0) to avoid reporting noise as signal, and
        appends an explanatory message to ``warnings_out`` if provided.

        Args:
            scores: Match scores for all candidates.
            group_labels: Demographic group label per candidate.
            outcomes: True labels (actual hired/not-hired decisions).
            groups: Unique group values to iterate over.
            selection_threshold: Score threshold used to derive predictions.
            real_outcomes: False when outcomes were derived from scores rather
                than actual hiring decisions.
            warnings_out: If provided, an explanatory warning is appended when
                ``real_outcomes`` is False.

        Returns:
            (equalized_odds_diff, tpr_diff, fpr_diff)
        """
        if not real_outcomes:
            if warnings_out is not None:
                warnings_out.append(
                    "Equalized odds requires real hiring outcomes (the `outcomes` argument) "
                    "to be meaningful. When outcomes are derived from scores, TPR/FPR values "
                    "are trivially determined by the threshold and do not measure bias. "
                    "Equalized odds values are set to 0 for this run."
                )
            return 0.0, 0.0, 0.0

        predicted: np.ndarray = scores >= selection_threshold
        group_tprs: dict[str, float] = {}
        group_fprs: dict[str, float] = {}

        for group in groups:
            mask: np.ndarray = group_labels == group
            group_predicted: np.ndarray = predicted[mask]
            group_outcomes: np.ndarray = outcomes[mask]

            actual_positives: int = int(np.sum(group_outcomes))
            actual_negatives: int = int(np.sum(~group_outcomes))

            # TPR = P(predicted=1 | outcome=1)
            tpr: float = (
                float(np.sum(group_predicted & group_outcomes)) / actual_positives
                if actual_positives > 0 else 0.0
            )
            # FPR = P(predicted=1 | outcome=0)
            fpr: float = (
                float(np.sum(group_predicted & ~group_outcomes)) / actual_negatives
                if actual_negatives > 0 else 0.0
            )

            group_tprs[group] = tpr
            group_fprs[group] = fpr

        tpr_values: list[float] = list(group_tprs.values())
        fpr_values: list[float] = list(group_fprs.values())

        tpr_diff: float = max(tpr_values) - min(tpr_values) if tpr_values else 0.0
        fpr_diff: float = max(fpr_values) - min(fpr_values) if fpr_values else 0.0
        eo_diff: float = max(tpr_diff, fpr_diff)

        return round(eo_diff, 4), round(tpr_diff, 4), round(fpr_diff, 4)

    def calculate_individual_fairness(
        self,
        candidate_scores: list[tuple[str, float, dict]],
        similarity_threshold: float = 0.1,
    ) -> dict[str, float | int]:
        """
        Calculate individual fairness metrics.

        Individual fairness requires that similar individuals
        receive similar scores.

        TODO: This method is not yet wired into the ranking pipeline.
              It should be integrated with FairnessReranker.apply() and
              covered by tests before being relied upon in production.

        Args:
            candidate_scores: List of (candidate_id, score, features) tuples
            similarity_threshold: Max allowed score difference for similar candidates

        Returns:
            Dictionary with individual fairness metrics.
        """
        if len(candidate_scores) < 2:
            return {"individual_fairness_score": 1.0, "violations": 0}

        violations = 0
        total_pairs = 0

        for i in range(len(candidate_scores)):
            for j in range(i + 1, len(candidate_scores)):
                _, score_i, features_i = candidate_scores[i]
                _, score_j, features_j = candidate_scores[j]

                # Calculate feature similarity (simplified)
                feature_sim = self._calculate_feature_similarity(features_i, features_j)

                if feature_sim > 0.8:  # Similar candidates
                    total_pairs += 1
                    score_diff = abs(score_i - score_j)
                    if score_diff > similarity_threshold:
                        violations += 1

        if total_pairs == 0:
            return {"individual_fairness_score": 1.0, "violations": 0}

        fairness_score = 1.0 - (violations / total_pairs)

        return {
            "individual_fairness_score": round(fairness_score, 4),
            "violations": violations,
            "total_similar_pairs": total_pairs,
        }

    def _calculate_feature_similarity(
        self,
        features1: dict,
        features2: dict,
    ) -> float:
        """Calculate similarity between two feature dictionaries."""
        if not features1 or not features2:
            return 0.0

        common_keys = set(features1.keys()) & set(features2.keys())
        if not common_keys:
            return 0.0

        matches = 0
        for key in common_keys:
            if features1[key] == features2[key]:
                matches += 1
            elif isinstance(features1[key], (int, float)) and isinstance(features2[key], (int, float)):
                # For numeric values, check if within 10%
                max_val = max(abs(features1[key]), abs(features2[key]))
                if max_val > 0 and abs(features1[key] - features2[key]) / max_val < 0.1:
                    matches += 1

        return matches / len(common_keys)


# Singleton instance — guarded by a lock to prevent duplicate construction
# when multiple threads access the calculator concurrently.
_calculator: Optional[FairnessCalculator] = None
_calculator_lock: threading.Lock = threading.Lock()


def get_fairness_calculator() -> FairnessCalculator:
    """Get the fairness calculator singleton instance (thread-safe)."""
    global _calculator
    if _calculator is None:
        with _calculator_lock:
            if _calculator is None:
                _calculator = FairnessCalculator()
    return _calculator
