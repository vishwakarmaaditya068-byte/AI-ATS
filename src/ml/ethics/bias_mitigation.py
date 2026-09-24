"""
Bias mitigation strategies for fair candidate evaluation.

Implements various techniques to reduce bias in the matching
and ranking process while maintaining prediction quality.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MitigationStrategy(str, Enum):
    """Available bias mitigation strategies."""

    # Pre-processing strategies
    REDACTION = "redaction"  # Remove protected attributes
    REWEIGHTING = "reweighting"  # Adjust sample weights

    # In-processing strategies
    SCORE_CALIBRATION = "score_calibration"  # Calibrate scores per group
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"  # Adjust selection thresholds

    # Post-processing strategies
    REJECT_OPTION = "reject_option"  # Flip predictions near threshold
    EQUALIZED_ODDS = "equalized_odds"  # Adjust to equalize odds


@dataclass
class MitigationResult:
    """Result of applying a bias mitigation strategy."""

    strategy_used: MitigationStrategy
    original_scores: list[float]
    mitigated_scores: list[float]
    changes_made: int
    fairness_improvement: Optional[dict[str, float]] = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RedactionResult:
    """Result of text redaction for protected attributes."""

    original_text: str
    redacted_text: str
    redactions_made: int
    redacted_terms: list[str] = field(default_factory=list)


class BiasMitigator:
    """
    Implements bias mitigation strategies.

    Provides both pre-processing (data-level) and post-processing
    (prediction-level) mitigation techniques.
    """

    def __init__(self):
        """Initialize the bias mitigator."""
        self._setup_redaction_rules()

    def _setup_redaction_rules(self):
        """Set up redaction rules for text anonymization."""
        import re

        # Patterns to redact (from protected attribute detector)
        self.redaction_patterns = {
            # Names (common patterns)
            "names": re.compile(
                r"\b(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
            ),

            # Gender pronouns
            "pronouns": re.compile(
                r"\b(he|she|him|her|his|hers|himself|herself)\b",
                re.IGNORECASE
            ),

            # Age indicators
            "age": re.compile(
                r"\b\d{2}\s*(?:years?\s+old|y\.?o\.?)\b|\b(?:born\s+(?:in\s+)?(?:19|20)\d{2})\b",
                re.IGNORECASE
            ),

            # Marital status
            "marital": re.compile(
                r"\b(married|single|divorced|widowed|spouse|wife|husband)\b",
                re.IGNORECASE
            ),

            # Nationality explicit mentions
            "nationality": re.compile(
                r"\b(citizen(?:ship)?|visa\s+(?:status|holder)|h-?1b|green\s+card)\b",
                re.IGNORECASE
            ),
        }

    def redact_protected_attributes(
        self,
        text: str,
        categories: Optional[list[str]] = None,
    ) -> RedactionResult:
        """
        Redact protected attributes from text.

        Args:
            text: The text to redact.
            categories: Optional list of categories to redact.
                       If None, redacts all categories.

        Returns:
            RedactionResult with redacted text and details.
        """
        if not text:
            return RedactionResult(
                original_text="",
                redacted_text="",
                redactions_made=0,
            )

        redacted_text = text
        redacted_terms = []
        patterns_to_use = self.redaction_patterns

        if categories:
            patterns_to_use = {
                k: v for k, v in self.redaction_patterns.items()
                if k in categories
            }

        for category, pattern in patterns_to_use.items():
            matches = list(pattern.finditer(redacted_text))
            for match in reversed(matches):  # Reverse to maintain positions
                term = match.group()
                redacted_terms.append(term)
                redacted_text = (
                    redacted_text[:match.start()] +
                    f"[REDACTED_{category.upper()}]" +
                    redacted_text[match.end():]
                )

        return RedactionResult(
            original_text=text,
            redacted_text=redacted_text,
            redactions_made=len(redacted_terms),
            redacted_terms=redacted_terms,
        )

    def calibrate_scores(
        self,
        scores: list[float],
        group_labels: list[str],
        target_mean: Optional[float] = None,
    ) -> MitigationResult:
        """
        Calibrate scores to have similar distributions across groups.

        This adjusts scores so that each group has similar mean scores,
        while preserving relative rankings within groups.

        Args:
            scores: Original match scores.
            group_labels: Group label for each score.
            target_mean: Target mean score. If None, uses overall mean.

        Returns:
            MitigationResult with calibrated scores.
        """
        scores = np.array(scores)
        group_labels = np.array(group_labels)

        if target_mean is None:
            target_mean = np.mean(scores)

        calibrated_scores = scores.copy()
        unique_groups = np.unique(group_labels)
        changes = 0
        group_adjustments = {}

        for group in unique_groups:
            mask = group_labels == group
            group_scores = scores[mask]
            group_mean = np.mean(group_scores)

            # Calculate adjustment factor
            adjustment = target_mean - group_mean
            group_adjustments[str(group)] = adjustment

            # Apply adjustment
            adjusted = group_scores + adjustment
            # Clip to valid range
            adjusted = np.clip(adjusted, 0, 1)

            # Count changes
            changes += np.sum(np.abs(adjusted - group_scores) > 0.01)

            calibrated_scores[mask] = adjusted

        return MitigationResult(
            strategy_used=MitigationStrategy.SCORE_CALIBRATION,
            original_scores=scores.tolist(),
            mitigated_scores=calibrated_scores.tolist(),
            changes_made=int(changes),
            details={
                "target_mean": target_mean,
                "group_adjustments": group_adjustments,
            },
        )

    def adjust_thresholds(
        self,
        scores: list[float],
        group_labels: list[str],
        base_threshold: float = 0.7,
        target_rate: Optional[float] = None,
    ) -> MitigationResult:
        """
        Adjust selection thresholds per group to achieve equal selection rates.

        Args:
            scores: Match scores.
            group_labels: Group label for each score.
            base_threshold: Base selection threshold.
            target_rate: Target selection rate. If None, uses overall rate.

        Returns:
            MitigationResult with adjusted outcomes (as 0/1 scores).
        """
        scores = np.array(scores)
        group_labels = np.array(group_labels)

        # Original selections
        original_selected = (scores >= base_threshold).astype(float)

        if target_rate is None:
            target_rate = np.mean(original_selected)

        unique_groups = np.unique(group_labels)
        mitigated_scores = scores.copy()
        group_thresholds = {}
        changes = 0

        for group in unique_groups:
            mask = group_labels == group
            group_scores = np.sort(scores[mask])[::-1]  # Descending

            # Find threshold that gives target rate
            n_to_select = int(np.ceil(target_rate * len(group_scores)))
            if n_to_select > 0 and n_to_select <= len(group_scores):
                group_threshold = group_scores[n_to_select - 1]
            else:
                group_threshold = base_threshold

            group_thresholds[str(group)] = float(group_threshold)

            # Apply group-specific threshold (encode in score as above/below)
            # This creates a normalized score based on group threshold
            group_mask = group_labels == group
            original_group = mitigated_scores[group_mask]
            adjusted_group = np.where(
                scores[group_mask] >= group_threshold,
                np.maximum(original_group, base_threshold),  # Ensure above base
                np.minimum(original_group, base_threshold - 0.01),  # Ensure below
            )
            changes += np.sum(np.abs(adjusted_group - original_group) > 0.01)
            mitigated_scores[group_mask] = adjusted_group

        return MitigationResult(
            strategy_used=MitigationStrategy.THRESHOLD_ADJUSTMENT,
            original_scores=scores.tolist(),
            mitigated_scores=mitigated_scores.tolist(),
            changes_made=int(changes),
            details={
                "base_threshold": base_threshold,
                "target_rate": target_rate,
                "group_thresholds": group_thresholds,
            },
        )

    def apply_reject_option(
        self,
        scores: list[float],
        group_labels: list[str],
        threshold: float = 0.7,
        margin: float = 0.1,
        favor_unprivileged: bool = True,
    ) -> MitigationResult:
        """
        Apply reject option classification for bias mitigation.

        For candidates near the decision boundary, flip predictions
        to favor the unprivileged group.

        Args:
            scores: Match scores.
            group_labels: Group label for each score.
            threshold: Selection threshold.
            margin: Margin around threshold to consider for flipping.
            favor_unprivileged: Whether to favor unprivileged groups.

        Returns:
            MitigationResult with adjusted scores.
        """
        scores = np.array(scores)
        group_labels = np.array(group_labels)

        # Identify privileged group (highest selection rate)
        unique_groups = np.unique(group_labels)
        group_rates = {}
        for group in unique_groups:
            mask = group_labels == group
            group_rates[group] = np.mean(scores[mask] >= threshold)

        if favor_unprivileged:
            privileged_group = max(group_rates, key=group_rates.get)
            unprivileged_groups = [g for g in unique_groups if g != privileged_group]
        else:
            unprivileged_groups = []

        mitigated_scores = scores.copy()
        changes = 0

        # Find candidates in the critical region
        critical_region = (scores >= threshold - margin) & (scores <= threshold + margin)

        for i, (score, group, in_critical) in enumerate(zip(scores, group_labels, critical_region)):
            if not in_critical:
                continue

            if favor_unprivileged:
                if group in unprivileged_groups and score < threshold:
                    # Boost unprivileged group members just below threshold
                    mitigated_scores[i] = threshold + 0.01
                    changes += 1
                elif group == privileged_group and score >= threshold:
                    # Slightly reduce privileged group members just above threshold
                    # (only if they're in the critical region)
                    if score < threshold + margin / 2:
                        mitigated_scores[i] = threshold - 0.01
                        changes += 1

        return MitigationResult(
            strategy_used=MitigationStrategy.REJECT_OPTION,
            original_scores=scores.tolist(),
            mitigated_scores=mitigated_scores.tolist(),
            changes_made=changes,
            details={
                "threshold": threshold,
                "margin": margin,
                "privileged_group": str(privileged_group) if favor_unprivileged else None,
                "group_rates": {str(k): v for k, v in group_rates.items()},
            },
        )

    def reweight_samples(
        self,
        group_labels: list[str],
        outcomes: list[bool],
    ) -> dict[str, float]:
        """
        Calculate reweighting factors for samples to reduce bias.

        Samples from underrepresented groups in positive outcomes
        get higher weights.

        Args:
            group_labels: Group label for each sample.
            outcomes: Outcome (positive/negative) for each sample.

        Returns:
            Dictionary mapping (group, outcome) to weight.
        """
        group_labels = np.array(group_labels)
        outcomes = np.array(outcomes)

        n_samples = len(group_labels)
        unique_groups = np.unique(group_labels)

        weights = {}

        # Calculate expected vs actual frequencies
        for group in unique_groups:
            group_mask = group_labels == group
            n_group = np.sum(group_mask)
            p_group = n_group / n_samples

            for outcome in [True, False]:
                outcome_mask = outcomes == outcome
                n_outcome = np.sum(outcome_mask)
                p_outcome = n_outcome / n_samples

                # Expected frequency (if group and outcome were independent)
                expected = p_group * p_outcome * n_samples

                # Actual frequency
                actual = np.sum(group_mask & outcome_mask)

                # Weight is expected / actual
                if actual > 0:
                    weight = expected / actual
                else:
                    weight = 1.0

                weights[(str(group), outcome)] = round(weight, 4)

        return weights

    def mitigate(
        self,
        scores: list[float],
        group_labels: list[str],
        strategy: MitigationStrategy = MitigationStrategy.SCORE_CALIBRATION,
        **kwargs,
    ) -> MitigationResult:
        """
        Apply a bias mitigation strategy.

        Args:
            scores: Match scores.
            group_labels: Group labels for each score.
            strategy: The mitigation strategy to use.
            **kwargs: Additional arguments for the strategy.

        Returns:
            MitigationResult with mitigated scores.
        """
        strategy_methods = {
            MitigationStrategy.SCORE_CALIBRATION: self.calibrate_scores,
            MitigationStrategy.THRESHOLD_ADJUSTMENT: self.adjust_thresholds,
            MitigationStrategy.REJECT_OPTION: self.apply_reject_option,
        }

        if strategy not in strategy_methods:
            raise ValueError(f"Unsupported mitigation strategy: {strategy}")

        method = strategy_methods[strategy]
        return method(scores, group_labels, **kwargs)


# Singleton instance
_mitigator: Optional[BiasMitigator] = None


def get_bias_mitigator() -> BiasMitigator:
    """Get the bias mitigator singleton instance."""
    global _mitigator
    if _mitigator is None:
        _mitigator = BiasMitigator()
    return _mitigator
