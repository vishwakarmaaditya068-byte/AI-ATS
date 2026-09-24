"""
SHAP-style explanations for match scores.

Implements Shapley Additive Explanations (SHAP) to provide
game-theoretic feature attributions for matching decisions.
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Optional

import numpy as np

from src.utils.constants import DEFAULT_SCORING_WEIGHTS
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SHAPValue:
    """SHAP value for a single feature."""

    feature_name: str
    shap_value: float
    base_value_contribution: float
    feature_value: float
    direction: str  # "positive" or "negative"


@dataclass
class SHAPExplanation:
    """SHAP explanation for a prediction."""

    expected_value: float  # Base/average prediction
    predicted_value: float
    shap_values: list[SHAPValue] = field(default_factory=list)
    sum_shap_values: float = 0.0  # Should equal predicted - expected

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for storage (feature -> SHAP value)."""
        return {sv.feature_name: sv.shap_value for sv in self.shap_values}

    def get_positive_contributors(self) -> list[tuple[str, float]]:
        """Get features that positively contribute to the score."""
        return [
            (sv.feature_name, sv.shap_value)
            for sv in sorted(self.shap_values, key=lambda x: x.shap_value, reverse=True)
            if sv.shap_value > 0
        ]

    def get_negative_contributors(self) -> list[tuple[str, float]]:
        """Get features that negatively contribute to the score."""
        return [
            (sv.feature_name, sv.shap_value)
            for sv in sorted(self.shap_values, key=lambda x: x.shap_value)
            if sv.shap_value < 0
        ]

    def explain_difference(self, threshold: float = 0.7) -> str:
        """Generate text explanation of why score is above/below threshold."""
        diff = self.predicted_value - threshold

        if diff >= 0:
            # Score is above threshold - explain why
            positive = self.get_positive_contributors()
            if positive:
                top_reasons = [f"{name} (+{val:.2f})" for name, val in positive[:3]]
                return (
                    f"Score ({self.predicted_value:.2f}) exceeds threshold ({threshold}) "
                    f"mainly due to: {', '.join(top_reasons)}"
                )
        else:
            # Score is below threshold - explain why
            negative = self.get_negative_contributors()
            if negative:
                top_reasons = [f"{name} ({val:.2f})" for name, val in negative[:3]]
                return (
                    f"Score ({self.predicted_value:.2f}) is below threshold ({threshold}) "
                    f"mainly due to: {', '.join(top_reasons)}"
                )

        return f"Score: {self.predicted_value:.2f}, Threshold: {threshold}"


class SHAPExplainer:
    """
    SHAP explainer for match scores.

    Uses Shapley values from cooperative game theory to fairly
    attribute the prediction to each feature.
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        baseline_values: Optional[dict[str, float]] = None,
    ):
        """
        Initialize the SHAP explainer.

        Args:
            weights: Scoring weights for the matching model.
            baseline_values: Baseline/average values for each feature.
                           Used as the "absence" of a feature.
        """
        self.weights = weights or DEFAULT_SCORING_WEIGHTS
        self.baseline_values = baseline_values or {
            "skills_match": 0.5,
            "experience_match": 0.5,
            "education_match": 0.5,
            "semantic_similarity": 0.5,
            "keyword_match": 0.5,
        }

    def explain(
        self,
        feature_values: dict[str, float],
        predict_fn: Optional[Callable[[dict[str, float]], float]] = None,
    ) -> SHAPExplanation:
        """
        Calculate SHAP values for a prediction.

        Uses exact Shapley value computation for the small number
        of features in our matching model.

        Args:
            feature_values: Dictionary of feature name -> value.
            predict_fn: Function that takes feature dict and returns score.
                       If None, uses weighted sum.

        Returns:
            SHAPExplanation with SHAP values for each feature.
        """
        if predict_fn is None:
            predict_fn = self._default_predict

        features = list(feature_values.keys())
        n_features = len(features)

        # Calculate expected value (prediction with all baseline values)
        expected_value = predict_fn(self.baseline_values)

        # Calculate actual prediction
        predicted_value = predict_fn(feature_values)

        # Calculate SHAP values using exact Shapley computation
        shap_values = self._compute_shapley_values(
            features, feature_values, predict_fn
        )

        # Create SHAP value objects
        shap_value_objects = []
        for feature in features:
            sv = shap_values[feature]
            shap_value_objects.append(SHAPValue(
                feature_name=self._format_name(feature),
                shap_value=round(sv, 4),
                base_value_contribution=round(
                    self.baseline_values.get(feature, 0.5) * self.weights.get(feature, 0), 4
                ),
                feature_value=feature_values[feature],
                direction="positive" if sv > 0 else "negative",
            ))

        # Sort by absolute SHAP value
        shap_value_objects.sort(key=lambda x: abs(x.shap_value), reverse=True)

        return SHAPExplanation(
            expected_value=round(expected_value, 4),
            predicted_value=round(predicted_value, 4),
            shap_values=shap_value_objects,
            sum_shap_values=round(sum(shap_values.values()), 4),
        )

    def _compute_shapley_values(
        self,
        features: list[str],
        feature_values: dict[str, float],
        predict_fn: Callable,
    ) -> dict[str, float]:
        """
        Compute exact Shapley values.

        For a small number of features, we can compute exact Shapley values
        by considering all possible coalitions.

        Args:
            features: List of feature names.
            feature_values: Actual feature values.
            predict_fn: Prediction function.

        Returns:
            Dictionary of feature -> Shapley value.
        """
        n = len(features)
        shapley_values = {f: 0.0 for f in features}

        # For each feature, calculate its Shapley value
        for i, feature in enumerate(features):
            other_features = [f for f in features if f != feature]

            # Consider all subsets of other features
            for subset_size in range(n):
                for subset in combinations(other_features, subset_size):
                    subset = set(subset)

                    # Coalition without feature i
                    coalition_without = self._create_coalition(
                        features, feature_values, subset
                    )
                    v_without = predict_fn(coalition_without)

                    # Coalition with feature i
                    coalition_with = self._create_coalition(
                        features, feature_values, subset | {feature}
                    )
                    v_with = predict_fn(coalition_with)

                    # Marginal contribution
                    marginal = v_with - v_without

                    # Shapley weight
                    weight = (
                        self._factorial(subset_size) *
                        self._factorial(n - subset_size - 1) /
                        self._factorial(n)
                    )

                    shapley_values[feature] += weight * marginal

        return shapley_values

    def _create_coalition(
        self,
        features: list[str],
        feature_values: dict[str, float],
        coalition: set[str],
    ) -> dict[str, float]:
        """
        Create feature values for a coalition.

        Features in the coalition use actual values,
        features not in coalition use baseline values.
        """
        result = {}
        for f in features:
            if f in coalition:
                result[f] = feature_values[f]
            else:
                result[f] = self.baseline_values.get(f, 0.5)
        return result

    def _factorial(self, n: int) -> int:
        """Calculate factorial."""
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def _default_predict(self, features: dict[str, float]) -> float:
        """Default prediction function using weighted sum."""
        total = 0
        for key, value in features.items():
            weight = self.weights.get(key, 0)
            total += value * weight
        return total

    def _format_name(self, key: str) -> str:
        """Format feature key to human-readable name."""
        return key.replace("_", " ").title()

    def explain_interaction(
        self,
        feature_values: dict[str, float],
        feature1: str,
        feature2: str,
        predict_fn: Optional[Callable] = None,
    ) -> float:
        """
        Calculate SHAP interaction value between two features.

        The interaction value captures the additional effect when
        both features are present together.

        Args:
            feature_values: Dictionary of feature values.
            feature1: First feature name.
            feature2: Second feature name.
            predict_fn: Prediction function.

        Returns:
            SHAP interaction value.
        """
        if predict_fn is None:
            predict_fn = self._default_predict

        # For a linear model, interactions are zero
        # This would be non-zero for non-linear models
        # Here we provide a simple approximation

        # Get predictions with different combinations
        both_present = feature_values.copy()
        neither_present = feature_values.copy()
        only_f1 = feature_values.copy()
        only_f2 = feature_values.copy()

        neither_present[feature1] = self.baseline_values.get(feature1, 0.5)
        neither_present[feature2] = self.baseline_values.get(feature2, 0.5)

        only_f1[feature2] = self.baseline_values.get(feature2, 0.5)
        only_f2[feature1] = self.baseline_values.get(feature1, 0.5)

        v_both = predict_fn(both_present)
        v_neither = predict_fn(neither_present)
        v_f1 = predict_fn(only_f1)
        v_f2 = predict_fn(only_f2)

        # Interaction = v(both) - v(f1 only) - v(f2 only) + v(neither)
        interaction = v_both - v_f1 - v_f2 + v_neither

        return round(interaction, 4)

    def generate_force_plot_data(
        self,
        explanation: SHAPExplanation,
    ) -> dict[str, Any]:
        """
        Generate data for a SHAP force plot visualization.

        Args:
            explanation: SHAP explanation to visualize.

        Returns:
            Dictionary with force plot data.
        """
        positive_features = []
        negative_features = []

        for sv in explanation.shap_values:
            feature_data = {
                "name": sv.feature_name,
                "value": sv.shap_value,
                "feature_value": sv.feature_value,
            }
            if sv.shap_value > 0:
                positive_features.append(feature_data)
            else:
                negative_features.append(feature_data)

        return {
            "base_value": explanation.expected_value,
            "output_value": explanation.predicted_value,
            "positive_features": sorted(
                positive_features, key=lambda x: x["value"], reverse=True
            ),
            "negative_features": sorted(
                negative_features, key=lambda x: x["value"]
            ),
        }


# Singleton instance
_explainer: Optional[SHAPExplainer] = None


def get_shap_explainer() -> SHAPExplainer:
    """Get the SHAP explainer singleton instance."""
    global _explainer
    if _explainer is None:
        _explainer = SHAPExplainer()
    return _explainer
