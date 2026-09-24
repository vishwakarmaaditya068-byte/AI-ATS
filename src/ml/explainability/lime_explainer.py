"""
LIME-style local explanations for match scores.

Implements Local Interpretable Model-agnostic Explanations (LIME)
to explain individual matching decisions by approximating the
model locally with an interpretable model.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from src.utils.constants import DEFAULT_SCORING_WEIGHTS
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LIMEFeatureWeight:
    """Weight assigned to a feature by LIME."""

    feature_name: str
    weight: float
    normalized_weight: float
    direction: str  # "positive" or "negative"
    importance_rank: int


@dataclass
class LIMEExplanation:
    """LIME explanation for a single prediction."""

    predicted_score: float
    local_prediction: float  # Prediction from the interpretable model
    intercept: float
    feature_weights: list[LIMEFeatureWeight] = field(default_factory=list)
    r_squared: float = 0.0  # How well the local model fits
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "predicted_score": self.predicted_score,
            "local_prediction": self.local_prediction,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "num_samples": self.num_samples,
            "feature_weights": {
                fw.feature_name: {
                    "weight": fw.weight,
                    "normalized": fw.normalized_weight,
                    "direction": fw.direction,
                    "rank": fw.importance_rank,
                }
                for fw in self.feature_weights
            },
        }

    def get_top_features(self, n: int = 5, positive_only: bool = False) -> list[str]:
        """Get top n features by absolute weight."""
        features = self.feature_weights
        if positive_only:
            features = [f for f in features if f.direction == "positive"]
        sorted_features = sorted(features, key=lambda f: abs(f.weight), reverse=True)
        return [f.feature_name for f in sorted_features[:n]]


class LIMEExplainer:
    """
    LIME-style explainer for match scores.

    Creates local linear approximations to explain why a
    candidate received a particular match score.
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        num_samples: int = 100,
        kernel_width: float = 0.25,
    ):
        """
        Initialize the LIME explainer.

        Args:
            weights: Scoring weights for the matching model.
            num_samples: Number of perturbed samples to generate.
            kernel_width: Width of the exponential kernel for weighting samples.
        """
        self.weights = weights or DEFAULT_SCORING_WEIGHTS
        self.num_samples = num_samples
        self.kernel_width = kernel_width

    def explain(
        self,
        feature_values: dict[str, float],
        predict_fn: Optional[Callable[[dict[str, float]], float]] = None,
    ) -> LIMEExplanation:
        """
        Generate LIME explanation for a prediction.

        Args:
            feature_values: Dictionary of feature name -> value.
            predict_fn: Function that takes feature dict and returns score.
                       If None, uses weighted sum.

        Returns:
            LIMEExplanation with feature weights.
        """
        if predict_fn is None:
            predict_fn = self._default_predict

        # Get the original prediction
        original_prediction = predict_fn(feature_values)

        # Generate perturbed samples
        samples, sample_weights = self._generate_samples(feature_values)

        # Get predictions for all samples
        predictions = np.array([
            predict_fn(dict(zip(feature_values.keys(), sample)))
            for sample in samples
        ])

        # Fit local linear model
        coefficients, intercept, r_squared = self._fit_local_model(
            samples, predictions, sample_weights
        )

        # Create feature weights
        feature_names = list(feature_values.keys())
        feature_weights = []

        # Normalize weights for interpretability
        total_abs_weight = sum(abs(c) for c in coefficients)
        if total_abs_weight == 0:
            total_abs_weight = 1

        for i, (name, coef) in enumerate(zip(feature_names, coefficients)):
            normalized = coef / total_abs_weight
            feature_weights.append(LIMEFeatureWeight(
                feature_name=self._format_name(name),
                weight=round(coef, 4),
                normalized_weight=round(normalized, 4),
                direction="positive" if coef > 0 else "negative",
                importance_rank=0,  # Will be set below
            ))

        # Assign importance ranks
        sorted_by_importance = sorted(
            enumerate(feature_weights),
            key=lambda x: abs(x[1].weight),
            reverse=True
        )
        for rank, (idx, _) in enumerate(sorted_by_importance, 1):
            feature_weights[idx].importance_rank = rank

        # Calculate local prediction
        original_features = np.array(list(feature_values.values()))
        local_prediction = float(np.dot(coefficients, original_features) + intercept)

        return LIMEExplanation(
            predicted_score=round(original_prediction, 4),
            local_prediction=round(local_prediction, 4),
            intercept=round(intercept, 4),
            feature_weights=feature_weights,
            r_squared=round(r_squared, 4),
            num_samples=self.num_samples,
        )

    def explain_text_features(
        self,
        resume_text: str,
        jd_text: str,
        base_score: float,
        text_processor: Optional[Callable] = None,
    ) -> dict[str, float]:
        """
        Explain which text features (words/phrases) contribute to the score.

        This is a simplified version that identifies important keywords.

        Args:
            resume_text: The resume text.
            jd_text: The job description text.
            base_score: The base match score.
            text_processor: Optional function to process text.

        Returns:
            Dictionary of text feature -> importance.
        """
        # Extract keywords from JD
        jd_words = set(self._extract_keywords(jd_text))
        resume_words = set(self._extract_keywords(resume_text))

        # Find overlapping and missing keywords
        matched = jd_words & resume_words
        missing = jd_words - resume_words

        # Calculate importance based on presence/absence
        importance = {}

        for word in matched:
            # Positive contribution for matched keywords
            importance[f"+{word}"] = round(0.1 / max(len(matched), 1), 4)

        for word in list(missing)[:10]:  # Limit missing to top 10
            # Negative contribution for missing keywords
            importance[f"-{word}"] = round(-0.05 / max(len(missing), 1), 4)

        return importance

    def _generate_samples(
        self,
        feature_values: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate perturbed samples around the original instance.

        Returns:
            Tuple of (samples array, sample weights array).
        """
        original = np.array(list(feature_values.values()))
        n_features = len(original)

        # Generate samples by perturbing the original
        samples = np.zeros((self.num_samples, n_features))
        samples[0] = original  # First sample is the original

        for i in range(1, self.num_samples):
            # Add Gaussian noise
            noise = np.random.normal(0, 0.2, n_features)
            perturbed = original + noise
            # Clip to valid range [0, 1]
            samples[i] = np.clip(perturbed, 0, 1)

        # Calculate sample weights using exponential kernel
        distances = np.sqrt(np.sum((samples - original) ** 2, axis=1))
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))

        return samples, weights

    def _fit_local_model(
        self,
        samples: np.ndarray,
        predictions: np.ndarray,
        sample_weights: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """
        Fit a weighted linear regression model.

        Returns:
            Tuple of (coefficients, intercept, r_squared).
        """
        # Weighted least squares
        W = np.diag(sample_weights)
        X = samples
        y = predictions

        # Add bias term
        X_bias = np.c_[X, np.ones(X.shape[0])]

        try:
            # Solve weighted least squares: (X'WX)^-1 X'Wy
            XtWX = X_bias.T @ W @ X_bias
            XtWy = X_bias.T @ W @ y

            # Add regularization for numerical stability
            XtWX += np.eye(XtWX.shape[0]) * 1e-6

            params = np.linalg.solve(XtWX, XtWy)

            coefficients = params[:-1]
            intercept = params[-1]

            # Calculate R-squared
            y_pred = X_bias @ params
            ss_res = np.sum(sample_weights * (y - y_pred) ** 2)
            ss_tot = np.sum(sample_weights * (y - np.average(y, weights=sample_weights)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        except np.linalg.LinAlgError:
            # Fallback to simple coefficients
            logger.warning("Linear algebra error in LIME, using fallback")
            coefficients = np.zeros(samples.shape[1])
            intercept = np.mean(predictions)
            r_squared = 0

        return coefficients, float(intercept), float(max(0, r_squared))

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

    def _extract_keywords(self, text: str, min_length: int = 4) -> list[str]:
        """Extract keywords from text."""
        import re

        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter by length and common stopwords
        stopwords = {
            'with', 'have', 'will', 'that', 'this', 'from', 'your', 'they',
            'their', 'what', 'when', 'where', 'which', 'would', 'could',
            'should', 'must', 'able', 'about', 'been', 'being', 'than',
            'then', 'into', 'over', 'such', 'only', 'other', 'some',
        }
        return [w for w in words if len(w) >= min_length and w not in stopwords]


# Singleton instance
_explainer: Optional[LIMEExplainer] = None


def get_lime_explainer() -> LIMEExplainer:
    """Get the LIME explainer singleton instance."""
    global _explainer
    if _explainer is None:
        _explainer = LIMEExplainer()
    return _explainer
