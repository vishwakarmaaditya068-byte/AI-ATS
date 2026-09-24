"""
Feature importance calculation for match score explanations.

Provides methods to calculate and visualize which features
contributed most to a candidate's match score.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.utils.constants import DEFAULT_SCORING_WEIGHTS
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureContribution:
    """Contribution of a single feature to the match score."""

    feature_name: str
    feature_value: Any
    raw_score: float  # The score for this feature (0-1)
    weight: float  # The weight applied to this feature
    weighted_contribution: float  # raw_score * weight
    contribution_percentage: float  # Percentage of total score
    direction: str  # "positive", "negative", "neutral"
    description: str  # Human-readable description


@dataclass
class FeatureImportanceResult:
    """Complete feature importance analysis."""

    total_score: float
    features: list[FeatureContribution] = field(default_factory=list)
    top_positive_features: list[str] = field(default_factory=list)
    top_negative_features: list[str] = field(default_factory=list)
    feature_ranking: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "total_score": self.total_score,
            "features": [
                {
                    "name": f.feature_name,
                    "value": f.feature_value,
                    "raw_score": f.raw_score,
                    "weight": f.weight,
                    "contribution": f.weighted_contribution,
                    "percentage": f.contribution_percentage,
                    "direction": f.direction,
                }
                for f in self.features
            ],
            "top_positive": self.top_positive_features,
            "top_negative": self.top_negative_features,
            "ranking": self.feature_ranking,
        }


class FeatureImportanceCalculator:
    """
    Calculates feature importance for match scores.

    Analyzes how each scoring component contributes to the
    final match score, providing transparency into the
    matching decision.
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        baseline_score: float = 0.5,
    ):
        """
        Initialize the feature importance calculator.

        Args:
            weights: Scoring weights for each feature.
            baseline_score: Baseline score for comparison (default 0.5).
        """
        self.weights = weights or DEFAULT_SCORING_WEIGHTS
        self.baseline_score = baseline_score

    def calculate(
        self,
        skills_score: float,
        experience_score: float,
        education_score: float,
        semantic_score: float,
        keyword_score: float,
        skill_details: Optional[dict] = None,
        experience_details: Optional[dict] = None,
        education_details: Optional[dict] = None,
    ) -> FeatureImportanceResult:
        """
        Calculate feature importance for a match result.

        Args:
            skills_score: Skills matching score (0-1).
            experience_score: Experience matching score (0-1).
            education_score: Education matching score (0-1).
            semantic_score: Semantic similarity score (0-1).
            keyword_score: Keyword matching score (0-1).
            skill_details: Optional detailed skill matching info.
            experience_details: Optional detailed experience info.
            education_details: Optional detailed education info.

        Returns:
            FeatureImportanceResult with detailed breakdown.
        """
        features = []

        # Calculate weighted contributions
        scores = {
            "skills_match": skills_score,
            "experience_match": experience_score,
            "education_match": education_score,
            "semantic_similarity": semantic_score,
            "keyword_match": keyword_score,
        }

        total_weighted = sum(
            scores[key] * self.weights[key]
            for key in scores
        )

        # Create feature contributions
        for feature_key, raw_score in scores.items():
            weight = self.weights[feature_key]
            weighted = raw_score * weight
            percentage = (weighted / total_weighted * 100) if total_weighted > 0 else 0

            # Determine direction relative to baseline
            if raw_score >= 0.7:
                direction = "positive"
            elif raw_score <= 0.3:
                direction = "negative"
            else:
                direction = "neutral"

            # Generate description
            description = self._generate_feature_description(
                feature_key, raw_score, skill_details, experience_details, education_details
            )

            features.append(FeatureContribution(
                feature_name=self._format_feature_name(feature_key),
                feature_value=raw_score,
                raw_score=raw_score,
                weight=weight,
                weighted_contribution=round(weighted, 4),
                contribution_percentage=round(percentage, 1),
                direction=direction,
                description=description,
            ))

        # Sort by contribution
        features.sort(key=lambda f: f.weighted_contribution, reverse=True)

        # Identify top positive and negative features
        top_positive = [
            f.feature_name for f in features
            if f.direction == "positive"
        ][:3]

        top_negative = [
            f.feature_name for f in features
            if f.direction == "negative"
        ][:3]

        # Feature ranking by importance
        feature_ranking = [f.feature_name for f in features]

        return FeatureImportanceResult(
            total_score=round(total_weighted, 4),
            features=features,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
            feature_ranking=feature_ranking,
        )

    def calculate_marginal_contributions(
        self,
        scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate marginal contribution of each feature.

        Shows how much each feature adds to the score above
        the baseline.

        Args:
            scores: Dictionary of feature scores.

        Returns:
            Dictionary of marginal contributions.
        """
        marginal = {}

        for feature_key, score in scores.items():
            weight = self.weights.get(feature_key, 0)
            # Marginal contribution = (score - baseline) * weight
            marginal[feature_key] = round(
                (score - self.baseline_score) * weight, 4
            )

        return marginal

    def calculate_permutation_importance(
        self,
        base_score: float,
        scores: dict[str, float],
        n_permutations: int = 10,
    ) -> dict[str, float]:
        """
        Calculate permutation importance for each feature.

        Measures how much the score drops when each feature
        is replaced with a random/baseline value.

        Args:
            base_score: The original total score.
            scores: Dictionary of feature scores.
            n_permutations: Number of permutations (for averaging).

        Returns:
            Dictionary of importance scores (higher = more important).
        """
        import random

        importance = {}

        for feature_key in scores:
            drops = []

            for _ in range(n_permutations):
                # Create perturbed scores
                perturbed = scores.copy()
                # Replace with random value between 0 and 1
                perturbed[feature_key] = random.random()

                # Calculate new score
                new_score = sum(
                    perturbed[k] * self.weights.get(k, 0)
                    for k in perturbed
                )

                drops.append(base_score - new_score)

            # Average drop is the importance
            importance[feature_key] = round(sum(drops) / len(drops), 4)

        return importance

    def _format_feature_name(self, key: str) -> str:
        """Format feature key to human-readable name."""
        name_map = {
            "skills_match": "Skills Match",
            "experience_match": "Experience Match",
            "education_match": "Education Match",
            "semantic_similarity": "Semantic Similarity",
            "keyword_match": "Keyword Match",
        }
        return name_map.get(key, key.replace("_", " ").title())

    def _generate_feature_description(
        self,
        feature_key: str,
        score: float,
        skill_details: Optional[dict],
        experience_details: Optional[dict],
        education_details: Optional[dict],
    ) -> str:
        """Generate human-readable description for a feature."""
        score_pct = int(score * 100)

        if feature_key == "skills_match":
            if skill_details:
                matched = skill_details.get("matched_count", 0)
                total = skill_details.get("total_required", 0)
                return f"Matched {matched}/{total} required skills ({score_pct}%)"
            return f"Skills alignment score: {score_pct}%"

        elif feature_key == "experience_match":
            if experience_details:
                candidate_years = experience_details.get("candidate_years", 0)
                required_years = experience_details.get("required_years", 0)
                return f"{candidate_years:.1f} years experience (required: {required_years:.1f})"
            return f"Experience match score: {score_pct}%"

        elif feature_key == "education_match":
            if education_details:
                candidate_deg = education_details.get("candidate_degree", "N/A")
                required_deg = education_details.get("required_degree", "N/A")
                return f"Has {candidate_deg} (required: {required_deg})"
            return f"Education match score: {score_pct}%"

        elif feature_key == "semantic_similarity":
            if score >= 0.8:
                return f"Strong semantic alignment ({score_pct}%) with job description"
            elif score >= 0.5:
                return f"Moderate semantic alignment ({score_pct}%) with job description"
            return f"Low semantic alignment ({score_pct}%) with job description"

        elif feature_key == "keyword_match":
            return f"Keyword overlap score: {score_pct}%"

        return f"{feature_key}: {score_pct}%"


# Singleton instance
_calculator: Optional[FeatureImportanceCalculator] = None


def get_feature_importance_calculator() -> FeatureImportanceCalculator:
    """Get the feature importance calculator singleton instance."""
    global _calculator
    if _calculator is None:
        _calculator = FeatureImportanceCalculator()
    return _calculator
