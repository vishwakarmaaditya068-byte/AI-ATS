"""
Main explainability orchestrator for AI-ATS.

Coordinates feature importance, LIME, and SHAP explanations
to provide comprehensive, interpretable match score explanations.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from src.utils.constants import DEFAULT_SCORING_WEIGHTS
from src.utils.logger import get_logger

from .feature_importance import (
    FeatureImportanceCalculator,
    FeatureImportanceResult,
    get_feature_importance_calculator,
)
from .lime_explainer import (
    LIMEExplainer,
    LIMEExplanation,
    get_lime_explainer,
)
from .shap_explainer import (
    SHAPExplainer,
    SHAPExplanation,
    get_shap_explainer,
)

logger = get_logger(__name__)


@dataclass
class MatchExplanation:
    """Complete explanation for a match score."""

    # Basic info
    candidate_name: str
    job_title: str
    overall_score: float

    # Feature importance
    feature_importance: Optional[FeatureImportanceResult] = None

    # LIME explanation
    lime_explanation: Optional[LIMEExplanation] = None

    # SHAP explanation
    shap_explanation: Optional[SHAPExplanation] = None

    # Human-readable summary
    summary: str = ""
    key_strengths: list[str] = field(default_factory=list)
    key_gaps: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "candidate_name": self.candidate_name,
            "job_title": self.job_title,
            "overall_score": self.overall_score,
            "summary": self.summary,
            "key_strengths": self.key_strengths,
            "key_gaps": self.key_gaps,
            "recommendations": self.recommendations,
            "feature_importance": (
                self.feature_importance.to_dict()
                if self.feature_importance else None
            ),
            "lime_explanation": (
                self.lime_explanation.to_dict()
                if self.lime_explanation else None
            ),
            "shap_values": (
                self.shap_explanation.to_dict()
                if self.shap_explanation else None
            ),
        }

    def get_lime_dict(self) -> Optional[dict[str, Any]]:
        """Get LIME explanation as dictionary for storage in Explanation model."""
        if self.lime_explanation:
            return self.lime_explanation.to_dict()
        return None

    def get_shap_dict(self) -> Optional[dict[str, float]]:
        """Get SHAP values as dictionary for storage in Explanation model."""
        if self.shap_explanation:
            return self.shap_explanation.to_dict()
        return None


class MatchExplainer:
    """
    Main explainer for match scores.

    Provides comprehensive explanations using multiple techniques:
    - Feature importance analysis
    - LIME local explanations
    - SHAP values
    - Human-readable summaries
    """

    def __init__(
        self,
        feature_calculator: Optional[FeatureImportanceCalculator] = None,
        lime_explainer: Optional[LIMEExplainer] = None,
        shap_explainer: Optional[SHAPExplainer] = None,
        weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize the match explainer.

        Args:
            feature_calculator: Optional custom feature importance calculator.
            lime_explainer: Optional custom LIME explainer.
            shap_explainer: Optional custom SHAP explainer.
            weights: Scoring weights for the matching model.
        """
        self._feature_calculator = feature_calculator
        self._lime_explainer = lime_explainer
        self._shap_explainer = shap_explainer
        self.weights = weights or DEFAULT_SCORING_WEIGHTS

    @property
    def feature_calculator(self) -> FeatureImportanceCalculator:
        """Get the feature importance calculator (lazy initialization)."""
        if self._feature_calculator is None:
            self._feature_calculator = get_feature_importance_calculator()
        return self._feature_calculator

    @property
    def lime_explainer(self) -> LIMEExplainer:
        """Get the LIME explainer (lazy initialization)."""
        if self._lime_explainer is None:
            self._lime_explainer = get_lime_explainer()
        return self._lime_explainer

    @property
    def shap_explainer(self) -> SHAPExplainer:
        """Get the SHAP explainer (lazy initialization)."""
        if self._shap_explainer is None:
            self._shap_explainer = get_shap_explainer()
        return self._shap_explainer

    def explain(
        self,
        candidate_name: str,
        job_title: str,
        skills_score: float,
        experience_score: float,
        education_score: float,
        semantic_score: float,
        keyword_score: float,
        overall_score: float,
        skill_details: Optional[dict] = None,
        experience_details: Optional[dict] = None,
        education_details: Optional[dict] = None,
        include_lime: bool = True,
        include_shap: bool = True,
    ) -> MatchExplanation:
        """
        Generate comprehensive explanation for a match.

        Args:
            candidate_name: Name of the candidate.
            job_title: Title of the job.
            skills_score: Skills matching score (0-1).
            experience_score: Experience matching score (0-1).
            education_score: Education matching score (0-1).
            semantic_score: Semantic similarity score (0-1).
            keyword_score: Keyword matching score (0-1).
            overall_score: Overall match score (0-1).
            skill_details: Optional detailed skill matching info.
            experience_details: Optional detailed experience info.
            education_details: Optional detailed education info.
            include_lime: Whether to include LIME explanation.
            include_shap: Whether to include SHAP explanation.

        Returns:
            MatchExplanation with all explanation components.
        """
        # Prepare feature values
        feature_values = {
            "skills_match": skills_score,
            "experience_match": experience_score,
            "education_match": education_score,
            "semantic_similarity": semantic_score,
            "keyword_match": keyword_score,
        }

        # Calculate feature importance
        feature_importance = self.feature_calculator.calculate(
            skills_score=skills_score,
            experience_score=experience_score,
            education_score=education_score,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            skill_details=skill_details,
            experience_details=experience_details,
            education_details=education_details,
        )

        # Generate LIME explanation
        lime_explanation = None
        if include_lime:
            try:
                lime_explanation = self.lime_explainer.explain(feature_values)
            except Exception as e:
                logger.warning(f"LIME explanation failed: {e}")

        # Generate SHAP explanation
        shap_explanation = None
        if include_shap:
            try:
                shap_explanation = self.shap_explainer.explain(feature_values)
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")

        # Generate human-readable components
        summary = self._generate_summary(
            candidate_name, job_title, overall_score, feature_importance
        )
        strengths = self._identify_strengths(feature_importance, shap_explanation)
        gaps = self._identify_gaps(feature_importance, shap_explanation)
        recommendations = self._generate_recommendations(
            feature_importance, skill_details, shap=shap_explanation
        )

        return MatchExplanation(
            candidate_name=candidate_name,
            job_title=job_title,
            overall_score=overall_score,
            feature_importance=feature_importance,
            lime_explanation=lime_explanation,
            shap_explanation=shap_explanation,
            summary=summary,
            key_strengths=strengths,
            key_gaps=gaps,
            recommendations=recommendations,
        )

    def explain_score_difference(
        self,
        score1: float,
        score2: float,
        features1: dict[str, float],
        features2: dict[str, float],
    ) -> dict[str, Any]:
        """
        Explain the difference between two match scores.

        Useful for comparing candidates or understanding why
        scores changed.

        Args:
            score1: First match score.
            score2: Second match score.
            features1: Feature values for first score.
            features2: Feature values for second score.

        Returns:
            Dictionary explaining the score difference.
        """
        differences = {}
        total_diff = 0

        for feature in features1:
            if feature in features2:
                value_diff = features2[feature] - features1[feature]
                weight = self.weights.get(feature, 0)
                weighted_diff = value_diff * weight

                differences[feature] = {
                    "value1": features1[feature],
                    "value2": features2[feature],
                    "difference": round(value_diff, 4),
                    "weighted_impact": round(weighted_diff, 4),
                }
                total_diff += weighted_diff

        # Sort by absolute impact
        sorted_features = sorted(
            differences.items(),
            key=lambda x: abs(x[1]["weighted_impact"]),
            reverse=True
        )

        return {
            "score1": score1,
            "score2": score2,
            "score_difference": round(score2 - score1, 4),
            "explained_difference": round(total_diff, 4),
            "feature_differences": dict(sorted_features),
            "top_contributing_features": [f[0] for f in sorted_features[:3]],
        }

    def explain_threshold_decision(
        self,
        score: float,
        threshold: float,
        feature_values: dict[str, float],
    ) -> dict[str, Any]:
        """
        Explain why a score is above or below a threshold.

        Args:
            score: The match score.
            threshold: Decision threshold.
            feature_values: Feature values for the score.

        Returns:
            Dictionary explaining the threshold decision.
        """
        shap_exp = self.shap_explainer.explain(feature_values)

        above_threshold = score >= threshold
        gap = score - threshold

        # Find features pushing toward/away from threshold
        if above_threshold:
            helping_features = shap_exp.get_positive_contributors()
            hurting_features = shap_exp.get_negative_contributors()
        else:
            helping_features = shap_exp.get_negative_contributors()
            hurting_features = shap_exp.get_positive_contributors()

        return {
            "score": score,
            "threshold": threshold,
            "above_threshold": above_threshold,
            "gap": round(gap, 4),
            "decision": "PASS" if above_threshold else "FAIL",
            "features_helping_pass": [
                {"name": n, "contribution": v}
                for n, v in helping_features[:3]
            ],
            "features_hurting_pass": [
                {"name": n, "contribution": v}
                for n, v in hurting_features[:3]
            ],
            "explanation": shap_exp.explain_difference(threshold),
        }

    def _generate_summary(
        self,
        candidate_name: str,
        job_title: str,
        score: float,
        importance: FeatureImportanceResult,
    ) -> str:
        """Generate human-readable summary."""
        score_pct: int = int(score * 100)

        # Determine match quality
        if score >= 0.85:
            quality = "excellent"
        elif score >= 0.70:
            quality = "good"
        elif score >= 0.50:
            quality = "fair"
        else:
            quality = "poor"

        # Get top contributing factor with its score
        top_feature: str = "overall profile"
        top_feature_pct: str = ""
        if importance.feature_ranking:
            top_feature = importance.feature_ranking[0]
            top_feat_obj = next(
                (f for f in importance.features if f.feature_name == top_feature),
                None,
            )
            if top_feat_obj:
                top_feature_pct = f" ({int(top_feat_obj.raw_score * 100)}%)"

        # Mention weakest area if any negative feature exists
        weak_clause: str = ""
        negative_features: list[str] = [
            f.feature_name for f in importance.features if f.direction == "negative"
        ]
        if negative_features:
            weak_clause = f" Weakest area: {negative_features[0]}."

        return (
            f"{candidate_name} is a {quality} match ({score_pct}%) for {job_title}. "
            f"The strongest factor is {top_feature}{top_feature_pct}.{weak_clause}"
        )

    def _identify_strengths(
        self,
        importance: FeatureImportanceResult,
        shap: Optional[SHAPExplanation],
    ) -> list[str]:
        """Identify key strengths. Primary source: SHAP positive contributors.
        Falls back to feature importance direction when SHAP is unavailable."""
        strengths: list[str] = []

        if shap is not None:
            for feature_name, shap_val in shap.get_positive_contributors():
                matching = next(
                    (f for f in importance.features if f.feature_name == feature_name),
                    None,
                )
                if matching:
                    strengths.append(
                        f"{feature_name}: {matching.description} "
                        f"(contributes +{shap_val:.2f} above baseline)"
                    )
                else:
                    strengths.append(
                        f"{feature_name}: above-average contribution (+{shap_val:.2f})"
                    )
        else:
            for feature in importance.features:
                if feature.direction == "positive":
                    strengths.append(f"{feature.feature_name}: {feature.description}")

        return strengths[:5]

    def _identify_gaps(
        self,
        importance: FeatureImportanceResult,
        shap: Optional[SHAPExplanation],
    ) -> list[str]:
        """Identify key gaps. Primary source: SHAP negative contributors.
        Falls back to feature importance direction when SHAP is unavailable."""
        gaps: list[str] = []

        if shap is not None:
            for feature_name, shap_val in shap.get_negative_contributors():
                matching = next(
                    (f for f in importance.features if f.feature_name == feature_name),
                    None,
                )
                if matching:
                    gaps.append(
                        f"{feature_name}: {matching.description} "
                        f"(contributes {shap_val:.2f} below baseline)"
                    )
                else:
                    gaps.append(
                        f"{feature_name}: below-average contribution ({shap_val:.2f})"
                    )
        else:
            for feature in importance.features:
                if feature.direction == "negative":
                    gaps.append(f"{feature.feature_name}: {feature.description}")

        return gaps[:5]

    def _generate_recommendations(
        self,
        importance: FeatureImportanceResult,
        skill_details: Optional[dict],
        shap: Optional[SHAPExplanation] = None,
    ) -> list[str]:
        """Generate score-specific, actionable recommendations for each weak area."""
        recommendations: list[str] = []

        for feature in importance.features:
            score_pct: int = int(feature.raw_score * 100)
            if feature.raw_score < 0.5:
                if "skills" in feature.feature_name.lower():
                    if skill_details and skill_details.get("missing_skills"):
                        missing: list[str] = skill_details["missing_skills"][:3]
                        recommendations.append(
                            f"Skills gap ({score_pct}%): candidate is missing "
                            f"{', '.join(missing)} — assess transferable experience "
                            "or training potential"
                        )
                    else:
                        recommendations.append(
                            f"Skills gap ({score_pct}%): evaluate adjacent skill sets "
                            "or potential for skill development"
                        )
                elif "experience" in feature.feature_name.lower():
                    recommendations.append(
                        f"Experience gap ({score_pct}%): candidate may benefit "
                        "from mentorship to bridge the experience gap"
                    )
                elif "education" in feature.feature_name.lower():
                    recommendations.append(
                        f"Education gap ({score_pct}%): consider equivalent work "
                        "experience or professional certifications"
                    )
                elif "semantic" in feature.feature_name.lower():
                    recommendations.append(
                        f"Domain fit ({score_pct}%): candidate's background may not "
                        "closely align — verify domain knowledge in interview"
                    )

        return list(dict.fromkeys(recommendations))[:5]


# Singleton instance
_explainer: Optional[MatchExplainer] = None


def get_match_explainer() -> MatchExplainer:
    """Get the match explainer singleton instance."""
    global _explainer
    if _explainer is None:
        _explainer = MatchExplainer()
    return _explainer
