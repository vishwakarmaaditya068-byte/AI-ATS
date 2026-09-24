"""
Main bias detection orchestrator for AI-ATS.

Coordinates protected attribute detection, fairness metrics calculation,
and bias mitigation to ensure fair candidate evaluation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from src.data.models import BiasCheckResult
from src.ml.nlp import ResumeParseResult
from src.utils.constants import FAIRNESS_THRESHOLDS
from src.utils.logger import get_logger

from .protected_attributes import (
    ProtectedAttributeDetector,
    AttributeDetectionResult,
    get_attribute_detector,
)
from .fairness_metrics import (
    FairnessCalculator,
    FairnessMetrics,
    get_fairness_calculator,
)
from .bias_mitigation import (
    BiasMitigator,
    MitigationStrategy,
    MitigationResult,
    get_bias_mitigator,
)

logger = get_logger(__name__)


@dataclass
class BiasAnalysisResult:
    """Complete result of bias analysis for a candidate or batch."""

    # Individual candidate analysis
    attribute_detection: Optional[AttributeDetectionResult] = None

    # Batch fairness analysis
    fairness_metrics: Optional[FairnessMetrics] = None

    # Mitigation applied
    mitigation_result: Optional[MitigationResult] = None

    # Overall assessment
    bias_detected: bool = False
    risk_level: str = "low"  # "low", "medium", "high"
    bias_types: list[str] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Audit trail
    analyzed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_version: str = "1.0"


class BiasDetector:
    """
    Main orchestrator for bias detection and mitigation.

    Integrates protected attribute detection, fairness metrics,
    and mitigation strategies into a cohesive pipeline.
    """

    def __init__(
        self,
        attribute_detector: Optional[ProtectedAttributeDetector] = None,
        fairness_calculator: Optional[FairnessCalculator] = None,
        bias_mitigator: Optional[BiasMitigator] = None,
        auto_mitigate: bool = False,
    ):
        """
        Initialize the bias detector.

        Args:
            attribute_detector: Optional custom attribute detector.
            fairness_calculator: Optional custom fairness calculator.
            bias_mitigator: Optional custom bias mitigator.
            auto_mitigate: Whether to automatically apply mitigation.
        """
        self._attribute_detector = attribute_detector
        self._fairness_calculator = fairness_calculator
        self._bias_mitigator = bias_mitigator
        self.auto_mitigate = auto_mitigate

    @property
    def attribute_detector(self) -> ProtectedAttributeDetector:
        """Get the attribute detector (lazy initialization)."""
        if self._attribute_detector is None:
            self._attribute_detector = get_attribute_detector()
        return self._attribute_detector

    @property
    def fairness_calculator(self) -> FairnessCalculator:
        """Get the fairness calculator (lazy initialization)."""
        if self._fairness_calculator is None:
            self._fairness_calculator = get_fairness_calculator()
        return self._fairness_calculator

    @property
    def bias_mitigator(self) -> BiasMitigator:
        """Get the bias mitigator (lazy initialization)."""
        if self._bias_mitigator is None:
            self._bias_mitigator = get_bias_mitigator()
        return self._bias_mitigator

    def analyze_candidate(
        self,
        resume_result: ResumeParseResult,
    ) -> BiasAnalysisResult:
        """
        Analyze a single candidate's resume for potential bias issues.

        Args:
            resume_result: Parsed resume data.

        Returns:
            BiasAnalysisResult with detection results.
        """
        # Extract text from resume
        resume_text = self._get_resume_text(resume_result)

        if not resume_text:
            return BiasAnalysisResult()

        # Detect protected attributes
        attr_result = self.attribute_detector.detect(resume_text)

        # Build result
        result = BiasAnalysisResult(
            attribute_detection=attr_result,
            bias_detected=attr_result.has_protected_attributes,
            risk_level=attr_result.risk_level,
            bias_types=attr_result.attribute_types_found,
            recommendations=attr_result.recommendations,
        )

        if attr_result.has_protected_attributes:
            logger.info(
                f"Bias indicators detected: {attr_result.attribute_types_found} "
                f"(risk: {attr_result.risk_level})"
            )

        return result

    def analyze_batch(
        self,
        scores: list[float],
        group_labels: list[str],
        outcomes: Optional[list[bool]] = None,
        mitigation_strategy: Optional[MitigationStrategy] = None,
    ) -> BiasAnalysisResult:
        """
        Analyze a batch of candidates for fairness across groups.

        Args:
            scores: Match scores for all candidates.
            group_labels: Demographic group for each candidate.
            outcomes: Optional actual outcomes (selected/not selected).
            mitigation_strategy: Optional strategy to apply if bias detected.

        Returns:
            BiasAnalysisResult with fairness metrics and mitigation.
        """
        # Calculate fairness metrics
        fairness = self.fairness_calculator.calculate(
            scores=scores,
            group_labels=group_labels,
            outcomes=outcomes,
        )

        # Determine overall bias detection
        bias_detected = not fairness.is_fair
        risk_level = self._calculate_batch_risk_level(fairness)
        bias_types = self._identify_bias_types(fairness)

        result = BiasAnalysisResult(
            fairness_metrics=fairness,
            bias_detected=bias_detected,
            risk_level=risk_level,
            bias_types=bias_types,
            recommendations=self._generate_batch_recommendations(fairness),
        )

        # Apply mitigation if requested or auto-enabled
        strategy = mitigation_strategy
        if bias_detected and self.auto_mitigate and strategy is None:
            strategy = MitigationStrategy.SCORE_CALIBRATION

        if bias_detected and strategy:
            result.mitigation_result = self.bias_mitigator.mitigate(
                scores=scores,
                group_labels=group_labels,
                strategy=strategy,
            )
            logger.info(
                f"Bias mitigation applied using {strategy.value}: "
                f"{result.mitigation_result.changes_made} changes made"
            )

        return result

    def check_match_for_bias(
        self,
        resume_result: ResumeParseResult,
        match_score: float,
    ) -> BiasCheckResult:
        """
        Check a specific match for bias, returning a BiasCheckResult
        compatible with the Match model.

        Args:
            resume_result: Parsed resume data.
            match_score: The match score assigned.

        Returns:
            BiasCheckResult for storing with the match.
        """
        # Analyze the candidate
        analysis = self.analyze_candidate(resume_result)

        # Create BiasCheckResult
        bias_check = BiasCheckResult(
            potential_bias_detected=analysis.bias_detected,
            bias_type=analysis.bias_types[0] if analysis.bias_types else None,
            bias_confidence=self._calculate_bias_confidence(analysis),
            bias_description=self._generate_bias_description(analysis),
            protected_attributes_found=analysis.bias_types,
            mitigation_applied=False,
            mitigation_description=None,
        )

        return bias_check

    def redact_resume(
        self,
        resume_text: str,
        categories: Optional[list[str]] = None,
    ) -> str:
        """
        Redact protected attributes from resume text.

        Args:
            resume_text: Original resume text.
            categories: Optional specific categories to redact.

        Returns:
            Redacted resume text.
        """
        result = self.bias_mitigator.redact_protected_attributes(
            text=resume_text,
            categories=categories,
        )
        return result.redacted_text

    def _get_resume_text(self, resume_result: ResumeParseResult) -> str:
        """Extract text content from parsed resume."""
        if resume_result.extraction_result:
            return resume_result.extraction_result.text
        elif resume_result.preprocessed:
            return resume_result.preprocessed.cleaned_text
        return ""

    def _calculate_batch_risk_level(self, fairness: FairnessMetrics) -> str:
        """Calculate risk level from fairness metrics."""
        if not fairness.violations:
            return "low"

        # Count severity of violations
        severe_violations = 0
        for violation in fairness.violations:
            if "disparate impact" in violation.lower():
                severe_violations += 2  # Disparate impact is legally significant
            else:
                severe_violations += 1

        if severe_violations >= 3:
            return "high"
        elif severe_violations >= 1:
            return "medium"
        return "low"

    def _identify_bias_types(self, fairness: FairnessMetrics) -> list[str]:
        """Identify types of bias from fairness metrics."""
        types = []

        if fairness.demographic_parity_difference > FAIRNESS_THRESHOLDS["demographic_parity_difference"]:
            types.append("demographic_parity")

        if fairness.equalized_odds_difference > FAIRNESS_THRESHOLDS["equalized_odds_difference"]:
            types.append("equalized_odds")

        if fairness.disparate_impact_ratio < FAIRNESS_THRESHOLDS["disparate_impact_ratio"]:
            types.append("disparate_impact")

        if fairness.score_gap > 0.2:
            types.append("score_distribution")

        return types

    def _generate_batch_recommendations(
        self,
        fairness: FairnessMetrics,
    ) -> list[str]:
        """Generate recommendations based on fairness analysis."""
        recommendations = []

        if not fairness.is_fair:
            recommendations.append(
                "Fairness violations detected. Review selection criteria "
                "and consider applying bias mitigation."
            )

        if fairness.disparate_impact_ratio < 0.8:
            recommendations.append(
                f"Disparate impact ratio ({fairness.disparate_impact_ratio:.2f}) "
                "is below the 4/5ths rule threshold. This may indicate "
                "discriminatory impact requiring immediate attention."
            )

        if fairness.demographic_parity_difference > 0.1:
            recommendations.append(
                "Selection rates differ significantly across groups. "
                "Consider reviewing job requirements for adverse impact."
            )

        if fairness.score_gap > 0.2:
            recommendations.append(
                f"Large score gap ({fairness.score_gap:.2f}) between groups. "
                "Review scoring criteria for potential bias sources."
            )

        # Group-specific recommendations
        if fairness.group_metrics:
            lowest_rate_group = min(
                fairness.group_metrics,
                key=lambda g: g.positive_rate,
            )
            highest_rate_group = max(
                fairness.group_metrics,
                key=lambda g: g.positive_rate,
            )

            if lowest_rate_group.positive_rate < highest_rate_group.positive_rate * 0.5:
                recommendations.append(
                    f"Group '{lowest_rate_group.group_name}' has significantly "
                    f"lower selection rate ({lowest_rate_group.positive_rate:.1%}) "
                    f"than '{highest_rate_group.group_name}' ({highest_rate_group.positive_rate:.1%}). "
                    "Investigate potential causes."
                )

        if not recommendations:
            recommendations.append("No significant fairness issues detected.")

        return recommendations

    def _calculate_bias_confidence(self, analysis: BiasAnalysisResult) -> float:
        """Calculate confidence score for bias detection."""
        if not analysis.attribute_detection:
            return 0.0

        if not analysis.attribute_detection.detected_attributes:
            return 0.0

        # Average confidence of detected attributes
        confidences = [
            attr.confidence
            for attr in analysis.attribute_detection.detected_attributes
        ]
        return sum(confidences) / len(confidences)

    def _generate_bias_description(self, analysis: BiasAnalysisResult) -> Optional[str]:
        """Generate human-readable bias description."""
        if not analysis.bias_detected:
            return None

        types_str = ", ".join(analysis.bias_types)
        return (
            f"Protected attributes detected: {types_str}. "
            f"Risk level: {analysis.risk_level}. "
            f"Review recommended to ensure fair evaluation."
        )


# Singleton instance
_detector: Optional[BiasDetector] = None


def get_bias_detector(auto_mitigate: bool = False) -> BiasDetector:
    """Get the bias detector singleton instance."""
    global _detector
    if _detector is None:
        _detector = BiasDetector(auto_mitigate=auto_mitigate)
    return _detector
