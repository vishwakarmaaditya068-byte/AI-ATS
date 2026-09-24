"""
Ethical AI: bias detection and fairness metrics.

This module provides comprehensive bias detection and mitigation
capabilities for the AI-ATS system, ensuring fair and ethical
candidate evaluation.

Components:
- ProtectedAttributeDetector: Detects protected attributes in text
- FairnessCalculator: Calculates fairness metrics across groups
- BiasMitigator: Implements bias mitigation strategies
- BiasDetector: Main orchestrator for bias analysis
"""

from .protected_attributes import (
    ProtectedAttributeDetector,
    DetectedAttribute,
    AttributeDetectionResult,
    get_attribute_detector,
)

from .fairness_metrics import (
    FairnessCalculator,
    FairnessMetrics,
    GroupMetrics,
    get_fairness_calculator,
)

from .bias_mitigation import (
    BiasMitigator,
    MitigationStrategy,
    MitigationResult,
    RedactionResult,
    get_bias_mitigator,
)

from .bias_detector import (
    BiasDetector,
    BiasAnalysisResult,
    get_bias_detector,
)

__all__ = [
    # Protected Attributes
    "ProtectedAttributeDetector",
    "DetectedAttribute",
    "AttributeDetectionResult",
    "get_attribute_detector",
    # Fairness Metrics
    "FairnessCalculator",
    "FairnessMetrics",
    "GroupMetrics",
    "get_fairness_calculator",
    # Bias Mitigation
    "BiasMitigator",
    "MitigationStrategy",
    "MitigationResult",
    "RedactionResult",
    "get_bias_mitigator",
    # Main Detector
    "BiasDetector",
    "BiasAnalysisResult",
    "get_bias_detector",
]
