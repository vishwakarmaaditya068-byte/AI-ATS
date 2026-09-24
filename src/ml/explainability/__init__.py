"""
Model explainability with LIME and SHAP.

This module provides comprehensive explainability capabilities for the
AI-ATS matching system, enabling transparent and interpretable decisions.

Components:
- FeatureImportanceCalculator: Calculates feature contributions to scores
- LIMEExplainer: Local Interpretable Model-agnostic Explanations
- SHAPExplainer: Shapley Additive Explanations
- MatchExplainer: Main orchestrator for comprehensive explanations
"""

from .feature_importance import (
    FeatureImportanceCalculator,
    FeatureContribution,
    FeatureImportanceResult,
    get_feature_importance_calculator,
)

from .lime_explainer import (
    LIMEExplainer,
    LIMEExplanation,
    LIMEFeatureWeight,
    get_lime_explainer,
)

from .shap_explainer import (
    SHAPExplainer,
    SHAPExplanation,
    SHAPValue,
    get_shap_explainer,
)

from .explainer import (
    MatchExplainer,
    MatchExplanation,
    get_match_explainer,
)

__all__ = [
    # Feature Importance
    "FeatureImportanceCalculator",
    "FeatureContribution",
    "FeatureImportanceResult",
    "get_feature_importance_calculator",
    # LIME
    "LIMEExplainer",
    "LIMEExplanation",
    "LIMEFeatureWeight",
    "get_lime_explainer",
    # SHAP
    "SHAPExplainer",
    "SHAPExplanation",
    "SHAPValue",
    "get_shap_explainer",
    # Main Explainer
    "MatchExplainer",
    "MatchExplanation",
    "get_match_explainer",
]
