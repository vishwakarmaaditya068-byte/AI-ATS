import os
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

import pytest
from unittest.mock import Mock

from src.ml.explainability.explainer import MatchExplainer, MatchExplanation
from src.ml.explainability.feature_importance import (
    FeatureImportanceCalculator,
    FeatureImportanceResult,
    FeatureContribution,
)
from src.ml.explainability.shap_explainer import SHAPExplainer, SHAPExplanation, SHAPValue


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_feature(
    name: str,
    raw_score: float,
    weight: float = 0.2,
    direction: str | None = None,
    description: str = "",
) -> FeatureContribution:
    weighted: float = raw_score * weight
    if direction is None:
        direction = "positive" if raw_score >= 0.7 else ("negative" if raw_score <= 0.3 else "neutral")
    return FeatureContribution(
        feature_name=name,
        feature_value=raw_score,
        raw_score=raw_score,
        weight=weight,
        weighted_contribution=round(weighted, 4),
        contribution_percentage=round(weighted / 1.0 * 100, 1),
        direction=direction,
        description=description or f"{name} score: {int(raw_score * 100)}%",
    )


def _make_importance(
    skills: float = 0.80,
    experience: float = 0.30,
    education: float = 0.60,
    semantic: float = 0.75,
    keyword: float = 0.40,
) -> FeatureImportanceResult:
    feats: list[FeatureContribution] = [
        _make_feature("Skills Match", skills, weight=0.35),
        _make_feature("Experience Match", experience, weight=0.25),
        _make_feature("Education Match", education, weight=0.15),
        _make_feature("Semantic Similarity", semantic, weight=0.20),
        _make_feature("Keyword Match", keyword, weight=0.05),
    ]
    feats.sort(key=lambda f: f.weighted_contribution, reverse=True)
    return FeatureImportanceResult(
        total_score=sum(f.weighted_contribution for f in feats),
        features=feats,
        top_positive_features=[f.feature_name for f in feats if f.direction == "positive"][:3],
        top_negative_features=[f.feature_name for f in feats if f.direction == "negative"][:3],
        feature_ranking=[f.feature_name for f in feats],
    )


def _make_shap_value(name: str, value: float, feature_value: float = 0.5) -> SHAPValue:
    return SHAPValue(
        feature_name=name,
        shap_value=value,
        base_value_contribution=0.1,
        feature_value=feature_value,
        direction="positive" if value > 0 else "negative",
    )


def _make_shap(
    positives: list[tuple[str, float]] | None = None,
    negatives: list[tuple[str, float]] | None = None,
    predicted_value: float = 0.65,
) -> SHAPExplanation:
    shap_vals: list[SHAPValue] = []
    for name, val in (positives or []):
        shap_vals.append(_make_shap_value(name, val))
    for name, val in (negatives or []):
        shap_vals.append(_make_shap_value(name, val))
    shap_vals.sort(key=lambda sv: abs(sv.shap_value), reverse=True)
    return SHAPExplanation(
        expected_value=0.5,
        predicted_value=predicted_value,
        shap_values=shap_vals,
        sum_shap_values=sum(sv.shap_value for sv in shap_vals),
    )


def _make_explainer() -> MatchExplainer:
    return MatchExplainer()


# ── _identify_strengths ───────────────────────────────────────────────────────

def test_identify_strengths_uses_shap_positive_contributors() -> None:
    """When SHAP is available, strengths are derived from SHAP positive contributors."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(skills=0.80, experience=0.30)
    shap: SHAPExplanation = _make_shap(
        positives=[("Skills Match", 0.18)],
        negatives=[("Experience Match", -0.09)],
    )

    strengths: list[str] = explainer._identify_strengths(importance, shap)

    assert any("Skills Match" in s for s in strengths)
    assert any("+0.18" in s or "0.18" in s for s in strengths), (
        "Strength should include the SHAP contribution value"
    )


def test_identify_strengths_excludes_negative_shap_contributors() -> None:
    """SHAP negative contributors must not appear as strengths."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(experience=0.25)
    shap: SHAPExplanation = _make_shap(
        negatives=[("Experience Match", -0.12)],
    )

    strengths: list[str] = explainer._identify_strengths(importance, shap)

    assert not any("Experience Match" in s for s in strengths)


def test_identify_strengths_falls_back_to_feature_importance_without_shap() -> None:
    """When SHAP is None, strengths come from feature importance direction."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(skills=0.85)

    strengths: list[str] = explainer._identify_strengths(importance, shap=None)

    assert any("Skills Match" in s for s in strengths)


# ── _identify_gaps ────────────────────────────────────────────────────────────

def test_identify_gaps_uses_shap_negative_contributors() -> None:
    """When SHAP is available, gaps are derived from SHAP negative contributors."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(experience=0.25)
    shap: SHAPExplanation = _make_shap(
        negatives=[("Experience Match", -0.11)],
    )

    gaps: list[str] = explainer._identify_gaps(importance, shap)

    assert any("Experience Match" in g for g in gaps)
    assert any("0.11" in g for g in gaps), "Gap should include the SHAP contribution value"


def test_identify_gaps_excludes_positive_shap_contributors() -> None:
    """SHAP positive contributors must not appear as gaps."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(skills=0.85)
    shap: SHAPExplanation = _make_shap(
        positives=[("Skills Match", 0.20)],
    )

    gaps: list[str] = explainer._identify_gaps(importance, shap)

    assert not any("Skills Match" in g for g in gaps)


def test_identify_gaps_falls_back_to_feature_importance_without_shap() -> None:
    """When SHAP is None, gaps come from feature importance direction."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(experience=0.20)

    gaps: list[str] = explainer._identify_gaps(importance, shap=None)

    assert any("Experience Match" in g for g in gaps)


# ── _generate_recommendations ─────────────────────────────────────────────────

def test_recommendations_include_score_percentage() -> None:
    """Each recommendation for a weak feature should state the actual score %."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(experience=0.28)

    recs: list[str] = explainer._generate_recommendations(importance, skill_details=None)

    assert any("28%" in r for r in recs), (
        "Recommendation for experience gap should include '28%'"
    )


def test_recommendations_name_specific_missing_skills() -> None:
    """Missing skills from skill_details should be named in the recommendation."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(skills=0.35)
    skill_details: dict = {"missing_skills": ["Python", "SQL", "Docker"]}

    recs: list[str] = explainer._generate_recommendations(importance, skill_details)

    all_text: str = " ".join(recs)
    assert "Python" in all_text or "SQL" in all_text, (
        "At least one missing skill should be named in recommendations"
    )


def test_recommendations_not_all_generic_templates() -> None:
    """Recommendations must not all be identical boilerplate strings."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(
        skills=0.30, experience=0.25, education=0.20
    )
    skill_details: dict = {"missing_skills": ["Python", "AWS"]}

    recs: list[str] = explainer._generate_recommendations(importance, skill_details)

    # All recs should differ from each other
    assert len(recs) == len(set(recs)), "Duplicate recommendations found"


# ── _generate_summary ─────────────────────────────────────────────────────────

def test_generate_summary_includes_top_feature_score() -> None:
    """Summary should show the score (%) of the strongest feature."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(skills=0.85)

    summary: str = explainer._generate_summary("Alice", "Engineer", 0.72, importance)

    # Top feature is Skills Match at 85%
    assert "85%" in summary or "Skills Match" in summary


def test_generate_summary_mentions_weak_area_when_present() -> None:
    """Summary should note the weakest area when a negative feature exists."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(experience=0.20)

    summary: str = explainer._generate_summary("Bob", "Manager", 0.55, importance)

    assert "Experience" in summary, (
        "Summary should mention the weak area when experience is negative"
    )


def test_generate_summary_no_weak_area_when_all_positive() -> None:
    """Summary should not mention a gap when all features are positive/neutral."""
    explainer: MatchExplainer = _make_explainer()
    importance: FeatureImportanceResult = _make_importance(
        skills=0.90, experience=0.80, education=0.75, semantic=0.85, keyword=0.70
    )

    summary: str = explainer._generate_summary("Carol", "Dev", 0.84, importance)

    # Should not contain any gap language
    assert "gap" not in summary.lower() and "missing" not in summary.lower()


# ── matching_engine integration ───────────────────────────────────────────────

def test_matching_engine_uses_explainer_enriched_outputs() -> None:
    """Explanation.strengths/gaps/recommendations should use explainer outputs when available."""
    from src.core.matching.matching_engine import MatchingEngine
    from src.ml.nlp.accurate_resume_parser import ParsedResume, SkillCategory, ExperienceEntry
    from src.data.models.job import Job, SkillRequirement

    engine: MatchingEngine = MatchingEngine(
        use_semantic=False, use_bias_detection=False, use_explainability=True
    )

    sentinel_strength: str = "SHAP-derived strength: Skills Match (+0.18 above baseline)"
    sentinel_gap: str = "SHAP-derived gap: Experience Match (-0.09 below baseline)"
    sentinel_rec: str = "SHAP recommendation: focus on Python skills (gap: 35%)"
    sentinel_summary: str = "SHAP summary: Alice is a good match (72%) for Engineer."

    mock_exp: Mock = Mock(spec=MatchExplanation)
    mock_exp.key_strengths = [sentinel_strength]
    mock_exp.key_gaps = [sentinel_gap]
    mock_exp.recommendations = [sentinel_rec]
    mock_exp.summary = sentinel_summary
    mock_exp.get_lime_dict.return_value = None
    mock_exp.get_shap_dict.return_value = {"Skills Match": 0.18}

    mock_explainer: Mock = Mock()
    mock_explainer.explain.return_value = mock_exp
    engine._explainer = mock_explainer

    parsed: ParsedResume = ParsedResume(
        summary="Python developer",
        skills=[SkillCategory(category="Tech", skills=["Python"])],
        experience=[ExperienceEntry(title="Dev", company="Acme", bullets=["Built APIs"])],
        raw_text="Python developer.",
    )
    job: Job = Job(
        title="Engineer",
        description="Build software.",
        responsibilities=["Write code"],
        company_name="Corp",
        skill_requirements=[SkillRequirement(name="Python", is_required=True)],
    )

    result = engine.match_from_parsed(parsed, job)

    assert result.explanation is not None
    assert sentinel_strength in result.explanation.strengths, (
        "matching_engine should use explainer.key_strengths, not its own basic strengths"
    )
    assert sentinel_gap in result.explanation.gaps, (
        "matching_engine should use explainer.key_gaps, not its own basic gaps"
    )
    assert sentinel_rec in result.explanation.recommendations, (
        "matching_engine should use explainer.recommendations"
    )


def test_matching_engine_passes_skill_details_to_explainer() -> None:
    """explainer.explain() should be called with skill_details containing missing_skills."""
    from src.core.matching.matching_engine import MatchingEngine
    from src.ml.nlp.accurate_resume_parser import ParsedResume, SkillCategory, ExperienceEntry
    from src.data.models.job import Job, SkillRequirement

    engine: MatchingEngine = MatchingEngine(
        use_semantic=False, use_bias_detection=False, use_explainability=True
    )

    mock_exp: Mock = Mock(spec=MatchExplanation)
    mock_exp.key_strengths = []
    mock_exp.key_gaps = []
    mock_exp.recommendations = []
    mock_exp.summary = "Test match"
    mock_exp.get_lime_dict.return_value = None
    mock_exp.get_shap_dict.return_value = {}

    mock_explainer: Mock = Mock()
    mock_explainer.explain.return_value = mock_exp
    engine._explainer = mock_explainer

    parsed: ParsedResume = ParsedResume(
        summary="Dev",
        skills=[SkillCategory(category="Tech", skills=["Python"])],
        experience=[ExperienceEntry(title="Dev", company="X", bullets=[])],
        raw_text="Dev",
    )
    job: Job = Job(
        title="Engineer",
        description="Build things.",
        responsibilities=[],
        company_name="Corp",
        skill_requirements=[
            SkillRequirement(name="Python", is_required=True),
            SkillRequirement(name="AWS", is_required=True),
        ],
    )

    engine.match_from_parsed(parsed, job)

    call_kwargs: dict = mock_explainer.explain.call_args.kwargs
    assert "skill_details" in call_kwargs, "explainer.explain() must receive skill_details"
    assert call_kwargs["skill_details"] is not None
