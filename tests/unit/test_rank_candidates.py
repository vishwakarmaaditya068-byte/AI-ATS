import os
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

import pytest
from src.core.matching.matching_engine import MatchResult
from src.core.ranking import rank_candidates, RankingConfig, RankedResult
from src.data.models import BiasCheckResult


def _mr(
    overall_score: float = 0.5,
    skills_score: float = 0.5,
    experience_score: float = 0.5,
    education_score: float = 0.5,
    semantic_score: float = 0.5,
    keyword_score: float = 0.5,
    protected_attrs: list[str] | None = None,
    name: str = "Candidate",
) -> MatchResult:
    return MatchResult(
        candidate_name=name,
        overall_score=overall_score,
        skills_score=skills_score,
        experience_score=experience_score,
        education_score=education_score,
        semantic_score=semantic_score,
        keyword_score=keyword_score,
        bias_check=BiasCheckResult(
            protected_attributes_found=protected_attrs or []
        ),
    )


def test_returns_list_of_ranked_results() -> None:
    results = [_mr(0.8), _mr(0.6)]
    ranked = rank_candidates(results)
    assert all(isinstance(r, RankedResult) for r in ranked)


def test_default_config_sorts_descending() -> None:
    results = [_mr(0.4), _mr(0.9), _mr(0.6)]
    ranked = rank_candidates(results)
    scores = [r.effective_score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_empty_input_returns_empty() -> None:
    assert rank_candidates([]) == []


def test_none_config_uses_defaults() -> None:
    results = [_mr(0.8), _mr(0.6)]
    ranked = rank_candidates(results, config=None)
    assert ranked[0].effective_score >= ranked[1].effective_score


def test_weight_override_applied_end_to_end() -> None:
    r1 = _mr(overall_score=0.5, skills_score=1.0, experience_score=0.0,
             education_score=0.0, semantic_score=0.0, keyword_score=0.0, name="SkillsFirst")
    r2 = _mr(overall_score=0.5, skills_score=0.0, experience_score=1.0,
             education_score=0.0, semantic_score=0.0, keyword_score=0.0, name="ExpFirst")
    config = RankingConfig(
        weights={
            "skills_match": 0.05,
            "experience_match": 0.95,
            "education_match": 0.0,
            "semantic_similarity": 0.0,
            "keyword_match": 0.0,
        },
        tiebreaker=None,
        fairness_mode="off",
    )
    ranked = rank_candidates([r1, r2], config)
    assert ranked[0].match_result.candidate_name == "ExpFirst"


def test_fairness_off_mode_no_flags() -> None:
    results = [_mr(0.9), _mr(0.7)]
    config = RankingConfig(fairness_mode="off")
    ranked = rank_candidates(results, config)
    assert all(r.fairness_flags == [] for r in ranked)


def test_rank_integers_are_one_based() -> None:
    results = [_mr(0.9), _mr(0.7), _mr(0.5)]
    ranked = rank_candidates(results, RankingConfig(fairness_mode="off"))
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
    assert ranked[2].rank == 3
