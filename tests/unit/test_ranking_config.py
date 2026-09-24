import os
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

import pytest
from src.core.ranking.ranking_config import RankingConfig, RankedResult, VALID_TIEBREAKERS
from src.core.matching.matching_engine import MatchResult


def test_default_weights_is_none() -> None:
    cfg = RankingConfig()
    assert cfg.weights is None


def test_default_tiebreaker_is_skills_score() -> None:
    cfg = RankingConfig()
    assert cfg.tiebreaker == "skills_score"


def test_default_fairness_mode_is_flag() -> None:
    cfg = RankingConfig()
    assert cfg.fairness_mode == "flag"


def test_none_tiebreaker_is_valid() -> None:
    cfg = RankingConfig(tiebreaker=None)
    assert cfg.tiebreaker is None


def test_invalid_tiebreaker_raises_value_error() -> None:
    with pytest.raises(ValueError, match="tiebreaker"):
        RankingConfig(tiebreaker="nonsense_score")


def test_invalid_fairness_mode_raises_value_error() -> None:
    with pytest.raises(ValueError, match="fairness_mode"):
        RankingConfig(fairness_mode="wrong")


def test_resolved_weights_none_returns_defaults() -> None:
    from src.utils.constants import DEFAULT_SCORING_WEIGHTS
    cfg = RankingConfig()
    resolved = cfg.resolved_weights
    assert resolved == dict(DEFAULT_SCORING_WEIGHTS)


def test_resolved_weights_full_dict_normalizes_to_one() -> None:
    cfg = RankingConfig(weights={
        "skills_match": 0.7,
        "experience_match": 0.3,
        "education_match": 0.0,
        "semantic_similarity": 0.0,
        "keyword_match": 0.0,
    })
    resolved = cfg.resolved_weights
    assert abs(sum(resolved.values()) - 1.0) < 1e-9
    assert abs(resolved["skills_match"] - 0.7) < 1e-9
    assert abs(resolved["experience_match"] - 0.3) < 1e-9


def test_resolved_weights_partial_dict_fills_missing_keys() -> None:
    cfg = RankingConfig(weights={"skills_match": 1.0})
    resolved = cfg.resolved_weights
    assert abs(sum(resolved.values()) - 1.0) < 1e-9
    assert "experience_match" in resolved
    assert "education_match" in resolved
    assert "semantic_similarity" in resolved
    assert "keyword_match" in resolved


def test_ranked_result_defaults() -> None:
    mr = MatchResult(candidate_name="Alice", overall_score=0.8)
    rr = RankedResult(match_result=mr, rank=1, effective_score=0.8)
    assert rr.fairness_flags == []
    assert rr.reranked is False


def test_negative_rerank_tolerance_raises() -> None:
    with pytest.raises(ValueError, match="rerank_tolerance"):
        RankingConfig(rerank_tolerance=-0.1)


def test_zero_fairness_min_group_size_raises() -> None:
    with pytest.raises(ValueError, match="fairness_min_group_size"):
        RankingConfig(fairness_min_group_size=0)


def test_negative_fairness_min_group_size_raises() -> None:
    with pytest.raises(ValueError, match="fairness_min_group_size"):
        RankingConfig(fairness_min_group_size=-1)


def test_unknown_weight_keys_raises() -> None:
    with pytest.raises(ValueError, match="Unknown weight keys"):
        RankingConfig(weights={"skill_match": 1.0})  # typo: missing 's'
