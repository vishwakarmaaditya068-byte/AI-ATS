import os
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

import pytest
from src.core.matching.matching_engine import MatchResult
from src.core.ranking.ranking_config import RankingConfig, RankedResult
from src.core.ranking.ranker import CandidateRanker


def _mr(
    overall_score: float = 0.5,
    skills_score: float = 0.5,
    experience_score: float = 0.5,
    education_score: float = 0.5,
    semantic_score: float = 0.5,
    keyword_score: float = 0.5,
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
    )


def test_empty_input_returns_empty() -> None:
    ranked = CandidateRanker(RankingConfig()).rank([])
    assert ranked == []


def test_single_result_gets_rank_one() -> None:
    ranked = CandidateRanker(RankingConfig()).rank([_mr(overall_score=0.8)])
    assert len(ranked) == 1
    assert ranked[0].rank == 1
    assert ranked[0].effective_score == pytest.approx(0.8)


def test_results_sorted_descending_by_overall_score() -> None:
    results = [_mr(overall_score=0.5), _mr(overall_score=0.9), _mr(overall_score=0.7)]
    ranked = CandidateRanker(RankingConfig(tiebreaker=None)).rank(results)
    scores = [r.effective_score for r in ranked]
    assert scores == pytest.approx([0.9, 0.7, 0.5])
    assert [r.rank for r in ranked] == [1, 2, 3]


def test_tied_scores_share_rank_non_dense() -> None:
    # Scores [0.9, 0.9, 0.7] → ranks [1, 1, 3] (non-dense, no tiebreaker)
    results = [_mr(overall_score=0.9), _mr(overall_score=0.9), _mr(overall_score=0.7)]
    ranked = CandidateRanker(RankingConfig(tiebreaker=None)).rank(results)
    assert [r.rank for r in ranked] == [1, 1, 3]


def test_tiebreaker_differentiates_equal_overall_scores() -> None:
    # Both overall=0.8; r1 has higher skills_score → r1 ranks first
    r1 = _mr(overall_score=0.8, skills_score=0.9, name="Alice")
    r2 = _mr(overall_score=0.8, skills_score=0.6, name="Bob")
    ranked = CandidateRanker(RankingConfig(tiebreaker="skills_score")).rank([r1, r2])
    assert ranked[0].match_result is r1
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2


def test_tiebreaker_none_gives_shared_rank_on_equal_overall() -> None:
    r1 = _mr(overall_score=0.8, skills_score=0.9, name="Alice")
    r2 = _mr(overall_score=0.8, skills_score=0.6, name="Bob")
    ranked = CandidateRanker(RankingConfig(tiebreaker=None)).rank([r1, r2])
    assert ranked[0].rank == 1
    assert ranked[1].rank == 1


def test_weight_override_re_scores_and_changes_order() -> None:
    # r1: skills=1.0, experience=0.0
    # r2: skills=0.0, experience=1.0
    # With experience-heavy weights r2 should rank first
    r1 = _mr(
        overall_score=0.5,
        skills_score=1.0, experience_score=0.0,
        education_score=0.0, semantic_score=0.0, keyword_score=0.0,
        name="SkillsHeavy",
    )
    r2 = _mr(
        overall_score=0.5,
        skills_score=0.0, experience_score=1.0,
        education_score=0.0, semantic_score=0.0, keyword_score=0.0,
        name="ExpHeavy",
    )
    config = RankingConfig(
        weights={
            "skills_match": 0.1,
            "experience_match": 0.9,
            "education_match": 0.0,
            "semantic_similarity": 0.0,
            "keyword_match": 0.0,
        },
        tiebreaker=None,
    )
    ranked = CandidateRanker(config).rank([r1, r2])
    assert ranked[0].match_result is r2
    assert ranked[0].effective_score > ranked[1].effective_score


def test_weight_override_uses_component_scores_not_overall() -> None:
    # overall_score is 0.9 for both but component scores differ
    r1 = _mr(overall_score=0.9, skills_score=0.2, experience_score=0.8,
             education_score=0.0, semantic_score=0.0, keyword_score=0.0)
    r2 = _mr(overall_score=0.9, skills_score=0.8, experience_score=0.2,
             education_score=0.0, semantic_score=0.0, keyword_score=0.0)
    config = RankingConfig(
        weights={
            "skills_match": 1.0,
            "experience_match": 0.0,
            "education_match": 0.0,
            "semantic_similarity": 0.0,
            "keyword_match": 0.0,
        },
        tiebreaker=None,
    )
    ranked = CandidateRanker(config).rank([r1, r2])
    assert ranked[0].match_result is r2  # r2 has higher skills_score


def test_ranked_results_have_empty_fairness_flags() -> None:
    ranked = CandidateRanker(RankingConfig()).rank([_mr()])
    assert ranked[0].fairness_flags == []


def test_ranked_results_reranked_is_false() -> None:
    # CandidateRanker never sets reranked=True; that is FairnessReranker's job
    ranked = CandidateRanker(RankingConfig()).rank([_mr(), _mr(overall_score=0.7)])
    assert all(r.reranked is False for r in ranked)


def test_four_item_non_dense_ranking_with_two_tie_groups() -> None:
    # Plan example: [0.90, 0.90, 0.75, 0.60] → ranks [1, 1, 3, 4]
    results = [
        _mr(overall_score=0.90),
        _mr(overall_score=0.90),
        _mr(overall_score=0.75),
        _mr(overall_score=0.60),
    ]
    ranked = CandidateRanker(RankingConfig(tiebreaker=None)).rank(results)
    assert [r.rank for r in ranked] == [1, 1, 3, 4]
