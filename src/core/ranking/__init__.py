"""Scoring and ranking algorithms module."""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.ranking.fairness_reranker import FairnessReranker
from src.core.ranking.ranker import CandidateRanker
from src.core.ranking.ranking_config import RankedResult, RankingConfig

if TYPE_CHECKING:
    from src.core.matching.matching_engine import MatchResult


def rank_candidates(
    results: list[MatchResult],
    config: RankingConfig | None = None,
) -> list[RankedResult]:
    """
    Rank a list of MatchResult objects.

    Args:
        results: Output from MatchingEngine.match() or match_from_parsed().
        config:  Per-session ranking configuration. Defaults to
                 RankingConfig() — no weight override, skills_score
                 tiebreaker, flag fairness mode.

    Returns:
        list[RankedResult] sorted descending by effective_score,
        with ranks assigned and fairness_flags populated.
    """
    cfg: RankingConfig = config or RankingConfig()
    ranked: list[RankedResult] = CandidateRanker(cfg).rank(results)
    ranked = FairnessReranker(cfg).apply(ranked)
    return ranked


__all__ = ["rank_candidates", "RankingConfig", "RankedResult"]
