"""
CandidateRanker — sorts MatchResult objects and assigns ranks.

Algorithm:
1. Compute effective_score per result (re-score if weights overridden).
2. Sort descending by (effective_score, tiebreaker_score).
3. Assign non-dense ranks: tied candidates share a rank;
   the next distinct score group skips positions.
   Example: effective_scores [0.90, 0.90, 0.75, 0.60] → ranks [1, 1, 3, 4].
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.core.ranking.ranking_config import RankingConfig, RankedResult

if TYPE_CHECKING:
    from src.core.matching.matching_engine import MatchResult


class CandidateRanker:
    """Sorts and ranks a list of MatchResult objects."""

    def __init__(self, config: RankingConfig) -> None:
        self._config: RankingConfig = config

    def rank(self, results: list[MatchResult]) -> list[RankedResult]:
        """
        Rank a list of MatchResult objects.

        Args:
            results: Output from MatchingEngine.match() or match_from_parsed().

        Returns:
            list[RankedResult] sorted descending by effective_score with
            non-dense rank integers assigned. fairness_flags is empty —
            FairnessReranker populates it in the next pipeline step.
        """
        if not results:
            return []

        weights: Optional[dict[str, float]] = (
            self._config.resolved_weights if self._config.weights is not None else None
        )
        tiebreaker: Optional[str] = self._config.tiebreaker

        # Build (effective_score, tiebreaker_score, result) tuples
        scored: list[tuple[float, float, MatchResult]] = []
        for result in results:
            effective: float = (
                self._compute_effective_score(result, weights)
                if weights is not None
                else result.overall_score
            )
            tb_score: float = (
                float(getattr(result, tiebreaker, 0.0))
                if tiebreaker is not None
                else 0.0
            )
            scored.append((effective, tb_score, result))

        # Sort descending by (effective_score, tiebreaker_score)
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

        # Assign non-dense ranks
        ranked: list[RankedResult] = []
        prev_effective: Optional[float] = None
        prev_tb: Optional[float] = None
        prev_rank: int = 1

        for i, (effective, tb_score, result) in enumerate(scored):
            position: int = i + 1
            rank: int = position
            if (
                prev_effective is not None
                and effective == prev_effective
                and tb_score == prev_tb
            ):
                rank = prev_rank

            # tiebreaker is validated by RankingConfig.__post_init__ against
            # VALID_TIEBREAKERS, all of which are attributes on MatchResult.
            # getattr fallback 0.0 is a safety net only.

            prev_effective = effective
            prev_tb = tb_score
            prev_rank = rank

            ranked.append(RankedResult(
                match_result=result,
                rank=rank,
                effective_score=round(effective, 6),
            ))

        return ranked

    @staticmethod
    def _compute_effective_score(
        result: MatchResult,
        weights: dict[str, float],
    ) -> float:
        """
        Re-compute score from component scores using provided weights.

        Component key → MatchResult attribute:
            "skills_match"        → skills_score
            "experience_match"    → experience_score
            "education_match"     → education_score
            "semantic_similarity" → semantic_score
            "keyword_match"       → keyword_score
        """
        score: float = (
            result.skills_score      * weights.get("skills_match", 0.0)
            + result.experience_score * weights.get("experience_match", 0.0)
            + result.education_score  * weights.get("education_match", 0.0)
            + result.semantic_score   * weights.get("semantic_similarity", 0.0)
            + result.keyword_score    * weights.get("keyword_match", 0.0)
        )
        return round(min(max(score, 0.0), 1.0), 6)
