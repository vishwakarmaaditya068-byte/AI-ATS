"""
RankingConfig and RankedResult.

RankingConfig: per-session parameters controlling weight overrides,
               tie-breaking, and fairness mode.
RankedResult:  MatchResult wrapper with rank, effective_score,
               fairness flags, and rerank marker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from src.utils.constants import DEFAULT_SCORING_WEIGHTS, FAIRNESS_MIN_GROUP_SIZE

if TYPE_CHECKING:
    from src.core.matching.matching_engine import MatchResult

VALID_TIEBREAKERS: frozenset[str] = frozenset({
    "skills_score",
    "experience_score",
    "education_score",
    "semantic_score",
    "keyword_score",
})

VALID_FAIRNESS_MODES: frozenset[str] = frozenset({"off", "flag", "rerank"})


@dataclass
class RankingConfig:
    """
    Per-session configuration for the ranking pipeline.

    weights:
        Keys match DEFAULT_SCORING_WEIGHTS:
            "skills_match", "experience_match", "education_match",
            "semantic_similarity", "keyword_match"
        Partial dicts accepted — missing keys filled from
        DEFAULT_SCORING_WEIGHTS, then normalized to sum to 1.0.
        If None, overall_score from MatchResult is used as-is.

    tiebreaker:
        MatchResult attribute used as secondary sort key.
        Must be one of VALID_TIEBREAKERS, or None for no tie-breaking.

    fairness_mode:
        "off"    — no fairness processing
        "flag"   — rank unchanged, attach FairnessMetrics warnings
        "rerank" — diversity re-rank within rerank_tolerance band

    rerank_tolerance:
        Score delta below the top candidate within which candidates
        are eligible for diversity reordering.

    fairness_min_group_size:
        Mirrors FairnessCalculator — groups smaller than this are skipped.
    """

    weights: Optional[dict[str, float]] = None
    tiebreaker: Optional[str] = "skills_score"
    fairness_mode: str = "flag"
    rerank_tolerance: float = 0.05
    fairness_min_group_size: int = FAIRNESS_MIN_GROUP_SIZE

    def __post_init__(self) -> None:
        if self.tiebreaker is not None and self.tiebreaker not in VALID_TIEBREAKERS:
            raise ValueError(
                f"Invalid tiebreaker {self.tiebreaker!r}. "
                f"Must be one of {sorted(VALID_TIEBREAKERS)} or None."
            )
        if self.fairness_mode not in VALID_FAIRNESS_MODES:
            raise ValueError(
                f"Invalid fairness_mode {self.fairness_mode!r}. "
                f"Must be one of {sorted(VALID_FAIRNESS_MODES)}."
            )
        if self.rerank_tolerance < 0.0:
            raise ValueError(
                f"rerank_tolerance must be >= 0.0, got {self.rerank_tolerance!r}."
            )
        if self.fairness_min_group_size < 1:
            raise ValueError(
                f"fairness_min_group_size must be >= 1, got {self.fairness_min_group_size!r}."
            )
        if self.weights is not None:
            unknown: set[str] = set(self.weights) - set(DEFAULT_SCORING_WEIGHTS)
            if unknown:
                raise ValueError(
                    f"Unknown weight keys: {sorted(unknown)}. "
                    f"Valid keys: {sorted(DEFAULT_SCORING_WEIGHTS)}."
                )

    @property
    def resolved_weights(self) -> dict[str, float]:
        """
        Return normalized weights merged with DEFAULT_SCORING_WEIGHTS.

        Steps:
        1. Start from DEFAULT_SCORING_WEIGHTS.
        2. Override with any keys from self.weights.
        3. Normalize so all values sum to 1.0.
        """
        if self.weights is None:
            return dict(DEFAULT_SCORING_WEIGHTS)
        merged: dict[str, float] = dict(DEFAULT_SCORING_WEIGHTS)
        merged.update(self.weights)
        total: float = sum(merged.values())
        if total <= 0.0:
            return dict(DEFAULT_SCORING_WEIGHTS)
        return {k: v / total for k, v in merged.items()}


@dataclass
class RankedResult:
    """MatchResult with ranking and fairness annotations."""

    match_result: MatchResult
    rank: int
    effective_score: float
    fairness_flags: list[str] = field(default_factory=list)
    reranked: bool = False
