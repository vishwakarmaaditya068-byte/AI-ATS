"""
FairnessReranker — applies fairness checking to a ranked candidate list.

Two modes (controlled by RankingConfig.fairness_mode):
  "flag"   — rank order unchanged; FairnessMetrics violations and warnings
             are attached to every RankedResult.fairness_flags.
  "rerank" — candidates within rerank_tolerance of the top score are
             diversity-reranked (MMR-style) to reduce group under-representation;
             positions outside the tolerance band are unchanged.

Uses the existing FairnessCalculator from src.ml.ethics.fairness_metrics.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from src.core.ranking.ranking_config import RankingConfig, RankedResult
from src.ml.ethics.fairness_metrics import FairnessCalculator, FairnessMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FairnessReranker:
    """Attaches fairness flags and optionally diversity-reranks a ranked list."""

    def __init__(self, config: RankingConfig) -> None:
        self._config: RankingConfig = config

    def apply(self, ranked: list[RankedResult]) -> list[RankedResult]:
        """
        Apply fairness processing to a ranked list.

        Args:
            ranked: Output of CandidateRanker.rank(), sorted descending.

        Returns:
            list[RankedResult] with fairness_flags populated.
            In "rerank" mode, order and ranks may change for in-band candidates.
            In "off" mode, returned unchanged.
        """
        if not ranked or self._config.fairness_mode == "off":
            return ranked

        scores: list[float] = [rr.effective_score for rr in ranked]
        group_labels: list[str] = [self._get_group(rr) for rr in ranked]

        calculator: FairnessCalculator = FairnessCalculator(
            min_group_size=self._config.fairness_min_group_size
        )
        metrics: FairnessMetrics = calculator.calculate(scores, group_labels)

        # violations — purely statistical messages (e.g. "Demographic parity difference
        # (0.12) exceeds threshold (0.1)"). Safe to surface in the UI.
        #
        # warnings — operational notes that may contain raw demographic group labels
        # (e.g. "Group 'gender_female' has only 3 sample(s)..."). These are logged
        # internally but NOT attached to fairness_flags to avoid leaking demographic
        # identifiers through the UI layer.
        if metrics.warnings:
            for msg in metrics.warnings:
                logger.debug("FairnessReranker [internal]: %s", msg)

        ui_flags: list[str] = list(metrics.violations)
        # Compare unique group count against eligible groups that produced metrics.
        # The >1 guard prevents the flag firing on single-group inputs where
        # FairnessCalculator returns an empty group_metrics list by design.
        _unique_group_count: int = len(set(group_labels))
        if _unique_group_count > 1 and _unique_group_count > len(metrics.group_metrics):
            ui_flags.append(
                f"Some groups excluded from fairness analysis "
                f"(minimum {self._config.fairness_min_group_size} candidates required per group)."
            )
        for rr in ranked:
            rr.fairness_flags.clear()
            rr.fairness_flags.extend(ui_flags)

        if self._config.fairness_mode == "flag":
            return ranked

        return self._diversity_rerank(ranked)

    def _diversity_rerank(self, ranked: list[RankedResult]) -> list[RankedResult]:
        """
        MMR-style diversity reranking within the tolerance band.

        Steps:
        1. Split ranked into in_band (score >= top - tolerance) and out_of_band.
        2. Greedily select from in_band, preferring under-represented groups.
        3. Concatenate reordered_band + out_of_band.
        4. Reassign rank = 1-based position.
        5. Set reranked=True for any candidate whose position changed.
        """
        if not ranked:
            return ranked

        top_score: float = ranked[0].effective_score
        threshold: float = top_score - self._config.rerank_tolerance

        in_band: list[RankedResult] = [
            rr for rr in ranked if rr.effective_score >= threshold
        ]
        out_of_band: list[RankedResult] = [
            rr for rr in ranked if rr.effective_score < threshold
        ]

        # With tolerance=0.0, float-exact equality means in_band may contain
        # only the top candidate — nothing to reorder, return unchanged.
        if len(in_band) <= 1:
            return ranked

        original_positions: dict[int, int] = {
            id(rr): i for i, rr in enumerate(ranked)
        }

        # Greedy selection from in_band preferring under-represented groups
        group_total: Counter[str] = Counter(self._get_group(rr) for rr in in_band)
        group_selected: defaultdict[str, int] = defaultdict(int)

        reordered_band: list[RankedResult] = []
        remaining: list[RankedResult] = list(in_band)

        while remaining:
            best_idx: int = self._pick_next(
                remaining, group_total, group_selected
            )
            best: RankedResult = remaining[best_idx]
            reordered_band.append(best)
            group_selected[self._get_group(best)] += 1
            # O(1) removal: overwrite selected slot with last item, then pop.
            # Order of remaining does not matter — _pick_next scans all entries anyway.
            remaining[best_idx] = remaining[-1]
            remaining.pop()

        # Concatenate and reassign ranks / reranked flag
        final: list[RankedResult] = reordered_band + out_of_band
        for new_pos, rr in enumerate(final):
            original_pos: int = original_positions[id(rr)]
            if new_pos != original_pos:
                rr.reranked = True
            rr.rank = new_pos + 1

        return final

    @staticmethod
    def _pick_next(
        remaining: list[RankedResult],
        group_total: Counter[str],
        group_selected: defaultdict[str, int],
    ) -> int:
        """
        Return the index within remaining of the next best candidate.

        Prefers the group with the lowest current selection ratio.
        Ties broken by effective_score descending.
        """
        best_idx: int = 0
        seed_group: str = FairnessReranker._get_group(remaining[0])
        best_ratio: float = group_selected[seed_group] / group_total[seed_group]
        best_score: float = remaining[0].effective_score

        for i in range(1, len(remaining)):
            rr = remaining[i]
            group: str = FairnessReranker._get_group(rr)
            ratio: float = group_selected[group] / group_total[group]
            if ratio < best_ratio or (
                ratio == best_ratio and rr.effective_score > best_score
            ):
                best_idx = i
                best_ratio = ratio
                best_score = rr.effective_score

        return best_idx

    @staticmethod
    def _get_group(rr: RankedResult) -> str:
        """
        Extract a stable composite group label from a RankedResult.

        Joins all protected_attributes_found sorted alphabetically so that
        intersectional candidates ("ethnicity_asian|gender_female") form their
        own group and attribute order in the source data is irrelevant.
        Falls back to "unknown" if no attributes are present.
        """
        bc = rr.match_result.bias_check
        if bc and bc.protected_attributes_found:
            return "|".join(sorted(bc.protected_attributes_found))
        return "unknown"
