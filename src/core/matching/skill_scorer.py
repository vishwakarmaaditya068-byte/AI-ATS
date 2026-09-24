"""
Embedding-based skill scorer for candidate-job matching.

Replaces the hardcoded 8-group related-skill lookup in MatchingEngine
with a vector-space approach: each job skill is matched against every
candidate skill via cosine similarity, and the best match determines
the per-skill score.

Standalone — no circular project imports at module level.
Lazy-loads EmbeddingModel on first use (same pattern as SemanticMatcher).
Falls back to exact-string matching when the embedding model is unavailable.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.data.models import SkillMatch
from src.utils.constants import (
    SKILL_EXACT_THRESHOLD,
    SKILL_STRONG_PARTIAL_THRESHOLD,
    SKILL_WEAK_PARTIAL_THRESHOLD,
)

logger = logging.getLogger(__name__)


class EmbeddingSkillScorer:
    """
    Score candidate skills against JD requirements using embedding similarity.

    Encodes candidate skills in one batch call and all JD skills in a second
    batch call, then resolves per-skill matches via dot-product similarity.

    Thresholds (from constants.py):
        >= SKILL_EXACT_THRESHOLD          -> exact match     (score 1.0)
        >= SKILL_STRONG_PARTIAL_THRESHOLD -> strong partial  (score 0.8)
        >= SKILL_WEAK_PARTIAL_THRESHOLD   -> weak partial    (score 0.5)
        <  SKILL_WEAK_PARTIAL_THRESHOLD   -> no match        (score 0.0)

    Falls back to exact string matching when the embedding model is
    unavailable, preserving correctness at the cost of alias detection.
    """

    def __init__(self, embedding_model: Optional[object] = None) -> None:
        """
        Args:
            embedding_model: Optional pre-built EmbeddingModel instance.
                             If None, lazy-loaded on first call to score_skills().
        """
        self._model: Optional[object] = embedding_model
        self._model_load_failed: bool = False

    @property
    def model(self) -> Optional[object]:
        """Lazy-load EmbeddingModel on first access."""
        if self._model is None and not self._model_load_failed:
            try:
                from src.ml.embeddings import get_embedding_model
                self._model = get_embedding_model()
                logger.info("EmbeddingSkillScorer: model loaded successfully")
            except Exception as exc:
                logger.warning(f"EmbeddingSkillScorer: model unavailable — {exc}")
                self._model_load_failed = True
        return self._model

    def score_skills(
        self,
        required_skills: list[str],
        preferred_skills: list[str],
        candidate_skills: list[str],
    ) -> tuple[list[SkillMatch], float]:
        """
        Score candidate skills against required and preferred job skills.

        Args:
            required_skills:  Skills the job requires (is_required=True).
            preferred_skills: Skills the job prefers  (is_required=False).
            candidate_skills: All skills the candidate has (from ParsedResume).

        Returns:
            (skill_match_list, overall_score_in_[0,1])
        """
        if self.model is not None:
            return self._score_with_embeddings(
                required_skills, preferred_skills, candidate_skills
            )
        return self._score_exact_fallback(
            required_skills, preferred_skills, candidate_skills
        )

    # ------------------------------------------------------------------
    # Embedding path
    # ------------------------------------------------------------------

    def _score_with_embeddings(
        self,
        required_skills: list[str],
        preferred_skills: list[str],
        candidate_skills: list[str],
    ) -> tuple[list[SkillMatch], float]:
        """Core embedding-based scoring path. Uses two batch encode calls total."""
        if not candidate_skills:
            skill_matches: list[SkillMatch] = [
                SkillMatch(skill_name=s, required=True,  candidate_has_skill=False, match_score=0.0)
                for s in required_skills
            ] + [
                SkillMatch(skill_name=s, required=False, candidate_has_skill=False, match_score=0.0)
                for s in preferred_skills
            ]
            return skill_matches, 0.0

        # Batch 1: encode all candidate skills
        candidate_lower: list[str] = [s.lower() for s in candidate_skills]
        candidate_embs: np.ndarray = self.model.encode(candidate_lower, normalize=True)  # type: ignore[union-attr]
        if candidate_embs.ndim == 1:
            candidate_embs = candidate_embs.reshape(1, -1)

        # Batch 2: encode all JD skills (required + preferred) in one call
        all_jd_skills: list[str] = required_skills + preferred_skills
        all_jd_lower: list[str] = [s.lower() for s in all_jd_skills]
        # Only encode JD skills that are NOT exact-string matches (optimization: skip exact matches)
        # But for simplicity and correctness, encode all — EmbeddingModel has per-string LRU cache
        jd_embs: np.ndarray = self.model.encode(all_jd_lower, normalize=True)  # type: ignore[union-attr]
        if jd_embs.ndim == 1:
            jd_embs = jd_embs.reshape(1, -1)

        skill_matches = []
        required_score_sum: float = 0.0
        preferred_score_sum: float = 0.0

        for i, skill in enumerate(required_skills):
            match: SkillMatch = self._match_single_skill(
                skill, required=True,
                candidate_lower=candidate_lower,
                candidate_embs=candidate_embs,
                jd_skill_emb=jd_embs[i],
            )
            required_score_sum += match.match_score
            skill_matches.append(match)

        offset: int = len(required_skills)
        for i, skill in enumerate(preferred_skills):
            match = self._match_single_skill(
                skill, required=False,
                candidate_lower=candidate_lower,
                candidate_embs=candidate_embs,
                jd_skill_emb=jd_embs[offset + i],
            )
            preferred_score_sum += match.match_score
            skill_matches.append(match)

        score: float = self._weighted_score(
            required_score_sum=required_score_sum,
            preferred_score_sum=preferred_score_sum,
            n_required=len(required_skills),
            n_preferred=len(preferred_skills),
            has_candidate_skills=bool(candidate_skills),
        )
        return skill_matches, score

    def _match_single_skill(
        self,
        skill: str,
        required: bool,
        candidate_lower: list[str],
        candidate_embs: np.ndarray,
        jd_skill_emb: np.ndarray,
    ) -> SkillMatch:
        """Find the best match for one JD skill using a pre-encoded JD skill vector."""
        skill_lower: str = skill.lower()

        # Fast-path: exact string match — no cosine computation needed
        if skill_lower in candidate_lower:
            return SkillMatch(
                skill_name=skill,
                required=required,
                candidate_has_skill=True,
                match_score=1.0,
            )

        # Use pre-encoded JD skill embedding, flatten to (dim,) for safety
        skill_emb: np.ndarray = jd_skill_emb.flatten()
        # candidate_embs: (n, dim), skill_emb: (dim,) -> sims: (n,)
        sims: np.ndarray = candidate_embs @ skill_emb
        best_idx: int = int(np.argmax(sims))
        best_sim: float = float(sims[best_idx])
        best_candidate_skill: str = candidate_lower[best_idx]

        if best_sim >= SKILL_EXACT_THRESHOLD:
            return SkillMatch(
                skill_name=skill, required=required,
                candidate_has_skill=True, match_score=1.0,
                related_skill=best_candidate_skill,
            )
        elif best_sim >= SKILL_STRONG_PARTIAL_THRESHOLD:
            return SkillMatch(
                skill_name=skill, required=required,
                candidate_has_skill=False, partial_match=True,
                related_skill=best_candidate_skill, match_score=0.8,
            )
        elif best_sim >= SKILL_WEAK_PARTIAL_THRESHOLD:
            return SkillMatch(
                skill_name=skill, required=required,
                candidate_has_skill=False, partial_match=True,
                related_skill=best_candidate_skill, match_score=0.5,
            )
        else:
            return SkillMatch(
                skill_name=skill, required=required,
                candidate_has_skill=False, match_score=0.0,
            )

    # ------------------------------------------------------------------
    # Fallback path
    # ------------------------------------------------------------------

    def _score_exact_fallback(
        self,
        required_skills: list[str],
        preferred_skills: list[str],
        candidate_skills: list[str],
    ) -> tuple[list[SkillMatch], float]:
        """Exact string match fallback when the embedding model is unavailable."""
        candidate_lower_set: set[str] = {s.lower() for s in candidate_skills}
        skill_matches: list[SkillMatch] = []
        required_matched: float = 0.0
        preferred_matched: float = 0.0

        for skill in required_skills:
            has_skill: bool = skill.lower() in candidate_lower_set
            skill_matches.append(SkillMatch(
                skill_name=skill, required=True,
                candidate_has_skill=has_skill,
                match_score=1.0 if has_skill else 0.0,
            ))
            if has_skill:
                required_matched += 1.0

        for skill in preferred_skills:
            has_skill = skill.lower() in candidate_lower_set
            skill_matches.append(SkillMatch(
                skill_name=skill, required=False,
                candidate_has_skill=has_skill,
                match_score=1.0 if has_skill else 0.0,
            ))
            if has_skill:
                preferred_matched += 1.0

        score: float = self._weighted_score(
            required_score_sum=required_matched,
            preferred_score_sum=preferred_matched,
            n_required=len(required_skills),
            n_preferred=len(preferred_skills),
            has_candidate_skills=bool(candidate_skills),
        )
        return skill_matches, score

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _weighted_score(
        required_score_sum: float,
        preferred_score_sum: float,
        n_required: int,
        n_preferred: int,
        has_candidate_skills: bool,
    ) -> float:
        """Compute weighted overall score (mirrors MatchingEngine weight structure)."""
        score: float = 0.0
        total_weight: float = 0.0

        if n_required > 0:
            req_weight: float = 0.7
            score += req_weight * (required_score_sum / n_required)
            total_weight += req_weight

        if n_preferred > 0:
            pref_weight: float = 0.3
            score += pref_weight * (preferred_score_sum / n_preferred)
            total_weight += pref_weight

        if total_weight > 0:
            return round(score / total_weight, 3)
        if has_candidate_skills:
            return 0.5
        return 0.0


def get_embedding_skill_scorer(
    embedding_model: Optional[object] = None,
) -> EmbeddingSkillScorer:
    """Get an EmbeddingSkillScorer instance."""
    return EmbeddingSkillScorer(embedding_model=embedding_model)
