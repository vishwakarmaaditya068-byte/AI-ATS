"""
Domain-aware experience scorer for candidate-job matching.

Replaces the hardcoded total-years thresholds in MatchingEngine._match_experience
with an embedding-based approach: each candidate experience entry is encoded and
compared against the job context (title + responsibilities) via cosine similarity.
Only the fraction of each entry's years proportional to its relevance score
contributes to the relevant-years total.

Standalone — no circular project imports at module level.
Lazy-loads EmbeddingModel on first use (same pattern as EmbeddingSkillScorer).
Falls back to total-years-based scoring when the embedding model is unavailable.
"""
from __future__ import annotations

import datetime
import logging
import re
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.data.models import ExperienceMatch
from src.utils.constants import EXP_RELEVANCE_TITLE_THRESHOLD

if TYPE_CHECKING:
    from src.ml.nlp.accurate_resume_parser import ExperienceEntry as ExperienceEntryType

logger = logging.getLogger(__name__)


class DomainAwareExperienceScorer:
    """
    Score candidate experience against job requirements using embedding similarity.

    Algorithm:
      1. Build entry_text_i = "{title} {company} {bullets[:5]}" for each entry.
      2. Batch-encode all entry texts in one .encode(list) call   → (n, dim).
      3. Encode job context "{job_title} {responsibilities[:10]}" as single string
         (benefits from EmbeddingModel's per-string LRU cache)    → (dim,).
      4. Cosine similarities: entry_embs @ job_emb                → (n,).
      5. relevance_i = max(0, sim_i).
      6. relevant_years_i = years_i * relevance_i.
      7. Score = _score_from_years(sum(relevant_years_i), required_years).
      8. relevant_titles_matched = [e.title for i if relevance_i >= EXP_RELEVANCE_TITLE_THRESHOLD].

    Falls back to total-years-based scoring when the embedding model is unavailable.
    """

    def __init__(self, embedding_model: Optional[object] = None) -> None:
        """
        Args:
            embedding_model: Optional pre-built EmbeddingModel instance.
                             If None, lazy-loaded on first call to score_experience().
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
                logger.info("DomainAwareExperienceScorer: model loaded successfully")
            except Exception as exc:
                logger.warning(f"DomainAwareExperienceScorer: model unavailable — {exc}")
                self._model_load_failed = True
        return self._model

    def score_experience(
        self,
        entries: list["ExperienceEntryType"],
        required_years: float,
        job_title: str,
        responsibilities: list[str],
    ) -> tuple[ExperienceMatch, float]:
        """
        Score candidate experience entries against a job context.

        Args:
            entries:          Candidate ExperienceEntry list (from ParsedResume).
            required_years:   Minimum years required by the job (0 = no requirement).
            job_title:        Job title string.
            responsibilities: Job responsibilities list (first 10 used for context).

        Returns:
            (ExperienceMatch, score_in_[0,1])
        """
        entry_years: list[float] = [_estimate_years(e.duration) for e in entries]
        total_candidate_years: float = sum(entry_years)

        if required_years == 0:
            score: float = 1.0 if total_candidate_years > 0 else 0.5
            match: ExperienceMatch = ExperienceMatch(
                required_years=0.0,
                candidate_years=total_candidate_years,
                years_difference=total_candidate_years,
                meets_minimum=True,
                score=score,
            )
            return match, round(score, 3)

        if self.model is not None and entries:
            return self._score_with_embeddings(
                entries,
                required_years,
                total_candidate_years,
                job_title,
                responsibilities,
                entry_years=entry_years,
            )
        return self._score_total_years_fallback(total_candidate_years, required_years)

    # ------------------------------------------------------------------
    # Embedding path
    # ------------------------------------------------------------------

    def _score_with_embeddings(
        self,
        entries: list["ExperienceEntryType"],
        required_years: float,
        total_candidate_years: float,
        job_title: str,
        responsibilities: list[str],
        entry_years: list[float],
    ) -> tuple[ExperienceMatch, float]:
        """Core embedding-based scoring path. Uses two encode calls total."""
        # Build entry texts: title + company + first 5 bullets
        entry_texts: list[str] = [
            f"{e.title} {e.company} {' '.join(e.bullets[:5])}".strip()
            for e in entries
        ]

        # Batch 1: encode all entry texts (list call → 2D array)
        entry_embs: np.ndarray = self.model.encode(entry_texts, normalize=True)  # type: ignore[union-attr]
        if entry_embs.ndim == 1:
            entry_embs = entry_embs.reshape(1, -1)

        # Batch 2: encode job context as single string (uses EmbeddingModel LRU cache)
        job_context: str = f"{job_title} {' '.join(responsibilities[:10])}".strip()
        job_emb: np.ndarray = self.model.encode(job_context, normalize=True)  # type: ignore[union-attr]
        job_emb = job_emb.flatten()

        # Cosine similarities: (n, dim) @ (dim,) → (n,)
        sims: np.ndarray = entry_embs @ job_emb

        relevant_years: float = 0.0
        relevant_titles: list[str] = []

        for i, entry in enumerate(entries):
            relevance: float = max(0.0, float(sims[i]))
            relevant_years += entry_years[i] * relevance
            if relevance >= EXP_RELEVANCE_TITLE_THRESHOLD and entry.title:
                relevant_titles.append(entry.title)

        score: float = self._score_from_years(relevant_years, required_years)
        match: ExperienceMatch = ExperienceMatch(
            required_years=required_years,
            candidate_years=total_candidate_years,
            years_difference=relevant_years - required_years,
            meets_minimum=relevant_years >= required_years,
            relevant_titles_matched=relevant_titles,
            score=score,
        )
        return match, round(score, 3)

    # ------------------------------------------------------------------
    # Fallback path
    # ------------------------------------------------------------------

    def _score_total_years_fallback(
        self,
        candidate_years: float,
        required_years: float,
    ) -> tuple[ExperienceMatch, float]:
        """Fallback: same tiers as legacy _match_experience(), no domain weighting."""
        score: float = self._score_from_years(candidate_years, required_years)
        match: ExperienceMatch = ExperienceMatch(
            required_years=required_years,
            candidate_years=candidate_years,
            years_difference=candidate_years - required_years,
            meets_minimum=candidate_years >= required_years,
            score=score,
        )
        return match, round(score, 3)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_from_years(years: float, required: float) -> float:
        """Map a years value to a [0,1] score using the same tiers as _match_experience."""
        if required <= 0.0:
            return 1.0 if years > 0.0 else 0.5
        if years >= required:
            return 1.0
        if years >= required * 0.7:
            return 0.7 + 0.3 * (years / required)
        if years > 0:
            return 0.5 * (years / required)
        return 0.0


def _estimate_years(duration: str) -> float:
    """Best-effort year estimate from a duration string like '2019-2022' or '2 years'.

    Standalone copy — DomainAwareExperienceScorer has no module-level dependency on
    matching_engine, so this helper is reproduced here rather than imported.
    """
    if not duration:
        return 1.0
    m: Optional[re.Match[str]] = re.search(r"(\d+(?:\.\d+)?)\s*year", duration, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+)\s*month", duration, re.IGNORECASE)
    if m:
        return round(float(m.group(1)) / 12, 1)
    if re.search(r"\b(present|current|now)\b", duration, re.IGNORECASE):
        m_start: list[str] = re.findall(r"\b(20\d{2}|19\d{2})\b", duration)
        if m_start:
            years: int = datetime.date.today().year - int(m_start[0])
            return float(max(years, 1))
    m_years: list[str] = re.findall(r"\b(20\d{2}|19\d{2})\b", duration)
    if len(m_years) >= 2:
        years = abs(int(m_years[-1]) - int(m_years[0]))
        return float(max(years, 1))
    return 1.0


def get_domain_aware_experience_scorer(
    embedding_model: Optional[object] = None,
) -> DomainAwareExperienceScorer:
    """Get a DomainAwareExperienceScorer instance."""
    return DomainAwareExperienceScorer(embedding_model=embedding_model)
