"""
Embedding-aware education scorer for candidate-job matching.

Extends the legacy degree-level comparison with a field-of-study relevance
signal: the candidate's study field is encoded and compared against the job
context (title + description) via cosine similarity. The final score is a
weighted blend of the degree-level score and the field similarity score.

Standalone — no circular project imports at module level.
Lazy-loads EmbeddingModel on first use (same pattern as EmbeddingSkillScorer).
Falls back to degree-level-only scoring when the embedding model is unavailable.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.data.models import EducationMatch
from src.utils.constants import (
    EDUCATION_LEVELS,
    EDU_FIELD_MATCH_THRESHOLD,
    EDU_FIELD_WEIGHT,
)

if TYPE_CHECKING:
    from src.ml.nlp.accurate_resume_parser import EducationEntry as EducationEntryType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level compiled patterns (created once)
# ---------------------------------------------------------------------------

_DEGREE_LEVEL_RE: re.Pattern[str] = re.compile(
    r"\b(ph\.?d\.?|doctor(?:ate)?|"
    r"m\.?b\.?a|"
    r"master(?:\'?s)?|m\.?tech|m\.?sc?\.?|m\.?s\.?|m\.?a\.?|m\.?e\.?|"
    r"bachelor(?:\'?s)?|b\.?tech|b\.?e\.?|b\.?sc?\.?|b\.?s\.?|b\.?a\.?|"
    r"associate(?:\'?s)?|a\.?a\.?|a\.?s\.?|"
    r"diploma|high\s+school|hsc|ssc|secondary|matriculation|"
    r"of|in|the|and|a|an|degree|programme|program)\b",
    re.IGNORECASE,
)

_LEVEL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(ph\.?d\.?|doctor(?:ate)?)\b", re.IGNORECASE), "phd"),
    (re.compile(r"\bm\.?b\.?a\b", re.IGNORECASE), "mba"),
    (re.compile(r"\b(master(?:\'?s)?|m\.?tech|m\.?sc?\.?|m\.?s\.?|m\.?a\.?|m\.?e\.?)\b", re.IGNORECASE), "master"),
    (re.compile(r"\b(bachelor(?:\'?s)?|b\.?tech|b\.?e\.?|b\.?sc?\.?|b\.?s\.?|b\.?a\.?)\b", re.IGNORECASE), "bachelor"),
    (re.compile(r"\b(associate(?:\'?s)?|a\.?a\.?|a\.?s\.?)\b", re.IGNORECASE), "associate"),
    (re.compile(r"\bdiploma\b", re.IGNORECASE), "diploma"),
    (re.compile(r"\bhigh\s+school\b", re.IGNORECASE), "high school"),
]


class EmbeddingEducationScorer:
    """
    Score candidate education against job requirements using degree level +
    field-of-study embedding similarity.

    Algorithm:
      1. Normalize candidate degree text to a key in EDUCATION_LEVELS via
         _normalize_degree_level() — e.g., "B.Tech CS" → "bachelor".
      2. Compute degree_score from the level comparison (same tiers as
         legacy _match_education()).
      3. Extract field-of-study substring via _extract_field().
      4. If field text is non-empty and model is available:
           a. Encode field text as single string (uses EmbeddingModel LRU cache).
           b. Encode job context "{job_title} {job_description[:200]}" as single string.
           c. field_sim = max(0, dot(field_emb, job_emb)).
           d. combined = degree_score * (1 - EDU_FIELD_WEIGHT) + field_sim * EDU_FIELD_WEIGHT.
      5. Set EducationMatch.field_match = field_sim >= EDU_FIELD_MATCH_THRESHOLD.

    Falls back to degree-level-only scoring when the embedding model is unavailable
    or when no field text can be extracted from the degree string.
    """

    def __init__(self, embedding_model: Optional[object] = None) -> None:
        """
        Args:
            embedding_model: Optional pre-built EmbeddingModel instance.
                             If None, lazy-loaded on first call to score_education().
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
                logger.info("EmbeddingEducationScorer: model loaded successfully")
            except Exception as exc:
                logger.warning(f"EmbeddingEducationScorer: model unavailable — {exc}")
                self._model_load_failed = True
        return self._model

    def score_education(
        self,
        education_entries: list["EducationEntryType"],
        required_degree: str,
        job_title: str,
        job_description: str,
    ) -> tuple[EducationMatch, float]:
        """
        Score candidate education entries against a job context.

        Args:
            education_entries: Candidate EducationEntry list (from ParsedResume).
            required_degree:   Minimum degree required by the job ('' = no requirement).
            job_title:         Job title string.
            job_description:   Job description (first 200 chars used for context).

        Returns:
            (EducationMatch, score_in_[0,1])
        """
        # No requirement path
        if not required_degree:
            score: float = 1.0 if education_entries else 0.7
            match: EducationMatch = EducationMatch(
                required_degree=None,
                candidate_degree=self._best_degree(education_entries) or None,
                meets_requirement=True,
                score=score,
            )
            return match, round(score, 3)

        # Select the highest-level degree across all education entries so that
        # a candidate with both B.Tech and M.Tech is compared using M.Tech.
        candidate_degree_text: str = self._best_degree(education_entries)
        if not candidate_degree_text:
            match = EducationMatch(
                required_degree=required_degree,
                candidate_degree=None,
                meets_requirement=False,
                score=0.3,
            )
            return match, 0.3

        # Compute degree-level score
        candidate_normalized: str = self._normalize_degree_level(candidate_degree_text)
        degree_score: float
        meets_req: bool
        degree_score, meets_req = self._degree_level_score(
            candidate_normalized, required_degree
        )

        # Attempt embedding-enhanced field scoring
        field_sim: float = 0.0
        combined_score: float = degree_score

        if self.model is not None:
            combined_score, field_sim = self._score_with_embeddings(
                degree_score, candidate_degree_text, job_title, job_description
            )

        field_match: bool = field_sim >= EDU_FIELD_MATCH_THRESHOLD
        match = EducationMatch(
            required_degree=required_degree,
            candidate_degree=candidate_degree_text,
            meets_requirement=meets_req,
            field_match=field_match,
            score=combined_score,
        )
        return match, round(combined_score, 3)

    # ------------------------------------------------------------------
    # Embedding path
    # ------------------------------------------------------------------

    def _score_with_embeddings(
        self,
        degree_score: float,
        candidate_degree_text: str,
        job_title: str,
        job_description: str,
    ) -> tuple[float, float]:
        """
        Blend degree-level score with field-of-study embedding similarity.

        Returns:
            (combined_score, field_sim) — field_sim is 0.0 if no field extracted.
        """
        field_text: str = self._extract_field(candidate_degree_text)
        if not field_text:
            return degree_score, 0.0

        # Encode field text (single string → 1D, uses EmbeddingModel LRU cache)
        field_emb: np.ndarray = self.model.encode(field_text.lower(), normalize=True)  # type: ignore[union-attr]
        field_emb = field_emb.flatten()

        # Encode job context (single string → 1D, uses EmbeddingModel LRU cache)
        job_context: str = f"{job_title} {job_description[:200]}".strip()
        job_emb: np.ndarray = self.model.encode(job_context.lower(), normalize=True)  # type: ignore[union-attr]
        job_emb = job_emb.flatten()

        field_sim: float = max(0.0, float(np.dot(field_emb, job_emb)))
        combined: float = (
            degree_score * (1.0 - EDU_FIELD_WEIGHT)
            + field_sim * EDU_FIELD_WEIGHT
        )
        return round(min(1.0, max(0.0, combined)), 3), field_sim

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _best_degree(
        education_entries: list["EducationEntryType"],
    ) -> str:
        """Return the degree text from the highest-level education entry.

        Iterates all entries and picks the one whose normalized level key
        maps to the highest integer in EDUCATION_LEVELS. Falls back to the
        first entry's degree when no level can be resolved.
        """
        if not education_entries:
            return ""
        best_text: str = ""
        best_level: int = -1
        for entry in education_entries:
            if not entry.degree:
                continue
            level_key: str = EmbeddingEducationScorer._normalize_degree_level(entry.degree)
            level: int = EDUCATION_LEVELS.get(level_key.lower(), 0)
            if level > best_level:
                best_level = level
                best_text = entry.degree
        return best_text or (education_entries[0].degree if education_entries else "")

    @staticmethod
    def _degree_level_score(
        candidate_normalized: str, required_degree: str
    ) -> tuple[float, bool]:
        """
        Compare candidate and required degree levels.

        Returns:
            (degree_score, meets_requirement)
        """
        required_level: int = EDUCATION_LEVELS.get(required_degree.lower(), 0)
        candidate_level: int = EDUCATION_LEVELS.get(candidate_normalized.lower(), 0)

        if required_level == 0:
            return 1.0, True

        if candidate_level >= required_level:
            return 1.0, True
        if candidate_level == required_level - 1:
            return 0.7, False
        return (
            max(0.3, candidate_level / required_level) if required_level > 0 else 0.3,
            False,
        )

    @staticmethod
    def _normalize_degree_level(degree_text: str) -> str:
        """Map a degree text string to its key in EDUCATION_LEVELS."""
        for pattern, level_key in _LEVEL_PATTERNS:
            if pattern.search(degree_text):
                return level_key
        return degree_text.lower()

    @staticmethod
    def _extract_field(degree_text: str) -> str:
        """Remove degree-level keywords from degree text to isolate the field of study."""
        field: str = _DEGREE_LEVEL_RE.sub(" ", degree_text)
        return re.sub(r"\s+", " ", field).strip()


def get_embedding_education_scorer(
    embedding_model: Optional[object] = None,
) -> EmbeddingEducationScorer:
    """Get an EmbeddingEducationScorer instance."""
    return EmbeddingEducationScorer(embedding_model=embedding_model)
