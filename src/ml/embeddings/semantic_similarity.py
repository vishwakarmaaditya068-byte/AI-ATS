"""
Semantic similarity computation for resume-job matching.

Provides high-level functions for computing semantic similarity
between resumes and job descriptions using embeddings.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.data.models import SemanticMatch
from src.ml.nlp import ResumeParseResult, JDParseResult
from src.utils.config import get_settings
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.ml.nlp.accurate_resume_parser import ParsedResume
    from src.data.models.job import Job

from .embedding_model import EmbeddingModel, get_embedding_model
from .vector_store import (
    VectorStore,
    SearchResult,
    get_resume_store,
    get_job_store,
)

logger = get_logger(__name__)

# Section type names used when scanning preprocessed resume sections.
# Centralised here so compute_similarity() and batch_compute_similarity() stay
# in sync when new section types are added to the parser.
_EXPERIENCE_SECTION_TYPES: frozenset[str] = frozenset({"experience", "work_history"})
_SUMMARY_SECTION_TYPES: frozenset[str] = frozenset({"summary", "objective", "profile"})

# Weights for combining per-section similarity scores into weighted_similarity.
# Values sum to 1.0. Tuned to reflect relative job-matching importance:
#   overall    — full-text contextual alignment
#   skills     — direct signal for technical/domain fit
#   experience — role and responsibility relevance
#   summary    — overall profile alignment
_SECTION_WEIGHTS: dict[str, float] = {
    "overall": 0.35,
    "skills": 0.30,
    "experience": 0.25,
    "summary": 0.10,
}


@dataclass
class CandidateMatch:
    """Result of matching a candidate to a job using semantic similarity."""

    candidate_id: str
    candidate_name: str
    similarity_score: float
    semantic_match: SemanticMatch
    resume_text: Optional[str] = None


class SemanticMatcher:
    """
    Computes semantic similarity between resumes and job descriptions.

    Uses sentence embeddings to capture semantic meaning beyond
    keyword matching.
    """

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        resume_store: Optional[VectorStore] = None,
        job_store: Optional[VectorStore] = None,
    ):
        """
        Initialize the semantic matcher.

        Args:
            embedding_model: Optional custom embedding model.
            resume_store: Optional custom vector store for resumes.
            job_store: Optional custom vector store for jobs.
        """
        self._embedding_model = embedding_model
        self._resume_store = resume_store
        self._job_store = job_store
        self._model_name_cache: Optional[str] = None

    @property
    def _model_name(self) -> str:
        """Lazily resolve the embedding model name from settings on first access."""
        if self._model_name_cache is None:
            self._model_name_cache = get_settings().ml.embedding_model
        return self._model_name_cache

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Get the embedding model (lazy initialization)."""
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model

    @property
    def resume_store(self) -> VectorStore:
        """Get the resume vector store (lazy initialization)."""
        if self._resume_store is None:
            self._resume_store = get_resume_store()
        return self._resume_store

    @property
    def job_store(self) -> VectorStore:
        """Get the job vector store (lazy initialization)."""
        if self._job_store is None:
            self._job_store = get_job_store()
        return self._job_store

    def compute_similarity(
        self,
        resume_result: ResumeParseResult,
        jd_result: JDParseResult,
    ) -> SemanticMatch:
        """
        Compute semantic similarity between a resume and job description.

        Batches all four resume-side section texts into one encode() call.
        JD-side texts are encoded individually so the EmbeddingModel LRU
        cache serves them for free when the same job is matched against
        multiple resumes.

        Args:
            resume_result: Parsed resume data.
            jd_result: Parsed job description data.

        Returns:
            SemanticMatch with similarity scores including weighted_similarity.
        """
        # ── Extract full texts ────────────────────────────────────────────
        resume_text: str = self._get_resume_text(resume_result)
        jd_text: str = jd_result.raw_text

        if not resume_text or not jd_text:
            return SemanticMatch(model_used=self._model_name)

        # ── Build resume-side section texts ──────────────────────────────
        candidate_skills: list[str] = [
            s.get("name", "") for s in resume_result.skills if s.get("name")
        ]
        skills_a: str = ", ".join(candidate_skills)

        exp_parts: list[str] = []
        if resume_result.preprocessed and resume_result.preprocessed.sections:
            for section in resume_result.preprocessed.sections:
                if section.section_type in _EXPERIENCE_SECTION_TYPES:
                    exp_parts.append(section.content)
        exp_text: str = " ".join(exp_parts)

        summary_text: str = ""
        if resume_result.preprocessed and resume_result.preprocessed.sections:
            for section in resume_result.preprocessed.sections:
                if section.section_type in _SUMMARY_SECTION_TYPES:
                    summary_text = section.content
                    break
        if not summary_text:
            summary_text = resume_text[:500]

        # ── Batch-encode all non-empty resume-side texts in one call ─────
        _resume_texts: list[str] = [resume_text, skills_a, exp_text, summary_text]
        _nonempty_idxs: list[int] = [i for i, t in enumerate(_resume_texts) if t]
        _nonempty_txts: list[str] = [_resume_texts[i] for i in _nonempty_idxs]
        _resume_embs: dict[int, np.ndarray] = {}
        # index 0 (resume_text) is always non-empty here — the early-exit guard above
        # ensures resume_text is truthy before we reach this point.
        if _nonempty_txts:
            _batch: np.ndarray = self.embedding_model.encode(_nonempty_txts)
            for _pos, _orig_idx in enumerate(_nonempty_idxs):
                _resume_embs[_orig_idx] = _batch[_pos]

        # ── JD-side: individual calls (LRU cache hits after first use) ────
        # NOTE: jd_summary (title + first 500 chars) is always a distinct string
        # from jd_text, so it is always a cache miss in this single-resume path.
        # For multi-resume matching use batch_compute_similarity(), which encodes
        # all four JD-side embeddings once and reuses them across all resumes.
        jd_full_emb: np.ndarray = self.embedding_model.encode(jd_text)

        required_skills: list[str] = jd_result.required_skills + jd_result.preferred_skills
        skills_b: str = ", ".join(required_skills)
        jd_skills_emb: Optional[np.ndarray] = (
            self.embedding_model.encode(skills_b) if skills_b else None
        )

        responsibilities_text: str = " ".join(jd_result.responsibilities)
        jd_resp_emb: Optional[np.ndarray] = (
            self.embedding_model.encode(responsibilities_text) if responsibilities_text else None
        )

        jd_summary: str = f"{jd_result.title}. {jd_text[:500]}"
        jd_summary_emb: np.ndarray = self.embedding_model.encode(jd_summary)

        # ── Compute section similarities ─────────────────────────────────
        overall_sim: float = (
            self.embedding_model.similarity(_resume_embs[0], jd_full_emb)
            if 0 in _resume_embs else 0.0
        )

        if 1 in _resume_embs and jd_skills_emb is not None:
            skills_sim: float = self.embedding_model.similarity(
                _resume_embs[1], jd_skills_emb
            )
        else:
            # 0.5 when neither side has skills (neutral); 0.0 when JD has required
            # skills but resume has none — consistent with batch_compute_similarity()
            # and compute_similarity_from_parsed().
            skills_sim = 0.5 if not skills_b else 0.0

        if 2 in _resume_embs and jd_resp_emb is not None:
            exp_sim: float = self.embedding_model.similarity(
                _resume_embs[2], jd_resp_emb
            )
        else:
            # 0.5 when the JD lists no responsibilities (neutral); 0.0 when JD has
            # responsibilities but resume has no experience sections.
            # Note: compute_similarity_from_parsed() uses 0.0 unconditionally here.
            exp_sim = 0.5 if not responsibilities_text else 0.0

        # index 3 (summary_text) is always present: the fallback to resume_text[:500]
        # above guarantees summary_text is non-empty whenever resume_text is non-empty.
        summary_sim: float = self.embedding_model.similarity(_resume_embs[3], jd_summary_emb)

        # ── Weighted combination (mirrors compute_similarity_from_parsed) ─
        weighted_sim: float = round(
            _SECTION_WEIGHTS["overall"]      * overall_sim
            + _SECTION_WEIGHTS["skills"]     * skills_sim
            + _SECTION_WEIGHTS["experience"] * exp_sim
            + _SECTION_WEIGHTS["summary"]    * summary_sim,
            4,
        )

        return SemanticMatch(
            overall_similarity=round(overall_sim, 4),
            summary_similarity=round(summary_sim, 4),
            skills_similarity=round(skills_sim, 4),
            experience_similarity=round(exp_sim, 4),
            weighted_similarity=weighted_sim,
            model_used=self._model_name,
        )

    def _get_resume_text(self, resume_result: ResumeParseResult) -> str:
        """Extract text content from parsed resume."""
        if resume_result.extraction_result:
            return resume_result.extraction_result.text
        elif resume_result.preprocessed:
            return resume_result.preprocessed.cleaned_text
        return ""

    def index_resume(
        self,
        resume_id: str,
        resume_result: ResumeParseResult,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Index a resume in the vector store for later search.

        Args:
            resume_id: Unique identifier for the resume.
            resume_result: Parsed resume data.
            metadata: Optional additional metadata.
        """
        resume_text = self._get_resume_text(resume_result)
        if not resume_text:
            logger.warning("Cannot index resume %s: no text content", resume_id)
            return

        embedding = self.embedding_model.encode(resume_text)

        # Prepare metadata
        meta = {
            "type": "resume",
            "candidate_name": self._get_candidate_name(resume_result),
            "skills": ",".join(s.get("name", "") for s in resume_result.skills[:20]),
            "experience_years": resume_result.total_experience_years,
            "education": resume_result.highest_education or "",
        }
        if metadata:
            meta.update(metadata)

        self.resume_store.upsert(
            ids=[resume_id],
            embeddings=embedding.reshape(1, -1),
            documents=[resume_text[:2000]],  # Truncate for storage
            metadatas=[meta],
        )

        logger.debug("Indexed resume: %s", resume_id)

    def index_job(
        self,
        job_id: str,
        jd_result: JDParseResult,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Index a job description in the vector store for later search.

        Args:
            job_id: Unique identifier for the job.
            jd_result: Parsed job description data.
            metadata: Optional additional metadata.
        """
        jd_text = jd_result.raw_text
        if not jd_text:
            logger.warning("Cannot index job %s: no text content", job_id)
            return

        embedding = self.embedding_model.encode(jd_text)

        # Prepare metadata
        meta = {
            "type": "job",
            "title": jd_result.title or "",
            "company": jd_result.company_name or "",
            "required_skills": ",".join(jd_result.required_skills[:20]),
            "experience_level": jd_result.experience_level or "",
        }
        if metadata:
            meta.update(metadata)

        self.job_store.upsert(
            ids=[job_id],
            embeddings=embedding.reshape(1, -1),
            documents=[jd_text[:2000]],
            metadatas=[meta],
        )

        logger.debug("Indexed job: %s", job_id)

    def find_matching_candidates(
        self,
        jd_result: JDParseResult,
        top_k: int = 20,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Find candidates that match a job description.

        Args:
            jd_result: Parsed job description.
            top_k: Maximum number of results.
            min_score: Minimum similarity score threshold.

        Returns:
            List of matching candidates sorted by similarity.
        """
        jd_embedding = self.embedding_model.encode(jd_result.raw_text)

        results = self.resume_store.search(
            query_embedding=jd_embedding,
            top_k=top_k,
        )

        # Filter by minimum score
        return [r for r in results if r.score >= min_score]

    def find_matching_jobs(
        self,
        resume_result: ResumeParseResult,
        top_k: int = 20,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Find jobs that match a candidate's resume.

        Args:
            resume_result: Parsed resume.
            top_k: Maximum number of results.
            min_score: Minimum similarity score threshold.

        Returns:
            List of matching jobs sorted by similarity.
        """
        resume_text = self._get_resume_text(resume_result)
        if not resume_text:
            return []

        resume_embedding = self.embedding_model.encode(resume_text)

        results = self.job_store.search(
            query_embedding=resume_embedding,
            top_k=top_k,
        )

        return [r for r in results if r.score >= min_score]

    def _get_candidate_name(self, resume_result: ResumeParseResult) -> str:
        """Extract candidate name from resume."""
        if resume_result.contact:
            name = resume_result.contact.get("full_name")
            if not name:
                first = resume_result.contact.get("first_name", "")
                last = resume_result.contact.get("last_name", "")
                name = f"{first} {last}".strip()
            return name or "Unknown"
        return "Unknown"

    # ------------------------------------------------------------------
    # ParsedResume/Job path — direct typed entry point
    # ------------------------------------------------------------------

    def compute_similarity_from_parsed(
        self,
        parsed: "ParsedResume",
        job: "Job",
    ) -> SemanticMatch:
        """
        Compute semantic similarity between a ParsedResume and a Job model.

        Uses the same text-building logic as EmbeddingService so that similarity
        is computed over identical text representations to what is stored in the
        vector stores — ensuring consistency between indexed and query vectors.

        Args:
            parsed: Output of AccurateResumeParser.
            job: The Job pydantic model from the database.

        Returns:
            SemanticMatch with overall, skills, experience, and summary scores.
        """
        resume_text: str = self._build_resume_text(parsed)
        jd_text: str = self._build_jd_text(job)

        if not resume_text or not jd_text:
            return SemanticMatch(model_used=self._model_name)

        # ── Build all section texts ───────────────────────────────────────────
        candidate_skill_names: list[str] = [
            skill for cat in parsed.skills for skill in cat.skills
        ]
        job_skill_names: list[str] = [s.name for s in job.skill_requirements]
        skills_a: str = ", ".join(candidate_skill_names)
        skills_b: str = ", ".join(job_skill_names)

        exp_text: str = " ".join(
            f"{e.title} {e.company} " + " ".join(e.bullets)
            for e in parsed.experience
        )
        resp_text: str = " ".join(job.responsibilities)

        summary_a: str = parsed.summary or resume_text[:500]
        summary_b: str = f"{job.title}. {(job.description or '')[:500]}"

        # ── Batch-encode all 4 resume-side texts in one model call ───────────
        # JD-side texts are kept as individual calls so the LRU cache in
        # EmbeddingModel can serve them for free when the same job is matched
        # against multiple resumes.
        _resume_section_texts: list[str] = [resume_text, skills_a, exp_text, summary_a]
        _nonempty_idxs: list[int] = [
            i for i, t in enumerate(_resume_section_texts) if t
        ]
        _nonempty_txts: list[str] = [_resume_section_texts[i] for i in _nonempty_idxs]
        _resume_embs: dict[int, np.ndarray] = {}
        if _nonempty_txts:
            _batch: np.ndarray = self.embedding_model.encode(_nonempty_txts)
            for _pos, _orig_idx in enumerate(_nonempty_idxs):
                _resume_embs[_orig_idx] = _batch[_pos]

        # ── JD side — individual calls (served from LRU cache after first use) ─
        jd_full_emb: np.ndarray = self.embedding_model.encode(jd_text)
        jd_skills_emb: Optional[np.ndarray] = (
            self.embedding_model.encode(skills_b) if skills_b else None
        )
        jd_resp_emb: Optional[np.ndarray] = (
            self.embedding_model.encode(resp_text) if resp_text else None
        )
        jd_summary_emb: np.ndarray = self.embedding_model.encode(summary_b)

        # ── Compute section similarities ──────────────────────────────────────
        overall_sim: float = (
            self.embedding_model.similarity(_resume_embs[0], jd_full_emb)
            if 0 in _resume_embs else 0.0
        )

        if 1 in _resume_embs and jd_skills_emb is not None:
            skills_sim: float = self.embedding_model.similarity(
                _resume_embs[1], jd_skills_emb
            )
        else:
            # 0.5 when neither side has skills (neutral); 0.0 when job has required
            # skills but candidate has none — matches compute_similarity() and
            # batch_compute_similarity() fallback behaviour.
            skills_sim = 0.5 if not skills_b else 0.0

        if 2 in _resume_embs and jd_resp_emb is not None:
            exp_sim: float = self.embedding_model.similarity(
                _resume_embs[2], jd_resp_emb
            )
        else:
            # 0.5 when job lists no responsibilities (neutral); 0.0 when job has
            # responsibilities but candidate has no experience sections — matches
            # compute_similarity() and batch_compute_similarity() fallback behaviour.
            exp_sim = 0.5 if not resp_text else 0.0

        # index 3 (summary_a) is always present: the fallback to resume_text[:500]
        # in _build_resume_text guarantees summary_a is non-empty when resume_text is.
        summary_sim: float = self.embedding_model.similarity(_resume_embs[3], jd_summary_emb)

        # ── Weighted combination ──────────────────────────────────────────────
        weighted_sim: float = round(
            _SECTION_WEIGHTS["overall"] * overall_sim
            + _SECTION_WEIGHTS["skills"] * skills_sim
            + _SECTION_WEIGHTS["experience"] * exp_sim
            + _SECTION_WEIGHTS["summary"] * summary_sim,
            4,
        )

        return SemanticMatch(
            overall_similarity=round(overall_sim, 4),
            skills_similarity=round(skills_sim, 4),
            experience_similarity=round(exp_sim, 4),
            summary_similarity=round(summary_sim, 4),
            weighted_similarity=weighted_sim,
            model_used=self._model_name,
        )

    def _build_resume_text(self, parsed: "ParsedResume") -> str:
        """Build embedding text from ParsedResume (mirrors EmbeddingService._build_text)."""
        parts: list[str] = []

        if parsed.summary:
            parts.append(parsed.summary)

        for cat in parsed.skills:
            parts.append(f"{cat.category}: {', '.join(cat.skills)}")

        for exp in parsed.experience:
            header: str = " at ".join(p for p in [exp.title, exp.company] if p).strip()
            parts.append(header)
            parts.extend(exp.bullets)

        for edu in parsed.education:
            parts.append(f"{edu.degree} {edu.institution}".strip())

        for proj in parsed.projects:
            parts.append(proj.name)
            parts.extend(proj.bullets[:2])

        text: str = " ".join(p for p in parts if p.strip())
        if not text and parsed.raw_text:
            text = parsed.raw_text
        return text

    def _build_jd_text(self, job: "Job") -> str:
        """Build embedding text from Job model (mirrors EmbeddingService._build_jd_text)."""
        parts: list[str] = []
        parts.append(job.title)
        if job.description:
            parts.append(job.description)
        for resp in job.responsibilities:
            parts.append(resp)
        required: list[str] = [s.name for s in job.skill_requirements if s.is_required]
        preferred: list[str] = [s.name for s in job.skill_requirements if not s.is_required]
        all_skills: list[str] = required + preferred
        if all_skills:
            parts.append("Skills: " + ", ".join(all_skills))
        if job.company_description:
            parts.append(job.company_description)
        return " ".join(p for p in parts if p.strip())

    def batch_compute_similarity(
        self,
        resume_results: list[ResumeParseResult],
        jd_result: JDParseResult,
        show_progress: bool = False,
    ) -> list[tuple[int, SemanticMatch]]:
        """
        Compute similarity for multiple resumes against a single job.

        All four JD-side embeddings are computed once and reused across every
        resume. Resume component texts (skills, experience, summary) are
        batch-encoded per section type to minimise model calls while still
        producing genuine per-component similarity scores.

        Args:
            resume_results: List of parsed resumes.
            jd_result: Parsed job description.
            show_progress: Whether to show progress bar.

        Returns:
            List of (index, SemanticMatch) tuples.
        """
        # ── JD embeddings — computed once, shared across all resumes ──────
        jd_full_emb = self.embedding_model.encode(jd_result.raw_text)

        jd_skills_text = ", ".join(
            jd_result.required_skills + jd_result.preferred_skills
        )
        jd_skills_emb = (
            self.embedding_model.encode(jd_skills_text) if jd_skills_text else None
        )

        jd_resp_text = " ".join(jd_result.responsibilities)
        jd_resp_emb = (
            self.embedding_model.encode(jd_resp_text) if jd_resp_text else None
        )

        jd_summary_emb = self.embedding_model.encode(
            f"{jd_result.title}. {jd_result.raw_text[:500]}"
        )

        # ── Filter to resumes that have extractable text ───────────────────
        resume_full_texts = [self._get_resume_text(r) for r in resume_results]
        valid_indices = [i for i, t in enumerate(resume_full_texts) if t]
        skipped_indices = [i for i, t in enumerate(resume_full_texts) if not t]

        for i in skipped_indices:
            logger.warning(
                "batch_compute_similarity: resume at index %d has no text — returning zero score", i
            )

        if not valid_indices:
            return [(i, SemanticMatch(model_used=self._model_name)) for i in skipped_indices]

        valid_resumes = [resume_results[i] for i in valid_indices]
        valid_full_texts = [resume_full_texts[i] for i in valid_indices]

        # Batch-encode full resume texts (for overall similarity)
        full_embeddings = self.embedding_model.encode(
            valid_full_texts, show_progress=show_progress
        )

        # ── Extract resume component texts ────────────────────────────────
        skills_texts = [
            ", ".join(s.get("name", "") for s in r.skills if s.get("name"))
            for r in valid_resumes
        ]

        exp_texts = [
            " ".join(
                sec.content
                for sec in (r.preprocessed.sections if r.preprocessed else [])
                if sec.section_type in _EXPERIENCE_SECTION_TYPES
            )
            for r in valid_resumes
        ]

        summary_texts = []
        for r in valid_resumes:
            text = ""
            if r.preprocessed and r.preprocessed.sections:
                for sec in r.preprocessed.sections:
                    if sec.section_type in _SUMMARY_SECTION_TYPES:
                        text = sec.content
                        break
            if not text:
                full = self._get_resume_text(r)
                text = full[:500] if full else ""
            summary_texts.append(text)

        # ── Batch-encode resume component texts ───────────────────────────
        def _batch_encode(texts: list[str]) -> list[Optional[np.ndarray]]:
            """Encode non-empty texts in one model call; return None for empty."""
            non_empty = [(i, t) for i, t in enumerate(texts) if t]
            result: list[Optional[np.ndarray]] = [None] * len(texts)
            if non_empty:
                idxs, txts = zip(*non_empty)
                # encode() returns 2D array when given a list
                embs = self.embedding_model.encode(list(txts))
                for idx, emb in zip(idxs, embs):
                    result[idx] = emb
            return result

        skills_embs = _batch_encode(skills_texts)
        exp_embs = _batch_encode(exp_texts)
        summary_embs = _batch_encode(summary_texts)

        # ── Compute per-resume matches ─────────────────────────────────────
        results = []
        for local_idx, i in enumerate(valid_indices):
            overall_sim = self.embedding_model.similarity(
                full_embeddings[local_idx], jd_full_emb
            )

            skills_sim = (
                self.embedding_model.similarity(skills_embs[local_idx], jd_skills_emb)
                if skills_embs[local_idx] is not None and jd_skills_emb is not None
                else (0.5 if not jd_skills_text else 0.0)
            )

            exp_sim = (
                self.embedding_model.similarity(exp_embs[local_idx], jd_resp_emb)
                if exp_embs[local_idx] is not None and jd_resp_emb is not None
                else (0.5 if not jd_resp_text else 0.0)
            )

            summary_sim = (
                self.embedding_model.similarity(summary_embs[local_idx], jd_summary_emb)
                if summary_embs[local_idx] is not None
                else 0.0
            )

            weighted_sim: float = round(
                _SECTION_WEIGHTS["overall"] * overall_sim
                + _SECTION_WEIGHTS["skills"] * skills_sim
                + _SECTION_WEIGHTS["experience"] * exp_sim
                + _SECTION_WEIGHTS["summary"] * summary_sim,
                4,
            )
            results.append((i, SemanticMatch(
                overall_similarity=round(overall_sim, 4),
                summary_similarity=round(summary_sim, 4),
                skills_similarity=round(skills_sim, 4),
                experience_similarity=round(exp_sim, 4),
                weighted_similarity=weighted_sim,
                model_used=self._model_name,
            )))

        # Append zero-score entries for resumes that had no extractable text so
        # that len(result) == len(resume_results) — callers can index by position.
        for i in skipped_indices:
            results.append((i, SemanticMatch(model_used=self._model_name)))

        return sorted(results, key=lambda t: t[0])


# Singleton instance — guarded by a lock to prevent duplicate model construction
# in multi-threaded contexts (e.g. PyQt6 worker threads for batch imports).
_semantic_matcher: Optional[SemanticMatcher] = None
_semantic_matcher_lock: threading.Lock = threading.Lock()


def get_semantic_matcher() -> SemanticMatcher:
    """Get the semantic matcher singleton instance (thread-safe)."""
    global _semantic_matcher
    if _semantic_matcher is None:
        with _semantic_matcher_lock:
            if _semantic_matcher is None:
                _semantic_matcher = SemanticMatcher()
    return _semantic_matcher
