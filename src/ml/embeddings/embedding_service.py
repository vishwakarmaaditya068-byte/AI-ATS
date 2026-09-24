"""
EmbeddingService — generates and stores resume and job embeddings after ingestion.

Flow (resume):
  ParsedResume → build text → EmbeddingModel.encode_resume() →
  VectorStore.upsert() → CandidateRepository.set_embedding_id()

Flow (job):
  Job → build text → EmbeddingModel.encode_job_description() →
  VectorStore.upsert() → JobRepository.set_embedding_id()
"""
from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

import numpy as np

from src.ml.embeddings.embedding_model import EmbeddingModel, get_embedding_model
from src.ml.embeddings.vector_store import VectorStore, get_resume_store, get_job_store
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.ml.nlp.accurate_resume_parser import ParsedResume
    from src.data.models.job import Job

logger = get_logger(__name__)


class EmbeddingService:
    """
    Thin orchestrator: build resume/job text → encode → upsert to VectorStore → link ID.

    Inject fakes for model and store in tests; uses real singletons in production.
    """

    def __init__(
        self,
        model: Optional[EmbeddingModel] = None,
        store: Optional[VectorStore] = None,
        job_store: Optional[VectorStore] = None,
        repo: Optional[Any] = None,
    ) -> None:
        self._model: EmbeddingModel = model or get_embedding_model()
        self._store: VectorStore = store or get_resume_store()   # candidate store
        self._job_store: Optional[VectorStore] = job_store       # job store — lazy
        self._repo: Optional[Any] = repo

    def embed_candidate(
        self,
        candidate_id: str,
        parsed: "ParsedResume",
    ) -> str:
        """
        Generate, store, and link an embedding for one candidate.

        Args:
            candidate_id: MongoDB ID string of the stored Candidate document.
            parsed: Fully parsed resume from AccurateResumeParser.

        Returns:
            embedding_id — the ChromaDB/FAISS document ID used to store the vector.
        """
        text: str = self._build_text(parsed)
        embedding: np.ndarray = self._model.encode_resume(text)
        embedding_id: str = f"candidate_{candidate_id}"

        self._store.upsert(
            ids=[embedding_id],
            embeddings=np.array([embedding]),
            documents=[text],
            metadatas=[{
                "candidate_id": candidate_id,
                "candidate_name": parsed.contact.name,
                # email intentionally omitted — PII should not be stored in
                # the vector store; look up contact info via candidate_id from
                # the primary database when needed.
            }],
        )

        if self._repo is not None:
            try:
                self._repo.set_embedding_id(candidate_id, embedding_id)
            except Exception:
                # Link failed — delete the just-inserted embedding so the
                # vector store and primary DB stay in sync (no orphaned vectors).
                try:
                    self._store.delete(ids=[embedding_id])
                except Exception as del_exc:
                    logger.warning(
                        f"Failed to clean up orphaned embedding {embedding_id}: {del_exc}"
                    )
                raise

        logger.info(f"Embedded candidate {candidate_id} → {embedding_id}")
        return embedding_id

    def _build_text(self, parsed: "ParsedResume") -> str:
        """
        Build a plain-text representation of a ParsedResume for embedding.

        Concatenates: summary, skill categories, experience titles + bullets,
        education entries, and project names + bullets.
        """
        parts: list[str] = []

        if parsed.summary:
            parts.append(parsed.summary)

        for cat in parsed.skills:
            line: str = f"{cat.category}: {', '.join(cat.skills)}"
            parts.append(line)

        for exp in parsed.experience:
            header: str = " at ".join(p for p in [exp.title, exp.company] if p).strip()
            parts.append(header)
            parts.extend(exp.bullets[:3])

        for edu in parsed.education:
            parts.append(f"{edu.degree} {edu.institution}".strip())

        for proj in parsed.projects:
            parts.append(proj.name)
            parts.extend(proj.bullets[:2])

        text: str = " ".join(p for p in parts if p.strip())

        # Fall back to raw_text when no structured sections produced content
        if not text and parsed.raw_text:
            text = parsed.raw_text

        return text

    def embed_job(
        self,
        job_id: str,
        job: "Job",
    ) -> str:
        """
        Generate, store, and link an embedding for one job.

        Args:
            job_id: MongoDB ID string of the stored Job document.
            job: Job model instance to embed.

        Returns:
            embedding_id — the ChromaDB/FAISS document ID used to store the vector.
        """
        text: str = self._build_jd_text(job)
        embedding: np.ndarray = self._model.encode_job_description(text)
        embedding_id: str = f"job_{job_id}"

        if self._job_store is None:
            self._job_store = get_job_store()
        store: VectorStore = self._job_store
        store.upsert(
            ids=[embedding_id],
            embeddings=np.array([embedding]),
            documents=[text],
            metadatas=[{
                "job_id": job_id,
                "job_title": job.title,
                "company_name": job.company_name,
            }],
        )

        if self._repo is not None:
            self._repo.set_embedding_id(job_id, embedding_id)

        logger.info(f"Embedded job {job_id} → {embedding_id}")
        return embedding_id

    def _build_jd_text(self, job: "Job") -> str:
        """
        Build a plain-text representation of a Job for embedding.

        Concatenates: title, description, responsibilities, required then
        preferred skills, and company description.
        """
        parts: list[str] = []

        parts.append(job.title)

        if job.description:
            parts.append(job.description)

        for responsibility in job.responsibilities:
            parts.append(responsibility)

        required_skills: list[str] = [
            sr.name for sr in job.skill_requirements if sr.is_required
        ]
        preferred_skills: list[str] = [
            sr.name for sr in job.skill_requirements if not sr.is_required
        ]
        all_skills: list[str] = required_skills + preferred_skills
        if all_skills:
            parts.append("Skills: " + ", ".join(all_skills))

        if job.company_description:
            parts.append(job.company_description)

        return " ".join(p for p in parts if p.strip())
