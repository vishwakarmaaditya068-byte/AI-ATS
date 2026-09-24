import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from src.services.file_validator import FileValidator, ValidationResult
from src.ml.nlp.accurate_resume_parser import AccurateResumeParser, ParsedResume
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Result type
@dataclass
class IngestionResult:
    status: str = "error"  # "success" | "duplicate" | "error"
    candidate_id: Optional[str] = None
    candidate_name: str = ""
    candidate_email: str = ""
    file_hash: str = ""
    embedding_id: Optional[str] = None  # set after successful embedding
    processing_time_ms: int = 0
    error_message: str = ""
    warnings: list[str] = field(default_factory=list)


# Repository protocol — lets tests inject a fake without importing pymongo
@runtime_checkable
class CandidateRepoProtocol(Protocol):
    def hash_exists(self, file_hash: str) -> bool: ...
    def upsert_by_email(
        self,
        parsed: ParsedResume,
        file_hash: str,
        filename: str,
    ) -> object: ...


# Service
class IngestionService:
    """
    Orchestrates resume ingestion end-to-end.

    Inject a repo stub in tests; call with the real repo in production.
    """

    def __init__(self, repo: Optional[CandidateRepoProtocol] = None) -> None:
        self._validator: FileValidator = FileValidator()
        self._parser: AccurateResumeParser = AccurateResumeParser()
        self._repo: CandidateRepoProtocol = repo or self._get_default_repo()

    @staticmethod
    def _get_default_repo() -> CandidateRepoProtocol:
        """Import the real repo lazily so tests can bypass DB entirely."""
        from src.data.repositories.candidate_repository import CandidateRepository

        return CandidateRepository()

    # Public API
    def ingest_file(self, path: Path | str) -> IngestionResult:
        """Ingest a resume from a filesystem path."""
        path = Path(path)
        if not path.exists():
            return IngestionResult(
                status="error",
                error_message=f"File not found: {path}",
            )
        try:
            content: bytes = path.read_bytes()
        except OSError as exc:
            return IngestionResult(status="error", error_message=str(exc))

        return self._ingest(content, path.name)

    def ingest_bytes(self, content: bytes, filename: str) -> IngestionResult:
        """Ingest a resume from raw bytes (e.g. downloaded from Google Drive)."""
        return self._ingest(content, filename)

    # Internal pipeline
    def _ingest(self, content: bytes, filename: str) -> IngestionResult:
        t0: float = time.monotonic()
        result: IngestionResult = IngestionResult()

        # 1. Validate
        validation: ValidationResult = self._validator.validate_bytes(content, filename)
        if not validation.ok:
            result.status = "error"
            result.error_message = validation.error
            result.processing_time_ms = int((time.monotonic() - t0) * 1000)
            return result

        result.file_hash = validation.file_hash

        # 2. Deduplicate by file hash
        try:
            if self._repo.hash_exists(validation.file_hash):
                result.status = "duplicate"
                result.processing_time_ms = int((time.monotonic() - t0) * 1000)
                logger.info(f"Duplicate file skipped: {filename} ({validation.file_hash[:8]}…)")
                return result
        except Exception as exc:
            logger.warning(f"Hash dedup check failed (continuing): {exc}")

        # 3. Parse — AccurateResumeParser needs a real file path
        try:
            suffix: str = Path(filename).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path: str = tmp.name
            try:
                parsed: ParsedResume = self._parser.parse(tmp_path)
            finally:
                os.unlink(tmp_path)
        except Exception as exc:
            logger.exception(f"Parsing failed for {filename}: {exc}")
            result.status = "error"
            result.error_message = f"Parse error: {exc}"
            result.processing_time_ms = int((time.monotonic() - t0) * 1000)
            return result

        result.candidate_name = parsed.contact.name
        result.candidate_email = parsed.contact.email

        # 4. Upsert candidate
        try:
            candidate_doc: object = self._repo.upsert_by_email(
                parsed, validation.file_hash, filename
            )
            result.candidate_id = str(candidate_doc.id) if candidate_doc else None  # type: ignore[union-attr]
            result.status = "success"
        except Exception as exc:
            logger.exception(f"DB upsert failed for {filename}: {exc}")
            result.status = "error"
            result.error_message = f"Storage error: {exc}"

        # 4b. Audit — fail-soft; never blocks ingestion
        if result.status == "success" and result.candidate_id:
            try:
                from src.services.audit_service import get_audit_service

                get_audit_service().emit_candidate_added(
                    candidate_mongo_id=result.candidate_id,
                    candidate_name=result.candidate_name or "Unknown",
                    source=filename,
                )
            except Exception as exc:
                logger.warning(f"Audit emit (candidate_added) failed: {exc}")

        # 5. Embed (non-fatal — embedding failure never blocks ingestion)
        if result.status == "success" and result.candidate_id:
            try:
                from src.ml.embeddings.embedding_service import EmbeddingService

                emb_svc: EmbeddingService = EmbeddingService(repo=self._repo)
                result.embedding_id = emb_svc.embed_candidate(result.candidate_id, parsed)
            except Exception as exc:
                logger.warning(f"Embedding step failed (non-fatal) for {filename}: {exc}")
                result.warnings.append(f"Embedding unavailable: {exc}")

        result.processing_time_ms = int((time.monotonic() - t0) * 1000)
        return result
