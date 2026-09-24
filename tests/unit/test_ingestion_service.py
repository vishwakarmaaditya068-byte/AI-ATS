"""
Unit tests for IngestionService.
MongoDB interactions are replaced with stubs — no live DB needed.
"""
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from src.services.ingestion_service import IngestionService, IngestionResult

VIVEK_PDF = Path("data/raw/resumes/vivek_resume.pdf")
FAKE_CANDIDATE_ID = "507f1f77bcf86cd799439011"


def _make_service(*, hash_exists=False, email_exists=False, created_id=FAKE_CANDIDATE_ID):
    """Return an IngestionService with DB calls stubbed out."""
    svc = IngestionService.__new__(IngestionService)
    svc._validator = __import__(
        "src.services.file_validator", fromlist=["FileValidator"]
    ).FileValidator()
    svc._parser = __import__(
        "src.ml.nlp.accurate_resume_parser", fromlist=["AccurateResumeParser"]
    ).AccurateResumeParser()

    # Stub repo
    repo = MagicMock()
    repo.hash_exists.return_value = hash_exists
    repo.email_exists.return_value = email_exists
    repo.upsert_by_email.return_value = MagicMock(id=created_id)
    svc._repo = repo
    return svc


class TestIngestFile:
    def test_valid_pdf_returns_success(self):
        svc = _make_service()
        result = svc.ingest_file(VIVEK_PDF)
        assert result.status == "success"
        assert result.candidate_id == FAKE_CANDIDATE_ID

    def test_extracts_name_into_result(self):
        svc = _make_service()
        result = svc.ingest_file(VIVEK_PDF)
        assert "vivek" in result.candidate_name.lower()

    def test_extracts_email_into_result(self):
        svc = _make_service()
        result = svc.ingest_file(VIVEK_PDF)
        assert result.candidate_email == "vaishvivek634@gmail.com"

    def test_duplicate_hash_returns_duplicate_status(self):
        svc = _make_service(hash_exists=True)
        result = svc.ingest_file(VIVEK_PDF)
        assert result.status == "duplicate"
        assert result.candidate_id is None  # no new record created

    def test_nonexistent_file_returns_error(self):
        svc = _make_service()
        result = svc.ingest_file(Path("no_such_file.pdf"))
        assert result.status == "error"
        assert result.error_message

    def test_wrong_extension_returns_error(self, tmp_path):
        svc = _make_service()
        f = tmp_path / "resume.exe"
        f.write_bytes(b"MZ fake")
        result = svc.ingest_file(f)
        assert result.status == "error"
        assert "extension" in result.error_message.lower()

    def test_empty_file_returns_error(self, tmp_path):
        svc = _make_service()
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        result = svc.ingest_file(f)
        assert result.status == "error"

    def test_repo_upsert_called_once_on_success(self):
        svc = _make_service()
        svc.ingest_file(VIVEK_PDF)
        svc._repo.upsert_by_email.assert_called_once()

    def test_repo_not_called_on_duplicate(self):
        svc = _make_service(hash_exists=True)
        svc.ingest_file(VIVEK_PDF)
        svc._repo.upsert_by_email.assert_not_called()


class TestIngestBytes:
    def test_valid_bytes_returns_success(self):
        svc = _make_service()
        content = VIVEK_PDF.read_bytes()
        result = svc.ingest_bytes(content, "vivek_resume.pdf")
        assert result.status == "success"

    def test_empty_bytes_returns_error(self):
        svc = _make_service()
        result = svc.ingest_bytes(b"", "resume.pdf")
        assert result.status == "error"


class TestIngestionResult:
    def test_result_has_processing_time(self):
        svc = _make_service()
        result = svc.ingest_file(VIVEK_PDF)
        assert result.processing_time_ms > 0

    def test_result_has_file_hash(self):
        svc = _make_service()
        result = svc.ingest_file(VIVEK_PDF)
        assert len(result.file_hash) == 64


class TestCandidateRepositoryMethods:
    """Verify the method signatures exist (no live DB)."""

    def test_candidate_repository_has_hash_exists(self):
        from src.data.repositories.candidate_repository import CandidateRepository
        assert hasattr(CandidateRepository, "hash_exists")

    def test_candidate_repository_has_upsert_by_email(self):
        from src.data.repositories.candidate_repository import CandidateRepository
        assert hasattr(CandidateRepository, "upsert_by_email")


class TestDriveIngestion:
    def test_google_drive_service_has_ingest_folder(self):
        from src.services.google_drive_service import GoogleDriveService
        assert hasattr(GoogleDriveService, "ingest_folder")


class TestIngestionEmbedding:
    def test_result_has_embedding_id_field(self) -> None:
        svc: IngestionService = _make_service()
        result: IngestionResult = svc.ingest_file(VIVEK_PDF)
        assert hasattr(result, "embedding_id")

    def test_embedding_failure_does_not_break_ingestion(self) -> None:
        """EmbeddingService raising should not change status to error."""
        from unittest.mock import patch
        svc: IngestionService = _make_service()
        with patch(
            "src.ml.embeddings.embedding_service.EmbeddingService.embed_candidate",
            side_effect=RuntimeError("GPU unavailable"),
        ):
            result: IngestionResult = svc.ingest_file(VIVEK_PDF)
        assert result.status == "success"

    def test_embedding_warning_added_on_failure(self) -> None:
        from unittest.mock import patch
        svc: IngestionService = _make_service()
        with patch(
            "src.ml.embeddings.embedding_service.EmbeddingService.embed_candidate",
            side_effect=RuntimeError("GPU unavailable"),
        ):
            result: IngestionResult = svc.ingest_file(VIVEK_PDF)
        assert any("embedding" in w.lower() for w in result.warnings)
