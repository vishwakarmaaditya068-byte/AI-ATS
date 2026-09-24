from pathlib import Path
import pytest
from src.services.file_validator import FileValidator, ValidationResult

RESUMES_DIR = Path("data/raw/resumes")
VIVEK_PDF = RESUMES_DIR / "vivek_resume.pdf"


@pytest.fixture
def validator():
    return FileValidator()


class TestValidPath:
    def test_valid_pdf_returns_ok(self, validator):
        result = validator.validate_path(VIVEK_PDF)
        assert result.ok is True
        assert result.error == ""

    def test_returns_file_hash(self, validator):
        result = validator.validate_path(VIVEK_PDF)
        assert len(result.file_hash) == 64  # sha256 hex


class TestExtension:
    def test_rejects_txt_extension(self, validator, tmp_path):
        f = tmp_path / "resume.txt"
        f.write_bytes(b"%PDF-1.4 fake")
        result = validator.validate_path(f)
        assert result.ok is False
        assert "extension" in result.error.lower()

    def test_rejects_exe_extension(self, validator, tmp_path):
        f = tmp_path / "resume.exe"
        f.write_bytes(b"MZ fake binary")
        result = validator.validate_path(f)
        assert result.ok is False


class TestSize:
    def test_rejects_empty_file(self, validator, tmp_path):
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        result = validator.validate_path(f)
        assert result.ok is False
        assert "empty" in result.error.lower()

    def test_rejects_file_over_50mb(self, validator, tmp_path):
        f = tmp_path / "huge.pdf"
        f.write_bytes(b"%PDF-1.4 " + b"x" * (51 * 1024 * 1024))
        result = validator.validate_path(f)
        assert result.ok is False
        assert "size" in result.error.lower() or "large" in result.error.lower()


class TestMagicBytes:
    def test_rejects_non_pdf_bytes_with_pdf_extension(self, validator, tmp_path):
        f = tmp_path / "fake.pdf"
        f.write_bytes(b"PK\x03\x04 this is a zip")  # ZIP magic bytes
        result = validator.validate_path(f)
        assert result.ok is False
        assert "magic" in result.error.lower() or "pdf" in result.error.lower()

    def test_accepts_real_pdf_magic(self, validator, tmp_path):
        f = tmp_path / "real.pdf"
        f.write_bytes(b"%PDF-1.4\n%%EOF")
        result = validator.validate_path(f)
        assert result.ok is True


class TestBytesValidation:
    def test_validate_bytes_valid_pdf(self, validator):
        content = VIVEK_PDF.read_bytes()
        result = validator.validate_bytes(content, "vivek_resume.pdf")
        assert result.ok is True

    def test_validate_bytes_empty(self, validator):
        result = validator.validate_bytes(b"", "resume.pdf")
        assert result.ok is False
        assert "empty" in result.error.lower()
