"""
PDF document text extractor.

Uses multiple extraction methods for robust text extraction:
1. pdfplumber - Primary method, good for structured text
2. pypdf - Fallback method
"""

import io
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

from .base import BaseExtractor, ExtractionResult

logger = get_logger(__name__)


class PDFExtractor(BaseExtractor):
    """Extractor for PDF documents."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".pdf",)

    def extract(self, file_path: str | Path) -> ExtractionResult:
        """Extract text from a PDF file."""
        try:
            path = self._validate_file(file_path)
            with open(path, "rb") as f:
                return self._extract_from_file_object(f, str(path))
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            return self._create_error_result(e)

    def extract_from_bytes(
        self, content: bytes, filename: str = "document.pdf"
    ) -> ExtractionResult:
        """Extract text from PDF bytes."""
        try:
            file_obj = io.BytesIO(content)
            return self._extract_from_file_object(file_obj, filename)
        except Exception as e:
            logger.error(f"PDF extraction from bytes failed: {e}")
            return self._create_error_result(e)

    def _extract_from_file_object(
        self, file_obj, filename: str
    ) -> ExtractionResult:
        """Extract text from a file-like object."""
        warnings = []
        metadata = {}

        # Try pdfplumber first (better for structured documents)
        text, page_count, meta = self._extract_with_pdfplumber(file_obj)

        if text and len(text.strip()) > 50:
            metadata.update(meta)
            return ExtractionResult(
                text=text,
                page_count=page_count,
                metadata=metadata,
                warnings=warnings,
            )

        # Fallback to pypdf
        warnings.append("pdfplumber extraction yielded limited text, trying pypdf")
        file_obj.seek(0)
        text, page_count, meta = self._extract_with_pypdf(file_obj)
        metadata.update(meta)

        if not text or len(text.strip()) < 10:
            warnings.append("PDF may be image-based or encrypted")

        return ExtractionResult(
            text=text,
            page_count=page_count,
            metadata=metadata,
            warnings=warnings,
        )

    def _extract_with_pdfplumber(
        self, file_obj
    ) -> tuple[str, int, dict]:
        """Extract text using pdfplumber."""
        try:
            import pdfplumber

            text_parts = []
            page_count = 0
            metadata = {"extractor": "pdfplumber"}

            with pdfplumber.open(file_obj) as pdf:
                page_count = len(pdf.pages)
                metadata["page_count"] = page_count

                if pdf.metadata:
                    metadata["pdf_metadata"] = {
                        k: v for k, v in pdf.metadata.items()
                        if v and isinstance(v, str)
                    }

                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            return "\n\n".join(text_parts), page_count, metadata

        except ImportError:
            logger.warning("pdfplumber not installed, skipping")
            return "", 0, {}
        except Exception as e:
            logger.debug(f"pdfplumber extraction error: {e}")
            return "", 0, {}

    def _extract_with_pypdf(
        self, file_obj
    ) -> tuple[str, int, dict]:
        """Extract text using pypdf."""
        try:
            from pypdf import PdfReader

            text_parts = []
            metadata = {"extractor": "pypdf"}

            reader = PdfReader(file_obj)
            page_count = len(reader.pages)
            metadata["page_count"] = page_count

            if reader.metadata:
                metadata["pdf_metadata"] = {
                    k: str(v) for k, v in reader.metadata.items()
                    if v and k.startswith("/")
                }

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            return "\n\n".join(text_parts), page_count, metadata

        except ImportError:
            logger.warning("pypdf not installed")
            return "", 0, {}
        except Exception as e:
            logger.debug(f"pypdf extraction error: {e}")
            return "", 0, {}
