"""
Plain text and RTF document extractor.
"""

import io
import re
from pathlib import Path

from src.utils.logger import get_logger

from .base import BaseExtractor, ExtractionResult

logger = get_logger(__name__)


class TextExtractor(BaseExtractor):
    """Extractor for plain text and RTF documents."""

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".txt", ".rtf", ".text", ".md")

    def extract(self, file_path: str | Path) -> ExtractionResult:
        """Extract text from a text file."""
        try:
            path = self._validate_file(file_path)

            if path.suffix.lower() == ".rtf":
                return self._extract_rtf(path)
            else:
                return self._extract_plain_text(path)

        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            return self._create_error_result(e)

    def extract_from_bytes(
        self, content: bytes, filename: str = "document.txt"
    ) -> ExtractionResult:
        """Extract text from bytes."""
        try:
            ext = Path(filename).suffix.lower()

            if ext == ".rtf":
                return self._extract_rtf_from_bytes(content)
            else:
                return self._extract_plain_text_from_bytes(content)

        except Exception as e:
            logger.error(f"Text extraction from bytes failed: {e}")
            return self._create_error_result(e)

    def _extract_plain_text(self, file_path: Path) -> ExtractionResult:
        """Extract content from a plain text file."""
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        text = None
        used_encoding = None

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                    used_encoding = encoding
                    break
            except UnicodeDecodeError:
                continue

        if text is None:
            # Last resort: read as binary and decode with errors ignored
            with open(file_path, "rb") as f:
                text = f.read().decode("utf-8", errors="ignore")
                used_encoding = "utf-8 (with errors ignored)"

        # Estimate page count (roughly 3000 chars per page)
        page_count = max(1, len(text) // 3000)

        return ExtractionResult(
            text=text,
            page_count=page_count,
            metadata={
                "extractor": "plain_text",
                "encoding": used_encoding,
            },
        )

    def _extract_plain_text_from_bytes(self, content: bytes) -> ExtractionResult:
        """Extract text from plain text bytes."""
        encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]
        text = None
        used_encoding = None

        for encoding in encodings:
            try:
                text = content.decode(encoding)
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            text = content.decode("utf-8", errors="ignore")
            used_encoding = "utf-8 (with errors ignored)"

        page_count = max(1, len(text) // 3000)

        return ExtractionResult(
            text=text,
            page_count=page_count,
            metadata={
                "extractor": "plain_text",
                "encoding": used_encoding,
            },
        )

    def _extract_rtf(self, file_path: Path) -> ExtractionResult:
        """Extract text from an RTF file."""
        with open(file_path, "rb") as f:
            content = f.read()
        return self._extract_rtf_from_bytes(content)

    def _extract_rtf_from_bytes(self, content: bytes) -> ExtractionResult:
        """Extract text from RTF bytes."""
        warnings = []

        # Try striprtf library first
        try:
            from striprtf.striprtf import rtf_to_text

            text = content.decode("utf-8", errors="ignore")
            plain_text = rtf_to_text(text)

            page_count = max(1, len(plain_text) // 3000)

            return ExtractionResult(
                text=plain_text,
                page_count=page_count,
                metadata={"extractor": "striprtf"},
                warnings=warnings,
            )

        except ImportError:
            logger.debug("striprtf not installed, using basic RTF extraction")
            warnings.append("striprtf not installed - using basic extraction")

        # Fallback: basic RTF stripping
        text = self._basic_rtf_strip(content.decode("utf-8", errors="ignore"))
        page_count = max(1, len(text) // 3000)

        return ExtractionResult(
            text=text,
            page_count=page_count,
            metadata={"extractor": "basic_rtf"},
            warnings=warnings,
        )

    def _basic_rtf_strip(self, rtf_text: str) -> str:
        """
        Basic RTF to plain text conversion.
        Strips RTF control words and extracts visible text.
        """
        # Remove RTF header
        text = re.sub(r"^\{\\rtf1.*?(?=\n)", "", rtf_text, flags=re.DOTALL)

        # Remove font tables, color tables, etc.
        text = re.sub(r"\{\\fonttbl[^}]*\}", "", text)
        text = re.sub(r"\{\\colortbl[^}]*\}", "", text)
        text = re.sub(r"\{\\stylesheet[^}]*\}", "", text)
        text = re.sub(r"\{\\info[^}]*\}", "", text)

        # Remove control words
        text = re.sub(r"\\[a-z]+\d*\s?", "", text)

        # Remove special characters
        text = re.sub(r"\\'[0-9a-fA-F]{2}", "", text)

        # Remove braces
        text = re.sub(r"[{}]", "", text)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text
