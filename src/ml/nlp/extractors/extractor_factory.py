"""
Factory for creating appropriate document extractors.
"""

from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

from .base import BaseExtractor, ExtractionResult
from .docx_extractor import DOCXExtractor
from .pdf_extractor import PDFExtractor
from .text_extractor import TextExtractor

logger = get_logger(__name__)


class ExtractorFactory:
    """
    Factory class for creating document extractors.

    Automatically selects the appropriate extractor based on file extension.
    """

    _extractors: list[BaseExtractor] = []
    _initialized: bool = False

    @classmethod
    def _initialize(cls) -> None:
        """Initialize available extractors."""
        if cls._initialized:
            return

        cls._extractors = [
            PDFExtractor(),
            DOCXExtractor(),
            TextExtractor(),
        ]
        cls._initialized = True

    @classmethod
    def get_extractor(cls, file_path: str | Path) -> Optional[BaseExtractor]:
        """
        Get the appropriate extractor for a file.

        Args:
            file_path: Path to the file or filename

        Returns:
            Appropriate extractor or None if no extractor supports the format
        """
        cls._initialize()

        path = Path(file_path)
        extension = path.suffix.lower()

        for extractor in cls._extractors:
            if extension in extractor.supported_extensions:
                return extractor

        logger.warning(f"No extractor found for extension: {extension}")
        return None

    @classmethod
    def extract(cls, file_path: str | Path) -> ExtractionResult:
        """
        Extract text from a file using the appropriate extractor.

        Args:
            file_path: Path to the file

        Returns:
            ExtractionResult with extracted text or error
        """
        extractor = cls.get_extractor(file_path)

        if extractor is None:
            return ExtractionResult(
                text="",
                success=False,
                error_message=f"Unsupported file format: {Path(file_path).suffix}",
            )

        return extractor.extract(file_path)

    @classmethod
    def extract_from_bytes(
        cls, content: bytes, filename: str
    ) -> ExtractionResult:
        """
        Extract text from file bytes using the appropriate extractor.

        Args:
            content: Raw file bytes
            filename: Original filename (for extension detection)

        Returns:
            ExtractionResult with extracted text or error
        """
        extractor = cls.get_extractor(filename)

        if extractor is None:
            return ExtractionResult(
                text="",
                success=False,
                error_message=f"Unsupported file format: {Path(filename).suffix}",
            )

        return extractor.extract_from_bytes(content, filename)

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of all supported file extensions."""
        cls._initialize()

        extensions = []
        for extractor in cls._extractors:
            extensions.extend(extractor.supported_extensions)
        return extensions

    @classmethod
    def is_supported(cls, file_path: str | Path) -> bool:
        """Check if a file format is supported."""
        return cls.get_extractor(file_path) is not None


# Convenience function
def get_extractor(file_path: str | Path) -> Optional[BaseExtractor]:
    """Get the appropriate extractor for a file."""
    return ExtractorFactory.get_extractor(file_path)
