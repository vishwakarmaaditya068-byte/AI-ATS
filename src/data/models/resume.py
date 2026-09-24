"""
Resume data models for AI-ATS.

Defines the schema for resume documents, including file metadata,
parsed content, and processing status.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from .base import BaseDocument, EmbeddedModel, PyObjectId


class ResumeFormat(str, Enum):
    """Supported resume file formats."""

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    TXT = "txt"
    RTF = "rtf"


class ProcessingStatus(str, Enum):
    """Status of resume processing pipeline."""

    PENDING = "pending"
    PROCESSING = "processing"
    PARSED = "parsed"
    ENRICHED = "enriched"  # After NLP enrichment
    EMBEDDED = "embedded"  # After vector embedding
    COMPLETED = "completed"
    FAILED = "failed"


class FileMetadata(EmbeddedModel):
    """Metadata about the uploaded resume file."""

    original_filename: str
    stored_filename: str  # UUID-based filename for storage
    file_format: ResumeFormat
    file_size_bytes: int
    file_hash: str  # SHA-256 hash for deduplication
    storage_path: str  # Relative path to file storage
    mime_type: Optional[str] = None
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("file_size_bytes")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        """Validate file size is reasonable (max 10MB)."""
        max_size = 10 * 1024 * 1024  # 10MB
        if v > max_size:
            raise ValueError(f"File size exceeds maximum of {max_size} bytes")
        return v


class ParsedSection(EmbeddedModel):
    """A parsed section from the resume."""

    section_type: str  # e.g., "summary", "experience", "education", "skills"
    title: Optional[str] = None  # Original section title from resume
    content: str  # Raw text content of the section
    start_position: Optional[int] = None  # Character position in raw text
    end_position: Optional[int] = None
    confidence: float = 1.0  # Confidence score for section detection


class ExtractedEntity(EmbeddedModel):
    """An entity extracted from the resume via NLP."""

    entity_type: str  # e.g., "SKILL", "ORG", "DATE", "DEGREE", "JOB_TITLE"
    value: str
    normalized_value: Optional[str] = None  # Normalized/canonical form
    start_position: int
    end_position: int
    confidence: float = 1.0
    context: Optional[str] = None  # Surrounding text for context


class ParsedContent(EmbeddedModel):
    """Structured content extracted from the resume."""

    raw_text: str  # Full extracted text
    cleaned_text: str  # Preprocessed/cleaned text
    sections: list[ParsedSection] = Field(default_factory=list)
    entities: list[ExtractedEntity] = Field(default_factory=list)
    word_count: int = 0
    language: str = "en"  # Detected language
    language_confidence: float = 1.0


class ProcessingError(EmbeddedModel):
    """Record of a processing error."""

    stage: str  # e.g., "parsing", "nlp", "embedding"
    error_type: str
    error_message: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_recoverable: bool = True


class ProcessingMetrics(EmbeddedModel):
    """Metrics about the processing pipeline."""

    parsing_duration_ms: Optional[int] = None
    nlp_duration_ms: Optional[int] = None
    embedding_duration_ms: Optional[int] = None
    total_duration_ms: Optional[int] = None
    parser_version: Optional[str] = None
    nlp_model_version: Optional[str] = None
    embedding_model_version: Optional[str] = None


class Resume(BaseDocument):
    """
    Main resume document model.

    Stores the uploaded resume file metadata, parsed content,
    and processing status.
    """

    # File Information
    file: FileMetadata

    # Candidate Reference (optional - may be parsed before candidate creation)
    candidate_id: Optional[PyObjectId] = None

    # Processing Status
    status: ProcessingStatus = ProcessingStatus.PENDING

    # Parsed Content
    parsed_content: Optional[ParsedContent] = None

    # Processing Information
    processing_errors: list[ProcessingError] = Field(default_factory=list)
    processing_metrics: Optional[ProcessingMetrics] = None

    # Vector Store Reference
    vector_ids: list[str] = Field(default_factory=list)  # IDs in vector store

    # Quality Metrics
    parse_quality_score: Optional[float] = None  # 0-1 score for parsing quality

    @property
    def has_errors(self) -> bool:
        """Check if resume has any processing errors."""
        return len(self.processing_errors) > 0

    @property
    def is_processed(self) -> bool:
        """Check if resume has been fully processed."""
        return self.status == ProcessingStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if resume processing has failed."""
        return self.status == ProcessingStatus.FAILED

    def add_error(
        self,
        stage: str,
        error_type: str,
        error_message: str,
        is_recoverable: bool = True,
    ) -> None:
        """Add a processing error to the resume."""
        self.processing_errors.append(
            ProcessingError(
                stage=stage,
                error_type=error_type,
                error_message=error_message,
                is_recoverable=is_recoverable,
            )
        )
        if not is_recoverable:
            self.status = ProcessingStatus.FAILED

    class Settings:
        """MongoDB collection settings."""

        name = "resumes"
        indexes = [
            "candidate_id",
            "status",
            "file.file_hash",
            "created_at",
        ]


class ResumeUpload(BaseModel):
    """Schema for uploading a new resume."""

    original_filename: str
    file_content: bytes  # Base64 or raw bytes
    candidate_id: Optional[str] = None  # Optional candidate to link


class ResumeParseResult(BaseModel):
    """Result of resume parsing operation."""

    resume_id: str
    status: ProcessingStatus
    parsed_content: Optional[ParsedContent] = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    metrics: Optional[ProcessingMetrics] = None
