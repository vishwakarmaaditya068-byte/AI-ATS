"""
Candidate data models for AI-ATS.

Defines the schema for candidate profiles, including personal information,
skills, work experience, and education.
"""

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

from src.utils.constants import CandidateStatus

from .base import BaseDocument, EmbeddedModel, PyObjectId


class ContactInfo(EmbeddedModel):
    """Candidate contact information."""

    email: EmailStr
    phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None


class Skill(EmbeddedModel):
    """Represents a single skill with metadata."""

    name: str
    category: Optional[str] = None  # e.g., "programming_languages", "frameworks"
    proficiency_level: Optional[str] = None  # e.g., "beginner", "intermediate", "expert"
    years_of_experience: Optional[float] = None
    is_verified: bool = False  # Whether skill was verified through assessment

    @field_validator("name")
    @classmethod
    def normalize_skill_name(cls, v: str) -> str:
        """Normalize skill names to lowercase for consistency."""
        return v.strip().lower()

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Skill):
            return NotImplemented
        return self.name == other.name


class WorkExperience(EmbeddedModel):
    """Represents a single work experience entry."""

    job_title: str
    company: str
    location: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None  # None indicates current position
    is_current: bool = False
    description: Optional[str] = None
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    skills_used: list[str] = Field(default_factory=list)

    @property
    def duration_months(self) -> Optional[int]:
        """Calculate duration of employment in months."""
        if not self.start_date:
            return None
        end = self.end_date or date.today()
        delta = end - self.start_date
        return max(1, delta.days // 30)


class Education(EmbeddedModel):
    """Represents a single education entry."""

    degree: str  # e.g., "Bachelor", "Master", "PhD"
    field_of_study: str  # e.g., "Computer Science"
    institution: str
    location: Optional[str] = None
    start_date: Optional[date] = None
    graduation_date: Optional[date] = None
    gpa: Optional[float] = None
    honors: Optional[str] = None
    relevant_coursework: list[str] = Field(default_factory=list)

    @field_validator("gpa")
    @classmethod
    def validate_gpa(cls, v: Optional[float]) -> Optional[float]:
        """Validate GPA is within reasonable range."""
        if v is not None and (v < 0 or v > 4.0):
            raise ValueError("GPA must be between 0 and 4.0")
        return v


class Certification(EmbeddedModel):
    """Represents a professional certification."""

    name: str
    issuing_organization: str
    issue_date: Optional[date] = None
    expiration_date: Optional[date] = None
    credential_id: Optional[str] = None
    credential_url: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if certification is still valid."""
        if not self.expiration_date:
            return True
        return self.expiration_date >= date.today()


class Language(EmbeddedModel):
    """Represents a language proficiency."""

    language: str
    proficiency: str = "conversational"  # native, fluent, professional, conversational, basic


class CandidateMetadata(EmbeddedModel):
    """Metadata about the candidate record."""

    source: Optional[str] = None  # e.g., "linkedin", "job_board", "referral"
    source_id: Optional[str] = None  # ID from the source system
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    recruiter_id: Optional[str] = None  # ID of assigned recruiter
    last_contacted: Optional[datetime] = None
    imported_at: Optional[datetime] = None


class Candidate(BaseDocument):
    """
    Main candidate model representing a job applicant.

    This is the primary document stored in the candidates collection.
    """

    # Personal Information
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    contact: ContactInfo

    # Professional Summary
    headline: Optional[str] = None  # e.g., "Senior Software Engineer"
    summary: Optional[str] = None  # Professional summary/bio

    # Qualifications
    skills: list[Skill] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)

    # Status
    status: CandidateStatus = CandidateStatus.NEW

    # Resume Reference
    resume_id: Optional[PyObjectId] = None  # Reference to Resume document

    # Metadata
    metadata: CandidateMetadata = Field(default_factory=CandidateMetadata)

    # Vector embedding for semantic search (stored separately in vector DB)
    embedding_id: Optional[str] = None

    # Deduplication — SHA-256 hashes of every resume file ingested for this candidate
    file_hashes: list[str] = Field(default_factory=list)

    @property
    def full_name(self) -> str:
        """Get candidate's full name."""
        return f"{self.first_name} {self.last_name}"

    @property
    def total_experience_years(self) -> float:
        """Calculate total years of work experience."""
        total_months = 0
        for exp in self.work_experience:
            if exp.duration_months:
                total_months += exp.duration_months
        return round(total_months / 12, 1)

    @property
    def highest_education_level(self) -> Optional[str]:
        """Get the highest education level achieved."""
        from src.utils.constants import EDUCATION_LEVELS

        if not self.education:
            return None

        highest_level = 0
        highest_degree = None

        for edu in self.education:
            degree_lower = edu.degree.lower()
            for level_name, level_value in EDUCATION_LEVELS.items():
                if level_name in degree_lower and level_value > highest_level:
                    highest_level = level_value
                    highest_degree = edu.degree

        return highest_degree

    @property
    def skill_names(self) -> list[str]:
        """Get list of all skill names."""
        return [skill.name for skill in self.skills]

    class Settings:
        """MongoDB collection settings."""

        name = "candidates"
        indexes = [
            "contact.email",
            "status",
            "metadata.tags",
            "created_at",
        ]


class CandidateCreate(BaseModel):
    """Schema for creating a new candidate."""

    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    contact: ContactInfo
    headline: Optional[str] = None
    summary: Optional[str] = None
    skills: list[Skill] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    metadata: Optional[CandidateMetadata] = None


class CandidateUpdate(BaseModel):
    """Schema for updating an existing candidate."""

    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    contact: Optional[ContactInfo] = None
    headline: Optional[str] = None
    summary: Optional[str] = None
    skills: Optional[list[Skill]] = None
    work_experience: Optional[list[WorkExperience]] = None
    education: Optional[list[Education]] = None
    certifications: Optional[list[Certification]] = None
    languages: Optional[list[Language]] = None
    status: Optional[CandidateStatus] = None
    metadata: Optional[CandidateMetadata] = None
