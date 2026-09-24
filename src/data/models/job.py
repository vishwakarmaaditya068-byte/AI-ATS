"""
Job posting data models for AI-ATS.

Defines the schema for job postings, including requirements,
qualifications, and matching criteria.
"""

from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.utils.constants import DEFAULT_SCORING_WEIGHTS, JobStatus

from .base import BaseDocument, EmbeddedModel, PyObjectId


class EmploymentType(str, Enum):
    """Type of employment for the position."""

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    TEMPORARY = "temporary"
    INTERNSHIP = "internship"
    FREELANCE = "freelance"


class WorkLocation(str, Enum):
    """Work location type."""

    ONSITE = "onsite"
    REMOTE = "remote"
    HYBRID = "hybrid"


class ExperienceLevel(str, Enum):
    """Required experience level for the position."""

    ENTRY = "entry"  # 0-2 years
    MID = "mid"  # 2-5 years
    SENIOR = "senior"  # 5-10 years
    LEAD = "lead"  # 10+ years
    EXECUTIVE = "executive"


class SalaryRange(EmbeddedModel):
    """Salary range for the position."""

    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    currency: str = "USD"
    pay_period: str = "yearly"  # yearly, monthly, hourly

    @field_validator("min_amount", "max_amount")
    @classmethod
    def validate_amount(cls, v: Optional[float]) -> Optional[float]:
        """Validate salary amount is positive."""
        if v is not None and v < 0:
            raise ValueError("Salary amount must be non-negative")
        return v


class Location(EmbeddedModel):
    """Physical location for the job."""

    city: Optional[str] = None
    state: Optional[str] = None
    country: str = "USA"
    postal_code: Optional[str] = None
    timezone: Optional[str] = None

    @property
    def display_string(self) -> str:
        """Get formatted location string."""
        parts = [p for p in [self.city, self.state, self.country] if p]
        return ", ".join(parts) if parts else "Location not specified"


class SkillRequirement(EmbeddedModel):
    """A required or preferred skill for the job."""

    name: str
    category: Optional[str] = None
    is_required: bool = True  # False = nice to have
    minimum_years: Optional[float] = None
    proficiency_level: Optional[str] = None  # beginner, intermediate, expert
    weight: float = 1.0  # Weight for scoring (higher = more important)

    @field_validator("name")
    @classmethod
    def normalize_skill_name(cls, v: str) -> str:
        """Normalize skill names to lowercase."""
        return v.strip().lower()


class EducationRequirement(EmbeddedModel):
    """Education requirement for the job."""

    minimum_degree: str  # e.g., "bachelor", "master"
    preferred_fields: list[str] = Field(default_factory=list)
    is_required: bool = True
    equivalent_experience_years: Optional[int] = None  # Years of experience that can substitute


class ExperienceRequirement(EmbeddedModel):
    """Experience requirement for the job."""

    minimum_years: float = 0
    maximum_years: Optional[float] = None  # None = no maximum
    required_titles: list[str] = Field(default_factory=list)  # Previous job titles
    required_industries: list[str] = Field(default_factory=list)


class ScoringWeights(EmbeddedModel):
    """Custom scoring weights for candidate matching."""

    skills_match: float = Field(default=0.35, ge=0, le=1)
    experience_match: float = Field(default=0.25, ge=0, le=1)
    education_match: float = Field(default=0.15, ge=0, le=1)
    semantic_similarity: float = Field(default=0.20, ge=0, le=1)
    keyword_match: float = Field(default=0.05, ge=0, le=1)

    @classmethod
    def from_defaults(cls) -> "ScoringWeights":
        """Create scoring weights from default constants."""
        return cls(**DEFAULT_SCORING_WEIGHTS)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "skills_match": self.skills_match,
            "experience_match": self.experience_match,
            "education_match": self.education_match,
            "semantic_similarity": self.semantic_similarity,
            "keyword_match": self.keyword_match,
        }

    @property
    def total_weight(self) -> float:
        """Calculate sum of all weights."""
        return (
            self.skills_match
            + self.experience_match
            + self.education_match
            + self.semantic_similarity
            + self.keyword_match
        )


class JobMetadata(EmbeddedModel):
    """Metadata about the job posting."""

    department: Optional[str] = None
    team: Optional[str] = None
    hiring_manager_id: Optional[str] = None
    recruiter_id: Optional[str] = None
    external_job_id: Optional[str] = None  # ID from external job board
    source: Optional[str] = None  # Where the job was posted
    tags: list[str] = Field(default_factory=list)
    views_count: int = 0
    applications_count: int = 0


class Job(BaseDocument):
    """
    Main job posting model.

    Represents a job opening with all requirements and matching criteria.
    """

    # Basic Information
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10)
    responsibilities: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)

    # Company Information
    company_name: str
    company_description: Optional[str] = None

    # Employment Details
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    work_location: WorkLocation = WorkLocation.ONSITE
    location: Location = Field(default_factory=Location)
    salary: Optional[SalaryRange] = None
    experience_level: ExperienceLevel = ExperienceLevel.MID

    # Requirements
    skill_requirements: list[SkillRequirement] = Field(default_factory=list)
    education_requirement: Optional[EducationRequirement] = None
    experience_requirement: Optional[ExperienceRequirement] = None

    # Additional Requirements
    certifications_required: list[str] = Field(default_factory=list)
    languages_required: list[str] = Field(default_factory=list)

    # Status & Dates
    status: JobStatus = JobStatus.DRAFT
    posted_date: Optional[datetime] = None
    closing_date: Optional[date] = None
    target_hire_date: Optional[date] = None
    positions_available: int = 1

    # Scoring Configuration
    scoring_weights: ScoringWeights = Field(default_factory=ScoringWeights.from_defaults)

    # Metadata
    metadata: JobMetadata = Field(default_factory=JobMetadata)

    # Vector embedding for semantic matching
    embedding_id: Optional[str] = None

    @property
    def required_skills(self) -> list[str]:
        """Get list of required skill names."""
        return [s.name for s in self.skill_requirements if s.is_required]

    @property
    def preferred_skills(self) -> list[str]:
        """Get list of preferred (nice-to-have) skill names."""
        return [s.name for s in self.skill_requirements if not s.is_required]

    @property
    def all_skills(self) -> list[str]:
        """Get all skill names."""
        return [s.name for s in self.skill_requirements]

    @property
    def is_active(self) -> bool:
        """Check if job is currently accepting applications."""
        if self.status != JobStatus.OPEN:
            return False
        if self.closing_date and self.closing_date < date.today():
            return False
        return True

    @property
    def days_open(self) -> Optional[int]:
        """Calculate days since job was posted."""
        if not self.posted_date:
            return None
        posted = self.posted_date
        if posted.tzinfo is None:
            posted = posted.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - posted
        return delta.days

    def publish(self) -> None:
        """Publish the job posting."""
        self.status = JobStatus.OPEN
        self.posted_date = datetime.now(timezone.utc)

    def close(self) -> None:
        """Close the job posting."""
        self.status = JobStatus.CLOSED

    class Settings:
        """MongoDB collection settings."""

        name = "jobs"
        indexes = [
            "status",
            "company_name",
            "employment_type",
            "experience_level",
            "metadata.tags",
            "posted_date",
            "created_at",
        ]


class JobCreate(BaseModel):
    """Schema for creating a new job posting."""

    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=10)
    responsibilities: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    company_name: str
    company_description: Optional[str] = None
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    work_location: WorkLocation = WorkLocation.ONSITE
    location: Optional[Location] = None
    salary: Optional[SalaryRange] = None
    experience_level: ExperienceLevel = ExperienceLevel.MID
    skill_requirements: list[SkillRequirement] = Field(default_factory=list)
    education_requirement: Optional[EducationRequirement] = None
    experience_requirement: Optional[ExperienceRequirement] = None
    certifications_required: list[str] = Field(default_factory=list)
    languages_required: list[str] = Field(default_factory=list)
    closing_date: Optional[date] = None
    target_hire_date: Optional[date] = None
    positions_available: int = 1
    scoring_weights: Optional[ScoringWeights] = None
    metadata: Optional[JobMetadata] = None


class JobUpdate(BaseModel):
    """Schema for updating an existing job posting."""

    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, min_length=10)
    responsibilities: Optional[list[str]] = None
    benefits: Optional[list[str]] = None
    company_name: Optional[str] = None
    company_description: Optional[str] = None
    employment_type: Optional[EmploymentType] = None
    work_location: Optional[WorkLocation] = None
    location: Optional[Location] = None
    salary: Optional[SalaryRange] = None
    experience_level: Optional[ExperienceLevel] = None
    skill_requirements: Optional[list[SkillRequirement]] = None
    education_requirement: Optional[EducationRequirement] = None
    experience_requirement: Optional[ExperienceRequirement] = None
    certifications_required: Optional[list[str]] = None
    languages_required: Optional[list[str]] = None
    status: Optional[JobStatus] = None
    closing_date: Optional[date] = None
    target_hire_date: Optional[date] = None
    positions_available: Optional[int] = None
    scoring_weights: Optional[ScoringWeights] = None
    metadata: Optional[JobMetadata] = None
