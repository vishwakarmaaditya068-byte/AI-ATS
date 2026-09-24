"""
Pydantic data models and schemas for AI-ATS.

This module provides all data models used throughout the application,
including database documents, embedded models, and API schemas.
"""

# Base models
from .base import BaseDocument, EmbeddedModel, PyObjectId, TimestampMixin

# Candidate models
from .candidate import (
    Candidate,
    CandidateCreate,
    CandidateMetadata,
    CandidateUpdate,
    Certification,
    ContactInfo,
    Education,
    Language,
    Skill,
    WorkExperience,
)

# Resume models
from .resume import (
    ExtractedEntity,
    FileMetadata,
    ParsedContent,
    ParsedSection,
    ProcessingError,
    ProcessingMetrics,
    ProcessingStatus,
    Resume,
    ResumeFormat,
    ResumeParseResult,
    ResumeUpload,
)

# Job models
from .job import (
    EducationRequirement,
    EmploymentType,
    ExperienceLevel,
    ExperienceRequirement,
    Job,
    JobCreate,
    JobMetadata,
    JobUpdate,
    Location,
    SalaryRange,
    ScoringWeights,
    SkillRequirement,
    WorkLocation,
)

# Match models
from .match import (
    BiasCheckResult,
    EducationMatch,
    Explanation,
    ExplanationFactor,
    ExperienceMatch,
    KeywordMatch,
    Match,
    MatchCreate,
    MatchStatus,
    MatchSummary,
    MatchUpdate,
    RecruiterFeedback,
    ScoreBreakdown,
    SemanticMatch,
    SkillMatch,
)

# Audit models
from .audit import (
    ActorInfo,
    AIDecisionInfo,
    AuditLog,
    AuditLogCreate,
    AuditLogQuery,
    AuditSummary,
    BiasAuditInfo,
    ChangeRecord,
    ResourceInfo,
    create_bias_detected_audit,
    create_candidate_added_audit,
    create_candidate_scored_audit,
    create_manual_override_audit,
)

__all__ = [
    # Base
    "BaseDocument",
    "EmbeddedModel",
    "PyObjectId",
    "TimestampMixin",
    # Candidate
    "Candidate",
    "CandidateCreate",
    "CandidateMetadata",
    "CandidateUpdate",
    "Certification",
    "ContactInfo",
    "Education",
    "Language",
    "Skill",
    "WorkExperience",
    # Resume
    "ExtractedEntity",
    "FileMetadata",
    "ParsedContent",
    "ParsedSection",
    "ProcessingError",
    "ProcessingMetrics",
    "ProcessingStatus",
    "Resume",
    "ResumeFormat",
    "ResumeParseResult",
    "ResumeUpload",
    # Job
    "EducationRequirement",
    "EmploymentType",
    "ExperienceLevel",
    "ExperienceRequirement",
    "Job",
    "JobCreate",
    "JobMetadata",
    "JobUpdate",
    "Location",
    "SalaryRange",
    "ScoringWeights",
    "SkillRequirement",
    "WorkLocation",
    # Match
    "BiasCheckResult",
    "EducationMatch",
    "Explanation",
    "ExplanationFactor",
    "ExperienceMatch",
    "KeywordMatch",
    "Match",
    "MatchCreate",
    "MatchStatus",
    "MatchSummary",
    "MatchUpdate",
    "RecruiterFeedback",
    "ScoreBreakdown",
    "SemanticMatch",
    "SkillMatch",
    # Audit
    "ActorInfo",
    "AIDecisionInfo",
    "AuditLog",
    "AuditLogCreate",
    "AuditLogQuery",
    "AuditSummary",
    "BiasAuditInfo",
    "ChangeRecord",
    "ResourceInfo",
    "create_bias_detected_audit",
    "create_candidate_added_audit",
    "create_candidate_scored_audit",
    "create_manual_override_audit",
]
