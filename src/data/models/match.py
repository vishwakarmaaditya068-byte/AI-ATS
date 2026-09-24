"""
Match and scoring data models for AI-ATS.

Defines the schema for candidate-job matches, scoring results,
and explainability components.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from src.utils.constants import MatchScoreLevel

from .base import BaseDocument, EmbeddedModel, PyObjectId


class MatchStatus(str, Enum):
    """Status of a match in the review pipeline."""

    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"
    SHORTLISTED = "shortlisted"
    REJECTED = "rejected"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    OFFER_EXTENDED = "offer_extended"


class SkillMatch(EmbeddedModel):
    """Detailed match result for a single skill."""

    skill_name: str
    required: bool = True
    candidate_has_skill: bool = False
    candidate_proficiency: Optional[str] = None
    candidate_years: Optional[float] = None
    required_proficiency: Optional[str] = None
    required_years: Optional[float] = None
    match_score: float = 0.0  # 0-1 score for this skill
    partial_match: bool = False  # True if related skill found
    related_skill: Optional[str] = None  # If partial match, what skill matched

    @field_validator("match_score", mode="before")
    @classmethod
    def clamp_match_score(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class ExperienceMatch(EmbeddedModel):
    """Detailed match result for experience requirements."""

    required_years: float
    candidate_years: float
    years_difference: float
    meets_minimum: bool
    relevant_titles_matched: list[str] = Field(default_factory=list)
    relevant_industries_matched: list[str] = Field(default_factory=list)
    score: float = 0.0  # 0-1 score

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class EducationMatch(EmbeddedModel):
    """Detailed match result for education requirements."""

    required_degree: Optional[str] = None
    candidate_degree: Optional[str] = None
    meets_requirement: bool = False
    field_match: bool = False  # Did the field of study match
    equivalent_experience_used: bool = False
    score: float = 0.0  # 0-1 score

    @field_validator("score", mode="before")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class SemanticMatch(EmbeddedModel):
    """Results from semantic/embedding-based matching."""

    overall_similarity: float = 0.0  # Cosine similarity 0-1
    summary_similarity: float = 0.0  # Candidate summary vs job description
    skills_similarity: float = 0.0  # Skills section similarity
    experience_similarity: float = 0.0  # Experience section similarity
    weighted_similarity: float = 0.0  # Weighted combination of all four section scores
    model_used: str = "all-MiniLM-L6-v2"

    @field_validator(
        "overall_similarity", "summary_similarity",
        "skills_similarity", "experience_similarity",
        "weighted_similarity",
        mode="before",
    )
    @classmethod
    def clamp_similarity(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class KeywordMatch(EmbeddedModel):
    """Results from keyword-based matching."""

    total_keywords: int = 0
    matched_keywords: int = 0
    match_percentage: float = 0.0
    matched_terms: list[str] = Field(default_factory=list)
    missing_terms: list[str] = Field(default_factory=list)


class ScoreBreakdown(EmbeddedModel):
    """Breakdown of the overall match score by component."""

    skills_score: float = 0.0
    skills_weight: float = 0.35
    skills_weighted: float = 0.0

    experience_score: float = 0.0
    experience_weight: float = 0.25
    experience_weighted: float = 0.0

    education_score: float = 0.0
    education_weight: float = 0.15
    education_weighted: float = 0.0

    semantic_score: float = 0.0
    semantic_weight: float = 0.20
    semantic_weighted: float = 0.0

    keyword_score: float = 0.0
    keyword_weight: float = 0.05
    keyword_weighted: float = 0.0

    @field_validator(
        "skills_score", "experience_score", "education_score",
        "semantic_score", "keyword_score",
        mode="before",
    )
    @classmethod
    def clamp_component_score(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @property
    def total_score(self) -> float:
        """Calculate total weighted score."""
        return (
            self.skills_weighted
            + self.experience_weighted
            + self.education_weighted
            + self.semantic_weighted
            + self.keyword_weighted
        )


class ExplanationFactor(EmbeddedModel):
    """A single factor contributing to the match explanation."""

    factor_name: str
    factor_type: str  # "positive", "negative", "neutral"
    description: str
    impact_score: float  # How much this factor affected the score
    evidence: Optional[str] = None  # Supporting evidence from resume/job


class Explanation(EmbeddedModel):
    """Human-readable explanation of the match score."""

    summary: str  # One-sentence summary
    factors: list[ExplanationFactor] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    # LIME/SHAP explanation data
    lime_explanation: Optional[dict[str, Any]] = None
    shap_values: Optional[dict[str, float]] = None


class BiasCheckResult(EmbeddedModel):
    """Results of bias detection check for this match."""

    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_version: str = "1.0"

    # Bias flags
    potential_bias_detected: bool = False
    bias_type: Optional[str] = None  # e.g., "gender", "age"
    bias_confidence: float = 0.0
    bias_description: Optional[str] = None

    # Protected attribute analysis
    protected_attributes_found: list[str] = Field(default_factory=list)
    mitigation_applied: bool = False
    mitigation_description: Optional[str] = None


class RecruiterFeedback(EmbeddedModel):
    """Feedback from recruiter on the match."""

    recruiter_id: str
    rating: Optional[int] = None  # 1-5 rating
    comments: Optional[str] = None
    decision: Optional[str] = None  # shortlist, reject, etc.
    feedback_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Match(BaseDocument):
    """
    Main match document representing a candidate-job match result.

    Stores scoring results, detailed breakdowns, and explanations.
    """

    # References
    candidate_id: PyObjectId
    job_id: PyObjectId

    # Overall Score
    overall_score: float = Field(0.0, ge=0, le=1)
    score_level: MatchScoreLevel = MatchScoreLevel.POOR
    rank: Optional[int] = None  # Rank among all matches for this job

    # Score Components
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)

    # Detailed Match Results
    skill_matches: list[SkillMatch] = Field(default_factory=list)
    experience_match: Optional[ExperienceMatch] = None
    education_match: Optional[EducationMatch] = None
    semantic_match: Optional[SemanticMatch] = None
    keyword_match: Optional[KeywordMatch] = None

    # Explainability
    explanation: Optional[Explanation] = None

    # Bias Detection
    bias_check: Optional[BiasCheckResult] = None

    # Status
    status: MatchStatus = MatchStatus.PENDING_REVIEW

    # Recruiter Interaction
    feedback: list[RecruiterFeedback] = Field(default_factory=list)
    manual_score_override: Optional[float] = None
    override_reason: Optional[str] = None

    # Processing Info
    scoring_model_version: str = "1.0"
    scored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("overall_score")
    @classmethod
    def set_score_level(cls, v: float, info) -> float:
        """Score level is set separately after validation."""
        return v

    def calculate_score_level(self) -> MatchScoreLevel:
        """Calculate and update score level from overall score."""
        self.score_level = MatchScoreLevel.from_score(self.overall_score)
        return self.score_level

    @property
    def effective_score(self) -> float:
        """Get the effective score (manual override if present)."""
        return self.manual_score_override if self.manual_score_override is not None else self.overall_score

    @property
    def is_shortlisted(self) -> bool:
        """Check if match is shortlisted."""
        return self.status == MatchStatus.SHORTLISTED

    @property
    def skills_match_percentage(self) -> float:
        """Calculate percentage of required skills matched."""
        required_skills = [s for s in self.skill_matches if s.required]
        if not required_skills:
            return 100.0
        matched = sum(1 for s in required_skills if s.candidate_has_skill)
        return (matched / len(required_skills)) * 100

    def add_feedback(
        self,
        recruiter_id: str,
        rating: Optional[int] = None,
        comments: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> None:
        """Add recruiter feedback to the match."""
        self.feedback.append(
            RecruiterFeedback(
                recruiter_id=recruiter_id,
                rating=rating,
                comments=comments,
                decision=decision,
            )
        )

    def override_score(self, new_score: float, reason: str) -> None:
        """Manually override the AI-generated score."""
        self.manual_score_override = new_score
        self.override_reason = reason
        self.calculate_score_level()

    class Settings:
        """MongoDB collection settings."""

        name = "matches"
        indexes = [
            [("candidate_id", 1), ("job_id", 1)],  # Compound unique index
            "job_id",
            "overall_score",
            "score_level",
            "status",
            "rank",
            "created_at",
        ]


class MatchCreate(BaseModel):
    """Schema for creating a new match (typically done by the system)."""

    candidate_id: str
    job_id: str


class MatchUpdate(BaseModel):
    """Schema for updating a match (recruiter actions)."""

    status: Optional[MatchStatus] = None
    manual_score_override: Optional[float] = Field(None, ge=0, le=1)
    override_reason: Optional[str] = None


class MatchSummary(BaseModel):
    """Summary view of a match for list displays."""

    match_id: str
    candidate_id: str
    candidate_name: str
    job_id: str
    job_title: str
    overall_score: float
    score_level: MatchScoreLevel
    status: MatchStatus
    skills_match_percentage: float
    top_strengths: list[str] = Field(default_factory=list)
    key_gaps: list[str] = Field(default_factory=list)
    created_at: datetime
