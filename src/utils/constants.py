"""
Application-wide constants for AI-ATS.

This module contains all constant values used throughout the application.
Modify these values to customize behavior without changing code logic.
"""

from enum import Enum, auto
from typing import Final


# =============================================================================
# Application Constants
# =============================================================================

APP_NAME: Final[str] = "AI-ATS"
APP_DISPLAY_NAME: Final[str] = "AI-Powered Applicant Tracking System"
VERSION: Final[str] = "0.1.0"


# =============================================================================
# File Types
# =============================================================================

SUPPORTED_RESUME_FORMATS: Final[tuple[str, ...]] = (
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".rtf",
)

SUPPORTED_JOB_FORMATS: Final[tuple[str, ...]] = (
    ".pdf",
    ".docx",
    ".txt",
    ".json",
)


# =============================================================================
# NLP Constants
# =============================================================================

# Common skill categories for extraction
SKILL_CATEGORIES: Final[dict[str, list[str]]] = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql",
    ],
    "frameworks": [
        "react", "angular", "vue", "django", "flask", "fastapi", "spring",
        "node.js", "express", ".net", "rails", "laravel", "tensorflow",
        "pytorch", "keras", "scikit-learn",
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
        "oracle", "sql server", "sqlite", "dynamodb", "firebase",
    ],
    "cloud_platforms": [
        "aws", "azure", "gcp", "google cloud", "heroku", "digitalocean",
        "kubernetes", "docker", "terraform", "ansible",
    ],
    "soft_skills": [
        "leadership", "communication", "teamwork", "problem-solving",
        "analytical", "creative", "adaptable", "organized", "detail-oriented",
    ],
}

# Education degree levels (ordered by level)
EDUCATION_LEVELS: Final[dict[str, int]] = {
    "high school": 1,
    "diploma": 2,
    "associate": 3,
    "bachelor": 4,
    "master": 5,
    "mba": 5,
    "phd": 6,
    "doctorate": 6,
}


# =============================================================================
# Scoring Constants
# =============================================================================

# Default weights for candidate scoring
DEFAULT_SCORING_WEIGHTS: Final[dict[str, float]] = {
    "skills_match": 0.35,
    "experience_match": 0.25,
    "education_match": 0.15,
    "semantic_similarity": 0.20,
    "keyword_match": 0.05,
}

# Score thresholds
SCORE_THRESHOLDS: Final[dict[str, float]] = {
    "excellent": 0.85,
    "good": 0.70,
    "fair": 0.50,
    "poor": 0.30,
}

# Skill embedding similarity thresholds
SKILL_EXACT_THRESHOLD: Final[float] = 0.92
"""Cosine similarity >= this value -> treat as exact match (score = 1.0)."""

SKILL_STRONG_PARTIAL_THRESHOLD: Final[float] = 0.75
"""Cosine similarity >= this value -> strong partial match (score = 0.8)."""

SKILL_WEAK_PARTIAL_THRESHOLD: Final[float] = 0.60
"""Cosine similarity >= this value -> weak partial match (score = 0.5)."""

EXP_RELEVANCE_TITLE_THRESHOLD: Final[float] = 0.40
"""Cosine similarity >= this value -> experience entry title added to relevant_titles_matched."""

EDU_FIELD_MATCH_THRESHOLD: Final[float] = 0.50
"""Cosine similarity >= this value -> EducationMatch.field_match = True."""

EDU_FIELD_WEIGHT: Final[float] = 0.40
"""Weight of field-of-study similarity in the combined education score (0-1).
The remaining (1 - EDU_FIELD_WEIGHT) is the degree-level score weight."""


# =============================================================================
# Ethical AI Constants
# =============================================================================

# Protected attributes for bias detection
PROTECTED_ATTRIBUTES: Final[list[str]] = [
    "gender",
    "age",
    "race",
    "ethnicity",
    "religion",
    "disability",
    "nationality",
    "marital_status",
]

# Fairness thresholds
FAIRNESS_THRESHOLDS: Final[dict[str, float]] = {
    "demographic_parity_difference": 0.1,
    "equalized_odds_difference": 0.1,
    "disparate_impact_ratio": 0.8,
}

# Minimum number of candidates required in a demographic group before that
# group is included in fairness metric calculations.  Groups smaller than this
# produce statistically unreliable rates (e.g. one extra selection/rejection
# swings the positive rate by 20–50 %), which could mask or falsely trigger
# fairness violations.  Configurable via FairnessCalculator(min_group_size=N).
FAIRNESS_MIN_GROUP_SIZE: Final[int] = 5


# =============================================================================
# Enums
# =============================================================================


class CandidateStatus(str, Enum):
    """Status of a candidate in the pipeline."""

    NEW = "new"
    SCREENING = "screening"
    SHORTLISTED = "shortlisted"
    INTERVIEWING = "interview"
    OFFERED = "offer"
    HIRED = "hired"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class JobStatus(str, Enum):
    """Status of a job posting."""

    DRAFT = "draft"
    OPEN = "open"
    PAUSED = "paused"
    CLOSED = "closed"
    FILLED = "filled"


class MatchScoreLevel(Enum):
    """Categorical levels for match scores."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

    @classmethod
    def from_score(cls, score: float) -> "MatchScoreLevel":
        """Convert a numeric score to a level."""
        if score >= SCORE_THRESHOLDS["excellent"]:
            return cls.EXCELLENT
        elif score >= SCORE_THRESHOLDS["good"]:
            return cls.GOOD
        elif score >= SCORE_THRESHOLDS["fair"]:
            return cls.FAIR
        return cls.POOR


class AuditAction(str, Enum):
    """Types of actions that can be audited."""

    CANDIDATE_ADDED = "candidate_added"
    CANDIDATE_SCORED = "candidate_scored"
    CANDIDATE_RANKED = "candidate_ranked"
    CANDIDATE_STATUS_CHANGED = "candidate_status_changed"
    JOB_CREATED = "job_created"
    JOB_CLOSED = "job_closed"
    BIAS_DETECTED = "bias_detected"
    MANUAL_OVERRIDE = "manual_override"
    REPORT_GENERATED = "report_generated"


# =============================================================================
# UI Constants
# =============================================================================

# Color palette — mutable so ThemeManager can swap dark ↔ light in-place.
# Default is dark (VSCode Dark+). Do not annotate as Final.
COLORS: dict[str, str] = {
    # ── Brand / Interactive ────────────────────────────────────────────────────
    "primary":              "#007ACC",   # VSCode blue
    "primary_dark":         "#005F9E",
    "primary_glow":         "#1A3F5C",
    "accent":               "#C586C0",   # VSCode purple
    "accent_dark":          "#9A4090",
    "accent_dim":           "#2D1F2D",
    "secondary":            "#858585",

    # ── Status ─────────────────────────────────────────────────────────────────
    "success":              "#4EC9B0",   # VSCode teal
    "success_dark":         "#3AA890",
    "success_dim":          "#1A3028",
    "warning":              "#CCA700",
    "warning_dim":          "#2D2800",
    "error":                "#F14C4C",
    "error_dim":            "#3D1010",
    "info":                 "#9CDCFE",   # VSCode light-blue

    # ── Surfaces (L0 → L3) — VSCode elevation model ───────────────────────────
    "background":           "#1E1E1E",   # Editor / main content
    "surface":              "#252526",   # Sidebar / panel
    "surface_elevated":     "#2D2D30",   # Input, tab bar, card bg
    "surface_overlay":      "#3C3C3C",   # Hover, dropdown, overlay

    # ── Borders ────────────────────────────────────────────────────────────────
    "border_subtle":        "#474747",
    "border_muted":         "#5A5A5A",

    # ── Typography ─────────────────────────────────────────────────────────────
    "text_primary":         "#D4D4D4",   # VSCode editor foreground
    "text_secondary":       "#858585",
    "text_tertiary":        "#4A4A4A",
    "text_on_primary":      "#FFFFFF",

    # ── VSCode chrome tokens ───────────────────────────────────────────────────
    "statusbar_bg":         "#007ACC",
    "statusbar_fg":         "#FFFFFF",
    "activitybar_bg":       "#333333",
    "activitybar_active":   "#FFFFFF",
    "activitybar_inactive": "#858585",
}

# ── Shadow definitions ─────────────────────────────────────────────────────────
# Each entry: (blur_radius, x_offset, y_offset, alpha) for QGraphicsDropShadowEffect
SHADOWS: Final[dict[str, tuple[int, int, int, int]]] = {
    "sm":  (8,  0, 1, 40),   # Subtle card lift
    "md":  (16, 0, 4, 60),   # Standard card elevation
    "lg":  (28, 0, 8, 80),   # Modal / floating panel
    "xl":  (48, 0, 16, 100), # Full overlay shadow
}

# Dashboard refresh intervals (in milliseconds)
REFRESH_INTERVALS: Final[dict[str, int]] = {
    "statistics": 30000,  # 30 seconds
    "candidate_list": 60000,  # 1 minute
    "notifications": 10000,  # 10 seconds
}
