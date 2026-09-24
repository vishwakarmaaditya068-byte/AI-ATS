"""
Shared test fixtures for the AI-ATS test suite.

Sets environment variables before any src imports to prevent config failures,
then provides factory fixtures for dataclasses and sample Pydantic model fixtures.
"""

import os

# === Set environment BEFORE any src imports ===
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

from datetime import date, datetime
from typing import Any, Optional

import pytest
from bson import ObjectId

from src.core.matching.matching_engine import MatchingEngine, MatchResult
from src.data.models import (
    ActorInfo,
    AuditLog,
    BiasCheckResult,
    Candidate,
    CandidateMetadata,
    Certification,
    ContactInfo,
    Education,
    EducationMatch,
    EducationRequirement,
    ExperienceMatch,
    ExperienceRequirement,
    Explanation,
    ExplanationFactor,
    FileMetadata,
    Job,
    JobMetadata,
    KeywordMatch,
    Language,
    Location,
    Match,
    MatchStatus,
    ParsedContent,
    ProcessingStatus,
    RecruiterFeedback,
    ResourceInfo,
    Resume,
    ResumeFormat,
    SalaryRange,
    ScoreBreakdown,
    ScoringWeights,
    Skill,
    SkillMatch,
    SkillRequirement,
    WorkExperience,
)
from src.ml.nlp.jd_parser import JDParseResult
from src.ml.nlp.resume_parser import ResumeParseResult
from src.utils.constants import (
    AuditAction,
    CandidateStatus,
    JobStatus,
    MatchScoreLevel,
)


# ---------------------------------------------------------------------------
# Factory fixtures for ML dataclasses (src.ml.nlp, NOT src.data.models)
# ---------------------------------------------------------------------------


@pytest.fixture
def make_resume_parse_result():
    """Factory that returns a callable to build ResumeParseResult dataclasses."""

    def _factory(
        contact: Optional[dict[str, Any]] = None,
        skills: Optional[list[dict[str, Any]]] = None,
        experience: Optional[list[dict[str, Any]]] = None,
        education: Optional[list[dict[str, Any]]] = None,
        certifications: Optional[list[dict[str, Any]]] = None,
        projects: Optional[list[dict[str, Any]]] = None,
        summary: Optional[str] = None,
        languages: Optional[list[str]] = None,
        total_experience_years: float = 5.0,
        highest_education: Optional[str] = "bachelor",
        overall_confidence: float = 0.8,
        parse_quality_score: float = 0.7,
        errors: Optional[list[str]] = None,
        **kwargs,
    ) -> ResumeParseResult:
        if contact is None:
            contact = {
                "first_name": "Jane",
                "last_name": "Smith",
                "full_name": "Jane Smith",
                "email": "jane.smith@example.com",
                "phone": "+1-555-0100",
                "linkedin_url": "https://linkedin.com/in/janesmith",
                "github_url": "https://github.com/janesmith",
                "portfolio_url": None,
                "city": "San Francisco",
                "state": "CA",
                "country": "US",
                "confidence": 0.9,
            }
        if skills is None:
            skills = [
                {"name": "python", "category": "programming_languages", "proficiency": "expert", "confidence": 0.95, "source": "section"},
                {"name": "django", "category": "frameworks", "proficiency": "advanced", "confidence": 0.9, "source": "section"},
                {"name": "javascript", "category": "programming_languages", "proficiency": "intermediate", "confidence": 0.85, "source": "text"},
                {"name": "postgresql", "category": "databases", "proficiency": "advanced", "confidence": 0.9, "source": "section"},
                {"name": "docker", "category": "devops", "proficiency": "intermediate", "confidence": 0.8, "source": "text"},
                {"name": "aws", "category": "cloud_platforms", "proficiency": "intermediate", "confidence": 0.8, "source": "text"},
            ]
        if experience is None:
            experience = [
                {
                    "job_title": "Senior Software Engineer",
                    "company": "TechCorp",
                    "location": "San Francisco, CA",
                    "start_date": "2020-01-15",
                    "end_date": None,
                    "is_current": True,
                    "responsibilities": ["Led backend development", "Designed REST APIs"],
                    "achievements": ["Reduced latency by 40%"],
                    "confidence": 0.9,
                },
                {
                    "job_title": "Software Engineer",
                    "company": "StartupInc",
                    "location": "New York, NY",
                    "start_date": "2017-06-01",
                    "end_date": "2019-12-31",
                    "is_current": False,
                    "responsibilities": ["Full-stack development"],
                    "achievements": [],
                    "confidence": 0.85,
                },
            ]
        if education is None:
            education = [
                {
                    "degree": "Bachelor of Science",
                    "degree_level": "bachelor",
                    "field_of_study": "Computer Science",
                    "institution": "MIT",
                    "location": "Cambridge, MA",
                    "graduation_date": "2017-05-15",
                    "gpa": 3.8,
                    "honors": "cum laude",
                    "confidence": 0.9,
                }
            ]

        return ResumeParseResult(
            contact=contact,
            skills=skills,
            experience=experience,
            education=education,
            certifications=certifications or [],
            projects=projects or [],
            summary=summary,
            languages=languages or [],
            total_experience_years=total_experience_years,
            highest_education=highest_education,
            skill_count=len(skills),
            overall_confidence=overall_confidence,
            parse_quality_score=parse_quality_score,
            errors=errors or [],
            **kwargs,
        )

    return _factory


@pytest.fixture
def make_jd_parse_result():
    """Factory that returns a callable to build JDParseResult dataclasses."""

    def _factory(
        title: str = "Senior Software Engineer",
        company_name: str = "Acme Corp",
        required_skills: Optional[list[str]] = None,
        preferred_skills: Optional[list[str]] = None,
        experience_years_min: Optional[float] = 5.0,
        experience_years_max: Optional[float] = None,
        education_requirement: Optional[str] = "bachelor",
        responsibilities: Optional[list[str]] = None,
        qualifications: Optional[list[str]] = None,
        raw_text: str = "",
        confidence: float = 0.8,
        errors: Optional[list[str]] = None,
        **kwargs,
    ) -> JDParseResult:
        if required_skills is None:
            required_skills = ["python", "django", "postgresql", "rest apis"]
        if preferred_skills is None:
            preferred_skills = ["docker", "kubernetes", "aws"]
        if responsibilities is None:
            responsibilities = [
                "Design and develop backend services",
                "Collaborate with cross-functional teams",
                "Participate in code reviews",
            ]
        if qualifications is None:
            qualifications = [
                "5+ years of experience in software development",
                "Proficient in Python and Django",
                "Experience with PostgreSQL",
            ]
        if not raw_text:
            raw_text = (
                f"Job Title: {title}\n"
                f"Company: {company_name}\n\n"
                "Responsibilities:\n"
                + "\n".join(f"- {r}" for r in responsibilities)
                + "\n\nRequirements:\n"
                + "\n".join(f"- {q}" for q in qualifications)
            )

        return JDParseResult(
            title=title,
            company_name=company_name,
            raw_text=raw_text,
            description=raw_text[:500],
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            experience_years_min=experience_years_min,
            experience_years_max=experience_years_max,
            education_requirement=education_requirement,
            responsibilities=responsibilities,
            qualifications=qualifications,
            confidence=confidence,
            errors=errors or [],
            **kwargs,
        )

    return _factory


# ---------------------------------------------------------------------------
# Sample Pydantic model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_contact_info():
    return ContactInfo(
        email="jane.smith@example.com",
        phone="+1-555-0100",
        linkedin_url="https://linkedin.com/in/janesmith",
        city="San Francisco",
        state="CA",
        country="US",
    )


@pytest.fixture
def sample_candidate(sample_contact_info):
    return Candidate(
        first_name="Jane",
        last_name="Smith",
        contact=sample_contact_info,
        headline="Senior Software Engineer",
        summary="Experienced backend developer specializing in Python.",
        skills=[
            Skill(name="Python", category="programming_languages", proficiency_level="expert", years_of_experience=7),
            Skill(name="Django", category="frameworks", proficiency_level="advanced", years_of_experience=5),
            Skill(name="PostgreSQL", category="databases", proficiency_level="advanced"),
        ],
        work_experience=[
            WorkExperience(
                job_title="Senior Software Engineer",
                company="TechCorp",
                start_date=date(2020, 1, 15),
                end_date=None,
                is_current=True,
                responsibilities=["Led backend development"],
            ),
            WorkExperience(
                job_title="Software Engineer",
                company="StartupInc",
                start_date=date(2017, 6, 1),
                end_date=date(2019, 12, 31),
            ),
        ],
        education=[
            Education(
                degree="Bachelor",
                field_of_study="Computer Science",
                institution="MIT",
                graduation_date=date(2017, 5, 15),
                gpa=3.8,
            ),
        ],
        certifications=[
            Certification(
                name="AWS Solutions Architect",
                issuing_organization="Amazon",
                issue_date=date(2022, 1, 1),
                expiration_date=date(2025, 1, 1),
            ),
        ],
        languages=[Language(language="English", proficiency="native")],
        status=CandidateStatus.NEW,
    )


@pytest.fixture
def sample_job():
    return Job(
        title="Senior Software Engineer",
        description="We are looking for an experienced backend engineer to join our team.",
        responsibilities=["Design APIs", "Code reviews", "Mentor juniors"],
        company_name="Acme Corp",
        skill_requirements=[
            SkillRequirement(name="python", is_required=True, weight=1.0),
            SkillRequirement(name="django", is_required=True, weight=1.0),
            SkillRequirement(name="postgresql", is_required=True, weight=0.8),
            SkillRequirement(name="docker", is_required=False, weight=0.5),
            SkillRequirement(name="kubernetes", is_required=False, weight=0.5),
        ],
        education_requirement=EducationRequirement(
            minimum_degree="bachelor",
            preferred_fields=["Computer Science", "Software Engineering"],
        ),
        experience_requirement=ExperienceRequirement(minimum_years=5),
        location=Location(city="San Francisco", state="CA", country="USA"),
        status=JobStatus.OPEN,
        posted_date=datetime(2024, 1, 15),
    )


@pytest.fixture
def sample_match():
    cid = ObjectId()
    jid = ObjectId()
    return Match(
        candidate_id=cid,
        job_id=jid,
        overall_score=0.82,
        score_level=MatchScoreLevel.GOOD,
        score_breakdown=ScoreBreakdown(
            skills_score=0.85,
            skills_weight=0.35,
            skills_weighted=0.2975,
            experience_score=0.9,
            experience_weight=0.25,
            experience_weighted=0.225,
            education_score=1.0,
            education_weight=0.15,
            education_weighted=0.15,
            semantic_score=0.7,
            semantic_weight=0.20,
            semantic_weighted=0.14,
            keyword_score=0.6,
            keyword_weight=0.05,
            keyword_weighted=0.03,
        ),
        skill_matches=[
            SkillMatch(skill_name="python", required=True, candidate_has_skill=True, match_score=1.0),
            SkillMatch(skill_name="django", required=True, candidate_has_skill=True, match_score=1.0),
            SkillMatch(skill_name="postgresql", required=True, candidate_has_skill=False, match_score=0.0),
            SkillMatch(skill_name="docker", required=False, candidate_has_skill=True, match_score=1.0),
        ],
        experience_match=ExperienceMatch(
            required_years=5.0,
            candidate_years=6.5,
            years_difference=1.5,
            meets_minimum=True,
            score=1.0,
        ),
        education_match=EducationMatch(
            required_degree="bachelor",
            candidate_degree="bachelor",
            meets_requirement=True,
            score=1.0,
        ),
        status=MatchStatus.PENDING_REVIEW,
    )


@pytest.fixture
def sample_resume():
    return Resume(
        file=FileMetadata(
            original_filename="jane_smith_resume.pdf",
            stored_filename="abc123.pdf",
            file_format=ResumeFormat.PDF,
            file_size_bytes=204800,
            file_hash="sha256_placeholder",
            storage_path="resumes/abc123.pdf",
        ),
        status=ProcessingStatus.COMPLETED,
        parsed_content=ParsedContent(
            raw_text="Jane Smith\nSenior Software Engineer...",
            cleaned_text="jane smith senior software engineer",
            word_count=150,
        ),
    )


@pytest.fixture
def sample_audit_log():
    return AuditLog(
        action=AuditAction.CANDIDATE_SCORED,
        action_description="Candidate scored 0.82 for Senior Engineer",
        actor=ActorInfo(actor_type="system"),
        resource=ResourceInfo(
            resource_type="match",
            resource_id=str(ObjectId()),
        ),
        compliance_relevant=True,
    )


# ---------------------------------------------------------------------------
# Matching engine fixture (no external services)
# ---------------------------------------------------------------------------


@pytest.fixture
def matching_engine():
    """MatchingEngine with all external services disabled."""
    return MatchingEngine(
        use_semantic=False,
        use_bias_detection=False,
        use_explainability=False,
    )
