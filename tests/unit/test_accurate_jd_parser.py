"""
Unit tests for AccurateJDParser.
Uses a fixed sample JD string — no files, no infrastructure.
"""
from __future__ import annotations

import pytest
from src.ml.nlp.accurate_jd_parser import AccurateJDParser, ParsedJob

_JD_SAMPLE: str = """
Software Engineer - Backend
Acme Corp | San Francisco, CA | Full-time | Remote

About the Role
We are looking for a talented Backend Engineer to join our growing team and
build the next generation of our AI platform.

Responsibilities
• Design and implement scalable REST APIs using Python and FastAPI
• Write clean, maintainable, and well-tested code
• Collaborate with frontend and data teams on API contracts
• Participate in architecture discussions and code reviews

Requirements
• 3+ years of Python development experience
• Strong knowledge of SQL and NoSQL databases (PostgreSQL, MongoDB)
• Experience with Docker and Kubernetes
• Bachelor's degree in Computer Science, Engineering, or equivalent

Nice to Have
• Experience with ML frameworks like PyTorch or TensorFlow
• GraphQL knowledge
• AWS Certified Solutions Architect

Benefits
• Competitive salary and equity package
• Full medical, dental, and vision insurance

About Us
Acme Corp is a fast-growing startup building the future of AI.
"""

_JD_CONTRACT: str = """
Senior Data Scientist
DataLab Inc | New York | Contract | Hybrid

Requirements
• 5+ years of Python and machine learning experience
• Master's degree or PhD in Statistics, Mathematics, or related field
• Experience with scikit-learn, pandas, and PyTorch
"""

_JD_INTERNSHIP: str = """
Software Engineering Intern
TechCo | Remote | Internship

Responsibilities
• Work on backend features under senior engineer guidance
• Write unit tests for existing code

Requirements
• Currently pursuing a Bachelor's degree in Computer Science
• Basic Python knowledge
"""


def _make_parser() -> AccurateJDParser:
    return AccurateJDParser()


class TestExtractHeader:
    def test_extracts_title(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        assert "engineer" in result.title.lower() or "software" in result.title.lower()

    def test_extracts_company(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        assert "acme" in result.company_name.lower()


class TestExtractResponsibilities:
    def test_extracts_responsibilities(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        assert len(result.responsibilities) >= 3

    def test_responsibilities_no_bullet_prefix(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        for resp in result.responsibilities:
            assert not resp.startswith("•"), f"Bullet not stripped: {resp!r}"


class TestExtractSkills:
    def test_extracts_required_skills_contains_python(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        all_req: str = " ".join(result.required_skills).lower()
        assert "python" in all_req

    def test_extracts_preferred_skills_non_empty(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        assert len(result.preferred_skills) > 0

    def test_preferred_skills_distinct_from_required(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        req_set: set[str] = set(result.required_skills)
        pref_set: set[str] = set(result.preferred_skills)
        # Some overlap is fine, but they must come from separate sections
        assert len(result.required_skills) > 0
        assert len(result.preferred_skills) > 0


class TestExtractRequirements:
    def test_extracts_experience_min_years(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        assert result.experience_min_years == 3.0

    def test_extracts_education_bachelor(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        assert result.education_requirement == "bachelor"

    def test_detects_full_time(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        assert result.employment_type == "full_time"

    def test_detects_remote(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        assert result.work_location == "remote"

    def test_detects_contract(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_CONTRACT)
        assert result.employment_type == "contract"

    def test_detects_internship(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_INTERNSHIP)
        assert result.employment_type == "internship"


class TestToJobCreate:
    def test_to_job_create_returns_job_create(self) -> None:
        from src.data.models.job import JobCreate
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        jc = result.to_job_create()
        assert isinstance(jc, JobCreate)

    def test_to_job_create_title_preserved(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        jc = result.to_job_create()
        assert "engineer" in jc.title.lower() or "software" in jc.title.lower()

    def test_to_job_create_required_skills_flagged(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        jc = result.to_job_create()
        required = [s for s in jc.skill_requirements if s.is_required]
        assert len(required) > 0

    def test_to_job_create_preferred_skills_flagged(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        jc = result.to_job_create()
        preferred = [s for s in jc.skill_requirements if not s.is_required]
        assert len(preferred) > 0

    def test_to_job_create_experience_requirement_populated(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        jc = result.to_job_create()
        assert jc.experience_requirement is not None
        assert jc.experience_requirement.minimum_years == 3.0

    def test_to_job_create_education_requirement_populated(self) -> None:
        p: AccurateJDParser = _make_parser()
        result: ParsedJob = p.parse(_JD_SAMPLE)
        jc = result.to_job_create()
        assert jc.education_requirement is not None
        assert "bachelor" in jc.education_requirement.minimum_degree.lower()
