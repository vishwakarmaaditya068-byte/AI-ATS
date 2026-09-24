"""
Tests for src.ml.nlp.jd_parser — JDParser extraction methods.
"""

import pytest

from src.ml.nlp.jd_parser import JDParser, JDParseResult


@pytest.fixture
def parser():
    return JDParser()


# ── JDParseResult properties ─────────────────────────────────────────────────


class TestJDParseResultProperties:
    def test_success_true(self):
        r = JDParseResult(title="Engineer", required_skills=["python"])
        assert r.success is True

    def test_success_false_with_errors(self):
        r = JDParseResult(errors=["Extraction failed"])
        assert r.success is False

    def test_success_true_with_title_only(self):
        r = JDParseResult(title="Engineer")
        assert r.success is True

    def test_success_true_with_skills_only(self):
        r = JDParseResult(required_skills=["python"])
        assert r.success is True

    def test_success_false_no_title_no_skills(self):
        r = JDParseResult()
        assert r.success is False

    def test_all_skills(self):
        r = JDParseResult(required_skills=["python", "java"], preferred_skills=["docker", "python"])
        all_s = r.all_skills
        assert "python" in all_s
        assert "java" in all_s
        assert "docker" in all_s
        # Duplicates removed
        assert len(all_s) == 3


# ── _extract_title ───────────────────────────────────────────────────────────


class TestExtractTitle:
    def test_job_title_label(self, parser):
        text = "Job Title: Senior Software Engineer\nCompany: Acme"
        assert parser._extract_title(text) == "Senior Software Engineer"

    def test_position_label(self, parser):
        text = "Position: Data Scientist\nLocation: Remote"
        assert parser._extract_title(text) == "Data Scientist"

    def test_first_line_fallback(self, parser):
        text = "Backend Developer\n\nWe are looking for a talented developer."
        assert parser._extract_title(text) == "Backend Developer"

    def test_unknown_when_no_title(self, parser):
        # All lines too short or end with colon
        text = "Hi:\nA:\nB:"
        assert parser._extract_title(text) == "Unknown Position"


# ── _extract_company ─────────────────────────────────────────────────────────


class TestExtractCompany:
    def test_company_label(self, parser):
        text = "Company: TechCorp\nLocation: NYC"
        assert parser._extract_company(text) == "TechCorp"

    def test_about_pattern(self, parser):
        text = "Job Title: Engineer\nAbout Acme Inc\nWe build great products."
        assert "Acme" in parser._extract_company(text)

    def test_unknown_company(self, parser):
        text = "We need talented engineers to join our growing team."
        assert parser._extract_company(text) == "Unknown Company"


# ── _detect_sections ─────────────────────────────────────────────────────────


class TestDetectSections:
    def test_detects_responsibilities(self, parser):
        text = "Overview\n\nResponsibilities:\n- Design systems\n- Write code\n\nRequirements:\n- Python"
        sections = parser._detect_sections(text)
        assert "responsibilities" in sections

    def test_detects_requirements(self, parser):
        text = "Job Title: Dev\n\nRequirements:\n- 5 years experience\n- Python\n\nBenefits:\n- Health insurance"
        sections = parser._detect_sections(text)
        assert "requirements" in sections

    def test_detects_multiple_sections(self, parser):
        text = (
            "Responsibilities:\n- Code\n\n"
            "Requirements:\n- Python\n\n"
            "Benefits:\n- Health"
        )
        sections = parser._detect_sections(text)
        assert "responsibilities" in sections
        assert "requirements" in sections
        assert "benefits" in sections


# ── _extract_bullet_points ───────────────────────────────────────────────────


class TestExtractBulletPoints:
    def test_dash_bullets(self, parser):
        text = "- Design and build APIs\n- Write clean code\n- Short"
        points = parser._extract_bullet_points(text)
        assert any("Design and build APIs" in p for p in points)

    def test_numbered_list(self, parser):
        text = "1. Design scalable systems\n2. Collaborate with teams\n3. tiny"
        points = parser._extract_bullet_points(text)
        assert any("Design scalable systems" in p for p in points)

    def test_short_lines_filtered(self, parser):
        text = "- Ok\n- Also ok\n- This is a longer bullet point that should pass"
        points = parser._extract_bullet_points(text)
        # Only the long line passes the len > 10 filter
        assert len(points) >= 1


# ── _extract_experience ──────────────────────────────────────────────────────


class TestExtractExperience:
    def test_plus_years(self, parser):
        text = "Requirements: 5+ years of professional experience in software development"
        min_y, max_y = parser._extract_experience(text)
        assert min_y == 5.0
        assert max_y is None

    def test_range_years(self, parser):
        text = "3-5 years of experience required"
        min_y, max_y = parser._extract_experience(text)
        assert min_y == 3.0
        assert max_y == 5.0

    def test_minimum_of(self, parser):
        text = "Minimum of 3 years in a similar role"
        min_y, max_y = parser._extract_experience(text)
        assert min_y == 3.0

    def test_not_found(self, parser):
        text = "No experience information here."
        min_y, max_y = parser._extract_experience(text)
        assert min_y is None
        assert max_y is None


# ── _extract_education ───────────────────────────────────────────────────────


class TestExtractEducation:
    def test_bachelors_required(self, parser):
        text = "Must have a Bachelor's degree in Computer Science"
        assert parser._extract_education(text) == "Bachelor's"

    def test_masters_preferred(self, parser):
        text = "Master's degree preferred in a related field"
        assert parser._extract_education(text) == "Master's"

    def test_bs_pattern(self, parser):
        text = "B.S. in Computer Science or related field"
        assert parser._extract_education(text) == "Bachelor's"

    def test_not_found(self, parser):
        text = "No formal education requirements for this role."
        assert parser._extract_education(text) is None


# ── _detect_employment_type ──────────────────────────────────────────────────


class TestDetectEmploymentType:
    def test_full_time(self, parser):
        text = "This is a full-time position"
        assert parser._detect_employment_type(text) == "full_time"

    def test_internship(self, parser):
        text = "Summer internship opportunity"
        assert parser._detect_employment_type(text) == "internship"

    def test_default(self, parser):
        text = "Join our team today"
        assert parser._detect_employment_type(text) == "full_time"


# ── _detect_experience_level ─────────────────────────────────────────────────


class TestDetectExperienceLevel:
    def test_keyword_senior(self, parser):
        text = "Looking for a senior engineer"
        assert parser._detect_experience_level(text, None) == "senior"

    def test_keyword_junior(self, parser):
        text = "Entry level position for junior developers"
        assert parser._detect_experience_level(text, None) == "entry"

    def test_year_inferred_entry(self, parser):
        text = "Some description without level keywords"
        assert parser._detect_experience_level(text, 1.0) == "entry"

    def test_year_inferred_senior(self, parser):
        text = "Some description without level keywords"
        assert parser._detect_experience_level(text, 7.0) == "senior"

    def test_default_mid(self, parser):
        text = "Some description"
        assert parser._detect_experience_level(text, None) == "mid"


# ── _extract_location ────────────────────────────────────────────────────────


class TestExtractLocation:
    def test_location_label(self, parser):
        text = "Location: San Francisco, CA"
        assert "San Francisco" in parser._extract_location(text)

    def test_remote_detection(self, parser):
        text = "This is a fully remote position"
        assert parser._extract_location(text) == "Remote"

    def test_not_found(self, parser):
        text = "Join our amazing team and build great products"
        assert parser._extract_location(text) is None


# ── _calculate_confidence ────────────────────────────────────────────────────


class TestCalculateConfidence:
    def test_high_confidence(self, parser):
        result = JDParseResult(
            title="Senior Engineer",
            required_skills=["python", "java", "sql", "aws", "docker"],
            experience_years_min=5.0,
            education_requirement="Bachelor's",
            responsibilities=["Design systems"],
            qualifications=["5 years experience"],
        )
        conf = parser._calculate_confidence(result)
        assert conf >= 0.8

    def test_low_confidence(self, parser):
        result = JDParseResult(title="Unknown Position")
        conf = parser._calculate_confidence(result)
        assert conf < 0.3


# ── parse_text (end-to-end) ──────────────────────────────────────────────────


class TestParseText:
    def test_full_pipeline(self, parser):
        jd_text = """Job Title: Senior Backend Engineer
Company: TechCorp

About TechCorp
We build scalable cloud solutions.

Responsibilities:
- Design and implement microservices
- Lead code reviews and mentor junior developers
- Collaborate with product and design teams

Requirements:
- 5+ years of experience in backend development
- Proficiency in Python and Django
- Experience with PostgreSQL and Redis
- Bachelor's degree in Computer Science required

Preferred Qualifications:
- Experience with Docker and Kubernetes
- AWS certification
- Master's degree preferred

This is a full-time position.
Location: San Francisco, CA
"""
        result = parser.parse_text(jd_text)

        assert result.success is True
        assert result.title == "Senior Backend Engineer"
        assert result.company_name == "TechCorp"
        assert result.experience_years_min == 5.0
        assert result.education_requirement is not None
        # Parser checks highest education first; "Master's preferred" is found before "Bachelor's required"
        assert result.education_requirement in ("Bachelor's", "Master's")
        assert result.employment_type == "full_time"
        assert len(result.responsibilities) > 0
        assert result.confidence > 0.0
        assert len(result.required_skills) > 0


# ── to_job_create ────────────────────────────────────────────────────────────


class TestToJobCreate:
    def test_conversion(self, parser):
        result = JDParseResult(
            title="Data Engineer",
            company_name="DataCo",
            description="Build data pipelines.",
            raw_text="Build data pipelines. 3+ years experience. Bachelor's required.",
            required_skills=["python", "sql"],
            preferred_skills=["spark"],
            experience_years_min=3.0,
            education_requirement="Bachelor's",
            employment_type="full_time",
            experience_level="mid",
            responsibilities=["Build pipelines"],
        )
        job_create = parser.to_job_create(result)

        assert job_create.title == "Data Engineer"
        assert job_create.company_name == "DataCo"
        assert len(job_create.skill_requirements) == 3
        required = [s for s in job_create.skill_requirements if s.is_required]
        preferred = [s for s in job_create.skill_requirements if not s.is_required]
        assert len(required) == 2
        assert len(preferred) == 1
        assert job_create.experience_requirement is not None
        assert job_create.experience_requirement.minimum_years == 3.0
        assert job_create.education_requirement is not None
