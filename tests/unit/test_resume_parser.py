"""
Tests for src.ml.nlp.resume_parser — ResumeParser logic.
"""

import pytest

from src.ml.nlp.resume_parser import ResumeParser, ResumeParseResult


@pytest.fixture
def parser():
    return ResumeParser()


# ── ResumeParseResult.success ────────────────────────────────────────────────


class TestResumeParseResultSuccess:
    def test_success_true(self):
        r = ResumeParseResult(overall_confidence=0.8, errors=[])
        assert r.success is True

    def test_success_false_with_errors(self):
        r = ResumeParseResult(overall_confidence=0.8, errors=["File not found"])
        assert r.success is False

    def test_success_false_low_confidence(self):
        r = ResumeParseResult(overall_confidence=0.2, errors=[])
        assert r.success is False

    def test_success_at_threshold(self):
        r = ResumeParseResult(overall_confidence=0.31, errors=[])
        assert r.success is True

    def test_success_at_exactly_threshold(self):
        r = ResumeParseResult(overall_confidence=0.3, errors=[])
        assert r.success is False  # > 0.3 required, not >=


# ── _calculate_overall_confidence ────────────────────────────────────────────


class TestCalculateOverallConfidence:
    def test_all_sections_present(self, parser):
        result = ResumeParseResult(
            contact={"confidence": 0.9, "email": "x@y.com"},
            skills=[{"name": "python", "confidence": 0.8, "category": None, "proficiency": None, "source": "text"}],
            experience=[{"confidence": 0.7, "job_title": "Dev", "company": "Co"}],
            education=[{"confidence": 0.9, "degree": "BS", "institution": "MIT"}],
        )
        conf = parser._calculate_overall_confidence(result)
        # contact: 0.23*0.9 + skills: 0.23*0.8 + experience: 0.28*0.7 + education: 0.18*0.9
        # certifications/projects default to [] so contribute 0
        expected = 0.23 * 0.9 + 0.23 * 0.8 + 0.28 * 0.7 + 0.18 * 0.9
        assert abs(conf - round(expected, 2)) < 0.02

    def test_partial_sections(self, parser):
        result = ResumeParseResult(
            contact={"confidence": 0.8, "email": "x@y.com"},
            skills=[],
            experience=[],
            education=[],
        )
        conf = parser._calculate_overall_confidence(result)
        # Only contact contributes: 0.23 * 0.8 = 0.184
        assert abs(conf - 0.184) < 0.02

    def test_empty_result(self, parser):
        result = ResumeParseResult(contact=None, skills=[], experience=[], education=[])
        conf = parser._calculate_overall_confidence(result)
        assert conf == 0.0


# ── _calculate_quality_score ─────────────────────────────────────────────────


class TestCalculateQualityScore:
    def test_full_quality(self, parser):
        result = ResumeParseResult(
            contact={"email": "x@y.com", "first_name": "Jane", "last_name": "Smith", "phone": "+1-555-0100"},
            skills=[{"name": f"skill{i}"} for i in range(6)],  # >= 5 skills
            experience=[{"job_title": "Dev", "company": "Co"}],
            total_experience_years=3.0,
            education=[{"degree": "BS"}],
            highest_education="Bachelor",
        )
        score = parser._calculate_quality_score(result)
        # email(0.15) + names(0.10) + phone(0.05) + skills>=5(0.20) +
        # experience(0.15) + exp_years(0.10) + education(0.15) + highest(0.10) = 1.0
        assert score == 1.0

    def test_minimal_quality(self, parser):
        result = ResumeParseResult(contact=None, skills=[], experience=[], education=[])
        score = parser._calculate_quality_score(result)
        assert score == 0.0

    def test_skills_threshold_at_two(self, parser):
        result = ResumeParseResult(
            contact=None,
            skills=[{"name": "a"}, {"name": "b"}],  # 2 skills
            experience=[],
            education=[],
        )
        score = parser._calculate_quality_score(result)
        assert abs(score - 0.10) < 0.02  # 2 skills gives 0.10

    def test_skills_threshold_at_five(self, parser):
        result = ResumeParseResult(
            contact=None,
            skills=[{"name": f"s{i}"} for i in range(5)],  # 5 skills
            experience=[],
            education=[],
        )
        score = parser._calculate_quality_score(result)
        assert abs(score - 0.20) < 0.02  # 5+ skills gives 0.20

    def test_capped_at_one(self, parser):
        """Score should never exceed 1.0."""
        result = ResumeParseResult(
            contact={"email": "x@y.com", "first_name": "Jane", "last_name": "Smith", "phone": "+1-555"},
            skills=[{"name": f"s{i}"} for i in range(10)],
            experience=[{"job_title": "Dev", "company": "Co"}],
            total_experience_years=5.0,
            education=[{"degree": "BS"}],
            highest_education="Bachelor",
        )
        score = parser._calculate_quality_score(result)
        assert score <= 1.0


# ── to_candidate_create ──────────────────────────────────────────────────────


class TestToCandidateCreate:
    def test_valid_data(self, parser, make_resume_parse_result):
        result = make_resume_parse_result()
        candidate = parser.to_candidate_create(result)
        assert candidate is not None
        assert candidate.first_name == "Jane"
        assert candidate.last_name == "Smith"
        assert candidate.contact.email == "jane.smith@example.com"
        assert len(candidate.skills) > 0

    def test_missing_email_returns_none(self, parser, make_resume_parse_result):
        result = make_resume_parse_result(
            contact={"first_name": "Jane", "last_name": "Smith", "email": None, "confidence": 0.5},
        )
        candidate = parser.to_candidate_create(result)
        assert candidate is None

    def test_failed_parse_returns_none(self, parser, make_resume_parse_result):
        result = make_resume_parse_result(errors=["Extraction failed"], overall_confidence=0.1)
        candidate = parser.to_candidate_create(result)
        assert candidate is None


# ── parse_text (end-to-end) ──────────────────────────────────────────────────


class TestParseText:
    def test_synthetic_resume(self, parser):
        resume_text = """Jane Smith
jane.smith@example.com | (555) 123-4567
San Francisco, CA

PROFESSIONAL SUMMARY
Experienced software engineer with 5+ years building scalable backend systems.

SKILLS
Python, Django, Flask, PostgreSQL, Docker, AWS, REST APIs, Git

EXPERIENCE

Senior Software Engineer
TechCorp, San Francisco, CA
January 2020 - Present
- Designed and implemented microservices architecture
- Led a team of 4 developers
- Reduced API response time by 40%

Software Engineer
StartupInc, New York, NY
June 2017 - December 2019
- Full-stack development using Python and React
- Built CI/CD pipelines

EDUCATION

Bachelor of Science in Computer Science
Massachusetts Institute of Technology
Graduated May 2017
GPA: 3.8
"""
        result = parser.parse_text(resume_text)

        assert isinstance(result, ResumeParseResult)
        assert result.processing_time_ms >= 0
        # Should find contact info
        assert result.contact is not None
        # Should find some skills
        assert result.skill_count > 0
        # Confidence should be reasonable
        assert result.overall_confidence > 0


# ── New field defaults ────────────────────────────────────────────────────────


class TestResumeParseResultNewFields:
    def test_default_certifications_is_empty_list(self):
        r = ResumeParseResult()
        assert r.certifications == []

    def test_default_projects_is_empty_list(self):
        r = ResumeParseResult()
        assert r.projects == []

    def test_default_summary_is_none(self):
        r = ResumeParseResult()
        assert r.summary is None

    def test_default_languages_is_empty_list(self):
        r = ResumeParseResult()
        assert r.languages == []


# ── to_candidate_create — new field mapping ───────────────────────────────────


class TestToCandidateCreateNewFields:
    def test_summary_mapped(self, parser, make_resume_parse_result):
        result = make_resume_parse_result(summary="Experienced Python engineer.")
        candidate = parser.to_candidate_create(result)
        assert candidate is not None
        assert candidate.summary == "Experienced Python engineer."

    def test_certifications_mapped(self, parser, make_resume_parse_result):
        result = make_resume_parse_result(
            certifications=[{
                "name": "AWS Solutions Architect",
                "issuer": "Amazon",
                "issue_date": "2022-01-01",
                "expiry_date": None,
                "credential_id": "ABC123",
                "credential_url": None,
                "confidence": 0.9,
            }]
        )
        candidate = parser.to_candidate_create(result)
        assert candidate is not None
        assert len(candidate.certifications) == 1
        assert candidate.certifications[0].name == "AWS Solutions Architect"

    def test_languages_mapped(self, parser, make_resume_parse_result):
        result = make_resume_parse_result(languages=["English", "Spanish"])
        candidate = parser.to_candidate_create(result)
        assert candidate is not None
        assert len(candidate.languages) == 2
