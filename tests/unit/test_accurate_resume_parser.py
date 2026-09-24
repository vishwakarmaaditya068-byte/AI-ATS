"""
Tests for AccurateResumeParser.

Written FIRST (TDD/RED) before any implementation exists.
Asserts against real PDFs in data/raw/resumes/.
"""

import json
from pathlib import Path

import pytest

# Parser does not exist yet — import will fail on RED run
from src.ml.nlp.accurate_resume_parser import AccurateResumeParser

RESUMES_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "resumes"
VIVEK_RESUME = RESUMES_DIR / "vivek_resume.pdf"


@pytest.fixture(scope="module")
def parser():
    return AccurateResumeParser()


@pytest.fixture(scope="module")
def vivek(parser):
    return parser.parse(VIVEK_RESUME)


# ---------------------------------------------------------------------------
# Contact extraction
# ---------------------------------------------------------------------------

class TestContactExtraction:
    def test_extracts_full_name(self, vivek):
        assert vivek.contact.name.upper() == "VIVEK VAISH"

    def test_extracts_email(self, vivek):
        assert vivek.contact.email == "vaishvivek634@gmail.com"

    def test_extracts_phone_number(self, vivek):
        phone = vivek.contact.phone
        assert phone, "Phone should not be empty"
        assert "7417378004" in phone.replace("-", "").replace(" ", "")

    def test_extracts_location(self, vivek):
        assert "noida" in vivek.contact.location.lower()


# ---------------------------------------------------------------------------
# Skills extraction
# ---------------------------------------------------------------------------

class TestSkillsExtraction:
    def test_extracts_python(self, vivek):
        all_skills = [s.lower() for cat in vivek.skills for s in cat.skills]
        assert "python" in all_skills

    def test_extracts_javascript(self, vivek):
        all_skills = [s.lower() for cat in vivek.skills for s in cat.skills]
        assert any("javascript" in s for s in all_skills)

    def test_extracts_react(self, vivek):
        all_skills = [s.lower() for cat in vivek.skills for s in cat.skills]
        assert any("react" in s for s in all_skills)

    def test_has_skill_categories(self, vivek):
        assert len(vivek.skills) > 0
        category_names = [c.category.lower() for c in vivek.skills]
        assert any(
            "programming" in c or "language" in c or "web" in c or "backend" in c or "tool" in c
            for c in category_names
        )

    def test_docker_in_tools(self, vivek):
        all_skills = [s.lower() for cat in vivek.skills for s in cat.skills]
        assert "docker" in all_skills


# ---------------------------------------------------------------------------
# Projects extraction
# ---------------------------------------------------------------------------

class TestProjectsExtraction:
    def test_extracts_at_least_two_projects(self, vivek):
        assert len(vivek.projects) >= 2

    def test_extracts_twitter_sentiment_project(self, vivek):
        names = [p.name.lower() for p in vivek.projects]
        assert any("twitter" in n or "sentiment" in n for n in names)

    def test_extracts_autobook_project(self, vivek):
        names = [p.name.lower() for p in vivek.projects]
        assert any("autobook" in n for n in names)

    def test_projects_have_bullet_points(self, vivek):
        assert any(len(p.bullets) > 0 for p in vivek.projects), (
            "At least one project must have parsed bullet points"
        )

    def test_twitter_project_has_bullets(self, vivek):
        twitter = next(
            (p for p in vivek.projects if "twitter" in p.name.lower() or "sentiment" in p.name.lower()),
            None,
        )
        assert twitter is not None
        assert len(twitter.bullets) >= 2, "Twitter project should have at least 2 bullet points"

    def test_autobook_project_has_bullets(self, vivek):
        autobook = next(
            (p for p in vivek.projects if "autobook" in p.name.lower()),
            None,
        )
        assert autobook is not None
        assert len(autobook.bullets) >= 2


# ---------------------------------------------------------------------------
# Education extraction
# ---------------------------------------------------------------------------

class TestEducationExtraction:
    def test_extracts_btech(self, vivek):
        degrees = [e.degree.lower() for e in vivek.education]
        assert any("b.tech" in d or "btech" in d or "b tech" in d for d in degrees)

    def test_extracts_galgotias(self, vivek):
        institutions = [e.institution.lower() for e in vivek.education if e.institution]
        assert any("galgotias" in i for i in institutions)

    def test_extracts_cgpa(self, vivek):
        scores = [e.score for e in vivek.education if e.score]
        combined = " ".join(scores).lower()
        assert "7.3" in combined or "cgpa" in combined

    def test_extracts_schooling(self, vivek):
        # Two school entries (class 10, class 12)
        assert len(vivek.education) >= 2


# ---------------------------------------------------------------------------
# Experience extraction
# ---------------------------------------------------------------------------

class TestExperienceExtraction:
    def test_extracts_at_least_one_experience(self, vivek):
        assert len(vivek.experience) >= 1

    def test_extracts_aiml_internship(self, vivek):
        titles = [e.title.lower() for e in vivek.experience]
        assert any("intern" in t or "aiml" in t for t in titles)

    def test_internship_has_duration(self, vivek):
        internship = vivek.experience[0]
        assert internship.duration, "Internship should have a duration string"

    def test_experience_has_bullet_points(self, vivek):
        assert any(len(e.bullets) > 0 for e in vivek.experience)


# ---------------------------------------------------------------------------
# Achievements extraction
# ---------------------------------------------------------------------------

class TestAchievementsExtraction:
    def test_extracts_achievements(self, vivek):
        assert len(vivek.achievements) >= 1

    def test_hackathon_mentioned(self, vivek):
        combined = " ".join(vivek.achievements).lower()
        assert "hackathon" in combined or "code kshetra" in combined


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------

class TestOutputFormat:
    def test_to_dict_has_all_keys(self, parser, vivek):
        d = parser.to_dict(vivek)
        for key in ("contact", "skills", "experience", "education", "projects", "achievements"):
            assert key in d, f"Missing key: {key}"

    def test_to_json_is_valid_json(self, parser, vivek):
        json_str = parser.to_json(vivek)
        parsed = json.loads(json_str)
        assert "contact" in parsed

    def test_contact_dict_has_name_and_email(self, parser, vivek):
        d = parser.to_dict(vivek)
        assert d["contact"]["name"]
        assert d["contact"]["email"]


# ---------------------------------------------------------------------------
# Robustness across all resumes
# ---------------------------------------------------------------------------

class TestMultipleResumes:
    def test_all_pdfs_parse_without_exception(self, parser):
        pdfs = list(RESUMES_DIR.glob("*.pdf"))
        assert len(pdfs) > 0, "No PDF files found in data/raw/resumes"
        for pdf in pdfs:
            result = parser.parse(pdf)
            assert result.raw_text, f"{pdf.name}: raw_text is empty"

    def test_majority_have_email(self, parser):
        pdfs = list(RESUMES_DIR.glob("*.pdf"))
        hits = sum(1 for pdf in pdfs if parser.parse(pdf).contact.email)
        assert hits / len(pdfs) >= 0.7, (
            f"Only {hits}/{len(pdfs)} resumes had an email parsed"
        )

    def test_majority_have_name(self, parser):
        pdfs = list(RESUMES_DIR.glob("*.pdf"))
        hits = sum(1 for pdf in pdfs if parser.parse(pdf).contact.name)
        assert hits / len(pdfs) >= 0.7, (
            f"Only {hits}/{len(pdfs)} resumes had a name parsed"
        )
