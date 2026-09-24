"""
Tests for src.ml.nlp.parsers.projects_parser — ProjectsParser.
"""

import pytest

from src.ml.nlp.parsers.projects_parser import ProjectsParser, ProjectsParseResult


@pytest.fixture
def parser():
    return ProjectsParser()


class TestProjectsParser:
    def test_project_name_from_first_line(self, parser):
        text = "E-Commerce Platform\nBuilt a scalable online store using Python and Django."
        result = parser.parse(text)
        assert len(result.projects) >= 1
        assert result.projects[0].name == "E-Commerce Platform"

    def test_tech_extraction_via_skills(self, parser):
        text = "My Project\nBuilt with Python, Django, and PostgreSQL for the backend."
        result = parser.parse(text)
        assert len(result.projects) >= 1
        tech = [t.lower() for t in result.projects[0].technologies]
        assert "python" in tech or "django" in tech or "postgresql" in tech

    def test_multi_project_blank_line_split(self, parser):
        text = (
            "Project Alpha\nA great project using Python.\n\n"
            "Project Beta\nAnother project using JavaScript and React."
        )
        result = parser.parse(text)
        assert len(result.projects) == 2

    def test_multi_project_numbered_header_split(self, parser):
        text = (
            "1. Project Alpha\nA great project using Python.\n"
            "2. Project Beta\nAnother project using JavaScript."
        )
        result = parser.parse(text)
        assert len(result.projects) == 2

    def test_github_url_detection(self, parser):
        text = "Open Source Tool\nA CLI utility.\nhttps://github.com/user/my-tool"
        result = parser.parse(text)
        assert result.projects[0].url is not None
        assert "github.com" in result.projects[0].url

    def test_general_https_url_detection(self, parser):
        text = "Portfolio Site\nMy personal website.\nhttps://myportfolio.io/work"
        result = parser.parse(text)
        assert result.projects[0].url is not None
        assert "https://" in result.projects[0].url

    def test_empty_input_returns_empty_result(self, parser):
        result = parser.parse("")
        assert result.projects == []
        assert result.confidence == 0.0

    def test_description_capped_at_300_chars(self, parser):
        long_line = "A" * 400
        text = f"My Project\n{long_line}"
        result = parser.parse(text)
        if result.projects and result.projects[0].description:
            assert len(result.projects[0].description) <= 300

    def test_name_stripped_of_bullet(self, parser):
        text = "- My Awesome Project\nBuilt with Python."
        result = parser.parse(text)
        assert len(result.projects) >= 1
        assert result.projects[0].name == "My Awesome Project"

    def test_confidence_increases_with_more_fields(self, parser):
        name_only = "Simple Project"
        full_project = (
            "Scalable API\nBuilt a REST API serving 1M+ requests.\n"
            "Tech: Python, FastAPI, PostgreSQL\nhttps://github.com/user/api"
        )
        r_name = parser.parse(name_only)
        r_full = parser.parse(full_project)
        if r_name.projects and r_full.projects:
            assert r_full.projects[0].confidence >= r_name.projects[0].confidence
