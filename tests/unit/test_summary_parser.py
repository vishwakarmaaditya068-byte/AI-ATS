"""
Tests for src.ml.nlp.parsers.summary_parser — SummaryParser.
"""

import pytest

from src.ml.nlp.parsers.summary_parser import SummaryParser, SummaryParseResult


@pytest.fixture
def parser():
    return SummaryParser()


class TestSummaryParser:
    def test_empty_input_returns_empty_result(self, parser):
        result = parser.parse("")
        assert result.text == ""
        assert result.confidence == 0.0
        assert result.themes == []

    def test_whitespace_only_returns_empty_result(self, parser):
        result = parser.parse("   \n   ")
        assert result.text == ""
        assert result.confidence == 0.0

    def test_whitespace_normalized(self, parser):
        text = "Experienced   engineer\nwith   multiple   spaces."
        result = parser.parse(text)
        assert "  " not in result.text

    def test_short_text_low_confidence(self, parser):
        text = "Python developer."  # < 50 chars
        result = parser.parse(text)
        assert result.confidence == pytest.approx(0.5)

    def test_long_text_high_confidence(self, parser):
        text = "Experienced software engineer with 6+ years building scalable distributed systems and APIs."
        result = parser.parse(text)
        assert result.confidence == pytest.approx(0.9)

    def test_leadership_theme_detected(self, parser):
        text = "Led a team of 8 engineers. Managed cross-functional projects. Mentor for junior developers."
        result = parser.parse(text)
        assert "leadership" in result.themes

    def test_technical_theme_detected(self, parser):
        text = "Software engineer who designed and implemented microservices architecture using Python."
        result = parser.parse(text)
        assert "technical" in result.themes

    def test_data_science_theme_detected(self, parser):
        text = "Machine learning engineer specializing in NLP and deep learning model development."
        result = parser.parse(text)
        assert "data_science" in result.themes

    def test_multiple_themes_detected(self, parser):
        text = (
            "Led a team of data scientists to design and implement machine learning models "
            "for real-time prediction systems."
        )
        result = parser.parse(text)
        assert len(result.themes) >= 2

    def test_theme_deduplication(self, parser):
        text = "Led projects. Managed teams. Supervised engineers. Coached junior staff."
        result = parser.parse(text)
        assert result.themes.count("leadership") == 1

    def test_text_capped_at_1000_chars(self, parser):
        text = "A" * 2000
        result = parser.parse(text)
        assert len(result.text) <= 1000

    def test_no_false_theme_from_unrelated_text(self, parser):
        text = "Organized office supplies and managed document filing for the legal department."
        result = parser.parse(text)
        assert "data_science" not in result.themes
        assert "domain_healthcare" not in result.themes
