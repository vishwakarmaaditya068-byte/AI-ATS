"""
Summary/objective/profile parser for resumes.

Extracts the professional summary text and detects key career themes.
"""

import re
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SummaryParseResult:
    """Result of summary parsing."""

    text: str = ""
    themes: list[str] = field(default_factory=list)
    confidence: float = 0.0


class SummaryParser:
    """Parser for extracting professional summaries from resume text."""

    THEME_KEYWORDS: dict[str, list[str]] = {
        "leadership": [
            "led", "lead", "managed", "director", "head of", "team lead",
            "mentor", "coach", "supervised", "oversaw", "spearheaded",
        ],
        "technical": [
            "engineer", "developer", "architect", "designed", "implemented",
            "built", "deployed", "optimized", "algorithm", "system",
        ],
        "data_science": [
            "machine learning", "data science", "analytics", "model",
            "prediction", "deep learning", "nlp", "artificial intelligence",
        ],
        "product": [
            "product", "roadmap", "strategy", "stakeholder", "requirements",
            "agile", "scrum", "sprint", "user stories",
        ],
        "domain_finance": [
            "fintech", "banking", "trading", "financial", "investment",
            "capital markets", "risk management",
        ],
        "domain_healthcare": [
            "healthcare", "medical", "clinical", "pharma", "biotech",
            "patient", "ehr", "hipaa",
        ],
    }

    def parse(self, section_text: str) -> SummaryParseResult:
        """Parse a professional summary section."""
        if not section_text or not section_text.strip():
            return SummaryParseResult()

        # Collapse multiple whitespace / newlines to single space
        cleaned = re.sub(r"\s+", " ", section_text.strip())

        # Cap length to prevent bloat, but break at the last sentence boundary
        # within the limit so we never split a sentence mid-way.
        if len(cleaned) > 1000:
            boundary = max(
                cleaned.rfind(".", 0, 1000),
                cleaned.rfind("!", 0, 1000),
                cleaned.rfind("?", 0, 1000),
            )
            cleaned = cleaned[: boundary + 1] if boundary > 0 else cleaned[:1000]

        themes = self._detect_themes(cleaned)
        confidence = 0.9 if len(cleaned) > 50 else 0.5

        return SummaryParseResult(text=cleaned, themes=themes, confidence=confidence)

    def _detect_themes(self, text: str) -> list[str]:
        """Detect career themes present in the summary text."""
        detected: list[str] = []
        for theme, keywords in self.THEME_KEYWORDS.items():
            for kw in keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
                    detected.append(theme)
                    break  # one match per theme is enough
        return detected
