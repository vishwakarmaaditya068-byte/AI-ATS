"""
Text preprocessing utilities for resume parsing.

Handles text cleaning, normalization, and section detection.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Common section headers in resumes
SECTION_HEADERS = {
    "contact": [
        "contact", "contact information", "personal information",
        "personal details", "contact details",
    ],
    "summary": [
        "summary", "professional summary", "profile", "objective",
        "career objective", "about me", "about", "overview",
        "professional profile", "executive summary",
    ],
    "experience": [
        "experience", "work experience", "employment history",
        "professional experience", "work history", "employment",
        "career history", "professional background", "positions held",
        "internship", "internships", "internship experience",
        "internship/experience", "experience/internship",
        "work experience/internship", "internship & experience",
        "industry experience",
    ],
    "education": [
        "education", "educational background", "academic background",
        "qualifications", "academic qualifications", "schooling",
        "educational qualifications", "degrees",
    ],
    "skills": [
        "skills", "technical skills", "core competencies",
        "competencies", "areas of expertise", "expertise",
        "key skills", "professional skills", "abilities",
        "technologies", "tools", "programming languages",
    ],
    "certifications": [
        "certifications", "certificates", "licenses",
        "professional certifications", "credentials",
        "training", "professional development",
    ],
    "projects": [
        "projects", "key projects", "notable projects",
        "project experience", "personal projects",
    ],
    "languages": [
        "languages", "language skills", "language proficiency",
    ],
    "references": [
        "references", "referees",
    ],
    "awards": [
        "awards", "honors", "achievements", "accomplishments",
        "recognition",
    ],
    "publications": [
        "publications", "papers", "research",
    ],
    "interests": [
        "interests", "hobbies", "activities",
    ],
}


@dataclass
class TextSection:
    """Represents a detected section in the resume."""

    section_type: str
    title: str
    content: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0


@dataclass
class PreprocessedText:
    """Result of text preprocessing."""

    original_text: str
    cleaned_text: str
    sections: list[TextSection] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    detected_language: str = "en"
    word_count: int = 0
    warnings: list[str] = field(default_factory=list)


class TextPreprocessor:
    """
    Preprocessor for resume text.

    Handles cleaning, normalization, and section detection.
    """

    def __init__(self):
        """Initialize the preprocessor."""
        self._section_pattern = self._build_section_pattern()

    def _build_section_pattern(self) -> re.Pattern:
        """Build regex pattern for section header detection."""
        all_headers = []
        for headers in SECTION_HEADERS.values():
            all_headers.extend(headers)

        # Sort by length (longest first) to match longer phrases first
        all_headers.sort(key=len, reverse=True)

        # Escape special regex characters
        escaped = [re.escape(h) for h in all_headers]

        # Build pattern that matches headers at line start
        pattern = r"^\s*(?:[\d\.\-\*•]+\s*)?(" + "|".join(escaped) + r")\s*:?\s*$"
        return re.compile(pattern, re.IGNORECASE | re.MULTILINE)

    def preprocess(self, text: str) -> PreprocessedText:
        """
        Preprocess resume text.

        Args:
            text: Raw extracted text

        Returns:
            PreprocessedText with cleaned content and detected sections
        """
        warnings = []

        # Clean the text
        cleaned = self._clean_text(text)

        # Split into lines
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]

        # Detect sections
        sections = self._detect_sections(cleaned)

        # Count words
        word_count = len(cleaned.split())

        # Detect language (basic)
        language = self._detect_language(cleaned)

        if word_count < 50:
            warnings.append("Very short text - may be incomplete extraction")

        if language != "en":
            warnings.append(
                f"Non-English resume detected (language: {language}). "
                "Parsing accuracy may be reduced for non-English content."
            )

        if not sections:
            warnings.append("Could not detect standard resume sections")

        return PreprocessedText(
            original_text=text,
            cleaned_text=cleaned,
            sections=sections,
            lines=lines,
            detected_language=language,
            word_count=word_count,
            warnings=warnings,
        )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Replace common problematic characters
        replacements = {
            "\u2018": "'",  # Left single quote
            "\u2019": "'",  # Right single quote
            "\u201c": '"',  # Left double quote
            "\u201d": '"',  # Right double quote
            "\u2013": "-",  # En dash
            "\u2014": "-",  # Em dash
            "\u2026": "...",  # Ellipsis
            "\u00a0": " ",  # Non-breaking space
            "\u00ad": "",  # Soft hyphen
            "\ufeff": "",  # BOM
            "\u200b": "",  # Zero-width space
            "\t": "    ",  # Tab to spaces
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove control characters (except newlines)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Remove excessive spaces within lines
        text = re.sub(r"[ ]{2,}", " ", text)

        return text.strip()

    def _detect_sections(self, text: str) -> list[TextSection]:
        """Detect sections in the resume text."""
        sections = []
        lines = text.split("\n")

        current_section: Optional[TextSection] = None
        content_lines: list[str] = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            # Check if this line is a section header
            section_type = self._identify_section_header(line_stripped)

            if section_type:
                # Save previous section
                if current_section:
                    current_section.content = "\n".join(content_lines).strip()
                    current_section.end_pos = (
                        sum(len(l) + 1 for l in lines[:i]) - 1
                    )
                    if current_section.content:
                        sections.append(current_section)

                # Start new section
                start_pos = sum(len(l) + 1 for l in lines[:i])
                current_section = TextSection(
                    section_type=section_type,
                    title=line_stripped,
                    content="",
                    start_pos=start_pos,
                    end_pos=start_pos,
                )
                content_lines = []
            else:
                content_lines.append(line)

        # Don't forget the last section
        if current_section:
            current_section.content = "\n".join(content_lines).strip()
            current_section.end_pos = len(text)
            if current_section.content:
                sections.append(current_section)

        # If no sections detected, create a single "unknown" section
        if not sections and text.strip():
            sections.append(
                TextSection(
                    section_type="unknown",
                    title="",
                    content=text,
                    start_pos=0,
                    end_pos=len(text),
                    confidence=0.5,
                )
            )

        return sections

    def _identify_section_header(self, line: str) -> Optional[str]:
        """
        Identify if a line is a section header.

        Returns the section type or None.
        """
        if not line:
            return None

        # Clean the line for matching
        clean_line = line.lower().strip()

        # Remove common prefixes/suffixes
        clean_line = re.sub(r"^[\d\.\-\*•:]+\s*", "", clean_line)
        clean_line = re.sub(r"\s*[:]\s*$", "", clean_line)
        clean_line = clean_line.strip()

        # Check against known section headers
        for section_type, headers in SECTION_HEADERS.items():
            for header in headers:
                if clean_line == header.lower():
                    return section_type
                # Also check if line starts with the header
                if clean_line.startswith(header.lower()) and len(clean_line) < len(header) + 5:
                    return section_type

        # Check for ALL CAPS short lines (common header format).
        # Require at least 2 words so single-word acronyms on their own line
        # (e.g. "AWS", "SQL", "API") are not misidentified as section headers.
        if (
            line.isupper()
            and 2 <= len(line.split()) <= 4
            and len(line) <= 40
        ):
            # Try to match to a section type
            for section_type, headers in SECTION_HEADERS.items():
                for header in headers:
                    if header.lower() in line.lower():
                        return section_type

        return None

    # High-frequency stop words that are strongly distinctive per language.
    # Words shared across languages (e.g. "a", "in") are intentionally excluded.
    _LANG_STOP_WORDS: dict[str, frozenset[str]] = {
        "en": frozenset([
            "the", "and", "for", "with", "of", "to", "in", "is", "are", "was",
            "has", "have", "this", "that", "which", "from", "by", "at", "an",
            "skills", "experience", "education", "work", "company",
        ]),
        "fr": frozenset([
            "le", "la", "les", "de", "du", "des", "et", "en", "un", "une",
            "est", "avec", "dans", "qui", "que", "sur", "par", "pour",
            "compétences", "expérience", "formation", "entreprise",
        ]),
        "de": frozenset([
            "der", "die", "das", "und", "in", "von", "mit", "ist", "für",
            "auf", "an", "den", "ein", "eine", "als", "auch", "bei",
            "kenntnisse", "erfahrung", "ausbildung", "entwicklung",
        ]),
        "es": frozenset([
            "el", "la", "los", "las", "de", "del", "y", "en", "con", "por",
            "que", "un", "una", "es", "para", "como", "sus", "del",
            "experiencia", "habilidades", "formación", "empresa",
        ]),
        "it": frozenset([
            "il", "lo", "la", "le", "di", "del", "e", "in", "con", "per",
            "che", "un", "una", "è", "da", "su", "si", "dei",
            "competenze", "esperienza", "formazione", "azienda",
        ]),
        "pt": frozenset([
            "o", "a", "os", "as", "de", "do", "da", "e", "em", "com",
            "que", "um", "uma", "por", "para", "das", "dos",
            "experiência", "habilidades", "formação", "empresa",
        ]),
    }

    def _detect_language(self, text: str) -> str:
        """
        Detect the language of resume text using stop-word frequency.

        Tokenises on word boundaries so substrings don't produce false
        positives (e.g. 'the' inside 'théâtre').  Returns the BCP-47
        language tag with the highest stop-word hit count, defaulting
        to 'en' when the text is empty or the scores are tied.

        Args:
            text: Cleaned resume text.

        Returns:
            Two-letter language code such as 'en', 'fr', 'de', 'es'.
        """
        if not text.strip():
            return "en"

        import re
        tokens: set[str] = set(re.findall(r"\b[a-záàâäãåçéèêëíìîïñóòôöõúùûüýÿæœ]+\b",
                                           text.lower()))

        scores: dict[str, int] = {
            lang: len(tokens & words)
            for lang, words in self._LANG_STOP_WORDS.items()
        }

        best_lang: str = max(scores, key=lambda lang: (scores[lang], lang == "en"))
        return best_lang if scores[best_lang] > 0 else "en"

    def get_section_content(
        self, preprocessed: PreprocessedText, section_type: str
    ) -> Optional[str]:
        """Get content of a specific section type."""
        for section in preprocessed.sections:
            if section.section_type == section_type:
                return section.content
        return None

    def get_sections_by_type(
        self, preprocessed: PreprocessedText, section_type: str
    ) -> list[TextSection]:
        """Get all sections of a specific type."""
        return [s for s in preprocessed.sections if s.section_type == section_type]
