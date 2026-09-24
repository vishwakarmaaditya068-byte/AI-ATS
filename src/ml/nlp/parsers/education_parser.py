"""
Education parser for resumes.

Extracts degrees, institutions, graduation dates, and fields of study.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from src.utils.constants import EDUCATION_LEVELS
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedEducation:
    """An education entry extracted from a resume."""

    degree: Optional[str] = None
    degree_level: Optional[str] = None  # bachelor, master, phd, etc.
    field_of_study: Optional[str] = None
    institution: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[date] = None
    graduation_date: Optional[date] = None
    gpa: Optional[float] = None
    honors: Optional[str] = None
    relevant_coursework: list[str] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0


@dataclass
class EducationParseResult:
    """Result of education parsing."""

    education: list[ExtractedEducation] = field(default_factory=list)
    highest_level: Optional[str] = None
    confidence: float = 0.0


class EducationParser:
    """Parser for extracting education from resume text."""

    # Degree patterns
    DEGREE_PATTERNS = {
        "phd": [
            r"ph\.?d\.?", r"doctor(?:ate)?(?:\s+of)?\s+philosophy",
            r"d\.?phil\.?", r"doctorate",
        ],
        "master": [
            r"master(?:'?s)?(?:\s+of)?", r"\bm\.?s\.?(?:\s|$)", r"\bm\.?a\.?(?:\s|$)",
            r"\bm\.?b\.?a\.?", r"\bm\.?eng\.?", r"\bm\.?sc\.?", r"\bm\.?ed\.?",
            r"\bmba\b", r"\bmsc\b", r"\bma(?:\s|$)",
        ],
        "bachelor": [
            r"bachelor(?:'?s)?(?:\s+of)?", r"\bb\.?s\.?(?:\s|$)", r"\bb\.?a\.?(?:\s|$)",
            r"\bb\.?eng\.?", r"\bb\.?sc\.?", r"\bb\.?tech\.?", r"\bb\.?e\.?(?:\s|$)",
            r"\bbsc\b", r"\bba(?:\s|$)", r"\bbs(?:\s|$)",
        ],
        "associate": [
            r"associate(?:'?s)?(?:\s+of)?", r"\ba\.?s\.?(?:\s|$)", r"\ba\.?a\.?(?:\s|$)",
        ],
        "diploma": [
            r"diploma", r"certificate", r"certification",
        ],
        "high school": [
            r"high\s*school", r"secondary\s*school", r"ged",
        ],
    }

    # Common fields of study
    FIELDS_OF_STUDY = [
        "computer science", "software engineering", "information technology",
        "data science", "artificial intelligence", "machine learning",
        "electrical engineering", "mechanical engineering", "civil engineering",
        "chemical engineering", "biomedical engineering", "aerospace engineering",
        "business administration", "finance", "accounting", "economics",
        "marketing", "management", "human resources", "operations",
        "mathematics", "statistics", "physics", "chemistry", "biology",
        "psychology", "sociology", "political science", "communications",
        "english", "history", "philosophy", "art", "music",
        "medicine", "nursing", "pharmacy", "public health",
        "law", "education", "architecture", "design",
        "information systems", "cybersecurity", "network engineering",
    ]

    # University/College indicators
    INSTITUTION_INDICATORS = [
        "university", "college", "institute", "school", "academy",
        "polytechnic", "conservatory",
    ]

    # Date pattern (reuse from experience parser)
    DATE_PATTERN = re.compile(
        r"(?:"
        r"(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"[,.\s]*\d{4})"
        r"|(?:\d{1,2}[/\-]\d{4})"
        r"|(?:\d{4})"
        r")",
        re.IGNORECASE,
    )

    # GPA pattern
    GPA_PATTERN = re.compile(
        r"(?:gpa|grade\s*point\s*average|cumulative)[:\s]*(\d+\.?\d*)\s*(?:/\s*(\d+\.?\d*))?",
        re.IGNORECASE,
    )

    # Honors patterns
    HONORS_PATTERNS = [
        r"summa\s*cum\s*laude",
        r"magna\s*cum\s*laude",
        r"cum\s*laude",
        r"with\s*(?:highest\s+)?honors?",
        r"dean'?s?\s*list",
        r"valedictorian",
        r"salutatorian",
        r"first\s*class\s*honors?",
        r"distinction",
    ]

    def parse(
        self,
        text: str,
        education_section: Optional[str] = None,
    ) -> EducationParseResult:
        """
        Parse education from resume text.

        Args:
            text: Full resume text
            education_section: Optional dedicated education section text

        Returns:
            EducationParseResult with extracted education entries
        """
        # Use education section if provided, otherwise use full text
        parse_text = education_section if education_section else text

        education_entries = self._extract_education(parse_text)

        # Determine highest level
        highest_level = self._get_highest_level(education_entries)

        # Calculate confidence
        if education_entries:
            avg_confidence = sum(e.confidence for e in education_entries) / len(education_entries)
        else:
            avg_confidence = 0.0

        return EducationParseResult(
            education=education_entries,
            highest_level=highest_level,
            confidence=avg_confidence,
        )

    def _extract_education(self, text: str) -> list[ExtractedEducation]:
        """Extract individual education entries from text."""
        entries = []

        # Split into blocks
        blocks = self._split_into_blocks(text)

        for block in blocks:
            entry = self._parse_education_block(block)
            if entry and (entry.degree or entry.institution):
                entries.append(entry)

        # Merge adjacent orphan blocks: a degree-only entry immediately
        # followed by an institution-only entry likely belongs together.
        merged: list[ExtractedEducation] = []
        i = 0
        while i < len(entries):
            current = entries[i]
            if (
                current.degree
                and not current.institution
                and i + 1 < len(entries)
            ):
                nxt = entries[i + 1]
                if nxt.institution and not nxt.degree:
                    # Merge: take institution, dates, gpa from the next entry
                    current.institution = nxt.institution
                    current.graduation_date = current.graduation_date or nxt.graduation_date
                    current.start_date = current.start_date or nxt.start_date
                    current.gpa = current.gpa or nxt.gpa
                    current.confidence = max(current.confidence, nxt.confidence)
                    merged.append(current)
                    i += 2
                    continue
            merged.append(current)
            i += 1

        # Sort by graduation date (most recent first)
        merged.sort(
            key=lambda e: e.graduation_date or date.min,
            reverse=True,
        )

        return merged

    def _split_into_blocks(self, text: str) -> list[str]:
        """Split text into potential education blocks."""
        lines = text.split("\n")
        blocks = []
        current_block = []

        for line in lines:
            line_stripped = line.strip()

            if not line_stripped:
                if current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                continue

            # Check if this line looks like a new entry
            if self._is_entry_header(line_stripped):
                if current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []

            current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))

        return blocks

    def _is_entry_header(self, line: str) -> bool:
        """Check if a line looks like a new education entry header."""
        line_lower = line.lower()

        # Contains degree keywords
        for patterns in self.DEGREE_PATTERNS.values():
            for pattern in patterns:
                if re.search(pattern, line_lower):
                    return True

        # Contains institution indicators
        if any(ind in line_lower for ind in self.INSTITUTION_INDICATORS):
            return True

        return False

    def _parse_education_block(self, block: str) -> Optional[ExtractedEducation]:
        """Parse a single education block."""
        if not block.strip():
            return None

        entry = ExtractedEducation(raw_text=block)

        # Extract degree and level
        degree_info = self._extract_degree(block)
        if degree_info:
            entry.degree = degree_info.get("degree")
            entry.degree_level = degree_info.get("level")

        # Extract field of study
        entry.field_of_study = self._extract_field_of_study(block)

        # Extract institution
        entry.institution = self._extract_institution(block)

        # Extract dates
        date_info = self._extract_dates(block)
        if date_info:
            entry.start_date = date_info.get("start")
            entry.graduation_date = date_info.get("end")

        # Extract GPA
        entry.gpa = self._extract_gpa(block)

        # Extract honors
        entry.honors = self._extract_honors(block)

        # Extract location
        entry.location = self._extract_location(block)

        # Extract coursework
        entry.relevant_coursework = self._extract_coursework(block)

        # Calculate confidence
        entry.confidence = self._calculate_confidence(entry)

        return entry

    def _extract_degree(self, text: str) -> Optional[dict]:
        """Extract degree and its level from text."""
        text_lower = text.lower()

        for level, patterns in self.DEGREE_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Find the line that contains the degree keyword and use
                    # it as the degree name — this handles titles like
                    # "Bachelor of Technology (B.Tech) in Computer Science"
                    for line in text.splitlines():
                        if re.search(pattern, line, re.IGNORECASE):
                            # Strip pipe-separated metadata (dates, GPA, etc.)
                            degree_text = re.split(r"\s*[|\-–—]\s*", line)[0]
                            # Remove trailing parenthetical abbreviations only
                            # if the main name is already long enough
                            degree_text = degree_text.strip(" ,.\n")
                            if degree_text and len(degree_text) >= 2:
                                return {
                                    "degree": degree_text,
                                    "level": level,
                                }
                    return {
                        "degree": match.group(0),
                        "level": level,
                    }

        return None

    def _extract_field_of_study(self, text: str) -> Optional[str]:
        """Extract field of study from text."""
        text_lower = text.lower()

        for field in self.FIELDS_OF_STUDY:
            if field.lower() in text_lower:
                # Get original case from text
                pattern = re.compile(re.escape(field), re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    return match.group(0)

        # Try to extract from "in [field]" pattern
        in_pattern = re.compile(
            r"(?:in|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        )
        match = in_pattern.search(text)
        if match:
            return match.group(1)

        return None

    def _extract_institution(self, text: str) -> Optional[str]:
        """Extract institution name from text."""
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            line_lower = line.lower()

            # Check if line contains institution indicators
            if any(ind in line_lower for ind in self.INSTITUTION_INDICATORS):
                # Clean up the line
                # Remove dates
                clean = self.DATE_PATTERN.sub("", line)
                # Remove common suffixes
                clean = re.sub(
                    r"\s*[-–—|,]\s*(?:present|current|expected).*$",
                    "",
                    clean,
                    flags=re.IGNORECASE,
                )
                clean = clean.strip(" ,.-–—|")

                if clean and len(clean) > 3:
                    return clean

        return None

    def _extract_dates(self, text: str) -> Optional[dict]:
        """Extract dates from education text."""
        dates = self.DATE_PATTERN.findall(text)

        if len(dates) >= 2:
            return {
                "start": self._parse_date(dates[0]),
                "end": self._parse_date(dates[1]),
            }
        elif len(dates) == 1:
            # Single date is likely graduation date
            return {
                "start": None,
                "end": self._parse_date(dates[0]),
            }

        # Check for "expected" graduation
        expected_pattern = re.compile(
            r"expected\s+(?:graduation\s+)?(\d{4})",
            re.IGNORECASE,
        )
        match = expected_pattern.search(text)
        if match:
            year = int(match.group(1))
            return {
                "start": None,
                "end": date(year, 5, 1),  # Assume May graduation
            }

        return None

    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse a date string into a date object."""
        if not date_str:
            return None

        date_str = date_str.strip().lower()

        month_names = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec",
        ]

        month = 5  # Default to May (common graduation month)

        # Check for month name
        for i, month_name in enumerate(month_names):
            if month_name in date_str:
                month = (i % 12) + 1
                break

        # Extract year
        year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if year_match:
            year = int(year_match.group(0))
            try:
                return date(year, month, 1)
            except ValueError:
                return None

        return None

    def _extract_gpa(self, text: str) -> Optional[float]:
        """Extract GPA from text."""
        match = self.GPA_PATTERN.search(text)
        if match:
            gpa = float(match.group(1))
            scale = float(match.group(2)) if match.group(2) else 4.0

            # Normalize to 4.0 scale if needed
            if scale > 4.0:
                gpa = (gpa / scale) * 4.0

            # Validate range
            if 0 <= gpa <= 4.0:
                return round(gpa, 2)

        return None

    def _extract_honors(self, text: str) -> Optional[str]:
        """Extract honors/distinctions from text."""
        text_lower = text.lower()

        for pattern in self.HONORS_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                # Get original case from text
                original = re.search(pattern, text, re.IGNORECASE)
                if original:
                    return original.group(0)

        return None

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from text."""
        # Pattern for City, State
        location_pattern = re.compile(
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*,\s*([A-Z]{2})\b"
        )
        match = location_pattern.search(text)
        if match:
            return f"{match.group(1)}, {match.group(2)}"

        return None

    def _extract_coursework(self, text: str) -> list[str]:
        """Extract relevant coursework from text."""
        coursework = []

        # Look for coursework section
        coursework_pattern = re.compile(
            r"(?:relevant\s+)?coursework[:\s]+(.+?)(?:\n\n|\n[A-Z]|$)",
            re.IGNORECASE | re.DOTALL,
        )
        match = coursework_pattern.search(text)
        if match:
            courses_text = match.group(1)
            # Split by common delimiters
            courses = re.split(r"[,;|•\n]", courses_text)
            for course in courses:
                course = course.strip()
                if course and len(course) > 3 and len(course) < 100:
                    coursework.append(course)

        return coursework[:10]  # Limit to 10 courses

    def _get_highest_level(
        self, entries: list[ExtractedEducation]
    ) -> Optional[str]:
        """Get the highest education level from entries."""
        highest_level = 0
        highest_name = None

        for entry in entries:
            if entry.degree_level:
                level_value = EDUCATION_LEVELS.get(entry.degree_level, 0)
                if level_value > highest_level:
                    highest_level = level_value
                    highest_name = entry.degree_level

        return highest_name

    def _calculate_confidence(self, entry: ExtractedEducation) -> float:
        """Calculate confidence score for an education entry."""
        score = 0.0

        if entry.degree or entry.degree_level:
            score += 0.3
        if entry.institution:
            score += 0.3
        if entry.field_of_study:
            score += 0.15
        if entry.graduation_date:
            score += 0.15
        if entry.gpa:
            score += 0.05
        if entry.honors:
            score += 0.05

        return min(score, 1.0)
