"""
Work experience parser for resumes.

Extracts job titles, companies, dates, and responsibilities.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedExperience:
    """A work experience entry extracted from a resume."""

    job_title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_current: bool = False
    description: Optional[str] = None
    responsibilities: list[str] = field(default_factory=list)
    achievements: list[str] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0


@dataclass
class ExperienceParseResult:
    """Result of experience parsing."""

    experiences: list[ExtractedExperience] = field(default_factory=list)
    total_years: float = 0.0
    confidence: float = 0.0


class ExperienceParser:
    """Parser for extracting work experience from resume text."""

    # Common job title patterns
    JOB_TITLE_KEYWORDS = [
        "engineer", "developer", "manager", "director", "analyst",
        "specialist", "consultant", "architect", "designer", "lead",
        "senior", "junior", "associate", "principal", "staff",
        "intern", "trainee", "coordinator", "administrator",
        "executive", "officer", "president", "vice president", "vp",
        "head", "chief", "cto", "ceo", "cfo", "coo", "cio",
        "scientist", "researcher", "professor", "instructor",
        "technician", "operator", "assistant", "support",
    ]

    # Date patterns
    MONTH_NAMES = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ]

    # Pattern for dates like "Jan 2020", "January 2020", "01/2020", "2020"
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

    # Pattern for date ranges
    DATE_RANGE_PATTERN = re.compile(
        r"("
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"[,.\s]*\d{4}|\d{1,2}[/\-]\d{4}|\d{4}"
        r")"
        r"\s*[-–—to]+\s*"
        r"("
        r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"[,.\s]*\d{4}|\d{1,2}[/\-]\d{4}|\d{4}|present|current|now|ongoing"
        r")",
        re.IGNORECASE,
    )

    # Bullet point patterns
    BULLET_PATTERN = re.compile(r"^[\s]*[•\-\*\u2022\u2023\u25E6\u2043\u2219]\s*")

    def parse(
        self,
        text: str,
        experience_section: Optional[str] = None,
    ) -> ExperienceParseResult:
        """
        Parse work experience from resume text.

        Args:
            text: Full resume text
            experience_section: Optional dedicated experience section text

        Returns:
            ExperienceParseResult with extracted experiences
        """
        # Use experience section if provided, otherwise use full text
        parse_text = experience_section if experience_section else text

        experiences = self._extract_experiences(parse_text)

        # Calculate total years
        total_years = self._calculate_total_years(experiences)

        # Calculate confidence
        if experiences:
            avg_confidence = sum(e.confidence for e in experiences) / len(experiences)
        else:
            avg_confidence = 0.0

        return ExperienceParseResult(
            experiences=experiences,
            total_years=total_years,
            confidence=avg_confidence,
        )

    def _extract_experiences(self, text: str) -> list[ExtractedExperience]:
        """Extract individual experience entries from text."""
        experiences = []

        # Split text into potential experience blocks
        blocks = self._split_into_blocks(text)

        for block in blocks:
            experience = self._parse_experience_block(block)
            if experience and (experience.job_title or experience.company):
                experiences.append(experience)

        # Sort by date (most recent first)
        experiences.sort(
            key=lambda e: e.start_date or date.min,
            reverse=True,
        )

        return experiences

    def _split_into_blocks(self, text: str) -> list[str]:
        """Split text into potential experience blocks."""
        lines = text.split("\n")
        blocks = []
        current_block = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()

            if not line_stripped:
                if current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []
                continue

            # Check if this line looks like a new entry header
            if self._is_entry_header(line_stripped, i, lines):
                if current_block:
                    blocks.append("\n".join(current_block))
                    current_block = []

            current_block.append(line)

        if current_block:
            blocks.append("\n".join(current_block))

        return blocks

    def _is_entry_header(
        self, line: str, index: int, all_lines: list[str]
    ) -> bool:
        """Check if a line looks like a new experience entry header."""
        # Contains a date range
        if self.DATE_RANGE_PATTERN.search(line):
            return True

        # Contains job title keywords and is relatively short
        line_lower = line.lower()
        has_title_keyword = any(
            keyword in line_lower for keyword in self.JOB_TITLE_KEYWORDS
        )

        if has_title_keyword and len(line) < 100:
            # Additional check: next line might have a date or company
            if index + 1 < len(all_lines):
                next_line = all_lines[index + 1].strip()
                if self.DATE_PATTERN.search(next_line) or self._looks_like_company(next_line):
                    return True

            return True

        return False

    def _looks_like_company(self, text: str) -> bool:
        """Check if text looks like a company name."""
        company_indicators = [
            "inc", "llc", "ltd", "corp", "corporation", "company",
            "co.", "technologies", "solutions", "systems", "group",
            "consulting", "services", "partners", "associates",
        ]
        text_lower = text.lower()
        return any(ind in text_lower for ind in company_indicators)

    def _parse_experience_block(self, block: str) -> Optional[ExtractedExperience]:
        """Parse a single experience block."""
        lines = block.split("\n")
        if not lines:
            return None

        experience = ExtractedExperience(raw_text=block)

        # Extract date range
        date_info = self._extract_dates(block)
        if date_info:
            experience.start_date = date_info.get("start")
            experience.end_date = date_info.get("end")
            experience.is_current = date_info.get("is_current", False)

        # Extract job title and company from first few lines
        header_lines = lines[:3]
        title_company = self._extract_title_and_company(header_lines, block)
        if title_company:
            experience.job_title = title_company.get("title")
            experience.company = title_company.get("company")
            experience.location = title_company.get("location")

        # Extract responsibilities and achievements
        bullet_points = self._extract_bullet_points(lines)
        for point in bullet_points:
            if self._is_achievement(point):
                experience.achievements.append(point)
            else:
                experience.responsibilities.append(point)

        # Build description
        if not bullet_points:
            # Use non-header lines as description
            desc_lines = [
                l.strip() for l in lines[2:]
                if l.strip() and not self.BULLET_PATTERN.match(l)
            ]
            if desc_lines:
                experience.description = " ".join(desc_lines)

        # Calculate confidence
        experience.confidence = self._calculate_confidence(experience)

        return experience

    # Pattern for "Month-Month Year" shorthand like "Apr–Jul 2024"
    COMPACT_RANGE_PATTERN = re.compile(
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)"
        r"\s*[-–—/]\s*"
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*)"
        r"\s+(\d{4})",
        re.IGNORECASE,
    )

    def _extract_dates(self, text: str) -> Optional[dict]:
        """Extract date range from text."""
        # Try compact "Mon-Mon Year" format first (e.g. "Apr–Jul 2024")
        compact = self.COMPACT_RANGE_PATTERN.search(text)
        if compact:
            start_date = self._parse_date(f"{compact.group(1)} {compact.group(3)}")
            end_date = self._parse_date(f"{compact.group(2)} {compact.group(3)}")
            return {"start": start_date, "end": end_date, "is_current": False}

        # Try to find date range first
        range_match = self.DATE_RANGE_PATTERN.search(text)
        if range_match:
            start_str = range_match.group(1)
            end_str = range_match.group(2)

            start_date = self._parse_date(start_str)
            is_current = end_str.lower() in ["present", "current", "now", "ongoing"]
            end_date = None if is_current else self._parse_date(end_str)

            return {
                "start": start_date,
                "end": end_date,
                "is_current": is_current,
            }

        # Try to find individual dates
        dates = self.DATE_PATTERN.findall(text)
        if len(dates) >= 2:
            return {
                "start": self._parse_date(dates[0]),
                "end": self._parse_date(dates[1]),
                "is_current": False,
            }
        elif len(dates) == 1:
            return {
                "start": self._parse_date(dates[0]),
                "end": None,
                "is_current": True,
            }

        return None

    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse a date string into a date object."""
        if not date_str:
            return None

        date_str = date_str.strip().lower()

        # Handle "present", "current", etc.
        if date_str in ["present", "current", "now", "ongoing"]:
            return None

        # Try to extract month and year
        month = 1  # Default to January
        year = None

        # Check for month name
        for i, month_name in enumerate(self.MONTH_NAMES):
            if month_name in date_str:
                month = (i % 12) + 1
                break

        # Extract year
        year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if year_match:
            year = int(year_match.group(0))

        if year:
            try:
                return date(year, month, 1)
            except ValueError:
                return None

        return None

    def _extract_title_and_company(
        self, header_lines: list[str], full_text: str
    ) -> Optional[dict]:
        """Extract job title and company from header lines."""
        result = {}

        for line in header_lines:
            line = line.strip()
            if not line:
                continue

            # Strip leading numbered list prefix: "1.", "2.", "1)", etc.
            line = re.sub(r"^\d+[.)]\s*", "", line)

            # Handle pipe-separated format: "Title - Company | Date" or
            # "Title | Organization | Date".  Extract segments before dates.
            if "|" in line:
                segments = [s.strip() for s in line.split("|")]
                for seg in segments:
                    seg_clean = self.DATE_RANGE_PATTERN.sub("", seg)
                    seg_clean = self.DATE_PATTERN.sub("", seg_clean).strip(" -–—,")
                    if not seg_clean:
                        continue
                    if not result.get("title"):
                        # First segment is title (may contain "Title - Company")
                        if " - " in seg_clean:
                            parts = seg_clean.split(" - ", 1)
                            result["title"] = parts[0].strip()
                            if not result.get("company"):
                                result["company"] = parts[1].strip()
                        else:
                            result["title"] = seg_clean
                    elif not result.get("company"):
                        result["company"] = seg_clean
                continue

            # Remove dates from line for cleaner parsing
            clean_line = self.DATE_RANGE_PATTERN.sub("", line)
            clean_line = self.DATE_PATTERN.sub("", clean_line)
            clean_line = clean_line.strip(" -–—|,")

            if not clean_line:
                continue

            # Check if this line looks like a job title
            line_lower = clean_line.lower()
            has_title = any(kw in line_lower for kw in self.JOB_TITLE_KEYWORDS)

            # Handle "Title - Company" on same line
            if " - " in clean_line and not result.get("title"):
                parts = clean_line.split(" - ", 1)
                result["title"] = parts[0].strip()
                if not result.get("company"):
                    result["company"] = parts[1].strip()
                continue

            # Check if this line looks like a company
            has_company = self._looks_like_company(clean_line)

            if has_title and not result.get("title"):
                result["title"] = clean_line
            elif has_company and not result.get("company"):
                result["company"] = clean_line
            elif not result.get("title") and not result.get("company"):
                # First non-empty line might be title
                if len(clean_line.split()) <= 8:
                    result["title"] = clean_line

        # Try to extract location
        location_pattern = re.compile(
            r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*,\s*([A-Z]{2})\b"
        )
        loc_match = location_pattern.search(full_text)
        if loc_match:
            result["location"] = f"{loc_match.group(1)}, {loc_match.group(2)}"

        return result if result else None

    def _extract_bullet_points(self, lines: list[str]) -> list[str]:
        """Extract bullet points from lines."""
        bullets = []

        for line in lines:
            line = line.strip()
            # Check if line starts with a bullet
            if self.BULLET_PATTERN.match(line):
                # Remove the bullet and clean
                clean = self.BULLET_PATTERN.sub("", line).strip()
                if clean and len(clean) > 10:
                    bullets.append(clean)

        return bullets

    def _is_achievement(self, text: str) -> bool:
        """Check if a bullet point is an achievement vs responsibility."""
        achievement_indicators = [
            "increased", "decreased", "improved", "reduced", "saved",
            "achieved", "awarded", "won", "recognized", "promoted",
            "generated", "grew", "expanded", "launched", "delivered",
            "%", "percent", "million", "billion", "revenue",
        ]
        text_lower = text.lower()
        return any(ind in text_lower for ind in achievement_indicators)

    def _calculate_total_years(
        self, experiences: list[ExtractedExperience]
    ) -> float:
        """Calculate total years of experience, merging overlapping date ranges.

        Concurrent jobs (e.g. a part-time role alongside a full-time role) would
        otherwise be double-counted. We sort all intervals by start date, merge
        any that overlap, then sum the merged spans.
        """
        today = date.today()
        intervals: list[tuple[date, date]] = []

        for exp in experiences:
            if exp.start_date:
                intervals.append((exp.start_date, exp.end_date or today))

        if not intervals:
            return 0.0

        # Sort by start date then merge overlapping / adjacent intervals.
        intervals.sort(key=lambda x: x[0])
        merged: list[tuple[date, date]] = [intervals[0]]
        for start, end in intervals[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                # Overlapping or adjacent — extend the previous interval if needed.
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        total_months = 0
        for start, end in merged:
            delta = end - start
            total_months += max(1, delta.days // 30)

        return round(total_months / 12, 1)

    def _calculate_confidence(self, experience: ExtractedExperience) -> float:
        """Calculate confidence score for an experience entry."""
        score = 0.0

        if experience.job_title:
            score += 0.3
        if experience.company:
            score += 0.25
        if experience.start_date:
            score += 0.2
        if experience.responsibilities or experience.achievements:
            score += 0.15
        if experience.location:
            score += 0.1

        return min(score, 1.0)
