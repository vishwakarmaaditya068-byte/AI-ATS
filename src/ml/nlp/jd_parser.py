"""
Job Description parser.

Extracts requirements, skills, and qualifications from job descriptions.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.data.models import (
    EducationRequirement,
    EmploymentType,
    ExperienceLevel,
    ExperienceRequirement,
    Job,
    JobCreate,
    SkillRequirement,
)
from src.utils.constants import EDUCATION_LEVELS, SKILL_CATEGORIES
from src.utils.logger import get_logger

from .extractors import ExtractorFactory
from .parsers import SkillsParser

logger = get_logger(__name__)


@dataclass
class JDParseResult:
    """Complete result of job description parsing."""

    file_path: Optional[str] = None
    raw_text: str = ""

    # Extracted information
    title: str = ""
    company_name: str = ""
    description: str = ""
    responsibilities: list[str] = field(default_factory=list)
    qualifications: list[str] = field(default_factory=list)

    # Requirements
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    experience_years_min: Optional[float] = None
    experience_years_max: Optional[float] = None
    education_requirement: Optional[str] = None

    # Employment details
    employment_type: Optional[str] = None
    experience_level: Optional[str] = None
    location: Optional[str] = None

    # Quality metrics
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if parsing was successful."""
        return len(self.errors) == 0 and bool(self.title or self.required_skills)

    @property
    def all_skills(self) -> list[str]:
        """Get all skills (required + preferred)."""
        return list(set(self.required_skills + self.preferred_skills))


class JDParser:
    """
    Parser for job descriptions.

    Extracts requirements, skills, and qualifications from job description documents.
    """

    # Patterns for section detection
    SECTION_PATTERNS = {
        "responsibilities": [
            r"responsibilities?\s*[:|-]?",
            r"what\s+you['\u2019]?ll\s+do",
            r"duties?\s*[:|-]?",
            r"role\s+responsibilities",
            r"key\s+responsibilities",
            r"your\s+responsibilities",
        ],
        "requirements": [
            r"requirements?\s*[:|-]?",
            r"qualifications?\s*[:|-]?",
            r"what\s+you['\u2019]?ll\s+need",
            r"what\s+we['\u2019]?re\s+looking\s+for",
            r"must\s+have",
            r"required\s+skills?",
            r"minimum\s+qualifications?",
        ],
        "preferred": [
            r"preferred\s+qualifications?",
            r"nice\s+to\s+have",
            r"bonus\s+points?",
            r"preferred\s+skills?",
            r"desired\s+qualifications?",
        ],
        "benefits": [
            r"benefits?\s*[:|-]?",
            r"what\s+we\s+offer",
            r"perks?\s*[:|-]?",
            r"compensation",
        ],
    }

    # Patterns for experience extraction (range pattern before single-number to avoid shadowing)
    EXPERIENCE_PATTERNS = [
        r"(\d+)\s*[-–]\s*(\d+)\s*(?:years?|yrs?)\s*(?:of)?\s*experience",
        r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:relevant|professional|industry|work)?\s*experience",
        r"minimum\s*(?:of)?\s*(\d+)\s*(?:years?|yrs?)",
        r"at\s+least\s+(\d+)\s*(?:years?|yrs?)",
        r"(\d+)\s*(?:years?|yrs?)\s*(?:or\s+more|minimum)",
    ]

    # Employment type keywords
    EMPLOYMENT_KEYWORDS = {
        "full_time": ["full-time", "full time", "permanent", "regular"],
        "part_time": ["part-time", "part time"],
        "contract": ["contract", "contractor", "consulting"],
        "internship": ["intern", "internship", "trainee", "co-op", "coop"],
        "temporary": ["temporary", "temp"],
        "freelance": ["freelance", "freelancer"],
    }

    # Experience level keywords
    LEVEL_KEYWORDS = {
        "entry": ["entry level", "entry-level", "junior", "fresher", "graduate", "0-2 years"],
        "mid": ["mid level", "mid-level", "intermediate", "2-5 years", "3-5 years"],
        "senior": ["senior", "sr.", "experienced", "5+ years", "5-10 years"],
        "lead": ["lead", "principal", "staff", "10+ years"],
        "executive": ["executive", "director", "vp", "c-level", "cto", "cio"],
    }

    def __init__(self):
        """Initialize the JD parser."""
        try:
            import spacy
            spacy.prefer_gpu()
        except Exception:
            # spaCy is optional — used only as a GPU-acceleration hint.
            # On Python 3.14 it raises pydantic.v1.errors.ConfigError, not
            # ImportError, so we catch all exceptions to stay portable.
            pass
        self.skills_parser = SkillsParser()
        self._build_skill_set()

    def _build_skill_set(self) -> None:
        """Build a set of known skills for quick lookup."""
        self.known_skills: set[str] = set()
        for skills in SKILL_CATEGORIES.values():
            self.known_skills.update(s.lower() for s in skills)
        # Add additional skills from skills parser
        for skills in self.skills_parser.ADDITIONAL_SKILLS.values():
            self.known_skills.update(s.lower() for s in skills)

    def parse_file(self, file_path: str | Path) -> JDParseResult:
        """
        Parse a job description from a file.

        Args:
            file_path: Path to the JD file (PDF, DOCX, TXT)

        Returns:
            JDParseResult with extracted information
        """
        result = JDParseResult(file_path=str(file_path))

        try:
            path = Path(file_path)
            extraction = ExtractorFactory.extract(path)

            if not extraction.success:
                result.errors.append(f"Extraction failed: {extraction.error_message}")
                return result

            if extraction.is_empty:
                result.errors.append("Extracted text is empty")
                return result

            result.raw_text = extraction.text
            result.warnings.extend(extraction.warnings)

            return self._process_text(extraction.text, result)

        except Exception as e:
            logger.exception(f"Error parsing JD: {e}")
            result.errors.append(str(e))
            return result

    def parse_text(self, text: str) -> JDParseResult:
        """
        Parse a job description from raw text.

        Args:
            text: Job description text content

        Returns:
            JDParseResult with extracted information
        """
        result = JDParseResult(raw_text=text)
        return self._process_text(text, result)

    def _process_text(self, text: str, result: JDParseResult) -> JDParseResult:
        """Process JD text and extract all information."""
        # Extract title (usually first non-empty line or prominent text)
        result.title = self._extract_title(text)

        # Extract company name
        result.company_name = self._extract_company(text)

        # Extract sections
        sections = self._detect_sections(text)

        # Extract responsibilities
        if "responsibilities" in sections:
            result.responsibilities = self._extract_bullet_points(sections["responsibilities"])

        # Extract requirements and qualifications
        if "requirements" in sections:
            result.qualifications = self._extract_bullet_points(sections["requirements"])

        # Extract skills
        result.required_skills, result.preferred_skills = self._extract_skills(
            text, sections.get("requirements", ""), sections.get("preferred", "")
        )

        # Extract experience requirement
        exp_min, exp_max = self._extract_experience(text)
        result.experience_years_min = exp_min
        result.experience_years_max = exp_max

        # Extract education requirement
        result.education_requirement = self._extract_education(text)

        # Detect employment type
        result.employment_type = self._detect_employment_type(text)

        # Detect experience level
        result.experience_level = self._detect_experience_level(text, exp_min)

        # Extract location
        result.location = self._extract_location(text)

        # Build description from sections
        if not result.description:
            result.description = self._build_description(text, sections)

        # Calculate confidence
        result.confidence = self._calculate_confidence(result)

        return result

    def _extract_title(self, text: str) -> str:
        """Extract job title from text."""
        lines = text.strip().split("\n")

        # Common title patterns
        title_patterns = [
            r"job\s*title\s*[:|-]?\s*(.+)",
            r"position\s*[:|-]?\s*(.+)",
            r"role\s*[:|-]?\s*(.+)",
            r"hiring\s*[:|-]?\s*(.+)",
        ]

        for pattern in title_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                if len(title) < 100:  # Reasonable title length
                    return title

        # Fall back to first non-empty line that looks like a title
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and 3 < len(line) < 100 and not line.endswith(":"):
                # Check if it's not a date or generic text
                if not re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", line):
                    return line

        return "Unknown Position"

    def _extract_company(self, text: str) -> str:
        """Extract company name from text."""
        patterns = [
            r"company\s*[:|-]?\s*(.+)",
            r"organization\s*[:|-]?\s*(.+)",
            r"employer\s*[:|-]?\s*(.+)",
            r"about\s+(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                # Take only first part if it's too long
                company = company.split("\n")[0][:50]
                if company:
                    return company

        return "Unknown Company"

    def _detect_sections(self, text: str) -> dict[str, str]:
        """Detect and extract sections from the text."""
        sections: dict[str, str] = {}

        for section_name, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    start = match.end()
                    # Find where the next section starts
                    end = len(text)
                    for other_name, other_patterns in self.SECTION_PATTERNS.items():
                        if other_name != section_name:
                            for other_pattern in other_patterns:
                                other_match = re.search(
                                    other_pattern, text[start:],
                                    re.IGNORECASE | re.MULTILINE
                                )
                                if other_match and start + other_match.start() < end:
                                    end = start + other_match.start()

                    sections[section_name] = text[start:end].strip()
                    break

        return sections

    def _extract_bullet_points(self, text: str) -> list[str]:
        """Extract bullet points or numbered items from text."""
        points = []

        # Split by bullet points or newlines
        lines = re.split(r"[\n•●◦○▪▸►\-\*]|\d+[.)]", text)

        for line in lines:
            line = line.strip()
            # Filter out empty lines and very short ones
            if line and len(line) > 10:
                # Clean up the line
                line = re.sub(r"^\s*[-•*]\s*", "", line)
                points.append(line)

        return points[:20]  # Limit to 20 points

    def _extract_skills(
        self,
        full_text: str,
        requirements_section: str,
        preferred_section: str,
    ) -> tuple[list[str], list[str]]:
        """Extract required and preferred skills."""
        required_skills: set[str] = set()
        preferred_skills: set[str] = set()

        # Parse skills from requirements section (text-only, no skills_section to avoid
        # treating bullet-point lines as skill candidates)
        if requirements_section:
            result = self.skills_parser.parse(requirements_section)
            for skill in result.skills:
                required_skills.add(skill.name.lower())

        # Parse skills from preferred section
        if preferred_section:
            result = self.skills_parser.parse(preferred_section)
            for skill in result.skills:
                preferred_skills.add(skill.name.lower())

        # Also extract from full text to catch any missed skills
        full_result = self.skills_parser.parse(full_text)
        for skill in full_result.skills:
            skill_lower = skill.name.lower()
            if skill_lower not in required_skills and skill_lower not in preferred_skills:
                required_skills.add(skill_lower)

        # Remove preferred skills from required if they appear in both
        preferred_skills -= required_skills

        return list(required_skills), list(preferred_skills)

    def _extract_experience(self, text: str) -> tuple[Optional[float], Optional[float]]:
        """Extract experience requirement in years."""
        for pattern in self.EXPERIENCE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2 and groups[1]:  # Range pattern
                    return float(groups[0]), float(groups[1])
                elif groups[0]:  # Single number pattern
                    return float(groups[0]), None

        return None, None

    def _extract_education(self, text: str) -> Optional[str]:
        """Extract education requirement."""
        text_lower = text.lower()

        # Check for education keywords in order of level (highest first)
        education_checks = [
            ("phd", "PhD"),
            ("doctorate", "Doctorate"),
            ("master", "Master's"),
            ("mba", "MBA"),
            ("bachelor", "Bachelor's"),
            ("associate", "Associate"),
            ("diploma", "Diploma"),
        ]

        for keyword, level in education_checks:
            if keyword in text_lower:
                # Check if it's mentioned as required
                if re.search(
                    rf"(require|must\s+have|need|minimum).*{keyword}|{keyword}.*(required|needed|preferred)",
                    text_lower,
                ):
                    return level

        # Also check for degree patterns
        degree_match = re.search(
            r"(bachelor'?s?|master'?s?|phd|doctorate|mba|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?)\s*(degree|in|of)?",
            text_lower,
        )
        if degree_match:
            degree = degree_match.group(1)
            if "bachelor" in degree or "b." in degree:
                return "Bachelor's"
            elif "master" in degree or "m." in degree:
                return "Master's"
            elif "phd" in degree or "doctor" in degree:
                return "PhD"
            elif "mba" in degree:
                return "MBA"

        return None

    def _detect_employment_type(self, text: str) -> Optional[str]:
        """Detect employment type from text."""
        text_lower = text.lower()

        for emp_type, keywords in self.EMPLOYMENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return emp_type

        return "full_time"  # Default

    def _detect_experience_level(
        self, text: str, min_years: Optional[float]
    ) -> Optional[str]:
        """Detect experience level from text and years."""
        text_lower = text.lower()

        # Check keywords first
        for level, keywords in self.LEVEL_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return level

        # Infer from years if not found
        if min_years is not None:
            if min_years <= 2:
                return "entry"
            elif min_years <= 5:
                return "mid"
            elif min_years <= 10:
                return "senior"
            else:
                return "lead"

        return "mid"  # Default

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract job location from text."""
        patterns = [
            r"location\s*[:|-]?\s*(.+?)(?:\n|$)",
            r"based\s+(?:in|at)\s+(.+?)(?:\n|$|,)",
            r"work\s+(?:from|in)\s+(.+?)(?:\n|$|,)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                if len(location) < 100:
                    return location

        # Check for remote
        if re.search(r"\b(remote|work\s+from\s+home|wfh)\b", text, re.IGNORECASE):
            return "Remote"

        return None

    def _build_description(self, text: str, sections: dict[str, str]) -> str:
        """Build a description from available text."""
        # Try to get description from before sections
        first_section_start = len(text)
        for section_content in sections.values():
            if section_content:
                idx = text.find(section_content[:50])
                if idx >= 0 and idx < first_section_start:
                    first_section_start = idx

        description = text[:first_section_start].strip()

        # Clean up
        lines = description.split("\n")
        # Skip title line
        if len(lines) > 1:
            description = "\n".join(lines[1:]).strip()

        return description[:2000] if description else text[:500]

    def _calculate_confidence(self, result: JDParseResult) -> float:
        """Calculate parsing confidence score."""
        score = 0.0

        if result.title and result.title != "Unknown Position":
            score += 0.2
        if result.required_skills:
            score += 0.3 * min(len(result.required_skills) / 5, 1.0)
        if result.experience_years_min is not None:
            score += 0.15
        if result.education_requirement:
            score += 0.1
        if result.responsibilities:
            score += 0.15
        if result.qualifications:
            score += 0.1

        return round(min(score, 1.0), 2)

    def to_job_create(self, result: JDParseResult) -> JobCreate:
        """Convert parse result to JobCreate schema."""
        # Build skill requirements
        skill_requirements = []
        for skill in result.required_skills:
            skill_requirements.append(
                SkillRequirement(name=skill, is_required=True, weight=1.0)
            )
        for skill in result.preferred_skills:
            skill_requirements.append(
                SkillRequirement(name=skill, is_required=False, weight=0.5)
            )

        # Build experience requirement
        experience_requirement = None
        if result.experience_years_min is not None:
            experience_requirement = ExperienceRequirement(
                minimum_years=result.experience_years_min,
                maximum_years=result.experience_years_max,
            )

        # Build education requirement
        education_requirement = None
        if result.education_requirement:
            education_requirement = EducationRequirement(
                minimum_degree=result.education_requirement.lower(),
                is_required=True,
            )

        # Map employment type
        employment_type = EmploymentType.FULL_TIME
        if result.employment_type:
            try:
                employment_type = EmploymentType(result.employment_type)
            except ValueError:
                pass

        # Map experience level
        experience_level = ExperienceLevel.MID
        if result.experience_level:
            try:
                experience_level = ExperienceLevel(result.experience_level)
            except ValueError:
                pass

        return JobCreate(
            title=result.title or "Unknown Position",
            description=result.description or result.raw_text[:500],
            responsibilities=result.responsibilities,
            company_name=result.company_name or "Unknown Company",
            employment_type=employment_type,
            experience_level=experience_level,
            skill_requirements=skill_requirements,
            education_requirement=education_requirement,
            experience_requirement=experience_requirement,
        )


# Singleton instance
_jd_parser: Optional[JDParser] = None


def get_jd_parser() -> JDParser:
    """Get the JD parser singleton instance."""
    global _jd_parser
    if _jd_parser is None:
        _jd_parser = JDParser()
    return _jd_parser
