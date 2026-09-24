"""
Projects parser for resumes.

Extracts project names, descriptions, technologies, and URLs
from a projects section of a resume.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedProject:
    """A project extracted from a resume."""

    name: Optional[str] = None
    description: Optional[str] = None
    technologies: list[str] = field(default_factory=list)
    url: Optional[str] = None
    raw_text: str = ""
    confidence: float = 0.0


@dataclass
class ProjectsParseResult:
    """Result of projects parsing."""

    projects: list[ExtractedProject] = field(default_factory=list)
    confidence: float = 0.0


class ProjectsParser:
    """Parser for extracting projects from resume text."""

    # GitHub / GitLab / Bitbucket repo URLs
    REPO_URL_PATTERN = re.compile(
        r"(?:https?://)?(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/[\w\-./]+",
        re.IGNORECASE,
    )

    # General HTTPS URL
    GENERAL_URL_PATTERN = re.compile(r"https?://[\w\-./?=#&%+]+", re.IGNORECASE)

    # "Tech stack:", "Technologies:", "Built with:", etc.
    TECH_LABEL_PATTERN = re.compile(
        r"(?:tech(?:nologies|nology|nical\s*stack)?|stack|tools?\s*used|built\s*with"
        r"|technologies\s*used|tech\s*stack)[:\s]+(.+?)(?:\n|$)",
        re.IGNORECASE,
    )

    # Numbered or bulleted project headers (e.g. "1. Project Name" or "• Project Name")
    NUMBERED_HEADER = re.compile(
        r"^\s*(?:\d+[.)]\s*|[•\-\*]\s*)([A-Z][^\n]{3,60})$",
        re.MULTILINE,
    )

    # Lines to skip when extracting description
    SKIP_LINE_PATTERNS = [
        re.compile(r"^\s*(?:https?://|github|gitlab|bitbucket)", re.IGNORECASE),
        re.compile(r"tech(?:nologies|nology|nical\s*stack)?[:\s]", re.IGNORECASE),
        re.compile(r"built\s*with[:\s]", re.IGNORECASE),
        re.compile(r"^\s*[\d.)\-•*]\s*$"),
    ]

    def __init__(self, skills_parser=None):
        """
        Initialize the projects parser.

        Args:
            skills_parser: Optional SkillsParser instance to reuse for
                           technology extraction. A new one is created if not provided.
        """
        if skills_parser is None:
            from src.ml.nlp.parsers.skills_parser import SkillsParser
            self._skills_parser = SkillsParser()
        else:
            self._skills_parser = skills_parser

    def parse(self, section_text: str) -> ProjectsParseResult:
        """Parse projects from a section of resume text."""
        if not section_text or not section_text.strip():
            return ProjectsParseResult()

        blocks = self._split_into_blocks(section_text)
        projects = []
        for block in blocks:
            project = self._parse_project_block(block)
            if project and project.name:
                projects.append(project)

        if projects:
            avg_conf = sum(p.confidence for p in projects) / len(projects)
        else:
            avg_conf = 0.0

        return ProjectsParseResult(projects=projects, confidence=avg_conf)

    def _split_into_blocks(self, text: str) -> list[str]:
        """Split text into project blocks."""
        # Primary: split on blank lines
        blocks = re.split(r"\n\s*\n", text.strip())
        blocks = [b.strip() for b in blocks if b.strip()]

        # Fallback: if only one block but has numbered headers, split on those
        if len(blocks) <= 1 and blocks:
            headers = list(self.NUMBERED_HEADER.finditer(text))
            if len(headers) >= 2:
                sub_blocks = []
                for i, match in enumerate(headers):
                    start = match.start()
                    end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
                    sub_blocks.append(text[start:end].strip())
                return [b for b in sub_blocks if b]

        return blocks

    def _parse_project_block(self, block: str) -> Optional[ExtractedProject]:
        """Extract a single project from a text block."""
        if not block:
            return None

        project = ExtractedProject(raw_text=block)
        lines = [ln for ln in block.splitlines() if ln.strip()]

        if not lines:
            return None

        # Name: first line, strip leading bullets/numbers
        first_line = re.sub(r"^\s*[\d.)\-•*]\s*", "", lines[0]).strip()
        if first_line and len(first_line) <= 80:
            project.name = first_line

        # URL: prefer repo URLs, fall back to general URL
        repo_match = self.REPO_URL_PATTERN.search(block)
        if repo_match:
            project.url = repo_match.group(0)
        else:
            url_match = self.GENERAL_URL_PATTERN.search(block)
            if url_match:
                project.url = url_match.group(0)

        # Technologies: check for explicit tech label first
        tech_match = self.TECH_LABEL_PATTERN.search(block)
        if tech_match:
            tech_text = tech_match.group(1)
            skills = self._skills_parser._extract_known_skills(tech_text)
            project.technologies = [s.name for s in skills]
        else:
            skills = self._skills_parser._extract_known_skills(block)
            project.technologies = [s.name for s in skills]

        # Description: first 1-2 content lines after name line
        description_lines = []
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
            if self._is_skip_line(stripped):
                continue
            description_lines.append(stripped)
            if len(description_lines) >= 2:
                break

        if description_lines:
            desc = " ".join(description_lines)
            project.description = desc[:300] if len(desc) > 300 else desc

        project.confidence = self._calculate_confidence(project)
        return project

    def _is_skip_line(self, line: str) -> bool:
        """Return True if this line should be excluded from description."""
        for pattern in self.SKIP_LINE_PATTERNS:
            if pattern.search(line):
                return True
        return False

    def _calculate_confidence(self, project: ExtractedProject) -> float:
        """Calculate confidence score for a project."""
        score = 0.0
        if project.name:
            score += 0.35
        if project.description and len(project.description) > 20:
            score += 0.30
        if project.technologies:
            score += 0.25
        if project.url:
            score += 0.10
        return round(score, 2)
