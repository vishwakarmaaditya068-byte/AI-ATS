"""
Accurate resume parser that preserves full structure.

Extracts: name, contact info, categorised skills, experience with bullets,
education with scores, projects with sub-bullets, achievements.

Standalone — no project infrastructure dependencies.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ContactInfo:
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""


@dataclass
class SkillCategory:
    category: str = ""
    skills: list[str] = field(default_factory=list)


@dataclass
class ExperienceEntry:
    title: str = ""
    company: str = ""
    duration: str = ""
    bullets: list[str] = field(default_factory=list)


@dataclass
class EducationEntry:
    degree: str = ""
    institution: str = ""
    year: str = ""
    score: str = ""


@dataclass
class ProjectEntry:
    name: str = ""
    technologies: str = ""
    duration: str = ""
    url: str = ""
    bullets: list[str] = field(default_factory=list)


@dataclass
class ParsedResume:
    contact: ContactInfo = field(default_factory=ContactInfo)
    summary: str = ""
    skills: list[SkillCategory] = field(default_factory=list)
    experience: list[ExperienceEntry] = field(default_factory=list)
    education: list[EducationEntry] = field(default_factory=list)
    projects: list[ProjectEntry] = field(default_factory=list)
    achievements: list[str] = field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+91[-\s]?\d{10}|\+91\d{10}|91[-\s]?\d{10}|\b\d{10}\b)")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w%-]+", re.IGNORECASE)
_GITHUB_RE = re.compile(r"github\.com/[\w%-]+", re.IGNORECASE)
_PORTFOLIO_RE = re.compile(r"https?://(?!.*linkedin)(?!.*github)[\w./-]+\.[a-z]{2,}", re.IGNORECASE)
_DATE_RE = re.compile(
    r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{4}",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_SCORE_RE = re.compile(r"(?:CGPA|GPA|Percentage|Grade)\s*:?\s*([\d.]+\s*%?)", re.IGNORECASE)

_CITY_RE = re.compile(
    r"(Greater Noida|Noida|New Delhi|Delhi|Mumbai|Bengaluru|Bangalore|"
    r"Hyderabad|Chennai|Pune|Kolkata|Ahmedabad|Jaipur|Lucknow|Chandigarh|"
    r"Gurgaon|Gurugram|Faridabad|Ghaziabad|Agra|Meerut|Varanasi|Patna|"
    r"Bhopal|Indore|Nagpur|Surat|Coimbatore|Kochi|Thiruvananthapuram)",
    re.IGNORECASE,
)

# Bullet helpers
# • ► ▸ ▶  → header-level bullets (new entry/project)
# – — - *  → sub-level bullets (responsibilities/details)
_HEADER_BULLET_RE = re.compile(r"^[•►▸▶]\s*")
_SUB_BULLET_RE = re.compile(r"^[–—\-*]\s*")
_ANY_BULLET_RE = re.compile(r"^[•►▸▶–—\-*]\s*")

_DEGREE_KW_RE = re.compile(
    r"\b(B\.?[Tt]ech|B\.?E\.?|B\.?Sc\.?|M\.?Tech|M\.?E\.?|MBA|Ph\.?D|"
    r"Bachelor|Master|Class\s*\d+|High\s*School|Secondary|"
    r"Matriculation|Intermediate|Diploma|BRB\s+Model|Model\s+School)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Section header map
# ---------------------------------------------------------------------------

_SECTION_ALIASES: dict[str, list[str]] = {
    "summary": [
        "summary", "objective", "profile", "about me", "about",
        "professional summary", "career objective", "professional profile",
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "competencies",
        "skill set", "technologies", "technical expertise",
    ],
    "experience": [
        "experience", "work experience", "employment history",
        "professional experience", "internship", "internships",
        "work history", "career",
    ],
    "education": [
        "education", "academic details", "academic background",
        "educational qualification", "qualifications", "academics",
        "academic detail",
    ],
    "projects": [
        "projects", "project", "personal projects", "academic projects",
        "key projects", "notable projects",
    ],
    "achievements": [
        "achievements", "awards", "accomplishments", "honors",
        "honours", "extracurricular", "activities",
    ],
    "certifications": [
        "certifications", "certificates", "courses", "training",
        "professional development",
    ],
}

_SECTION_MAP: dict[str, str] = {}
for _sec, _aliases in _SECTION_ALIASES.items():
    for _alias in _aliases:
        _SECTION_MAP[_alias] = _sec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Collapse whitespace, lowercase, strip trailing punctuation."""
    cleaned = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[:\-_•]+$", "", cleaned).strip()


def _is_header_bullet(line: str) -> bool:
    return bool(_HEADER_BULLET_RE.match(line.strip()))


def _is_sub_bullet(line: str) -> bool:
    return bool(_SUB_BULLET_RE.match(line.strip()))


def _strip_header_bullet(line: str) -> str:
    return _HEADER_BULLET_RE.sub("", line.strip()).strip()


def _strip_sub_bullet(line: str) -> str:
    return _SUB_BULLET_RE.sub("", line.strip()).strip()


def _strip_any_bullet(line: str) -> str:
    return _ANY_BULLET_RE.sub("", line.strip()).strip()


def _collapse_spaces(line: str) -> str:
    """Collapse multiple spaces to one (pdfplumber layout=True artefact)."""
    return re.sub(r" {2,}", " ", line)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class AccurateResumeParser:
    """
    Accurately parse PDF resumes into structured data.

    Preserves:
    - Contact: name, email, phone, location, social links
    - Skills: per-category (Programming Languages, Web Tech, etc.)
    - Experience: title, company, duration + bullet points
    - Education: degree, institution, year, score
    - Projects: name, technologies, date, URL + bullet points
    - Achievements: list of strings
    """

    def parse(self, pdf_path: str | Path) -> ParsedResume:
        """Parse a PDF resume file into a ParsedResume dataclass."""
        text = self._extract_text(pdf_path)
        return self._parse_text(text)

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_text(self, pdf_path: str | Path) -> str:
        """Extract text from PDF using pdfplumber with layout preservation."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("pdfplumber is required: pip install pdfplumber")

        pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True)
                if text:
                    pages.append(text)
        return "\n".join(pages)

    # ------------------------------------------------------------------
    # Section splitting
    # ------------------------------------------------------------------

    def _split_sections(self, lines: list[str]) -> dict[str, list[str]]:
        """Return {section_key: [lines]} mapping."""
        sections: dict[str, list[str]] = {}
        current_key: Optional[str] = None
        current_lines: list[str] = []

        for line in lines:
            # Normalise for matching but keep original for content
            norm = _normalise(line)
            # Also try after stripping leading bullet/number
            norm_stripped = _normalise(re.sub(r"^[\d.•\-–]+\s*", "", line))

            matched = _SECTION_MAP.get(norm) or _SECTION_MAP.get(norm_stripped)

            if matched:
                if current_key is not None:
                    sections[current_key] = current_lines
                current_key = matched
                current_lines = []
            elif current_key is not None:
                current_lines.append(line)

        if current_key is not None and current_lines:
            sections[current_key] = current_lines

        return sections

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def _parse_text(self, text: str) -> ParsedResume:
        result = ParsedResume(raw_text=text)
        # Collapse multi-space runs (pdfplumber layout=True artefact)
        lines = [_collapse_spaces(l) for l in text.splitlines()]

        result.contact = self._extract_contact(lines)
        sections = self._split_sections(lines)

        result.summary = self._parse_summary(sections.get("summary", []))
        result.skills = self._parse_skills(sections.get("skills", []))
        result.experience = self._parse_experience(sections.get("experience", []))
        result.education = self._parse_education(sections.get("education", []))
        result.projects = self._parse_projects(sections.get("projects", []))
        result.achievements = self._parse_list_section(sections.get("achievements", []))

        return result

    # ------------------------------------------------------------------
    # Contact extraction
    # ------------------------------------------------------------------

    def _extract_contact(self, lines: list[str]) -> ContactInfo:
        contact = ContactInfo()

        # Name: first meaningful line in the header block
        for line in lines[:8]:
            stripped = _collapse_spaces(line).strip()
            if not stripped or len(stripped) < 3 or len(stripped) > 80:
                continue
            if _EMAIL_RE.search(stripped):
                continue
            if _PHONE_RE.search(stripped):
                continue
            if re.search(r"https?://|linkedin|github|portfolio|@|⋄|◇", stripped, re.I):
                continue
            if _SECTION_MAP.get(_normalise(stripped)):
                continue
            contact.name = stripped
            break

        # Scan first 25 lines for contact fields
        header_block = "\n".join(lines[:25])

        m = _EMAIL_RE.search(header_block)
        if m:
            contact.email = m.group()

        m = _PHONE_RE.search(header_block)
        if m:
            contact.phone = m.group().strip()

        m = _LINKEDIN_RE.search(header_block)
        if m:
            contact.linkedin = "https://www." + m.group()

        m = _GITHUB_RE.search(header_block)
        if m:
            contact.github = "https://" + m.group()

        for m in _PORTFOLIO_RE.finditer(header_block):
            url = m.group()
            if "linkedin" not in url.lower() and "github" not in url.lower():
                contact.portfolio = url
                break

        m = _CITY_RE.search(header_block)
        if m:
            contact.location = m.group()

        return contact

    # ------------------------------------------------------------------
    # Section parsers
    # ------------------------------------------------------------------

    def _parse_summary(self, lines: list[str]) -> str:
        return " ".join(l.strip() for l in lines if l.strip())

    def _parse_skills(self, lines: list[str]) -> list[SkillCategory]:
        """
        Parse skills. Handles:
        1. "• Category: skill1, skill2"  (bullet-prefixed category lines)
        2. "Category: skill1, skill2"    (plain category lines)
        3. Raw comma-separated list
        """
        categories: list[SkillCategory] = []
        uncategorised: list[str] = []

        for line in lines:
            stripped = _strip_any_bullet(line.strip())
            if not stripped:
                continue

            m = re.match(r"^([A-Za-z][A-Za-z\s/&()+#.]{2,50}):\s*(.+)$", stripped)
            if m:
                cat_label = m.group(1).strip()
                skills_raw = m.group(2).strip()
                skills = [s.strip() for s in re.split(r"[,;]", skills_raw) if s.strip()]
                if skills:
                    categories.append(SkillCategory(category=cat_label, skills=skills))
                continue

            tokens = [s.strip() for s in re.split(r"[,;]", stripped) if s.strip()]
            uncategorised.extend(tokens)

        if uncategorised:
            categories.append(SkillCategory(category="General", skills=uncategorised))

        return categories

    def _parse_experience(self, lines: list[str]) -> list[ExperienceEntry]:
        """
        Parse experience section.

        Layout:
          • AIML INTERNSHIP (VIRTUAL)     October 2024 - December 2024
            – Engineered …
              through advanced deep learning.   ← wrapped continuation

        - • bullet → new entry header (title + possibly duration)
        - – bullet → sub-point
        - indented plain text → continuation of previous sub-point
        """
        entries: list[ExperienceEntry] = []
        current: Optional[ExperienceEntry] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Sub-point bullet (– or -)
            if _is_sub_bullet(stripped):
                if current is not None:
                    current.bullets.append(_strip_sub_bullet(stripped))
                continue

            # Wrapped continuation of a sub-point:
            # - has any leading whitespace (indented), OR
            # - starts with lowercase (sentence fragment carrying over)
            if (
                current is not None
                and current.bullets
                and stripped
                and stripped[0].islower()
                and not _is_header_bullet(stripped)
                and not _is_sub_bullet(stripped)
            ):
                current.bullets[-1] += " " + stripped
                continue

            # Header bullet (•) → new entry
            if _is_header_bullet(stripped):
                content = _strip_header_bullet(stripped)
                title, duration = content, ""
                date_m = _DATE_RE.search(content)
                if date_m:
                    duration = content[date_m.start():].strip()
                    title = content[: date_m.start()].strip().rstrip("-–, ").strip()
                current = ExperienceEntry(title=title, duration=duration)
                entries.append(current)
                continue

            # Duration line (month+year or "present")
            if _DATE_RE.search(stripped) or re.search(r"\bpresent\b|\bcurrent\b", stripped, re.I):
                if current is not None and not current.duration:
                    current.duration = stripped
                continue

            # Plain uppercase → title fallback
            alpha = re.sub(r"[^A-Za-z\s]", "", stripped)
            upper_ratio = sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)
            if (
                len(stripped.split()) <= 10
                and upper_ratio >= 0.75
                and re.search(r"[A-Z]{2,}", stripped)
                and not _EMAIL_RE.search(stripped)
            ):
                current = ExperienceEntry(title=stripped)
                entries.append(current)
                continue

            # First untagged text after entry → company
            if current is not None and not current.company:
                current.company = stripped

        return entries

    def _parse_education(self, lines: list[str]) -> list[EducationEntry]:
        """
        Parse education section.

        Layout:
          B.tech CSE (AIML), Galgotias University     Expected 2026
          CGPA: 7.3
          BRB Model School                            2022
          Percentage: 85.4%
        """
        entries: list[EducationEntry] = []
        current: Optional[EducationEntry] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            score_m = _SCORE_RE.search(stripped)
            year_m = _YEAR_RE.search(stripped)

            # Score line → attach to current entry
            if score_m:
                if current is not None:
                    current.score = stripped
                continue

            # Line contains degree keywords → new education entry
            if _DEGREE_KW_RE.search(stripped):
                current = EducationEntry(degree=stripped)
                if year_m:
                    current.year = year_m.group()
                # Institution may be on the same line (e.g. "B.Tech CSE, Galgotias University  2026")
                inst_m = re.search(
                    r"([\w\s]+(?:University|College|Institute|School|Academy|Polytechnic))",
                    stripped,
                    re.IGNORECASE,
                )
                if inst_m:
                    current.institution = inst_m.group(1).strip()
                entries.append(current)
                continue

            # Line with year but no degree keyword → institution or standalone school
            if year_m:
                if current is not None and not current.institution:
                    current.institution = stripped
                    if not current.year:
                        current.year = year_m.group()
                else:
                    current = EducationEntry(institution=stripped, year=year_m.group())
                    entries.append(current)
                continue

            # Plain line after degree entry → institution
            if current is not None and not current.institution:
                current.institution = stripped

        return entries

    def _parse_projects(self, lines: list[str]) -> list[ProjectEntry]:
        """
        Parse projects section preserving bullet points per project.

        Layout:
          • Twitter Sentiment Analysis - Python, Scikit-learn…  Link  March 2025
            – Built a robust …
              data using TF-IDF.    ← wrapped continuation (indented, lowercase)

        - • bullet → project header (name, tech, date extracted from same line)
        - – bullet → sub-point under current project
        - indented lowercase → continuation of previous sub-point
        """
        entries: list[ProjectEntry] = []
        current: Optional[ProjectEntry] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Sub-point (– or -)
            if _is_sub_bullet(stripped):
                text = _strip_sub_bullet(stripped)
                if current is not None and text:
                    current.bullets.append(text)
                continue

            # Wrapped continuation of a sub-bullet (starts lowercase after space collapse)
            if (
                current is not None
                and current.bullets
                and stripped
                and stripped[0].islower()
            ):
                current.bullets[-1] += " " + stripped
                continue

            # Project header — may or may not have leading •
            proj_line = _strip_header_bullet(stripped) if _is_header_bullet(stripped) else stripped

            # Skip empty results
            if not proj_line:
                continue

            # Extract URL
            url = ""
            url_m = re.search(r"https?://\S+", proj_line)
            if url_m:
                url = url_m.group()
                proj_line = proj_line.replace(url, "").strip()

            # Remove plain "Link" anchor text
            proj_line = re.sub(r"\s+Link\b", "", proj_line, flags=re.IGNORECASE).strip()

            # Extract date from end of line
            duration = ""
            date_m = _DATE_RE.search(proj_line)
            if date_m:
                duration = date_m.group()
                proj_line = proj_line[: date_m.start()].strip().rstrip("-–, ").strip()

            # Extract technologies after " - " or " – "
            technologies = ""
            dash_m = re.search(r"\s[-–]\s+(.+)$", proj_line)
            if dash_m:
                technologies = dash_m.group(1).strip()
                proj_line = proj_line[: dash_m.start()].strip()

            if proj_line:
                current = ProjectEntry(
                    name=proj_line,
                    technologies=technologies,
                    duration=duration,
                    url=url,
                )
                entries.append(current)

        return entries

    def _parse_list_section(self, lines: list[str]) -> list[str]:
        """Parse a section that is a simple list (achievements, certifications)."""
        items: list[str] = []
        current_item: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if _is_header_bullet(stripped) or _is_sub_bullet(stripped):
                if current_item:
                    items.append(current_item)
                current_item = _strip_any_bullet(stripped)
            elif current_item is not None:
                current_item += " " + stripped
            else:
                current_item = stripped

        if current_item:
            items.append(current_item)

        return items

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self, result: ParsedResume) -> dict:
        """Convert ParsedResume to a plain dict."""
        return asdict(result)

    def to_json(self, result: ParsedResume, indent: int = 2) -> str:
        """Serialise ParsedResume to a JSON string."""
        return json.dumps(self.to_dict(result), indent=indent, ensure_ascii=False)
