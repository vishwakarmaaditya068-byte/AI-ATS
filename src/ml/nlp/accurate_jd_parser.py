"""
Accurate job description parser.

Extracts: title, company, description, responsibilities, required skills,
preferred skills, experience requirements, education requirements,
employment type, work location, benefits, certifications.

Standalone — no project infrastructure dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ParsedJob:
    """Structured output of AccurateJDParser — mirrors ParsedResume on the job side."""

    title: str = ""
    company_name: str = ""
    company_description: str = ""
    description: str = ""
    responsibilities: list[str] = field(default_factory=list)
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    experience_min_years: float = 0.0
    experience_max_years: Optional[float] = None
    experience_level: str = ""        # "entry" | "mid" | "senior" | "lead" | "executive"
    education_requirement: str = ""   # "bachelor" | "master" | "phd" | "diploma"
    employment_type: str = ""         # "full_time" | "part_time" | "contract" | "internship"
    work_location: str = ""           # "onsite" | "remote" | "hybrid"
    benefits: list[str] = field(default_factory=list)
    certifications_required: list[str] = field(default_factory=list)
    languages_required: list[str] = field(default_factory=list)
    raw_text: str = ""

    def to_job_create(self) -> "JobCreate":
        """
        Convert ParsedJob to a JobCreate schema for database storage or matching.

        Uses lazy import so this module remains standalone at the module level.
        Maps employment_type/work_location/experience_level strings to their
        respective Enum values, defaulting safely when a value is unrecognised.
        """
        from src.data.models.job import (
            JobCreate,
            SkillRequirement,
            EducationRequirement,
            ExperienceRequirement,
            EmploymentType,
            WorkLocation,
            ExperienceLevel,
        )

        # Resolve enums — use get() with safe fallback
        _emp_values: set[str] = {e.value for e in EmploymentType}
        emp_type: EmploymentType = EmploymentType(
            self.employment_type if self.employment_type in _emp_values else "full_time"
        )

        _loc_values: set[str] = {e.value for e in WorkLocation}
        work_loc: WorkLocation = WorkLocation(
            self.work_location if self.work_location in _loc_values else "onsite"
        )

        _lvl_values: set[str] = {e.value for e in ExperienceLevel}
        exp_level: ExperienceLevel = ExperienceLevel(
            self.experience_level if self.experience_level in _lvl_values else "mid"
        )

        skill_reqs: list[SkillRequirement] = [
            SkillRequirement(name=s, is_required=True) for s in self.required_skills
        ] + [
            SkillRequirement(name=s, is_required=False) for s in self.preferred_skills
        ]

        edu_req: Optional[EducationRequirement] = (
            EducationRequirement(minimum_degree=self.education_requirement)
            if self.education_requirement
            else None
        )

        exp_req: Optional[ExperienceRequirement] = (
            ExperienceRequirement(
                minimum_years=self.experience_min_years,
                maximum_years=self.experience_max_years,
            )
            if self.experience_min_years > 0
            else None
        )

        # description must be >= 10 chars per Job model validator
        description: str = (
            self.description
            or self.raw_text[:500]
            or "No description provided."
        )
        if len(description) < 10:
            description = "No description provided."

        return JobCreate(
            title=self.title or "Untitled Position",
            description=description,
            responsibilities=self.responsibilities,
            company_name=self.company_name or "Unknown Company",
            company_description=self.company_description or None,
            benefits=self.benefits,
            employment_type=emp_type,
            work_location=work_loc,
            experience_level=exp_level,
            skill_requirements=skill_reqs,
            education_requirement=edu_req,
            experience_requirement=exp_req,
            certifications_required=self.certifications_required,
            languages_required=self.languages_required,
        )


# ---------------------------------------------------------------------------
# Compiled patterns — module-level (mirrors accurate_resume_parser.py style)
# ---------------------------------------------------------------------------

_EXP_RANGE_RE = re.compile(
    r"(\d+)\s*(?:to|-|–)\s*(\d+)\s*\+?\s*years?",
    re.IGNORECASE,
)
_EXP_MIN_RE = re.compile(
    r"(\d+)\s*\+?\s*years?",
    re.IGNORECASE,
)
_EDU_RANK: list[tuple[str, re.Pattern[str]]] = [
    ("phd", re.compile(r"\b(ph\.?d|doctorate)\b", re.IGNORECASE)),
    ("master", re.compile(
        r"\b(master|m\.?tech|m\.?e\.?|m\.?sc\.?|m\.?s\.?|mba)\b", re.IGNORECASE
    )),
    ("bachelor", re.compile(
        r"\b(bachelor|b\.?tech|b\.?e\.?|b\.?sc\.?|b\.?s\.?|undergraduate|degree)\b",
        re.IGNORECASE,
    )),
    ("diploma", re.compile(r"\bdiploma\b", re.IGNORECASE)),
]
_REMOTE_RE = re.compile(r"\b(remote|work\s+from\s+home|wfh)\b", re.IGNORECASE)
_HYBRID_RE = re.compile(r"\bhybrid\b", re.IGNORECASE)
_FULLTIME_RE = re.compile(r"\b(full[-\s]?time|permanent)\b", re.IGNORECASE)
_PARTTIME_RE = re.compile(r"\bpart[-\s]?time\b", re.IGNORECASE)
_CONTRACT_RE = re.compile(r"\b(contract(?:or)?|consulting)\b", re.IGNORECASE)
_INTERNSHIP_RE = re.compile(r"\b(intern|internship|trainee|co-?op)\b", re.IGNORECASE)
_CERT_RE = re.compile(
    r"\b(?:"
    r"(?:AWS|GCP|Azure|Google)(?:\s+\w+){0,3}?\s+(?:Certified|Certification|Certificate)"
    r"|CISSP|PMP|ITIL|CPA|CompTIA(?:\s+\w+)?"
    r"|Scrum\s+(?:Master|Developer|Product\s+Owner)"
    r")\b",
    re.IGNORECASE,
)
_LANG_RE = re.compile(
    r"\b(English|French|Spanish|German|Japanese|Mandarin|Hindi|Arabic|Portuguese)\b",
    re.IGNORECASE,
)
_ANY_BULLET_RE = re.compile(r"^[•►▸▶–—\-*✓✗]\s*")
_GENERIC_TOKENS: frozenset[str] = frozenset({
    "experience", "knowledge", "understanding", "skills", "ability", "strong",
    "working", "with", "of", "and", "the", "or", "in", "to", "a", "an",
    "excellent", "good", "proven", "demonstrated", "solid", "hands-on",
    "proficiency", "proficient", "exposure", "background", "required", "preferred",
    # extra tokens from original local set
    "years", "year", "using", "minimum", "plus", "degree", "bachelor",
    "master", "phd", "field", "related", "computer", "science",
    "mathematics", "statistics", "equivalent", "development", "engineering",
})


# ---------------------------------------------------------------------------
# Section header map — mirrors _SECTION_MAP in accurate_resume_parser.py
# ---------------------------------------------------------------------------

_JD_SECTION_ALIASES: dict[str, list[str]] = {
    "overview": [
        "overview", "about the role", "about this role", "job summary",
        "role summary", "position summary", "about the position",
        "the role", "role overview", "job overview", "description",
        "job description", "about the job", "what you'll do overview",
    ],
    "responsibilities": [
        "responsibilities", "what you'll do", "what you will do",
        "key responsibilities", "role responsibilities", "your responsibilities",
        "duties", "job duties", "day to day", "day-to-day",
        "what we need you to do", "what you'll be doing",
        "what you will be doing", "your role",
    ],
    "requirements": [
        "requirements", "qualifications", "what you'll need",
        "what you will need", "what we're looking for",
        "what we are looking for", "must have", "required skills",
        "minimum qualifications", "basic qualifications",
        "about you", "you should have", "you must have",
        "we are looking for", "candidate profile",
        "skills and experience", "skills & experience",
        "who you are", "your background",
    ],
    "preferred": [
        "preferred", "preferred qualifications", "preferred skills",
        "nice to have", "nice-to-have", "bonus", "bonus points",
        "desired qualifications", "desired skills", "plus if you have",
        "good to have", "preferred experience", "would be nice",
        "additional qualifications",
    ],
    "benefits": [
        "benefits", "what we offer", "perks", "compensation",
        "why join us", "what you get", "our offer", "we offer",
        "rewards", "total rewards", "why us",
    ],
    "about": [
        "about us", "about the company", "company overview",
        "who we are", "our story", "the company",
        "company description", "about acme", "about our company",
    ],
    "certifications": [
        "certifications", "certificates", "certification requirements",
    ],
}

_JD_SECTION_MAP: dict[str, str] = {
    alias: section
    for section, aliases in _JD_SECTION_ALIASES.items()
    for alias in aliases
}


# ---------------------------------------------------------------------------
# Helpers — mirrors helpers in accurate_resume_parser.py
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Collapse whitespace, lowercase, strip trailing punctuation."""
    cleaned: str = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[:\-_•|]+$", "", cleaned).strip()


def _strip_bullet(line: str) -> str:
    """Remove any leading bullet character from a line."""
    return _ANY_BULLET_RE.sub("", line.strip()).strip()


def _collapse_spaces(line: str) -> str:
    """Collapse multiple spaces to one (pdfplumber layout=True artefact)."""
    return re.sub(r" {2,}", " ", line)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class AccurateJDParser:
    """
    Accurately parse job descriptions into structured ParsedJob data.

    Mirrors AccurateResumeParser — standalone, no infrastructure dependencies.
    Uses compiled regex patterns and a section-alias map identical in structure
    to those in accurate_resume_parser.py.

    Entry points:
        parse(text: str) -> ParsedJob          — from raw text
        parse_file(path: str | Path) -> ParsedJob  — from PDF/TXT file
    """

    def parse(self, text: str) -> ParsedJob:
        """Parse job description text into a ParsedJob dataclass."""
        result: ParsedJob = ParsedJob(raw_text=text)
        lines: list[str] = [_collapse_spaces(ln) for ln in text.splitlines()]

        result.title, result.company_name = self._extract_header(lines)
        sections: dict[str, list[str]] = self._split_sections(lines)

        result.description = self._parse_overview(sections.get("overview", []))
        if not result.description:
            result.description = self._build_fallback_description(lines)

        result.responsibilities = self._parse_bullets(
            sections.get("responsibilities", [])
        )

        req_text: str = "\n".join(sections.get("requirements", []))
        pref_text: str = "\n".join(sections.get("preferred", []))
        result.required_skills = self._extract_skills(req_text)
        result.preferred_skills = self._extract_skills(pref_text)

        # Scan requirements + top of document for experience/education
        scan_text: str = req_text + "\n" + text[:400]
        result.experience_min_years, result.experience_max_years = (
            self._extract_experience(scan_text)
        )
        result.education_requirement = self._extract_education(scan_text)

        result.employment_type = self._detect_employment_type(text)
        result.work_location = self._detect_work_location(text)
        result.experience_level = self._detect_experience_level(
            text, result.experience_min_years
        )

        result.benefits = self._parse_bullets(sections.get("benefits", []))
        result.company_description = " ".join(
            ln.strip() for ln in sections.get("about", []) if ln.strip()
        )
        result.certifications_required = self._extract_certifications(req_text)
        result.languages_required = self._extract_languages(text)

        return result

    def parse_file(self, path: str | Path) -> ParsedJob:
        """Parse a job description from a PDF or plain-text file."""
        try:
            import pdfplumber
            pages: list[str] = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    t: Optional[str] = page.extract_text(layout=True)
                    if t:
                        pages.append(t)
            text: str = "\n".join(pages)
        except Exception:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        return self.parse(text)

    # ------------------------------------------------------------------
    # Section splitting (mirrors _split_sections in AccurateResumeParser)
    # ------------------------------------------------------------------

    def _split_sections(self, lines: list[str]) -> dict[str, list[str]]:
        """Return {section_key: [lines]} mapping."""
        sections: dict[str, list[str]] = {}
        current_key: Optional[str] = None
        current_lines: list[str] = []

        for line in lines:
            norm: str = _normalise(line)
            norm_stripped: str = _normalise(re.sub(r"^[\d.•\-–]+\s*", "", line))
            matched: Optional[str] = (
                _JD_SECTION_MAP.get(norm) or _JD_SECTION_MAP.get(norm_stripped)
            )

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
    # Header extraction
    # ------------------------------------------------------------------

    def _extract_header(self, lines: list[str]) -> tuple[str, str]:
        """Extract title and company from the first non-empty lines."""
        title: str = ""
        company: str = ""

        for line in lines[:12]:
            stripped: str = _collapse_spaces(line).strip()
            if not stripped or len(stripped) < 3:
                continue
            # Stop at first recognised section header
            if _JD_SECTION_MAP.get(_normalise(stripped)):
                break
            # Skip lines that are purely numeric/separators
            if re.fullmatch(r"[\d\s@|,./:()\-]+", stripped):
                continue
            if not title:
                title = stripped
            elif not company:
                # Take only the part before a pipe/comma separator
                company = re.split(r"\s*[|,]\s*", stripped)[0].strip()
                break

        return title, company

    # ------------------------------------------------------------------
    # Section content parsers
    # ------------------------------------------------------------------

    def _parse_overview(self, lines: list[str]) -> str:
        return " ".join(ln.strip() for ln in lines if ln.strip())

    def _parse_bullets(self, lines: list[str]) -> list[str]:
        """
        Parse a bullet-point section into a list of strings.
        Handles wrapped continuations (lowercase continuation lines).
        """
        bullets: list[str] = []
        current: str = ""

        for line in lines:
            stripped: str = line.strip()
            if not stripped:
                if current:
                    bullets.append(current)
                    current = ""
                continue
            if _ANY_BULLET_RE.match(stripped):
                if current:
                    bullets.append(current)
                current = _strip_bullet(stripped)
            elif current and stripped[0].islower():
                current += " " + stripped  # wrapped continuation
            else:
                if current:
                    bullets.append(current)
                current = stripped

        if current:
            bullets.append(current)

        return [b for b in bullets if b]

    # Technology keywords to pull out of experience/education lines
    _TECH_TOKEN_RE: re.Pattern[str] = re.compile(
        r"\b([A-Z][a-zA-Z0-9+#.]*(?:\s+[A-Z][a-zA-Z0-9+#.]*){0,2}|"
        r"[a-z][a-zA-Z0-9+#.]{1,})\b",
    )

    def _extract_skills(self, text: str) -> list[str]:
        """
        Extract skill tokens from requirement/preferred section text.
        For experience-quantity lines, still pulls out capitalised technology
        tokens (e.g. "Python" from "3+ years of Python experience") rather
        than skipping the whole line.
        Deduplicates while preserving insertion order. Caps at 30 items.
        """
        skills: list[str] = []
        seen: set[str] = set()

        def _add(tok: str) -> bool:
            """Add tok to skills; return True if cap reached."""
            normalised: str = re.sub(r"\s+", " ", tok).strip().lower()
            if (
                1 < len(normalised) <= 60
                and normalised not in seen
                and not re.fullmatch(r"[\d\s]+", normalised)
            ):
                seen.add(normalised)
                skills.append(normalised)
                return len(skills) >= 30
            return False

        for line in text.splitlines():
            stripped: str = _strip_bullet(line.strip())
            if not stripped:
                continue

            is_exp_line: bool = bool(
                _EXP_MIN_RE.search(stripped)
                and re.search(r"\bexperience\b|\bworking\b", stripped, re.IGNORECASE)
            )
            is_edu_line: bool = bool(
                re.search(
                    r"\b(degree|bachelor|master|phd|diploma)\b.*\bin\b",
                    stripped,
                    re.IGNORECASE,
                )
            )

            if is_exp_line or is_edu_line:
                # Harvest capitalised technology tokens only
                for m in self._TECH_TOKEN_RE.finditer(stripped):
                    word: str = m.group()
                    if word.lower() not in _GENERIC_TOKENS and len(word) > 2:
                        if _add(word):
                            return skills
                continue

            tokens: list[str] = [
                t.strip() for t in re.split(r"[,;/]", stripped) if t.strip()
            ]
            for tok in tokens:
                if _add(tok):
                    return skills

        return skills

    def _extract_experience(self, text: str) -> tuple[float, Optional[float]]:
        """Return (min_years, max_years) from the first experience pattern found."""
        m_range = _EXP_RANGE_RE.search(text)
        if m_range:
            return float(m_range.group(1)), float(m_range.group(2))
        for m_min in _EXP_MIN_RE.finditer(text):
            start: int = max(0, m_min.start() - 60)
            end: int = min(len(text), m_min.end() + 60)
            context: str = text[start:end].lower()
            if re.search(r"\b(experience|expertise|working|proficiency|required|minimum)\b", context):
                return float(m_min.group(1)), None
        return 0.0, None

    def _extract_education(self, text: str) -> str:
        """Return the highest education level found (phd > master > bachelor > diploma)."""
        for level, pattern in _EDU_RANK:
            if pattern.search(text):
                return level
        return ""

    def _detect_employment_type(self, text: str) -> str:
        """Detect employment type by scanning for keywords in order of specificity."""
        if _INTERNSHIP_RE.search(text):
            return "internship"
        if _PARTTIME_RE.search(text):
            return "part_time"
        if _CONTRACT_RE.search(text):
            return "contract"
        if _FULLTIME_RE.search(text):
            return "full_time"
        return "full_time"  # safe default

    def _detect_work_location(self, text: str) -> str:
        """Detect work location: hybrid wins over remote, both win over onsite."""
        if _HYBRID_RE.search(text):
            return "hybrid"
        if _REMOTE_RE.search(text):
            return "remote"
        return "onsite"

    def _detect_experience_level(self, text: str, min_years: float) -> str:
        """Detect experience level from keyword scan; falls back to min_years bucket."""
        t: str = text.lower()
        if re.search(r"\b(executive|director|vp|c-level|cto|cio)\b", t):
            return "executive"
        if re.search(r"\b(lead|principal|staff\s+engineer)\b", t):
            return "lead"
        if re.search(r"\b(senior|sr\.?)\b", t):
            return "senior"
        if re.search(r"\b(junior|entry[\s-]level|fresher|graduate|new\s+grad)\b", t):
            return "entry"
        # Fallback: bucket by minimum years
        if min_years >= 7:
            return "senior"
        if min_years >= 3:
            return "mid"
        if 0 < min_years <= 1:
            return "entry"
        return "mid"

    def _extract_certifications(self, text: str) -> list[str]:
        certs: list[str] = []
        for m in _CERT_RE.finditer(text):
            cert: str = m.group().strip()
            if cert not in certs:
                certs.append(cert)
        return certs

    def _extract_languages(self, text: str) -> list[str]:
        langs: list[str] = []
        for m in _LANG_RE.finditer(text):
            lang: str = m.group().capitalize()
            if lang not in langs:
                langs.append(lang)
        return langs

    def _build_fallback_description(self, lines: list[str]) -> str:
        """First 3 content lines (non-header, non-empty) as description fallback."""
        parts: list[str] = []
        for line in lines:
            stripped: str = line.strip()
            if not stripped:
                continue
            if _JD_SECTION_MAP.get(_normalise(stripped)):
                continue
            parts.append(stripped)
            if len(parts) >= 3:
                break
        return " ".join(parts)


def get_accurate_jd_parser() -> AccurateJDParser:
    """Get an AccurateJDParser instance."""
    return AccurateJDParser()
