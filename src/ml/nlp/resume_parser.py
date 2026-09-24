"""
Main resume parser orchestrator.

Coordinates text extraction, preprocessing, and information extraction
to parse resumes into structured data.
"""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.data.models import (
    Candidate,
    CandidateCreate,
    Certification,
    ContactInfo as CandidateContactInfo,
    Education,
    FileMetadata,
    Language,
    ParsedContent,
    ParsedSection,
    ProcessingMetrics,
    ProcessingStatus,
    Resume,
    ResumeFormat,
    Skill,
    WorkExperience,
)
from src.utils.config import get_settings
from src.utils.logger import get_logger

from .extractors import ExtractorFactory, ExtractionResult
from .parsers import (
    CertificationsParser,
    ContactParser,
    EducationParser,
    ExperienceParser,
    ProjectsParser,
    SkillsParser,
    SummaryParser,
)
from .preprocessor import PreprocessedText, TextPreprocessor

logger = get_logger(__name__)


@dataclass
class ResumeParseResult:
    """Complete result of resume parsing."""

    # File information
    file_path: Optional[str] = None
    file_hash: Optional[str] = None

    # Extraction results
    extraction_result: Optional[ExtractionResult] = None
    preprocessed: Optional[PreprocessedText] = None

    # Parsed information
    contact: Optional[dict[str, Any]] = None
    skills: list[dict[str, Any]] = field(default_factory=list)
    experience: list[dict[str, Any]] = field(default_factory=list)
    education: list[dict[str, Any]] = field(default_factory=list)
    certifications: list[dict[str, Any]] = field(default_factory=list)
    projects: list[dict[str, Any]] = field(default_factory=list)
    summary: Optional[str] = None
    languages: list[str] = field(default_factory=list)

    # Summary
    total_experience_years: float = 0.0
    highest_education: Optional[str] = None
    skill_count: int = 0

    # Quality metrics
    overall_confidence: float = 0.0
    parse_quality_score: float = 0.0

    # Processing info
    processing_time_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Check if parsing was successful.

        Requires no errors AND overall_confidence at or above the configured
        threshold (default 0.5).  The threshold is intentionally higher than
        the old value of 0.3 so that a document with only an email address
        does not pass — meaningful contact + at least one substantive section
        (skills, experience, or education) must be extracted.
        """
        threshold = get_settings().ml.resume_success_threshold
        return len(self.errors) == 0 and self.overall_confidence > threshold


class ResumeParser:
    """
    Main resume parser that orchestrates the parsing pipeline.

    Pipeline:
    1. Extract text from document (PDF, DOCX, etc.)
    2. Preprocess and detect sections
    3. Extract contact information
    4. Extract skills
    5. Extract work experience
    6. Extract education
    7. Compile results and calculate quality metrics
    """

    def __init__(self):
        """Initialize the resume parser with all component parsers."""
        try:
            import spacy
            spacy.prefer_gpu()
        except Exception:
            # spaCy is optional — used only as a GPU-acceleration hint.
            # On Python 3.14 it raises pydantic.v1.errors.ConfigError, not
            # ImportError, so we catch all exceptions to stay portable.
            pass
        self.preprocessor = TextPreprocessor()
        self.contact_parser = ContactParser()
        self.skills_parser = SkillsParser()
        self.experience_parser = ExperienceParser()
        self.education_parser = EducationParser()
        self.certifications_parser = CertificationsParser()
        self.projects_parser = ProjectsParser(skills_parser=self.skills_parser)
        self.summary_parser = SummaryParser()

    # Maximum file size to process (50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024

    def parse_file(self, file_path: str | Path) -> ResumeParseResult:
        """
        Parse a resume from a file.

        Args:
            file_path: Path to the resume file

        Returns:
            ResumeParseResult with all extracted information

        Security:
            - Validates file path to prevent traversal attacks
            - Limits file size to prevent resource exhaustion
        """
        start_time = time.time()
        result = ResumeParseResult(file_path=str(file_path))

        try:
            path = Path(file_path).resolve()

            # Security: Check file exists and is not too large
            if not path.exists():
                result.errors.append(f"File not found: {file_path}")
                return result

            if not path.is_file():
                result.errors.append(f"Not a file: {file_path}")
                return result

            file_size = path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                result.errors.append(f"File too large: {file_size} bytes (max: {self.MAX_FILE_SIZE})")
                return result

            # Calculate file hash
            with open(path, "rb") as f:
                result.file_hash = hashlib.sha256(f.read()).hexdigest()

            # Extract text
            extraction = ExtractorFactory.extract(path)
            result.extraction_result = extraction

            if not extraction.success:
                result.errors.append(f"Extraction failed: {extraction.error_message}")
                return result

            if extraction.is_empty:
                result.errors.append("Extracted text is empty")
                return result

            result.warnings.extend(extraction.warnings)

            # Process the extracted text
            result = self._process_text(extraction.text, result)

        except Exception as e:
            logger.exception(f"Error parsing resume: {e}")
            result.errors.append(str(e))

        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result

    def parse_bytes(
        self, content: bytes, filename: str
    ) -> ResumeParseResult:
        """
        Parse a resume from bytes.

        Args:
            content: Raw file bytes
            filename: Original filename (for format detection)

        Returns:
            ResumeParseResult with all extracted information

        Security:
            - Limits content size to prevent resource exhaustion
        """
        start_time = time.time()
        result = ResumeParseResult()

        # Security: Check content size
        if len(content) > self.MAX_FILE_SIZE:
            result.errors.append(f"Content too large: {len(content)} bytes (max: {self.MAX_FILE_SIZE})")
            return result

        result.file_hash = hashlib.sha256(content).hexdigest()

        try:
            # Extract text
            extraction = ExtractorFactory.extract_from_bytes(content, filename)
            result.extraction_result = extraction

            if not extraction.success:
                result.errors.append(f"Extraction failed: {extraction.error_message}")
                return result

            if extraction.is_empty:
                result.errors.append("Extracted text is empty")
                return result

            result.warnings.extend(extraction.warnings)

            # Process the extracted text
            result = self._process_text(extraction.text, result)

        except Exception as e:
            logger.exception(f"Error parsing resume from bytes: {e}")
            result.errors.append(str(e))

        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result

    def parse_text(self, text: str) -> ResumeParseResult:
        """
        Parse a resume from raw text.

        Args:
            text: Resume text content

        Returns:
            ResumeParseResult with all extracted information
        """
        start_time = time.time()
        result = ResumeParseResult()

        try:
            result = self._process_text(text, result)
        except Exception as e:
            logger.exception(f"Error parsing resume text: {e}")
            result.errors.append(str(e))

        result.processing_time_ms = int((time.time() - start_time) * 1000)
        return result

    def _process_text(
        self, text: str, result: ResumeParseResult
    ) -> ResumeParseResult:
        """Process extracted text through all parsers."""
        # Preprocess text
        preprocessed = self.preprocessor.preprocess(text)
        result.preprocessed = preprocessed
        result.warnings.extend(preprocessed.warnings)

        # Get section content helpers
        def get_section(section_type: str) -> Optional[str]:
            return self.preprocessor.get_section_content(preprocessed, section_type)

        # Extract contact information
        contact_result = self.contact_parser.parse(text)
        result.contact = {
            "first_name": contact_result.first_name,
            "last_name": contact_result.last_name,
            "full_name": contact_result.full_name,
            "email": contact_result.email,
            "phone": contact_result.phone,
            "linkedin_url": contact_result.linkedin_url,
            "github_url": contact_result.github_url,
            "portfolio_url": contact_result.portfolio_url,
            "city": contact_result.city,
            "state": contact_result.state,
            "country": contact_result.country,
            "confidence": contact_result.confidence,
        }

        # Extract skills
        skills_section = get_section("skills")
        skills_result = self.skills_parser.parse(text, skills_section)
        result.skills = [
            {
                "name": s.name,
                "category": s.category,
                "proficiency": s.proficiency,
                "confidence": s.confidence,
                "source": s.source,
            }
            for s in skills_result.skills
        ]
        result.skill_count = len(result.skills)

        # Extract work experience
        experience_section = get_section("experience")
        experience_result = self.experience_parser.parse(text, experience_section)
        result.experience = [
            {
                "job_title": e.job_title,
                "company": e.company,
                "location": e.location,
                "start_date": e.start_date.isoformat() if e.start_date else None,
                "end_date": e.end_date.isoformat() if e.end_date else None,
                "is_current": e.is_current,
                "responsibilities": e.responsibilities,
                "achievements": e.achievements,
                "confidence": e.confidence,
            }
            for e in experience_result.experiences
        ]
        result.total_experience_years = experience_result.total_years

        # Extract education
        education_section = get_section("education")
        education_result = self.education_parser.parse(text, education_section)
        result.education = [
            {
                "degree": e.degree,
                "degree_level": e.degree_level,
                "field_of_study": e.field_of_study,
                "institution": e.institution,
                "location": e.location,
                "graduation_date": e.graduation_date.isoformat() if e.graduation_date else None,
                "gpa": e.gpa,
                "honors": e.honors,
                "confidence": e.confidence,
            }
            for e in education_result.education
        ]
        result.highest_education = education_result.highest_level

        # Extract certifications
        certifications_section = get_section("certifications")
        if certifications_section:
            cert_result = self.certifications_parser.parse(certifications_section)
            result.certifications = [
                {
                    "name": c.name,
                    "issuer": c.issuer,
                    "issue_date": c.issue_date.isoformat() if c.issue_date else None,
                    "expiry_date": c.expiry_date.isoformat() if c.expiry_date else None,
                    "credential_id": c.credential_id,
                    "credential_url": c.credential_url,
                    "confidence": c.confidence,
                }
                for c in cert_result.certifications
            ]

        # Extract projects
        projects_section = get_section("projects")
        if projects_section:
            proj_result = self.projects_parser.parse(projects_section)
            result.projects = [
                {
                    "name": p.name,
                    "description": p.description,
                    "technologies": p.technologies,
                    "url": p.url,
                    "confidence": p.confidence,
                }
                for p in proj_result.projects
            ]

        # Extract professional summary
        summary_section = get_section("summary")
        if summary_section:
            summary_result = self.summary_parser.parse(summary_section)
            result.summary = summary_result.text or None

        # Extract languages (simple split — no full parser needed for plain lists)
        import re as _re
        languages_section = get_section("languages")
        if languages_section:
            result.languages = [
                lang.strip()
                for lang in _re.split(r"[,;\n•\-]", languages_section)
                if lang.strip() and 2 <= len(lang.strip()) <= 40
            ]

        # Calculate overall confidence and quality
        result.overall_confidence = self._calculate_overall_confidence(result)
        result.parse_quality_score = self._calculate_quality_score(result)

        return result

    def _calculate_overall_confidence(self, result: ResumeParseResult) -> float:
        """Calculate overall parsing confidence."""
        weights = {
            "contact": 0.23,
            "skills": 0.23,
            "experience": 0.28,
            "education": 0.18,
            "certifications": 0.04,
            "projects": 0.04,
        }

        score = 0.0

        # Contact confidence
        if result.contact:
            score += weights["contact"] * result.contact.get("confidence", 0)

        # Skills confidence
        if result.skills:
            avg_skill_conf = sum(s["confidence"] for s in result.skills) / len(result.skills)
            score += weights["skills"] * avg_skill_conf

        # Experience confidence
        if result.experience:
            avg_exp_conf = sum(e["confidence"] for e in result.experience) / len(result.experience)
            score += weights["experience"] * avg_exp_conf

        # Education confidence
        if result.education:
            avg_edu_conf = sum(e["confidence"] for e in result.education) / len(result.education)
            score += weights["education"] * avg_edu_conf

        # Certifications confidence
        if result.certifications:
            avg_cert_conf = sum(c["confidence"] for c in result.certifications) / len(result.certifications)
            score += weights["certifications"] * avg_cert_conf

        # Projects confidence
        if result.projects:
            avg_proj_conf = sum(p["confidence"] for p in result.projects) / len(result.projects)
            score += weights["projects"] * avg_proj_conf

        return round(score, 2)

    def _calculate_quality_score(self, result: ResumeParseResult) -> float:
        """Calculate overall parsing quality score."""
        score = 0.0

        # Contact info quality
        if result.contact:
            if result.contact.get("email"):
                score += 0.15
            if result.contact.get("first_name") and result.contact.get("last_name"):
                score += 0.10
            if result.contact.get("phone"):
                score += 0.05

        # Skills quality
        if len(result.skills) >= 5:
            score += 0.20
        elif len(result.skills) >= 2:
            score += 0.10

        # Experience quality
        if result.experience:
            score += 0.15
            if result.total_experience_years > 0:
                score += 0.10

        # Education quality
        if result.education:
            score += 0.15
            if result.highest_education:
                score += 0.10

        # Certifications quality bonus
        if result.certifications:
            score += 0.05

        # Projects quality bonus
        if result.projects:
            score += 0.05

        return round(min(score, 1.0), 2)

    def to_candidate_create(
        self, result: ResumeParseResult, source: Optional[str] = None
    ) -> Optional[CandidateCreate]:
        """
        Convert parse result to CandidateCreate schema.

        Args:
            result: ResumeParseResult from parsing
            source: Optional source identifier (e.g., "linkedin", "upload")

        Returns:
            CandidateCreate schema or None if insufficient data
        """
        if not result.success:
            return None

        contact = result.contact or {}

        # Require at least name and email
        if not contact.get("email"):
            logger.warning("Cannot create candidate: no email found")
            return None

        first_name = contact.get("first_name") or "Unknown"
        last_name = contact.get("last_name") or "Candidate"

        # Build contact info
        contact_info = CandidateContactInfo(
            email=contact["email"],
            phone=contact.get("phone"),
            linkedin_url=contact.get("linkedin_url"),
            github_url=contact.get("github_url"),
            portfolio_url=contact.get("portfolio_url"),
            city=contact.get("city"),
            state=contact.get("state"),
            country=contact.get("country"),
        )

        # Build skills
        skills = [
            Skill(
                name=s["name"],
                category=s.get("category"),
                proficiency_level=s.get("proficiency"),
            )
            for s in result.skills
        ]

        # Build work experience
        work_experience = []
        for exp in result.experience:
            from datetime import date as date_type

            start_date = None
            end_date = None

            if exp.get("start_date"):
                try:
                    start_date = date_type.fromisoformat(exp["start_date"])
                except ValueError:
                    pass

            if exp.get("end_date"):
                try:
                    end_date = date_type.fromisoformat(exp["end_date"])
                except ValueError:
                    pass

            work_experience.append(
                WorkExperience(
                    job_title=exp.get("job_title") or "Unknown Position",
                    company=exp.get("company") or "Unknown Company",
                    location=exp.get("location"),
                    start_date=start_date,
                    end_date=end_date,
                    is_current=exp.get("is_current", False),
                    responsibilities=exp.get("responsibilities", []),
                    achievements=exp.get("achievements", []),
                )
            )

        # Build education
        education = []
        for edu in result.education:
            from datetime import date as date_type

            graduation_date = None
            if edu.get("graduation_date"):
                try:
                    graduation_date = date_type.fromisoformat(edu["graduation_date"])
                except ValueError:
                    pass

            education.append(
                Education(
                    degree=edu.get("degree") or edu.get("degree_level") or "Unknown",
                    field_of_study=edu.get("field_of_study") or "Unknown",
                    institution=edu.get("institution") or "Unknown Institution",
                    location=edu.get("location"),
                    graduation_date=graduation_date,
                    gpa=edu.get("gpa"),
                    honors=edu.get("honors"),
                )
            )

        # Build headline from most recent experience
        headline = None
        if result.experience and result.experience[0].get("job_title"):
            headline = result.experience[0]["job_title"]

        # Build certifications
        from datetime import date as date_type

        certifications = []
        for cert in result.certifications:
            if not cert.get("name"):
                continue
            issue_date = None
            expiry_date = None
            if cert.get("issue_date"):
                try:
                    issue_date = date_type.fromisoformat(cert["issue_date"])
                except ValueError:
                    pass
            if cert.get("expiry_date"):
                try:
                    expiry_date = date_type.fromisoformat(cert["expiry_date"])
                except ValueError:
                    pass
            certifications.append(
                Certification(
                    name=cert["name"],
                    issuing_organization=cert.get("issuer") or "Unknown",
                    issue_date=issue_date,
                    expiration_date=expiry_date,
                    credential_id=cert.get("credential_id"),
                    credential_url=cert.get("credential_url"),
                )
            )

        # Build languages
        languages = [Language(language=lang) for lang in result.languages if lang]

        return CandidateCreate(
            first_name=first_name,
            last_name=last_name,
            contact=contact_info,
            headline=headline,
            summary=result.summary,
            skills=skills,
            work_experience=work_experience,
            education=education,
            certifications=certifications,
            languages=languages,
        )

    def to_parsed_content(self, result: ResumeParseResult) -> ParsedContent:
        """Convert parse result to ParsedContent model for Resume document."""
        sections = []

        if result.preprocessed:
            for section in result.preprocessed.sections:
                sections.append(
                    ParsedSection(
                        section_type=section.section_type,
                        title=section.title,
                        content=section.content,
                        start_position=section.start_pos,
                        end_position=section.end_pos,
                        confidence=section.confidence,
                    )
                )

        raw_text = ""
        if result.extraction_result:
            raw_text = result.extraction_result.text

        cleaned_text = ""
        if result.preprocessed:
            cleaned_text = result.preprocessed.cleaned_text

        return ParsedContent(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            sections=sections,
            word_count=result.preprocessed.word_count if result.preprocessed else 0,
            language=result.preprocessed.detected_language if result.preprocessed else "en",
        )


# Singleton instance
_resume_parser: Optional[ResumeParser] = None


def get_resume_parser() -> ResumeParser:
    """Get the resume parser singleton instance."""
    global _resume_parser
    if _resume_parser is None:
        _resume_parser = ResumeParser()
    return _resume_parser
