"""
NLP pipeline for AI-ATS.

Provides resume parsing, text preprocessing, and information extraction
capabilities for processing candidate resumes.

Main Components:
- ResumeParser: Main orchestrator for parsing resumes
- TextPreprocessor: Text cleaning and section detection
- ExtractorFactory: Document text extraction (PDF, DOCX, TXT)
- ContactParser: Contact information extraction
- SkillsParser: Skills extraction and categorization
- ExperienceParser: Work experience extraction
- EducationParser: Education extraction
"""

from .resume_parser import (
    ResumeParser,
    ResumeParseResult,
    get_resume_parser,
)

from .preprocessor import (
    TextPreprocessor,
    PreprocessedText,
    TextSection,
    SECTION_HEADERS,
)

from .extractors import (
    ExtractorFactory,
    ExtractionResult,
    BaseExtractor,
    PDFExtractor,
    DOCXExtractor,
    TextExtractor,
    get_extractor,
)

from .parsers import (
    ContactParser,
    ContactInfo,
    SkillsParser,
    ExtractedSkill,
    ExperienceParser,
    ExtractedExperience,
    EducationParser,
    ExtractedEducation,
)

from .jd_parser import (
    JDParser,
    JDParseResult,
    get_jd_parser,
)

from .accurate_jd_parser import (
    AccurateJDParser,
    ParsedJob,
    get_accurate_jd_parser,
)

__all__ = [
    # Main parser
    "ResumeParser",
    "ResumeParseResult",
    "get_resume_parser",
    # Preprocessor
    "TextPreprocessor",
    "PreprocessedText",
    "TextSection",
    "SECTION_HEADERS",
    # Extractors
    "ExtractorFactory",
    "ExtractionResult",
    "BaseExtractor",
    "PDFExtractor",
    "DOCXExtractor",
    "TextExtractor",
    "get_extractor",
    # Parsers
    "ContactParser",
    "ContactInfo",
    "SkillsParser",
    "ExtractedSkill",
    "ExperienceParser",
    "ExtractedExperience",
    "EducationParser",
    "ExtractedEducation",
    # JD Parser
    "JDParser",
    "JDParseResult",
    "get_jd_parser",
    # Accurate JD Parser
    "AccurateJDParser",
    "ParsedJob",
    "get_accurate_jd_parser",
]
