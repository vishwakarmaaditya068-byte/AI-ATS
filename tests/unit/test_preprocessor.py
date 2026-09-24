import os
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

import pytest

from src.ml.nlp.preprocessor import TextPreprocessor, PreprocessedText


def _pp() -> TextPreprocessor:
    return TextPreprocessor()


# ── _detect_language ───────────────────────────────────────────────────────────

def test_detect_language_english_returns_en() -> None:
    """Standard English resume text must be detected as 'en'."""
    text: str = (
        "Professional summary: experienced software engineer with expertise "
        "in Python and cloud infrastructure. Strong skills in distributed systems."
    )
    pp: TextPreprocessor = _pp()
    assert pp._detect_language(text) == "en"


def test_detect_language_french_returns_fr() -> None:
    """French resume with accented characters and stop words must return 'fr'."""
    text: str = (
        "Expérience professionnelle : développeur logiciel avec des compétences "
        "en Python et en Java. Maîtrise de la conception des systèmes distribués."
    )
    pp: TextPreprocessor = _pp()
    assert pp._detect_language(text) == "fr"


def test_detect_language_german_returns_de() -> None:
    """German resume with umlauts and stop words must return 'de'."""
    text: str = (
        "Berufserfahrung: Softwareentwickler mit Kenntnissen in Python und Java. "
        "Erfahrung in der Entwicklung von verteilten Systemen und Datenbanken."
    )
    pp: TextPreprocessor = _pp()
    assert pp._detect_language(text) == "de"


def test_detect_language_spanish_returns_es() -> None:
    """Spanish resume with ñ and stop words must return 'es'."""
    text: str = (
        "Experiencia profesional: desarrollador de software con conocimientos "
        "en Python y Java. Habilidades en diseño de sistemas y bases de datos. "
        "Implementación de soluciones técnicas para empresas de tecnología."
    )
    pp: TextPreprocessor = _pp()
    assert pp._detect_language(text) == "es"


def test_detect_language_empty_returns_en() -> None:
    """Empty text must return the default 'en'."""
    pp: TextPreprocessor = _pp()
    assert pp._detect_language("") == "en"


# ── non-English warning in preprocess() ───────────────────────────────────────

def test_non_english_adds_warning() -> None:
    """preprocess() must add a warning when the detected language is not English."""
    french_resume: str = (
        "Expérience professionnelle\n"
        "Développeur logiciel chez ABC avec des compétences en Python et Java.\n"
        "Formation\n"
        "Licence en informatique de l'Université de Paris."
    )
    result: PreprocessedText = _pp().preprocess(french_resume)

    assert any("language" in w.lower() or "english" in w.lower() for w in result.warnings), (
        "A non-English resume must produce a language warning"
    )


def test_english_resume_no_language_warning() -> None:
    """preprocess() must NOT add a language warning for English resumes."""
    english_resume: str = (
        "Professional Experience\n"
        "Software Engineer at ABC Corp with strong Python and Java skills.\n"
        "Education\n"
        "Bachelor of Science in Computer Science from State University."
    )
    result: PreprocessedText = _pp().preprocess(english_resume)

    assert not any("language" in w.lower() and "non-english" in w.lower() for w in result.warnings)


# ── _detect_sections() position accuracy ──────────────────────────────────────

def test_section_start_pos_matches_text_position() -> None:
    """section.start_pos must equal the character index of the header in cleaned text."""
    text: str = "EXPERIENCE\nWorked at Corp\n\nEDUCATION\nBS Computer Science"
    result: PreprocessedText = _pp().preprocess(text)

    edu_section = next(
        (s for s in result.sections if s.section_type == "education"), None
    )
    assert edu_section is not None, "Education section not found"
    cleaned: str = result.cleaned_text
    assert cleaned[edu_section.start_pos:].startswith("EDUCATION"), (
        f"start_pos={edu_section.start_pos} does not point to 'EDUCATION' in cleaned text"
    )


def test_section_end_pos_does_not_exceed_text_length() -> None:
    """section.end_pos must never exceed the total length of cleaned text."""
    text: str = (
        "SKILLS\nPython, Java, SQL\n\n"
        "EXPERIENCE\nSoftware Engineer at Corp\n\n"
        "EDUCATION\nBS Computer Science"
    )
    result: PreprocessedText = _pp().preprocess(text)

    cleaned_len: int = len(result.cleaned_text)
    for section in result.sections:
        assert section.end_pos <= cleaned_len, (
            f"Section '{section.section_type}' has end_pos={section.end_pos} "
            f"which exceeds cleaned text length={cleaned_len}"
        )
