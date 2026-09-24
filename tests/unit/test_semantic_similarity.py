import os
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

import pytest
import numpy as np
from unittest.mock import Mock

from src.data.models.match import SemanticMatch
from src.ml.nlp.accurate_resume_parser import (
    ParsedResume, ExperienceEntry, SkillCategory, ContactInfo,
)
from src.data.models.job import Job, SkillRequirement


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_job(
    title: str = "Software Engineer",
    description: str = "Build reliable software.",
    responsibilities: list[str] | None = None,
    skill_names: list[str] | None = None,
) -> Job:
    return Job(
        title=title,
        description=description,
        responsibilities=responsibilities or ["Design systems", "Write code"],
        company_name="Acme Corp",
        skill_requirements=[
            SkillRequirement(name=s, is_required=True)
            for s in (skill_names or ["Python", "SQL"])
        ],
    )


def _make_parsed(
    summary: str = "Experienced developer.",
    skill_names: list[str] | None = None,
    exp_bullets: list[str] | None = None,
    raw_text: str = "Developer with Python experience.",
) -> ParsedResume:
    return ParsedResume(
        summary=summary,
        skills=[SkillCategory(category="Technical", skills=skill_names or ["Python", "Django"])],
        experience=[ExperienceEntry(
            title="Developer",
            company="Tech Co",
            bullets=exp_bullets or ["Built REST APIs", "Led code reviews"],
        )],
        raw_text=raw_text,
    )


def _mock_model(similarity_values: list[float] | None = None):
    """
    Mock EmbeddingModel.

    encode(str)  → 1-D unit array (JD side; in production served from LRU cache)
    encode(list) → 2-D array, one distinct unit vector per text (resume batch)
    similarity   → values consumed from similarity_values in call order
    """
    from src.ml.embeddings.embedding_model import EmbeddingModel

    mock = Mock(spec=EmbeddingModel)
    DIM: int = 8

    def _encode(texts, **kwargs):
        if isinstance(texts, list):
            n: int = len(texts)
            embs: np.ndarray = np.zeros((n, DIM))
            for i in range(n):
                embs[i, i % DIM] = 1.0
            return embs
        # Single string (JD-side): derive a distinct unit vector from the text
        # so that per-section routing bugs (wrong emb compared to wrong JD emb)
        # are detectable — all-identical vectors would mask them.
        dim_idx: int = abs(hash(texts)) % DIM
        vec: np.ndarray = np.zeros(DIM)
        vec[dim_idx] = 1.0
        return vec

    mock.encode.side_effect = _encode

    if similarity_values is not None:
        mock.similarity.side_effect = list(similarity_values)
    else:
        mock.similarity.return_value = 0.6

    return mock


# ── SemanticMatch model ───────────────────────────────────────────────────────

def test_semantic_match_has_weighted_similarity_field() -> None:
    sm: SemanticMatch = SemanticMatch()
    assert sm.weighted_similarity == 0.0


def test_semantic_match_weighted_similarity_clamped_above() -> None:
    sm: SemanticMatch = SemanticMatch(weighted_similarity=1.5)
    assert sm.weighted_similarity == pytest.approx(1.0)


def test_semantic_match_weighted_similarity_clamped_below() -> None:
    sm: SemanticMatch = SemanticMatch(weighted_similarity=-0.2)
    assert sm.weighted_similarity == pytest.approx(0.0)


# ── _build_resume_text ────────────────────────────────────────────────────────

def test_build_resume_text_includes_all_bullets() -> None:
    """Removing the [:3] cap means all bullets appear in the embedding text."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    bullets: list[str] = [f"task_{i}" for i in range(6)]
    parsed: ParsedResume = _make_parsed(exp_bullets=bullets)
    matcher: SemanticMatcher = SemanticMatcher(embedding_model=_mock_model())

    text: str = matcher._build_resume_text(parsed)

    for bullet in bullets:
        assert bullet in text, f"Expected bullet '{bullet}' in resume text"


def test_build_resume_text_three_bullets_unaffected() -> None:
    """Regression: resumes with ≤3 bullets still work correctly."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    bullets: list[str] = ["alpha", "beta", "gamma"]
    parsed: ParsedResume = _make_parsed(exp_bullets=bullets)
    matcher: SemanticMatcher = SemanticMatcher(embedding_model=_mock_model())

    text: str = matcher._build_resume_text(parsed)

    for bullet in bullets:
        assert bullet in text


# ── compute_similarity_from_parsed ───────────────────────────────────────────

def test_weighted_similarity_formula() -> None:
    """weighted_similarity = 0.35*overall + 0.30*skills + 0.25*exp + 0.10*summary."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher, _SECTION_WEIGHTS

    overall, skills, exp, summary = 0.80, 0.60, 0.70, 0.50
    expected: float = (
        _SECTION_WEIGHTS["overall"] * overall
        + _SECTION_WEIGHTS["skills"] * skills
        + _SECTION_WEIGHTS["experience"] * exp
        + _SECTION_WEIGHTS["summary"] * summary
    )

    result: SemanticMatch = SemanticMatcher(
        embedding_model=_mock_model([overall, skills, exp, summary])
    ).compute_similarity_from_parsed(_make_parsed(), _make_job())

    assert result.weighted_similarity == pytest.approx(expected, abs=1e-4)


def test_all_four_section_scores_are_populated() -> None:
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    overall, skills, exp, summary = 0.80, 0.60, 0.70, 0.50
    result: SemanticMatch = SemanticMatcher(
        embedding_model=_mock_model([overall, skills, exp, summary])
    ).compute_similarity_from_parsed(_make_parsed(), _make_job())

    assert result.overall_similarity == pytest.approx(overall, abs=1e-4)
    assert result.skills_similarity == pytest.approx(skills, abs=1e-4)
    assert result.experience_similarity == pytest.approx(exp, abs=1e-4)
    assert result.summary_similarity == pytest.approx(summary, abs=1e-4)


def test_resume_side_uses_one_batch_encode_call() -> None:
    """All 4 resume-side texts are batched into exactly one encode(list) call."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    mock: Mock = _mock_model()
    SemanticMatcher(embedding_model=mock).compute_similarity_from_parsed(
        _make_parsed(), _make_job()
    )

    list_calls: list = [
        c for c in mock.encode.call_args_list if isinstance(c.args[0], list)
    ]
    assert len(list_calls) == 1, (
        f"Expected exactly 1 batch encode call, got {len(list_calls)}"
    )


def test_batch_encode_contains_all_bullet_text() -> None:
    """The experience text passed in the batch includes all bullets, not just 3."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    bullets: list[str] = [f"bullet_{i}" for i in range(6)]
    parsed: ParsedResume = _make_parsed(exp_bullets=bullets)

    mock: Mock = _mock_model()
    SemanticMatcher(embedding_model=mock).compute_similarity_from_parsed(
        parsed, _make_job()
    )

    list_calls: list = [
        c for c in mock.encode.call_args_list if isinstance(c.args[0], list)
    ]
    all_batch_text: str = " ".join(list_calls[0].args[0])
    for bullet in bullets:
        assert bullet in all_batch_text, f"'{bullet}' missing from batch texts"


# ── matching_engine integration ───────────────────────────────────────────────

def test_matching_engine_uses_weighted_similarity_as_semantic_score() -> None:
    """semantic_score in MatchResult equals weighted_similarity, not overall_similarity."""
    from src.core.matching.matching_engine import MatchingEngine

    engine: MatchingEngine = MatchingEngine(
        use_semantic=True, use_bias_detection=False, use_explainability=False
    )

    known_weighted: float = 0.73
    known_overall: float = 0.55

    mock_sm: SemanticMatch = SemanticMatch(
        overall_similarity=known_overall,
        weighted_similarity=known_weighted,
        skills_similarity=0.80,
        experience_similarity=0.70,
        summary_similarity=0.50,
    )
    mock_matcher: Mock = Mock()
    mock_matcher.compute_similarity_from_parsed.return_value = mock_sm
    engine._semantic_matcher = mock_matcher

    result = engine.match_from_parsed(_make_parsed(), _make_job())

    assert result.semantic_score == pytest.approx(known_weighted, abs=1e-3), (
        f"Expected semantic_score={known_weighted}, got {result.semantic_score}. "
        "matching_engine should use weighted_similarity, not overall_similarity."
    )


# ── compute_similarity() helpers and fixture ─────────────────────────────────

def _make_resume_result(
    text: str = "Python developer with 5 years of experience building web apps.",
    skills: list[dict] | None = None,
) -> Mock:
    """Minimal ResumeParseResult mock for compute_similarity() tests."""
    result = Mock()
    result.extraction_result.text = text
    result.skills = skills or [{"name": "Python"}, {"name": "Django"}]
    result.preprocessed = None  # forces fallback to extraction_result.text
    return result


def _make_jd_result(
    raw_text: str = "Software engineer role requiring Python and SQL skills.",
    required_skills: list[str] | None = None,
    responsibilities: list[str] | None = None,
    title: str = "Software Engineer",
) -> Mock:
    """Minimal JDParseResult mock for compute_similarity() tests."""
    result = Mock()
    result.raw_text = raw_text
    result.required_skills = required_skills or ["Python", "SQL"]
    result.preferred_skills = []
    result.responsibilities = responsibilities or ["Build APIs", "Write tests"]
    result.title = title
    return result


@pytest.fixture
def mock_embedding_model() -> Mock:
    """
    Deterministic embedding model stub for compute_similarity() tests.

    encode(str)  → content-hashed unit vector (distinct per text, catches
                   per-section routing bugs where wrong resume emb is compared
                   against wrong JD emb — aligned with _mock_model() behaviour)
    encode(list) → one distinct unit vector per text (resume batch side)
    similarity   → dot product of the two unit vectors
    """
    DIM: int = 4
    model = Mock()

    def _encode(texts, normalize=True, show_progress=False):
        if isinstance(texts, str):
            dim_idx: int = abs(hash(texts)) % DIM
            vec = np.zeros(DIM)
            vec[dim_idx] = 1.0
            return vec
        n = len(texts)
        embs = np.zeros((n, DIM))
        for i in range(n):
            embs[i, i % DIM] = 1.0
        return embs

    model.encode.side_effect = _encode
    model.similarity.side_effect = lambda a, b: float(np.dot(a, b))
    return model


def test_compute_similarity_returns_nonzero_weighted_similarity() -> None:
    """compute_similarity() must return weighted_similarity > 0 when both sides have content."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher
    # _mock_model() with no args returns similarity=0.6 for all section pairs,
    # ensuring weighted_similarity > 0 without depending on vector alignment.
    matcher = SemanticMatcher(embedding_model=_mock_model())
    resume = _make_resume_result()
    jd = _make_jd_result()

    result = matcher.compute_similarity(resume, jd)

    assert result.weighted_similarity > 0.0, (
        "compute_similarity() returned weighted_similarity=0.0; "
        "it must compute the weighted combination of section scores."
    )


def test_compute_similarity_empty_inputs_return_zero_weighted_similarity(
    mock_embedding_model: Mock,
) -> None:
    """compute_similarity() must return weighted_similarity=0.0 on empty resume or JD."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher
    matcher = SemanticMatcher(embedding_model=mock_embedding_model)

    empty_resume = Mock()
    empty_resume.extraction_result = None
    empty_resume.preprocessed = None
    empty_resume.skills = []
    jd = _make_jd_result()

    result = matcher.compute_similarity(empty_resume, jd)
    assert result.weighted_similarity == 0.0


def test_compute_similarity_empty_jd_returns_zero_weighted_similarity(
    mock_embedding_model: Mock,
) -> None:
    """compute_similarity() must return weighted_similarity=0.0 when JD has no text."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher
    matcher = SemanticMatcher(embedding_model=mock_embedding_model)

    resume = _make_resume_result()
    empty_jd = _make_jd_result(raw_text="")

    result = matcher.compute_similarity(resume, empty_jd)
    assert result.weighted_similarity == 0.0
    # encode must never be called when the guard triggers
    mock_embedding_model.encode.assert_not_called()
    # _model_name lazy property must resolve to a non-empty string even on early exit
    assert result.model_used is not None and result.model_used != "", (
        "model_used must be a non-empty string even when returning a zero-score match"
    )


# ── batch_compute_similarity ──────────────────────────────────────────────────

def test_batch_compute_similarity_returns_one_result_per_resume() -> None:
    """Result list length must equal the number of input resumes."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    resumes = [_make_resume_result() for _ in range(4)]
    jd = _make_jd_result()
    matcher = SemanticMatcher(embedding_model=_mock_model())

    results = matcher.batch_compute_similarity(resumes, jd)

    assert len(results) == 4, "One (index, SemanticMatch) tuple expected per resume"


def test_batch_compute_similarity_indices_are_sorted() -> None:
    """Returned tuples must be sorted by the original resume index."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    resumes = [_make_resume_result(text=f"resume {i}") for i in range(3)]
    jd = _make_jd_result()
    matcher = SemanticMatcher(embedding_model=_mock_model())

    results = matcher.batch_compute_similarity(resumes, jd)
    indices = [idx for idx, _ in results]

    assert indices == sorted(indices), "Results must be sorted by original resume index"


def test_batch_compute_similarity_all_resumes_empty_returns_zero_scores() -> None:
    """When every resume has no extractable text, all weighted_similarity values are 0."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    empty_resume = Mock()
    empty_resume.extraction_result = None
    empty_resume.preprocessed = None
    empty_resume.skills = []
    jd = _make_jd_result()
    matcher = SemanticMatcher(embedding_model=_mock_model())

    results = matcher.batch_compute_similarity([empty_resume, empty_resume], jd)

    assert len(results) == 2
    for _, sm in results:
        assert sm.weighted_similarity == 0.0


def test_batch_compute_similarity_nonzero_scores_for_valid_resumes() -> None:
    """Valid resumes must produce weighted_similarity > 0."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    resumes = [_make_resume_result() for _ in range(3)]
    jd = _make_jd_result()
    matcher = SemanticMatcher(embedding_model=_mock_model())

    results = matcher.batch_compute_similarity(resumes, jd)

    for _, sm in results:
        assert sm.weighted_similarity > 0.0, (
            "batch_compute_similarity should return non-zero scores for valid resumes"
        )


def test_batch_compute_similarity_mixed_valid_and_empty_resumes() -> None:
    """Empty resumes get zero score; valid resumes get non-zero score in the same batch."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    valid = _make_resume_result()
    empty = Mock()
    empty.extraction_result = None
    empty.preprocessed = None
    empty.skills = []

    matcher = SemanticMatcher(embedding_model=_mock_model())
    results = matcher.batch_compute_similarity([valid, empty, valid], jd_result=_make_jd_result())

    result_map = {idx: sm for idx, sm in results}
    assert result_map[0].weighted_similarity > 0.0, "Index 0 (valid) should have non-zero score"
    assert result_map[1].weighted_similarity == 0.0, "Index 1 (empty) should have zero score"
    assert result_map[2].weighted_similarity > 0.0, "Index 2 (valid) should have non-zero score"


def test_batch_compute_similarity_jd_embeddings_encoded_once() -> None:
    """All four JD-side embeddings must be computed exactly once, regardless of resume count."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    mock = _mock_model()
    resumes = [_make_resume_result() for _ in range(5)]
    jd = _make_jd_result(
        raw_text="Engineering role at Acme.",
        required_skills=["Python"],
        responsibilities=["Build APIs"],
    )

    SemanticMatcher(embedding_model=mock).batch_compute_similarity(resumes, jd)

    # Each string call to encode() is a JD-side or a batch call.
    # String calls (non-list) are JD-side — there should be exactly 4
    # (jd_full, jd_skills, jd_resp, jd_summary).
    str_calls = [c for c in mock.encode.call_args_list if isinstance(c.args[0], str)]
    assert len(str_calls) == 4, (
        f"Expected 4 JD-side encode calls (full, skills, resp, summary), got {len(str_calls)}"
    )


# ── index_resume / index_job ──────────────────────────────────────────────────

def test_index_resume_calls_upsert_with_resume_id() -> None:
    """index_resume must call vector store upsert with the supplied resume_id."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    mock_store = Mock()
    mock_store.upsert = Mock()
    matcher = SemanticMatcher(
        embedding_model=_mock_model(),
        resume_store=mock_store,
    )

    resume = _make_resume_result(text="Python developer.")
    matcher.index_resume("resume-42", resume)

    mock_store.upsert.assert_called_once()
    call_kwargs = mock_store.upsert.call_args
    ids_arg: list[str] = call_kwargs.kwargs.get("ids") or call_kwargs.args[0]
    assert ids_arg == ["resume-42"], "upsert must be called with the provided resume_id"


def test_index_resume_skips_upsert_when_no_text() -> None:
    """index_resume must not call upsert and must log a warning when resume has no text."""
    from unittest.mock import patch
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    mock_store = Mock()
    empty_resume = Mock()
    empty_resume.extraction_result = None
    empty_resume.preprocessed = None

    matcher = SemanticMatcher(embedding_model=_mock_model(), resume_store=mock_store)

    with patch("src.ml.embeddings.semantic_similarity.logger") as mock_logger:
        matcher.index_resume("no-text-resume", empty_resume)

    mock_store.upsert.assert_not_called()
    mock_logger.warning.assert_called_once()
    warning_args: tuple = mock_logger.warning.call_args.args
    assert "no-text-resume" in warning_args[1], (
        "The warning must include the resume_id so operators can trace the skipped entry"
    )


def test_index_job_calls_upsert_with_job_id() -> None:
    """index_job must call vector store upsert with the supplied job_id."""
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    mock_store = Mock()
    mock_store.upsert = Mock()
    matcher = SemanticMatcher(
        embedding_model=_mock_model(),
        job_store=mock_store,
    )

    jd = _make_jd_result(raw_text="Build scalable APIs in Python.")
    matcher.index_job("job-99", jd)

    mock_store.upsert.assert_called_once()
    call_kwargs = mock_store.upsert.call_args
    ids_arg: list[str] = call_kwargs.kwargs.get("ids") or call_kwargs.args[0]
    assert ids_arg == ["job-99"], "upsert must be called with the provided job_id"


def test_index_job_skips_upsert_when_no_text() -> None:
    """index_job must not call upsert and must log a warning when JD has no text."""
    from unittest.mock import patch
    from src.ml.embeddings.semantic_similarity import SemanticMatcher

    mock_store = Mock()
    empty_jd = _make_jd_result(raw_text="")
    matcher = SemanticMatcher(embedding_model=_mock_model(), job_store=mock_store)

    with patch("src.ml.embeddings.semantic_similarity.logger") as mock_logger:
        matcher.index_job("no-text-job", empty_jd)

    mock_store.upsert.assert_not_called()
    mock_logger.warning.assert_called_once()
    warning_args: tuple = mock_logger.warning.call_args.args
    assert "no-text-job" in warning_args[1], (
        "The warning must include the job_id so operators can trace the skipped entry"
    )
