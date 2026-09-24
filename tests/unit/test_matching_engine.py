"""
Tests for src.core.matching.matching_engine — MatchingEngine business logic.

All tests run with use_semantic=False, use_bias_detection=False,
use_explainability=False so no external services are required.
"""

import pytest

from src.core.matching.matching_engine import MatchingEngine, MatchResult
from src.data.models import (
    ScoreBreakdown,
    SkillMatch,
)
from src.ml.nlp.jd_parser import JDParseResult
from src.ml.nlp.resume_parser import ResumeParseResult
from src.utils.constants import MatchScoreLevel


# ── _match_skills ────────────────────────────────────────────────────────────


class TestMatchSkills:
    def test_all_required_matched(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(skills=[
            {"name": "python", "category": "pl", "proficiency": "expert", "confidence": 0.9, "source": "section"},
            {"name": "django", "category": "fw", "proficiency": "advanced", "confidence": 0.9, "source": "section"},
        ])
        jd = make_jd_parse_result(required_skills=["python", "django"], preferred_skills=[])
        matches, score = matching_engine._match_skills(resume, jd)
        assert score == 1.0
        assert all(m.candidate_has_skill for m in matches)

    def test_none_matched(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(skills=[
            {"name": "rust", "category": "pl", "proficiency": "expert", "confidence": 0.9, "source": "section"},
        ])
        jd = make_jd_parse_result(required_skills=["java", "spring"], preferred_skills=[])
        matches, score = matching_engine._match_skills(resume, jd)
        assert score == 0.0

    def test_partial_match_via_related_skill(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        # Candidate has "django" which is related to "python"
        resume = make_resume_parse_result(skills=[
            {"name": "django", "category": "fw", "proficiency": "advanced", "confidence": 0.9, "source": "section"},
        ])
        jd = make_jd_parse_result(required_skills=["python"], preferred_skills=[])
        matches, score = matching_engine._match_skills(resume, jd)
        assert any(m.partial_match for m in matches)
        assert score == 0.5  # Partial match gives 0.5

    def test_preferred_skills_weighting(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(skills=[
            {"name": "docker", "category": "devops", "proficiency": "intermediate", "confidence": 0.9, "source": "text"},
        ])
        jd = make_jd_parse_result(required_skills=[], preferred_skills=["docker", "kubernetes"])
        matches, score = matching_engine._match_skills(resume, jd)
        # 1 of 2 preferred skills matched
        assert 0.0 < score < 1.0

    def test_case_insensitive(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(skills=[
            {"name": "Python", "category": "pl", "proficiency": "expert", "confidence": 0.9, "source": "section"},
        ])
        jd = make_jd_parse_result(required_skills=["PYTHON"], preferred_skills=[])
        matches, score = matching_engine._match_skills(resume, jd)
        assert score == 1.0

    def test_empty_skills(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(skills=[])
        jd = make_jd_parse_result(required_skills=[], preferred_skills=[])
        matches, score = matching_engine._match_skills(resume, jd)
        assert score == 0.0
        assert len(matches) == 0

    def test_candidate_has_skills_but_no_requirements(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(skills=[
            {"name": "python", "category": "pl", "proficiency": "expert", "confidence": 0.9, "source": "section"},
        ])
        jd = make_jd_parse_result(required_skills=[], preferred_skills=[])
        matches, score = matching_engine._match_skills(resume, jd)
        assert score == 0.5  # Has skills but no requirements


# ── _find_related_skill ──────────────────────────────────────────────────────


class TestFindRelatedSkill:
    def test_python_django_group(self, matching_engine):
        candidate_skills = {"django", "flask"}
        result = matching_engine._find_related_skill("python", candidate_skills)
        assert result in ("django", "flask")

    def test_javascript_react_group(self, matching_engine):
        candidate_skills = {"react", "vue"}
        result = matching_engine._find_related_skill("javascript", candidate_skills)
        assert result in ("react", "vue")

    def test_no_match(self, matching_engine):
        candidate_skills = {"python", "django"}
        result = matching_engine._find_related_skill("java", candidate_skills)
        assert result is None

    def test_same_skill_not_returned(self, matching_engine):
        candidate_skills = {"python"}
        result = matching_engine._find_related_skill("python", candidate_skills)
        assert result is None


# ── _match_experience ────────────────────────────────────────────────────────


class TestMatchExperience:
    def test_meets_requirement(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(total_experience_years=7.0)
        jd = make_jd_parse_result(experience_years_min=5.0)
        match, score = matching_engine._match_experience(resume, jd)
        assert match is not None
        assert match.meets_minimum is True
        assert score == 1.0

    def test_close_to_requirement(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        # 70% of 5 = 3.5, candidate has 4.0
        resume = make_resume_parse_result(total_experience_years=4.0)
        jd = make_jd_parse_result(experience_years_min=5.0)
        match, score = matching_engine._match_experience(resume, jd)
        assert match is not None
        assert match.meets_minimum is False
        assert 0.7 < score < 1.0

    def test_below_requirement(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(total_experience_years=1.0)
        jd = make_jd_parse_result(experience_years_min=5.0)
        match, score = matching_engine._match_experience(resume, jd)
        assert match is not None
        assert score < 0.5

    def test_zero_experience(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(total_experience_years=0.0)
        jd = make_jd_parse_result(experience_years_min=5.0)
        match, score = matching_engine._match_experience(resume, jd)
        assert score == 0.0

    def test_no_requirement(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(total_experience_years=3.0)
        jd = make_jd_parse_result(experience_years_min=None)
        match, score = matching_engine._match_experience(resume, jd)
        assert match is None
        assert score == 1.0

    def test_no_requirement_no_experience(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(total_experience_years=0.0)
        jd = make_jd_parse_result(experience_years_min=None)
        match, score = matching_engine._match_experience(resume, jd)
        assert score == 0.5


# ── _match_education ─────────────────────────────────────────────────────────


class TestMatchEducation:
    def test_meets_requirement(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(highest_education="bachelor")
        jd = make_jd_parse_result(education_requirement="bachelor")
        match, score = matching_engine._match_education(resume, jd)
        assert match is not None
        assert match.meets_requirement is True
        assert score == 1.0

    def test_exceeds_requirement(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(highest_education="master")
        jd = make_jd_parse_result(education_requirement="bachelor")
        match, score = matching_engine._match_education(resume, jd)
        assert match.meets_requirement is True
        assert score == 1.0

    def test_one_level_below(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(highest_education="bachelor")
        jd = make_jd_parse_result(education_requirement="master")
        match, score = matching_engine._match_education(resume, jd)
        assert score == 0.7

    def test_no_requirement(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(highest_education="bachelor")
        jd = make_jd_parse_result(education_requirement=None)
        match, score = matching_engine._match_education(resume, jd)
        assert match is None
        assert score == 1.0

    def test_no_candidate_degree(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(highest_education=None)
        jd = make_jd_parse_result(education_requirement="bachelor")
        match, score = matching_engine._match_education(resume, jd)
        assert score == 0.3

    def test_equal_level(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result(highest_education="phd")
        jd = make_jd_parse_result(education_requirement="phd")
        match, score = matching_engine._match_education(resume, jd)
        assert match.meets_requirement is True
        assert score == 1.0


# ── _match_keywords ──────────────────────────────────────────────────────────


class TestMatchKeywords:
    def test_keyword_extraction_and_matching(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        from src.ml.nlp.extractors.base import ExtractionResult
        resume = make_resume_parse_result()
        resume.extraction_result = ExtractionResult(
            text="Experienced in backend development with microservices architecture and REST APIs.",
        )
        jd = make_jd_parse_result(
            responsibilities=["Design backend microservices", "Build REST APIs"],
            qualifications=["Strong architecture knowledge"],
        )
        match, score = matching_engine._match_keywords(resume, jd)
        assert match is not None
        assert match.total_keywords > 0
        assert score >= 0.0

    def test_missing_resume_text(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result()
        resume.extraction_result = None
        resume.preprocessed = None
        jd = make_jd_parse_result(responsibilities=["Design APIs"])
        match, score = matching_engine._match_keywords(resume, jd)
        assert score == 0.0

    def test_empty_jd_responsibilities(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        from src.ml.nlp.extractors.base import ExtractionResult
        resume = make_resume_parse_result()
        resume.extraction_result = ExtractionResult(text="Some resume text here with content")
        jd = make_jd_parse_result(responsibilities=[], qualifications=[])
        match, score = matching_engine._match_keywords(resume, jd)
        # No keywords to extract => 0.5 fallback
        assert score == 0.5


# ── _calculate_breakdown ─────────────────────────────────────────────────────


class TestCalculateBreakdown:
    def test_default_weights(self, matching_engine):
        result = MatchResult(
            skills_score=1.0,
            experience_score=1.0,
            education_score=1.0,
            keyword_score=1.0,
            semantic_score=0.0,  # No semantic match
        )
        bd = matching_engine._calculate_breakdown(result)
        assert isinstance(bd, ScoreBreakdown)
        # Without semantic match, semantic_score falls back to keyword_score (1.0)
        expected = 1.0 * 0.35 + 1.0 * 0.25 + 1.0 * 0.15 + 1.0 * 0.20 + 1.0 * 0.05
        assert abs(bd.total_score - expected) < 1e-6

    def test_custom_weights(self):
        engine = MatchingEngine(
            weights={"skills_match": 0.5, "experience_match": 0.3, "education_match": 0.1, "semantic_similarity": 0.05, "keyword_match": 0.05},
            use_semantic=False,
            use_bias_detection=False,
            use_explainability=False,
        )
        result = MatchResult(
            skills_score=0.8,
            experience_score=0.6,
            education_score=1.0,
            keyword_score=0.5,
            semantic_score=0.0,
        )
        bd = engine._calculate_breakdown(result)
        expected = 0.8 * 0.5 + 0.6 * 0.3 + 1.0 * 0.1 + 0.5 * 0.05 + 0.5 * 0.05
        assert abs(bd.total_score - expected) < 1e-6

    def test_semantic_fallback_to_keyword(self, matching_engine):
        """When no semantic_match, semantic_score uses keyword_score."""
        result = MatchResult(
            skills_score=0.8,
            experience_score=0.7,
            education_score=0.9,
            keyword_score=0.6,
            semantic_score=0.0,
            semantic_match=None,  # Not available
        )
        bd = matching_engine._calculate_breakdown(result)
        # Semantic weight (0.20) should be applied to keyword_score (0.6) as fallback
        assert bd.semantic_score == 0.6


# ── _generate_explanation ────────────────────────────────────────────────────


class TestGenerateExplanation:
    def test_excellent_summary(self, matching_engine):
        result = MatchResult(
            candidate_name="Jane Smith",
            job_title="Senior Engineer",
            overall_score=0.90,
            skills_score=0.95,
            skill_matches=[
                SkillMatch(skill_name="python", required=True, candidate_has_skill=True, match_score=1.0),
            ],
        )
        explanation = matching_engine._generate_explanation(result)
        assert "excellent match" in explanation.summary.lower()
        assert len(explanation.strengths) > 0

    def test_poor_summary(self, matching_engine):
        result = MatchResult(
            candidate_name="John Doe",
            job_title="Lead Engineer",
            overall_score=0.30,
            skills_score=0.2,
            skill_matches=[
                SkillMatch(skill_name="python", required=True, candidate_has_skill=False, match_score=0.0),
            ],
        )
        explanation = matching_engine._generate_explanation(result)
        assert "not be the best fit" in explanation.summary.lower()
        assert len(explanation.gaps) > 0

    def test_strengths_populated(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        from src.data.models import ExperienceMatch
        result = MatchResult(
            candidate_name="Jane",
            job_title="Engineer",
            overall_score=0.75,
            skills_score=0.8,
            experience_score=0.9,
            skill_matches=[
                SkillMatch(skill_name="python", required=True, candidate_has_skill=True, match_score=1.0),
            ],
            experience_match=ExperienceMatch(
                required_years=3.0,
                candidate_years=5.0,
                years_difference=2.0,
                meets_minimum=True,
                score=1.0,
            ),
        )
        explanation = matching_engine._generate_explanation(result)
        assert len(explanation.strengths) >= 2  # skills + experience

    def test_gaps_populated(self, matching_engine):
        result = MatchResult(
            candidate_name="Jane",
            job_title="Engineer",
            overall_score=0.55,
            skills_score=0.3,
            skill_matches=[
                SkillMatch(skill_name="python", required=True, candidate_has_skill=False, match_score=0.0),
                SkillMatch(skill_name="java", required=True, candidate_has_skill=False, match_score=0.0),
            ],
        )
        explanation = matching_engine._generate_explanation(result)
        assert len(explanation.gaps) > 0
        assert len(explanation.recommendations) > 0


# ── rank_candidates ──────────────────────────────────────────────────────────


class TestRankCandidates:
    def test_descending_sort(self, matching_engine):
        c1 = MatchResult(overall_score=0.50)
        c2 = MatchResult(overall_score=0.90)
        c3 = MatchResult(overall_score=0.70)
        ranked = matching_engine.rank_candidates([c1, c2, c3])
        assert ranked[0].overall_score == 0.90
        assert ranked[1].overall_score == 0.70
        assert ranked[2].overall_score == 0.50

    def test_empty_list(self, matching_engine):
        assert matching_engine.rank_candidates([]) == []


# ── match() full pipeline ────────────────────────────────────────────────────


class TestMatchFullPipeline:
    def test_produces_populated_result(self, matching_engine, make_resume_parse_result, make_jd_parse_result):
        resume = make_resume_parse_result()
        jd = make_jd_parse_result()
        result = matching_engine.match(resume, jd)

        assert isinstance(result, MatchResult)
        assert result.candidate_name == "Jane Smith"
        assert result.job_title == "Senior Software Engineer"
        assert 0.0 <= result.overall_score <= 1.0
        assert result.score_level in list(MatchScoreLevel)
        assert result.score_breakdown is not None
        assert result.explanation is not None
        assert len(result.skill_matches) > 0

    def test_no_contact_defaults_to_unknown(self, matching_engine, make_jd_parse_result):
        resume = ResumeParseResult(contact=None, skills=[], total_experience_years=0)
        jd = make_jd_parse_result()
        result = matching_engine.match(resume, jd)
        assert result.candidate_name == "Unknown Candidate"
