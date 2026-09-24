"""
Candidate-Job matching engine.

Scores and ranks candidates against job requirements using multiple
matching strategies including skills matching, experience matching,
education matching, keyword matching, and semantic similarity.
"""
from __future__ import annotations

import datetime
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from src.ml.nlp.accurate_resume_parser import ParsedResume as ParsedResumeType
    from src.data.models.job import Job as JobType
    from src.core.matching.skill_scorer import EmbeddingSkillScorer as EmbeddingSkillScorerType
    from src.core.matching.experience_scorer import DomainAwareExperienceScorer as DomainAwareExperienceScorerType
    from src.core.matching.education_scorer import EmbeddingEducationScorer as EmbeddingEducationScorerType

from src.data.models import (
    BiasCheckResult,
    EducationMatch,
    ExperienceMatch,
    Explanation,
    ExplanationFactor,
    KeywordMatch,
    Match,
    MatchStatus,
    ScoreBreakdown,
    SemanticMatch,
    SkillMatch,
)
from src.ml.nlp import ResumeParseResult, JDParseResult
from src.utils.constants import (
    DEFAULT_SCORING_WEIGHTS,
    EDUCATION_LEVELS,
    MatchScoreLevel,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


_KEYWORD_STOPWORDS: frozenset[str] = frozenset({
    "with", "have", "will", "that", "this", "from", "your", "they",
    "their", "what", "when", "where", "which", "would", "could",
    "should", "must", "able", "about", "experience", "work", "team",
})


@dataclass
class _MinimalPreprocessed:
    """Thin preprocessed shim used by match_from_parsed() to satisfy _match_keywords."""

    cleaned_text: str
    sections: list = field(default_factory=list)


def _estimate_years(duration: str) -> float:
    """Best-effort year estimate from a duration string like '2019-2022' or '2 years'."""
    if not duration:
        return 1.0
    # "N years" or "N year"
    m = re.search(r"(\d+(?:\.\d+)?)\s*year", duration, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # "N months"
    m = re.search(r"(\d+)\s*month", duration, re.IGNORECASE)
    if m:
        return round(float(m.group(1)) / 12, 1)
    # "YYYY - present/current/now"
    if re.search(r"\b(present|current|now)\b", duration, re.IGNORECASE):
        m_start = re.findall(r"\b(20\d{2}|19\d{2})\b", duration)
        if m_start:
            years = datetime.date.today().year - int(m_start[0])
            return float(max(years, 1))
    # "YYYY - YYYY" or "YYYY–YYYY"
    m = re.findall(r"\b(20\d{2}|19\d{2})\b", duration)
    if len(m) >= 2:
        years = abs(int(m[-1]) - int(m[0]))
        return float(max(years, 1))
    # fallback: assume 1 year per entry
    return 1.0


@dataclass
class MatchResult:
    """Complete result of matching a candidate to a job."""

    # Basic info
    candidate_name: str = ""
    job_title: str = ""

    # Scores
    overall_score: float = 0.0
    score_level: MatchScoreLevel = MatchScoreLevel.POOR

    # Component scores
    skills_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    keyword_score: float = 0.0
    semantic_score: float = 0.0

    # Detailed matches
    skill_matches: list[SkillMatch] = field(default_factory=list)
    experience_match: Optional[ExperienceMatch] = None
    education_match: Optional[EducationMatch] = None
    keyword_match: Optional[KeywordMatch] = None
    semantic_match: Optional[SemanticMatch] = None

    # Bias detection
    bias_check: Optional[BiasCheckResult] = None

    # Breakdown
    score_breakdown: Optional[ScoreBreakdown] = None

    # Explanation
    explanation: Optional[Explanation] = None

    # Raw data references
    resume_result: Optional[ResumeParseResult] = None
    jd_result: Optional[JDParseResult] = None

    @property
    def matched_skills(self) -> list[str]:
        """Get list of matched skills."""
        return [s.skill_name for s in self.skill_matches if s.candidate_has_skill]

    @property
    def missing_skills(self) -> list[str]:
        """Get list of missing required skills."""
        return [
            s.skill_name for s in self.skill_matches
            if s.required and not s.candidate_has_skill
        ]


class MatchingEngine:
    """
    Engine for scoring candidates against job requirements.

    Uses a multi-factor approach:
    - Skills matching (required vs preferred)
    - Experience years matching
    - Education level matching
    - Keyword/terminology matching
    - Semantic similarity (using embeddings)
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        use_semantic: bool = True,
        use_bias_detection: bool = True,
        use_explainability: bool = True,
    ):
        """
        Initialize the matching engine.

        Args:
            weights: Optional custom scoring weights
            use_semantic: Whether to use semantic similarity matching
            use_bias_detection: Whether to perform bias detection
            use_explainability: Whether to generate detailed explanations
        """
        self.weights = weights or DEFAULT_SCORING_WEIGHTS
        self.use_semantic = use_semantic
        self.use_bias_detection = use_bias_detection
        self.use_explainability = use_explainability
        self._semantic_matcher = None
        self._bias_detector = None
        self._explainer = None
        self._skill_scorer: Optional["EmbeddingSkillScorerType"] = None
        self._use_embedding_skill_scorer: bool = True
        self._experience_scorer: Optional["DomainAwareExperienceScorerType"] = None
        self._use_domain_experience_scorer: bool = True
        self._education_scorer: Optional["EmbeddingEducationScorerType"] = None
        self._use_embedding_education_scorer: bool = True

    @property
    def semantic_matcher(self):
        """Get the semantic matcher (lazy initialization)."""
        if self._semantic_matcher is None and self.use_semantic:
            try:
                from src.ml.embeddings import get_semantic_matcher
                self._semantic_matcher = get_semantic_matcher()
                logger.info("Semantic matcher initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize semantic matcher: {e}")
                self.use_semantic = False
        return self._semantic_matcher

    @property
    def bias_detector(self):
        """Get the bias detector (lazy initialization)."""
        if self._bias_detector is None and self.use_bias_detection:
            try:
                from src.ml.ethics import get_bias_detector
                self._bias_detector = get_bias_detector()
                logger.info("Bias detector initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize bias detector: {e}")
                self.use_bias_detection = False
        return self._bias_detector

    @property
    def explainer(self):
        """Get the match explainer (lazy initialization)."""
        if self._explainer is None and self.use_explainability:
            try:
                from src.ml.explainability import get_match_explainer
                self._explainer = get_match_explainer()
                logger.info("Match explainer initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize explainer: {e}")
                self.use_explainability = False
        return self._explainer

    @property
    def skill_scorer(self) -> Optional["EmbeddingSkillScorerType"]:
        """Get the embedding skill scorer (lazy initialization)."""
        if self._skill_scorer is None and self._use_embedding_skill_scorer:
            try:
                from src.core.matching.skill_scorer import EmbeddingSkillScorer
                self._skill_scorer = EmbeddingSkillScorer()
                logger.info("EmbeddingSkillScorer initialized successfully")
            except Exception as exc:
                logger.warning(f"Could not initialize EmbeddingSkillScorer: {exc}")
                self._use_embedding_skill_scorer = False
        return self._skill_scorer

    @property
    def experience_scorer(self) -> Optional["DomainAwareExperienceScorerType"]:
        """Get the domain-aware experience scorer (lazy initialization)."""
        if self._experience_scorer is None and self._use_domain_experience_scorer:
            try:
                from src.core.matching.experience_scorer import DomainAwareExperienceScorer
                self._experience_scorer = DomainAwareExperienceScorer()
                logger.info("DomainAwareExperienceScorer initialized successfully")
            except Exception as exc:
                logger.warning(f"Could not initialize DomainAwareExperienceScorer: {exc}")
                self._use_domain_experience_scorer = False
        return self._experience_scorer

    @property
    def education_scorer(self) -> Optional["EmbeddingEducationScorerType"]:
        """Get the embedding education scorer (lazy initialization)."""
        if self._education_scorer is None and self._use_embedding_education_scorer:
            try:
                from src.core.matching.education_scorer import EmbeddingEducationScorer
                self._education_scorer = EmbeddingEducationScorer()
                logger.info("EmbeddingEducationScorer initialized successfully")
            except Exception as exc:
                logger.warning(f"Could not initialize EmbeddingEducationScorer: {exc}")
                self._use_embedding_education_scorer = False
        return self._education_scorer

    def match(
        self,
        resume_result: ResumeParseResult,
        jd_result: JDParseResult,
    ) -> MatchResult:
        """
        Match a candidate resume against a job description.

        Args:
            resume_result: Parsed resume data
            jd_result: Parsed job description data

        Returns:
            MatchResult with scores and explanations
        """
        result = MatchResult(
            resume_result=resume_result,
            jd_result=jd_result,
        )

        # Extract candidate name
        if resume_result.contact:
            name = resume_result.contact.get("full_name")
            if not name:
                first = resume_result.contact.get("first_name", "")
                last = resume_result.contact.get("last_name", "")
                name = f"{first} {last}".strip()
            result.candidate_name = name or "Unknown Candidate"
        else:
            result.candidate_name = "Unknown Candidate"

        result.job_title = jd_result.title or "Unknown Position"

        # Calculate individual scores
        result.skill_matches, result.skills_score = self._match_skills(
            resume_result, jd_result
        )

        result.experience_match, result.experience_score = self._match_experience(
            resume_result, jd_result
        )

        result.education_match, result.education_score = self._match_education(
            resume_result, jd_result
        )

        result.keyword_match, result.keyword_score = self._match_keywords(
            resume_result, jd_result
        )

        # Calculate semantic similarity if enabled
        result.semantic_match, result.semantic_score = self._match_semantic(
            resume_result, jd_result
        )

        # Calculate weighted overall score
        result.score_breakdown = self._calculate_breakdown(result)
        result.overall_score = result.score_breakdown.total_score
        result.score_level = MatchScoreLevel.from_score(result.overall_score)

        # Generate explanation
        result.explanation = self._generate_explanation(result)

        # Perform bias detection if enabled
        result.bias_check = self._check_bias(resume_result, result.overall_score)

        return result

    def match_from_parsed(
        self,
        parsed: "ParsedResumeType",
        job: "JobType",
    ) -> MatchResult:
        """
        Match a ParsedResume (AccurateResumeParser output) against a Job model.

        Primary entry point for the new accurate-parser pipeline. Converts
        ParsedResume + Job into thin shims for the existing _match_* sub-methods,
        then routes semantic similarity through compute_similarity_from_parsed()
        for real vector-based scores instead of the old ResumeParseResult path.

        Args:
            parsed: Output of AccurateResumeParser.
            job: The Job pydantic model from the database.

        Returns:
            MatchResult with all scores, breakdown, and explanation.
        """
        # -- Build minimal JDParseResult shim from Job model ------------------
        jd_shim: JDParseResult = JDParseResult(
            raw_text=(
                f"{job.title} {job.description or ''} "
                + " ".join(job.responsibilities)
            ),
            title=job.title,
            company_name=job.company_name,
            description=job.description or "",
            responsibilities=list(job.responsibilities),
            required_skills=[s.name for s in job.skill_requirements if s.is_required],
            preferred_skills=[s.name for s in job.skill_requirements if not s.is_required],
            experience_years_min=(
                job.experience_requirement.minimum_years
                if job.experience_requirement else None
            ),
            education_requirement=(
                job.education_requirement.minimum_degree
                if job.education_requirement else None
            ),
        )

        # -- Build minimal ResumeParseResult shim from ParsedResume -----------
        skills_list: list[dict[str, Any]] = [
            {"name": skill, "category": cat.category}
            for cat in parsed.skills
            for skill in cat.skills
        ]
        resume_shim: ResumeParseResult = ResumeParseResult()
        resume_shim.contact = {
            "full_name": parsed.contact.name,
            "email": parsed.contact.email,
        }
        resume_shim.skills = skills_list
        resume_shim.total_experience_years = sum(
            _estimate_years(e.duration) for e in parsed.experience
        )
        resume_shim.highest_education = (
            parsed.education[0].degree if parsed.education else None
        )

        # Populate preprocessed so _match_keywords can find resume text
        resume_shim.preprocessed = _MinimalPreprocessed(
            cleaned_text=parsed.raw_text,
        )

        # -- Run all existing match sub-methods --------------------------------
        result: MatchResult = MatchResult(
            resume_result=resume_shim,
            jd_result=jd_shim,
        )
        result.candidate_name = parsed.contact.name or "Unknown Candidate"
        result.job_title = job.title

        # Skills: use EmbeddingSkillScorer for alias/variant matching when available
        emb_skill_matches: list[SkillMatch] | None
        emb_skills_score: float
        emb_skill_matches, emb_skills_score = self._match_skills_from_parsed(parsed, job)
        if emb_skill_matches is not None:
            result.skill_matches = emb_skill_matches
            result.skills_score = emb_skills_score
        else:
            result.skill_matches, result.skills_score = self._match_skills(resume_shim, jd_shim)
        # Experience: use DomainAwareExperienceScorer for domain-weighted scoring when available
        emb_exp_match: ExperienceMatch | None
        emb_exp_score: float
        emb_exp_match, emb_exp_score = self._match_experience_from_parsed(parsed, job)
        if emb_exp_match is not None:
            result.experience_match = emb_exp_match
            result.experience_score = emb_exp_score
        else:
            result.experience_match, result.experience_score = self._match_experience(resume_shim, jd_shim)
        # Education: use EmbeddingEducationScorer for field-aware scoring when available
        emb_edu_match: EducationMatch | None
        emb_edu_score: float
        emb_edu_match, emb_edu_score = self._match_education_from_parsed(parsed, job)
        if emb_edu_match is not None:
            result.education_match = emb_edu_match
            result.education_score = emb_edu_score
        else:
            result.education_match, result.education_score = self._match_education(resume_shim, jd_shim)
        result.keyword_match, result.keyword_score = self._match_keywords(resume_shim, jd_shim)

        # -- Semantic step: new typed path if matcher available ----------------
        if self.use_semantic and self.semantic_matcher is not None:
            try:
                semantic_match = self.semantic_matcher.compute_similarity_from_parsed(
                    parsed, job
                )
                result.semantic_match = semantic_match
                # Use weighted combination of section scores when available;
                # fall back to overall_similarity for legacy SemanticMatch objects
                # (those leave weighted_similarity at its default of 0.0).
                _sem_score: float = (
                    semantic_match.weighted_similarity
                    if semantic_match.weighted_similarity > 0.0
                    else semantic_match.overall_similarity
                )
                result.semantic_score = round(_sem_score, 3)
            except Exception as exc:
                logger.warning(f"compute_similarity_from_parsed failed: {exc}")
                result.semantic_match = None
                result.semantic_score = 0.0
        else:
            result.semantic_match = None
            result.semantic_score = 0.0

        result.score_breakdown = self._calculate_breakdown(result)
        result.overall_score = result.score_breakdown.total_score
        result.score_level = MatchScoreLevel.from_score(result.overall_score)
        result.explanation = self._generate_explanation(result)
        result.bias_check = self._check_bias(resume_shim, result.overall_score)

        return result

    def _check_bias(
        self,
        resume: ResumeParseResult,
        score: float,
    ) -> Optional[BiasCheckResult]:
        """
        Check for potential bias in the candidate evaluation.

        Args:
            resume: Parsed resume data.
            score: The assigned match score.

        Returns:
            BiasCheckResult if bias detection is enabled, None otherwise.
        """
        if not self.use_bias_detection or self.bias_detector is None:
            return None

        try:
            return self.bias_detector.check_match_for_bias(resume, score)
        except Exception as e:
            logger.warning(f"Bias detection failed: {e}")
            return None

    def _match_skills(
        self,
        resume: ResumeParseResult,
        jd: JDParseResult,
    ) -> tuple[list[SkillMatch], float]:
        """Match candidate skills against job requirements."""
        skill_matches = []

        # Get candidate skills (lowercase for comparison)
        candidate_skills = {
            s.get("name", "").lower(): s for s in resume.skills
            if s.get("name")
        }

        # Get all required skills
        required_skills = set(s.lower() for s in jd.required_skills)
        preferred_skills = set(s.lower() for s in jd.preferred_skills)

        # Score required skills
        required_matched = 0
        for skill in required_skills:
            has_skill = skill in candidate_skills
            match = SkillMatch(
                skill_name=skill,
                required=True,
                candidate_has_skill=has_skill,
                match_score=1.0 if has_skill else 0.0,
            )

            # Check for partial match (related skills)
            if not has_skill:
                related = self._find_related_skill(skill, candidate_skills.keys())
                if related:
                    match.partial_match = True
                    match.related_skill = related
                    match.match_score = 0.5

            if has_skill:
                required_matched += 1
            elif match.partial_match:
                required_matched += 0.5

            skill_matches.append(match)

        # Score preferred skills
        preferred_matched = 0
        for skill in preferred_skills:
            has_skill = skill in candidate_skills
            match = SkillMatch(
                skill_name=skill,
                required=False,
                candidate_has_skill=has_skill,
                match_score=1.0 if has_skill else 0.0,
            )

            if has_skill:
                preferred_matched += 1

            skill_matches.append(match)

        # Calculate overall skills score
        score = 0.0
        total_weight = 0.0

        if required_skills:
            # Required skills are worth more
            required_weight = 0.7
            score += required_weight * (required_matched / len(required_skills))
            total_weight += required_weight

        if preferred_skills:
            # Preferred skills contribute less
            preferred_weight = 0.3
            score += preferred_weight * (preferred_matched / len(preferred_skills))
            total_weight += preferred_weight

        if total_weight > 0:
            score = score / total_weight
        elif candidate_skills:
            # If no requirements specified but candidate has skills
            score = 0.5

        return skill_matches, round(score, 3)

    def _find_related_skill(
        self, target: str, candidate_skills: set[str]
    ) -> Optional[str]:
        """Find a related skill in candidate's skill set."""
        # Simple related skill mappings
        related_groups = [
            {"python", "django", "flask", "fastapi"},
            {"javascript", "typescript", "node.js", "react", "angular", "vue"},
            {"java", "spring", "hibernate", "kotlin"},
            {"c#", ".net", "asp.net"},
            {"sql", "mysql", "postgresql", "oracle"},
            {"aws", "gcp", "azure", "cloud"},
            {"docker", "kubernetes", "containerization"},
            {"machine learning", "deep learning", "tensorflow", "pytorch"},
        ]

        target_lower = target.lower()
        for group in related_groups:
            if target_lower in group:
                for skill in candidate_skills:
                    if skill.lower() in group and skill.lower() != target_lower:
                        return skill
        return None

    def _match_skills_from_parsed(
        self,
        parsed: "ParsedResumeType",
        job: "JobType",
    ) -> tuple[list[SkillMatch] | None, float]:
        """
        Score skills using EmbeddingSkillScorer when available.

        Works directly on ParsedResume + Job model — no shim required.
        Returns (None, 0.0) when scorer is unavailable (caller falls back to
        _match_skills(shim)).
        """
        if not self._use_embedding_skill_scorer or self.skill_scorer is None:
            return None, 0.0

        required_skills: list[str] = [
            s.name for s in job.skill_requirements if s.is_required
        ]
        preferred_skills: list[str] = [
            s.name for s in job.skill_requirements if not s.is_required
        ]
        candidate_skills: list[str] = [
            skill for cat in parsed.skills for skill in cat.skills
        ]

        scorer: EmbeddingSkillScorerType = self.skill_scorer
        return scorer.score_skills(required_skills, preferred_skills, candidate_skills)

    def _match_experience_from_parsed(
        self,
        parsed: "ParsedResumeType",
        job: "JobType",
    ) -> tuple[ExperienceMatch | None, float]:
        """
        Score experience using DomainAwareExperienceScorer when available.

        Works directly on ParsedResume + Job model — no shim required.
        Returns (None, 0.0) when scorer is unavailable (caller falls back to
        _match_experience(shim)).
        """
        if not self._use_domain_experience_scorer or self.experience_scorer is None:
            return None, 0.0

        required_years: float = (
            job.experience_requirement.minimum_years
            if job.experience_requirement
            else 0.0
        )
        responsibilities: list[str] = list(job.responsibilities)

        scorer: DomainAwareExperienceScorerType = self.experience_scorer
        return scorer.score_experience(
            entries=parsed.experience,
            required_years=required_years,
            job_title=job.title,
            responsibilities=responsibilities,
        )

    def _match_education_from_parsed(
        self,
        parsed: "ParsedResumeType",
        job: "JobType",
    ) -> tuple[EducationMatch | None, float]:
        """
        Score education using EmbeddingEducationScorer when available.

        Works directly on ParsedResume + Job model — no shim required.
        Returns (None, 0.0) when scorer is unavailable (caller falls back to
        _match_education(shim)).
        """
        if not self._use_embedding_education_scorer or self.education_scorer is None:
            return None, 0.0

        required_degree: str = (
            job.education_requirement.minimum_degree
            if job.education_requirement
            else ""
        )
        job_description: str = job.description or ""

        scorer: "EmbeddingEducationScorerType" = self.education_scorer
        return scorer.score_education(
            education_entries=parsed.education,
            required_degree=required_degree,
            job_title=job.title,
            job_description=job_description,
        )

    def _match_experience(
        self,
        resume: ResumeParseResult,
        jd: JDParseResult,
    ) -> tuple[Optional[ExperienceMatch], float]:
        """Match candidate experience against job requirements."""
        candidate_years = resume.total_experience_years
        required_years = jd.experience_years_min or 0

        if required_years == 0:
            # No experience requirement
            return None, 1.0 if candidate_years > 0 else 0.5

        years_diff = candidate_years - required_years

        match = ExperienceMatch(
            required_years=required_years,
            candidate_years=candidate_years,
            years_difference=years_diff,
            meets_minimum=candidate_years >= required_years,
            score=0.0,
        )

        # Calculate score
        if candidate_years >= required_years:
            # Meets or exceeds requirement
            match.score = 1.0
        elif candidate_years >= required_years * 0.7:
            # Close to requirement (within 30%)
            match.score = 0.7 + 0.3 * (candidate_years / required_years)
        elif candidate_years > 0:
            # Has some experience
            match.score = 0.5 * (candidate_years / required_years)
        else:
            # No experience
            match.score = 0.0

        return match, round(match.score, 3)

    def _match_education(
        self,
        resume: ResumeParseResult,
        jd: JDParseResult,
    ) -> tuple[Optional[EducationMatch], float]:
        """Match candidate education against job requirements."""
        required_degree = jd.education_requirement
        candidate_degree = resume.highest_education

        if not required_degree:
            # No education requirement
            return None, 1.0 if candidate_degree else 0.7

        match = EducationMatch(
            required_degree=required_degree,
            candidate_degree=candidate_degree,
            meets_requirement=False,
            score=0.0,
        )

        if not candidate_degree:
            # No education info found
            match.score = 0.3
            return match, match.score

        # Get education levels
        required_level = EDUCATION_LEVELS.get(required_degree.lower(), 0)
        candidate_level = EDUCATION_LEVELS.get(candidate_degree.lower(), 0)

        if candidate_level >= required_level:
            match.meets_requirement = True
            match.score = 1.0
        elif candidate_level == required_level - 1:
            # One level below
            match.score = 0.7
        else:
            # More than one level below
            match.score = max(0.3, candidate_level / required_level) if required_level > 0 else 0.3

        return match, round(match.score, 3)

    def _match_keywords(
        self,
        resume: ResumeParseResult,
        jd: JDParseResult,
    ) -> tuple[Optional[KeywordMatch], float]:
        """
        Match keywords from JD in resume, weighted by term frequency.

        Keywords mentioned more often in the JD (across responsibilities and
        qualifications) carry proportionally more weight in the final score,
        so the score reflects how well the resume covers the *emphasized*
        requirements rather than treating every word equally.
        """
        resume_text = ""
        if resume.extraction_result:
            resume_text = resume.extraction_result.text.lower()
        elif resume.preprocessed:
            resume_text = resume.preprocessed.cleaned_text.lower()

        if not resume_text:
            return None, 0.0

        # Count term frequency across responsibilities and qualifications.
        # Words that appear multiple times are more heavily emphasized by the
        # employer and therefore deserve a higher weight in the score.
        freq: Counter[str] = Counter()
        for resp in jd.responsibilities:
            for word in resp.lower().split():
                if len(word) > 3 and word not in _KEYWORD_STOPWORDS:
                    freq[word] += 1
        for qual in jd.qualifications:
            for word in qual.lower().split():
                if len(word) > 3 and word not in _KEYWORD_STOPWORDS:
                    freq[word] += 1

        if not freq:
            return None, 0.5

        total_weight = sum(freq.values())

        # TF-weighted score: matched frequency / total frequency
        matched = [kw for kw in freq if kw in resume_text]
        missing = [kw for kw in freq if kw not in resume_text]

        matched_weight = sum(freq[kw] for kw in matched)
        tf_score = matched_weight / total_weight if total_weight > 0 else 0.0

        # Present matched/missing lists ordered by frequency (most important first)
        matched_by_freq = sorted(matched, key=lambda k: freq[k], reverse=True)
        missing_by_freq = sorted(missing, key=lambda k: freq[k], reverse=True)

        match = KeywordMatch(
            total_keywords=len(freq),
            matched_keywords=len(matched),
            match_percentage=tf_score,
            matched_terms=matched_by_freq[:20],
            missing_terms=missing_by_freq[:10],
        )

        return match, round(tf_score, 3)

    def _match_semantic(
        self,
        resume: ResumeParseResult,
        jd: JDParseResult,
    ) -> tuple[Optional[SemanticMatch], float]:
        """
        Compute semantic similarity between resume and job description.

        Uses embedding-based matching to capture semantic meaning
        beyond simple keyword matching.
        """
        if not self.use_semantic or self.semantic_matcher is None:
            # Fallback to keyword score if semantic matching not available
            return None, 0.0

        try:
            semantic_match = self.semantic_matcher.compute_similarity(resume, jd)
            return semantic_match, round(semantic_match.overall_similarity, 3)
        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}")
            return None, 0.0

    def _calculate_breakdown(self, result: MatchResult) -> ScoreBreakdown:
        """Calculate weighted score breakdown.

        When semantic matching is unavailable its weight is NOT redistributed —
        instead keyword_score is used as a proxy for the semantic slot so that
        the total still sums to 1.0 and the original weights are preserved.
        """
        if result.semantic_match is not None:
            # Semantic matching succeeded — use all weights as configured.
            skills_w: float = self.weights["skills_match"]
            experience_w: float = self.weights["experience_match"]
            education_w: float = self.weights["education_match"]
            semantic_w: float = self.weights["semantic_similarity"]
            keyword_w: float = self.weights["keyword_match"]
            semantic_score: float = result.semantic_score
        else:
            # Semantic unavailable — keep original weights, fall back to
            # keyword_score as the semantic proxy so the formula is unchanged.
            skills_w = self.weights["skills_match"]
            experience_w = self.weights["experience_match"]
            education_w = self.weights["education_match"]
            semantic_w = self.weights["semantic_similarity"]
            keyword_w = self.weights["keyword_match"]
            semantic_score = result.keyword_score   # fallback: keyword as proxy

        breakdown = ScoreBreakdown(
            skills_score=result.skills_score,
            skills_weight=skills_w,
            skills_weighted=result.skills_score * skills_w,

            experience_score=result.experience_score,
            experience_weight=experience_w,
            experience_weighted=result.experience_score * experience_w,

            education_score=result.education_score,
            education_weight=education_w,
            education_weighted=result.education_score * education_w,

            semantic_score=semantic_score,
            semantic_weight=semantic_w,
            semantic_weighted=semantic_score * semantic_w,

            keyword_score=result.keyword_score,
            keyword_weight=keyword_w,
            keyword_weighted=result.keyword_score * keyword_w,
        )

        return breakdown

    def _generate_explanation(self, result: MatchResult) -> Explanation:
        """Generate human-readable explanation of the match."""
        factors = []
        strengths = []
        gaps = []
        recommendations = []

        # Skills analysis
        matched_skills = result.matched_skills
        missing_skills = result.missing_skills

        if matched_skills:
            strengths.append(f"Matches {len(matched_skills)} required skills: {', '.join(matched_skills[:5])}")
            factors.append(ExplanationFactor(
                factor_name="Skills Match",
                factor_type="positive" if result.skills_score > 0.5 else "neutral",
                description=f"Candidate has {len(matched_skills)} of the required skills",
                impact_score=result.skills_score,
            ))

        if missing_skills:
            gaps.append(f"Missing skills: {', '.join(missing_skills[:5])}")
            recommendations.append(f"Consider candidates who also have: {', '.join(missing_skills[:3])}")

        # Experience analysis
        if result.experience_match:
            exp = result.experience_match
            if exp.meets_minimum:
                strengths.append(f"Has {exp.candidate_years:.1f} years experience (required: {exp.required_years:.1f})")
                factors.append(ExplanationFactor(
                    factor_name="Experience",
                    factor_type="positive",
                    description=f"Meets experience requirement with {exp.candidate_years:.1f} years",
                    impact_score=result.experience_score,
                ))
            else:
                gaps.append(f"Has {exp.candidate_years:.1f} years experience (required: {exp.required_years:.1f})")
                factors.append(ExplanationFactor(
                    factor_name="Experience",
                    factor_type="negative",
                    description=f"Below required experience ({exp.years_difference:.1f} years short)",
                    impact_score=result.experience_score,
                ))

        # Education analysis
        if result.education_match:
            edu = result.education_match
            if edu.meets_requirement:
                strengths.append(f"Has {edu.candidate_degree} degree (required: {edu.required_degree})")
            else:
                gaps.append(f"Has {edu.candidate_degree or 'no degree listed'} (required: {edu.required_degree})")

        # Semantic similarity analysis
        if result.semantic_match:
            sem = result.semantic_match
            if sem.overall_similarity >= 0.7:
                strengths.append(f"High semantic match ({sem.overall_similarity:.0%}) with job description")
                factors.append(ExplanationFactor(
                    factor_name="Semantic Similarity",
                    factor_type="positive",
                    description=f"Resume content strongly aligns with job requirements",
                    impact_score=sem.overall_similarity,
                ))
            elif sem.overall_similarity >= 0.5:
                factors.append(ExplanationFactor(
                    factor_name="Semantic Similarity",
                    factor_type="neutral",
                    description=f"Resume content moderately aligns with job requirements",
                    impact_score=sem.overall_similarity,
                ))
            else:
                gaps.append(f"Low semantic alignment ({sem.overall_similarity:.0%}) with job description")
                factors.append(ExplanationFactor(
                    factor_name="Semantic Similarity",
                    factor_type="negative",
                    description=f"Resume content may not fully align with job requirements",
                    impact_score=sem.overall_similarity,
                ))

        # Generate summary
        if result.overall_score >= 0.85:
            summary = f"{result.candidate_name} is an excellent match for {result.job_title}"
        elif result.overall_score >= 0.70:
            summary = f"{result.candidate_name} is a good match for {result.job_title}"
        elif result.overall_score >= 0.50:
            summary = f"{result.candidate_name} is a fair match for {result.job_title}"
        else:
            summary = f"{result.candidate_name} may not be the best fit for {result.job_title}"

        # Generate LIME and SHAP explanations if enabled
        lime_explanation = None
        shap_values = None

        if self.use_explainability and self.explainer:
            try:
                _skill_details: dict[str, Any] = {
                    "missing_skills": missing_skills,
                    "matched_skills": matched_skills,
                }
                _experience_details: Optional[dict[str, Any]] = (
                    {
                        "meets_minimum": result.experience_match.meets_minimum,
                        "candidate_years": result.experience_match.candidate_years,
                        "required_years": result.experience_match.required_years,
                    }
                    if result.experience_match else None
                )
                _education_details: Optional[dict[str, Any]] = (
                    {
                        "meets_requirement": result.education_match.meets_requirement,
                        "candidate_degree": result.education_match.candidate_degree,
                        "required_degree": result.education_match.required_degree,
                    }
                    if result.education_match else None
                )
                detailed_explanation = self.explainer.explain(
                    candidate_name=result.candidate_name,
                    job_title=result.job_title,
                    skills_score=result.skills_score,
                    experience_score=result.experience_score,
                    education_score=result.education_score,
                    semantic_score=result.semantic_score,
                    keyword_score=result.keyword_score,
                    overall_score=result.overall_score,
                    skill_details=_skill_details,
                    experience_details=_experience_details,
                    education_details=_education_details,
                )
                lime_explanation = detailed_explanation.get_lime_dict()
                shap_values = detailed_explanation.get_shap_dict()
                summary = detailed_explanation.summary
                strengths = detailed_explanation.key_strengths
                gaps = detailed_explanation.key_gaps
                recommendations = detailed_explanation.recommendations
            except Exception as e:
                logger.warning(f"Failed to generate detailed explanation: {e}")

        return Explanation(
            summary=summary,
            factors=factors,
            strengths=strengths,
            gaps=gaps,
            recommendations=recommendations,
            lime_explanation=lime_explanation,
            shap_values=shap_values,
        )

    def rank_candidates(
        self,
        candidates: list[MatchResult],
    ) -> list[MatchResult]:
        """
        Rank candidates by their overall match score.

        Tie-breaking order (all descending, then name ascending as last resort):
          1. overall_score  — primary criterion
          2. skills_score   — highest-weight component (0.35)
          3. experience_score — second-highest component (0.25)
          4. candidate_name — alphabetical, makes ordering fully deterministic

        Args:
            candidates: List of match results

        Returns:
            Sorted list with highest scores first
        """
        return sorted(
            candidates,
            key=lambda x: (
                -x.overall_score,
                -x.skills_score,
                -x.experience_score,
                x.candidate_name,
            ),
        )


# Singleton instance
_matching_engine: Optional[MatchingEngine] = None


def get_matching_engine() -> MatchingEngine:
    """Get the matching engine singleton instance."""
    global _matching_engine
    if _matching_engine is None:
        _matching_engine = MatchingEngine()
    return _matching_engine
