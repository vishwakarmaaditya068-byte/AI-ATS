"""Candidate-job matching engine module."""

from .matching_engine import (
    MatchingEngine,
    MatchResult,
    get_matching_engine,
)
from .skill_scorer import (
    EmbeddingSkillScorer,
    get_embedding_skill_scorer,
)
from .experience_scorer import (
    DomainAwareExperienceScorer,
    get_domain_aware_experience_scorer,
)
from .education_scorer import (
    EmbeddingEducationScorer,
    get_embedding_education_scorer,
)

__all__ = [
    "MatchingEngine",
    "MatchResult",
    "get_matching_engine",
    "EmbeddingSkillScorer",
    "get_embedding_skill_scorer",
    "DomainAwareExperienceScorer",
    "get_domain_aware_experience_scorer",
    "EmbeddingEducationScorer",
    "get_embedding_education_scorer",
]
