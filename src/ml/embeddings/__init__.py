"""
Document embedding, vectorization, and similarity search.

This module provides semantic embedding capabilities for the AI-ATS system,
enabling intelligent matching between resumes and job descriptions using
vector similarity search.

Components:
- EmbeddingModel: Wrapper for sentence-transformers models
- VectorStore: Abstraction for vector storage (ChromaDB/FAISS)
- SemanticMatcher: High-level semantic similarity computation
"""

from .embedding_model import (
    EmbeddingModel,
    get_embedding_model,
)

from .vector_store import (
    VectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    SearchResult,
    get_vector_store,
    get_resume_store,
    get_job_store,
)

from .semantic_similarity import (
    SemanticMatcher,
    CandidateMatch,
    get_semantic_matcher,
)

__all__ = [
    # Embedding model
    "EmbeddingModel",
    "get_embedding_model",
    # Vector stores
    "VectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "SearchResult",
    "get_vector_store",
    "get_resume_store",
    "get_job_store",
    # Semantic matching
    "SemanticMatcher",
    "CandidateMatch",
    "get_semantic_matcher",
]
