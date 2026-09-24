"""
Embedding model wrapper for generating text embeddings.

Uses sentence-transformers library for generating high-quality
semantic embeddings from text content.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding models.

    Provides a unified interface for generating text embeddings
    with support for batching and device selection.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_maxsize: int = 512,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to config setting.
            device: Device to run model on ('cpu', 'cuda', 'mps').
                   Defaults to config setting.
            cache_maxsize: Maximum number of single-text embeddings to keep
                          in the LRU cache. Set to 0 to disable caching.
        """
        settings = get_settings()
        self.model_name = model_name or settings.ml.embedding_model
        self.device = device or settings.ml.device
        self.dimension = settings.ml.embedding_dimension
        self.batch_size = settings.ml.batch_size
        self.cache_maxsize = cache_maxsize

        self._model = None
        self._initialized = False
        # LRU cache for single-string encode() calls.
        # Key: (text, normalize)  Value: embedding ndarray
        # OrderedDict gives O(1) move-to-end for LRU bookkeeping.
        self._cache: OrderedDict[tuple[str, bool], np.ndarray] = OrderedDict()

    def _load_model(self) -> None:
        """Lazy load the embedding model."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            self._initialized = True
            logger.info(f"Embedding model loaded on device: {self.device}")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    @property
    def model(self):
        """Get the underlying sentence-transformer model."""
        if not self._initialized:
            self._load_model()
        return self._model

    def encode(
        self,
        texts: str | list[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Single-string calls are served from an LRU cache so that identical
        texts (e.g. the same JD encoded once per candidate) never hit the
        model more than once per process lifetime.  List inputs bypass the
        cache because batch calls are typically unique and large.

        Args:
            texts: Single text string or list of texts to encode.
            normalize: Whether to L2-normalize embeddings (for cosine similarity).
            show_progress: Whether to show progress bar for large batches.

        Returns:
            numpy array of shape (n_texts, embedding_dim) or (embedding_dim,)
            for single text input.
        """
        if not self._initialized:
            self._load_model()

        single_input = isinstance(texts, str)

        if single_input and self.cache_maxsize > 0:
            cache_key = (texts, normalize)
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

        texts_list = [texts] if single_input else texts

        embeddings = self.model.encode(
            texts_list,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )

        if single_input:
            result = embeddings[0]
            if self.cache_maxsize > 0:
                self._cache[cache_key] = result
                self._cache.move_to_end(cache_key)
                if len(self._cache) > self.cache_maxsize:
                    self._cache.popitem(last=False)
            return result

        return embeddings

    def cache_clear(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    @property
    def cache_info(self) -> dict:
        """Return current cache occupancy."""
        return {"size": len(self._cache), "maxsize": self.cache_maxsize}

    def encode_resume(self, resume_text: str) -> np.ndarray:
        """
        Generate embedding for resume content.

        Args:
            resume_text: Full resume text content.

        Returns:
            Normalized embedding vector.
        """
        return self.encode(resume_text, normalize=True)

    def encode_job_description(self, jd_text: str) -> np.ndarray:
        """
        Generate embedding for job description.

        Args:
            jd_text: Full job description text.

        Returns:
            Normalized embedding vector.
        """
        return self.encode(jd_text, normalize=True)

    def encode_skills(self, skills: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of skills.

        Args:
            skills: List of skill names/descriptions.

        Returns:
            Array of embedding vectors, one per skill.
        """
        if not skills:
            return np.array([])
        return self.encode(skills, normalize=True)

    def similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between 0 and 1.
        """
        # For normalized vectors, cosine similarity = dot product
        sim = np.dot(embedding1, embedding2)
        # Clip to valid range (numerical precision issues)
        return float(np.clip(sim, 0.0, 1.0))

    def batch_similarity(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate similarity between a query and multiple corpus embeddings.

        Args:
            query_embedding: Single query embedding vector.
            corpus_embeddings: Array of corpus embedding vectors.

        Returns:
            Array of similarity scores.
        """
        if len(corpus_embeddings) == 0:
            return np.array([])

        similarities = np.dot(corpus_embeddings, query_embedding)
        return np.clip(similarities, 0.0, 1.0)


# Singleton instance
_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model() -> EmbeddingModel:
    """Get the embedding model singleton instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model
