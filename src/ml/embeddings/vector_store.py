"""
Vector store abstraction for storing and searching embeddings.

Supports ChromaDB and FAISS backends for efficient similarity search.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchResult:
    """Result from a vector similarity search."""

    id: str
    score: float
    metadata: dict[str, Any]
    document: Optional[str] = None


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Add embeddings to the store."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for similar embeddings."""
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete embeddings by ID."""
        pass

    @abstractmethod
    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get embeddings and metadata by ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Get total number of embeddings in store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all embeddings from store."""
        pass


class ChromaVectorStore(VectorStore):
    """
    ChromaDB-based vector store implementation.

    Provides persistent storage with metadata filtering support.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[Path] = None,
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            collection_name: Name of the collection to use.
            persist_directory: Directory for persistent storage.
        """
        settings = get_settings()
        self.collection_name = collection_name or settings.vector_store.collection_name
        self.persist_directory = persist_directory or settings.vector_store.persist_directory

        self._client = None
        self._collection = None
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy initialization of ChromaDB client."""
        if self._initialized:
            return

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            # Ensure persist directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)

            logger.info(f"Initializing ChromaDB at: {self.persist_directory}")

            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            self._initialized = True
            logger.info(
                f"ChromaDB initialized with collection: {self.collection_name} "
                f"({self._collection.count()} documents)"
            )

        except ImportError:
            logger.error(
                "chromadb not installed. Install with: pip install chromadb"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    @property
    def collection(self):
        """Get the ChromaDB collection."""
        if not self._initialized:
            self._initialize()
        return self._collection

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """
        Add embeddings to the collection.

        Args:
            ids: Unique identifiers for each embedding.
            embeddings: Embedding vectors to store.
            documents: Optional original text documents.
            metadatas: Optional metadata for each embedding.
        """
        if len(ids) == 0:
            return

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

        logger.debug(f"Added {len(ids)} embeddings to collection")

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """
        Add or update embeddings in the collection.

        Args:
            ids: Unique identifiers for each embedding.
            embeddings: Embedding vectors to store.
            documents: Optional original text documents.
            metadatas: Optional metadata for each embedding.
        """
        if len(ids) == 0:
            return

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

        logger.debug(f"Upserted {len(ids)} embeddings to collection")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter.

        Returns:
            List of SearchResult objects sorted by similarity.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distance, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                # For cosine distance: similarity = 1 - distance
                similarity = 1.0 - distance

                search_results.append(SearchResult(
                    id=doc_id,
                    score=similarity,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                    document=results["documents"][0][i] if results["documents"] else None,
                ))

        return search_results

    def delete(self, ids: list[str]) -> None:
        """Delete embeddings by ID."""
        if ids:
            self.collection.delete(ids=ids)
            logger.debug(f"Deleted {len(ids)} embeddings from collection")

    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get embeddings and metadata by ID."""
        if not ids:
            return []

        results = self.collection.get(
            ids=ids,
            include=["embeddings", "documents", "metadatas"],
        )

        items = []
        for i, doc_id in enumerate(results["ids"]):
            items.append({
                "id": doc_id,
                "embedding": results["embeddings"][i] if results["embeddings"] else None,
                "document": results["documents"][i] if results["documents"] else None,
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            })

        return items

    def count(self) -> int:
        """Get total number of embeddings in collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all embeddings from collection."""
        # Delete and recreate collection
        if self._client and self._collection:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Cleared collection: {self.collection_name}")


class FAISSVectorStore(VectorStore):
    """
    FAISS-based vector store implementation.

    Provides fast in-memory similarity search with optional persistence.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        persist_path: Optional[Path] = None,
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension size.
            persist_path: Path for saving/loading index.
        """
        settings = get_settings()
        self.dimension = dimension or settings.ml.embedding_dimension
        self.persist_path = persist_path or (
            settings.vector_store.persist_directory / "faiss_index"
        )

        self._index = None
        self._id_map: dict[int, str] = {}  # Internal ID -> External ID
        self._ext_to_internal: dict[str, int] = {}  # External ID -> Internal ID
        self._metadata: dict[str, dict[str, Any]] = {}  # External ID -> Metadata
        self._documents: dict[str, str] = {}  # External ID -> Document
        self._next_id = 0
        self._initialized = False

    def _initialize(self) -> None:
        """Lazy initialization of FAISS index."""
        if self._initialized:
            return

        try:
            import faiss

            logger.info(f"Initializing FAISS index with dimension: {self.dimension}")

            # Use IndexFlatIP for cosine similarity (with normalized vectors)
            self._index = faiss.IndexFlatIP(self.dimension)
            self._initialized = True

            # Try to load existing index
            if self.persist_path.exists():
                self._load()

            logger.info(f"FAISS index initialized ({self._index.ntotal} vectors)")

        except ImportError:
            logger.error(
                "faiss not installed. Install with: pip install faiss-cpu"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise

    @property
    def index(self):
        """Get the FAISS index."""
        if not self._initialized:
            self._initialize()
        return self._index

    def _should_compact(self) -> bool:
        """Return True when the stale-vector ratio in the FAISS index exceeds 20%.

        Stale vectors are rows that remain in the FAISS index after their IDs
        have been removed from the external-ID maps.  A ratio above 20 % means
        the index has grown meaningfully beyond its useful size and a rebuild
        is worth the cost.
        """
        if not self._initialized or self._index is None:
            return False
        total: int = self._index.ntotal
        if total == 0:
            return False
        live: int = len(self._id_map)
        stale_ratio: float = (total - live) / total
        return stale_ratio > 0.20

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Add embeddings to the index."""
        if len(ids) == 0:
            return

        # Validate shape/dimension before touching FAISS so callers get a
        # clear error instead of a cryptic FAISS assertion or silent mismatch.
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2D array, got shape {embeddings.shape}"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        if not self._initialized:
            self._initialize()

        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store mappings
        for i, ext_id in enumerate(ids):
            internal_id = self._next_id + i
            self._id_map[internal_id] = ext_id
            self._ext_to_internal[ext_id] = internal_id

            if metadatas and i < len(metadatas):
                self._metadata[ext_id] = metadatas[i]
            if documents and i < len(documents):
                self._documents[ext_id] = documents[i]

        self._next_id += len(ids)
        logger.debug(f"Added {len(ids)} embeddings to FAISS index")

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Add or update embeddings in the index.

        Compaction is deferred until the stale-vector ratio exceeds 20 %.
        Until then, searches already skip stale rows via the ID-map lookup,
        so correctness is maintained without an O(N) rebuild on every update.
        """
        if len(ids) == 0:
            return

        # Validate shape/dimension eagerly — before map mutations or FAISS calls.
        if embeddings.ndim != 2:
            raise ValueError(
                f"Embeddings must be a 2D array, got shape {embeddings.shape}"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        if not self._initialized:
            self._initialize()

        # Remove existing entries for IDs that already exist
        existing: list[str] = [eid for eid in ids if eid in self._ext_to_internal]
        if existing:
            self._remove_from_maps(existing)
            if self._should_compact():
                self._compact()

        self.add(ids, embeddings, documents, metadatas)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Search for similar embeddings."""
        if not self._initialized:
            self._initialize()

        if self.index.ntotal == 0:
            return []

        # Ensure query is float32 and 2D
        query = query_embedding.astype(np.float32).reshape(1, -1)

        # Search
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            ext_id = self._id_map.get(idx)
            if not ext_id:
                continue

            metadata = self._metadata.get(ext_id, {})

            # Apply metadata filter if specified
            if filter_metadata:
                if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append(SearchResult(
                id=ext_id,
                score=float(score),
                metadata=metadata,
                document=self._documents.get(ext_id),
            ))

        return results

    def _remove_from_maps(self, ids: list[str]) -> None:
        """Remove IDs from all mapping dicts (does not touch the FAISS index)."""
        for ext_id in ids:
            internal_id = self._ext_to_internal.pop(ext_id, None)
            if internal_id is not None:
                self._id_map.pop(internal_id, None)
            self._metadata.pop(ext_id, None)
            self._documents.pop(ext_id, None)

    def _compact(self) -> None:
        """
        Rebuild the FAISS index containing only live (non-deleted) vectors.

        Uses IndexFlatIP.reconstruct() to retrieve each live vector by its
        current internal row index, then builds a fresh index so that no
        stale vectors remain.
        """
        import faiss

        live_internal_ids = sorted(self._id_map.keys())

        if not live_internal_ids:
            self._index = faiss.IndexFlatIP(self.dimension)
            self._next_id = 0
            return

        # Reconstruct the live vectors from the current index
        live_vectors = np.array(
            [self._index.reconstruct(i) for i in live_internal_ids],
            dtype=np.float32,
        )

        # Build a fresh index and add the live vectors
        new_index = faiss.IndexFlatIP(self.dimension)
        new_index.add(live_vectors)

        # Remap internal IDs to new sequential positions
        new_id_map: dict[int, str] = {}
        new_ext_to_internal: dict[str, int] = {}
        for new_internal, old_internal in enumerate(live_internal_ids):
            ext_id = self._id_map[old_internal]
            new_id_map[new_internal] = ext_id
            new_ext_to_internal[ext_id] = new_internal

        self._index = new_index
        self._id_map = new_id_map
        self._ext_to_internal = new_ext_to_internal
        self._next_id = len(new_id_map)
        logger.debug(f"Compacted FAISS index: {len(new_id_map)} live vectors")

    def delete(self, ids: list[str]) -> None:
        """Delete embeddings by ID and rebuild index to remove stale vectors."""
        if not ids or not self._initialized:
            return

        to_remove = [eid for eid in ids if eid in self._ext_to_internal]
        if not to_remove:
            return

        self._remove_from_maps(to_remove)
        self._compact()
        logger.debug(f"Deleted {len(to_remove)} embeddings from FAISS index")

    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get metadata by ID."""
        items = []
        for ext_id in ids:
            if ext_id in self._metadata or ext_id in self._documents:
                items.append({
                    "id": ext_id,
                    "metadata": self._metadata.get(ext_id, {}),
                    "document": self._documents.get(ext_id),
                })
        return items

    def count(self) -> int:
        """Get total number of embeddings in index."""
        if not self._initialized:
            return 0
        return self.index.ntotal

    def clear(self) -> None:
        """Clear all embeddings from index."""
        if self._initialized:
            import faiss
            self._index = faiss.IndexFlatIP(self.dimension)
            self._id_map.clear()
            self._ext_to_internal.clear()
            self._metadata.clear()
            self._documents.clear()
            self._next_id = 0
            logger.info("Cleared FAISS index")

    def save(self) -> None:
        """Save index to disk."""
        if not self._initialized or self.index.ntotal == 0:
            return

        import faiss
        import json

        self.persist_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.persist_path / "index.faiss"))

        # Save metadata
        meta = {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "metadata": self._metadata,
            "documents": self._documents,
            "next_id": self._next_id,
        }
        with open(self.persist_path / "metadata.json", "w") as f:
            json.dump(meta, f)

        logger.info(f"Saved FAISS index to {self.persist_path}")

    def _load(self) -> None:
        """Load index from disk."""
        import faiss
        import json

        index_path = self.persist_path / "index.faiss"
        meta_path = self.persist_path / "metadata.json"

        if not index_path.exists():
            return

        self._index = faiss.read_index(str(index_path))

        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._id_map = {int(k): v for k, v in meta["id_map"].items()}
            self._metadata = meta["metadata"]
            self._documents = meta["documents"]
            self._next_id = meta["next_id"]
            # Rebuild reverse map from loaded id_map
            self._ext_to_internal = {v: int(k) for k, v in meta["id_map"].items()}

        logger.info(f"Loaded FAISS index from {self.persist_path}")


def get_vector_store(
    provider: Optional[str] = None,
    **kwargs,
) -> VectorStore:
    """
    Factory function to get a vector store instance.

    Args:
        provider: Vector store provider ('chromadb' or 'faiss').
                 Defaults to config setting.
        **kwargs: Additional arguments for the vector store.

    Returns:
        VectorStore instance.
    """
    settings = get_settings()
    provider = provider or settings.vector_store.provider

    if provider == "chromadb":
        return ChromaVectorStore(**kwargs)
    elif provider == "faiss":
        return FAISSVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store provider: {provider}")


# Default vector stores for different purposes
_resume_store: Optional[VectorStore] = None
_job_store: Optional[VectorStore] = None


def get_resume_store() -> VectorStore:
    """Get vector store for resume embeddings."""
    global _resume_store
    if _resume_store is None:
        settings = get_settings()
        provider: str = settings.vector_store.provider
        if provider == "faiss":
            _resume_store = FAISSVectorStore(
                persist_path=settings.vector_store.persist_directory / "resumes",
            )
        else:
            _resume_store = ChromaVectorStore(
                collection_name="resume_embeddings",
                persist_directory=settings.vector_store.persist_directory / "resumes",
            )
    return _resume_store


def get_job_store() -> VectorStore:
    """Get vector store for job description embeddings."""
    global _job_store
    if _job_store is None:
        settings = get_settings()
        provider: str = settings.vector_store.provider
        if provider == "faiss":
            _job_store = FAISSVectorStore(
                persist_path=settings.vector_store.persist_directory / "jobs",
            )
        else:
            _job_store = ChromaVectorStore(
                collection_name="job_embeddings",
                persist_directory=settings.vector_store.persist_directory / "jobs",
            )
    return _job_store
