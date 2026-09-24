import os
os.environ.setdefault("APP_ENVIRONMENT", "testing")
os.environ.setdefault("DB_NAME", "ai_ats_test")

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ml.embeddings.vector_store import FAISSVectorStore


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_store(dimension: int = 4, ntotal: int = 0, live_count: int = 0) -> FAISSVectorStore:
    """Build a FAISSVectorStore with a mock FAISS index, bypassing real FAISS."""
    store = FAISSVectorStore.__new__(FAISSVectorStore)
    store.dimension = dimension
    store.persist_path = Path("/nonexistent")
    store._initialized = True
    store._next_id = live_count

    mock_index = MagicMock()
    mock_index.ntotal = ntotal
    store._index = mock_index

    # Populate id maps with `live_count` entries
    store._id_map = {i: f"id{i}" for i in range(live_count)}
    store._ext_to_internal = {f"id{i}": i for i in range(live_count)}
    store._metadata = {}
    store._documents = {}
    return store


# ── _should_compact ────────────────────────────────────────────────────────────

def test_should_compact_false_when_stale_ratio_low() -> None:
    """_should_compact() must return False when stale ratio is at or below 20%."""
    store = _make_store(ntotal=10, live_count=9)  # 1/10 = 10% stale

    assert store._should_compact() is False


def test_should_compact_false_at_exactly_threshold() -> None:
    """_should_compact() returns False when stale ratio is exactly 20% (not strictly above)."""
    store = _make_store(ntotal=10, live_count=8)  # 2/10 = 20% stale → NOT > 0.20

    assert store._should_compact() is False


def test_should_compact_true_when_stale_ratio_high() -> None:
    """_should_compact() must return True when stale ratio exceeds 20%."""
    store = _make_store(ntotal=10, live_count=7)  # 3/10 = 30% stale

    assert store._should_compact() is True


def test_should_compact_false_when_index_empty() -> None:
    """_should_compact() returns False when the FAISS index is empty."""
    store = _make_store(ntotal=0, live_count=0)

    assert store._should_compact() is False


# ── upsert lazy compaction ─────────────────────────────────────────────────────

def test_upsert_skips_compact_when_stale_ratio_below_threshold() -> None:
    """upsert() must NOT call _compact() when stale ratio stays at or below 20%."""
    # 5 total vectors, 5 live — removing 1 existing → 1/5 = 20% stale → no compact
    store = _make_store(ntotal=5, live_count=5)
    store._compact = MagicMock()

    embeddings = np.ones((1, 4), dtype=np.float32)
    store.upsert(["id0"], embeddings)

    store._compact.assert_not_called()


def test_upsert_triggers_compact_when_stale_ratio_exceeds_threshold() -> None:
    """upsert() must call _compact() when removing an ID pushes stale ratio above 20%."""
    # ntotal=10, live=8 → stale=(10-8)/10=20%. After removing "id7" → live=7, stale=30% → compact.
    store = _make_store(ntotal=10, live_count=8)
    store._compact = MagicMock()
    store.add = MagicMock()

    embeddings = np.ones((1, 4), dtype=np.float32)
    store.upsert(["id7"], embeddings)  # "id7" is one of the 8 live IDs

    store._compact.assert_called_once()


# ── dimension validation ───────────────────────────────────────────────────────

def test_add_raises_on_dimension_mismatch() -> None:
    """add() must raise ValueError before touching FAISS when dimension is wrong."""
    store = _make_store(dimension=4)
    store._initialized = False  # Ensure _initialize() is never called

    wrong_dim = np.ones((2, 8), dtype=np.float32)  # dim=8, expected 4

    with pytest.raises(ValueError, match="dimension"):
        store.add(["id1", "id2"], wrong_dim)


def test_add_raises_on_wrong_ndim() -> None:
    """add() must raise ValueError when embeddings is not a 2D array."""
    store = _make_store(dimension=4)
    store._initialized = False

    flat = np.ones((8,), dtype=np.float32)  # 1D, not 2D

    with pytest.raises(ValueError, match="2D"):
        store.add(["id1"], flat)


def test_upsert_raises_on_dimension_mismatch() -> None:
    """upsert() must raise ValueError before any FAISS operation when dimension is wrong."""
    store = _make_store(dimension=4)
    store._compact = MagicMock()
    store.add = MagicMock()

    wrong_dim = np.ones((1, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="dimension"):
        store.upsert(["id1"], wrong_dim)

    store._compact.assert_not_called()
    store.add.assert_not_called()
