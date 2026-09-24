"""FlatIPIndex: the unit-norm guard, exact recall, and the save/load round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from mmu.embed.hashing import HashingEmbedder
from mmu.retrieve.dense import FlatIPIndex

TEXTS = [
    "the controller shall notify the supervisory authority within 72 hours",
    "the processor shall notify the controller without undue delay",
    "a recipe for slow-cooked beans with paprika",
]


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64)


def test_indexed_vector_is_its_own_top_hit(embedder: HashingEmbedder) -> None:
    vectors = embedder.encode(TEXTS, kind="document").dense
    index = FlatIPIndex(embedder.dim)
    index.add(vectors)

    for row, text in enumerate(TEXTS):
        query = embedder.encode([text], kind="query").dense[0]
        hits = index.search(query, k=3)
        assert hits[0][0] == row
        # Exact search over a unit vector against itself: cosine 1.0, no approximation.
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_non_normalized_vectors_are_refused(embedder: HashingEmbedder) -> None:
    """The guard that keeps IndexFlatIP meaning cosine rather than magnitude.

    An unnormalized vector raises nowhere downstream — FAISS happily ranks by inner
    product, so the longest chunks win every query and retrieval is mysteriously bad.
    """
    index = FlatIPIndex(embedder.dim)
    scaled = embedder.encode(TEXTS, kind="document").dense * 3.0
    with pytest.raises(ValueError, match="L2-normalized"):
        index.add(scaled)


def test_k_larger_than_corpus_returns_everything(embedder: HashingEmbedder) -> None:
    index = FlatIPIndex(embedder.dim)
    index.add(embedder.encode(TEXTS, kind="document").dense)
    query = embedder.encode(["notify"], kind="query").dense[0]
    assert len(index.search(query, k=99)) == len(TEXTS)
    # No -1 padding rows leak out.
    assert all(row >= 0 for row, _ in index.search(query, k=99))


def test_empty_index_returns_nothing(embedder: HashingEmbedder) -> None:
    index = FlatIPIndex(embedder.dim)
    assert index.size == 0
    assert index.search(np.zeros(embedder.dim, dtype=np.float32), k=5) == []


def test_wrong_dimension_is_rejected(embedder: HashingEmbedder) -> None:
    index = FlatIPIndex(embedder.dim)
    with pytest.raises(ValueError, match="expected"):
        index.add(np.zeros((2, embedder.dim + 1), dtype=np.float32))


def test_save_load_preserves_row_order_and_dim(tmp_path, embedder: HashingEmbedder) -> None:
    vectors = embedder.encode(TEXTS, kind="document").dense
    index = FlatIPIndex(embedder.dim)
    index.add(vectors)
    path = tmp_path / "faiss.bin"
    index.save(path)

    reloaded = FlatIPIndex.load(path)
    assert reloaded.dim == embedder.dim
    assert reloaded.size == len(TEXTS)
    # Row order is the mapping ChunkStore relies on; a reordered index would silently
    # attach every score to the wrong chunk.
    query = embedder.encode([TEXTS[1]], kind="query").dense[0]
    assert reloaded.search(query, k=1)[0][0] == 1
