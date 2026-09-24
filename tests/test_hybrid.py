"""The hybrid retriever: the ablation switch, the union effect, and rank provenance."""

from __future__ import annotations

import inspect

import pytest

from mmu.core.models import Chunk
from mmu.embed.hashing import HashingEmbedder
from mmu.retrieve.dense import FlatIPIndex
from mmu.retrieve.hybrid import HybridRetriever
from mmu.retrieve.sparse import Bm25Index
from mmu.retrieve.store import ChunkStore

TEXTS = [
    "the controller shall notify the competent supervisory authority without undue delay",
    "the data controller determines the purposes and means of the processing of personal data",
    "the data processor acts only on documented instructions from the controller",
    "escalate every security incident to the data protection officer within four hours",
    "a slow-cooked bean stew benefits from smoked paprika and patience",
]


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"doc{i % 2}:{i}",
            doc_id=f"doc{i % 2}",
            filename=f"doc{i % 2}.txt",
            chunk_index=i,
            text=text,
            char_start=0,
            char_end=len(text),
        )
        for i, text in enumerate(TEXTS)
    ]


@pytest.fixture
def retriever() -> HybridRetriever:
    embedder = HashingEmbedder(dim=64)
    chunks = _chunks()
    store = ChunkStore(chunks)
    dense = FlatIPIndex(embedder.dim)
    dense.add(embedder.encode([c.text for c in chunks], kind="document").dense)
    return HybridRetriever(store, dense, Bm25Index(store.texts), embedder)


def test_search_is_not_a_coroutine_function() -> None:
    """Pins the sync/async split that the whole thread-pool offload depends on.

    If this ever becomes `async def`, `api/deps.py`'s `asyncio.to_thread` silently stops
    being an offload and the event loop starts stalling on every retrieval again — with
    no error anywhere, just a server that reads as "down" whenever one query is slow.
    """
    assert not inspect.iscoroutinefunction(HybridRetriever.search)


def test_rank_provenance_records_which_channel_found_each_chunk(
    retriever: HybridRetriever,
) -> None:
    result = retriever.search("data controller purposes", k=5)
    assert result.chunks
    for rc in result.chunks:
        assert rc.dense_rank is not None or rc.sparse_rank is not None
    # At least one chunk should be found by both channels on this query.
    assert any(rc.dense_rank and rc.sparse_rank for rc in result.chunks)


def test_channel_ablation_returns_only_that_channels_results(
    retriever: HybridRetriever,
) -> None:
    dense_only = retriever.search("data controller", k=5, channels=["dense"])
    assert all(rc.sparse_rank is None for rc in dense_only.chunks)
    assert dense_only.timings.sparse_ms == 0.0

    sparse_only = retriever.search("data controller", k=5, channels=["sparse"])
    assert all(rc.dense_rank is None for rc in sparse_only.chunks)
    # The dense channel was not merely down-weighted; it was never run.
    assert sparse_only.timings.embed_ms == 0.0
    assert sparse_only.timings.dense_ms == 0.0


def test_a_chunk_found_by_only_one_channel_still_surfaces(
    retriever: HybridRetriever,
) -> None:
    """The union effect — the reason to fuse rather than intersect."""
    fused = retriever.search("data controller purposes", k=5)
    ids = {rc.chunk_id for rc in fused.chunks}
    single_channel = {
        rc.chunk_id for rc in fused.chunks if (rc.dense_rank is None) != (rc.sparse_rank is None)
    }
    assert single_channel, "expected at least one chunk from exactly one channel"
    assert single_channel <= ids


def test_fused_order_matches_hand_computed_rrf(retriever: HybridRetriever) -> None:
    result = retriever.search("data controller", k=5, rrf_k=60)
    expected = []
    for rc in result.chunks:
        score = 0.0
        if rc.dense_rank is not None:
            score += 1.0 / (60 + rc.dense_rank)
        if rc.sparse_rank is not None:
            score += 1.0 / (60 + rc.sparse_rank)
        expected.append(score)
        assert rc.rrf_score == pytest.approx(score)
    assert expected == sorted(expected, reverse=True)
    assert [rc.rank for rc in result.chunks] == list(range(1, len(result.chunks) + 1))


def test_weights_shift_the_fused_order(retriever: HybridRetriever) -> None:
    balanced = retriever.search("controller processing", k=5)
    sparse_heavy = retriever.search("controller processing", k=5, sparse_weight=20.0)
    assert [c.chunk_id for c in balanced.chunks] != [c.chunk_id for c in sparse_heavy.chunks]


def test_over_fetch_depth_exceeds_k(retriever: HybridRetriever) -> None:
    """Each channel is asked for `depth`, not `k` — fusing two already-final lists is
    nearly a no-op, and the union effect is where hybrid earns its keep."""
    result = retriever.search("controller", k=2, depth=5)
    assert len(result.chunks) == 2
    # Ranks beyond k were considered: a chunk ranked 3rd+ by a channel can still appear.
    deep = retriever.search("controller", k=5, depth=5)
    assert max((rc.dense_rank or 0) for rc in deep.chunks) > 2


def test_unknown_or_empty_channels_raise(retriever: HybridRetriever) -> None:
    with pytest.raises(ValueError, match="at least one"):
        retriever.search("x", channels=[])
    with pytest.raises(ValueError, match="unknown channel"):
        retriever.search("x", channels=["colbert"])


def test_timings_are_populated(retriever: HybridRetriever) -> None:
    t = retriever.search("controller", k=3).timings
    assert t.embed_ms > 0 and t.total_ms > 0
    assert t.retrieval_ms == pytest.approx(
        t.embed_ms + t.dense_ms + t.sparse_ms + t.fuse_ms
    )
