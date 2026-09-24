"""The thread-pool offload, proven behaviourally rather than by inspection.

This is the test for the most consequential decision in the serving layer. The bug it
guards against has no error message: if retrieval runs on the event loop the server
keeps answering — it just serializes every concurrent request behind whichever one is
retrieving, and the symptom is "the server is down" rather than "one query is slow".
Asserting that `asyncio.to_thread` appears in the source would not catch a regression
that moves the call; asserting that N concurrent requests finish in well under N x the
single-request time does.
"""

from __future__ import annotations

import asyncio
import inspect
import time

import pytest

from mmu.api.app import create_app
from mmu.core.models import Chunk, RetrievedChunk, SearchResult, Timings
from mmu.retrieve.hybrid import HybridRetriever
from tests.conftest import FakeGenerator, app_client

BLOCKING_SECONDS = 0.2
CONCURRENT_REQUESTS = 8


class BlockingRetriever:
    """A retriever whose `search` blocks the calling thread, like the real one."""

    def __init__(self) -> None:
        self.calls = 0
        self.store = None

    def search(self, query: str, **kwargs) -> SearchResult:
        self.calls += 1
        # time.sleep, not asyncio.sleep: this must block a REAL thread, exactly as a
        # torch forward pass and a FAISS scan do. asyncio.sleep would yield to the loop
        # and the test would pass even with the offload removed.
        time.sleep(BLOCKING_SECONDS)
        chunk = Chunk(
            chunk_id="d:0", doc_id="d", filename="d.txt", chunk_index=0,
            text="text", char_start=0, char_end=4,
        )
        return SearchResult(
            query=query, k=1,
            chunks=[RetrievedChunk(chunk=chunk, rank=1, rrf_score=0.5, dense_rank=1)],
            timings=Timings(total_ms=BLOCKING_SECONDS * 1000),
        )


@pytest.fixture
def app():
    return create_app(retriever=BlockingRetriever(), generator=FakeGenerator())


async def test_concurrent_searches_do_not_serialize(app) -> None:
    """N blocking retrievals must not take N x the single-request time."""
    async with app_client(app) as client:
        started = time.perf_counter()
        responses = await asyncio.gather(
            *[
                client.post("/search", json={"query": f"q{i}", "k": 1})
                for i in range(CONCURRENT_REQUESTS)
            ]
        )
        elapsed = time.perf_counter() - started

    assert all(r.status_code == 200 for r in responses), [r.status_code for r in responses]
    serial = CONCURRENT_REQUESTS * BLOCKING_SECONDS
    # Generous margin: the point is the difference between ~0.2-0.6s and ~1.6s, not a
    # precise speedup on a machine whose core count this test does not control.
    assert elapsed < serial * 0.75, (
        f"{CONCURRENT_REQUESTS} concurrent searches took {elapsed:.2f}s against a "
        f"serial bound of {serial:.2f}s — retrieval is running on the event loop"
    )


async def test_health_stays_responsive_during_a_blocking_search(app) -> None:
    """The property the offload actually buys: one slow query must not make the process
    look dead to everything else, including the health check."""
    async with app_client(app) as client:
        search = asyncio.create_task(client.post("/search", json={"query": "q", "k": 1}))
        await asyncio.sleep(BLOCKING_SECONDS / 4)  # let the search get into its sleep

        started = time.perf_counter()
        health = await client.get("/health")
        health_latency = time.perf_counter() - started

        assert health.status_code == 200
        assert health_latency < BLOCKING_SECONDS / 2, (
            f"/health took {health_latency:.3f}s while a search was in flight — "
            "the event loop is pinned"
        )
        await search


def test_retriever_search_is_not_a_coroutine_function() -> None:
    """Structural guard. If someone makes `search` async to 'modernize' it, the
    to_thread call silently stops being an offload and the block returns."""
    assert not inspect.iscoroutinefunction(HybridRetriever.search)


async def test_search_without_an_index_is_503_not_500(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A missing index must not be fatal: /health is the endpoint that explains it."""
    # retriever=None means "not injected — build one from disk", not "there is no
    # retriever". So this test must control what there is to build from: point
    # MMU_DATA_DIR at an empty tmp_path before create_app() so _build_retriever fails
    # for a real, deterministic reason (no .index under it). Without this, the test
    # only fails the way it's meant to by accident, on a machine that happens to have
    # no real data/.index built yet — and it silently starts loading BGE-M3 on any
    # machine that does have one, which this fake-injecting test should never do.
    #
    # get_settings() is @lru_cache'd, and `mmu.api.app` calls it once already at
    # import time (the module-level `app = create_app()`), so the cached Settings
    # from that first import — built from whatever MMU_DATA_DIR was at process
    # start — would otherwise survive untouched and silently ignore the env var set
    # below. Clear the cache so create_app()'s own `get_settings()` call re-reads
    # the environment.
    from mmu.config import get_settings

    monkeypatch.setenv("MMU_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    app = create_app(retriever=None, generator=FakeGenerator())
    get_settings.cache_clear()
    async with app_client(app) as client:
        response = await client.post("/search", json={"query": "q", "k": 1})
        assert response.status_code == 503
        assert "mmu index build" in response.json()["detail"]

        health = await client.get("/health")
        assert health.status_code == 200 and health.json()["ready"] is False
