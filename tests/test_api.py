"""The HTTP surface: shapes, the SSE event sequence, and the failure modes.

Everything here runs against injected fakes through ``create_app(...)``, so no model is
loaded and no socket is opened — but the app's real lifespan runs (see
``conftest.app_client``), so the dependency wiring under test is the production wiring.
"""

from __future__ import annotations

import json

import pytest

from mmu.api.app import create_app
from mmu.core.models import Chunk, RetrievedChunk, SearchResult, Timings
from mmu.generate.base import GeneratorUnavailable
from tests.conftest import FakeGenerator, app_client

CORPUS = [
    ("gdpr", "the controller shall notify the supervisory authority within 72 hours"),
    ("company-policy", "escalate every security incident to the data protection officer"),
    ("national-law", "the competent supervisory authority is the federal commissioner"),
]


class StubRetriever:
    """Returns a fixed fused result with believable rank provenance."""

    def __init__(self) -> None:
        self.last_kwargs: dict = {}
        self.store = None

    def search(self, query: str, **kwargs) -> SearchResult:
        self.last_kwargs = kwargs
        chunks = []
        for i, (doc_id, text) in enumerate(CORPUS):
            chunk = Chunk(
                chunk_id=f"{doc_id}:0", doc_id=doc_id, filename=f"{doc_id}.txt",
                chunk_index=0, text=text, char_start=0, char_end=len(text), page_from=1,
            )
            chunks.append(
                RetrievedChunk(
                    chunk=chunk, rank=i + 1, rrf_score=1.0 / (60 + i + 1),
                    dense_rank=i + 1,
                    # The second chunk is a sparse-only find: the hybrid union effect,
                    # visible on the wire.
                    sparse_rank=1 if i == 1 else None,
                )
            )
        return SearchResult(query=query, k=len(chunks), chunks=chunks,
                            timings=Timings(embed_ms=1, dense_ms=1, total_ms=3))


@pytest.fixture
def app():
    return create_app(retriever=StubRetriever(), generator=FakeGenerator())


def _events(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE stream into (event, data) pairs."""
    out = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        name = payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: "):])
        out.append((name, payload))
    return out


async def test_search_returns_rank_provenance(app) -> None:
    async with app_client(app) as client:
        response = await client.post("/search", json={"query": "breach", "k": 3})
    assert response.status_code == 200
    body = response.json()
    assert [c["chunk"]["doc_id"] for c in body["chunks"]] == [d for d, _ in CORPUS]
    # A chunk found only by BM25 is the whole argument for two channels, made visible.
    sparse_only = [c for c in body["chunks"] if c["dense_rank"] and c["sparse_rank"]]
    assert sparse_only
    assert body["timings"]["total_ms"] > 0


async def test_channels_ablation_reaches_the_retriever(app) -> None:
    async with app_client(app) as client:
        await client.post("/search", json={"query": "q", "k": 3, "channels": ["dense"]})
    assert app.state.retriever.last_kwargs["channels"] == ["dense"]


async def test_search_validates_its_input(app) -> None:
    async with app_client(app) as client:
        assert (await client.post("/search", json={"query": "", "k": 3})).status_code == 422
        assert (await client.post("/search", json={"query": "q", "k": 0})).status_code == 422
        bad = await client.post("/search", json={"query": "q", "channels": ["colbert"]})
        assert bad.status_code == 422


async def test_answer_returns_citations_and_compliance(app) -> None:
    async with app_client(app) as client:
        response = await client.post("/answer", json={"query": "when must we notify?"})
    assert response.status_code == 200
    body = response.json()
    assert body["text"]
    assert body["marker_compliance"] == 1.0
    assert body["citations"][0]["doc_id"] == "gdpr"
    assert body["timings"]["generate_ms"] >= 0


async def test_sse_event_sequence(app) -> None:
    """`retrieval` -> `token`* -> `usage` -> `done`.

    The retrieval event must come FIRST, before any token: first-token latency
    necessarily includes the whole retrieval phase, and emitting the sources up front is
    what gives a client something real to render during the generation gap.
    """
    async with app_client(app) as client:
        response = await client.post("/answer/stream", json={"query": "when?"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        # Without this a reverse proxy buffers the stream and nothing is streamed at all.
        assert response.headers["x-accel-buffering"] == "no"
        events = _events(response.text)

    names = [name for name, _ in events]
    assert names[0] == "retrieval"
    assert names[-1] == "done"
    assert names.count("usage") == 1
    assert names.index("usage") == len(names) - 2
    assert "token" in names
    assert names.index("retrieval") < names.index("token")

    retrieval_payload = events[0][1]
    assert len(retrieval_payload["chunks"]) == len(CORPUS)

    # The tokens reassemble into the full answer.
    streamed = "".join(p["delta"] for n, p in events if n == "token")
    assert streamed == FakeGenerator().reply

    usage = events[names.index("usage")][1]
    assert usage["marker_compliance"] == 1.0
    assert usage["citations"]


async def test_mid_stream_failure_emits_error_and_never_switches_model() -> None:
    """Once tokens are on screen, a silent fallback would splice two models' output."""
    app = create_app(retriever=StubRetriever(), generator=FakeGenerator(fail_after=2))
    async with app_client(app) as client:
        response = await client.post("/answer/stream", json={"query": "when?"})
        events = _events(response.text)

    names = [name for name, _ in events]
    assert names[0] == "retrieval"
    assert "token" in names          # a partial answer did reach the client
    assert names[-1] == "error"      # and it ended as an error, not a silent recovery
    assert "done" not in names
    assert "usage" not in names


async def test_generator_unavailable_is_503() -> None:
    app = create_app(retriever=StubRetriever(), generator=None)
    async with app_client(app) as client:
        response = await client.post("/answer", json={"query": "q"})
    assert response.status_code == 503


async def test_health_and_index_stats_without_an_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # retriever=None means "not injected — build one from disk", not "there is no
    # retriever" (see mmu.api.app's lifespan). So this test must control what there
    # is to build from: point MMU_DATA_DIR at an empty tmp_path before create_app()
    # so _build_retriever fails for a real, deterministic reason (no .index under
    # it), instead of silently succeeding against a real data/.index if one has
    # been built on disk.
    #
    # get_settings() is @lru_cache'd, and mmu.api.app calls it once already at
    # import time (the module-level `app = create_app()`), so the cached Settings
    # from that first import would otherwise survive untouched and silently ignore
    # the env var set below. Clear the cache before AND after so the tmp-path
    # Settings doesn't leak into later tests via the process-global cache.
    from mmu.config import get_settings

    monkeypatch.setenv("MMU_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    app = create_app(retriever=None, generator=FakeGenerator())
    get_settings.cache_clear()
    async with app_client(app) as client:
        health = (await client.get("/health")).json()
        stats = (await client.get("/index/stats")).json()
    assert health["ready"] is False
    assert health["index"]["chunks"] == 0
    assert health["index"]["error"]          # says WHY, rather than 500-ing
    assert health["generator"]["available"] is True
    assert stats["loaded"] is False and stats["documents"] == []
