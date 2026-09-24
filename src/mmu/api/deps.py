"""The thread-pool offload. This module is the entire async/sync boundary.

:meth:`mmu.retrieve.hybrid.HybridRetriever.search` is a plain ``def`` holding three
CPU-bound blocking calls — the query encode, the FAISS scan and the BM25 pass. Calling
it directly from an ``async def`` handler pins the event loop for its whole duration,
which stalls *every other request in the process*, including ``GET /health``. That is
what makes the failure so misleading: one slow query and the server reads as down
rather than as busy.

``asyncio.to_thread`` moves the work to the default executor. faiss and numpy both
release the GIL for their heavy sections, so the worker threads genuinely run in
parallel rather than round-robining on one core.

**Rejected alternative: declare the route handler as a plain ``def``** and let
Starlette's threadpool do it. That works for ``POST /search`` — but ``POST
/answer/stream`` must be a coroutine, because it returns an async generator. The split
would then be inconsistent, and the offload would silently not apply to the endpoint
that needs it most.

**The semaphore is not redundant with ``to_thread``.** ``to_thread`` keeps the loop
responsive; it does nothing to bound concurrency. Two hundred in-flight requests become
two hundred queued thread tasks all contending for the same cores, which converts a
latency problem into a thrashing problem — and the default executor's own queue is
unbounded, so the failure arrives as memory growth rather than as backpressure. The
bound is ``cpu_count`` by default (``MMU_SEARCH_CONCURRENCY``).

Note what is *not* here: generation. Both generator backends are ``httpx.AsyncClient``
calls, and wrapping async network I/O in a thread is the exact inversion of this fix —
it burns a worker thread to wait on a socket the event loop waits on for free.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from fastapi import HTTPException, Request

from mmu.core.models import SearchResult
from mmu.retrieve.hybrid import HybridRetriever

#: Created once per process in the lifespan, not per request: a Semaphore built per call
#: bounds nothing at all.
_SLOTS: asyncio.Semaphore | None = None


def init_slots(limit: int) -> None:
    global _SLOTS
    _SLOTS = asyncio.Semaphore(limit)


def _slots() -> asyncio.Semaphore:
    if _SLOTS is None:  # a test that built the app without the lifespan
        init_slots(4)
    assert _SLOTS is not None
    return _SLOTS


def get_retriever(request: Request) -> HybridRetriever:
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail="no index loaded. Run `mmu index build` and restart the server.",
        )
    return retriever


def get_generator(request: Request):
    generator = getattr(request.app.state, "generator", None)
    if generator is None:
        # 503 with a reason, never a 500 on a missing attribute: "generation is
        # unavailable" is a state the client can render, "server error" is not.
        raise HTTPException(
            status_code=503,
            detail=getattr(request.app.state, "generator_error", "generation is unavailable"),
        )
    return generator


async def retrieve(
    retriever: HybridRetriever,
    query: str,
    *,
    k: int,
    depth: int | None = None,
    channels: Sequence[str] = ("dense", "sparse"),
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60,
) -> SearchResult:
    """The one place blocking retrieval is awaited. See the module docstring."""
    async with _slots():
        return await asyncio.to_thread(
            retriever.search,
            query,
            k=k,
            depth=depth,
            channels=list(channels),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_k=rrf_k,
        )
