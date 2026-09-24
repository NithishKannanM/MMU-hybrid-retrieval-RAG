"""Grounded answer routes: one JSON, one SSE.

``POST /answer`` and ``POST /answer/stream`` are two routes rather than one route with a
``stream: bool`` flag. A flag would make a single endpoint return two incompatible
content types, forcing every client into a special case before it can even parse the
response; two routes keep the JSON path trivially consumable.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from mmu.api.deps import get_generator, get_retriever, retrieve
from mmu.api.schemas import AnswerRequest
from mmu.core.models import Answer
from mmu.generate.base import GeneratorUnavailable
from mmu.generate.prompt import build_messages, marker_compliance, resolve_citations
from mmu.retrieve.hybrid import HybridRetriever

router = APIRouter(tags=["answer"])


def _resolve_generator(request: Request, alias: str | None):
    """Resolve an alias to a generator, or fall back to the process default.

    An alias, never a raw model id — the parameter is client-controlled, and a request
    must not be able to name a cloud model and route around the configured local default.
    """
    generator = get_generator(request)
    if alias is None:
        return generator
    from mmu.config import get_settings
    from mmu.generate.registry import build_generator

    try:
        return build_generator(get_settings(), alias)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except GeneratorUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None


@router.post("/answer")
async def answer(
    body: AnswerRequest,
    request: Request,
    retriever: HybridRetriever = Depends(get_retriever),
) -> Answer:
    generator = _resolve_generator(request, body.generator)
    started = time.perf_counter()

    result = await retrieve(
        retriever, body.query, k=body.k, depth=body.depth, channels=body.channels,
        dense_weight=body.dense_weight, sparse_weight=body.sparse_weight, rrf_k=body.rrf_k,
    )

    t = time.perf_counter()
    try:
        completion = await generator.complete(
            build_messages(body.query, result.chunks),
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    except GeneratorUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None

    timings = result.timings
    timings.generate_ms = (time.perf_counter() - t) * 1000
    timings.total_ms = (time.perf_counter() - started) * 1000

    return Answer(
        query=body.query,
        text=completion.text,
        citations=resolve_citations(completion.text, result.chunks),
        chunks=result.chunks,
        model=completion.model,
        timings=timings,
        marker_compliance=marker_compliance(completion.text, len(result.chunks)),
    )


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(jsonable_encoder(data))}\n\n"


@router.post("/answer/stream")
async def answer_stream(
    body: AnswerRequest,
    request: Request,
    retriever: HybridRetriever = Depends(get_retriever),
):
    """Server-sent events: ``retrieval`` -> ``token``* -> ``usage`` -> ``done``, or ``error``.

    **What this endpoint is, and what it is not.** Streaming here is a *token-level UX*
    concern: a grounded answer over eight legal chunks takes 10-30s on a local 7B, and a
    client staring at a spinner cannot distinguish a slow answer from a hung process.
    Streaming lets the answer form on screen.

    It is explicitly **not** the fix for blocking retrieval. That fix lives in
    :mod:`mmu.api.deps` and has already run to completion before the first byte of this
    stream is written. Left un-offloaded, this endpoint would stream tokens beautifully
    for one client while every concurrent request sat behind a pinned event loop — and
    the failure would look like a server problem rather than a retrieval one. Two
    independent concerns, fixed in two different places.

    **The ``retrieval`` event fires first**, before generation begins, carrying the
    citations. First-token latency necessarily includes the whole retrieval phase, so
    emitting the sources immediately gives the client something real to render during
    the generation gap — and makes the retrieve/generate split visible on the wire.
    """
    generator = _resolve_generator(request, body.generator)
    started = time.perf_counter()

    result = await retrieve(
        retriever, body.query, k=body.k, depth=body.depth, channels=body.channels,
        dense_weight=body.dense_weight, sparse_weight=body.sparse_weight, rrf_k=body.rrf_k,
    )

    async def events() -> AsyncIterator[str]:
        yield _sse(
            "retrieval",
            {
                "chunks": result.chunks,
                "timings": result.timings,
                "model": generator.name,
            },
        )
        pieces: list[str] = []
        t = time.perf_counter()
        try:
            async for delta in generator.stream(
                build_messages(body.query, result.chunks),
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            ):
                pieces.append(delta.text)
                yield _sse("token", {"delta": delta.text})
        except GeneratorUnavailable as exc:
            # Weaker recovery than POST /answer, deliberately: once a token has been
            # sent, the partial answer is already on the user's screen. Silently
            # switching models would splice two systems' output into one answer, so a
            # mid-stream failure surfaces as an error event instead.
            yield _sse("error", {"detail": str(exc)})
            return

        text = "".join(pieces)
        timings = result.timings
        timings.generate_ms = (time.perf_counter() - t) * 1000
        timings.total_ms = (time.perf_counter() - started) * 1000
        yield _sse(
            "usage",
            {
                "model": generator.name,
                "citations": resolve_citations(text, result.chunks),
                "marker_compliance": marker_compliance(text, len(result.chunks)),
                "timings": timings,
            },
        )
        yield _sse("done", {})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "cache-control": "no-cache",
            "connection": "keep-alive",
            # Without this, a reverse proxy buffers the whole response and the client
            # sees nothing until generation finishes — i.e. not streaming at all.
            "x-accel-buffering": "no",
        },
    )
