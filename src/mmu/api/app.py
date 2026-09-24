"""Application factory and the per-subsystem lifespan.

``create_app(retriever=..., generator=...)`` injects ready subsystems — tests pass fakes
and never touch a model or the filesystem. Anything left as ``None`` is built by the
lifespan at startup. Routes never construct a subsystem; they resolve it from
``app.state`` through a dependency in :mod:`mmu.api.deps`, so the same handlers serve an
injected test app and a real one unchanged.

**Each subsystem is resolved independently, and a failure to build one is recorded
rather than raised.** A missing index must not prevent the process from starting,
because then ``GET /health`` — the one endpoint that could *explain* the missing index —
would not be up either. The same goes for the generator: ``ollama list`` is empty on a
fresh machine, and "no model pulled" should be a 503 with instructions on /answer, not a
server that refuses to boot.

A module-level ``app = create_app()`` is provided so ``uvicorn mmu.api.app:app`` works.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from mmu.api import deps
from mmu.config import Settings, get_settings

log = logging.getLogger(__name__)


def _build_retriever(settings: Settings):
    """Load the persisted index. Deferred import: faiss and torch are not free, and a
    test that injects a fake retriever must not pay for either."""
    from mmu.embed.bge_m3 import BgeM3Embedder
    from mmu.retrieve.build import load_index
    from mmu.retrieve.hybrid import HybridRetriever

    embedder = BgeM3Embedder(
        model_name=settings.embed_model,
        device=settings.embed_device,
        batch_size=settings.embed_batch_size,
    )
    store, dense, sparse = load_index(settings.index_dir, embedder)
    return HybridRetriever(store, dense, sparse, embedder)


def _build_generator(settings: Settings):
    from mmu.generate.registry import build_generator

    return build_generator(settings)


def create_app(*, retriever=None, generator=None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The semaphore must be created inside the running loop, and exactly once per
        # process — a Semaphore built per request bounds nothing at all.
        deps.init_slots(settings.search_concurrency)

        app.state.retriever = retriever
        app.state.generator = generator
        app.state.index_error = None
        app.state.generator_error = None
        app.state.aliases = []

        if app.state.retriever is None:
            try:
                app.state.retriever = _build_retriever(settings)
            except Exception as exc:  # noqa: BLE001 — surfaced on /health, not fatal
                app.state.index_error = str(exc)
                log.warning("no index loaded: %s", exc)

        if app.state.generator is None:
            try:
                app.state.generator = _build_generator(settings)
                from mmu.generate.registry import available_aliases

                app.state.aliases = available_aliases(settings)
            except Exception as exc:  # noqa: BLE001 — /answer answers 503 with this
                app.state.generator_error = str(exc)
                log.warning("no generator available: %s", exc)

        warning = settings.context_budget_warning()
        if warning:
            # Loud, because the failure it describes is silent: an over-long prompt is
            # left-truncated and every metric then measures truncation.
            log.warning("%s", warning)

        yield

    app = FastAPI(
        title="MMU",
        description="Modular Memory Unit — hybrid-retrieval RAG over legal/compliance documents",
        version="0.1.0",
        lifespan=lifespan,
    )

    from mmu.api import routes_admin, routes_answer, routes_search

    app.include_router(routes_admin.router)
    app.include_router(routes_search.router)
    app.include_router(routes_answer.router)
    return app


app = create_app()
