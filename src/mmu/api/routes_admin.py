"""Health and index introspection.

``GET /index/stats`` exists for a specific workflow reason: it serves the **doc_id
table**. Those slugs are what ``data/questions.jsonl`` refers to, and a question naming
a doc_id that is not in the index scores 0.0 on every per-document metric — which is
indistinguishable from a model failure unless you can see the real list.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["admin"])


@router.get("/health")
async def health(request: Request) -> dict:
    """Readiness, with the reason when something is missing.

    Reports capability, never a secret: a configured cloud generator appears as
    ``is_local: false`` while the key itself stays server-side.
    """
    state = request.app.state
    retriever = getattr(state, "retriever", None)
    generator = getattr(state, "generator", None)
    store = retriever.store if retriever is not None else None
    return {
        "ready": retriever is not None,
        "index": {
            "loaded": store is not None,
            "chunks": len(store) if store is not None else 0,
            "documents": len(store.doc_ids()) if store is not None else 0,
            "built_at": store.manifest.built_at
            if store is not None and store.manifest is not None
            else None,
            "embed_model": store.manifest.embed_model
            if store is not None and store.manifest is not None
            else None,
            "error": getattr(state, "index_error", None),
        },
        "generator": {
            "available": generator is not None,
            "model": generator.name if generator is not None else None,
            "error": getattr(state, "generator_error", None),
        },
        "aliases": getattr(state, "aliases", []),
    }


@router.get("/index/stats")
async def index_stats(request: Request) -> dict:
    """Per-document chunk counts — the doc_id table for questions.jsonl."""
    retriever = getattr(request.app.state, "retriever", None)
    if retriever is None:
        return {"loaded": False, "documents": [], "chunks": 0}
    store = retriever.store
    counts = store.counts_by_doc()
    by_doc = {c.doc_id: c.filename for c in store.chunks}
    return {
        "loaded": True,
        "chunks": len(store),
        "documents": [
            {"doc_id": doc_id, "filename": by_doc[doc_id], "chunks": n}
            for doc_id, n in sorted(counts.items())
        ],
        "manifest": store.manifest.__dict__ if store.manifest is not None else None,
    }
