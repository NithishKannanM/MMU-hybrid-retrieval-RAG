"""Retrieval-only route. No generation, so no LLM is required to exercise the stack."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from mmu.api.deps import get_retriever, retrieve
from mmu.api.schemas import SearchRequest
from mmu.core.models import SearchResult
from mmu.retrieve.hybrid import HybridRetriever

router = APIRouter(tags=["search"])


@router.post("/search")
async def search(
    body: SearchRequest, retriever: HybridRetriever = Depends(get_retriever)
) -> SearchResult:
    """Fused retrieval, with per-channel rank provenance on every chunk.

    ``dense_rank``/``sparse_rank`` are the point of this endpoint beyond debugging: a
    chunk with ``dense_rank: null`` and a strong ``sparse_rank`` is BM25 catching an
    exact defined term the embedder blurred — which is the whole argument for running
    two channels, made visible on one request.
    """
    return await retrieve(
        retriever,
        body.query,
        k=body.k,
        depth=body.depth,
        channels=body.channels,
        dense_weight=body.dense_weight,
        sparse_weight=body.sparse_weight,
        rrf_k=body.rrf_k,
    )
