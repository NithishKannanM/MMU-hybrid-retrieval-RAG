"""Cross-encoder reranking as a post-stage over fused candidates.

**Why this is a separate stage and not a better retriever.** BGE-M3 is a *bi-encoder*:
it embeds the query and each chunk independently, so chunk vectors are computed once at
index time and a query costs one embedding plus a cheap similarity sweep. That is what
makes it an index. A cross-encoder takes the *pair* — query and chunk concatenated into
one input — and runs the full transformer over both together, so attention can relate
"notification deadline" in the question directly to "72 hours" in the text. Nothing can
be precomputed, because the chunk's representation now depends on which query is asking.
It is a predicate that cannot be indexed, and can only be evaluated by scanning.

That forces the architecture: the cheap index narrows 378 chunks (or a million) to a
pool of `depth` candidates, optimising for *recall*, and the expensive model reorders
only those, optimising for *precision*. The same shape as an index scan followed by a
filter that no index can serve.

**The ceiling is recall@depth, not recall@k.** A reranker can only promote what the
candidate pool already contains, so anything retrieval missed entirely is out of reach —
raising `depth` is the lever for that, and it costs a linear number of cross-encoder
forward passes. ``scripts/rerank_headroom.py`` measures both numbers against the real
index before any of this is worth running.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mmu.core.models import SearchResult

if TYPE_CHECKING:
    from mmu.core.models import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    """Scores (query, text) pairs. Higher is more relevant.

    Scores are compared only against each other within one call, never thresholded, so
    no backend has to promise a calibrated range: BGE-reranker emits unbounded logits,
    a hypothetical API reranker might emit 0..1, and both order identically.
    """

    @property
    def name(self) -> str: ...

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """BGE-reranker-v2-m3 (or any cross-encoder) via ``sentence_transformers``.

    Deliberately `sentence_transformers.CrossEncoder` and not `FlagEmbedding`, following
    the decision already recorded in :mod:`mmu.embed.base`: FlagEmbedding pulls a much
    heavier dependency tree for the same weights. The import is deferred to first use so
    that selecting no reranker — or constructing one in a test — never loads torch.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None

    @property
    def name(self) -> str:
        return self._model_id

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_id, device=self._device, max_length=self._max_length
            )
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        model = self._load()
        scores = model.predict(
            [(query, t) for t in texts],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


def rerank(result: SearchResult, reranker: Reranker, *, k: int) -> SearchResult:
    """Reorder ``result.chunks`` by cross-encoder score and keep the top ``k``.

    Returns a new :class:`SearchResult`; the input is not mutated.

    Three properties the caller depends on:

    - **``dense_rank`` and ``sparse_rank`` are preserved, not renumbered.** They record
      *which channel found this chunk and at what position*, which is the question the
      whole ablation exists to answer. Reranking changes where a chunk sits in the
      context; it does not change the history of how it got into the pool. Only
      ``rank``, the fused position, is renumbered to the new 1-based order.
    - **``rrf_score`` is left intact and the cross-encoder score goes in
      ``rerank_score``.** Overwriting the fused score would destroy the evidence for a
      promotion — with both present, a chunk that entered the pool at fused rank 19 and
      left at context rank 1 can be shown to have done so, which is exactly the claim a
      reranker makes.
    - **The sort is stable**, so chunks the reranker scores equally keep their fused
      order. A cross-encoder that cannot separate two candidates should not be allowed
      to permute them arbitrarily; falling back to the previous stage's opinion is the
      conservative choice, and it keeps this function deterministic.

    ``rerank_ms`` is recorded on the returned timings. It is a real cost paid per query
    and must be visible as its own line: attributing it to retrieval or to generation is
    how a latency comparison ends up measuring the wrong thing.
    """
    if not result.chunks:
        return result

    started = time.perf_counter()
    scores = reranker.score(result.query, [c.text for c in result.chunks])
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if len(scores) != len(result.chunks):
        raise ValueError(
            f"reranker {reranker.name!r} returned {len(scores)} score(s) for "
            f"{len(result.chunks)} chunk(s); scores must be positionally aligned"
        )

    scored: list[tuple[RetrievedChunk, float]] = list(zip(result.chunks, scores))
    # `sorted` is stable and `-score` never reorders equal scores, so a tie keeps the
    # fused ordering rather than whatever the sort happens to do.
    ordered = sorted(scored, key=lambda pair: -pair[1])[:k]

    chunks = [
        replace(chunk, rank=position, rerank_score=score)
        for position, (chunk, score) in enumerate(ordered, start=1)
    ]
    timings = replace(result.timings, rerank_ms=elapsed_ms)
    return SearchResult(query=result.query, k=len(chunks), chunks=chunks, timings=timings)
