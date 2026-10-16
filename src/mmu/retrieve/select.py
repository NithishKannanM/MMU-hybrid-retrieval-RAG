"""Context selection: choosing *which* k of the reranked candidates reach the prompt.

**Why this is a stage of its own.** A cross-encoder scores each ``(query, chunk)`` pair
independently, so its ranking expresses relevance per chunk and nothing about the
selected *set*. Measured consequence on this corpus (``reports/rerank/``, 30 questions):
reranking lifted recall@8 from 0.469 to 0.632 and MRR from 0.641 to 0.748, while
coverage fell from 0.929 to 0.583 and XDR from 0.452 to 0.106 — and **11 of the 14
cross-document questions landed at exactly ``Cov = 0.50``**, one of two required
documents present. With three slots and no notion of a set, the reranker filled all
three from whichever document was most on-topic.

A cross-document question needs one passage from each required document. That is a
property of the chosen set, so no amount of better pointwise scoring can produce it, and
the fix belongs after reranking rather than inside it:

    fuse `depth` candidates  ->  rerank (pointwise)  ->  select k (set-aware)  ->  prompt

Two selectors, both operating on the reranked order and neither touching the candidate
pool or the generator:

- :class:`MMRSelector` — maximal marginal relevance. Penalises a candidate by how
  similar it is to what has already been chosen, so near-duplicates lose to the next
  distinct passage. The general tool: it knows nothing about documents.
- :class:`DocQuotaSelector` — caps how many chunks any one document may contribute.
  The targeted tool: it addresses exactly the measured failure and nothing else.

Both are in because they fail differently, and the ablation is what decides. MMR can
diversify *within* a document and still return three chunks from it; the quota cannot
express that two passages from one document are near-identical.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mmu.core.models import SearchResult

if TYPE_CHECKING:
    import numpy as np

    from mmu.core.models import RetrievedChunk
    from mmu.retrieve.base import DenseIndex
    from mmu.retrieve.store import ChunkStore

#: Similarity between two candidates, in [0, 1]. Injected rather than hard-coded so the
#: redundancy notion is swappable and so tests need no index and no model.
Similarity = Callable[["RetrievedChunk", "RetrievedChunk"], float]


@runtime_checkable
class Selector(Protocol):
    """Chooses which ``k`` candidates reach the prompt, from a ranked list."""

    @property
    def name(self) -> str: ...

    def select(
        self, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]: ...


def _renumber(chosen: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Renumber ``rank`` to the final 1-based context order.

    Only ``rank`` changes. ``dense_rank``, ``sparse_rank``, ``rrf_score`` and
    ``rerank_score`` are the record of how each chunk got here and what every earlier
    stage thought of it, which is what makes a selection auditable after the fact.
    """
    return [replace(c, rank=i) for i, c in enumerate(chosen, start=1)]


class TopKSelector:
    """Keep the first ``k``. The identity selector, and the control arm.

    Named and registered rather than expressed as ``selector=None`` so that "take the
    reranker's order as-is" is a choice an arm states explicitly and the report prints,
    instead of an absence that has to be inferred.
    """

    @property
    def name(self) -> str:
        return "top-k"

    def select(
        self, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]:
        return _renumber(list(candidates[:k]))


def _normalized_relevance(candidates: Sequence[RetrievedChunk]) -> list[float]:
    """Each candidate's relevance, min-max scaled into [0, 1] across the pool.

    Scaling is not cosmetic. MMR mixes a relevance term with a similarity term under
    one ``lambda_``, and cross-encoder scores are unbounded logits (observed roughly
    -11..+9 on this corpus) while cosine similarity lives in [0, 1]. Mixed raw, the
    relevance term dominates for any ``lambda_`` worth using and MMR silently degrades
    to top-k — which would have looked like "diversity selection does not help" rather
    than a units bug.

    Uses ``rerank_score`` where a reranker ran and ``rrf_score`` otherwise, so the
    selector works with or without the cross-encoder. A pool whose scores are all equal
    maps to all-zero relevance, leaving MMR driven purely by the redundancy term; that
    is the right degenerate behaviour, since the ranker has expressed no preference.
    """
    scores = [
        c.rerank_score if c.rerank_score is not None else c.rrf_score
        for c in candidates
    ]
    if not scores:
        return []
    low, high = min(scores), max(scores)
    span = high - low
    if span <= 0:
        return [0.0] * len(scores)
    return [(s - low) / span for s in scores]


class MMRSelector:
    """Maximal marginal relevance over the reranked pool.

    At each step picks the candidate maximising
    ``lambda_ * rel(c) - (1 - lambda_) * max(sim(c, s) for s in selected)``,
    with ``rel`` min-max scaled (see :func:`_normalized_relevance`) so the two terms
    share a scale. The first pick has an empty ``selected`` and so is always the
    top-ranked candidate, which means MMR can only reorder what comes *after* the best
    chunk — it cannot damage MRR@1.

    ``lambda_ = 1.0`` is exactly top-k; ``0.0`` ignores relevance entirely and chases
    novelty. The default 0.5 is the midpoint and is **not tuned** — tuning it against
    the same 30 questions used to evaluate it would be fitting the test set.
    """

    def __init__(self, *, similarity: Similarity, lambda_: float = 0.5) -> None:
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_}")
        self._similarity = similarity
        self._lambda = lambda_

    @property
    def name(self) -> str:
        return f"mmr(lambda={self._lambda:g})"

    def select(
        self, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]:
        pool = list(candidates)
        if k <= 0 or not pool:
            return []
        relevance = _normalized_relevance(pool)
        remaining = list(range(len(pool)))
        chosen: list[int] = []

        while remaining and len(chosen) < k:
            best_index, best_score = remaining[0], -math.inf
            for index in remaining:
                redundancy = max(
                    (self._similarity(pool[index], pool[s]) for s in chosen),
                    default=0.0,
                )
                score = self._lambda * relevance[index] - (1 - self._lambda) * redundancy
                # Strict `>` keeps the earlier (better-ranked) candidate on a tie, so
                # the selector never reorders candidates it cannot distinguish.
                if score > best_score:
                    best_index, best_score = index, score
            chosen.append(best_index)
            remaining.remove(best_index)

        return _renumber([pool[i] for i in chosen])


class DocQuotaSelector:
    """Cap how many chunks any single document may contribute to the context.

    Walks the reranked order and admits a candidate while its document is under quota.
    If fewer than ``k`` chunks pass, it makes a second pass admitting the best remaining
    candidates regardless of quota — **the context is always filled to ``k`` when the
    pool allows**. A quota that shrank the prompt would confound this experiment with a
    prompt-length change, and prompt length is the thing the latency result turns on.

    ``max_per_doc=None`` derives the cap as ``ceil(k / documents present in the pool)``:
    with k=3 over a 3-document corpus that is 1 per document, which is the configuration
    the measured ``Cov = 0.50`` failures call for. Derived from the pool rather than from
    the question's required documents on purpose — the selector must not see the labels
    it is about to be scored against.
    """

    def __init__(self, *, max_per_doc: int | None = None) -> None:
        if max_per_doc is not None and max_per_doc < 1:
            raise ValueError(f"max_per_doc must be >= 1, got {max_per_doc}")
        self._max_per_doc = max_per_doc

    @property
    def name(self) -> str:
        cap = "auto" if self._max_per_doc is None else str(self._max_per_doc)
        return f"doc-quota(max={cap})"

    def select(
        self, candidates: Sequence[RetrievedChunk], k: int
    ) -> list[RetrievedChunk]:
        pool = list(candidates)
        if k <= 0 or not pool:
            return []

        if self._max_per_doc is not None:
            cap = self._max_per_doc
        else:
            n_docs = len({c.doc_id for c in pool})
            cap = max(1, math.ceil(k / n_docs))

        taken: dict[str, int] = {}
        chosen: list[RetrievedChunk] = []
        overflow: list[RetrievedChunk] = []
        for candidate in pool:
            if len(chosen) >= k:
                break
            if taken.get(candidate.doc_id, 0) < cap:
                chosen.append(candidate)
                taken[candidate.doc_id] = taken.get(candidate.doc_id, 0) + 1
            else:
                overflow.append(candidate)

        # Backfill in reranked order, so a corpus that simply has no second document to
        # offer degrades to top-k instead of returning a short context.
        for candidate in overflow:
            if len(chosen) >= k:
                break
            chosen.append(candidate)

        return _renumber(chosen)


def embedding_similarity(store: ChunkStore, dense: DenseIndex) -> Similarity:
    """Cosine between two chunks' stored dense vectors.

    Free: the vectors were computed at index time and :meth:`add` guarantees they are
    L2-normalized, so a dot product *is* the cosine and no model is loaded. Results are
    memoized per chunk because MMR asks for the same candidate's vector once per
    iteration.

    This is the semantic notion of redundancy, and on this corpus it also catches the
    mechanical kind: the splitter duplicates 64 tokens of lead-in between adjacent
    chunks, so neighbours are partly the same text by construction.
    """
    cache: dict[str, np.ndarray] = {}

    def vector(chunk: RetrievedChunk) -> np.ndarray:
        if chunk.chunk_id not in cache:
            row = store.row_of(chunk.chunk_id)
            cache[chunk.chunk_id] = dense.vectors_for([row])[0]
        return cache[chunk.chunk_id]

    def similarity(a: RetrievedChunk, b: RetrievedChunk) -> float:
        if a.chunk_id == b.chunk_id:
            return 1.0
        # Clamped: float error can put a dot product a hair outside [-1, 1], and a
        # negative cosine would turn the redundancy penalty into a reward.
        return max(0.0, min(1.0, float(vector(a) @ vector(b))))

    return similarity


_TOKEN = re.compile(r"[a-z0-9]+")


def lexical_similarity() -> Similarity:
    """Jaccard overlap of lowercased word tokens — no index, no model, no vectors.

    Kept as a usable alternative rather than only a test double: it needs nothing but
    the chunk text, which makes it the only redundancy measure available to a caller
    holding chunks without the index that produced them.
    """

    def tokens(chunk: RetrievedChunk) -> frozenset[str]:
        return frozenset(_TOKEN.findall(chunk.text.lower()))

    def similarity(a: RetrievedChunk, b: RetrievedChunk) -> float:
        ta, tb = tokens(a), tokens(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    return similarity


def apply(result: SearchResult, selector: Selector, *, k: int) -> SearchResult:
    """Run ``selector`` over ``result``'s candidates and keep ``k``, timing the stage.

    Returns a new :class:`SearchResult`; the input is not mutated. Reports its own
    ``select_ms`` for the same reason reranking does: a stage whose cost is folded into
    a neighbour cannot be argued about.
    """
    if not result.chunks:
        return result
    started = time.perf_counter()
    chosen = selector.select(result.chunks, k)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    timings = replace(result.timings, select_ms=elapsed_ms)
    return SearchResult(
        query=result.query, k=len(chosen), chunks=chosen, timings=timings
    )
