"""The hybrid retriever: encode, scan both channels, fuse by rank.

**``search`` is a plain ``def``, and that is a load-bearing decision.** It contains the
three CPU-bound blocking operations in the whole request path:

1. **query encoding** — a torch forward pass, tens of milliseconds on CPU. The original
   design note for this system listed FAISS and BM25 as the blocking calls and omitted
   this one, which is backwards: the encode is the *largest* of the three. A flat scan
   over a few hundred 1024-dim vectors is microseconds; a transformer forward is not.
2. **FAISS ``IndexFlatIP.search``** — a BLAS-backed exact scan.
3. **``BM25Okapi.get_scores``** — a numpy pass over the whole corpus.

Keeping all three inside one synchronous function means the thread-pool offload in
:mod:`mmu.api.deps` covers all three *by construction* rather than by anyone remembering
to wrap each one. If this ever becomes ``async def``, the offload silently stops being
an offload and the event loop starts stalling again — so ``test_offload.py`` asserts
that this method is not a coroutine function.

**Channels are a first-class parameter, not a weight set to zero.** ``channels=["dense"]``
is the ablation that justifies the architecture: if hybrid does not beat dense-only on
exact-defined-term questions, the sparse tokenizer is wrong and that is the finding.
Reproducing an ablation by zeroing a weight would still pay for the channel's latency
and still let it influence the over-fetch, which makes the comparison dishonest.

**Over-fetch before fusing.** Each channel is asked for ``depth`` results, not ``k``.
Fusing two lists that have already been cut to the final size is nearly a no-op — the
union effect, where a chunk ranked 12th by dense and 2nd by BM25 reaches the top, is
precisely where hybrid retrieval earns its keep, and a depth of ``k`` throws it away.
"""

from __future__ import annotations

import time
from typing import Sequence

from mmu.core.fusion import RRF_K, rrf_scores
from mmu.core.models import RetrievedChunk, SearchResult, Timings
from mmu.embed.base import Embedder
from mmu.retrieve.base import DenseIndex
from mmu.retrieve.sparse import Bm25Index
from mmu.retrieve.store import ChunkStore

Channel = str  # "dense" | "sparse"
DEFAULT_CHANNELS: tuple[Channel, ...] = ("dense", "sparse")


class HybridRetriever:
    def __init__(
        self,
        store: ChunkStore,
        dense: DenseIndex,
        sparse: Bm25Index,
        embedder: Embedder,
    ) -> None:
        self._store = store
        self._dense = dense
        self._sparse = sparse
        self._embedder = embedder

    @property
    def store(self) -> ChunkStore:
        return self._store

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        depth: int | None = None,
        channels: Sequence[Channel] = DEFAULT_CHANNELS,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        rrf_k: int = RRF_K,
    ) -> SearchResult:
        """Retrieve ``k`` chunks. Blocking and CPU-bound — see the module docstring."""
        if not channels:
            raise ValueError("at least one of 'dense' or 'sparse' must be enabled")
        unknown = set(channels) - {"dense", "sparse"}
        if unknown:
            raise ValueError(f"unknown channel(s): {sorted(unknown)}")

        depth = depth if depth is not None else max(4 * k, 20)
        timings = Timings()
        started = time.perf_counter()

        dense_rows: list[int] = []
        sparse_rows: list[int] = []

        if "dense" in channels:
            t = time.perf_counter()
            # kind="query" — the other side of the pair the indexer encoded with
            # kind="document". See mmu.embed.prefix for why this is mandatory.
            vector = self._embedder.encode([query], kind="query").dense[0]
            timings.embed_ms = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            dense_rows = [row for row, _ in self._dense.search(vector, depth)]
            timings.dense_ms = (time.perf_counter() - t) * 1000

        if "sparse" in channels:
            t = time.perf_counter()
            sparse_rows = [row for row, _ in self._sparse.search(query, depth)]
            timings.sparse_ms = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        rankings: list[tuple[float, Sequence[int]]] = []
        if "dense" in channels:
            rankings.append((dense_weight, dense_rows))
        if "sparse" in channels:
            rankings.append((sparse_weight, sparse_rows))
        scores = rrf_scores(rankings, k=rrf_k)

        # 1-based rank lookups, so a chunk can report which channel found it and where.
        dense_at = {row: i + 1 for i, row in enumerate(dense_rows)}
        sparse_at = {row: i + 1 for i, row in enumerate(sparse_rows)}

        # Ties break by first appearance across the input rankings, matching rrf_fuse.
        first_seen = {row: i for i, row in enumerate([*dense_rows, *sparse_rows])}
        ordered = sorted(scores, key=lambda r: (-scores[r], first_seen[r]))[:k]
        timings.fuse_ms = (time.perf_counter() - t) * 1000

        chunks = [
            RetrievedChunk(
                chunk=self._store.row(row),
                rank=i + 1,
                rrf_score=scores[row],
                dense_rank=dense_at.get(row),
                sparse_rank=sparse_at.get(row),
            )
            for i, row in enumerate(ordered)
        ]
        timings.total_ms = (time.perf_counter() - started) * 1000
        return SearchResult(query=query, k=k, chunks=chunks, timings=timings)
