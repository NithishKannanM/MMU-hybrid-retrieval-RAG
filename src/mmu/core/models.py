"""The dataclasses every package speaks. This module is the contract.

Ingest produces :class:`Chunk`. Retrieval turns those into :class:`RetrievedChunk` by
adding rank provenance. Generation consumes retrieved chunks and produces
:class:`Answer`. The eval harness consumes :class:`Answer` and :class:`Timings` as
*data* — it never imports a retriever or a generator, which is what lets the metric
math be developed and tested independently of either.

Nothing here imports torch, faiss, fastapi or httpx. That is deliberate: this module is
imported by all seven packages, so anything expensive in it is paid for seven times and
would make the pure-function unit tests drag in the whole stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    """One indexed span of one document.

    ``char_start``/``char_end`` are offsets into the document's *extracted text*, which
    makes ``chunk.text == source[char_start:char_end]`` an invariant a test can assert
    and lets a citation point at the exact source region rather than a re-joined
    approximation of it.
    """

    chunk_id: str          # f"{doc_id}:{chunk_index}" — stable across rebuilds of the same text
    doc_id: str            # slugified filename stem; the id questions.jsonl refers to
    filename: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    page_from: int | None = None   # None for formats with no page structure (txt/md)
    page_to: int | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk as returned by retrieval, carrying *which channel found it*.

    ``dense_rank`` and ``sparse_rank`` are the point of this type. Without them you
    cannot tell whether a good result came from the hybrid union or would have appeared
    from dense alone — which is the single most interesting question about this
    architecture, and the one the eval ablation exists to answer. ``None`` means that
    channel did not return this chunk within its over-fetch depth at all.
    """

    chunk: Chunk
    rank: int              # 1-based position in the fused ordering
    rrf_score: float
    dense_rank: int | None = None    # 1-based rank within the dense channel, or None
    sparse_rank: int | None = None   # 1-based rank within the sparse channel, or None

    # Convenience passthroughs — the API and the metrics both address chunks by these,
    # and reaching through `.chunk.` at every call site obscures the metric formulas.
    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass(frozen=True)
class Citation:
    """A resolved ``[Sn]`` marker: which source number pointed at which chunk."""

    marker: int            # the n in [Sn], 1-based, as it appears in the answer
    chunk_id: str
    doc_id: str
    filename: str
    page_from: int | None = None


@dataclass
class Timings:
    """Wall-clock split of one request. Present on every response.

    This is what makes LpQU measurable, and — more importantly — what makes an LpQU
    change *diagnosable*. Generation dominates retrieval by roughly 50x on a local
    model, so a quality-per-latency improvement is almost always a generation
    improvement unless this breakdown proves otherwise.
    """

    embed_ms: float = 0.0
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    fuse_ms: float = 0.0
    generate_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def retrieval_ms(self) -> float:
        return self.embed_ms + self.dense_ms + self.sparse_ms + self.fuse_ms


@dataclass
class SearchResult:
    """Retrieval output, with no generation. What `POST /search` returns."""

    query: str
    k: int
    chunks: list[RetrievedChunk] = field(default_factory=list)
    timings: Timings = field(default_factory=Timings)


@dataclass
class Answer:
    """A grounded answer plus everything the harness needs to score it.

    ``marker_compliance`` is 1.0 when the answer contains at least one well-formed
    ``[Sn]`` marker with n <= k, else 0.0. It is recorded per-answer rather than derived
    later because the whole deterministic-metric design (Cite, XDR, every CUR variant)
    rests on markers being parseable, and a run where the model dropped the format must
    be visibly flagged rather than silently scored as if it had complied.
    """

    query: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    model: str = ""
    timings: Timings = field(default_factory=Timings)
    marker_compliance: float = 0.0
