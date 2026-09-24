"""Corpus -> chunks -> vectors -> index. The one place an index is created.

Runs from the CLI, never from an HTTP route: a rebuild re-embeds every document, which
takes minutes, and an HTTP endpoint doing that holds a connection past every proxy
timeout. ``GET /health`` reports a stale or missing index instead.

The document side of the encode is here, and it is the *only* place ``kind="document"``
is passed. Its counterpart, ``kind="query"``, appears exactly once too, in
:meth:`mmu.retrieve.hybrid.HybridRetriever.search`. Those two call sites are the whole
query/document asymmetry surface; ``test_prefix.py`` pins them with a recording embedder
because a mismatch here produces no error at all, only quietly worse rankings.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from mmu.core.models import Chunk
from mmu.embed.base import Embedder
from mmu.ingest.corpus import CorpusDocument, discover, scanned_pdf_warning
from mmu.ingest.reader import extract_text
from mmu.ingest.splitter import page_of, split_text
from mmu.retrieve.dense import FlatIPIndex
from mmu.retrieve.sparse import Bm25Index
from mmu.retrieve.store import INDEX_FILE, ChunkStore, Manifest

log = logging.getLogger(__name__)


def chunk_document(
    doc: CorpusDocument,
    *,
    max_tokens: int = 384,
    min_tokens: int = 100,
    overlap_tokens: int = 64,
) -> tuple[list[Chunk], list[str]]:
    """Read and split one document into :class:`Chunk`s, plus any warnings.

    The splitter returns offset spans and knows nothing about documents; this is where
    a span becomes a chunk with an identity. Page numbers come from the reader's page
    offsets, so a citation can name a page for a PDF and honestly name none for a
    Markdown file that has no pages until something renders it.
    """
    data = doc.path.read_bytes()
    extracted = extract_text(data, doc.kind)
    warnings: list[str] = []
    warning = scanned_pdf_warning(extracted, doc.filename)
    if warning:
        warnings.append(warning)

    spans = split_text(
        extracted.text,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_tokens=overlap_tokens,
    )
    chunks = [
        Chunk(
            chunk_id=f"{doc.doc_id}:{span.index}",
            doc_id=doc.doc_id,
            filename=doc.filename,
            chunk_index=span.index,
            text=span.text,
            char_start=span.char_start,
            char_end=span.char_end,
            page_from=page_of(extracted.page_starts, span.char_start),
            page_to=page_of(extracted.page_starts, max(span.char_end - 1, span.char_start)),
        )
        for span in spans
    ]
    return chunks, warnings


def embed_chunks(
    embedder: Embedder,
    chunks: Sequence[Chunk],
    *,
    batch_size: int = 16,
    on_batch: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Encode every chunk with ``kind="document"``, in batches.

    Batched because a single encode call over a whole corpus holds one allocation for
    every chunk's activations at once, and on a 6GB machine indexing on CPU that is the
    difference between slow and killed.
    """
    if not chunks:
        return np.zeros((0, embedder.dim), dtype=np.float32)
    out: list[np.ndarray] = []
    for start in range(0, len(chunks), batch_size):
        batch = [c.text for c in chunks[start : start + batch_size]]
        out.append(embedder.encode(batch, kind="document").dense)
        if on_batch is not None:
            on_batch(min(start + batch_size, len(chunks)), len(chunks))
    return np.vstack(out).astype(np.float32)


def build_index(
    corpus_dir: Path,
    index_dir: Path,
    embedder: Embedder,
    *,
    max_tokens: int = 384,
    min_tokens: int = 100,
    overlap_tokens: int = 64,
    batch_size: int = 16,
    on_batch: Callable[[int, int], None] | None = None,
) -> tuple[ChunkStore, FlatIPIndex, Bm25Index, list[str]]:
    """Build and persist the index. Returns the live objects plus collected warnings."""
    docs = discover(corpus_dir)
    if not docs:
        raise FileNotFoundError(
            f"no supported documents in {corpus_dir}. Drop PDF/txt/md files there first."
        )

    chunks: list[Chunk] = []
    warnings: list[str] = []
    for doc in docs:
        doc_chunks, doc_warnings = chunk_document(
            doc,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            overlap_tokens=overlap_tokens,
        )
        if not doc_chunks:
            # An empty document is not an error — but it must be visible, because every
            # downstream metric would otherwise read its absence as a model failure.
            warnings.append(f"{doc.filename}: produced no chunks (empty or unreadable)")
        chunks.extend(doc_chunks)
        warnings.extend(doc_warnings)

    if not chunks:
        raise ValueError(
            f"{len(docs)} document(s) in {corpus_dir} produced zero chunks. "
            "If these are scanned PDFs, they have no text layer and need OCR first."
        )

    vectors = embed_chunks(embedder, chunks, batch_size=batch_size, on_batch=on_batch)
    dense = FlatIPIndex(embedder.dim)
    dense.add(vectors)
    sparse = Bm25Index([c.text for c in chunks])

    manifest = Manifest(
        embed_model=embedder.model_name,
        dim=embedder.dim,
        chunk_max_tokens=max_tokens,
        chunk_min_tokens=min_tokens,
        chunk_overlap_tokens=overlap_tokens,
        n_chunks=len(chunks),
        doc_ids=[d.doc_id for d in docs],
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    store = ChunkStore(chunks, manifest)
    store.save(index_dir)
    dense.save(index_dir / INDEX_FILE)
    return store, dense, sparse, warnings


def load_index(index_dir: Path, embedder: Embedder) -> tuple[ChunkStore, FlatIPIndex, Bm25Index]:
    """Load a persisted index, refusing one built with incompatible settings.

    BM25 is rebuilt in memory rather than persisted: it is a tokenized term-frequency
    table over text the store already holds, it takes milliseconds to construct at this
    scale, and persisting it would add a second artifact that can silently fall out of
    sync with chunks.jsonl.
    """
    store = ChunkStore.load(index_dir)
    if store.manifest is not None:
        current = Manifest(
            embed_model=embedder.model_name,
            dim=embedder.dim,
            chunk_max_tokens=store.manifest.chunk_max_tokens,
            chunk_min_tokens=store.manifest.chunk_min_tokens,
            chunk_overlap_tokens=store.manifest.chunk_overlap_tokens,
            n_chunks=store.manifest.n_chunks,
            doc_ids=store.manifest.doc_ids,
            built_at=store.manifest.built_at,
        )
        problem = store.manifest.mismatch(current)
        if problem:
            raise ValueError(problem)
    dense = FlatIPIndex.load(index_dir / INDEX_FILE)
    sparse = Bm25Index(store.texts)
    return store, dense, sparse
