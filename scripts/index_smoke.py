"""LIVE gate: build a real index with the real embedder and query it.

Run this after `uv sync` and after dropping documents into `data/corpus/`. It is a
script rather than a test because it downloads BGE-M3 (~2.3GB) on first run and takes
minutes on a cold cache — `uv run pytest` must stay offline and fast.

    uv run python scripts/index_smoke.py

If `data/corpus/` is empty it synthesizes a three-document corpus in a temp directory
(using the same in-memory PDF builder the tests use) so the path can be exercised before
you have real documents.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from mmu.config import get_settings  # noqa: E402
from mmu.embed.bge_m3 import BgeM3Embedder  # noqa: E402
from mmu.ingest.corpus import discover  # noqa: E402
from mmu.retrieve.build import build_index  # noqa: E402
from mmu.retrieve.hybrid import HybridRetriever  # noqa: E402

QUERIES = [
    "personal data breach notification deadline",
    "data controller",           # the exact-defined-term case BM25 exists for
    "who do we tell and how fast",
]


def _fallback_corpus() -> Path:
    from pdf_fixture import build_pdf

    tmp = Path(tempfile.mkdtemp(prefix="mmu-smoke-"))
    (tmp / "gdpr.txt").write_text(
        "Article 33. In the case of a personal data breach, the controller shall "
        "without undue delay and, where feasible, not later than 72 hours after having "
        "become aware of it, notify the personal data breach to the competent "
        "supervisory authority.\n",
        encoding="utf-8",
    )
    (tmp / "national-law.txt").write_text(
        "Section 40. The competent supervisory authority for controllers established in "
        "this Member State is the Federal Commissioner for Data Protection.\n",
        encoding="utf-8",
    )
    (tmp / "company-policy.pdf").write_bytes(
        build_pdf([[
            "Security incident response policy.",
            "Any suspected security incident must be escalated to the",
            "Data Protection Officer within four hours of detection.",
        ]])
    )
    return tmp


def main() -> int:
    settings = get_settings()
    corpus_dir = settings.corpus_dir
    if not discover(corpus_dir):
        corpus_dir = _fallback_corpus()
        print(f"data/corpus/ is empty — using a synthesized corpus at {corpus_dir}\n")

    print(f"loading {settings.embed_model} on {settings.embed_device} "
          f"(first run downloads ~2.3GB)...")
    embedder = BgeM3Embedder(
        model_name=settings.embed_model,
        device=settings.embed_device,
        batch_size=settings.embed_batch_size,
    )
    print(f"  dim = {embedder.dim}\n")

    store, dense, sparse, warnings = build_index(
        corpus_dir, settings.index_dir, embedder,
        max_tokens=settings.chunk_max_tokens,
        min_tokens=settings.chunk_min_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        batch_size=settings.embed_batch_size,
        on_batch=lambda done, total: print(f"  embedded {done}/{total}", end="\r"),
    )
    print()
    for w in warnings:
        print(f"WARNING: {w}")
    print(f"\nindexed {len(store)} chunks from {len(store.doc_ids())} documents: "
          f"{', '.join(store.doc_ids())}\n")

    retriever = HybridRetriever(store, dense, sparse, embedder)
    for query in QUERIES:
        result = retriever.search(query, k=5)
        print(f"query: {query!r}  ({result.timings.total_ms:.0f}ms)")
        for rc in result.chunks:
            channels = []
            if rc.dense_rank:
                channels.append(f"dense#{rc.dense_rank}")
            if rc.sparse_rank:
                channels.append(f"sparse#{rc.sparse_rank}")
            flag = "  <- sparse-only" if rc.dense_rank is None else ""
            print(f"  {rc.rank}. [{rc.doc_id}] {rc.rrf_score:.4f} "
                  f"({'+'.join(channels)}){flag}")
            print(f"     {rc.text[:90].strip()}...")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
