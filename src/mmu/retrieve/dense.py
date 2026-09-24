"""The dense channel: exact inner-product search over BGE-M3 vectors.

**``IndexFlatIP``, deliberately — not HNSW, not IVF.** A flat index scans every vector:
O(n·d) per query, no training step, no ``nprobe``/``efSearch`` to tune, and — the reason
it is here — **zero approximation recall loss**. At benchmark scale (a handful of legal
documents, order 10²–10³ chunks) a 1024-dim flat scan costs well under a millisecond,
and the generation call that follows costs three orders of magnitude more.

The argument is not that approximate search would be slow. It is that every
retrieval-quality number the eval harness reports would silently carry an unmeasured ANN
recall penalty baked into it. Context Precision would then be measuring the *index*
rather than the retrieval design, and the ablation between dense-only, sparse-only and
hybrid — the experiment this whole project exists to run — would be contaminated by a
variable nobody is tracking.

**This is the first thing to swap when scaling, and ``_make_index`` is the only line
that changes.** At ~10⁵ vectors: ``faiss.IndexHNSWFlat(dim, 32)`` with
``efConstruction=200``, ``efSearch=64`` — no training call, roughly 95–98% recall@10,
sub-millisecond. At ~10⁶: ``IndexIVFFlat(quantizer, dim, nlist≈4·√n)``, which needs a
``train()`` pass over a sample and is therefore not the first step. In both cases, run
``mmu eval run`` before and after and report the delta: an ANN swap that costs three
points of Context Precision is a decision, not a free win.

**Cosine equals inner product only for unit-norm vectors**, which is why ``add`` asserts
the norm instead of trusting the embedder. An unnormalized vector does not raise — it
just ranks by magnitude, so the longest chunks quietly win every query, and the only
symptom is that retrieval is mysteriously bad.

**Row addressing, not ids.** ``IndexFlatIP`` addresses vectors by position and
:class:`mmu.retrieve.store.ChunkStore` keeps the parallel ``row -> chunk_id`` list. That
makes deletion impossible without a rebuild, which is the right trade for a corpus that
is rebuilt from ``data/corpus/`` anyway; ``IndexIDMap2`` is the upgrade if incremental
delete is ever needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: How far a stored vector's L2 norm may drift from 1.0 before we refuse it. Loose
#: enough for float32 accumulation over 1024 dims, tight enough that an unnormalized
#: vector (norm in the tens) cannot slip through.
_NORM_TOLERANCE = 1e-3


class FlatIPIndex:
    """Exact inner-product index. Implements :class:`mmu.retrieve.base.DenseIndex`."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = self._make_index(dim)

    @staticmethod
    def _make_index(dim: int):
        # <-- THE SEAM. Everything about swapping to an approximate index is in the
        # module docstring; this is the one line that changes when you do.
        import faiss

        return faiss.IndexFlatIP(dim)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"expected (N, {self._dim}) vectors, got {vectors.shape}"
            )
        if len(vectors) == 0:
            return
        norms = np.linalg.norm(vectors, axis=1)
        worst = float(np.max(np.abs(norms - 1.0)))
        if worst > _NORM_TOLERANCE:
            # Not a warning. A non-normalized vector turns this cosine index into a
            # magnitude ranker, which does not error anywhere downstream — it just
            # returns the longest chunks for every query.
            raise ValueError(
                f"vectors must be L2-normalized for IndexFlatIP to mean cosine; "
                f"worst deviation {worst:.4g} exceeds {_NORM_TOLERANCE}"
            )
        self._index.add(vectors)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(row, score)`` pairs, best first."""
        if self.size == 0 or k <= 0:
            return []
        q = np.ascontiguousarray(np.atleast_2d(query), dtype=np.float32)
        # faiss pads with row -1 when k exceeds ntotal; filter rather than clamp so the
        # caller can ask for more than the corpus holds without special-casing.
        scores, rows = self._index.search(q, min(k, self.size))
        return [(int(r), float(s)) for r, s in zip(rows[0], scores[0]) if r != -1]

    def save(self, path: Path) -> None:
        import faiss

        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path) -> "FlatIPIndex":
        import faiss

        index = faiss.read_index(str(path))
        obj = cls.__new__(cls)
        obj._index = index
        obj._dim = int(index.d)
        return obj
