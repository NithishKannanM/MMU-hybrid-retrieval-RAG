"""The dense-index contract.

Kept in its own module, apart from the ``IndexFlatIP`` implementation, for one reason:
:mod:`mmu.retrieve.dense` imports faiss, and the Protocol is imported by the store, the
hybrid retriever, the API layer and half the test suite — none of which should pay for
a faiss import to reference a type.

Implementations address vectors by **row position**, not by id. The mapping from row to
``chunk_id`` is a parallel list in :class:`mmu.retrieve.store.ChunkStore`. That makes
deletion impossible without a rebuild, which is the right trade for a corpus that is
rebuilt from ``data/corpus/`` anyway; ``faiss.IndexIDMap2`` is the upgrade if
incremental delete is ever needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DenseIndex(Protocol):
    def add(self, vectors: np.ndarray) -> None:
        """Append ``(N, dim)`` float32 rows. Must reject non-unit-norm input."""
        ...

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(row, score)`` pairs, best first."""
        ...

    def save(self, path: Path) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def dim(self) -> int: ...
