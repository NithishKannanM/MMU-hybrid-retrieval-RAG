"""The embedder contract — and the one argument that cannot be forgotten.

**Why ``kind`` is a required keyword argument with no default.** A bi-encoder used for
retrieval can be asymmetric: the query side and the document side may need different
treatment at encode time. When it is needed and you skip it, nothing raises — retrieval
just gets quietly worse, which is the most expensive class of bug in a system whose
whole output is a ranking. Making ``kind`` mandatory converts that silent quality
regression into a ``TypeError`` at the call site, caught by the type checker and by any
test that exercises the path. That is a structural guarantee; "remember the prefix" is
a convention, and conventions do not survive a refactor.

See :mod:`mmu.embed.prefix` for what each model actually does with ``kind``. For
BGE-M3 specifically the answer is "nothing" — and that emptiness is a registered,
asserted decision rather than a missing feature.

**Why ``Encoding`` has three fields when only one is used.** BGE-M3 produces dense,
sparse (lexical weights) and ColBERT-style multi-vector representations from a single
forward pass; that is the reason it pairs so well with hybrid retrieval without
doubling inference infrastructure. MMU fuses dense + BM25 today, so only ``dense`` is
populated. The other two fields exist now so that wiring them later is one new class
behind this same Protocol plus two more channels in the ``rrf_fuse`` call — with zero
changes to the index, the store, the API or the eval harness.

**The library tradeoff, recorded.** ``sentence-transformers`` exposes BGE-M3's dense
head only; the sparse and ColBERT heads are extra layers that only ``FlagEmbedding``'s
``BGEM3FlagModel`` loads. FlagEmbedding was rejected as a dependency because it pulls
the entire training stack (datasets, accelerate, peft, ir-datasets) to do inference and
declares no ``requires-python``. If the other two heads are ever needed, that is the
swap — a new ``Embedder`` implementation, nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np

#: Which side of the retrieval pair a text is being encoded for.
InputKind = Literal["query", "document"]


@dataclass(frozen=True)
class Encoding:
    """The output of one encode call.

    ``dense`` is ``(N, dim)`` float32 and **L2-normalized**, so inner product equals
    cosine similarity. That normalization is what makes ``faiss.IndexFlatIP`` a cosine
    index; :mod:`mmu.retrieve.dense` asserts it on add rather than trusting it, because
    an unnormalized vector does not error — it just ranks by magnitude, and long chunks
    quietly win every query.
    """

    dense: np.ndarray
    sparse: list[dict[int, float]] | None = None   # token_id -> weight; None until wired
    colbert: list[np.ndarray] | None = None        # (T_i, dim) per text; None until wired


@runtime_checkable
class Embedder(Protocol):
    """Text -> vectors. Implementations must be safe to call from a worker thread."""

    def encode(self, texts: Sequence[str], *, kind: InputKind) -> Encoding: ...

    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...


def encode_one(embedder: Embedder, text: str, *, kind: InputKind) -> np.ndarray:
    """Encode a single text to a single dense vector.

    A helper rather than a Protocol method: every implementation would write the same
    two lines, and a Protocol is cheaper to satisfy the fewer methods it demands.
    ``kind`` stays mandatory here too — a convenience wrapper that made it optional
    would be a hole straight through the guarantee this module exists to provide.
    """
    return embedder.encode([text], kind=kind).dense[0]
