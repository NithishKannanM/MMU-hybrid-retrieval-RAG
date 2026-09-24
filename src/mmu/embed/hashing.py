"""A deterministic, dependency-free embedder. The keystone of the test strategy.

Real retrieval quality needs BGE-M3 and a 2.3GB download. Almost none of the *code*
does: the index, the store, the fusion, the hybrid retriever, the API layer and the
whole eval harness only need vectors that behave consistently — same text in, same
vector out, and texts sharing words landing near each other. This produces exactly that
from ``hashlib``, which is why ``uv run pytest`` downloads nothing and makes no network
call while still exercising the real code paths.

**How.** Each token is hashed to a seed, the seed drives a fixed-size pseudo-random unit
vector, and a text's vector is the normalized sum of its tokens'. That makes cosine
similarity a smooth function of token overlap — enough for "the chunk containing the
query's words ranks first" to be a meaningful assertion, and not remotely enough to say
anything about semantic quality. It is a test double, never a fallback: nothing in the
serving or indexing path may select it silently.

``blake2b`` rather than ``hash()``: Python's built-in string hash is salted per process,
so a ``hash()``-seeded embedder would produce a different index on every run and every
"deterministic" test would be a coin flip that passes locally and fails in CI.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from mmu.embed.base import Encoding, InputKind

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.'][a-z0-9]+)*")


def _seed(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


class HashingEmbedder:
    """Deterministic pseudo-embeddings. Implements :class:`mmu.embed.base.Embedder`."""

    def __init__(self, dim: int = 256, *, asymmetric: bool = False) -> None:
        self._dim = dim
        #: When True, ``kind`` perturbs the vector, so a query and an identical document
        #: encode differently. Default False because that is what BGE-M3 actually does
        #: (no retrieval instruction, see mmu.embed.prefix) and because a retriever test
        #: asserting "the chunk containing the query's words ranks first" needs the two
        #: sides to live in the same space. Set True only to test that an asymmetric
        #: model's two sides are in fact distinguishable.
        self._asymmetric = asymmetric

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "hashing-test-embedder"

    def encode(self, texts: Sequence[str], *, kind: InputKind) -> Encoding:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        salt = kind if self._asymmetric else ""
        for i, text in enumerate(texts):
            tokens = _TOKEN.findall(text.lower())
            if not tokens:
                # An all-punctuation chunk still needs a unit vector: a zero row would
                # make FAISS return NaN scores and the unit-norm assertion would fire on
                # input the corpus can legitimately contain.
                out[i, 0] = 1.0
                continue
            acc = np.zeros(self._dim, dtype=np.float64)
            for token in tokens:
                rng = np.random.default_rng(_seed(salt + token))
                acc += rng.standard_normal(self._dim)
            norm = float(np.linalg.norm(acc))
            out[i] = (acc / norm).astype(np.float32) if norm else np.eye(1, self._dim)[0]
        return Encoding(dense=out)
