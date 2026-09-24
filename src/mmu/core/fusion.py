"""Reciprocal Rank Fusion — the one place MMU's two retrievers are combined.

RRF:  score(d) = Σ_i  w_i / (k + rank_i(d)),  rank 1-based, k=60 by convention
(Cormack et al., 2009).

**Why rank and not score.** The two channels produce numbers that are not comparable:
FAISS returns a cosine similarity in [-1, 1] and BM25Okapi returns an unbounded
term-frequency score whose scale depends on the corpus, the document lengths and the
query. Adding them requires a normalization scheme, and every normalization scheme is
itself a tuning knob and a source of silent bugs (min-max is dominated by outliers;
z-score assumes a distribution BM25 does not have; both shift when the corpus changes).
RRF sidesteps the calibration problem entirely by looking only at rank position, which
is why it is the default fusion method in hybrid IR rather than weighted score combination.

**An absent id contributes exactly zero.** Do not "fix" this by assigning missing
documents a sentinel rank of n+1 — that gives every document a floor score from every
channel, which compresses the gap between "found by both channels" and "found by one",
and that gap is the entire signal hybrid retrieval exists to produce. ``test_fusion.py``
asserts the zero-contribution semantics so the sentinel cannot come back.

Pure and generic over any hashable id, so it unit-tests without a model or an index.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Hashable, TypeVar

T = TypeVar("T", bound=Hashable)

RRF_K = 60


def rrf_fuse(rankings: Sequence[tuple[float, Sequence[T]]], *, k: int = RRF_K) -> list[T]:
    """Fuse weighted rankings (best-first id lists) into one, highest score first.

    ``rankings`` is a list of ``(weight, ids)``. An id absent from a ranking simply
    contributes nothing from it. Ties break by first appearance, so the ordering is
    deterministic for a fixed input.
    """
    scores: dict[T, float] = {}
    order: dict[T, int] = {}
    seq = 0
    for weight, ids in rankings:
        for rank, item in enumerate(ids):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)
            if item not in order:
                order[item] = seq
                seq += 1
    return sorted(scores, key=lambda it: (-scores[it], order[it]))


def rrf_scores(
    rankings: Sequence[tuple[float, Sequence[T]]], *, k: int = RRF_K
) -> dict[T, float]:
    """The same fusion, but returning the scores themselves.

    ``rrf_fuse`` returns order only, which is all a retriever needs. The API surfaces
    ``rrf_score`` per chunk so a caller can see *how far apart* two results were, and
    the eval harness reports it — an order alone cannot distinguish a decisive win from
    a coin flip. Kept as a separate function so the hot path stays allocation-free.
    """
    scores: dict[T, float] = {}
    for weight, ids in rankings:
        for rank, item in enumerate(ids):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)
    return scores
