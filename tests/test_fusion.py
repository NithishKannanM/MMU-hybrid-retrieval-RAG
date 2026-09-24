"""RRF: the exact scores, the tie rule, and the sentinel that must never come back."""

from __future__ import annotations

import pytest

from mmu.core.fusion import RRF_K, rrf_fuse, rrf_scores


def test_hand_computed_scores() -> None:
    rankings = [(1.0, ["a", "b", "c"]), (1.0, ["c", "a", "d"])]
    scores = rrf_scores(rankings)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["d"] == pytest.approx(1 / 63)
    # 'a' (top of one list, second of the other) beats 'c' (third and first).
    assert rrf_fuse(rankings) == ["a", "c", "b", "d"]


def test_absent_id_contributes_exactly_zero() -> None:
    """The regression guard against the rejected `n+1` sentinel.

    An earlier implementation in a sibling project gave documents missing from a channel
    a sentinel rank of ``n + 1``, so every document collected a floor score from every
    channel. That compresses the gap between 'found by both channels' and 'found by one'
    — which is the entire signal hybrid retrieval produces. Here, absence is silence.
    """
    only_dense = rrf_scores([(1.0, ["x"]), (1.0, ["y"])])
    assert only_dense["x"] == pytest.approx(1 / (RRF_K + 1))
    assert only_dense["y"] == pytest.approx(1 / (RRF_K + 1))

    # A doc found by both channels must outscore one found by a single channel at the
    # same rank. Under the sentinel scheme this margin shrinks toward zero.
    both = rrf_scores([(1.0, ["p", "q"]), (1.0, ["p", "r"])])
    assert both["p"] > both["q"] + 1e-9
    assert both["p"] == pytest.approx(2 / (RRF_K + 1))


def test_ties_break_by_first_appearance() -> None:
    fused = rrf_fuse([(1.0, ["a", "b"]), (1.0, ["b", "a"])])
    assert fused == ["a", "b"]  # identical scores; 'a' was seen first
    assert rrf_fuse([(1.0, ["b", "a"]), (1.0, ["a", "b"])]) == ["b", "a"]


def test_weights_change_the_order() -> None:
    rankings_sparse_heavy = [(1.0, ["a", "b"]), (5.0, ["b", "a"])]
    assert rrf_fuse(rankings_sparse_heavy)[0] == "b"


def test_k_controls_rank_discrimination() -> None:
    """Small k sharpens the gap between rank 1 and rank 2; large k flattens it."""
    sharp = rrf_scores([(1.0, ["a", "b"])], k=1)
    flat = rrf_scores([(1.0, ["a", "b"])], k=1000)
    assert sharp["a"] / sharp["b"] > flat["a"] / flat["b"]


def test_empty_input() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([(1.0, [])]) == []
