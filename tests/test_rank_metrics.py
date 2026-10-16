"""Recall@k, MRR and NDCG@k — the metrics a reranker can actually move.

The project's existing retrieval metrics are blind to reordering by construction:
`coverage` is a set intersection over `doc_id`, and `context_precision` normalises over
the relevant chunks it *retrieved*. A reranker reorders a fixed candidate pool, so both
of those are invariant under exactly the operation being evaluated. These tests pin
that claim down first (`test_reordering_is_invisible_to_coverage_*`) and then establish
what the new metrics do instead.
"""

from __future__ import annotations

import math

import pytest

from mmu.eval.metrics import (
    context_precision,
    count_relevant_chunks,
    cross_document_reasoning,
    rank_metrics,
)

from .conftest import make_chunk, make_retrieved

_SPAN = "not later than 72 hours"
_MUST = {"gdpr": [_SPAN]}


def _context(*relevant_positions: int, size: int = 8):
    """`size` retrieved gdpr chunks, relevant at the given 0-based positions."""
    texts = [f"filler chunk {i} about unrelated recitals" for i in range(size)]
    for p in relevant_positions:
        texts[p] = f"the controller shall notify {_SPAN} after becoming aware"
    return make_retrieved([make_chunk("gdpr", i, t) for i, t in enumerate(texts)])


# ------------------------------------------------- why the old metrics cannot see this


def test_reordering_is_invisible_to_coverage() -> None:
    """The finding that redirected this whole work package.

    Coverage asks whether each required `doc_id` appears anywhere in the context, so it
    is computed from a *set*. Reversing the ranking leaves that set identical, which is
    why a reranker cannot move the metric the ablation currently leads with — and hybrid
    already scores 1.000, so there was no headroom in either direction.
    """
    best_first = _context(0) + make_retrieved([make_chunk("company-policy", 9, "four hours")])
    worst_last = list(reversed(best_first))

    def cov(ctx):
        return cross_document_reasoning(
            required_doc_ids=("gdpr", "company-policy"),
            retrieved_doc_ids={c.doc_id for c in ctx},
            cited_doc_ids={"gdpr", "company-policy"},
            synthesis_score=1.0,
        ).coverage

    assert cov(best_first) == cov(worst_last) == 1.0

    # The same two orderings are plainly different to a rank-aware metric.
    kw = dict(relevant_doc_ids={"gdpr"}, must_contain=_MUST, n_relevant_total=1)
    assert rank_metrics(best_first, **kw).mrr == 1.0
    assert rank_metrics(worst_last, **kw).mrr == pytest.approx(1 / 9)


def test_reordering_is_visible_to_mrr_and_ndcg_where_cp_barely_moves() -> None:
    at_rank_1 = rank_metrics(_context(0), relevant_doc_ids={"gdpr"}, must_contain=_MUST,
                             n_relevant_total=1)
    at_rank_8 = rank_metrics(_context(7), relevant_doc_ids={"gdpr"}, must_contain=_MUST,
                             n_relevant_total=1)

    # Same chunks, same recall: the relevant one is in the top 8 either way.
    assert at_rank_1.recall == at_rank_8.recall == 1.0
    # But the ordering differs, and MRR/NDCG say so.
    assert at_rank_1.mrr == 1.0
    assert at_rank_8.mrr == pytest.approx(0.125)
    assert at_rank_1.ndcg == 1.0
    assert at_rank_8.ndcg < 0.4


# ------------------------------------------------------------------------------- MRR


def test_mrr_is_the_reciprocal_of_the_first_relevant_rank() -> None:
    for position in range(8):
        m = rank_metrics(_context(position), relevant_doc_ids={"gdpr"}, must_contain=_MUST)
        assert m.first_relevant_rank == position + 1
        assert m.mrr == pytest.approx(1.0 / (position + 1))


def test_mrr_only_counts_the_first_hit_so_later_relevant_chunks_do_not_raise_it() -> None:
    one = rank_metrics(_context(2), relevant_doc_ids={"gdpr"}, must_contain=_MUST)
    many = rank_metrics(_context(2, 3, 4), relevant_doc_ids={"gdpr"}, must_contain=_MUST)
    assert one.mrr == many.mrr == pytest.approx(1 / 3)
    assert many.n_relevant_retrieved == 3


def test_mrr_is_zero_not_none_when_nothing_relevant_was_retrieved() -> None:
    """A measured miss has a defined value; only ratios without denominators are None."""
    m = rank_metrics(_context(size=8), relevant_doc_ids={"gdpr"}, must_contain=_MUST,
                     n_relevant_total=4)
    assert m.n_relevant_retrieved == 0
    assert m.first_relevant_rank is None
    assert m.mrr == 0.0
    assert m.recall == 0.0


# ---------------------------------------------------------------------------- recall


def test_recall_divides_by_the_corpus_relevant_count_not_the_retrieved_one() -> None:
    """The denominator is the whole point. Two of five relevant chunks found is 0.4,
    not the 1.0 that normalising over what you happened to retrieve would report."""
    m = rank_metrics(_context(0, 1), relevant_doc_ids={"gdpr"}, must_contain=_MUST,
                     n_relevant_total=5)
    assert m.n_relevant_retrieved == 2
    assert m.recall == pytest.approx(0.4)


def test_shrinking_k_can_only_lower_recall_which_is_the_experiment() -> None:
    """Reranking is evaluated by whether recall@3 after reranking matches recall@8
    before it. This pins the direction that makes that comparison meaningful."""
    ctx = _context(0, 5)
    at_8 = rank_metrics(ctx, relevant_doc_ids={"gdpr"}, must_contain=_MUST, k=8,
                        n_relevant_total=2)
    at_3 = rank_metrics(ctx, relevant_doc_ids={"gdpr"}, must_contain=_MUST, k=3,
                        n_relevant_total=2)
    assert at_8.recall == 1.0
    assert at_3.recall == pytest.approx(0.5)
    assert at_3.recall < at_8.recall
    assert at_3.k == 3 and at_8.k == 8


def test_a_good_reranker_keeps_recall_at_k_3_that_a_plain_run_needed_k_8_for() -> None:
    """The result Phase 1 is trying to produce, expressed as arithmetic.

    Unreranked, the relevant chunk sits at rank 6 and is lost when k shrinks to 3.
    Reranked, the same chunk is at rank 1, so k=3 keeps the recall that previously
    needed k=8 — and a prompt of 3 chunks instead of 8 is where the latency comes from.
    """
    unreranked = _context(5)
    reranked = _context(0)

    assert rank_metrics(unreranked, relevant_doc_ids={"gdpr"}, must_contain=_MUST,
                        k=3, n_relevant_total=1).recall == 0.0
    assert rank_metrics(unreranked, relevant_doc_ids={"gdpr"}, must_contain=_MUST,
                        k=8, n_relevant_total=1).recall == 1.0
    assert rank_metrics(reranked, relevant_doc_ids={"gdpr"}, must_contain=_MUST,
                        k=3, n_relevant_total=1).recall == 1.0


# ------------------------------------------------------------------------------ NDCG


def test_ndcg_is_one_only_when_every_relevant_chunk_is_at_the_top() -> None:
    m = rank_metrics(_context(0, 1, 2), relevant_doc_ids={"gdpr"}, must_contain=_MUST,
                     n_relevant_total=3)
    assert m.ndcg == pytest.approx(1.0)


def test_ndcg_uses_the_corpus_ideal_so_finding_one_of_five_cannot_score_one() -> None:
    """The failure this guards against: an ideal ranking built from the *retrieved*
    relevant count would score a single hit at rank 1 as a perfect ranking, hiding
    four missed chunks. The ideal has to know how many existed."""
    found_one_of_five = rank_metrics(
        _context(0), relevant_doc_ids={"gdpr"}, must_contain=_MUST, n_relevant_total=5
    )
    found_one_of_one = rank_metrics(
        _context(0), relevant_doc_ids={"gdpr"}, must_contain=_MUST, n_relevant_total=1
    )
    assert found_one_of_one.ndcg == pytest.approx(1.0)
    assert found_one_of_five.ndcg < 0.45
    assert found_one_of_five.mrr == found_one_of_one.mrr == 1.0


def test_ndcg_discount_matches_the_log2_definition() -> None:
    """Spelled out rather than trusted: rel at ranks 1 and 3, two relevant in the
    corpus, so DCG = 1/log2(2) + 1/log2(4) and IDCG = 1/log2(2) + 1/log2(3)."""
    m = rank_metrics(_context(0, 2), relevant_doc_ids={"gdpr"}, must_contain=_MUST,
                     n_relevant_total=2)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert m.ndcg == pytest.approx(dcg / idcg)


def test_the_ideal_is_capped_at_k_so_recall_limited_cases_stay_scorable() -> None:
    """With 10 relevant chunks in the corpus and k=3, the best any ranking can do is 3
    relevant in the top 3. NDCG@3 must call that 1.0, or it would penalise a perfect
    ranking for the cutoff rather than for its ordering."""
    m = rank_metrics(_context(0, 1, 2), relevant_doc_ids={"gdpr"}, must_contain=_MUST,
                     k=3, n_relevant_total=10)
    assert m.ndcg == pytest.approx(1.0)
    assert m.recall == pytest.approx(0.3)


# ---------------------------------------------------- undefined rather than zero


def test_recall_and_ndcg_are_none_when_the_corpus_count_is_unknown() -> None:
    """Following the `cp=None` convention. Reporting 0.0 for an unknown denominator is
    the bug that was fixed twice already, in AF and in marker_compliance."""
    m = rank_metrics(_context(0), relevant_doc_ids={"gdpr"}, must_contain=_MUST)
    assert m.recall is None
    assert m.ndcg is None
    assert m.mrr == 1.0  # still measured


def test_an_unanswerable_question_yields_no_rank_metrics_at_all() -> None:
    """No relevant set means no relevant chunks: every ratio is undefined. It must drop
    out of the averages rather than enter them as a zero."""
    m = rank_metrics(_context(0), relevant_doc_ids=set(), must_contain={},
                     n_relevant_total=0)
    assert m.n_relevant_retrieved == 0
    assert m.recall is None
    assert m.ndcg is None


# ------------------------------------------------------- the corpus-wide denominator


def test_count_relevant_chunks_counts_over_the_whole_corpus() -> None:
    corpus = [
        ("gdpr", f"chunk about {_SPAN} notification"),
        ("gdpr", "chunk about something else entirely"),
        ("gdpr", f"another chunk quoting {_SPAN} again"),
        ("national-law", f"a chunk that also says {_SPAN}"),
    ]
    # must_contain narrows to the two gdpr chunks holding the span; national-law is not
    # in relevant_doc_ids at all.
    assert count_relevant_chunks(corpus, relevant_doc_ids={"gdpr"}, must_contain=_MUST) == 2


def test_a_relevant_doc_with_no_spans_makes_all_of_its_chunks_relevant() -> None:
    """Documented behaviour inherited from `_chunk_is_relevant`, and the reason
    `RankMetrics.recall` is not comparable across questions: without `must_contain`
    the denominator is the document's entire chunk count."""
    corpus = [("gdpr", f"chunk {i}") for i in range(271)]
    assert count_relevant_chunks(corpus, relevant_doc_ids={"gdpr"}, must_contain={}) == 271
    loose = rank_metrics(_context(0), relevant_doc_ids={"gdpr"}, must_contain={},
                         n_relevant_total=271)
    assert loose.recall < 0.03  # bounded near zero by construction, not a retrieval bug


def test_cp_and_rank_metrics_disagree_on_purpose() -> None:
    """A context with one relevant chunk at rank 1 out of five that exist in the corpus:
    CP reports a perfect 1.0 because it only grades what it retrieved, while recall
    reports 0.2. Both are correct about different questions, and the reranker
    experiment needs the second one."""
    ctx = _context(0)
    cp = context_precision(ctx, relevant_doc_ids={"gdpr"}, must_contain=_MUST)
    m = rank_metrics(ctx, relevant_doc_ids={"gdpr"}, must_contain=_MUST, n_relevant_total=5)
    assert cp == pytest.approx(1.0)
    assert m.recall == pytest.approx(0.2)
