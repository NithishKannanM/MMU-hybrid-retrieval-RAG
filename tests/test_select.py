"""Context selection: MMR and the per-document quota.

These exist because of a measured failure, not a hypothesis. In `reports/rerank/`
(30 questions, 90 generations) the cross-encoder lifted recall@8 0.469 -> 0.632 and MRR
0.641 -> 0.748 while coverage fell 0.929 -> 0.583 and XDR 0.452 -> 0.106, and **11 of
the 14 cross-document questions landed at exactly Cov = 0.50** — the reranker filled
all three context slots from whichever document was most on-topic. A cross-encoder
scores each (query, chunk) pair independently, so it cannot express a property of the
chosen *set*, and one-passage-per-required-document is exactly such a property.

`test_the_measured_failure_*` reproduces that shape directly; everything else pins the
behaviour the two selectors promise.
"""

from __future__ import annotations

import pytest

from mmu.core.models import SearchResult, Timings
from mmu.retrieve.select import (
    DocQuotaSelector,
    _normalized_relevance,
    MMRSelector,
    Selector,
    TopKSelector,
    apply,
    lexical_similarity,
)

from .conftest import make_chunk, make_retrieved


def _pool(*spec: tuple[str, str], scores: list[float] | None = None):
    """Candidates as (doc_id, text) in ranked order, optionally with rerank scores."""
    chunks = make_retrieved(
        [make_chunk(doc, i, text) for i, (doc, text) in enumerate(spec)]
    )
    if scores is not None:
        from dataclasses import replace

        chunks = [replace(c, rerank_score=s) for c, s in zip(chunks, scores)]
    return chunks


def _docs(chosen):
    return [c.doc_id for c in chosen]


def test_all_three_selectors_satisfy_the_protocol() -> None:
    for selector in (
        TopKSelector(),
        MMRSelector(similarity=lexical_similarity()),
        DocQuotaSelector(),
    ):
        assert isinstance(selector, Selector)


# ------------------------------------------------------ the failure being fixed


def _monopolised_pool():
    """A pool shaped like the measured failure: the three best candidates all come from
    one document, and the only passage from the second required document sits at rank 4.
    """
    return _pool(
        ("gdpr", "notification to the supervisory authority without undue delay"),
        ("gdpr", "notification to the supervisory authority not later than 72 hours"),
        ("gdpr", "notification to the supervisory authority with reasons for delay"),
        ("company-policy", "escalate to the Data Protection Officer within four hours"),
        scores=[9.0, 8.5, 8.0, 2.0],
    )


def test_the_measured_failure_reproduces_under_plain_truncation() -> None:
    chosen = TopKSelector().select(_monopolised_pool(), 3)
    assert _docs(chosen) == ["gdpr", "gdpr", "gdpr"]
    assert "company-policy" not in _docs(chosen), "this is Cov = 0.50"


def test_the_quota_recovers_the_second_document() -> None:
    chosen = DocQuotaSelector().select(_monopolised_pool(), 3)
    assert set(_docs(chosen)) == {"gdpr", "company-policy"}
    assert chosen[0].text.endswith("without undue delay"), "best chunk still leads"


def test_mmr_does_NOT_recover_it_when_the_redundant_chunks_are_also_far_more_relevant() -> None:
    """Measured, not assumed — and it is the honest prediction for the experiment.

    MMR admits a candidate when its relevance advantage is smaller than its redundancy
    penalty. In `_monopolised_pool` the scores are [9.0, 8.5, 8.0, 2.0], so after
    min-max scaling the second gdpr chunk has relevance 0.929 against the policy
    chunk's 0.0, while their redundancy differs by only 0.385 - 0.133 = 0.252. The
    relevance gap wins, at every lambda tried (0.3 / 0.5 / 0.7).

    So MMR's effect on coverage is *contingent on the cross-encoder's score
    distribution*, while the quota's is structural. If the real pools look like this
    one, MMR will not fix the Cov = 0.50 failures and the quota will. That is a
    prediction the ablation is there to test, and writing a test that asserted the
    opposite would have been rigging it.
    """
    for lam in (0.3, 0.5, 0.7):
        chosen = MMRSelector(similarity=lexical_similarity(), lambda_=lam).select(
            _monopolised_pool(), 3
        )
        assert _docs(chosen) == ["gdpr", "gdpr", "gdpr"], lam


def test_mmr_recovers_the_second_document_once_relevance_is_bunched() -> None:
    """The other side of the same rule: compress the score spread and the redundancy
    term starts deciding. Same texts, same similarity, only the scores differ."""
    pool = _pool(
        ("gdpr", "notification to the supervisory authority without undue delay"),
        ("gdpr", "notification to the supervisory authority not later than 72 hours"),
        ("gdpr", "notification to the supervisory authority with reasons for delay"),
        ("company-policy", "escalate to the Data Protection Officer within four hours"),
        scores=[9.0, 5.0, 4.5, 4.0],
    )
    chosen = MMRSelector(similarity=lexical_similarity(), lambda_=0.5).select(pool, 3)
    assert "company-policy" in _docs(chosen)
    assert chosen[0].rerank_score == 9.0, "the top candidate is never displaced"


# --------------------------------------------------------------------------- MMR


def test_mmr_always_picks_the_top_candidate_first() -> None:
    """With nothing selected yet the redundancy term is 0 for every candidate, so the
    first pick is the most relevant one. This is why MMR cannot damage MRR@1."""
    for lam in (0.0, 0.3, 0.5, 1.0):
        chosen = MMRSelector(similarity=lexical_similarity(), lambda_=lam).select(
            _monopolised_pool(), 3
        )
        assert chosen[0].rerank_score == 9.0, lam


def test_mmr_with_lambda_one_is_exactly_top_k() -> None:
    pool = _monopolised_pool()
    assert [c.text for c in MMRSelector(similarity=lexical_similarity(), lambda_=1.0).select(pool, 3)] == [
        c.text for c in TopKSelector().select(pool, 3)
    ]


def test_relevance_is_rescaled_into_the_unit_interval() -> None:
    """The units guard.

    Cross-encoder scores are unbounded logits (roughly -11..+9 observed on this corpus)
    while similarity lives in [0, 1]. Mixed raw under one lambda the relevance term is
    arithmetically incomparable to the redundancy term, and MMR degrades toward top-k
    for any usable lambda — which would read as "diversity selection does not help"
    rather than as a scaling mistake. Scaling makes `lambda_` mean what it says.
    """
    pool = _pool(("a", "x"), ("b", "y"), ("c", "z"), scores=[9.0, -1.0, -11.0])
    assert _normalized_relevance(pool) == pytest.approx([1.0, 0.5, 0.0])


def test_rescaling_can_change_which_candidate_mmr_admits() -> None:
    """Not merely cosmetic: with these scores the raw-score variant keeps the
    near-duplicate (0.5*5.0 - 0.5*0.385 = 2.31 beats 0.5*4.0 - 0.5*0.133 = 1.93) while
    the scaled one admits the distinct document. The scaling is load-bearing."""
    pool = _pool(
        ("gdpr", "notification to the supervisory authority without undue delay"),
        ("gdpr", "notification to the supervisory authority not later than 72 hours"),
        ("company-policy", "escalate to the Data Protection Officer within four hours"),
        scores=[9.0, 5.0, 4.0],
    )
    chosen = MMRSelector(similarity=lexical_similarity(), lambda_=0.5).select(pool, 2)
    assert _docs(chosen) == ["gdpr", "company-policy"]


def test_mmr_falls_back_to_the_fused_score_when_no_reranker_ran() -> None:
    pool = _pool(("gdpr", "alpha beta"), ("gdpr", "alpha beta gamma"))
    assert all(c.rerank_score is None for c in pool)
    chosen = MMRSelector(similarity=lexical_similarity()).select(pool, 2)
    assert len(chosen) == 2  # uses rrf_score; no crash, no empty relevance


def test_mmr_with_all_equal_scores_is_driven_purely_by_novelty() -> None:
    """A degenerate but correct case: the ranker expressed no preference, so the only
    signal left is redundancy."""
    pool = _pool(
        ("a", "one two three"),
        ("a", "one two three four"),      # near-duplicate of the first
        ("b", "entirely different words"),
        scores=[1.0, 1.0, 1.0],
    )
    chosen = MMRSelector(similarity=lexical_similarity(), lambda_=0.5).select(pool, 2)
    assert _docs(chosen) == ["a", "b"]


def test_mmr_keeps_the_better_ranked_candidate_on_a_tie() -> None:
    pool = _pool(("a", "xxx"), ("b", "yyy"), scores=[1.0, 1.0])
    chosen = MMRSelector(similarity=lambda x, y: 0.0, lambda_=1.0).select(pool, 1)
    assert chosen[0].doc_id == "a"


def test_mmr_rejects_a_lambda_outside_the_unit_interval() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="lambda_"):
            MMRSelector(similarity=lexical_similarity(), lambda_=bad)


# ------------------------------------------------------------------- doc quota


def test_the_auto_cap_is_k_over_the_documents_present() -> None:
    pool = _pool(*[("gdpr", f"g{i}") for i in range(5)],
                 *[("national-law", f"n{i}") for i in range(5)],
                 *[("company-policy", f"c{i}") for i in range(5)])
    # k=3 over 3 documents -> one each
    assert sorted(_docs(DocQuotaSelector().select(pool, 3))) == [
        "company-policy", "gdpr", "national-law"
    ]
    # k=8 over 3 documents -> ceil(8/3) = 3 each, filling 8
    chosen = DocQuotaSelector().select(pool, 8)
    assert len(chosen) == 8
    assert all(_docs(chosen).count(d) <= 3 for d in set(_docs(chosen)))


def test_an_explicit_cap_overrides_the_derived_one() -> None:
    pool = _pool(("gdpr", "a"), ("gdpr", "b"), ("national-law", "c"))
    chosen = DocQuotaSelector(max_per_doc=2).select(pool, 3)
    assert _docs(chosen) == ["gdpr", "gdpr", "national-law"]


def test_the_quota_backfills_rather_than_returning_a_short_context() -> None:
    """A single-document pool has no second document to offer. The quota must degrade
    to top-k rather than shrink the prompt: a shorter prompt would confound this
    experiment with a prompt-length change, and prompt length is what the latency
    result turns on."""
    pool = _pool(*[("gdpr", f"chunk {i}") for i in range(6)])
    chosen = DocQuotaSelector().select(pool, 3)
    assert len(chosen) == 3
    assert _docs(chosen) == ["gdpr"] * 3


def test_the_quota_preserves_reranked_order_among_what_it_admits() -> None:
    pool = _pool(("a", "a1"), ("b", "b1"), ("a", "a2"), ("c", "c1"))
    chosen = DocQuotaSelector(max_per_doc=1).select(pool, 3)
    assert [c.text for c in chosen] == ["a1", "b1", "c1"]


def test_the_quota_rejects_a_cap_below_one() -> None:
    with pytest.raises(ValueError, match="max_per_doc"):
        DocQuotaSelector(max_per_doc=0)


# --------------------------------------------------- shared contract + the stage


@pytest.mark.parametrize(
    "selector",
    [TopKSelector(), MMRSelector(similarity=lexical_similarity()), DocQuotaSelector()],
    ids=["top-k", "mmr", "quota"],
)
class TestEverySelector:
    def test_returns_exactly_k_when_the_pool_allows(self, selector) -> None:
        pool = _pool(*[("gdpr", f"c{i}") for i in range(10)])
        assert len(selector.select(pool, 4)) == 4

    def test_renumbers_rank_to_the_context_position(self, selector) -> None:
        pool = _pool(("a", "x"), ("b", "y"), ("c", "z"))
        assert [c.rank for c in selector.select(pool, 3)] == [1, 2, 3]

    def test_preserves_the_provenance_of_every_chunk_it_keeps(self, selector) -> None:
        """`dense_rank`/`sparse_rank`/`rrf_score`/`rerank_score` are the record of how a
        chunk got here and what each earlier stage thought of it. Selection decides
        which chunks reach the prompt; it must not rewrite their history."""
        pool = _pool(("a", "x"), ("b", "y"), ("c", "z"), scores=[3.0, 2.0, 1.0])
        before = {
            c.chunk_id: (c.dense_rank, c.sparse_rank, c.rrf_score, c.rerank_score)
            for c in pool
        }
        for c in selector.select(pool, 2):
            assert (c.dense_rank, c.sparse_rank, c.rrf_score, c.rerank_score) == before[
                c.chunk_id
            ]

    def test_an_empty_pool_selects_nothing(self, selector) -> None:
        assert selector.select([], 3) == []

    def test_a_non_positive_k_selects_nothing(self, selector) -> None:
        assert selector.select(_pool(("a", "x")), 0) == []

    def test_k_beyond_the_pool_returns_the_whole_pool(self, selector) -> None:
        pool = _pool(("a", "x"), ("b", "y"))
        assert len(selector.select(pool, 9)) == 2

    def test_apply_times_the_stage_and_does_not_mutate_its_input(self, selector) -> None:
        pool = _pool(("a", "x"), ("b", "y"), ("c", "z"))
        result = SearchResult(query="q", k=3, chunks=pool, timings=Timings(total_ms=5.0))
        before = [(c.chunk_id, c.rank) for c in result.chunks]

        out = apply(result, selector, k=2)
        assert out.k == 2 and len(out.chunks) == 2
        assert out.timings.select_ms >= 0.0
        assert out.timings.total_ms == 5.0, "retrieval timing untouched"
        assert [(c.chunk_id, c.rank) for c in result.chunks] == before


def test_apply_passes_an_empty_result_straight_through() -> None:
    empty = SearchResult(query="q", k=0, chunks=[], timings=Timings())
    assert apply(empty, TopKSelector(), k=3) is empty
