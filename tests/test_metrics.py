"""The biggest test file: hand-computed fixtures for all five metrics.

Every metric in mmu.eval.metrics is a pure function, so every test here builds its
inputs by hand rather than running a retriever or a generator.
"""

from __future__ import annotations

import pytest

from mmu.eval.metrics import (
    MARKER_COMPLIANCE_FLOOR,
    ClaimVerification,
    XDRResult,
    answer_faithfulness,
    context_precision,
    context_utilization,
    corpus_answer_faithfulness,
    corpus_lpqu,
    corpus_xdr,
    cross_document_reasoning,
    is_marker_compliance_reliable,
    lpqu,
    marker_compliance_rate,
    question_quality,
    synthesis_score_fallback,
)
from tests.conftest import make_chunk, make_retrieved

# ------------------------------------------------------------------- Answer Faithfulness


def test_af_supported_unsupported_mix() -> None:
    claims = [
        ClaimVerification(claim="a", supported=True),
        ClaimVerification(claim="b", supported=False),
        ClaimVerification(claim="c", supported=True),
    ]
    assert answer_faithfulness(claims, is_refusal=False) == pytest.approx(2 / 3)


def test_af_refusal_with_zero_claims_scores_one() -> None:
    assert answer_faithfulness([], is_refusal=True) == 1.0


def test_af_empty_degenerate_with_zero_claims_scores_zero() -> None:
    # Never award vacuous truth: an empty/degenerate answer that isn't a refusal must
    # not score as if it had said something true.
    assert answer_faithfulness([], is_refusal=False) == 0.0


def test_af_all_supported_is_one() -> None:
    claims = [ClaimVerification(claim="a", supported=True)] * 4
    assert answer_faithfulness(claims, is_refusal=False) == 1.0


def test_corpus_af_is_macro_mean_not_micro_mean() -> None:
    # q1: one claim, supported -> AF=1.0. q2: nine claims, none supported -> AF=0.0.
    q1_claims = [ClaimVerification(claim="only", supported=True)]
    q2_claims = [ClaimVerification(claim=f"c{i}", supported=False) for i in range(9)]
    af1 = answer_faithfulness(q1_claims, is_refusal=False)
    af2 = answer_faithfulness(q2_claims, is_refusal=False)
    assert af1 == 1.0
    assert af2 == 0.0

    macro = corpus_answer_faithfulness([af1, af2])
    assert macro == pytest.approx(0.5)

    # A micro-mean (supported claims / total claims across the corpus) would be
    # dominated by q2's nine claims: 1 supported / 10 total = 0.1. Macro must differ.
    total_supported = sum(c.supported for c in q1_claims + q2_claims)
    total_claims = len(q1_claims) + len(q2_claims)
    micro = total_supported / total_claims
    assert micro == pytest.approx(0.1)
    assert macro != pytest.approx(micro)


def test_corpus_af_empty_is_zero() -> None:
    assert corpus_answer_faithfulness([]) == 0.0


# --------------------------------------------------------------------- Context Precision


def test_cp_relevant_at_rank_one_beats_same_count_at_rank_five() -> None:
    span = "UNIQUE NOTIFICATION SPAN"
    must_contain = {"gdpr": [span]}

    def _chunks(relevant_index: int) -> list:
        texts = ["filler text about something else" for _ in range(5)]
        texts[relevant_index] = f"text containing {span} here"
        return make_retrieved([make_chunk("gdpr", i, t) for i, t in enumerate(texts)])

    cp_rank1 = context_precision(_chunks(0), relevant_doc_ids={"gdpr"}, must_contain=must_contain)
    cp_rank5 = context_precision(_chunks(4), relevant_doc_ids={"gdpr"}, must_contain=must_contain)

    assert cp_rank1 == pytest.approx(1.0)
    assert cp_rank5 == pytest.approx(0.2)
    assert cp_rank1 > cp_rank5


def test_cp_zero_when_nothing_relevant() -> None:
    chunks = make_retrieved([make_chunk("unrelated-doc", i, "text") for i in range(3)])
    assert context_precision(chunks, relevant_doc_ids={"gdpr"}) == 0.0


def test_cp_must_contain_narrows_doc_level_match_to_a_subset_of_chunks() -> None:
    span = "SPAN"
    chunks = make_retrieved(
        [
            make_chunk("gdpr", 0, f"has {span} in it"),
            make_chunk("gdpr", 1, "does not have the marker"),
            make_chunk("gdpr", 2, f"also has {span} in it"),
        ]
    )
    # Without must_contain, doc-level relevance alone marks all three chunks relevant —
    # an entire GDPR PDF being "relevant" is too loose (see context_precision docstring).
    cp_doc_level = context_precision(chunks, relevant_doc_ids={"gdpr"})
    assert cp_doc_level == pytest.approx(1.0)

    # With must_contain, only the chunks whose text actually contains the span count,
    # which narrows relevance to a strict subset (chunk 1 drops out) and changes CP.
    cp_span_level = context_precision(
        chunks, relevant_doc_ids={"gdpr"}, must_contain={"gdpr": [span]}
    )
    assert cp_span_level < cp_doc_level
    assert cp_span_level == pytest.approx((1 / 1 + 2 / 3) / 2)


# ----------------------------------------------------------------- Cross-Document Reasoning


def test_xdr_factor_decomposition_matches_worked_example() -> None:
    required = ("gdpr", "national-law", "company-policy")
    result = cross_document_reasoning(
        required_doc_ids=required,
        retrieved_doc_ids={"gdpr", "national-law", "company-policy"},  # Cov = 1.0
        cited_doc_ids={"gdpr", "national-law"},  # Cite = 2/3
        synthesis_score=0.5,  # Syn = 0.5
    )
    assert isinstance(result, XDRResult)
    assert result.coverage == pytest.approx(1.0)
    assert result.cite == pytest.approx(2 / 3)
    assert result.synthesis == pytest.approx(0.5)
    assert result.xdr == pytest.approx(1.0 * (2 / 3) * 0.5)
    assert result.xdr == pytest.approx(0.3333, abs=1e-3)
    assert result.n_required == 3
    # All four numbers must be present, never collapsed to just .xdr.
    assert (result.coverage, result.cite, result.synthesis, result.xdr) == (
        pytest.approx(1.0), pytest.approx(2 / 3), pytest.approx(0.5), pytest.approx(1 / 3),
    )


def test_xdr_low_coverage_vs_low_cite_are_distinguishable() -> None:
    required = ("a", "b")
    retrieval_bug = cross_document_reasoning(
        required_doc_ids=required, retrieved_doc_ids={"a"}, cited_doc_ids={"a"}, synthesis_score=1.0
    )
    generation_bug = cross_document_reasoning(
        required_doc_ids=required, retrieved_doc_ids={"a", "b"}, cited_doc_ids={"a"}, synthesis_score=1.0
    )
    # Both may end with the same product, but the factor that is low differs.
    assert retrieval_bug.coverage < 1.0
    assert generation_bug.coverage == 1.0
    assert generation_bug.cite < 1.0


def test_xdr_requires_at_least_two_required_docs() -> None:
    with pytest.raises(ValueError):
        cross_document_reasoning(
            required_doc_ids=("only-one",), retrieved_doc_ids={"only-one"},
            cited_doc_ids={"only-one"}, synthesis_score=1.0,
        )


def test_corpus_xdr_averages_only_given_results_and_reports_n() -> None:
    # Caller is responsible for filtering to cross-document (n>=2) questions before
    # calling this — corpus_xdr just aggregates and reports how many it averaged.
    results = [
        cross_document_reasoning(
            required_doc_ids=("a", "b"), retrieved_doc_ids={"a", "b"},
            cited_doc_ids={"a", "b"}, synthesis_score=1.0,
        ),
        cross_document_reasoning(
            required_doc_ids=("a", "b", "c"), retrieved_doc_ids={"a"},
            cited_doc_ids=set(), synthesis_score=0.0,
        ),
        cross_document_reasoning(
            required_doc_ids=("a", "b"), retrieved_doc_ids={"a", "b"},
            cited_doc_ids={"a"}, synthesis_score=0.5,
        ),
    ]
    agg = corpus_xdr(results)
    assert agg.n == 3
    assert agg.mean_xdr == pytest.approx((1.0 + 0.0 + 0.25) / 3)


def test_corpus_xdr_empty_is_zero_with_n_zero() -> None:
    agg = corpus_xdr([])
    assert agg.mean_xdr == 0.0
    assert agg.n == 0


def test_synthesis_score_fallback_thresholds() -> None:
    keys = ("72 hours", "supervisory authority", "escalate")
    assert synthesis_score_fallback("nothing relevant here", keys) == 0.0
    assert synthesis_score_fallback("we escalate promptly", keys) == 0.0  # 1/3 < 0.5
    assert synthesis_score_fallback(
        "notify the supervisory authority and escalate immediately", keys
    ) == 0.5  # 2/3 in [0.5, 1.0)
    assert synthesis_score_fallback(
        "within 72 hours, notify the supervisory authority and escalate", keys
    ) == 1.0


def test_synthesis_score_fallback_fuzzy_match_counts_as_present() -> None:
    # Reordered near-paraphrase of "supervisory authority" (not a substring) should
    # still count as present via rapidfuzz token_set_ratio.
    keys = ("supervisory authority",)
    assert synthesis_score_fallback("the authority supervisory was notified", keys) == 1.0


def test_synthesis_score_fallback_empty_keys_is_zero() -> None:
    assert synthesis_score_fallback("anything", ()) == 0.0


# -------------------------------------------------------------------------------- LpQU


def test_lpqu_eps_clamp_at_zero_quality() -> None:
    assert lpqu(10.0, 0.0, epsilon=0.01) == pytest.approx(10.0 / 0.01)


def test_question_quality_weighted_sum() -> None:
    q = question_quality(af=1.0, cp=0.5, xdr=0.0, weight_af=0.4, weight_cp=0.2, weight_xdr=0.4)
    assert q == pytest.approx(0.4 * 1.0 + 0.2 * 0.5 + 0.4 * 0.0)


def test_corpus_lpqu_aggregate_ratio_differs_from_mean_of_ratios_on_skewed_fixture() -> None:
    # Deliberately skewed: one question has near-zero quality (Q=0.01), dragging its
    # own t/Q ratio to 100 while everything else is fine. This test exists so nobody
    # "simplifies" corpus_lpqu from sum(t)/sum(Q) back into mean(t_i/Q_i) — see
    # corpus_lpqu's docstring for why a mean is the wrong aggregate.
    t_totals = [1.0, 1.0]
    qs = [0.01, 1.0]
    result = corpus_lpqu(t_totals, qs, epsilon=0.01)

    per_question_mean = sum(result.per_question) / len(result.per_question)
    assert result.per_question == pytest.approx((100.0, 1.0))
    assert per_question_mean == pytest.approx(50.5)
    assert result.corpus_lpqu == pytest.approx(2.0 / 1.01)
    assert result.corpus_lpqu != pytest.approx(per_question_mean)
    # The aggregate must be far less dominated by the bad question than the mean is.
    assert result.corpus_lpqu < per_question_mean / 10


def test_corpus_lpqu_reports_p50_p95_latency() -> None:
    t_totals = [1.0, 2.0, 3.0, 4.0, 100.0]
    qs = [1.0] * 5
    result = corpus_lpqu(t_totals, qs, epsilon=0.01)
    assert result.p50_latency_s == pytest.approx(3.0)
    assert result.p95_latency_s >= result.p50_latency_s


def test_corpus_lpqu_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        corpus_lpqu([1.0, 2.0], [1.0], epsilon=0.01)


def test_corpus_lpqu_empty_is_zero() -> None:
    result = corpus_lpqu([], [], epsilon=0.01)
    assert result.corpus_lpqu == 0.0
    assert result.per_question == ()


# ----------------------------------------------------------------- Context Utilization


def test_cur_cited_and_attr_diverge_on_citation_theater() -> None:
    chunks = make_retrieved([make_chunk("gdpr", 0, "first chunk text"), make_chunk("gdpr", 1, "second chunk text")])
    # The model cites chunk 0's marker, but AF never verified any claim against it.
    result = context_utilization(chunks, cited_chunk_ids={chunks[0].chunk_id}, claims=[])
    assert result.k == 2
    assert result.cited == pytest.approx(0.5)
    assert result.attr == pytest.approx(0.0)
    assert result.gap == pytest.approx(0.5)


def test_cur_attr_counts_only_verified_spans_with_matching_chunk() -> None:
    chunks = make_retrieved(
        [make_chunk("gdpr", 0, "notify within 72 hours"), make_chunk("gdpr", 1, "other content")]
    )
    claims = [
        ClaimVerification(claim="c1", supported=True, chunk_id=chunks[0].chunk_id, span="72 hours"),
        # Unsupported claim must not count even if it names a chunk_id/span.
        ClaimVerification(claim="c2", supported=False, chunk_id=chunks[1].chunk_id, span="other"),
    ]
    result = context_utilization(chunks, cited_chunk_ids=set(), claims=claims)
    assert result.attr == pytest.approx(0.5)


def test_cur_tok_merges_overlapping_verified_spans() -> None:
    text = "The quick brown fox jumps over the lazy dog"
    chunks = make_retrieved([make_chunk("gdpr", 0, text)])
    claims = [
        ClaimVerification(claim="c1", supported=True, chunk_id=chunks[0].chunk_id, span="quick brown fox"),
        ClaimVerification(claim="c2", supported=True, chunk_id=chunks[0].chunk_id, span="brown fox jumps"),
    ]
    result = context_utilization(chunks, cited_chunk_ids=set(), claims=claims)

    start1, end1 = text.index("quick brown fox"), text.index("quick brown fox") + len("quick brown fox")
    start2, end2 = text.index("brown fox jumps"), text.index("brown fox jumps") + len("brown fox jumps")
    merged_len = max(end1, end2) - min(start1, start2)  # the two spans overlap
    expected_tok = merged_len / len(text)

    assert result.tok == pytest.approx(expected_tok)
    assert result.tok <= 1.0
    # Sanity: naive (unmerged) double counting would exceed the merged value.
    naive = (len("quick brown fox") + len("brown fox jumps")) / len(text)
    assert result.tok < naive


def test_cur_k_zero_is_all_zero() -> None:
    result = context_utilization([], cited_chunk_ids=set(), claims=[])
    assert (result.cited, result.attr, result.tok, result.gap, result.k) == (0.0, 0.0, 0.0, 0.0, 0)


# ------------------------------------------------------------------- marker_compliance


def test_marker_compliance_rate_is_mean() -> None:
    assert marker_compliance_rate([1.0, 1.0, 0.0, 0.0]) == pytest.approx(0.5)


def test_marker_compliance_reliability_floor() -> None:
    assert is_marker_compliance_reliable(MARKER_COMPLIANCE_FLOOR) is True
    assert is_marker_compliance_reliable(MARKER_COMPLIANCE_FLOOR - 0.01) is False
    assert is_marker_compliance_reliable(1.0) is True
    assert is_marker_compliance_reliable(0.0) is False
