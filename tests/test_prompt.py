"""The prompt seam: marker numbering, parsing, citation resolution, and refusal
detection. Everything here is pure — no network, no model, nothing async."""

from __future__ import annotations

from mmu.generate.prompt import (
    MARKER_RE,
    SYSTEM_PROMPT,
    build_messages,
    is_refusal,
    marker_compliance,
    parse_markers,
    resolve_citations,
)
from tests.conftest import make_chunk, make_retrieved


def _chunks(n: int):
    return make_retrieved([make_chunk(f"doc{i}", 0, f"text of doc {i}") for i in range(n)])


# --------------------------------------------------------------------- build_messages


def test_build_messages_numbers_context_1based_contiguous() -> None:
    chunks = _chunks(4)
    messages = build_messages("What is the deadline?", chunks)

    assert messages[0].role == "system"
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[1].role == "user"

    body = messages[1].content
    # Every source appears, numbered 1..k in the order the chunks were given.
    for n in range(1, 5):
        assert f"[S{n}]" in body
    # And no stray S5 leaked in from anywhere.
    assert "[S5]" not in body
    assert "What is the deadline?" in body


def test_system_prompt_mandates_markers_and_explicit_refusal() -> None:
    # These are the load-bearing clauses the brief calls out; assert their presence
    # directly so a future edit to the wording cannot silently drop one.
    assert "[S" in SYSTEM_PROMPT  # cites the marker format by example
    assert "context" in SYSTEM_PROMPT.lower()
    assert "does not contain" in SYSTEM_PROMPT.lower()


def test_build_messages_empty_context_does_not_crash() -> None:
    messages = build_messages("anything?", [])
    assert "Question: anything?" in messages[1].content


# ----------------------------------------------------------------------- parse_markers


def test_parse_markers_1based_first_appearance_order() -> None:
    text = "First claim [S2]. Second claim [S1][S3]. Repeat of [S2]."
    assert parse_markers(text, k=3) == [2, 1, 3]


def test_parse_markers_drops_out_of_range_without_crashing() -> None:
    # k=3, but the model invented S9 and S0 (0 is out of range — markers are 1-based).
    text = "Claim [S9] and another [S0] and a real one [S2]."
    assert parse_markers(text, k=3) == [2]


def test_parse_markers_no_markers_returns_empty() -> None:
    assert parse_markers("No citations here at all.", k=8) == []


def test_marker_re_matches_expected_shape() -> None:
    assert MARKER_RE.findall("[S1][S12] not [S] not [Sx]") == ["1", "12"]


# ------------------------------------------------------------------- resolve_citations


def test_resolve_citations_maps_to_right_doc_id() -> None:
    chunks = _chunks(3)
    text = "The controller must notify within 72 hours [S2]."
    citations = resolve_citations(text, chunks)

    assert len(citations) == 1
    c = citations[0]
    assert c.marker == 2
    assert c.doc_id == chunks[1].doc_id
    assert c.chunk_id == chunks[1].chunk_id


def test_resolve_citations_preserves_first_appearance_order_and_drops_bad() -> None:
    chunks = _chunks(2)
    text = "Combines [S2] and [S1], and an invented [S7]."
    citations = resolve_citations(text, chunks)
    assert [c.marker for c in citations] == [2, 1]
    assert [c.doc_id for c in citations] == [chunks[1].doc_id, chunks[0].doc_id]


# ------------------------------------------------------------------- marker_compliance


def test_marker_compliance_zero_with_no_markers() -> None:
    assert marker_compliance("An answer with no citations at all.", k=5) == 0.0


def test_marker_compliance_zero_when_only_marker_is_out_of_range() -> None:
    assert marker_compliance("Only source is [S9].", k=3) == 0.0


def test_marker_compliance_one_with_at_least_one_in_range_marker() -> None:
    assert marker_compliance("Supported by [S1] and the invented [S99].", k=3) == 1.0


# -------------------------------------------------------------------------- is_refusal


def test_is_refusal_true_for_explicit_refusal() -> None:
    assert is_refusal("The provided context does not contain this information.")


def test_is_refusal_true_for_cannot_be_answered_phrasing() -> None:
    assert is_refusal(
        "This question cannot be answered from the provided context; no retention "
        "period is given."
    )


def test_is_refusal_true_for_no_information_found_phrasing() -> None:
    assert is_refusal("No relevant information is found in the provided context.")


def test_is_refusal_false_for_grounded_answer_with_citation() -> None:
    assert not is_refusal("The controller must notify within 72 hours [S1].")


def test_is_refusal_false_near_miss_negation_about_claim_not_context() -> None:
    # The near-miss: "does not" appears, but it negates the *content* of a cited claim
    # ("does not require... only 72 hours"), not whether the context contains an
    # answer. A refusal detector that fires on any negation would misclassify this real,
    # grounded answer as a refusal and hand it a false faithfulness score of 1.0.
    text = (
        "The policy does not require notification within 24 hours; it requires "
        "notification within 72 hours [S1]."
    )
    assert not is_refusal(text)


def test_is_refusal_false_for_empty_answer_without_refusal_language() -> None:
    assert not is_refusal("")
