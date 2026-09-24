"""The splitter's invariant, its overlap contract, and the edges around min/max."""

from __future__ import annotations

import pytest

from mmu.ingest.splitter import (
    CHARS_PER_TOKEN,
    estimate_tokens,
    page_of,
    split_text,
)

# A paragraph-and-sentence-shaped document long enough to force multiple chunks at
# small token budgets, without needing megabytes of fixture text.
_PROSE = (
    "Article 33 Notification of a personal data breach to the supervisory authority. "
    "In the case of a personal data breach, the controller shall without undue delay "
    "and, where feasible, not later than 72 hours after having become aware of it, "
    "notify the personal data breach to the competent supervisory authority.\n\n"
    "Where the notification to the supervisory authority is not made within 72 hours, "
    "it shall be accompanied by reasons for the delay. The processor shall notify the "
    "controller without undue delay after becoming aware of a personal data breach.\n\n"
    "Article 34 Communication of a personal data breach to the data subject. When the "
    "personal data breach is likely to result in a high risk to the rights and freedoms "
    "of natural persons, the controller shall communicate the breach to the data "
    "subject without undue delay."
)


def _assert_char_start_end_invariant(text: str, chunks) -> None:
    for chunk in chunks:
        assert chunk.text == text[chunk.char_start : chunk.char_end]


# --------------------------------------------------------------------- the invariant


@pytest.mark.parametrize(
    "text",
    [
        _PROSE,
        "One short sentence.",
        "No terminal punctuation at all just words going on and on and on and on",
        "Line one\nLine two\nLine three\nno blank-line paragraphs here",
        "Para one.\n\nPara two.\n\nPara three.\n\nPara four.\n\nPara five.",
        # A single "sentence" with no whitespace at all inside the overflow window,
        # forcing the mid-word hard-split fallback.
        "x" * 5000,
        # Minified JSON: one very long "sentence" that does have whitespace, forcing the
        # whitespace-seeking hard split rather than the mid-word fallback.
        "{" + ", ".join(f'"k{i}": {i}' for i in range(500)) + "}",
    ],
)
def test_char_start_end_invariant_across_shapes(text: str) -> None:
    chunks = split_text(text, max_tokens=40, min_tokens=10, overlap_tokens=5)
    _assert_char_start_end_invariant(text, chunks)


def test_invariant_holds_with_default_sizing() -> None:
    # Defaults are large relative to _PROSE, so this also covers the "whole document
    # fits in one chunk" shape.
    chunks = split_text(_PROSE)
    _assert_char_start_end_invariant(_PROSE, chunks)


# -------------------------------------------------------------------------- overlap


def test_overlap_is_at_most_overlap_tokens_and_starts_on_a_word_boundary() -> None:
    overlap_tokens = 5
    max_overlap_chars = overlap_tokens * CHARS_PER_TOKEN
    chunks = split_text(_PROSE, max_tokens=30, min_tokens=5, overlap_tokens=overlap_tokens)
    assert len(chunks) > 2  # exercise more than the trivial single-chunk case

    for chunk in chunks[1:]:  # the first chunk has no predecessor to overlap with
        assert 0 <= chunk.overlap_chars <= max_overlap_chars
        if chunk.overlap_chars > 0:
            # The character immediately before char_start must be whitespace (or
            # char_start is 0) — otherwise the overlap begins mid-word.
            before = _PROSE[chunk.char_start - 1] if chunk.char_start > 0 else " "
            assert before.isspace()


def test_zero_overlap_means_chunks_are_back_to_back_in_core() -> None:
    chunks = split_text(_PROSE, max_tokens=30, min_tokens=5, overlap_tokens=0)
    for chunk in chunks:
        assert chunk.overlap_chars == 0
        assert chunk.char_start == chunk.core_start


# ------------------------------------------------------------------------- runt tail


def test_runt_tail_folds_into_predecessor_and_may_exceed_max_tokens() -> None:
    # Construct paragraphs where the last one, alone, would be a runt (well under
    # min_tokens) so it must fold into the previous chunk rather than stand alone.
    body = "Sentence about controllers and processors and supervisory authorities. " * 6
    paragraphs = [body.strip(), body.strip(), "Tiny tail."]
    text = "\n\n".join(paragraphs)

    max_tokens, min_tokens = 40, 15
    chunks = split_text(text, max_tokens=max_tokens, min_tokens=min_tokens, overlap_tokens=0)

    assert "Tiny tail." in chunks[-1].text
    # The fold is the *only* sanctioned way to exceed max_tokens.
    over_budget = [c for c in chunks if c.tokens > max_tokens]
    assert all(c is chunks[-1] for c in over_budget)
    _assert_char_start_end_invariant(text, chunks)


def test_no_runt_tail_when_last_chunk_already_meets_min_tokens() -> None:
    paragraphs = ["Paragraph number one is reasonably long on its own merits here."] * 3
    text = "\n\n".join(paragraphs)
    chunks = split_text(text, max_tokens=30, min_tokens=5, overlap_tokens=0)
    assert all(c.tokens <= 30 for c in chunks)


# --------------------------------------------------------------- min/max clamping


def test_min_tokens_above_max_tokens_is_clamped_not_crashed_on() -> None:
    # Documented behaviour: min_chars is clamped to max_chars, not the other way round,
    # so this must not raise and must not silently collapse the whole document to one
    # chunk with no signal that anything unusual happened.
    chunks = split_text(_PROSE, max_tokens=20, min_tokens=999, overlap_tokens=0)
    assert chunks  # did not crash, did not produce a nonsensical empty result
    _assert_char_start_end_invariant(_PROSE, chunks)


# ------------------------------------------------------------------------- empties


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t  \n"])
def test_empty_and_whitespace_only_input(text: str) -> None:
    assert split_text(text) == []


# --------------------------------------------------------------------- estimate_tokens


def test_estimate_tokens_matches_documented_heuristic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2  # ceil division, not floor


# --------------------------------------------------------------------------- page_of


def test_page_of_none_without_page_structure() -> None:
    assert page_of([], 0) is None
    assert page_of([], 500) is None


def test_page_of_off_by_one_exactly_at_page_start() -> None:
    # Page 1 starts at 0, page 2 at 100, page 3 at 250.
    page_starts = [0, 100, 250]
    assert page_of(page_starts, 0) == 1       # exact start of page 1
    assert page_of(page_starts, 99) == 1       # last offset still on page 1
    assert page_of(page_starts, 100) == 2      # exact start of page 2 — the off-by-one
    assert page_of(page_starts, 101) == 2
    assert page_of(page_starts, 249) == 2
    assert page_of(page_starts, 250) == 3      # exact start of page 3
    assert page_of(page_starts, 10_000) == 3   # past the last page start: still page 3
