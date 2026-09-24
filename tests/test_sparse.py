"""BM25: the tokenizer is the channel, so most of this file tests the tokenizer."""

from __future__ import annotations

from mmu.retrieve.sparse import Bm25Index, tokenize


def test_exact_defined_term_beats_the_semantic_neighbour() -> None:
    """The reason a lexical channel is in the design at all.

    'data controller' and 'data processor' are semantically adjacent and legally
    opposite. A dense embedder will happily rank the wrong one first; BM25 scores the
    literal term, so the chunk that actually contains it surfaces.
    """
    texts = [
        "the data processor shall assist the entity that determines the purposes",
        "the data controller determines the purposes and means of processing",
        "cooking beans slowly improves their texture",
    ]
    index = Bm25Index(texts)
    hits = index.search("data controller", k=3)
    assert hits[0][0] == 1


def test_numeric_and_hyphenated_forms_stay_whole() -> None:
    """The tokenizer rule that actually matters: no-space forms must not fragment.

    "Art. 33" legitimately tokenizes as two terms — the period there is sentence
    punctuation followed by a space, and both `art` and `33` still match. The rule earns
    its keep on `13.2` and `data-controller`, which would otherwise split into the very
    words they are defined against.
    """
    tokens = tokenize("See Art. 33 and section 13.2 for the notification duty.")
    assert "art" in tokens and "33" in tokens
    assert "13.2" in tokens, "a section number must not fragment into 13 and 2"
    # The sentence-final period is not internal, so it is not glued on.
    assert "duty" in tokens


def test_hyphenated_defined_terms_stay_whole() -> None:
    tokens = tokenize("A data-controller is not a data processor.")
    assert "data-controller" in tokens
    assert "data" in tokens and "processor" in tokens


def test_case_is_folded() -> None:
    assert tokenize("Controller") == tokenize("controller") == ["controller"]


def test_zero_scoring_rows_are_dropped() -> None:
    """A zero score means the query contributed no evidence for that row.

    Returning it would hand RRF a rank position implying evidence the channel does not
    have — the same mistake as the rejected `n+1` sentinel in the fusion step.
    """
    index = Bm25Index(["alpha beta", "gamma delta", "gamma epsilon", "delta zeta"])
    hits = index.search("alpha", k=5)
    assert [row for row, _ in hits] == [0]


def test_idf_floor_at_small_n_is_a_known_property() -> None:
    """BM25Okapi gives IDF exactly 0.0 to a term in >=half the corpus.

    Documented as a test rather than left to be rediscovered: such a term scores 0 in
    every row, so the channel returns nothing for a query made only of ubiquitous words.
    That is correct — those terms carry no discriminative lexical signal and dense
    retrieval is what should answer — but it surprises people at small N. This channel
    indexes chunks, not documents, so N is in the hundreds in practice.
    """
    half = Bm25Index(["alpha beta", "alpha gamma", "delta", "epsilon"])
    assert half.search("alpha", k=5) == []

    # One document fewer containing the term, and it discriminates again.
    fewer = Bm25Index(["alpha beta", "gamma", "delta", "epsilon"])
    assert [row for row, _ in fewer.search("alpha", k=5)] == [0]


def test_empty_query_and_empty_corpus() -> None:
    assert Bm25Index(["alpha"]).search("", k=5) == []
    assert Bm25Index(["alpha"]).search("!!! ???", k=5) == []
    assert Bm25Index([]).search("alpha", k=5) == []
    # A corpus of pure punctuation must not divide by a zero average length.
    assert Bm25Index(["!!!", "???"]).search("alpha", k=5) == []
