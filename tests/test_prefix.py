"""The asymmetry suite: prefix policy correctness plus the structural guarantee.

Downloads nothing, makes no network call — the real BGE-M3 model is never
instantiated outside the ``live``-marked test at the bottom, which is deselected by
the default ``addopts = "-m 'not live'"``.
"""

from __future__ import annotations

import pytest

from mmu.embed.hashing import HashingEmbedder
from mmu.embed.prefix import POLICIES, PrefixPolicy, UnknownEmbeddingModel, policy_for
from tests.conftest import RecordingEmbedder


def test_bge_m3_has_no_retrieval_prefix() -> None:
    # This is the spec risk, asserted as a decision rather than a trivial default: the
    # original spec claimed BGE-M3 needs an asymmetric query instruction like
    # bge-large-en-v1.5 does. That is false — BAAI's guidance and FlagEmbedding's own
    # BGEM3FlagModel default (query_instruction_for_retrieval=None) agree BGE-M3 takes
    # no retrieval instruction. If this assertion ever fails because someone "fixed" the
    # registry to add a query prefix, that is the regression this test exists to catch.
    policy = policy_for("BAAI/bge-m3")
    assert policy.query == ""
    assert policy.document == ""


def test_bge_large_v15_prefixes_queries_only() -> None:
    policy = policy_for("BAAI/bge-large-en-v1.5")
    assert policy.query == "Represent this sentence for searching relevant passages: "
    assert policy.document == ""

    text = "what is the notification deadline?"
    assert policy.apply(text, "query") == policy.query + text
    # apply() leaves documents untouched for this model.
    assert policy.apply(text, "document") == text


def test_unregistered_model_raises_with_actionable_message() -> None:
    with pytest.raises(UnknownEmbeddingModel) as exc_info:
        policy_for("some/unregistered-model")
    message = str(exc_info.value)
    assert "some/unregistered-model" in message
    # The error must name known models so the fix (register it) is discoverable
    # without reading source.
    assert "BAAI/bge-m3" in message


def test_hashing_test_embedder_is_registered() -> None:
    # The test double must resolve through the same registry path as a real model, with
    # empty/empty prefixes, so retrieval tests built on HashingEmbedder don't need a
    # special case.
    assert POLICIES["hashing-test-embedder"] == PrefixPolicy(query="", document="")


def test_policy_for_never_defaults_silently() -> None:
    # policy_for must not fall back to an empty PrefixPolicy for anything not in the
    # registry — a wrong (including missing) prefix is worse than a loud failure. This
    # is a `type`-level check that UnknownEmbeddingModel is a KeyError subclass, since
    # callers may reasonably catch KeyError generically.
    assert issubclass(UnknownEmbeddingModel, KeyError)


@pytest.mark.parametrize("kind", ["query", "document"])
def test_apply_round_trips_and_does_not_mutate_input(kind: str) -> None:
    policy = PrefixPolicy(query="Q: ", document="D: ")
    original = "the controller must notify within 72 hours"
    result = policy.apply(original, kind)  # type: ignore[arg-type]
    expected_prefix = policy.query if kind == "query" else policy.document
    assert result == expected_prefix + original
    # str is immutable, but assert explicitly that the input object itself is unchanged
    # (no accidental in-place buffer trickery, no leaking of prefix into the source).
    assert original == "the controller must notify within 72 hours"


def test_encode_without_kind_raises_type_error() -> None:
    # The structural guarantee this whole package exists to provide: `kind` is a
    # required keyword-only argument with no default on Embedder.encode. Omitting it
    # must be a TypeError at the call site — a stronger guarantee than "remember the
    # prefix", because it converts what would otherwise be a silent ranking regression
    # (querying and indexing with mismatched or missing prefix treatment) into a loud,
    # immediate failure that any test or type checker catches.
    embedder = HashingEmbedder()
    with pytest.raises(TypeError):
        embedder.encode(["x"])  # type: ignore[call-arg]


def test_recording_embedder_sees_different_kinds_for_query_and_document() -> None:
    # Proves the actual bug this module exists to prevent is *observable*: a caller
    # that encodes the document side and the query side must pass different `kind`
    # values, not just "call encode() twice". RecordingEmbedder (tests/conftest.py)
    # records every (texts, kind) call so this is assertable without a real index.
    #
    # This test proves the recording mechanism and the kind distinction. The real
    # indexer/searcher wiring is pinned by
    # `test_indexer_and_searcher_encode_with_opposite_kinds` below.
    recorder = RecordingEmbedder()

    recorder.encode(["a legal clause about data breaches"], kind="document")
    recorder.encode(["what is the notification deadline?"], kind="query")

    kinds = recorder.kinds()
    assert kinds == ["document", "query"]
    assert len(set(kinds)) == 2  # a caller that encodes both sides must vary kind


@pytest.mark.live
def test_bge_m3_real_model_dim_and_ranking() -> None:
    """Needs the real BGE-M3 (~2.3GB download). Deselected by default (-m 'not live')."""
    import numpy as np

    from mmu.embed.bge_m3 import BgeM3Embedder

    embedder = BgeM3Embedder()
    assert embedder.dim == 1024

    query = "How quickly must a data controller notify a supervisory authority of a personal data breach?"
    relevant_doc = (
        "The controller shall without undue delay and, where feasible, not later than "
        "72 hours after having become aware of a personal data breach, notify the "
        "breach to the competent supervisory authority."
    )
    irrelevant_doc = (
        "The office cafeteria will be closed for renovations starting next Monday; "
        "employees should use the third-floor break room instead."
    )

    encoded = embedder.encode([query, relevant_doc, irrelevant_doc], kind="query")
    dense = encoded.dense
    norms = np.linalg.norm(dense, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)  # unit-norm output

    q_vec = embedder.encode([query], kind="query").dense[0]
    rel_vec = embedder.encode([relevant_doc], kind="document").dense[0]
    irr_vec = embedder.encode([irrelevant_doc], kind="document").dense[0]

    rel_score = float(np.dot(q_vec, rel_vec))
    irr_score = float(np.dot(q_vec, irr_vec))
    assert rel_score > irr_score


def test_indexer_and_searcher_encode_with_opposite_kinds(tmp_path) -> None:
    """The round-trip this whole module exists to protect.

    The bug is not "the prefix function is wrong" — that is easy to see and easy to
    test. The bug is that the *indexer* and the *searcher* encoded with the same
    ``kind``, so documents and queries land in mismatched spaces. Nothing raises. No
    log line appears. Retrieval just gets quietly worse, and the only symptom is a
    benchmark number that is lower than it should be for reasons nobody can locate.

    So this asserts the wiring rather than the function: build the index through the
    real ``build_index``, run the real ``HybridRetriever.search``, and check what the
    embedder was actually asked for at each call site.
    """
    from mmu.retrieve.build import build_index
    from mmu.retrieve.hybrid import HybridRetriever

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "gdpr.txt").write_text(
        "The controller shall notify the supervisory authority within 72 hours.",
        encoding="utf-8",
    )

    recorder = RecordingEmbedder(inner=HashingEmbedder(dim=32))
    store, dense, sparse, _ = build_index(
        corpus, tmp_path / ".index", recorder, max_tokens=64, min_tokens=8, overlap_tokens=8
    )
    # Indexing encoded the document side, and ONLY the document side.
    assert recorder.kinds() == ["document"], recorder.kinds()

    HybridRetriever(store, dense, sparse, recorder).search("notification deadline", k=3)
    # Searching added exactly one query-side encode on top.
    assert recorder.kinds() == ["document", "query"], recorder.kinds()

    # And the two call sites are genuinely distinct — if someone "simplifies" either to
    # share a single kind, this is the assertion that fires.
    assert recorder.calls[0][1] == "document"
    assert recorder.calls[-1][1] == "query"
