"""Invariants of the seed question set itself, checked against the real corpus.

These tests exist because `data/questions.jsonl` is *data that metrics are computed
from*, so a mistake in it does not raise -- it silently moves a score. The question
set went from 6 to 30 entries, and nothing in the suite would have noticed if half of
them had quietly stopped contributing to a denominator.

The expensive fixture here is chunking 500 KB of legal text with the production
splitter. It is done once per session and reused, which keeps the whole module well
inside the offline suite's time budget.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mmu.eval.metrics import paired_sign_test
from mmu.eval.questions import load_questions, validate_questions
from mmu.ingest.splitter import split_text

_CORPUS = Path("data/corpus")
_QUESTIONS = Path("data/questions.jsonl")
_DOC_IDS = ("gdpr", "national-law", "company-policy")


def _norm(s: str) -> str:
    """The same whitespace collapse `questions._normalize_ws` and `metrics` apply."""
    return re.sub(r"\s+", " ", s).strip()


@pytest.fixture(scope="module")
def doc_texts() -> dict[str, str]:
    return {d: (_CORPUS / f"{d}.txt").read_text(encoding="utf-8") for d in _DOC_IDS}


@pytest.fixture(scope="module")
def questions():
    return load_questions(_QUESTIONS)


@pytest.fixture(scope="module")
def normalized_chunks(doc_texts) -> dict[str, list[str]]:
    """Every document split by the *production* splitter at its default settings.

    Deliberately not read from `data/.index/chunks.jsonl`: that file is gitignored, so
    a test depending on it would pass on this machine and fail on a fresh clone.
    """
    return {
        doc_id: [_norm(span.text) for span in split_text(text)]
        for doc_id, text in doc_texts.items()
    }


def test_the_question_file_validates_against_the_real_corpus(questions, doc_texts):
    report = validate_questions(
        questions, known_doc_ids=set(_DOC_IDS), doc_texts=doc_texts
    )
    assert report.errors == ()
    assert report.warnings == ()


def test_every_must_contain_span_fits_inside_a_single_chunk(questions, normalized_chunks):
    """The gap `validate_questions` leaves open, and the reason this module exists.

    `validate_questions` checks each `must_contain` span appears in the *document*.
    But `metrics._chunk_is_relevant` asks whether the span appears in *one specific
    chunk*, so a span straddling a chunk boundary is present in the document and
    present in no chunk at all. Nothing would raise: context precision would simply
    compute over zero relevant chunks and report a confidently wrong number, and the
    rank-aware metrics would have no relevant item to rank.

    A span that fails here is fixed by shortening it, not by retuning the splitter.
    """
    orphans = [
        (q.id, doc_id, span)
        for q in questions
        for doc_id, spans in q.must_contain.items()
        for span in spans
        if not any(_norm(span) in chunk for chunk in normalized_chunks[doc_id])
    ]
    assert orphans == []


def test_each_metric_still_has_a_denominator(questions):
    """Each corpus metric averages over the questions that define it. If any of these
    cohorts empties out, the corresponding column reports `None` (or vanishes) and the
    ablation stops being able to see that dimension at all."""
    cross_doc = [q for q in questions if q.is_cross_document]
    with_relevant_set = [q for q in questions if q.relevant_doc_ids]
    unanswerable = [q for q in questions if q.answer_type == "unanswerable"]

    assert len(cross_doc) >= 10, "XDR/Cov/Cite/Syn average over cross-document questions"
    assert len(with_relevant_set) >= 20, "CP averages over questions with a relevant set"
    assert len(unanswerable) >= 3, "AF's refusal_is_correct path needs refusable questions"


def test_every_document_is_required_by_some_question(questions):
    """A document no question requires is dead weight in the index: a retrieval arm
    could fail on it completely and no metric would move. `national-law` is the one at
    risk, being the document that appears mostly inside cross-document questions."""
    required = {d for q in questions for d in q.required_doc_ids}
    assert required == set(_DOC_IDS)


def test_every_document_is_the_sole_subject_of_some_question(questions):
    """Stronger than the above: each document needs at least one question answerable
    from *it alone*, so that a single-document retrieval failure is visible directly
    instead of only as a diluted cross-document score."""
    sole = {q.required_doc_ids[0] for q in questions if len(q.required_doc_ids) == 1}
    assert sole == set(_DOC_IDS)


def test_unanswerable_questions_require_and_label_nothing(questions):
    """A refusable question with a non-empty required set would be scored as a
    retrieval failure instead of a correct abstention -- the three-way rule in
    `answer_faithfulness` keys off exactly these fields being empty."""
    for q in questions:
        if q.answer_type == "unanswerable":
            assert q.required_doc_ids == (), q.id
            assert q.relevant_doc_ids == (), q.id
            assert q.must_contain == {}, q.id


def test_the_corpus_is_large_enough_for_a_realistic_effect_to_be_detectable(questions):
    """The reason the set was expanded, expressed as a test.

    At n=6 the sign test could not report significance for anything short of unanimity
    (best attainable p = 0.031), and a reranker winning 70% of questions would have
    reported p=0.69. This asserts the floor that made the expansion worth doing, so
    that deleting questions back down to a decorative handful fails loudly.
    """
    n = len(questions)
    assert paired_sign_test([1.0] * n).best_attainable_p < 1e-6

    wins = round(n * 0.7)
    realistic = paired_sign_test([1.0] * wins + [-1.0] * (n - wins))
    assert realistic.p_two_sided < 0.05, (
        f"a 70/30 effect over {n} questions reports p={realistic.p_two_sided:.3f}; "
        "the corpus is too small to detect an effect worth shipping"
    )
