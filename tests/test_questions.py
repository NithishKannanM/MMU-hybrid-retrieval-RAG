"""Tests for mmu.eval.questions: parsing, and the validate_questions hard gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmu.eval.questions import (
    Question,
    QuestionParseError,
    ValidationReport,
    load_questions,
    validate_questions,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _row(**overrides) -> dict:
    base = {
        "id": "q1",
        "question": "What is the notification deadline?",
        "answer_type": "single_document",
        "required_doc_ids": ["gdpr"],
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- parsing


def test_load_questions_minimal_row_applies_defaults(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "q.jsonl", [_row()])
    questions = load_questions(path)
    assert len(questions) == 1
    q = questions[0]
    assert q.id == "q1"
    assert q.required_doc_ids == ("gdpr",)
    # relevant_doc_ids defaults to required_doc_ids when omitted.
    assert q.relevant_doc_ids == ("gdpr",)
    assert q.must_contain == {}
    assert q.expected_synthesis_keys == ()
    assert q.reference_answer is None


def test_load_questions_full_row(tmp_path: Path) -> None:
    row = _row(
        id="xdr-001",
        answer_type="cross_document",
        required_doc_ids=["gdpr", "national-law", "company-policy"],
        relevant_doc_ids=["gdpr", "national-law", "company-policy", "recital-1"],
        must_contain={"gdpr": ["not later than 72 hours"]},
        expected_synthesis_keys=["72 hours", "supervisory authority"],
        reference_answer="some reference text",
        difficulty="hard",
        notes="seed question",
    )
    q = load_questions(_write_jsonl(tmp_path / "q.jsonl", [row]))[0]
    assert q.relevant_doc_ids == ("gdpr", "national-law", "company-policy", "recital-1")
    assert q.must_contain == {"gdpr": ("not later than 72 hours",)}
    assert q.expected_synthesis_keys == ("72 hours", "supervisory authority")
    assert q.reference_answer == "some reference text"
    assert q.is_cross_document is True


def test_load_questions_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(_row()) + "\n\n\n", encoding="utf-8")
    assert len(load_questions(path)) == 1


def test_load_questions_invalid_json_carries_line_number(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(_row(id="q1")) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(QuestionParseError) as exc_info:
        load_questions(path)
    assert "line 2" in str(exc_info.value)


def test_load_questions_missing_required_field_carries_line_number(tmp_path: Path) -> None:
    row = _row()
    del row["required_doc_ids"]
    path = _write_jsonl(tmp_path / "q.jsonl", [_row(id="ok"), row])
    with pytest.raises(QuestionParseError) as exc_info:
        load_questions(path)
    msg = str(exc_info.value)
    assert "line 2" in msg
    assert "required_doc_ids" in msg


def test_load_questions_bad_answer_type_raises(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "q.jsonl", [_row(answer_type="sideways")])
    with pytest.raises(QuestionParseError):
        load_questions(path)


def test_load_questions_duplicate_id_is_hard_error_with_both_lines(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "q.jsonl", [_row(id="dup"), _row(id="dup")])
    with pytest.raises(QuestionParseError) as exc_info:
        load_questions(path)
    msg = str(exc_info.value)
    assert "line 2" in msg
    assert "line 1" in msg
    assert "dup" in msg


# ------------------------------------------------------------------- is_cross_document


def test_is_cross_document_requires_at_least_two_required_docs() -> None:
    one_doc = Question(
        id="a", question="q", answer_type="single_document",
        required_doc_ids=("gdpr",), relevant_doc_ids=("gdpr",),
    )
    two_docs = Question(
        id="b", question="q", answer_type="cross_document",
        required_doc_ids=("gdpr", "national-law"), relevant_doc_ids=("gdpr", "national-law"),
    )
    assert one_doc.is_cross_document is False
    assert two_docs.is_cross_document is True


# --------------------------------------------------------------- validate_questions


DOC_TEXTS = {
    "gdpr": "The controller shall notify not later than 72 hours after becoming aware.",
    "national-law": "Notify the competent Supervisory Authority in the official language.",
    "company-policy": "Escalate any security incident within four hours.",
}
KNOWN_DOC_IDS = set(DOC_TEXTS)


def _q(**overrides) -> Question:
    base = dict(
        id="q1",
        question="text",
        answer_type="single_document",
        required_doc_ids=("gdpr",),
        relevant_doc_ids=("gdpr",),
        must_contain={},
        expected_synthesis_keys=(),
    )
    base.update(overrides)
    return Question(**base)


def test_validate_questions_clean_set_has_no_errors() -> None:
    questions = [
        _q(id="single", required_doc_ids=("gdpr",), relevant_doc_ids=("gdpr",)),
        _q(
            id="cross",
            answer_type="cross_document",
            required_doc_ids=("gdpr", "national-law"),
            relevant_doc_ids=("gdpr", "national-law"),
            expected_synthesis_keys=("72 hours",),
        ),
        _q(id="unanswerable", answer_type="unanswerable", required_doc_ids=(), relevant_doc_ids=()),
    ]
    report = validate_questions(questions, known_doc_ids=KNOWN_DOC_IDS, doc_texts=DOC_TEXTS)
    assert report.errors == ()
    assert report.ok is True
    assert "no 'unanswerable'" not in " ".join(report.warnings)


def test_validate_questions_unknown_required_doc_id_is_error() -> None:
    questions = [_q(required_doc_ids=("not-a-real-doc",), relevant_doc_ids=("not-a-real-doc",))]
    report = validate_questions(questions, known_doc_ids=KNOWN_DOC_IDS, doc_texts=DOC_TEXTS)
    assert not report.ok
    assert any("not-a-real-doc" in e for e in report.errors)


def test_validate_questions_unknown_relevant_doc_id_is_error() -> None:
    questions = [_q(required_doc_ids=("gdpr",), relevant_doc_ids=("gdpr", "ghost-doc"))]
    report = validate_questions(questions, known_doc_ids=KNOWN_DOC_IDS, doc_texts=DOC_TEXTS)
    assert not report.ok
    assert any("ghost-doc" in e for e in report.errors)


def test_validate_questions_must_contain_span_found_normalizing_whitespace() -> None:
    # The span in the question has different line-wrapping whitespace than the
    # "extracted text", mirroring a PDF reflow — normalization must still match it.
    questions = [_q(must_contain={"gdpr": ["notify\nnot later\tthan 72 hours"]})]
    report = validate_questions(questions, known_doc_ids=KNOWN_DOC_IDS, doc_texts=DOC_TEXTS)
    assert report.ok


def test_validate_questions_must_contain_span_missing_is_error() -> None:
    questions = [_q(must_contain={"gdpr": ["within 24 hours"]})]
    report = validate_questions(questions, known_doc_ids=KNOWN_DOC_IDS, doc_texts=DOC_TEXTS)
    assert not report.ok
    assert any("within 24 hours" in e for e in report.errors)


def test_validate_questions_must_contain_is_case_sensitive() -> None:
    # DOC_TEXTS["national-law"] has "Supervisory Authority" capitalized; the lowercase
    # form must NOT match, documenting that span matching does not case-fold (legal
    # defined terms are case-significant enough that folding would hide a real miss).
    questions = [_q(
        required_doc_ids=("national-law",),
        relevant_doc_ids=("national-law",),
        must_contain={"national-law": ["supervisory authority"]},
    )]
    report = validate_questions(questions, known_doc_ids=KNOWN_DOC_IDS, doc_texts=DOC_TEXTS)
    assert not report.ok


def test_validate_questions_cross_document_needs_two_required_docs() -> None:
    questions = [_q(answer_type="cross_document", required_doc_ids=("gdpr",), relevant_doc_ids=("gdpr",))]
    report = validate_questions(questions, known_doc_ids=KNOWN_DOC_IDS, doc_texts=DOC_TEXTS)
    assert not report.ok
    assert any("cross_document" in e for e in report.errors)


def test_validate_questions_duplicate_id_is_error() -> None:
    questions = [_q(id="dup"), _q(id="dup")]
    report = validate_questions(questions, known_doc_ids=KNOWN_DOC_IDS, doc_texts=DOC_TEXTS)
    assert not report.ok
    assert any("duplicate" in e for e in report.errors)


def test_validate_questions_warns_on_cross_document_without_synthesis_keys() -> None:
    questions = [_q(
        answer_type="cross_document",
        required_doc_ids=("gdpr", "national-law"),
        relevant_doc_ids=("gdpr", "national-law"),
        expected_synthesis_keys=(),
    )]
    report = validate_questions(questions, known_doc_ids=KNOWN_DOC_IDS, doc_texts=DOC_TEXTS)
    assert report.ok  # a warning, not an error
    assert any("expected_synthesis_keys" in w for w in report.warnings)


def test_validate_questions_warns_when_no_unanswerable_question_present() -> None:
    questions = [_q()]
    report = validate_questions(questions, known_doc_ids=KNOWN_DOC_IDS, doc_texts=DOC_TEXTS)
    assert any("unanswerable" in w for w in report.warnings)


def test_validate_questions_no_warning_when_unanswerable_present() -> None:
    questions = [_q(), _q(id="q2", answer_type="unanswerable", required_doc_ids=(), relevant_doc_ids=())]
    report = validate_questions(questions, known_doc_ids=KNOWN_DOC_IDS, doc_texts=DOC_TEXTS)
    assert not any("no 'unanswerable'" in w for w in report.warnings)


def test_validation_report_ok_reflects_errors_only() -> None:
    assert ValidationReport(errors=(), warnings=("w",)).ok is True
    assert ValidationReport(errors=("e",), warnings=()).ok is False
