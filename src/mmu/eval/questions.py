"""Parse and validate ``data/questions.jsonl`` — the eval harness's ground truth.

A :class:`Question` is the join point between three things built independently: the
corpus on disk (via ``doc_id``), the metrics in :mod:`mmu.eval.metrics` (via
``required_doc_ids``/``relevant_doc_ids``/``must_contain``), and a person's judgment
about what a correct answer looks like. Nothing here imports a retriever, a generator,
or a judge — this module's only job is turning JSON lines into typed, checked data.

**``reference_answer`` is used by no metric.** All five metrics (AF, CP, XDR, LpQU,
CUR) are reference-free by design: they score an answer against the retrieved *context*
and the *question's structure* (required docs, must-contain spans, synthesis keys), not
against a gold answer string. This is deliberate — a reference-answer metric like
ROUGE/BLEU rewards paraphrase overlap with one hand-written answer, which is a proxy for
grounding at best and actively misleading at worst (a well-grounded answer phrased
differently scores low; a fluent hallucination that happens to echo the reference's
wording scores high). ``reference_answer`` is kept in the schema purely as a human
sanity-check a report author can eyeball — it is intentionally never threaded into
:mod:`mmu.eval.metrics`, and nobody should add a metric that reads it without revisiting
this decision explicitly.

**Why ``validate_questions`` is a hard gate, not a lint pass.** Every error class here
maps to a failure mode that is *indistinguishable from a model failure* if left
unchecked. The worst offender: a typo in ``required_doc_ids`` (a doc_id that does not
exist in the built index) makes ``Cov`` read 0 for every retrieval, which makes XDR read
0.0 for that question — and a 0.0 XDR looks exactly like "the retriever and generator
failed to reason across documents." Someone then spends an hour staring at retrieval
logs and generation prompts for a bug that is actually a one-character typo in
``questions.jsonl``. Running this gate first (``mmu questions validate``) converts that
hour into an immediate, specific error message.

The ``must_contain`` span check exists for the same reason at one remove: a span
hand-copied from a rendered PDF can differ from the PDF's *extracted* text layer —
ligatures (``fi`` becomes one glyph), soft hyphens, or a multi-column layout reflowed
in the wrong reading order. A span that looks identical on screen may simply not be a
substring of ``doc_texts[doc_id]``, and without this check that silently zeroes CP for
every chunk of that document (see :func:`mmu.eval.metrics.context_precision`) rather
than raising at validation time.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

AnswerType = Literal["cross_document", "single_document", "unanswerable"]

_ANSWER_TYPES: frozenset[str] = frozenset({"cross_document", "single_document", "unanswerable"})

_REQUIRED_FIELDS: tuple[str, ...] = ("id", "question", "answer_type", "required_doc_ids")


class QuestionParseError(ValueError):
    """A malformed line in ``questions.jsonl``. Always carries the 1-based line number
    — the whole point of parsing line-by-line rather than ``json.load``-ing the file as
    one blob is that a parse failure must be locatable without a binary search."""


def _normalize_ws(s: str) -> str:
    """Collapse all whitespace runs (including newlines) to single spaces.

    **Deliberately not case-folded.** Legal defined terms are case-significant enough
    (e.g. a capitalized "Controller" as a defined term vs. lowercase "controller" used
    generically) that a case-insensitive match would hide a real mismatch between a
    ``must_contain`` span and the document's actual text — exactly the class of bug this
    check exists to catch. Whitespace, by contrast, carries no legal meaning and differs
    for uninteresting reasons (PDF line wrapping, column reflow), so it is safe to fold.
    """
    return " ".join(s.split())


@dataclass(frozen=True)
class Question:
    """One eval seed. See the module docstring for why ``reference_answer`` is inert.

    ``must_contain`` maps a ``doc_id`` (which must be one of this question's
    ``required_doc_ids`` or ``relevant_doc_ids``) to a list of verbatim spans that must
    appear in that document's extracted text. It narrows :func:`context_precision`'s
    doc-level relevance to chunk-level relevance — see that function's docstring.
    """

    id: str
    question: str
    answer_type: AnswerType
    required_doc_ids: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]
    must_contain: dict[str, tuple[str, ...]] = field(default_factory=dict)
    expected_synthesis_keys: tuple[str, ...] = field(default_factory=tuple)
    #: Human sanity-check only. See module docstring — used by no metric.
    reference_answer: str | None = None
    difficulty: str | None = None
    notes: str | None = None

    @property
    def is_cross_document(self) -> bool:
        """Only questions with >=2 required docs count toward XDR's corpus average.

        ``len(required_doc_ids) >= 2`` rather than ``answer_type == "cross_document"``:
        the field is the operative contract (it is literally XDR's denominator), and
        ``validate_questions`` separately enforces that the two never disagree.
        """
        return len(self.required_doc_ids) >= 2


def _parse_line(raw: dict, lineno: int) -> Question:
    for name in _REQUIRED_FIELDS:
        if name not in raw or raw[name] in (None, ""):
            raise QuestionParseError(f"questions.jsonl line {lineno}: missing required field {name!r}")

    qid = raw["id"]
    if not isinstance(qid, str):
        raise QuestionParseError(f"questions.jsonl line {lineno}: 'id' must be a string")

    answer_type = raw["answer_type"]
    if answer_type not in _ANSWER_TYPES:
        raise QuestionParseError(
            f"questions.jsonl line {lineno}: 'answer_type' must be one of "
            f"{sorted(_ANSWER_TYPES)}, got {answer_type!r}"
        )

    required_raw = raw["required_doc_ids"]
    if not isinstance(required_raw, list) or not all(isinstance(d, str) for d in required_raw):
        raise QuestionParseError(
            f"questions.jsonl line {lineno}: 'required_doc_ids' must be a list of strings"
        )
    required_doc_ids = tuple(required_raw)

    relevant_raw = raw.get("relevant_doc_ids", list(required_doc_ids))
    if not isinstance(relevant_raw, list) or not all(isinstance(d, str) for d in relevant_raw):
        raise QuestionParseError(
            f"questions.jsonl line {lineno}: 'relevant_doc_ids' must be a list of strings"
        )
    relevant_doc_ids = tuple(relevant_raw)

    must_contain_raw = raw.get("must_contain", {})
    if not isinstance(must_contain_raw, dict):
        raise QuestionParseError(f"questions.jsonl line {lineno}: 'must_contain' must be an object")
    must_contain: dict[str, tuple[str, ...]] = {}
    for doc_id, spans in must_contain_raw.items():
        if not isinstance(spans, list) or not all(isinstance(s, str) for s in spans):
            raise QuestionParseError(
                f"questions.jsonl line {lineno}: 'must_contain[{doc_id!r}]' must be a list of strings"
            )
        must_contain[doc_id] = tuple(spans)

    keys_raw = raw.get("expected_synthesis_keys", [])
    if not isinstance(keys_raw, list) or not all(isinstance(s, str) for s in keys_raw):
        raise QuestionParseError(
            f"questions.jsonl line {lineno}: 'expected_synthesis_keys' must be a list of strings"
        )

    return Question(
        id=qid,
        question=raw["question"],
        answer_type=answer_type,
        required_doc_ids=required_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        must_contain=must_contain,
        expected_synthesis_keys=tuple(keys_raw),
        reference_answer=raw.get("reference_answer"),
        difficulty=raw.get("difficulty"),
        notes=raw.get("notes"),
    )


def load_questions(path: Path) -> list[Question]:
    """Parse ``path`` (one JSON object per line). Blank lines are skipped.

    A duplicate ``id`` is a hard error raised here, not deferred to
    :func:`validate_questions` — an eval run addresses questions by id, and a silent
    duplicate would make half the report attribute to the wrong seed. The error message
    carries both line numbers.
    """
    questions: list[Question] = []
    seen_at: dict[str, int] = {}
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QuestionParseError(f"questions.jsonl line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise QuestionParseError(f"questions.jsonl line {lineno}: expected a JSON object")

        question = _parse_line(raw, lineno)
        if question.id in seen_at:
            raise QuestionParseError(
                f"questions.jsonl line {lineno}: duplicate id {question.id!r} "
                f"(first seen at line {seen_at[question.id]})"
            )
        seen_at[question.id] = lineno
        questions.append(question)
    return questions


@dataclass(frozen=True)
class ValidationReport:
    """The hard-gate result. ``errors`` must be empty before an eval run proceeds;
    ``warnings`` are printed but never block — see module docstring for why errors are
    hard and the specific failure each one prevents."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def validate_questions(
    questions: Sequence[Question],
    *,
    known_doc_ids: Collection[str],
    doc_texts: Mapping[str, str],
) -> ValidationReport:
    """Validate a set of questions against a built index. See module docstring for why
    this is a hard gate rather than a warning pass.

    ``known_doc_ids`` is the built index's doc_id set (e.g. ``ChunkStore.doc_ids()``).
    ``doc_texts`` maps doc_id to that document's full extracted text (e.g.
    ``ChunkStore.text_of_doc``), used only for the ``must_contain`` verbatim check.

    Duplicate ids are checked again here (independently of :func:`load_questions`,
    which already refuses to produce a duplicate-id list) because this function must be
    a correct hard gate for *any* list of :class:`Question`, including one assembled
    programmatically rather than loaded from disk.
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_at: dict[str, int] = {}
    any_unanswerable = False

    for i, q in enumerate(questions):
        if q.id in seen_at:
            errors.append(f"{q.id!r}: duplicate question id (also at position {seen_at[q.id]})")
        else:
            seen_at[q.id] = i

        if q.answer_type == "unanswerable":
            any_unanswerable = True

        if q.answer_type == "cross_document" and len(q.required_doc_ids) < 2:
            errors.append(
                f"{q.id!r}: answer_type='cross_document' requires >=2 required_doc_ids, "
                f"got {len(q.required_doc_ids)}"
            )

        for doc_id in q.required_doc_ids:
            if doc_id not in known_doc_ids:
                errors.append(
                    f"{q.id!r}: required_doc_ids contains unknown doc_id {doc_id!r} "
                    f"(not in the built index)"
                )
        for doc_id in q.relevant_doc_ids:
            if doc_id not in known_doc_ids:
                errors.append(
                    f"{q.id!r}: relevant_doc_ids contains unknown doc_id {doc_id!r} "
                    f"(not in the built index)"
                )

        for doc_id, spans in q.must_contain.items():
            doc_text = doc_texts.get(doc_id)
            if doc_text is None:
                errors.append(
                    f"{q.id!r}: must_contain references unknown doc_id {doc_id!r} "
                    f"(not in the built index)"
                )
                continue
            norm_doc = _normalize_ws(doc_text)
            for span in spans:
                if _normalize_ws(span) not in norm_doc:
                    errors.append(
                        f"{q.id!r}: must_contain span not found verbatim in {doc_id!r}: "
                        f"{span!r} (check for PDF ligatures, soft hyphens, or column "
                        f"reflow between the source and the extracted text)"
                    )

        if q.is_cross_document and not q.expected_synthesis_keys:
            warnings.append(
                f"{q.id!r}: cross_document question has no expected_synthesis_keys "
                f"(the offline Syn fallback will always score 0 for it)"
            )

    if not any_unanswerable:
        warnings.append("no 'unanswerable' question in the file")

    return ValidationReport(errors=tuple(errors), warnings=tuple(warnings))
