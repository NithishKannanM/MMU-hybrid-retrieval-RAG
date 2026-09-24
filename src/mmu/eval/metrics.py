"""The five metrics, as pure functions over data — no I/O, no judge calls inside.

Every function here takes judge verdicts, citations and timings *as arguments*. That is
what makes this module's own tests run in milliseconds with no model and no network:
the interesting, arguable part of this benchmark is the arithmetic (what counts as
"relevant", how a rank-1 hit differs from a rank-5 hit, why an aggregate ratio and not a
mean of ratios), and that arithmetic can and should be exercised with hand-built
fixtures. :mod:`mmu.eval.judge` and (a later work package) ``eval/runner.py`` are the
only places that call a model; this module never imports either.

**Why :class:`ClaimVerification` and not :class:`mmu.eval.judge.Judgment`.** AF and CUR
both need "was this claim supported, and if so by which chunk and which span" — but
that is a *result*, not a judge call. Defining a small local record for it here (rather
than importing ``judge.Judgment``) keeps the dependency one-directional: ``judge.py`` is
free to import from here (it reuses :func:`synthesis_score_fallback`), and this module
never has to know a Judge Protocol, a Generator, or rapidfuzz's judge-side thresholds
exist. ``eval/runner.py`` is the adapter that turns a stream of ``Judgment`` objects
into a list of these.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from rapidfuzz import fuzz

from mmu.core.models import RetrievedChunk

# ----------------------------------------------------------------------------- helpers


def _normalize_ws(s: str) -> str:
    """Collapse whitespace runs to single spaces. Duplicated (not imported) from
    :mod:`mmu.eval.questions` on purpose: that module owns *schema* validation, this one
    owns *metric* arithmetic, and a one-line helper is cheaper to repeat than to couple
    two otherwise-independent modules over."""
    return " ".join(s.split())


@dataclass(frozen=True)
class ClaimVerification:
    """One atomic claim from an answer, verified against a specific retrieved chunk.

    ``supported`` is the judge's verdict *after* span-invention downgrading has already
    happened (see :mod:`mmu.eval.judge`) — by the time this reaches AF or CUR math, a
    "supported" claim is guaranteed to have a judge-provided ``span`` that really is a
    substring of the chunk it names, or it is not supported at all.
    """

    claim: str
    supported: bool
    chunk_id: str | None = None
    span: str | None = None


# --------------------------------------------------------------- 1. Answer Faithfulness


def answer_faithfulness(claims: Sequence[ClaimVerification], *, is_refusal: bool) -> float:
    """``AF = supported_claims / m`` over the atomic claims decomposed from an answer.

    ``m == 0`` has exactly two possible causes and they must score oppositely:
    the model correctly refused an unanswerable question (no claims to make is the
    *correct* output) -> **1.0**; or the model produced an empty/degenerate answer that
    happens to decompose into zero claims -> **0.0**. Scoring the latter as 1.0 would be
    vacuous truth — an AF metric that rewards saying nothing is trivially gameable by a
    model that just stops generating, which defeats the entire point of measuring
    faithfulness. ``is_refusal`` must therefore be supplied by the caller (it is a
    property of the answer text, e.g. ``generate/prompt.py``'s refusal detector — not of
    the claims, since there are none to inspect).
    """
    m = len(claims)
    if m == 0:
        return 1.0 if is_refusal else 0.0
    supported = sum(1 for c in claims if c.supported)
    return supported / m


def corpus_answer_faithfulness(per_question_af: Sequence[float]) -> float:
    """Macro-mean over questions: every question's AF counts once, regardless of how
    many claims it decomposed into.

    Rejected alternative: a *micro*-mean (``total_supported / total_claims`` across the
    whole corpus) would let one verbose, heavily-decomposed answer dominate the corpus
    number — a question whose answer happens to produce 20 claims counts 20x as much as
    one that produces 1, even though both are one seed question worth one vote on "is
    this system faithful." ``test_metrics.py`` builds a fixture where the two diverge.
    """
    if not per_question_af:
        return 0.0
    return sum(per_question_af) / len(per_question_af)


# ----------------------------------------------------------------- 2. Context Precision


def _chunk_is_relevant(
    doc_id: str,
    text: str,
    relevant_doc_ids: Collection[str],
    must_contain: Mapping[str, Sequence[str]],
) -> bool:
    """``rel(c_i) = 1`` iff ``c_i.doc_id`` is relevant AND (no must_contain entry for
    that doc, or one of its spans appears in this specific chunk).

    Doc-level relevance alone is too loose: an entire multi-page GDPR PDF is "relevant"
    to a GDPR question, which would score CP=1.0 for eight retrieved chunks that are all
    from the GDPR document but none of which contain the actual notification-deadline
    clause the question is about. ``must_contain`` narrows relevance from "this
    document" to "this specific passage", which is what context precision is supposed
    to be measuring.
    """
    if doc_id not in relevant_doc_ids:
        return False
    spans = must_contain.get(doc_id)
    if not spans:
        return True
    norm_text = _normalize_ws(text)
    return any(_normalize_ws(span) in norm_text for span in spans)


def context_precision(
    chunks: Sequence[RetrievedChunk],
    *,
    relevant_doc_ids: Collection[str],
    must_contain: Mapping[str, Sequence[str]] | None = None,
) -> float:
    """Rank-weighted average precision over the fused context ``C = [c_1..c_k]``.

    ``CP = sum_i rel(c_i) * P@i / sum_i rel(c_i)``, ``0.0`` when nothing relevant was
    retrieved. This is deliberately *not* a flat hit-rate (``relevant_count / k``): a
    relevant chunk at rank 1 survives context truncation (and is the first thing the
    generator reads); one at rank 8 may be cut entirely if ``k`` shrinks or the prompt
    is truncated. A flat hit-rate scores "1 relevant at rank 1" identically to "1
    relevant at rank 8", throwing away exactly the ordering information retrieval exists
    to produce.
    """
    must_contain = must_contain or {}
    rel_flags = [
        _chunk_is_relevant(c.doc_id, c.text, relevant_doc_ids, must_contain) for c in chunks
    ]
    total_relevant = sum(rel_flags)
    if total_relevant == 0:
        return 0.0
    hits = 0
    running_sum = 0.0
    for i, rel in enumerate(rel_flags, start=1):
        if rel:
            hits += 1
            running_sum += hits / i  # P@i
    return running_sum / total_relevant


# ------------------------------------------------------------ 3. Cross-Document Reasoning


@dataclass(frozen=True)
class XDRResult:
    """All four numbers, always — never just the product.

    ``XDR = Cov * Cite * Syn`` collapses three independent failure modes into one
    number, and each points at a *different* subsystem: a low ``Cov`` (required docs
    missing from the retrieved context) is a retrieval bug — fix the chunker or the
    sparse channel. A low ``Cite`` with ``Cov=1.0`` (the model had every required
    document in context but only cited one) is a generation bug — the prompt or model
    isn't using what it was given. Fixing the wrong one wastes a debugging session, and
    the product alone cannot tell you which of the two happened, so callers must report
    all four fields, never collapse to ``.xdr`` alone.
    """

    coverage: float
    cite: float
    synthesis: float
    xdr: float
    n_required: int


def cross_document_reasoning(
    *,
    required_doc_ids: Sequence[str],
    retrieved_doc_ids: Collection[str],
    cited_doc_ids: Collection[str],
    synthesis_score: float,
) -> XDRResult:
    """``Cov`` = fraction of ``required_doc_ids`` present in the retrieved context.
    ``Cite`` = fraction cited in the answer — deterministic, resolved from ``[Sn]``
    markers upstream (``generate/prompt.py``), not judged. ``synthesis_score`` (``Syn``)
    is the one genuinely-LLM factor (or its deterministic fallback, see
    :func:`synthesis_score_fallback`) and must already be one of ``{0, 0.5, 1}``.

    Only defined for cross-document questions (``len(required_doc_ids) >= 2``); a
    single-document question has no XDR score and is excluded from the corpus average
    by the caller (see :func:`corpus_xdr`), not by returning a degenerate value here.
    """
    if len(required_doc_ids) < 2:
        raise ValueError(
            "cross_document_reasoning is only defined for questions with "
            f">=2 required_doc_ids, got {len(required_doc_ids)}"
        )
    n = len(required_doc_ids)
    coverage = sum(1 for d in required_doc_ids if d in retrieved_doc_ids) / n
    cite = sum(1 for d in required_doc_ids if d in cited_doc_ids) / n
    xdr = coverage * cite * synthesis_score
    return XDRResult(coverage=coverage, cite=cite, synthesis=synthesis_score, xdr=xdr, n_required=n)


def synthesis_score_fallback(
    answer_text: str, expected_synthesis_keys: Sequence[str], *, fuzzy_threshold: float = 85.0
) -> float:
    """The offline ``Syn`` fallback used when no LLM judge is available.

    Measures **presence of the joint elements, not inference over them** — this is a
    deliberate and named gap, not an approximation error. The real rubric asks whether
    the answer's *conclusion* depends on synthesizing two-plus documents together; the
    fallback can only check whether the answer's *text* mentions the key facts each
    document contributes (e.g. "72 hours", "supervisory authority", "escalate"). A model
    that lists all three facts without connecting them still passes this fallback. The
    gap between local Syn and LLM Syn on the same answer *is* the
    concatenation-vs-synthesis measurement the report is meant to surface.

    A key counts as present via normalized substring match, or — to tolerate light
    paraphrase without opening the door to unrelated matches — ``rapidfuzz``
    ``token_set_ratio >= 85``. The presence fraction is then snapped to ``{0, 0.5, 1}``:
    ``<0.5 -> 0``, ``<1.0 -> 0.5``, ``==1.0 -> 1``, matching the three-point LLM rubric
    so local and LLM Syn are comparable on the same scale.
    """
    if not expected_synthesis_keys:
        return 0.0
    norm_answer = _normalize_ws(answer_text)
    present = 0
    for key in expected_synthesis_keys:
        norm_key = _normalize_ws(key)
        if norm_key and norm_key in norm_answer:
            present += 1
        elif fuzz.token_set_ratio(norm_key, norm_answer) >= fuzzy_threshold:
            present += 1
    fraction = present / len(expected_synthesis_keys)
    if fraction < 0.5:
        return 0.0
    if fraction < 1.0:
        return 0.5
    return 1.0


@dataclass(frozen=True)
class CorpusXDR:
    """XDR averaged over cross-document questions only. ``n`` must always be printed
    alongside ``mean_xdr`` — with ~6 seed questions, ``n=3`` carries enormous sampling
    noise, and a bare mean invites treating it as more precise than it is."""

    mean_xdr: float
    n: int


def corpus_xdr(results: Sequence[XDRResult]) -> CorpusXDR:
    """Average ``.xdr`` over the given (already cross-document-only) results.

    Callers must filter to ``question.is_cross_document`` *before* calling this — this
    function does not accept single-document questions and cannot detect if one slipped
    in, because an ``XDRResult`` carries no question-level answer_type. The filtering
    responsibility is the caller's precisely so this stays a pure aggregate.
    """
    if not results:
        return CorpusXDR(mean_xdr=0.0, n=0)
    return CorpusXDR(mean_xdr=sum(r.xdr for r in results) / len(results), n=len(results))


# -------------------------------------------------------------------------------- 4. LpQU


def question_quality(
    af: float, cp: float, xdr: float, *, weight_af: float, weight_cp: float, weight_xdr: float
) -> float:
    """``Q = w_af*AF + w_cp*CP + w_xdr*XDR``. Weights come from ``Settings`` and must be
    printed in every report header — a tradeoff a reader is meant to be able to argue
    with requires seeing the weights, not just the resulting number."""
    return weight_af * af + weight_cp * cp + weight_xdr * xdr


def lpqu(t_total_s: float, q: float, *, epsilon: float) -> float:
    """``LpQU = t_total / max(Q, epsilon)`` — seconds per quality unit, lower is better.

    The ``epsilon`` clamp exists so a ``Q=0`` answer (fast but worthless) does not
    divide by zero; clamping to ``epsilon`` instead makes it score as *the worst
    possible* LpQU for its latency rather than undefined, which correctly ranks a fast
    wrong answer below a slow correct one. ``epsilon`` is not a tuning knob a caller
    should vary per-call: :mod:`mmu.config` fixes it and prints it in the report header,
    because LpQU values computed with different epsilon are not comparable.
    """
    return t_total_s / max(q, epsilon)


@dataclass(frozen=True)
class CorpusLpQU:
    """Both aggregate forms, so neither can be silently substituted for the other.

    ``corpus_lpqu`` is the number to report: the aggregate ratio ``sum(t) / sum(Q)``.
    ``per_question`` is ``t_i / max(Q_i, epsilon)`` for each question — diagnostic only,
    never averaged into a headline number. See :func:`corpus_lpqu` for why the
    difference matters.
    """

    corpus_lpqu: float
    per_question: tuple[float, ...]
    p50_latency_s: float
    p95_latency_s: float


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return sorted_values[lo]
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def corpus_lpqu(
    t_totals_s: Sequence[float], qs: Sequence[float], *, epsilon: float
) -> CorpusLpQU:
    """The corpus LpQU is ``sum(t_totals_s) / max(sum(qs), epsilon)`` — an aggregate
    ratio, **never** ``mean(t_i / max(q_i, epsilon))``.

    A mean of per-question ratios is dominated by the single worst-quality question: one
    seed with ``Q=0.01`` produces a ``t/Q`` on the order of 100x every other question's
    ratio, and averaging pulls the whole corpus number toward that one outlier — a
    corpus of five good answers and one degenerate one would report as "bad" even though
    five-sixths of the system worked. The aggregate ratio instead sums total time and
    total quality across the corpus before dividing, so one bad question contributes its
    actual (small) share of total quality, not an outsized share of the average.
    ``test_metrics.py`` asserts the two diverge on a deliberately skewed fixture so this
    cannot be "simplified" back into a mean.
    """
    if len(t_totals_s) != len(qs):
        raise ValueError("t_totals_s and qs must have the same length")
    if not t_totals_s:
        return CorpusLpQU(corpus_lpqu=0.0, per_question=(), p50_latency_s=0.0, p95_latency_s=0.0)
    per_question = tuple(t / max(q, epsilon) for t, q in zip(t_totals_s, qs))
    aggregate = sum(t_totals_s) / max(sum(qs), epsilon)
    sorted_t = sorted(t_totals_s)
    return CorpusLpQU(
        corpus_lpqu=aggregate,
        per_question=per_question,
        p50_latency_s=_percentile(sorted_t, 50),
        p95_latency_s=_percentile(sorted_t, 95),
    )


# ----------------------------------------------------------------- 5. Context Utilization


@dataclass(frozen=True)
class CURResult:
    """Three deterministic forms plus the citation-theater gap. ``k`` travels with the
    result because CUR only means something next to the ``k`` it was computed at — see
    the module-level note on why CUR is a diagnostic, never a target."""

    cited: float
    attr: float
    tok: float
    gap: float  # cited - attr: cited a source but never drew a verified span from it
    k: int


def _span_pattern(span: str) -> re.Pattern[str]:
    """A regex that matches ``span`` inside a chunk's original (non-normalized) text,
    tolerant of whitespace differences the way the judge's own verification is (see
    :mod:`mmu.eval.judge`). Built from the normalized span so the interval it finds maps
    back to real character offsets in the chunk, which is what interval merging needs."""
    tokens = _normalize_ws(span).split(" ")
    return re.compile(r"\s+".join(re.escape(t) for t in tokens if t))


def _find_span_interval(text: str, span: str) -> tuple[int, int] | None:
    if not span.strip():
        return None
    match = _span_pattern(span).search(text)
    return (match.start(), match.end()) if match else None


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Standard interval-merge, sorted by start. Overlapping/adjacent verified spans
    within one chunk must be merged before summing characters, or a heavily-quoted
    chunk (multiple claims verified against overlapping text) counts the same
    characters more than once and pushes CUR_tok above 1.0 for that chunk."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def context_utilization(
    chunks: Sequence[RetrievedChunk],
    *,
    cited_chunk_ids: Collection[str],
    claims: Sequence[ClaimVerification],
) -> CURResult:
    """Three deterministic measures of how much of the retrieved context the answer
    actually used, computed at zero extra judge cost (``claims`` is AF's own output).

    ``CUR_cited`` = distinct in-range ``[Sn]`` markers / k — purely syntactic, what the
    model *claimed* to use. ``CUR_attr`` = chunks with >=1 *verified* span / k — what AF
    actually confirmed the answer drew from. ``CUR_cited - CUR_attr`` is the
    citation-theater gap: a model that cites ``[S3]`` next to a sentence AF could not
    verify against chunk 3's text is citing for the *appearance* of grounding, and this
    number is the only place in the report that catches it directly.

    ``CUR_tok`` = verified span characters / total context characters, with overlapping
    verified spans merged into intervals first (see :func:`_merge_intervals`) — without
    merging, two claims separately verified against overlapping text in the same chunk
    would double-count those characters and could push a heavily-quoted chunk's
    contribution above what the chunk actually contains.

    **CUR is a diagnostic, never a target.** ``CUR_cited == 1.0`` at ``k=8`` most likely
    means ``k`` is too small (the model would use more context if given more); a low
    CUR alongside high AF and CP=1.0 can mean healthy over-fetch feeding a model that
    correctly filters down to what it needs. Shrinking ``k`` to chase a higher CUR
    number will reduce AF and XDR (less material to ground a cross-document answer in),
    so ``report.py`` (a later work package) must print CUR next to ``k`` and must never
    rank runs by it.
    """
    k = len(chunks)
    if k == 0:
        return CURResult(cited=0.0, attr=0.0, tok=0.0, gap=0.0, k=0)

    chunk_by_id = {c.chunk_id: c for c in chunks}
    cited_in_range = {cid for cid in cited_chunk_ids if cid in chunk_by_id}
    cur_cited = len(cited_in_range) / k

    verified_spans_by_chunk: dict[str, list[str]] = {}
    for claim in claims:
        if claim.supported and claim.span and claim.chunk_id in chunk_by_id:
            verified_spans_by_chunk.setdefault(claim.chunk_id, []).append(claim.span)
    cur_attr = len(verified_spans_by_chunk) / k

    total_chars = sum(len(c.text) for c in chunks)
    verified_chars = 0
    for chunk_id, spans in verified_spans_by_chunk.items():
        text = chunk_by_id[chunk_id].text
        intervals = [iv for iv in (_find_span_interval(text, s) for s in spans) if iv is not None]
        merged = _merge_intervals(intervals)
        verified_chars += sum(end - start for start, end in merged)
    cur_tok = verified_chars / total_chars if total_chars else 0.0

    return CURResult(cited=cur_cited, attr=cur_attr, tok=cur_tok, gap=cur_cited - cur_attr, k=k)


# ----------------------------------------------------------------- marker_compliance


#: Below this, Cite/XDR/every CUR variant rest on a denominator that is mostly
#: unparsed markers, not mostly-correct citations — the numbers are noise, not signal.
MARKER_COMPLIANCE_FLOOR = 0.8


def marker_compliance_rate(per_answer_compliance: Sequence[float]) -> float:
    """Corpus-level fraction of answers with >=1 well-formed ``[Sn]`` marker
    (``n <= k``). Each answer's own ``marker_compliance`` is 1.0 or 0.0 (see
    ``core.models.Answer``); this is their mean."""
    if not per_answer_compliance:
        return 0.0
    return sum(per_answer_compliance) / len(per_answer_compliance)


def is_marker_compliance_reliable(rate: float) -> bool:
    """``False`` below :data:`MARKER_COMPLIANCE_FLOOR`. When this is ``False``, the
    report must print a loud warning and still show ``Cite``, ``XDR`` and every CUR
    variant rather than hiding them — the fragility must be surfaced, not smoothed over
    by silently omitting the columns."""
    return rate >= MARKER_COMPLIANCE_FLOOR
