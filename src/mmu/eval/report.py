"""Rendering and persistence of a run — where the honesty rules are enforced.

Three of this project's design decisions only exist at the point of *reporting*, so
they live here rather than in the metric functions:

* **XDR is printed as four columns.** ``Cov``, ``Cite``, ``Syn`` and the product. An
  XDR of 0 caused by ``Cov=0.66`` is a retrieval bug; one caused by ``Cov=1.0,
  Cite=0.33`` is a generation bug. They demand opposite fixes, and the product alone
  cannot distinguish them.
* **Marker compliance gates the columns that depend on it.** Below 0.8, ``Cite``, XDR
  and every CUR variant are computed from markers the model did not reliably emit. They
  are printed with a warning rather than silently presented as clean numbers.
* **``compare`` never declares a winner.** With ~6 seed questions, the 95% confidence
  interval on any proportion is roughly +/-0.2. Printing "hybrid wins" off a 0.05
  difference would be the most misleading thing this harness could do, so it prints the
  deltas, the n, and a noise band, and stops there.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from mmu.eval.metrics import paired_sign_test
from mmu.eval.runner import RunReport

#: Below this, markers are unreliable and every marker-derived column is suspect.
MARKER_RELIABILITY_FLOOR = 0.8

#: Paired service-latency differences at or under this many seconds count as ties.
#: Generation here takes tens of seconds, so this only absorbs timer jitter; without a
#: threshold a 2ms difference would be scored as a win and inflate the sign test's n.
LATENCY_TIE_S = 0.05


def _noise_band(n: int) -> float:
    """Half-width of a 95% CI for a proportion at the worst case p=0.5.

    ``1.96 * sqrt(0.25 / n)``. At n=6 that is +/-0.40, which is the point: the band is
    printed so nobody reads a 0.05 difference between two runs as a result.
    """
    return 1.96 * math.sqrt(0.25 / n) if n > 0 else 0.0


def save(report: RunReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render(report: RunReport, *, verbose: bool = False) -> str:
    w_af, w_cp, w_xdr = report.weights
    lines: list[str] = []
    lines.append(
        f"corpus: {report.n_documents} docs / {report.n_chunks} chunks   "
        f"questions: {len(report.runs)} (XDR set: {report.xdr_n})   "
        f"k={report.k}   rrf_k={report.rrf_k}   channels={'+'.join(report.channels)}"
        + (
            f"   rerank={report.rerank} over {report.rerank_depth} candidates"
            if report.rerank
            else "   rerank=none"
        )
    )
    lines.append(
        f"judge: {report.judge}   model: {report.model}   embed: {report.embed_model or 'n/a'}"
    )
    lines.append(
        f"weights: af={w_af:.2f} cp={w_cp:.2f} xdr={w_xdr:.2f}   eps={report.epsilon}"
    )

    # Branch on None before consulting markers_reliable: that property answers "is a
    # warning warranted", and for an unmeasurable run the answer is no — but printing
    # "OK" would claim markers were checked when nothing in the run carried one.
    if report.marker_compliance is None:
        lines.append(
            "marker_compliance: n/a  [no answer was measurable — every one correctly "
            "cited nothing]"
        )
    else:
        mark = "OK" if report.markers_reliable else "UNRELIABLE"
        excluded = len(report.runs) - report.marker_compliance_n
        over = f"over {report.marker_compliance_n}/{len(report.runs)} answer(s)"
        note = (
            f", {excluded} correct refusal(s) excluded as undefined"
            if excluded
            else ""
        )
        lines.append(
            f"marker_compliance: {report.marker_compliance:.2f}  [{mark}]  "
            f"{over}{note}"
        )
    lines.append("")

    header = (
        f"{'metric':<22}{'value':>8}   notes"
    )
    lines.append(header)
    lines.append("-" * len(header))

    af_note = (
        "lexical overlap, NOT entailment — CI tripwire only"
        if report.judge == "local"
        else "judge-verified, spans checked"
    )
    lines.append(f"{'Answer Faithfulness':<22}{report.af:>8.3f}   {af_note}")
    lines.append(
        f"{'Context Precision':<22}{report.cp:>8.3f}   rank-weighted AP, "
        f"n={report.cp_n} question(s) with a relevant set"
    )
    lines.append("")
    lines.append(f"{'  Cov (retrieval)':<22}{report.cov_mean:>8.3f}   required docs present in context")
    lines.append(f"{'  Cite (generation)':<22}{report.cite_mean:>8.3f}   required docs cited in answer")
    lines.append(f"{'  Syn (synthesis)':<22}{report.syn_mean:>8.3f}   combined vs. merely listed")
    lines.append(
        f"{'= XDR':<22}{report.xdr_mean:>8.3f}   product, n={report.xdr_n} "
        f"cross-document question(s)"
    )
    lines.append("")
    if report.mrr is None:
        lines.append(
            f"{'MRR / NDCG / recall':<22}{'n/a':>8}   no question had a relevant set"
        )
    else:
        lines.append(
            f"{'MRR':<22}{report.mrr:>8.3f}   1/rank of first relevant chunk, "
            f"n={report.mrr_n}"
        )
        if report.mean_ndcg is None:
            lines.append(
                f"{'  NDCG@k / recall@k':<22}{'n/a':>8}   corpus-wide relevant count "
                f"unavailable, so both are undefined (not 0.0)"
            )
        else:
            lines.append(
                f"{'  NDCG@k':<22}{report.mean_ndcg:>8.3f}   vs. ideal ranking, "
                f"n={report.rank_n}"
            )
            lines.append(
                f"{'  recall@k':<22}{report.mean_recall:>8.3f}   relevant chunks found "
                f"/ relevant in corpus  [per-run summary; denominators differ per "
                f"question, so do NOT compare across question sets]"
            )
    lines.append("")
    span_note = " [n/a: local judge emits no spans]" if report.judge == "local" else ""
    lines.append(f"{'CUR cited':<22}{report.cur_cited:>8.3f}   markers emitted / k={report.k}")
    lines.append(f"{'CUR attributed':<22}{report.cur_attr:>8.3f}   verified spans / k{span_note}")
    lines.append(f"{'CUR tokens':<22}{report.cur_tok:>8.3f}   verified chars / context chars{span_note}")
    lines.append(f"{'  citation-theater gap':<22}{report.cur_gap:>8.3f}   cited but never verified{span_note}")
    lines.append("")
    lines.append(f"{'LpQU (corpus)':<22}{report.corpus_lpqu:>8.2f}   seconds per quality unit, SUM(t)/SUM(Q)")
    lines.append(f"{'  p50 latency':<22}{report.p50_latency_s:>8.2f}s")
    lines.append(f"{'  p95 latency':<22}{report.p95_latency_s:>8.2f}s")

    notes: list[str] = []
    if report.marker_compliance is not None and not report.markers_reliable:
        notes.append(
            f"WARNING: marker_compliance {report.marker_compliance:.2f} < "
            f"{MARKER_RELIABILITY_FLOOR}. Cite, XDR and every CUR number above are "
            f"derived from [Sn] markers this model did not reliably emit — treat them "
            f"as unreliable for this run, not as low scores."
        )
    if report.judge == "local":
        notes.append(
            "NOTE: the local judge is negation-blind ('must notify' vs 'must not "
            "notify' score ~0.95 against the same chunk). Use --judge ollama or "
            "--judge anthropic for any Answer Faithfulness number you intend to report."
        )
        notes.append(
            "NOTE: CUR attributed, CUR tokens and the citation-theater gap are NOT "
            "measurements in this run. LocalJudge scores vocabulary overlap and never "
            "localizes a supporting span, so 'verified spans' is structurally zero and "
            "the gap is just CUR cited. Read them only from an LLM-judge run — "
            "a 0.000 here says nothing about whether the model cited honestly."
        )
    if report.degraded_claims:
        notes.append(
            f"NOTE: {report.degraded_claims} judge item(s) fell back to the local judge "
            f"after malformed JSON. Judges were not silently mixed — these are flagged."
        )
    if report.invented_spans:
        notes.append(
            f"NOTE: {report.invented_spans} verdict(s) were downgraded because the "
            f"judge's quoted span was not in the chunk it cited. A high count means the "
            f"judge model is too small."
        )
    band = _noise_band(len(report.runs))
    notes.append(
        f"n={len(report.runs)} questions: the 95% noise band on any proportion here is "
        f"about +/-{band:.2f}. Differences smaller than that are not results."
    )
    if notes:
        lines.append("")
        lines.extend(notes)

    if verbose:
        lines.append("")
        lines.append(f"{'id':<18}{'AF':>6}{'CP':>6}{'Cov':>6}{'Cite':>6}{'Syn':>6}{'XDR':>7}{'t(s)':>7}")
        for r in report.runs:
            x = r.xdr
            # "--" rather than nan for a dimension the question does not have. A nan
            # reads as a computation that went wrong; a dash reads as "not applicable",
            # which is what a single-document question's XDR actually is.
            def cell(value: float | None, width: int = 6) -> str:
                return f"{value:>{width}.2f}" if value is not None else f"{'--':>{width}}"

            lines.append(
                f"{r.question_id:<18}{r.af:>6.2f}{cell(r.cp)}"
                f"{cell(x.coverage if x else None)}"
                f"{cell(x.cite if x else None)}"
                f"{cell(x.synthesis if x else None)}"
                f"{cell(x.xdr if x else None, 7)}{r.latency_s:>7.2f}"
            )
            if r.error:
                lines.append(f"{'':<18}ERROR: {r.error}")
    return "\n".join(lines)


def _runs_by_id(report: dict) -> dict[str, dict]:
    return {r["question_id"]: r for r in report.get("runs", [])}


def _mrr(run: dict) -> float | None:
    """MRR out of a serialized ``QuestionRun``, or None where it is undefined.

    ``rank`` is ``None`` for a question with no relevant set, so this is not a
    ``.get`` with a 0.0 default: a missing value and a measured zero mean different
    things here, and conflating them is the bug class this project has already fixed
    three times.
    """
    rank = run.get("rank")
    return None if rank is None else rank.get("mrr")


#: The timing fields that make up service time, i.e. everything a served request would
#: actually pay for. Kept as a list so that adding a stage to `QuestionRun` and
#: forgetting to add it here is a *test* failure rather than a silently understated
#: latency column — which is exactly what happened when `rerank_s` was introduced: the
#: cross-encoder's ~19s per query was omitted from the paired table, and a reranked arm
#: that was genuinely faster was reported as faster by 45s instead of its real 26s.
_SERVICE_FIELDS: tuple[str, ...] = ("retrieval_s", "rerank_s", "generate_s")


def _service_s(run: dict) -> float:
    """Every stage a served request pays for, excluding judging — mirrors
    ``QuestionRun.service_s`` and must stay in step with it.

    Recomputed from the component fields rather than read from ``latency_s`` because
    ``latency_s`` includes judge time, which arms of an interleaved run share through
    the judgment cache: where two arms produce an identical answer over identical
    chunks, whichever ran second got its verdicts for free and would look faster for no
    retrieval-related reason.

    Summed over :data:`_SERVICE_FIELDS` rather than spelled out, so that a new stage is
    added in one place. A missing field reads as 0.0, which is correct for an older
    report written before the stage existed.
    """
    return sum(run.get(field, 0.0) for field in _SERVICE_FIELDS)


def _signed(value: float, width: int) -> str:
    """A signed delta cell. A difference that rounds to zero prints unsigned.

    ``f"{-0.0001:+.2f}"`` is ``-0.00``, which reads as "the arm was slower" when the
    honest reading is "these were the same". The sign test already treats such a value
    as a tie; the table must not contradict it.
    """
    if round(value, 2) == 0:
        return f"{0.0:>{width}.2f}"
    return f"{value:>+{width}.2f}"


def paired_cohort_error(reports: Sequence[dict], labels: Sequence[str]) -> str | None:
    """``None`` if these reports may be compared pairwise, else why they may not.

    The check is not bureaucratic. A paired test over runs that were *executed*
    sequentially is strictly worse than no test: it inherits all the between-run drift
    the pairing was supposed to remove, and then reports a p-value about it. So pairing
    is permitted only for reports that recorded themselves as members of one
    interleaved cohort, and only when their question sequences actually line up.
    """
    if len(reports) < 2:
        return "a paired comparison needs at least two runs"
    cohorts = [tuple(r.get("interleaved_with") or ()) for r in reports]
    if not all(cohorts[0] and c == cohorts[0] for c in cohorts):
        return (
            "these runs were not interleaved with each other, so they cannot be paired. "
            "Latency in particular is not comparable across separately-executed runs: "
            "generator throughput on this project varied 3x between sequential runs of "
            "the same model. Use `mmu eval ablate` to run the configurations "
            "question-major in one process."
        )
    for label, r in zip(labels, reports):
        own = r.get("arm_label") or label
        if own not in cohorts[0]:
            return f"{own!r} is not a member of the cohort {list(cohorts[0])}"
    ids = [[r["question_id"] for r in rep.get("runs", [])] for rep in reports]
    if any(seq != ids[0] for seq in ids):
        return "the runs do not cover the same questions in the same order"
    if not ids[0]:
        return "no questions in these runs"
    return None


def render_paired(reports: Sequence[dict], labels: Sequence[str]) -> str:
    """Per-question paired differences against the first run, with exact sign tests.

    The first run is the baseline purely by position, not by merit. Each row is one
    question that both arms answered seconds apart on the same loaded model, which is
    what licenses attributing the difference to the configuration — see
    ``runner.run_interleaved`` for why the unpaired version of this table could not.

    Significance is reported for the *difference*, and the function still never names a
    winner: it prints which arm was higher, how often, and the p-value, and where the
    design could not have reached significance at all it says so instead of letting a
    non-significant p be read as evidence of no effect.
    """
    problem = paired_cohort_error(reports, labels)
    if problem is not None:
        return f"PAIRED COMPARISON UNAVAILABLE: {problem}"

    base_label, base = labels[0], reports[0]
    base_runs = _runs_by_id(base)
    qids = [r["question_id"] for r in base.get("runs", [])]

    lines: list[str] = []
    lines.append(
        f"PAIRED vs {base_label} (interleaved, question-major, arm order rotated "
        f"per question)"
    )
    lines.append(
        "Each row is one question answered by both arms back to back on the same loaded "
        "model, so the\nwithin-pair difference is attributable to the configuration and "
        "not to host drift between runs."
    )

    for label, rep in list(zip(labels, reports))[1:]:
        arm_runs = _runs_by_id(rep)
        lines.append("")
        head = (
            f"{'question':<18}{'t_base':>8}{'t_arm':>8}{'dt':>9}"
            f"{'AF_base':>9}{'AF_arm':>8}{'dAF':>7}{'dQ':>7}{'dMRR':>8}"
        )
        lines.append(f"-- {label} vs {base_label}")
        lines.append(head)
        lines.append("-" * len(head))

        dt: list[float] = []
        daf: list[float] = []
        dq: list[float] = []
        dmrr: list[float] = []
        for qid in qids:
            b, a = base_runs[qid], arm_runs[qid]
            tb, ta = _service_s(b), _service_s(a)
            dt.append(ta - tb)
            daf.append(a["af"] - b["af"])
            dq.append(a["quality"] - b["quality"])
            # MRR is undefined where the question has no relevant set, and that is a
            # property of the question, so it is undefined for both arms at once. Such
            # a pair is omitted from the delta list rather than entered as a 0.0 tie:
            # a tie is dropped from n by the sign test anyway, but entering it would
            # misreport `ties` as an observation about the arms.
            mb, ma = _mrr(b), _mrr(a)
            if mb is None or ma is None:
                lines.append(
                    f"{qid:<18}{tb:>8.2f}{ta:>8.2f}{_signed(ta - tb, 9)}"
                    f"{b['af']:>9.2f}{a['af']:>8.2f}{_signed(a['af'] - b['af'], 7)}"
                    f"{_signed(a['quality'] - b['quality'], 7)}{'n/a':>8}"
                )
                continue
            dmrr.append(ma - mb)
            lines.append(
                f"{qid:<18}{tb:>8.2f}{ta:>8.2f}{_signed(ta - tb, 9)}"
                f"{b['af']:>9.2f}{a['af']:>8.2f}{_signed(a['af'] - b['af'], 7)}"
                f"{_signed(a['quality'] - b['quality'], 7)}{_signed(ma - mb, 8)}"
            )

        lines.append("")
        for name, deltas, eps, unit in (
            ("service latency", dt, LATENCY_TIE_S, "s"),
            ("Answer Faithfulness", daf, 0.0, ""),
            ("quality (composite)", dq, 0.0, ""),
            ("MRR (retrieval rank)", dmrr, 0.0, ""),
        ):
            if not deltas:
                lines.append(f"{name:<22}not measurable on any paired question")
                continue
            t = paired_sign_test(deltas, epsilon=eps)
            mean = sum(deltas) / len(deltas) if deltas else 0.0
            direction = (
                f"{label} higher in {t.wins}, lower in {t.losses}, tied {t.ties}"
            )
            verdict = f"p={t.p_two_sided:.3f}"
            if not t.conclusive_possible:
                verdict += (
                    f" — but n={t.n_pairs} CANNOT reach p<0.05 "
                    f"(best possible {t.best_attainable_p:.3f}); this is a power "
                    f"failure, not evidence of no difference"
                )
            elif not t.significant:
                verdict += f" (floor {t.best_attainable_p:.3f}) — not a result"
            else:
                verdict += (
                    f" (floor {t.best_attainable_p:.3f}) — consistent in sign across "
                    f"every informative pair"
                )
            lines.append(f"{name:<22}mean {_signed(mean, 8)}{unit:<2} {direction}")
            lines.append(f"{'':<22}{verdict}")

    lines.append("")
    unanimous = paired_sign_test([1.0] * len(qids))
    lines.append(
        "A sign test asks only whether one arm is consistently higher, never by how "
        "much. The means\nabove are descriptive. With "
        f"{len(qids)} question(s) the strongest result any of these tests could report "
        f"is\np={unanimous.best_attainable_p:.3f}, and only if every question agrees in "
        "sign; ties lower it further. Pairing buys\nprecision per question, not more "
        "questions."
    )
    return "\n".join(lines)


def render_comparison(reports: Sequence[dict], labels: Sequence[str]) -> str:
    """Side-by-side runs. Prints deltas and a noise band; never names a winner."""
    lines: list[str] = []
    n = min((len(r.get("runs", [])) for r in reports), default=0)
    band = _noise_band(n)
    head = (
        f"{'run':<16}{'AF':>7}{'CP':>7}{'Cov':>7}{'Cite':>7}{'Syn':>7}{'XDR':>7}"
        f"{'CURcit':>8}{'CURatt':>8}{'p50':>7}{'LpQU':>8}"
    )
    lines.append(head)
    lines.append("-" * len(head))
    for label, r in zip(labels, reports):
        lines.append(
            f"{label:<16}{r['af']:>7.3f}{r['cp']:>7.3f}{r['cov_mean']:>7.3f}"
            f"{r['cite_mean']:>7.3f}{r['syn_mean']:>7.3f}{r['xdr_mean']:>7.3f}"
            f"{r['cur_cited']:>8.3f}{r['cur_attr']:>8.3f}"
            f"{r['p50_latency_s']:>7.2f}{r['corpus_lpqu']:>8.2f}"
        )
    lines.append("")
    lines.append(
        f"n={n} questions per run. The 95% noise band on any proportion is about "
        f"+/-{band:.2f} — differences smaller than that are not results, and this "
        f"table deliberately does not declare a winner."
    )
    # The latency columns are the ones most easily misread, because a 3x difference
    # looks far too large to be noise. Measured on this project: it was. Unless the runs
    # were interleaved, p50 and LpQU carry whatever the host was doing at the time.
    if len(reports) < 2:
        pass  # nothing to compare latency across
    elif paired_cohort_error(reports, labels) is None:
        lines.append(
            "These runs were interleaved question-major, so the paired table below "
            "tests the differences directly."
        )
    else:
        lines.append(
            "p50/LpQU are NOT comparable across these rows: they were executed as "
            "separate runs, and generator throughput on this host varied 3x between "
            "sequential runs of the same model while retrieval accounted for under 2% "
            "of wall clock. Run `mmu eval ablate` for a latency comparison that means "
            "something."
        )
    # `is not None` matters: the rate is None for a run in which no answer was
    # measurable, and `None < float` raises rather than comparing.
    unreliable = [
        label
        for label, r in zip(labels, reports)
        if r.get("marker_compliance") is not None
        and r["marker_compliance"] < MARKER_RELIABILITY_FLOOR
    ]
    if unreliable:
        lines.append(
            f"WARNING: marker compliance below {MARKER_RELIABILITY_FLOOR} in: "
            f"{', '.join(unreliable)}. Their Cite/XDR/CUR columns are unreliable."
        )
    return "\n".join(lines)
