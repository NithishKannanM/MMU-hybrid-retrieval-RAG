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

from mmu.eval.runner import RunReport

#: Below this, markers are unreliable and every marker-derived column is suspect.
MARKER_RELIABILITY_FLOOR = 0.8


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
    )
    lines.append(
        f"judge: {report.judge}   model: {report.model}   embed: {report.embed_model or 'n/a'}"
    )
    lines.append(
        f"weights: af={w_af:.2f} cp={w_cp:.2f} xdr={w_xdr:.2f}   eps={report.epsilon}"
    )

    mark = "OK" if report.markers_reliable else "UNRELIABLE"
    lines.append(f"marker_compliance: {report.marker_compliance:.2f}  [{mark}]")
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
    if not report.markers_reliable:
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
    unreliable = [
        label
        for label, r in zip(labels, reports)
        if r.get("marker_compliance", 1.0) < MARKER_RELIABILITY_FLOOR
    ]
    if unreliable:
        lines.append(
            f"WARNING: marker compliance below {MARKER_RELIABILITY_FLOOR} in: "
            f"{', '.join(unreliable)}. Their Cite/XDR/CUR columns are unreliable."
        )
    return "\n".join(lines)
