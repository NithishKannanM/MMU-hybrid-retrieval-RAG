"""Drives retrieve -> generate -> judge -> score for each question.

This module is orchestration only; every formula lives in :mod:`mmu.eval.metrics` and
every verdict in :mod:`mmu.eval.judge`. That separation is what lets the metric math be
tested exhaustively without a model, and it is worth preserving: if a number starts
looking wrong, it is computed in one place and that place has no I/O in it.

**Claim verification is O(claims x chunks) in the worst case, with an early exit.** A
claim is supported if *any* retrieved chunk supports it, so the runner walks chunks in
fused rank order and stops at the first supporting verdict. On a grounded answer the
supporting chunk is usually rank 1-3, so the typical cost is far below the bound; on a
hallucinated claim it pays the full k, which is the right place to spend judge calls
since that is the case the metric exists to catch.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from mmu.config import Settings
from mmu.core.models import RetrievedChunk
from mmu.eval import metrics
from mmu.eval.judge import CachingJudge, Judge
from mmu.eval.metrics import ClaimVerification, CURResult, XDRResult
from mmu.eval.questions import Question
from mmu.generate.base import Generator, GeneratorUnavailable
from mmu.generate.prompt import (
    build_messages,
    is_refusal,
    marker_compliance,
    resolve_citations,
)
from mmu.retrieve.hybrid import HybridRetriever


@dataclass
class QuestionRun:
    """Everything scored for one question, kept flat so the report needs no re-derivation."""

    question_id: str
    question: str
    answer_type: str
    answer_text: str
    model: str
    k: int
    latency_s: float
    retrieval_s: float
    generate_s: float
    af: float
    #: None when the question has no relevant set at all (an `unanswerable` question).
    #: Context Precision is *undefined* there, not zero: a system that correctly has
    #: nothing to retrieve would otherwise be scored as having retrieved badly, and the
    #: corpus average would be capped below 1.0 by the mere presence of such a question.
    cp: float | None
    xdr: XDRResult | None                 # None for single-document questions
    cur: CURResult
    quality: float
    marker_compliance: float
    n_claims: int
    n_supported: int
    degraded_claims: int = 0
    invented_spans: int = 0
    error: str | None = None
    retrieved_doc_ids: tuple[str, ...] = ()
    cited_doc_ids: tuple[str, ...] = ()


@dataclass
class RunReport:
    """Corpus aggregates plus the run's configuration.

    The configuration is part of the result, not metadata around it: LpQU is
    incomparable across different quality weights or a different epsilon, and XDR is
    incomparable across a different k. A report that does not carry them cannot be
    compared to another report, so they travel together.
    """

    runs: list[QuestionRun] = field(default_factory=list)
    # config header
    judge: str = ""
    model: str = ""
    embed_model: str = ""
    channels: tuple[str, ...] = ()
    k: int = 0
    rrf_k: int = 60
    weights: tuple[float, float, float] = (0.4, 0.2, 0.4)
    epsilon: float = 0.01
    n_documents: int = 0
    n_chunks: int = 0

    # aggregates
    af: float = 0.0
    cp: float = 0.0
    cp_n: int = 0          # questions that HAVE a relevant set; see QuestionRun.cp
    xdr_mean: float = 0.0
    xdr_n: int = 0
    cov_mean: float = 0.0
    cite_mean: float = 0.0
    syn_mean: float = 0.0
    cur_cited: float = 0.0
    cur_attr: float = 0.0
    cur_tok: float = 0.0
    cur_gap: float = 0.0
    corpus_lpqu: float = 0.0
    p50_latency_s: float = 0.0
    p95_latency_s: float = 0.0
    marker_compliance: float = 0.0
    degraded_claims: int = 0
    invented_spans: int = 0

    @property
    def markers_reliable(self) -> bool:
        return metrics.is_marker_compliance_reliable(self.marker_compliance)


async def _verify_claims(
    judge: Judge, claims: Sequence[str], chunks: Sequence[RetrievedChunk]
) -> tuple[list[ClaimVerification], int, int]:
    """Verify each claim against the retrieved chunks, best-ranked first."""
    verified: list[ClaimVerification] = []
    degraded = invented = 0
    for claim in claims:
        outcome = ClaimVerification(claim=claim, supported=False)
        for chunk in chunks:
            judgment = await judge.verify_claim(claim, chunk.text, chunk.chunk_id)
            degraded += int(judgment.degraded)
            invented += int(judgment.invented_span)
            if judgment.verdict:
                outcome = ClaimVerification(
                    claim=claim, supported=True, chunk_id=judgment.source, span=judgment.span
                )
                break  # a claim needs one supporting chunk, not the best one
        verified.append(outcome)
    return verified, degraded, invented


async def run_question(
    question: Question,
    retriever: HybridRetriever,
    generator: Generator,
    judge: Judge,
    settings: Settings,
    *,
    k: int,
    channels: Sequence[str],
    rrf_k: int,
    cache_path: Path | None = None,
) -> QuestionRun:
    started = time.perf_counter()
    if cache_path is not None:
        judge = CachingJudge(judge, question.id, cache_path)

    # to_thread for the same reason the API does it: search is CPU-bound and blocking,
    # and the judge calls interleaved with it are network I/O that should not be stalled.
    result = await asyncio.to_thread(
        retriever.search, question.question, k=k, channels=list(channels), rrf_k=rrf_k
    )
    retrieval_s = result.timings.total_ms / 1000

    t = time.perf_counter()
    try:
        completion = await generator.complete(
            build_messages(question.question, result.chunks),
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        answer_text, model = completion.text, completion.model
        error = None
    except GeneratorUnavailable as exc:
        answer_text, model, error = "", generator.name, str(exc)
    generate_s = time.perf_counter() - t

    citations = resolve_citations(answer_text, result.chunks)
    cited_chunk_ids = {c.chunk_id for c in citations}
    cited_doc_ids = {c.doc_id for c in citations}
    retrieved_doc_ids = {c.doc_id for c in result.chunks}

    decomposition = await judge.decompose_claims(answer_text)
    claims, degraded, invented = await _verify_claims(
        judge, decomposition.claims, result.chunks
    )
    degraded += int(decomposition.degraded)

    af = metrics.answer_faithfulness(claims, is_refusal=is_refusal(answer_text))
    cp: float | None = None
    if question.relevant_doc_ids:
        cp = metrics.context_precision(
            result.chunks,
            relevant_doc_ids=question.relevant_doc_ids,
            must_contain=question.must_contain,
        )

    xdr: XDRResult | None = None
    if question.is_cross_document:
        if judge.name == "local":
            syn = metrics.synthesis_score_fallback(
                answer_text, question.expected_synthesis_keys
            )
        else:
            judgment = await judge.score_synthesis(
                question=question.question,
                answer_text=answer_text,
                required_doc_texts={
                    d: retriever.store.text_of_doc(d) for d in question.required_doc_ids
                },
                expected_synthesis_keys=question.expected_synthesis_keys,
            )
            degraded += int(judgment.degraded)
            syn = judgment.score if judgment.score is not None else 0.0
        xdr = metrics.cross_document_reasoning(
            required_doc_ids=list(question.required_doc_ids),
            retrieved_doc_ids=retrieved_doc_ids,
            cited_doc_ids=cited_doc_ids,
            synthesis_score=syn,
        )

    cur = metrics.context_utilization(
        result.chunks, cited_chunk_ids=cited_chunk_ids, claims=claims
    )
    # Quality is the weighted mean over the dimensions this question actually has.
    # Folding a zero in for a dimension the question was never asked about would
    # penalize it for its own shape: a single-document question has no XDR, and an
    # unanswerable one has neither XDR nor CP. Renormalizing over the applicable
    # weights keeps every question's quality on the same 0-1 scale.
    applicable: list[tuple[float, float]] = [(af, settings.weight_af)]
    if cp is not None:
        applicable.append((cp, settings.weight_cp))
    if xdr is not None:
        applicable.append((xdr.xdr, settings.weight_xdr))
    total_weight = sum(w for _, w in applicable)
    quality = (
        sum(v * w for v, w in applicable) / total_weight if total_weight else 0.0
    )

    return QuestionRun(
        question_id=question.id,
        question=question.question,
        answer_type=question.answer_type,
        answer_text=answer_text,
        model=model,
        k=len(result.chunks),
        latency_s=time.perf_counter() - started,
        retrieval_s=retrieval_s,
        generate_s=generate_s,
        af=af,
        cp=cp,
        xdr=xdr,
        cur=cur,
        quality=quality,
        marker_compliance=marker_compliance(answer_text, len(result.chunks)),
        n_claims=len(claims),
        n_supported=sum(1 for c in claims if c.supported),
        degraded_claims=degraded,
        invented_spans=invented,
        error=error,
        retrieved_doc_ids=tuple(sorted(retrieved_doc_ids)),
        cited_doc_ids=tuple(sorted(cited_doc_ids)),
    )


async def run_eval(
    questions: Sequence[Question],
    retriever: HybridRetriever,
    generator: Generator,
    judge: Judge,
    settings: Settings,
    *,
    k: int | None = None,
    channels: Sequence[str] = ("dense", "sparse"),
    rrf_k: int | None = None,
    cache_path: Path | None = None,
    on_question=None,
) -> RunReport:
    """Run every question sequentially and aggregate.

    Sequential, not gathered: the generator is a single local model with one GPU-backed
    slot, so concurrent requests queue inside Ollama anyway while making the per-question
    latency numbers — which LpQU is built on — measure queueing instead of generation.
    """
    k = k if k is not None else settings.top_k
    rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
    report = RunReport(
        judge=judge.name,
        model=generator.name,
        embed_model=(
            retriever.store.manifest.embed_model if retriever.store.manifest else ""
        ),
        channels=tuple(channels),
        k=k,
        rrf_k=rrf_k,
        weights=(settings.weight_af, settings.weight_cp, settings.weight_xdr),
        epsilon=settings.lpqu_epsilon,
        n_documents=len(retriever.store.doc_ids()),
        n_chunks=len(retriever.store),
    )

    for question in questions:
        run = await run_question(
            question, retriever, generator, judge, settings,
            k=k, channels=channels, rrf_k=rrf_k, cache_path=cache_path,
        )
        report.runs.append(run)
        if on_question is not None:
            on_question(run)

    _aggregate(report)
    return report


def _aggregate(report: RunReport) -> None:
    runs = report.runs
    if not runs:
        return
    report.af = metrics.corpus_answer_faithfulness([r.af for r in runs])
    # Only questions that have a relevant set — see QuestionRun.cp.
    cps = [r.cp for r in runs if r.cp is not None]
    report.cp = sum(cps) / len(cps) if cps else 0.0
    report.cp_n = len(cps)

    xdrs = [r.xdr for r in runs if r.xdr is not None]
    corpus = metrics.corpus_xdr(xdrs)
    report.xdr_mean, report.xdr_n = corpus.mean_xdr, corpus.n
    if xdrs:
        report.cov_mean = sum(x.coverage for x in xdrs) / len(xdrs)
        report.cite_mean = sum(x.cite for x in xdrs) / len(xdrs)
        report.syn_mean = sum(x.synthesis for x in xdrs) / len(xdrs)

    report.cur_cited = sum(r.cur.cited for r in runs) / len(runs)
    report.cur_attr = sum(r.cur.attr for r in runs) / len(runs)
    report.cur_tok = sum(r.cur.tok for r in runs) / len(runs)
    report.cur_gap = report.cur_cited - report.cur_attr

    lpqu = metrics.corpus_lpqu(
        [r.latency_s for r in runs], [r.quality for r in runs], epsilon=report.epsilon
    )
    report.corpus_lpqu = lpqu.corpus_lpqu
    report.p50_latency_s = lpqu.p50_latency_s
    report.p95_latency_s = lpqu.p95_latency_s

    report.marker_compliance = metrics.marker_compliance_rate(
        [r.marker_compliance for r in runs]
    )
    report.degraded_claims = sum(r.degraded_claims for r in runs)
    report.invented_spans = sum(r.invented_spans for r in runs)
