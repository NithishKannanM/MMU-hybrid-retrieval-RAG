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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

from mmu.config import Settings
from mmu.core.models import RetrievedChunk
from mmu.eval import metrics
from mmu.eval.judge import CachingJudge, Judge
from mmu.eval.metrics import ClaimVerification, CURResult, RankMetrics, XDRResult
from mmu.retrieve.registry import build_reranker
from mmu.retrieve.rerank import Reranker, rerank as rerank_result
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
    #: None for a *correct refusal* — an answer that refuses and cites nothing. Marker
    #: compliance is **undefined** there, not zero, for the same reason :attr:`cp` is:
    #: the answer was never supposed to carry an ``[Sn]`` marker, so scoring it 0.0 on a
    #: metric about citing measures the question's shape rather than the model's
    #: behaviour, and drags a gating rate below its own floor. Set from exactly the same
    #: condition as AF's refusal carve-out, so the two can never disagree about what a
    #: refusal is.
    marker_compliance: float | None
    n_claims: int
    n_supported: int
    #: Cross-encoder time, 0.0 when no reranker ran. Its own field, and counted inside
    #: :attr:`service_s`, because a reranking arm's whole claim is that the shorter
    #: prompt it buys saves more generation time than the cross-encoder costs — and that
    #: claim is unfalsifiable if the cost is hidden inside `retrieval_s`.
    rerank_s: float = 0.0
    degraded_claims: int = 0
    invented_spans: int = 0
    error: str | None = None
    retrieved_doc_ids: tuple[str, ...] = ()
    cited_doc_ids: tuple[str, ...] = ()
    #: Set only by :func:`run_interleaved`. ``arm_index`` is this configuration's
    #: position in the rotation *for this question*, which is what makes the
    #: counterbalancing auditable after the fact rather than merely asserted.
    arm_label: str = ""
    arm_index: int = 0
    #: Rank-aware retrieval quality (:func:`metrics.rank_metrics`). ``None`` when the
    #: question has no relevant set, exactly like :attr:`cp`. These are the only
    #: retrieval metrics that can see a reranker: ``Cov`` is a set intersection and
    #: ``cp`` normalises over what was retrieved, so both are invariant under the
    #: reordering a reranker performs. ``recall`` and ``ndcg`` are additionally ``None``
    #: when the corpus-wide relevant count could not be computed, rather than 0.0.
    rank: RankMetrics | None = None

    @property
    def service_s(self) -> float:
        """Retrieval + generation, excluding judging.

        ``latency_s`` wraps the whole of :func:`run_question`, judge calls included, so
        it is not comparable between two arms of an interleaved run: ``CachingJudge`` is
        keyed by ``question_id``, which the arms share, so whichever arm runs second
        gets its identical claim/chunk verdicts for free and looks faster for a reason
        that has nothing to do with retrieval. This is the quantity a paired latency
        comparison must use, and it is also the only part of the two that a served
        request would actually pay.
        """
        return self.retrieval_s + self.rerank_s + self.generate_s


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
    #: This run's label, and every label it was interleaved with (itself included), set
    #: only by :func:`run_interleaved`. Empty for a standalone run. ``report.py`` refuses
    #: to print a paired comparison unless the reports agree on a non-empty cohort,
    #: which is what stops a paired test being computed over runs that were executed
    #: sequentially and are therefore not paired at all.
    arm_label: str = ""
    interleaved_with: tuple[str, ...] = ()
    #: The reranker alias this arm used, or "" for none. Part of the configuration for
    #: the same reason `k` and `rrf_k` are: two reports are not comparable unless they
    #: agree about it, and a reranked run that does not say so is a result with its
    #: cause missing.
    rerank: str = ""
    #: Candidate pool the reranker reordered. Meaningless without `rerank`, and recorded
    #: because the reranker's ceiling is recall@depth — a reranked arm's result cannot be
    #: interpreted without knowing how much it was allowed to see.
    rerank_depth: int = 0

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
    #: Rank-aware retrieval means, over the questions where each is *defined*.
    #: ``mrr`` is defined wherever there is a relevant set; ``recall``/``ndcg``
    #: additionally need the corpus-wide relevant count, hence the separate ``n``.
    #: ``None`` rather than 0.0 when nothing was measurable, like ``marker_compliance``.
    #: ``mean_recall`` is a mean of per-question recalls whose denominators differ, so
    #: it is a summary of this run and **not** comparable across question sets; the
    #: paired per-question deltas in ``report.render_paired`` are the comparable form.
    mrr: float | None = None
    mrr_n: int = 0
    mean_recall: float | None = None
    mean_ndcg: float | None = None
    rank_n: int = 0
    #: None when no answer was measurable; see :func:`metrics.marker_compliance_rate`.
    marker_compliance: float | None = 0.0
    #: Answers the rate was computed over — the corpus total minus correct refusals.
    #: Printed next to the rate for the same reason ``cp_n`` is: a rate over 5 of 6
    #: answers and a rate over 6 of 6 are different claims.
    marker_compliance_n: int = 0
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


def retriever_depth(settings: Settings, k: int) -> int:
    """The candidate pool a reranker sees, defaulting to the project's over-fetch depth.

    Reusing `Settings.resolved_depth` rather than inventing a second notion of depth:
    the pool the reranker reorders should be the pool fusion already produces, or the
    reranked arm would differ from its control in two ways at once and the comparison
    would not isolate reranking.
    """
    return settings.resolved_depth(k)


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
    n_relevant_total: int | None = None,
    reranker: Reranker | None = None,
    rerank_depth: int | None = None,
) -> QuestionRun:
    started = time.perf_counter()
    if cache_path is not None:
        judge = CachingJudge(judge, question.id, cache_path)

    # to_thread for the same reason the API does it: search is CPU-bound and blocking,
    # and the judge calls interleaved with it are network I/O that should not be stalled.
    # With a reranker, retrieval fetches the *candidate pool* and the reranker decides
    # which k survive. Without one, it fetches k directly. `fetch_k` is therefore the
    # pool size, not the context size, and the two are only equal in the unreranked arm.
    fetch_k = k if reranker is None else (rerank_depth or retriever_depth(settings, k))
    result = await asyncio.to_thread(
        retriever.search,
        question.question,
        k=fetch_k,
        depth=fetch_k if reranker is not None else None,
        channels=list(channels),
        rrf_k=rrf_k,
    )
    retrieval_s = result.timings.total_ms / 1000

    rerank_s = 0.0
    if reranker is not None:
        result = await asyncio.to_thread(rerank_result, result, reranker, k=k)
        rerank_s = result.timings.rerank_ms / 1000

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

    # A *pure* refusal asserts nothing about the corpus, so it has no claims to verify
    # and AF's m==0 carve-out should award it 1.0 (see questions.jsonl's
    # unanswerable-001: "A correct refusal must score 1.0 on Answer Faithfulness").
    # Measured: the judge decomposes "The provided context does not contain this
    # information." — verbatim the wording prompt.py suggests — into one unsupported
    # claim, so m==1, the carve-out never fires, and a textbook refusal scored 0.0.
    #
    # `cited_chunk_ids` is the laundering guard, and it is why this is not simply
    # `if is_refusal(...)`. is_refusal uses `search`, so it also matches a refusal
    # sentence tacked onto a real answer; prompt.py calls that false positive the worse
    # failure, since it would launder an unsupported answer into a perfect score. An
    # answer that cites nothing cannot be laundering support it never claimed.
    pure_refusal = is_refusal(answer_text) and not cited_chunk_ids

    # Was abstaining the right call? Two situations where it was: the question is
    # unanswerable (no required documents exist to find), or a required document was
    # not retrieved, in which case refusing beats inventing an answer and the failure
    # belongs to retrieval — `Cov` and `CP` already carry it, and double-charging it to
    # AF would turn a generation metric into a retrieval one.
    #
    # Everything else is an unjustified refusal: the model had the required document in
    # context and declined anyway. Measured: the generator refused `single-002`, an
    # "easy" single-document question, under dense-only and sparse-only with gdpr
    # retrieved at CP 0.74 and 1.00 — and the old carve-out scored it 1.00, because it
    # keyed on "does this look like a refusal" and never asked whether it should have
    # been one. That made corpus AF 1.000 reachable by refusing everything.
    required = set(question.required_doc_ids)
    refusal_is_correct = pure_refusal and (
        not required or not required.issubset(retrieved_doc_ids)
    )
    if pure_refusal:
        claims = ()
    af = metrics.answer_faithfulness(claims, refusal_is_correct=refusal_is_correct)
    cp: float | None = None
    rank: RankMetrics | None = None
    if question.relevant_doc_ids:
        cp = metrics.context_precision(
            result.chunks,
            relevant_doc_ids=question.relevant_doc_ids,
            must_contain=question.must_contain,
        )
        # Same chunks, same relevance predicate, different question: `cp` grades the
        # ordering of what was found, `rank` grades the finding against how much there
        # was to find. `n_relevant_total` comes from the orchestrator because it is a
        # property of the corpus, not of this retrieval.
        rank = metrics.rank_metrics(
            result.chunks,
            relevant_doc_ids=question.relevant_doc_ids,
            must_contain=question.must_contain,
            n_relevant_total=n_relevant_total,
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
        # Keyed on `pure_refusal`, deliberately *not* on `refusal_is_correct`. An
        # unjustified refusal is a real failure, but it is an *answering* failure, and
        # AF now scores it 0.0. Counting it 0.0 here as well would charge it a second
        # time to the rate that gates Cite/XDR/CUR — resurrecting the exact bug this
        # None exists to fix, where a gating metric trips for a reason that has nothing
        # to do with how well the model cites. No refusal carries a marker by
        # construction, so for all of them compliance is undefined rather than failed.
        marker_compliance=(
            None if pure_refusal else marker_compliance(answer_text, len(result.chunks))
        ),
        n_claims=len(claims),
        n_supported=sum(1 for c in claims if c.supported),
        rerank_s=rerank_s,
        degraded_claims=degraded,
        invented_spans=invented,
        error=error,
        retrieved_doc_ids=tuple(sorted(retrieved_doc_ids)),
        cited_doc_ids=tuple(sorted(cited_doc_ids)),
        rank=rank,
    )


def relevant_chunk_counts(
    questions: Sequence[Question], retriever: HybridRetriever
) -> dict[str, int]:
    """Per-question count of relevant chunks in the whole index — recall's denominator.

    Computed once per run rather than once per question-arm: it depends only on the
    question and the corpus, so recomputing it inside each arm would normalise 378 chunk
    texts three times over for an identical answer. Questions with no relevant set are
    omitted entirely, so :func:`metrics.rank_metrics` receives ``None`` for them and
    reports recall and NDCG as undefined rather than zero.
    """
    corpus = [(c.doc_id, c.text) for c in retriever.store.chunks]
    counts: dict[str, int] = {}
    for q in questions:
        if not q.relevant_doc_ids:
            continue
        counts[q.id] = metrics.count_relevant_chunks(
            corpus, relevant_doc_ids=q.relevant_doc_ids, must_contain=q.must_contain
        )
    return counts


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

    n_relevant = relevant_chunk_counts(questions, retriever)
    for question in questions:
        run = await run_question(
            question, retriever, generator, judge, settings,
            k=k, channels=channels, rrf_k=rrf_k, cache_path=cache_path,
            n_relevant_total=n_relevant.get(question.id),
        )
        report.runs.append(run)
        if on_question is not None:
            on_question(run)

    _aggregate(report)
    return report


@dataclass(frozen=True)
class Arm:
    """One configuration under test in an interleaved run.

    ``rerank`` is a *registry alias*, not a model id — see
    :mod:`mmu.retrieve.registry` for why that distinction is a boundary and not a
    style choice. ``None`` means this arm deliberately does not rerank.

    ``k`` and ``depth`` override the run defaults for this arm alone. ``k`` is what
    makes the shrink-the-context experiment expressible: an arm can keep 3 chunks where
    its control keeps 8. ``depth`` should be set to the **same** value on every arm in a
    reranking comparison, because `Settings.resolved_depth` is a function of k
    (``max(4k, 20)``) — so leaving it unset would hand a k=3 arm a 20-candidate pool and
    a k=8 arm a 32-candidate pool, and the arms would then differ in two ways at once.
    """

    label: str
    channels: tuple[str, ...]
    rerank: str | None = None
    k: int | None = None
    depth: int | None = None


#: The standard three-way retrieval ablation, interleaved. ``hybrid`` is first so it is
#: the baseline the others are differenced against in the paired report.
DEFAULT_ARMS: tuple[Arm, ...] = (
    Arm("hybrid", ("dense", "sparse")),
    Arm("dense-only", ("dense",)),
    Arm("sparse-only", ("sparse",)),
)

#: The reranking comparison. All three arms share ``depth=32`` so the only differences
#: are whether the cross-encoder runs and how many chunks reach the prompt:
#:
#: - ``hybrid-k8``        — the control: today's production configuration.
#: - ``hybrid-rerank-k8`` — same context size, better-chosen contents. Isolates
#:   reranking's effect on *quality* with prompt length held constant.
#: - ``hybrid-rerank-k3`` — the one that could pay for itself. If recall@3 after
#:   reranking matches recall@8 without it, the prompt shrinks by five chunks, and
#:   since generation is ~99% of wall clock that is the only lever that moves
#:   end-to-end latency.
#:
#: Splitting quality and length into two arms is deliberate: a single
#: "rerank and shrink" arm that came back faster and worse could not say which change
#: did which.
RERANK_ARMS: tuple[Arm, ...] = (
    Arm("hybrid-k8", ("dense", "sparse"), k=8, depth=32),
    Arm("hybrid-rerank-k8", ("dense", "sparse"), rerank="default", k=8, depth=32),
    Arm("hybrid-rerank-k3", ("dense", "sparse"), rerank="default", k=3, depth=32),
)


async def run_interleaved(
    questions: Sequence[Question],
    retriever: HybridRetriever,
    generator: Generator,
    judge: Judge,
    settings: Settings,
    *,
    arms: Sequence[Arm] = DEFAULT_ARMS,
    k: int | None = None,
    rrf_k: int | None = None,
    cache_path: Path | None = None,
    on_question=None,
) -> dict[str, RunReport]:
    """Run every arm on every question *within one process*, question-major.

    This exists because the sequential ablation — one full ``run_eval`` per
    configuration, one after another — cannot support a latency comparison, and this
    project measured that the hard way. Across five sequential runs of the same
    ``qwen2.5:7b-instruct-q4_K_M``, generator throughput ranged from 6.7 to 20.5
    characters per second while retrieval accounted for 0.0-1.4% of wall clock. The
    "3x faster" p50 that fell out of that table was host state, not retrieval: a 3B
    model in the same series was *slower* than the fast 7B run, which is backwards and
    gives the game away. Differencing corpus means across separately-executed runs
    differences the configuration and everything else that drifted, and on this hardware
    everything else dominates.

    Question-major fixes it by construction. Both arms answer ``xdr-001`` seconds apart,
    on the same loaded model, at the same temperature, under the same thermal and
    contention state, so the within-pair difference has one plausible cause left.

    **The arm order rotates per question.** Within a question the first arm pays any
    cold-start cost and later arms benefit from a warm KV cache and a resident model, so
    a fixed order would simply relabel that bias as an arm effect. Rotating by question
    index spreads each position evenly across arms (with 3 arms and 6 questions, each
    arm leads exactly twice) and :attr:`QuestionRun.arm_index` records where each run
    actually landed, so the balance can be checked rather than trusted.

    What this still does not fix: an arm's own ``latency_s`` and therefore ``LpQU``
    remain contaminated by judge-cache sharing between arms (see
    :attr:`QuestionRun.service_s`), and ``n`` is unchanged — pairing buys precision per
    question, not more questions. Six questions is still six questions.
    """
    if not arms:
        raise ValueError("run_interleaved needs at least one arm")
    labels = [a.label for a in arms]
    if len(set(labels)) != len(labels):
        raise ValueError(f"arm labels must be unique, got {labels}")

    k = k if k is not None else settings.top_k
    rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
    cohort = tuple(sorted(labels))

    reports: dict[str, RunReport] = {
        arm.label: RunReport(
            judge=judge.name,
            model=generator.name,
            embed_model=(
                retriever.store.manifest.embed_model if retriever.store.manifest else ""
            ),
            channels=tuple(arm.channels),
            k=arm.k or k,
            rrf_k=rrf_k,
            weights=(settings.weight_af, settings.weight_cp, settings.weight_xdr),
            epsilon=settings.lpqu_epsilon,
            n_documents=len(retriever.store.doc_ids()),
            n_chunks=len(retriever.store),
            arm_label=arm.label,
            interleaved_with=cohort,
            rerank=arm.rerank or "",
            rerank_depth=(arm.depth or 0) if arm.rerank else 0,
        )
        for arm in arms
    }

    n_relevant = relevant_chunk_counts(questions, retriever)
    # One reranker instance per distinct alias, built once before the loop. A
    # cross-encoder is ~568M parameters: constructing it per question-arm would load the
    # weights 90 times over a 30-question run, and the load time would land inside
    # `rerank_s` and be reported as the cross-encoder's per-query cost.
    rerankers: dict[str, Reranker | None] = {}
    for arm in arms:
        if arm.rerank not in rerankers:
            rerankers[arm.rerank] = build_reranker(settings, arm.rerank)

    for i, question in enumerate(questions):
        shift = i % len(arms)
        rotated = list(arms[shift:]) + list(arms[:shift])
        for position, arm in enumerate(rotated):
            run = await run_question(
                question, retriever, generator, judge, settings,
                k=arm.k or k, channels=arm.channels, rrf_k=rrf_k,
                cache_path=cache_path,
                n_relevant_total=n_relevant.get(question.id),
                reranker=rerankers[arm.rerank],
                rerank_depth=arm.depth,
            )
            run = replace(run, arm_label=arm.label, arm_index=position)
            reports[arm.label].runs.append(run)
            if on_question is not None:
                on_question(arm.label, run)

    for report in reports.values():
        _aggregate(report)
    return reports


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

    ranked = [r.rank for r in runs if r.rank is not None]
    report.mrr_n = len(ranked)
    report.mrr = (sum(x.mrr for x in ranked) / len(ranked)) if ranked else None
    # recall/ndcg have a narrower denominator than mrr: both are None where the
    # corpus-wide relevant count was unavailable, and averaging over a different set of
    # questions than mrr is exactly why `rank_n` is reported separately from `mrr_n`.
    recalls = [x.recall for x in ranked if x.recall is not None]
    ndcgs = [x.ndcg for x in ranked if x.ndcg is not None]
    report.rank_n = len(recalls)
    report.mean_recall = (sum(recalls) / len(recalls)) if recalls else None
    report.mean_ndcg = (sum(ndcgs) / len(ndcgs)) if ndcgs else None

    report.marker_compliance = metrics.marker_compliance_rate(
        [r.marker_compliance for r in runs]
    )
    report.marker_compliance_n = sum(1 for r in runs if r.marker_compliance is not None)
    report.degraded_claims = sum(r.degraded_claims for r in runs)
    report.invented_spans = sum(r.invented_spans for r in runs)
