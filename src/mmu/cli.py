"""The ``mmu`` command: build an index, validate the questions, run the benchmark.

Index building lives here and not behind an HTTP route on purpose — it re-embeds the
whole corpus, which takes minutes, and an endpoint that does that holds a connection
past every proxy timeout. ``GET /health`` reports a missing or stale index instead.

``questions validate`` is a **hard gate** that exits non-zero. It is cheap to run and it
prevents the most expensive failure mode this harness has: a typo in a ``doc_id`` makes
every per-document metric read 0.0 for that question, which is indistinguishable from a
model failure and gets debugged in entirely the wrong place.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mmu.config import get_settings


def _embedder(settings):
    """Deferred: importing torch costs seconds, and `questions validate` never needs it."""
    from mmu.embed.bge_m3 import BgeM3Embedder

    return BgeM3Embedder(
        model_name=settings.embed_model,
        device=settings.embed_device,
        batch_size=settings.embed_batch_size,
    )


# ------------------------------------------------------------------------------ index


def cmd_index_build(args) -> int:
    from mmu.retrieve.build import build_index

    settings = get_settings()
    warning = settings.context_budget_warning(args.k)
    if warning:
        print(f"WARNING: {warning}\n", file=sys.stderr)

    def progress(done: int, total: int) -> None:
        if args.verbose:
            print(f"  embedding {done}/{total} chunks", end="\r", file=sys.stderr)

    print(f"corpus: {settings.corpus_dir}")
    store, dense, sparse, warnings = build_index(
        settings.corpus_dir,
        settings.index_dir,
        _embedder(settings),
        max_tokens=settings.chunk_max_tokens,
        min_tokens=settings.chunk_min_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        batch_size=settings.embed_batch_size,
        on_batch=progress,
    )
    if args.verbose:
        print(file=sys.stderr)

    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    print(f"\nindexed {len(store)} chunks from {len(store.doc_ids())} document(s)")
    print(f"written to {settings.index_dir}\n")
    _print_doc_table(store)
    return 0


def _print_doc_table(store) -> None:
    """The doc_id table. These slugs are what data/questions.jsonl must refer to."""
    counts = store.counts_by_doc()
    by_doc = {c.doc_id: c.filename for c in store.chunks}
    width = max((len(d) for d in counts), default=8)
    print(f"{'doc_id':<{width}}  {'chunks':>6}  filename")
    print(f"{'-' * width}  {'-' * 6}  {'-' * 30}")
    for doc_id, n in sorted(counts.items()):
        print(f"{doc_id:<{width}}  {n:>6}  {by_doc[doc_id]}")
    print("\nUse these doc_id values in data/questions.jsonl.")


def cmd_index_stats(args) -> int:
    from mmu.retrieve.store import ChunkStore

    settings = get_settings()
    try:
        store = ChunkStore.load(settings.index_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if store.manifest is not None:
        m = store.manifest
        print(f"built {m.built_at} with {m.embed_model} (dim {m.dim})")
        print(
            f"chunking: max={m.chunk_max_tokens} min={m.chunk_min_tokens} "
            f"overlap={m.chunk_overlap_tokens}\n"
        )
    _print_doc_table(store)
    return 0


# -------------------------------------------------------------------------- questions


def cmd_questions_validate(args) -> int:
    from mmu.eval.questions import load_questions, validate_questions
    from mmu.retrieve.store import ChunkStore

    settings = get_settings()
    try:
        questions = load_questions(settings.questions_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    try:
        store = ChunkStore.load(settings.index_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report = validate_questions(
        questions,
        known_doc_ids=store.doc_ids(),
        doc_texts={d: store.text_of_doc(d) for d in store.doc_ids()},
    )
    for w in report.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    for e in report.errors:
        print(f"ERROR: {e}", file=sys.stderr)

    if not report.ok:
        print(
            f"\n{len(report.errors)} error(s). These are a hard gate: an unknown doc_id "
            f"scores 0.0 on every per-document metric, which looks exactly like a model "
            f"failure.",
            file=sys.stderr,
        )
        return 1
    print(f"{len(questions)} question(s) OK against {len(store.doc_ids())} document(s).")
    cross = [q for q in questions if q.is_cross_document]
    print(f"{len(cross)} count toward the Cross-Document Reasoning average.")
    return 0


# ------------------------------------------------------------------------------- eval


def _build_judge(settings, backend: str):
    from mmu.eval.judge import LocalJudge

    if backend == "local":
        return LocalJudge()
    from mmu.eval.judge import LlmJudge
    from mmu.generate.registry import build_generator

    return LlmJudge(build_generator(settings), fallback=LocalJudge())


def cmd_eval_run(args) -> int:
    from mmu.eval import report as report_mod
    from mmu.eval.questions import load_questions
    from mmu.eval.runner import run_eval
    from mmu.generate.registry import build_generator
    from mmu.retrieve.build import load_index
    from mmu.retrieve.hybrid import HybridRetriever

    settings = get_settings()
    questions = load_questions(settings.questions_path)
    embedder = _embedder(settings)
    store, dense, sparse = load_index(settings.index_dir, embedder)
    retriever = HybridRetriever(store, dense, sparse, embedder)

    try:
        generator = build_generator(settings, args.generator)
    except Exception as exc:  # noqa: BLE001 — actionable message, not a traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    channels = tuple(c.strip() for c in args.channels.split(",") if c.strip())
    judge = _build_judge(settings, args.judge)

    def on_question(run) -> None:
        if args.verbose:
            # CP is None, not 0.0, for a question with no relevant set — see
            # QuestionRun.cp in eval/runner.py. Formatting it with `:.2f` raised
            # TypeError and killed the whole run at the unanswerable question, after
            # every other question had already been generated and paid for. "--" here
            # matches report.py's `cell()`: a dash reads as "not applicable", which is
            # what it is, where nan would read as a computation that went wrong.
            cp = f"{run.cp:.2f}" if run.cp is not None else "  --"
            print(f"  {run.question_id:<18} AF={run.af:.2f} CP={cp} "
                  f"t={run.latency_s:.1f}s", file=sys.stderr)

    result = asyncio.run(
        run_eval(
            questions, retriever, generator, judge, settings,
            k=args.k, channels=channels, rrf_k=args.rrf_k,
            cache_path=settings.cache_dir / "judgments.jsonl",
            on_question=on_question,
        )
    )
    print(report_mod.render(result, verbose=args.verbose))
    if args.out:
        out = Path(args.out)
        report_mod.save(result, out)
        print(f"\nwritten to {out}")
    return 0


def _parse_arm(spec: str):
    """``label=dense,sparse[:rerank=alias][:k=N][:depth=N]`` -> an :class:`Arm`.

    The optional suffixes are what make a reranking arm expressible on the command
    line. ``rerank`` takes a registry *alias*, never a model id — see
    :mod:`mmu.retrieve.registry`.
    """
    from mmu.eval.runner import Arm

    head, *options = spec.split(":")
    label, _, channels = head.partition("=")
    label = label.strip()
    chans = tuple(c.strip() for c in channels.split(",") if c.strip())
    if not label or not chans:
        raise ValueError(
            f"--arm expects label=channel[,channel], got {spec!r} "
            f"(e.g. --arm hybrid=dense,sparse or "
            f"--arm reranked=dense,sparse:rerank=default:k=3:depth=32)"
        )

    rerank: str | None = None
    select: str | None = None
    k: int | None = None
    depth: int | None = None
    for option in options:
        name, _, value = option.partition("=")
        name, value = name.strip(), value.strip()
        if not value:
            raise ValueError(f"--arm option {option!r} needs a value, in {spec!r}")
        if name == "rerank":
            rerank = value
        elif name == "select":
            select = value
        elif name in ("k", "depth"):
            try:
                number = int(value)
            except ValueError:
                raise ValueError(f"--arm {name} must be an integer, got {value!r}") from None
            if number < 1:
                raise ValueError(f"--arm {name} must be >= 1, got {number}")
            if name == "k":
                k = number
            else:
                depth = number
        else:
            raise ValueError(
                f"unknown --arm option {name!r} in {spec!r}; "
                f"valid options are rerank, select, k, depth"
            )
    return Arm(label, chans, rerank=rerank, select=select, k=k, depth=depth)


def cmd_eval_ablate(args) -> int:
    """Run several retrieval configurations question-major in one process.

    This is the ablation that can support a statistical claim. ``eval run`` executed
    three times writes three reports whose latency columns differ mostly by what the
    host was doing, which is how this project initially read a 3x retrieval speedup off
    pure generator variance. Here every arm answers the same question seconds apart on
    the same loaded model, so the differences are paired and ``eval compare`` can print
    an exact sign test over them.
    """
    from mmu.eval import report as report_mod
    from mmu.eval.questions import load_questions
    from mmu.eval.runner import DEFAULT_ARMS, RERANK_ARMS, SELECT_ARMS, run_interleaved
    from mmu.generate.registry import build_generator
    from mmu.retrieve.build import load_index
    from mmu.retrieve.hybrid import HybridRetriever

    settings = get_settings()
    try:
        if args.arm:
            arms = tuple(_parse_arm(a) for a in args.arm)
        elif getattr(args, "select", False):
            arms = SELECT_ARMS
        elif getattr(args, "rerank", False):
            arms = RERANK_ARMS
        else:
            arms = DEFAULT_ARMS
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    questions = load_questions(settings.questions_path)
    embedder = _embedder(settings)
    store, dense, sparse = load_index(settings.index_dir, embedder)
    retriever = HybridRetriever(store, dense, sparse, embedder)

    try:
        generator = build_generator(settings, args.generator)
    except Exception as exc:  # noqa: BLE001 — actionable message, not a traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    judge = _build_judge(settings, args.judge)
    labels = [a.label for a in arms]
    print(
        f"{len(questions)} question(s) x {len(arms)} arm(s) = "
        f"{len(questions) * len(arms)} generations, interleaved: {', '.join(labels)}"
    )

    def on_question(arm_label, run) -> None:
        if args.verbose:
            cp = f"{run.cp:.2f}" if run.cp is not None else "  --"
            print(
                f"  [{run.arm_index}] {arm_label:<12} {run.question_id:<18} "
                f"AF={run.af:.2f} CP={cp} t={run.service_s:.1f}s",
                file=sys.stderr,
            )

    reports = asyncio.run(
        run_interleaved(
            questions, retriever, generator, judge, settings,
            arms=arms, k=args.k, rrf_k=args.rrf_k,
            cache_path=settings.cache_dir / "judgments.jsonl",
            on_question=on_question,
        )
    )

    out_dir = Path(args.out_dir)
    paths = []
    for label in labels:
        path = out_dir / f"{label}.json"
        report_mod.save(reports[label], path)
        paths.append(path)

    as_dicts = [report_mod.load(p) for p in paths]
    print()
    print(report_mod.render_comparison(as_dicts, labels))
    print()
    print(report_mod.render_paired(as_dicts, labels))
    print(f"\nwritten to {out_dir}/" + "{" + ",".join(labels) + "}.json")
    return 0


def cmd_eval_compare(args) -> int:
    from mmu.eval import report as report_mod

    reports = [report_mod.load(Path(p)) for p in args.reports]
    labels = [Path(p).stem for p in args.reports]
    print(report_mod.render_comparison(reports, labels))
    # Only for a genuine interleaved cohort; render_paired refuses otherwise, and that
    # refusal is worth printing rather than swallowing — it names the reason.
    if report_mod.paired_cohort_error(reports, labels) is None:
        print()
        print(report_mod.render_paired(reports, labels))
    return 0


# ------------------------------------------------------------------------------ parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmu", description="Modular Memory Unit")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="build or inspect the index").add_subparsers(
        dest="index_command", required=True
    )
    build = index.add_parser("build", help="build the index from data/corpus/")
    build.add_argument("--verbose", "-v", action="store_true")
    build.add_argument("--k", type=int, default=None, help="k to check the context budget against")
    build.set_defaults(func=cmd_index_build)

    stats = index.add_parser("stats", help="print the doc_id table")
    stats.set_defaults(func=cmd_index_stats)

    questions = sub.add_parser("questions", help="work with data/questions.jsonl").add_subparsers(
        dest="questions_command", required=True
    )
    validate = questions.add_parser("validate", help="hard gate against the built index")
    validate.set_defaults(func=cmd_questions_validate)

    ev = sub.add_parser("eval", help="run or compare benchmarks").add_subparsers(
        dest="eval_command", required=True
    )
    run = ev.add_parser("run", help="run the five-metric benchmark")
    run.add_argument("--judge", default="local", choices=["local", "ollama", "anthropic"])
    run.add_argument("--channels", default="dense,sparse",
                     help="comma-separated: dense,sparse (the ablation switch)")
    run.add_argument("--k", type=int, default=None)
    run.add_argument("--rrf-k", dest="rrf_k", type=int, default=None)
    run.add_argument("--generator", default=None, help="alias: default | fast")
    run.add_argument("--out", default=None)
    run.add_argument("--verbose", "-v", action="store_true")
    run.set_defaults(func=cmd_eval_run)

    ablate = ev.add_parser(
        "ablate",
        help="run several channel configurations interleaved, question-major, and pair them",
    )
    ablate.add_argument("--arm", action="append", default=None,
                        help="label=dense,sparse — repeatable; defaults to the "
                             "standard hybrid/dense-only/sparse-only three-way")
    ablate.add_argument("--judge", default="local", choices=["local", "ollama", "anthropic"])
    ablate.add_argument("--k", type=int, default=None)
    ablate.add_argument("--rrf-k", dest="rrf_k", type=int, default=None)
    ablate.add_argument("--generator", default=None, help="alias: default | fast")
    ablate.add_argument("--out-dir", dest="out_dir", default="reports/ablate")
    ablate.add_argument(
        "--rerank", action="store_true",
        help="use the reranking cohort (hybrid-k8 / hybrid-rerank-k8 / hybrid-rerank-k3) "
             "instead of the channel ablation; ignored when --arm is given",
    )
    ablate.add_argument(
        "--select", action="store_true",
        help="use the diversity-selection cohort (hybrid-k8 / rerank-k3 / mmr-k3 / "
             "quota-k3); ignored when --arm is given",
    )
    ablate.add_argument("--verbose", "-v", action="store_true")
    ablate.set_defaults(func=cmd_eval_ablate)

    compare = ev.add_parser("compare", help="print runs side by side; never names a winner")
    compare.add_argument("reports", nargs="+")
    compare.set_defaults(func=cmd_eval_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
