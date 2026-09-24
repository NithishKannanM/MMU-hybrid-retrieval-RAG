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
            print(f"  {run.question_id:<18} AF={run.af:.2f} CP={run.cp:.2f} "
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


def cmd_eval_compare(args) -> int:
    from mmu.eval import report as report_mod

    reports = [report_mod.load(Path(p)) for p in args.reports]
    labels = [Path(p).stem for p in args.reports]
    print(report_mod.render_comparison(reports, labels))
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

    compare = ev.add_parser("compare", help="print runs side by side; never names a winner")
    compare.add_argument("reports", nargs="+")
    compare.set_defaults(func=cmd_eval_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
