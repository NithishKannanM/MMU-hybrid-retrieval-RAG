"""The CLI, driven through its real entry points against a temporary data directory.

The one thing stubbed is :func:`mmu.cli._embedder`, which would otherwise download
BGE-M3. Everything else — argument parsing, the index build, the doc_id table, the hard
gate's exit codes, the eval run and the comparison — is the production path. Exit codes
matter here specifically: `mmu questions validate` is meant to be usable in CI, and a
gate that prints an error but exits 0 is not a gate.
"""

from __future__ import annotations

import json

import pytest

from mmu import cli
from mmu.config import Settings, get_settings
from mmu.embed.hashing import HashingEmbedder


@pytest.fixture
def data_dir(tmp_path, mini_corpus, monkeypatch):
    """A complete data/ tree: the mini corpus, plus questions matching it."""
    data = tmp_path / "data"
    (data / "corpus").mkdir(parents=True)
    for src in mini_corpus.iterdir():
        (data / "corpus" / src.name).write_text(src.read_text(), encoding="utf-8")
    # Two questions, not one, and the second is unanswerable on purpose. A seed set of
    # only answerable questions is a world where `QuestionRun.cp` is never None, and the
    # CLI's --verbose printer formatted it with `:.2f` and crashed the entire run at the
    # unanswerable question — after every other answer had been generated and paid for.
    # The fixture has to contain the case that breaks the code, or it asserts nothing.
    (data / "questions.jsonl").write_text(
        json.dumps(
            {
                "id": "xdr-001",
                "question": "How fast must we notify, and who owns it internally?",
                "answer_type": "cross_document",
                "required_doc_ids": ["gdpr", "company-policy"],
                "relevant_doc_ids": ["gdpr", "company-policy"],
                "must_contain": {
                    "gdpr": ["72 hours"],
                    "company-policy": ["Data Protection Officer"],
                },
                "expected_synthesis_keys": ["72 hours", "Data Protection Officer"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "unanswerable-001",
                "question": "What is the deductible on our cyber-liability insurance?",
                "answer_type": "unanswerable",
                "required_doc_ids": [],
                "relevant_doc_ids": [],
                "must_contain": {},
                "expected_synthesis_keys": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    get_settings.cache_clear()
    monkeypatch.setattr(
        cli, "get_settings", lambda: Settings(data_dir=data, chunk_max_tokens=120,
                                              chunk_min_tokens=20, chunk_overlap_tokens=16)
    )
    monkeypatch.setattr(cli, "_embedder", lambda settings: HashingEmbedder(dim=64))
    yield data
    get_settings.cache_clear()


def test_index_build_prints_the_doc_id_table(data_dir, capsys) -> None:
    assert cli.main(["index", "build", "--verbose"]) == 0
    out = capsys.readouterr().out
    # The doc_id table is the workflow-critical output: these slugs are what
    # questions.jsonl must refer to.
    assert "doc_id" in out
    for doc_id in ("gdpr", "national-law", "company-policy"):
        assert doc_id in out
    assert (data_dir / ".index" / "chunks.jsonl").exists()
    assert (data_dir / ".index" / "faiss.bin").exists()
    assert (data_dir / ".index" / "manifest.json").exists()


def test_index_stats_requires_a_built_index(data_dir, capsys) -> None:
    assert cli.main(["index", "stats"]) == 1
    assert "mmu index build" in capsys.readouterr().err

    cli.main(["index", "build"])
    assert cli.main(["index", "stats"]) == 0
    assert "gdpr" in capsys.readouterr().out


def test_questions_validate_is_a_hard_gate(data_dir, capsys) -> None:
    cli.main(["index", "build"])
    assert cli.main(["questions", "validate"]) == 0
    assert "OK" in capsys.readouterr().out

    # A single typo'd doc_id must FAIL the command, not merely warn.
    path = data_dir / "questions.jsonl"
    path.write_text(path.read_text().replace('"gdpr"', '"gdrp"'), encoding="utf-8")
    assert cli.main(["questions", "validate"]) == 1
    err = capsys.readouterr().err
    assert "gdrp" in err and "hard gate" in err


def test_eval_run_and_compare(data_dir, tmp_path, monkeypatch, capsys) -> None:
    cli.main(["index", "build"])
    capsys.readouterr()

    from tests.conftest import FakeGenerator

    monkeypatch.setattr(
        "mmu.generate.registry.build_generator",
        lambda settings, alias=None: FakeGenerator(
            reply="Notify within 72 hours [S1]; escalate to the Data Protection Officer [S2]."
        ),
    )

    hybrid = tmp_path / "hybrid.json"
    assert cli.main(["eval", "run", "--judge", "local", "--k", "6",
                     "--out", str(hybrid), "--verbose"]) == 0
    captured = capsys.readouterr()
    out = captured.out
    assert "Answer Faithfulness" in out and "Cov (retrieval)" in out
    assert "negation-blind" in out       # the local-judge caveat is never omitted
    assert hybrid.exists()

    # --verbose must survive a question whose Context Precision is undefined. This is
    # the regression guard: formatting None with `:.2f` raised TypeError and aborted
    # the run at its last question, losing every result already computed.
    assert "unanswerable-001" in captured.err
    assert "CP=  --" in captured.err

    # The ablation is a first-class CLI argument.
    dense_only = tmp_path / "dense-only.json"
    assert cli.main(["eval", "run", "--judge", "local", "--k", "6",
                     "--channels", "dense", "--out", str(dense_only)]) == 0
    assert json.loads(dense_only.read_text())["channels"] == ["dense"]
    capsys.readouterr()

    assert cli.main(["eval", "compare", str(hybrid), str(dense_only)]) == 0
    compare = capsys.readouterr().out
    assert "hybrid" in compare and "dense-only" in compare
    # It reports a noise band and deliberately does not name a winner.
    assert "noise band" in compare
    assert "winner" not in compare.lower() or "does not declare a winner" in compare


def test_eval_ablate_interleaves_and_pairs(data_dir, tmp_path, monkeypatch, capsys) -> None:
    """`eval ablate` is the command that exists so a latency comparison can mean
    something: it runs each arm against the same question back to back in one process,
    rather than as separate end-to-end runs whose latency columns differ by whatever the
    host was doing. It must write one report per arm and print the paired block."""
    cli.main(["index", "build"])
    capsys.readouterr()

    from tests.conftest import FakeGenerator

    monkeypatch.setattr(
        "mmu.generate.registry.build_generator",
        lambda settings, alias=None: FakeGenerator(
            reply="Notify within 72 hours [S1]; escalate to the Data Protection Officer [S2]."
        ),
    )

    out_dir = tmp_path / "ablate"
    assert cli.main([
        "eval", "ablate",
        "--arm", "hybrid=dense,sparse",
        "--arm", "dense-only=dense",
        "--judge", "local", "--k", "6",
        "--out-dir", str(out_dir), "--verbose",
    ]) == 0
    captured = capsys.readouterr()

    # One report per arm, each declaring the cohort it was interleaved against.
    for label in ("hybrid", "dense-only"):
        payload = json.loads((out_dir / f"{label}.json").read_text())
        assert payload["arm_label"] == label
        assert payload["interleaved_with"] == ["dense-only", "hybrid"]

    # 2 questions x 2 arms, and the rotation is visible in the verbose stream.
    assert "2 question(s) x 2 arm(s) = 4 generations" in captured.out
    assert captured.err.count("hybrid") == 2
    assert captured.err.count("dense-only") == 2

    # The paired block replaces the "not comparable" warning that sequential runs get.
    assert "PAIRED vs hybrid" in captured.out
    assert "service latency" in captured.out
    assert "p50/LpQU are NOT comparable" not in captured.out


def test_eval_compare_refuses_to_pair_separately_executed_runs(
    data_dir, tmp_path, monkeypatch, capsys
) -> None:
    """Two `eval run` invocations are not a paired experiment, and `compare` has to say
    so where it prints their latency columns — this is the exact misreading the project
    made: a 3x p50 gap that was generator variance between runs."""
    cli.main(["index", "build"])
    from tests.conftest import FakeGenerator

    monkeypatch.setattr(
        "mmu.generate.registry.build_generator",
        lambda settings, alias=None: FakeGenerator(reply="Within 72 hours [S1]."),
    )
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    cli.main(["eval", "run", "--judge", "local", "--k", "6", "--out", str(a)])
    cli.main(["eval", "run", "--judge", "local", "--k", "6", "--channels", "dense",
              "--out", str(b)])
    capsys.readouterr()

    assert cli.main(["eval", "compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "p50/LpQU are NOT comparable" in out
    assert "mmu eval ablate" in out
    assert "PAIRED vs" not in out          # and no sign test is computed over them
