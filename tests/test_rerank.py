"""The reranking post-stage: ordering, the audit trail, and the registry boundary.

No cross-encoder is loaded here. `sentence_transformers` is imported lazily inside
`CrossEncoderReranker._load`, so these tests exercise the staging logic with a fake
scorer and stay inside the offline suite's budget — the real model is exercised by
`scripts/rerank_headroom.py` and the live gates.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from mmu.config import Settings
from mmu.core.models import SearchResult, Timings
from mmu.eval.runner import RERANK_ARMS, Arm
from mmu.retrieve.registry import available_rerankers, build_reranker
from mmu.retrieve.rerank import Reranker, rerank

from .conftest import make_chunk, make_retrieved
from .test_e2e import built  # noqa: F401  (fixture used by the interleaved tests)


class ScriptedReranker:
    """Returns a fixed score per chunk text, so the expected order is unambiguous."""

    def __init__(self, scores: dict[str, float], *, label: str = "scripted") -> None:
        self._scores = scores
        self._label = label
        self.calls: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return self._label

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls.append((query, len(texts)))
        return [self._scores.get(t, 0.0) for t in texts]


def _result(*texts: str, query: str = "what is the deadline") -> SearchResult:
    chunks = make_retrieved([make_chunk("gdpr", i, t) for i, t in enumerate(texts)])
    return SearchResult(query=query, k=len(chunks), chunks=chunks,
                        timings=Timings(total_ms=12.0))


def test_a_scripted_reranker_satisfies_the_protocol() -> None:
    assert isinstance(ScriptedReranker({}), Reranker)


# --------------------------------------------------------------------------- ordering


def test_rerank_promotes_the_highest_scored_chunk_and_truncates() -> None:
    res = _result("irrelevant", "also irrelevant", "the 72 hour rule")
    out = rerank(res, ScriptedReranker({"the 72 hour rule": 9.0}), k=2)

    assert [c.text for c in out.chunks] == ["the 72 hour rule", "irrelevant"]
    assert [c.rank for c in out.chunks] == [1, 2]
    assert out.k == 2


def test_rerank_can_rescue_a_chunk_from_outside_the_old_context_window() -> None:
    """The measured headroom, as arithmetic: five of the seed questions have nothing
    relevant in the top 8 while the relevant chunk sits at fused rank 10-19. Promoting
    it is the entire mechanism by which reranking can help here."""
    pool = _result(*[f"filler {i}" for i in range(9)], "the relevant clause")
    assert [c.text for c in pool.chunks][:8] == [f"filler {i}" for i in range(8)]

    out = rerank(pool, ScriptedReranker({"the relevant clause": 5.0}), k=8)
    assert out.chunks[0].text == "the relevant clause"
    assert len(out.chunks) == 8


def test_a_tie_keeps_the_fused_order() -> None:
    """A cross-encoder that cannot separate two candidates must not be allowed to
    permute them: falling back to the previous stage's opinion keeps the function
    deterministic, which a benchmark depends on."""
    res = _result("first", "second", "third")
    out = rerank(res, ScriptedReranker({}), k=3)   # every score 0.0
    assert [c.text for c in out.chunks] == ["first", "second", "third"]


def test_rerank_is_not_destructive_to_its_input() -> None:
    res = _result("a", "b", "c")
    before = [(c.text, c.rank) for c in res.chunks]
    rerank(res, ScriptedReranker({"c": 1.0}), k=1)
    assert [(c.text, c.rank) for c in res.chunks] == before


def test_an_empty_result_is_returned_unchanged() -> None:
    empty = SearchResult(query="q", k=0, chunks=[], timings=Timings())
    assert rerank(empty, ScriptedReranker({}), k=8) is empty


def test_k_larger_than_the_pool_still_reorders_without_padding() -> None:
    res = _result("low", "high")
    out = rerank(res, ScriptedReranker({"high": 3.0}), k=8)
    assert [c.text for c in out.chunks] == ["high", "low"]
    assert out.k == 2


# ------------------------------------------------------------------- the audit trail


def test_the_channel_ranks_survive_reranking() -> None:
    """`dense_rank`/`sparse_rank` record which channel found a chunk and where — the
    question the whole ablation exists to answer. Reranking changes a chunk's position
    in the context; it must not rewrite the history of how the chunk got into the pool.
    """
    res = _result("a", "b", "c")
    original = {c.text: (c.dense_rank, c.sparse_rank) for c in res.chunks}
    out = rerank(res, ScriptedReranker({"c": 2.0, "a": 1.0}), k=3)
    for c in out.chunks:
        assert (c.dense_rank, c.sparse_rank) == original[c.text], c.text


def test_both_scores_are_kept_so_a_promotion_is_provable() -> None:
    res = _result("filler", "the clause")
    rrf_before = {c.text: c.rrf_score for c in res.chunks}
    out = rerank(res, ScriptedReranker({"the clause": 7.5}), k=2)

    promoted = out.chunks[0]
    assert promoted.text == "the clause"
    assert promoted.rerank_score == 7.5           # why it was promoted
    assert promoted.rrf_score == rrf_before["the clause"]   # and where it came from


def test_rerank_score_is_none_when_no_reranker_ran() -> None:
    """Distinguishable from a reranker that ran and scored 0.0 — the same
    undefined-versus-zero distinction the scoring metrics already depend on."""
    res = _result("a", "b")
    assert all(c.rerank_score is None for c in res.chunks)
    out = rerank(res, ScriptedReranker({}), k=2)
    assert all(c.rerank_score == 0.0 for c in out.chunks)


def test_the_cross_encoder_cost_is_recorded_on_its_own_timing_line() -> None:
    """Charging it to retrieval or generation is how this project previously reported
    a 3x speedup that did not exist."""
    res = _result("a", "b")
    out = rerank(res, ScriptedReranker({}), k=2)
    assert out.timings.rerank_ms >= 0.0
    assert out.timings.total_ms == res.timings.total_ms  # retrieval cost untouched


def test_a_reranker_returning_the_wrong_number_of_scores_raises() -> None:
    class Broken:
        name = "broken"

        def score(self, query, texts):
            return [1.0]

    with pytest.raises(ValueError, match="positionally aligned"):
        rerank(_result("a", "b", "c"), Broken(), k=3)


# ---------------------------------------------------------------- registry boundary


def test_the_registry_resolves_only_aliases_never_model_ids() -> None:
    settings = Settings()
    with pytest.raises(ValueError, match="unknown reranker alias"):
        build_reranker(settings, "BAAI/bge-reranker-v2-m3")
    assert "default" in available_rerankers()


def test_disabled_aliases_mean_no_reranker_rather_than_an_error() -> None:
    settings = Settings()
    assert build_reranker(settings, "none") is None
    assert build_reranker(settings, "off") is None
    assert build_reranker(settings, None) is None


def test_an_unknown_alias_raises_instead_of_silently_not_reranking() -> None:
    """An arm labelled as reranked that quietly did not rerank would turn the whole
    comparison into a measurement of nothing, invisibly."""
    with pytest.raises(ValueError):
        build_reranker(Settings(), "crossencoder")


def test_building_a_reranker_does_not_import_torch() -> None:
    import sys

    sys.modules.pop("sentence_transformers", None)
    reranker = build_reranker(Settings(), "default")
    assert reranker is not None
    assert reranker.name == Settings().rerank_model
    assert "sentence_transformers" not in sys.modules


# ------------------------------------------------------------------------ arm design


def test_every_rerank_arm_shares_one_candidate_depth() -> None:
    """`Settings.resolved_depth` is max(4k, 20), so leaving depth unset would give the
    k=3 arm a 20-candidate pool and the k=8 arms a 32-candidate one — the arms would
    differ in two ways and the comparison would not isolate reranking."""
    depths = {arm.depth for arm in RERANK_ARMS}
    assert depths == {32}


def test_the_rerank_cohort_separates_quality_from_prompt_length() -> None:
    """Three arms, not two: a single "rerank and shrink" arm that came back faster and
    worse could not say which of the two changes did which."""
    by_label = {arm.label: arm for arm in RERANK_ARMS}
    assert by_label["hybrid-k8"].rerank is None
    assert by_label["hybrid-rerank-k8"].rerank == "default"
    assert by_label["hybrid-rerank-k8"].k == by_label["hybrid-k8"].k == 8
    assert by_label["hybrid-rerank-k3"].k == 3
    assert RERANK_ARMS[0].label == "hybrid-k8"  # baseline is first, for the paired table


def test_an_arm_defaults_to_no_reranking() -> None:
    arm = Arm("plain", ("dense", "sparse"))
    assert arm.rerank is None and arm.k is None and arm.depth is None


# ------------------------------------------------------- through the interleaved run


async def test_run_interleaved_applies_the_reranker_and_charges_it_separately(
    tmp_path, built, monkeypatch
) -> None:
    """The wiring, end to end: the reranked arms must actually rerank, the control must
    not, and the cross-encoder's cost must land in `rerank_s` rather than `retrieval_s`.
    """
    import mmu.eval.runner as runner_mod
    from mmu.config import Settings as S
    from mmu.eval.judge import LocalJudge
    from mmu.eval.questions import load_questions
    from mmu.retrieve.hybrid import HybridRetriever

    from .conftest import FakeGenerator
    from .test_e2e import QUESTIONS

    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(QUESTIONS, encoding="utf-8")

    scripted = ScriptedReranker({}, label="fake-cross-encoder")
    monkeypatch.setattr(
        runner_mod, "build_reranker",
        lambda settings, alias: scripted if alias else None,
    )

    arms = (
        runner_mod.Arm("plain-k3", ("dense", "sparse"), k=3, depth=3),
        runner_mod.Arm("reranked-k3", ("dense", "sparse"), rerank="default", k=3, depth=3),
    )
    reports = await runner_mod.run_interleaved(
        load_questions(path), retriever, FakeGenerator(), LocalJudge(),
        S(data_dir=tmp_path), arms=arms, cache_path=tmp_path / "j.jsonl",
    )

    plain, reranked = reports["plain-k3"], reports["reranked-k3"]
    assert plain.rerank == "" and reranked.rerank == "default"
    assert reranked.rerank_depth == 3

    # The control never called the reranker; the reranked arm called it once per question.
    assert len(scripted.calls) == len(reranked.runs)
    assert all(r.rerank_s == 0.0 for r in plain.runs)
    assert all(r.rerank_s > 0.0 for r in reranked.runs)
    # And the cost is counted in service_s, which is what the paired latency test uses.
    for r in reranked.runs:
        assert r.service_s == pytest.approx(r.retrieval_s + r.rerank_s + r.generate_s)


async def test_a_reranked_arm_records_its_own_k_not_the_run_default(
    tmp_path, built, monkeypatch
) -> None:
    """The shrink-the-context experiment depends on each arm's report carrying the k it
    actually used; a shared k would make the two arms indistinguishable in the output."""
    import mmu.eval.runner as runner_mod
    from mmu.config import Settings as S
    from mmu.eval.judge import LocalJudge
    from mmu.eval.questions import load_questions
    from mmu.retrieve.hybrid import HybridRetriever

    from .conftest import FakeGenerator
    from .test_e2e import QUESTIONS

    embedder, _, store, dense, sparse, _ = built
    retriever = HybridRetriever(store, dense, sparse, embedder)
    path = tmp_path / "questions.jsonl"
    path.write_text(QUESTIONS, encoding="utf-8")
    monkeypatch.setattr(
        runner_mod, "build_reranker",
        lambda settings, alias: ScriptedReranker({}) if alias else None,
    )

    arms = (
        runner_mod.Arm("wide", ("dense", "sparse"), k=3, depth=3),
        runner_mod.Arm("narrow", ("dense", "sparse"), rerank="default", k=1, depth=3),
    )
    reports = await runner_mod.run_interleaved(
        load_questions(path), retriever, FakeGenerator(), LocalJudge(),
        S(data_dir=tmp_path), arms=arms, cache_path=tmp_path / "j.jsonl", k=3,
    )
    assert reports["wide"].k == 3
    assert reports["narrow"].k == 1
    assert all(len(r.retrieved_doc_ids) >= 1 for r in reports["narrow"].runs)
    assert all(r.k == 1 for r in reports["narrow"].runs)


# ------------------------------------------------------------------ the --arm grammar


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("hybrid=dense,sparse", ("hybrid", ("dense", "sparse"), None, None, None)),
        ("d=dense:rerank=default", ("d", ("dense",), "default", None, None)),
        ("r=dense,sparse:rerank=bge:k=3:depth=32",
         ("r", ("dense", "sparse"), "bge", 3, 32)),
        ("r=sparse:depth=64:k=5", ("r", ("sparse",), None, 5, 64)),
    ],
)
def test_parse_arm_accepts_the_documented_grammar(spec, expected) -> None:
    from mmu.cli import _parse_arm

    arm = _parse_arm(spec)
    assert (arm.label, arm.channels, arm.rerank, arm.k, arm.depth) == expected


@pytest.mark.parametrize(
    "spec",
    [
        "noequals",
        "label=",
        "=dense",
        "x=dense:unknown=1",
        "x=dense:k=0",
        "x=dense:k=-2",
        "x=dense:k=notanumber",
        "x=dense:k=",
        "x=dense:depth=0",
    ],
)
def test_parse_arm_rejects_malformed_specs(spec) -> None:
    """Every rejection here would otherwise become a silently-misconfigured arm, and a
    misconfigured arm produces a number that looks like a result."""
    from mmu.cli import _parse_arm

    with pytest.raises(ValueError):
        _parse_arm(spec)


def test_the_rerank_flag_selects_the_rerank_cohort() -> None:
    from mmu.cli import build_parser

    args = build_parser().parse_args(["eval", "ablate", "--rerank"])
    assert args.rerank is True
    assert args.arm is None


def test_an_explicit_arm_list_overrides_the_rerank_flag() -> None:
    """Documented in the flag's help text; asserted so the precedence cannot drift."""
    from mmu.cli import build_parser

    args = build_parser().parse_args(
        ["eval", "ablate", "--rerank", "--arm", "only=dense"]
    )
    assert args.arm == ["only=dense"]


# ------------------------------------- the paired latency column must pay for rerank


def test_report_service_time_matches_the_runner_field_by_field() -> None:
    """The bug this guards, which shipped once and was caught only by reconciling the
    report against its own JSON.

    `QuestionRun.service_s` is a dataclass property; `report._service_s` re-derives the
    same quantity from the serialized dict. When `rerank_s` was added to the first and
    not the second, the paired table omitted ~19s per query of real cross-encoder cost
    and reported a reranked arm as 45s faster when it was 26s faster. Both numbers
    favoured the same arm, which is exactly why nothing looked wrong.

    Asserting the *field list* rather than one example keeps a future stage from
    repeating it: adding a timing field to the runner and not to `_SERVICE_FIELDS`
    fails here.
    """
    from dataclasses import asdict

    from mmu.eval.report import _SERVICE_FIELDS, _service_s
    from mmu.eval.runner import QuestionRun

    run = QuestionRun(
        question_id="q", question="q?", answer_type="single_document", answer_text="a",
        model="m", k=3, latency_s=99.0, retrieval_s=0.34, generate_s=26.87,
        af=1.0, cp=None, xdr=None, cur=None, quality=1.0, marker_compliance=1.0,
        n_claims=0, n_supported=0, rerank_s=20.58,
    )
    assert run.service_s == pytest.approx(47.79)
    assert _service_s(asdict(run)) == pytest.approx(run.service_s)

    # Every field the report sums must exist on the runner's record, or it is silently
    # contributing 0.0 forever.
    serialized = asdict(run)
    for field in _SERVICE_FIELDS:
        assert field in serialized, field

    # And every nonzero timing stage the runner records must be in the report's sum.
    stages = {"retrieval_s", "rerank_s", "generate_s"}
    assert stages == set(_SERVICE_FIELDS), (
        "a timing stage exists on QuestionRun that the paired latency column ignores"
    )


def test_a_reranked_arm_is_charged_for_the_cross_encoder_in_the_paired_table() -> None:
    """End to end through `render_paired`: an arm that generates faster but pays a
    large reranking cost must not appear faster by the generation saving alone."""
    from mmu.eval.report import render_paired

    def run(qid, *, retrieval, rerank, generate):
        return {
            "question_id": qid, "retrieval_s": retrieval, "rerank_s": rerank,
            "generate_s": generate, "af": 1.0, "quality": 1.0, "rank": None,
        }

    base = {
        "interleaved_with": ["base", "arm"], "arm_label": "base",
        "runs": [run("q1", retrieval=0.3, rerank=0.0, generate=60.0)],
    }
    arm = {
        "interleaved_with": ["base", "arm"], "arm_label": "arm",
        "runs": [run("q1", retrieval=0.3, rerank=20.0, generate=20.0)],
    }
    out = render_paired([base, arm], ["base", "arm"])

    # 60.3 -> 40.3 is a 20s saving, not the 40s the generation column alone would show.
    assert "-20.00" in out
    assert "-40.00" not in out
