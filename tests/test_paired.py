"""The interleaved ablation and its paired statistics.

These tests exist because of a measured mistake, not a hypothetical one. The project's
first ablation ran each retrieval configuration as its own end-to-end `eval run` and
differenced the corpus means. That read a "3x p50 speedup" for hybrid off pure generator
variance: across five sequential runs of the same model, throughput ranged 6.7-20.5
chars/s while retrieval was under 2% of wall clock, and a 3B model came out *slower*
than the fastest 7B run. So the two things covered here are (a) that the runner really
does interleave and counterbalance, and (b) that the reporting path refuses to compute a
paired test over runs that were not actually paired — because such a test would inherit
exactly the drift the pairing exists to remove and then attach a p-value to it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from mmu.config import Settings
from mmu.embed.hashing import HashingEmbedder
from mmu.eval import report as report_mod
from mmu.eval.judge import LocalJudge
from mmu.eval.metrics import paired_sign_test
from mmu.eval.questions import load_questions
from mmu.eval.runner import Arm, run_interleaved
from mmu.retrieve.build import build_index
from mmu.retrieve.hybrid import HybridRetriever
from tests.conftest import FakeGenerator

QUESTIONS = """\
{"id":"xdr-001","question":"How fast must we notify and who owns it?",\
"answer_type":"cross_document","required_doc_ids":["gdpr","company-policy"],\
"relevant_doc_ids":["gdpr","company-policy"],\
"must_contain":{"gdpr":["72 hours"],"company-policy":["Data Protection Officer"]},\
"expected_synthesis_keys":["72 hours","Data Protection Officer"]}
{"id":"single-001","question":"What is the statutory notification deadline?",\
"answer_type":"single_document","required_doc_ids":["gdpr"],"relevant_doc_ids":["gdpr"],\
"must_contain":{"gdpr":["72 hours"]},"expected_synthesis_keys":["72 hours"]}
{"id":"single-002","question":"Who owns external notifications?",\
"answer_type":"single_document","required_doc_ids":["company-policy"],\
"relevant_doc_ids":["company-policy"],\
"must_contain":{"company-policy":["Data Protection Officer"]},\
"expected_synthesis_keys":["Data Protection Officer"]}
"""


# ------------------------------------------------------------------ the sign test math


def test_all_pairs_agreeing_at_n6_is_the_best_this_design_can_do() -> None:
    t = paired_sign_test([1.0] * 6)
    assert (t.n_pairs, t.wins, t.losses, t.ties) == (6, 6, 0, 0)
    assert t.p_two_sided == pytest.approx(0.03125)
    # Unanimity at n=6 *is* the floor: there is no stronger outcome available.
    assert t.best_attainable_p == pytest.approx(0.03125)
    assert t.significant and t.conclusive_possible


def test_one_disagreeing_pair_destroys_significance_at_n6() -> None:
    # 5-1 is p=0.219. This is the whole reason the project does not claim results off
    # six questions: a single question flipping moves p by an order of magnitude.
    assert paired_sign_test([1.0] * 5 + [-1.0]).p_two_sided == pytest.approx(0.21875)


def test_a_design_that_cannot_reach_significance_reports_that_it_cannot() -> None:
    """n=5 is unwinnable, and that must be distinguishable from "no difference"."""
    t = paired_sign_test([1.0] * 5)
    assert t.wins == 5 and t.losses == 0          # a perfectly consistent effect
    assert t.p_two_sided == pytest.approx(0.0625)  # and still not significant
    assert not t.significant
    assert not t.conclusive_possible               # the design's fault, not the data's


def test_ties_are_dropped_and_shrink_the_attainable_floor() -> None:
    # Six questions, four of them scored identically by both arms. The test runs on
    # two pairs, where the best possible p is 0.5 — unanimous and meaningless.
    t = paired_sign_test([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    assert (t.n_pairs, t.ties) == (2, 4)
    assert t.best_attainable_p == pytest.approx(0.5)
    assert not t.conclusive_possible


def test_every_pair_tied_is_no_information_rather_than_a_null_result() -> None:
    t = paired_sign_test([0.0, 0.0, 0.0])
    assert t.n_pairs == 0 and t.p_two_sided == 1.0 and t.best_attainable_p == 1.0


def test_epsilon_absorbs_timer_jitter_instead_of_scoring_it_as_a_win() -> None:
    deltas = [0.002, -0.001, 0.003, 12.0, 9.0, 11.0]
    assert paired_sign_test(deltas, epsilon=0.0).n_pairs == 6        # jitter counted
    assert paired_sign_test(deltas, epsilon=0.05).n_pairs == 3       # jitter tied off


def test_the_test_is_exact_not_normal_approximated() -> None:
    # The normal approximation at n=6, 6 wins gives z=2.449 -> p=0.0143, which is less
    # than half the exact 0.03125 and would be reported as a stronger result than the
    # data can support. Asserting the exact value pins that down.
    assert paired_sign_test([1.0] * 6).p_two_sided == pytest.approx(2 / 64)


# -------------------------------------------------------------------- the runner itself


@pytest.fixture
def harness(tmp_path, mini_corpus):
    embedder = HashingEmbedder(dim=64)
    store, dense, sparse, _ = build_index(
        mini_corpus, tmp_path / ".index", embedder,
        max_tokens=120, min_tokens=20, overlap_tokens=16,
    )
    retriever = HybridRetriever(store, dense, sparse, embedder)
    qpath = tmp_path / "questions.jsonl"
    qpath.write_text(QUESTIONS, encoding="utf-8")
    questions = load_questions(qpath)
    settings = Settings(data_dir=tmp_path)
    return questions, retriever, settings


ARMS = (Arm("hybrid", ("dense", "sparse")), Arm("dense-only", ("dense",)),
        Arm("sparse-only", ("sparse",)))


def _interleave(harness, arms=ARMS, **kw):
    questions, retriever, settings = harness
    return asyncio.run(
        run_interleaved(
            questions, retriever,
            FakeGenerator(reply="Notify within 72 hours [S1]; the Data Protection "
                                "Officer owns it [S2]."),
            LocalJudge(), settings, arms=arms, k=6, **kw,
        )
    )


def test_every_arm_answers_every_question(harness) -> None:
    reports = _interleave(harness)
    assert set(reports) == {"hybrid", "dense-only", "sparse-only"}
    for label, report in reports.items():
        assert [r.question_id for r in report.runs] == [
            "xdr-001", "single-001", "single-002"
        ]
        assert report.arm_label == label
        assert report.channels == dict(
            (a.label, a.channels) for a in ARMS
        )[label]


def test_the_arm_order_rotates_so_no_arm_always_pays_the_cold_start(harness) -> None:
    """Within a question the leading arm absorbs warm-up; a fixed order would relabel
    that as an arm effect. Three arms over three questions must give each arm each
    position exactly once."""
    reports = _interleave(harness)
    positions = {
        label: sorted(r.arm_index for r in report.runs)
        for label, report in reports.items()
    }
    assert positions == {
        "hybrid": [0, 1, 2], "dense-only": [0, 1, 2], "sparse-only": [0, 1, 2]
    }
    # And per question, the three arms occupy three distinct positions.
    for qid in ("xdr-001", "single-001", "single-002"):
        taken = [
            next(r.arm_index for r in rep.runs if r.question_id == qid)
            for rep in reports.values()
        ]
        assert sorted(taken) == [0, 1, 2]


def test_each_report_records_the_whole_cohort_it_was_paired_against(harness) -> None:
    reports = _interleave(harness)
    for report in reports.values():
        assert report.interleaved_with == ("dense-only", "hybrid", "sparse-only")


def test_a_standalone_run_claims_no_cohort(harness) -> None:
    """`run_eval` must not produce something that looks paired, or the guard is moot."""
    from mmu.eval.runner import run_eval

    questions, retriever, settings = harness
    report = asyncio.run(
        run_eval(questions, retriever, FakeGenerator(), LocalJudge(), settings, k=6)
    )
    assert report.interleaved_with == () and report.arm_label == ""


def test_duplicate_arm_labels_are_rejected(harness) -> None:
    with pytest.raises(ValueError, match="unique"):
        _interleave(harness, arms=(Arm("a", ("dense",)), Arm("a", ("sparse",))))


def test_no_arms_is_rejected(harness) -> None:
    with pytest.raises(ValueError, match="at least one arm"):
        _interleave(harness, arms=())


def test_service_s_excludes_the_judge_time_that_arms_share(harness) -> None:
    """`latency_s` covers judging too, and the judgment cache is keyed by question_id —
    which the arms share — so the arm that runs second gets identical verdicts free.
    The paired comparison must use retrieval+generation, which no arm can get free."""
    reports = _interleave(harness, cache_path=harness[2].cache_dir / "j.jsonl")
    for report in reports.values():
        for run in report.runs:
            assert run.service_s == pytest.approx(run.retrieval_s + run.generate_s)
            assert run.service_s <= run.latency_s + 1e-9


# ---------------------------------------------------------------------- the report path


def _save_and_load(tmp_path, reports, labels):
    paths = []
    for label in labels:
        p = tmp_path / f"{label}.json"
        report_mod.save(reports[label], p)
        paths.append(p)
    return [report_mod.load(p) for p in paths]


def test_an_interleaved_cohort_renders_a_paired_table_with_a_sign_test(
    harness, tmp_path
) -> None:
    labels = ["hybrid", "dense-only", "sparse-only"]
    dicts = _save_and_load(tmp_path, _interleave(harness), labels)

    assert report_mod.paired_cohort_error(dicts, labels) is None
    out = report_mod.render_paired(dicts, labels)
    assert "PAIRED vs hybrid" in out
    assert "dense-only vs hybrid" in out and "sparse-only vs hybrid" in out
    for metric in ("service latency", "Answer Faithfulness", "quality (composite)"):
        assert metric in out
    assert "p=" in out
    # Every question appears as its own row — the pairing has to be auditable.
    for qid in ("xdr-001", "single-001", "single-002"):
        assert qid in out
    # And it still refuses to name a winner.
    assert "winner" not in out.lower()


def test_the_paired_table_says_so_when_n_makes_significance_impossible(
    harness, tmp_path
) -> None:
    """Three questions can reach at best p=0.25. The report must call that a power
    failure rather than print a non-significant p and let it read as "no difference"."""
    labels = ["hybrid", "dense-only"]
    reports = _interleave(harness)
    dicts = _save_and_load(tmp_path, reports, labels)
    out = report_mod.render_paired(dicts, labels)
    assert "CANNOT reach p<0.05" in out
    assert "not evidence of no difference" in out


def test_runs_that_were_not_interleaved_are_refused_a_paired_test(tmp_path) -> None:
    """The guard that matters: pairing sequentially-executed runs would reintroduce the
    exact between-run drift that produced this project's false 3x latency claim."""
    a = {"runs": [{"question_id": "q1", "af": 1.0, "quality": 1.0,
                   "retrieval_s": 0.3, "generate_s": 20.0}],
         "interleaved_with": []}
    b = json.loads(json.dumps(a))
    labels = ["a", "b"]
    problem = report_mod.paired_cohort_error([a, b], labels)
    assert problem is not None and "not interleaved" in problem
    assert "UNAVAILABLE" in report_mod.render_paired([a, b], labels)


def test_a_cohort_covering_different_questions_is_refused(tmp_path) -> None:
    mk = lambda qid: {  # noqa: E731
        "runs": [{"question_id": qid, "af": 1.0, "quality": 1.0,
                  "retrieval_s": 0.1, "generate_s": 1.0}],
        "interleaved_with": ["a", "b"], "arm_label": "a",
    }
    err = report_mod.paired_cohort_error([mk("q1"), mk("q2")], ["a", "b"])
    assert err is not None and "same questions" in err


def test_a_subset_of_a_cohort_may_still_be_paired(harness, tmp_path) -> None:
    """Running three arms then comparing two of them is legitimate — they were still
    executed question-major against each other."""
    reports = _interleave(harness)
    dicts = _save_and_load(tmp_path, reports, ["hybrid", "sparse-only"])
    assert report_mod.paired_cohort_error(dicts, ["hybrid", "sparse-only"]) is None


def test_the_paired_delta_follows_service_time_not_total_latency(tmp_path) -> None:
    """Hand-built so the two disagree: arm `b` has the larger `latency_s` but the
    smaller retrieval+generate. A table reading `latency_s` would report b as slower."""
    def rep(label, ret, gen, lat):
        return {
            "runs": [{"question_id": "q1", "af": 1.0, "quality": 1.0,
                      "retrieval_s": ret, "generate_s": gen, "latency_s": lat}],
            "interleaved_with": ["a", "b"], "arm_label": label,
        }
    a = rep("a", 0.5, 10.0, 11.0)     # service 10.5, total 11.0
    b = rep("b", 0.5, 4.0, 99.0)      # service  4.5, total 99.0 (judge-dominated)
    out = report_mod.render_paired([a, b], ["a", "b"])
    assert "-6.00" in out             # 4.5 - 10.5, not 99.0 - 11.0
    assert "b higher in 0, lower in 1" in out


def test_compare_warns_that_latency_is_incomparable_across_separate_runs() -> None:
    sequential = {
        "af": 0.7, "cp": 0.8, "cov_mean": 1.0, "cite_mean": 0.8, "syn_mean": 1.0,
        "xdr_mean": 0.8, "cur_cited": 0.4, "cur_attr": 0.0, "p50_latency_s": 18.8,
        "corpus_lpqu": 30.0, "runs": [{"question_id": "q1"}], "interleaved_with": [],
    }
    other = dict(sequential, p50_latency_s=61.8)
    out = report_mod.render_comparison([sequential, other], ["hybrid", "dense-only"])
    assert "p50/LpQU are NOT comparable" in out
    assert "mmu eval ablate" in out


def test_compare_does_not_warn_when_the_runs_were_interleaved(harness, tmp_path) -> None:
    labels = ["hybrid", "dense-only"]
    dicts = _save_and_load(tmp_path, _interleave(harness), labels)
    out = report_mod.render_comparison(dicts, labels)
    assert "NOT comparable" not in out
    assert "interleaved question-major" in out
