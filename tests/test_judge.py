"""Tests for mmu.eval.judge: LocalJudge's documented limitations, LlmJudge's span
verification and degrade-on-malformed-JSON path, and CachingJudge's persistence.

No network, no real model: LlmJudge is exercised entirely through FakeGenerator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from rapidfuzz import fuzz

from mmu.eval.judge import (
    LOCAL_SUPPORT_THRESHOLD,
    CachingJudge,
    Judge,
    Judgment,
    LlmJudge,
    LocalJudge,
)
from tests.conftest import FakeGenerator

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------- LocalJudge


async def test_local_judge_decompose_splits_on_sentence_boundaries() -> None:
    judge = LocalJudge()
    result = await judge.decompose_claims("First claim. Second claim! Third claim?")
    assert result.task == "decompose"
    assert result.claims == ("First claim.", "Second claim!", "Third claim?")


async def test_local_judge_decompose_empty_answer_is_no_claims() -> None:
    judge = LocalJudge()
    result = await judge.decompose_claims("   ")
    assert result.claims == ()


async def test_local_judge_verify_above_threshold_is_supported() -> None:
    judge = LocalJudge()
    result = await judge.verify_claim(
        "The controller must notify within 72 hours.",
        "The controller must notify within 72 hours of becoming aware.",
        "gdpr:0",
    )
    assert result.verdict is True
    assert result.source == "gdpr:0"
    # LocalJudge never claims to localize a span — only LlmJudge does, and is held to it.
    assert result.span is None


async def test_local_judge_verify_below_threshold_is_unsupported() -> None:
    judge = LocalJudge()
    result = await judge.verify_claim(
        "Bananas are a good source of potassium.",
        "The controller must notify within 72 hours of becoming aware.",
        "gdpr:0",
    )
    assert result.verdict is False
    assert result.source is None


async def test_local_judge_threshold_is_configurable() -> None:
    strict = LocalJudge(threshold=99.9)
    result = await strict.verify_claim(
        "The controller must notify within 72 hours.",
        "The controller must notify within 72 hours of becoming aware.",
        "gdpr:0",
    )
    assert result.verdict is False  # a near-identical but not perfect match now fails


async def test_local_judge_negation_false_positive_is_a_known_limitation() -> None:
    """LocalJudge is lexical-overlap only, so a negation flip barely moves the score.

    "must notify" and "must not notify" against the same source chunk differ by exactly
    one token, so token_set_ratio stays far above the support threshold and LocalJudge
    reports the negated claim as *supported*. This is the exact failure the plan calls
    out: in a compliance corpus, inverting an obligation is the failure that matters
    most, and no lexical judge can catch it. This test documents the limitation as
    permanent (see LocalJudge's docstring) rather than letting it be silently
    "fixed"/forgotten — any reported (non-CI-tripwire) AF number needs an LLM judge.
    """
    judge = LocalJudge()
    chunk = "The controller must notify the supervisory authority within 72 hours."
    positive_claim = "The controller must notify the supervisory authority within 72 hours."
    negated_claim = "The controller must not notify the supervisory authority within 72 hours."

    score = fuzz.token_set_ratio(negated_claim, chunk)
    assert score >= LOCAL_SUPPORT_THRESHOLD  # the dangerous false positive, quantified

    positive_result = await judge.verify_claim(positive_claim, chunk, "gdpr:0")
    negated_result = await judge.verify_claim(negated_claim, chunk, "gdpr:0")
    assert positive_result.verdict is True
    assert negated_result.verdict is True  # WRONG, and known to be wrong — see docstring


async def test_local_judge_score_synthesis_uses_deterministic_fallback() -> None:
    judge = LocalJudge()
    result = await judge.score_synthesis(
        question="q",
        answer_text="notify within 72 hours and escalate to the supervisory authority",
        required_doc_texts={"gdpr": "...", "national-law": "...", "company-policy": "..."},
        expected_synthesis_keys=("72 hours", "supervisory authority", "escalate"),
    )
    assert result.task == "syn"
    assert result.score == 1.0


# ----------------------------------------------------------------------------- LlmJudge


async def test_llm_judge_verify_claim_accepts_a_real_verbatim_span() -> None:
    chunk_text = "The controller shall notify within 72 hours of becoming aware."
    fake = FakeGenerator(
        reply='{"verdict": true, "source": "gdpr:0", "span": "notify within 72 hours", "reason": "matches"}'
    )
    judge = LlmJudge(fake)
    result = await judge.verify_claim("Notification is due within 72 hours.", chunk_text, "gdpr:0")
    assert result.verdict is True
    assert result.span == "notify within 72 hours"
    assert result.invented_span is False
    assert result.degraded is False


async def test_llm_judge_downgrades_and_counts_an_invented_span() -> None:
    chunk_text = "The controller shall notify within 72 hours of becoming aware."
    fake = FakeGenerator(
        reply=(
            '{"verdict": true, "source": "gdpr:0", '
            '"span": "this sentence does not appear in the chunk at all", '
            '"reason": "looks plausible"}'
        )
    )
    judge = LlmJudge(fake)
    result = await judge.verify_claim("Some claim.", chunk_text, "gdpr:0")
    # A judge that cannot point at the text does not get to say "supported".
    assert result.verdict is False
    assert result.invented_span is True
    assert "judge_invented_span" in result.reason


async def test_llm_judge_verdict_true_with_no_span_is_also_invented() -> None:
    chunk_text = "The controller shall notify within 72 hours of becoming aware."
    fake = FakeGenerator(reply='{"verdict": true, "source": "gdpr:0", "span": "", "reason": "trust me"}')
    judge = LlmJudge(fake)
    result = await judge.verify_claim("Some claim.", chunk_text, "gdpr:0")
    assert result.verdict is False
    assert result.invented_span is True


async def test_llm_judge_malformed_json_retries_once_then_falls_back_degraded() -> None:
    fake = FakeGenerator(reply="not json, sorry, here is a paragraph instead")
    judge = LlmJudge(fake)
    result = await judge.decompose_claims("First sentence. Second sentence.")

    # Exactly one retry: two calls to the generator (original + one retry), not more.
    assert len(fake.prompts) == 2
    assert result.degraded is True
    # The fallback is LocalJudge's sentence splitter.
    assert result.claims == ("First sentence.", "Second sentence.")


async def test_llm_judge_verify_claim_malformed_json_falls_back_degraded() -> None:
    fake = FakeGenerator(reply="{{not valid json")
    judge = LlmJudge(fake)
    result = await judge.verify_claim(
        "The controller must notify within 72 hours.",
        "The controller must notify within 72 hours of becoming aware.",
        "gdpr:0",
    )
    assert result.degraded is True
    assert result.task == "verify"
    assert result.verdict is True  # LocalJudge fallback: high lexical overlap


async def test_llm_judge_score_synthesis_malformed_json_falls_back_degraded() -> None:
    fake = FakeGenerator(reply="nonsense")
    judge = LlmJudge(fake)
    result = await judge.score_synthesis(
        question="q",
        answer_text="72 hours, supervisory authority, escalate",
        required_doc_texts={"a": "...", "b": "..."},
        expected_synthesis_keys=("72 hours", "supervisory authority", "escalate"),
    )
    assert result.degraded is True
    assert result.task == "syn"
    assert result.score == 1.0


async def test_llm_judge_out_of_rubric_score_snaps_to_nearest_allowed_value() -> None:
    fake = FakeGenerator(reply='{"score": 0.8, "reason": "close enough"}')
    judge = LlmJudge(fake)
    result = await judge.score_synthesis(
        question="q", answer_text="a", required_doc_texts={"a": "x"}, expected_synthesis_keys=(),
    )
    assert result.score == 1.0  # nearest of {0, 0.5, 1} to 0.8
    assert result.degraded is False


async def test_llm_judge_strips_markdown_json_fence() -> None:
    fake = FakeGenerator(reply='```json\n{"claims": ["a claim"]}\n```')
    judge = LlmJudge(fake)
    result = await judge.decompose_claims("irrelevant, scripted reply")
    assert result.claims == ("a claim",)
    assert result.degraded is False


# ------------------------------------------------------------------------- CachingJudge


@dataclass
class CountingJudge:
    """Wraps a Judge and counts calls, so tests can distinguish a cache hit (no
    increment) from a miss (inner judge actually invoked)."""

    inner: Judge
    calls: int = field(default=0)

    @property
    def name(self) -> str:
        return self.inner.name

    async def decompose_claims(self, answer_text: str) -> Judgment:
        self.calls += 1
        return await self.inner.decompose_claims(answer_text)

    async def verify_claim(self, claim: str, chunk_text: str, chunk_id: str) -> Judgment:
        self.calls += 1
        return await self.inner.verify_claim(claim, chunk_text, chunk_id)

    async def score_synthesis(self, **kwargs) -> Judgment:
        self.calls += 1
        return await self.inner.score_synthesis(**kwargs)


async def test_caching_judge_hits_on_repeat_call_same_instance(tmp_path) -> None:
    cache_path = tmp_path / "judgments.jsonl"
    counting = CountingJudge(inner=LocalJudge())
    cached = CachingJudge(counting, "q1", cache_path)

    args = ("The controller must notify.", "The controller must notify within 72 hours.", "gdpr:0")
    r1 = await cached.verify_claim(*args)
    assert counting.calls == 1
    r2 = await cached.verify_claim(*args)
    assert counting.calls == 1  # hit: inner not called a second time
    assert r2 == r1


async def test_caching_judge_different_question_id_is_a_separate_cache_key(tmp_path) -> None:
    cache_path = tmp_path / "judgments.jsonl"
    counting = CountingJudge(inner=LocalJudge())
    args = ("The controller must notify.", "The controller must notify within 72 hours.", "gdpr:0")

    cached_q1 = CachingJudge(counting, "q1", cache_path)
    await cached_q1.verify_claim(*args)
    assert counting.calls == 1

    cached_q2 = CachingJudge(counting, "q2", cache_path)
    await cached_q2.verify_claim(*args)
    assert counting.calls == 2  # different question_id -> different key -> miss


async def test_caching_judge_persists_to_jsonl_and_round_trips_across_instances(tmp_path) -> None:
    cache_path = tmp_path / "judgments.jsonl"
    args = ("The controller must notify.", "The controller must notify within 72 hours.", "gdpr:0")

    counting1 = CountingJudge(inner=LocalJudge())
    cached1 = CachingJudge(counting1, "q1", cache_path)
    r1 = await cached1.verify_claim(*args)
    assert counting1.calls == 1
    assert cache_path.exists()
    assert cache_path.read_text(encoding="utf-8").strip() != ""

    # A brand-new CachingJudge instance pointed at the same file must observe the
    # persisted entry without calling its (fresh) inner judge at all.
    counting2 = CountingJudge(inner=LocalJudge())
    cached2 = CachingJudge(counting2, "q1", cache_path)
    r2 = await cached2.verify_claim(*args)
    assert counting2.calls == 0
    assert r2 == r1


async def test_caching_judge_name_passthrough(tmp_path) -> None:
    inner = LocalJudge()
    cached = CachingJudge(inner, "q1", tmp_path / "judgments.jsonl")
    assert cached.name == inner.name
