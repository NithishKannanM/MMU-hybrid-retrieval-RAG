"""Judges: turn an answer and its retrieved context into the verdicts AF and XDR need.

Mirrors the shape of ``local_infra/bench/judge.py`` (``Judge`` Protocol / ``LocalJudge``
/ ``CachingJudge``), extended here because MMU's metrics need three distinct judge
tasks where the IRR judge needed only one ("did retrieval recall this fact"):

1. **Claim decomposition** — split an answer into atomic, individually-checkable claims.
2. **Claim verification** — for one claim against one candidate source chunk, does the
   chunk support it, and *where* (a verbatim span), and why.
3. **The XDR ``Syn`` rubric** — does the answer's conclusion actually depend on
   synthesizing multiple required documents together, or just mention all of them.

Two constraints run through every implementation in this file:

1. **A verdict without a locatable span does not get to say "supported."**
   :class:`LocalJudge` never claims to localize a span at all (it is lexical-overlap
   only — see its docstring for exactly what that does and does not catch).
   :class:`LlmJudge` does claim to, and is therefore held to it: after parsing, the
   returned span is checked as a real (whitespace-normalized) substring of the cited
   chunk, and a verdict whose span does not check out is downgraded to unsupported and
   counted as ``invented_span``. A judge that cannot point at the text is not entitled
   to vouch for it.
2. **Judges are never silently mixed.** When :class:`LlmJudge` cannot get parseable
   JSON even after one retry, it falls back to :class:`LocalJudge` for that single item
   and marks the returned :class:`Judgment` ``degraded=True`` rather than quietly
   substituting a different judge's opinion into what looks like an LLM verdict. A
   report reading these must be able to say "3 of 30 items fell back", not just print a
   clean-looking number that was secretly computed two different ways.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from rapidfuzz import fuzz

from mmu.eval.metrics import synthesis_score_fallback
from mmu.generate.base import Generator, Message

JudgeTask = Literal["decompose", "verify", "syn"]


def _normalize_ws(s: str) -> str:
    """Duplicated from :mod:`mmu.eval.questions` / :mod:`mmu.eval.metrics` on purpose —
    see either module's docstring for why a one-line helper is cheaper to repeat than
    to couple these modules over."""
    return " ".join(s.split())


@dataclass(frozen=True)
class Judgment:
    """One judge call's result. Which fields are meaningful depends on ``task``:

    * ``"decompose"``: ``claims`` holds the atomic claims extracted from an answer.
    * ``"verify"``: ``verdict`` (supported or not), ``source`` (the chunk_id the judge
      says supports the claim, only set when ``verdict`` is True), ``span`` (the
      verbatim quote — see the invented-span check), ``reason``.
    * ``"syn"``: ``score`` in ``{0, 0.5, 1}`` per the XDR rubric, ``reason``.

    ``degraded`` is True when an ``LlmJudge`` call fell back to ``LocalJudge`` after
    malformed JSON survived one retry. ``invented_span`` is True when a verdict was
    downgraded because its span did not check out against the cited chunk — both flags
    exist so a report can count and surface these without re-deriving them.
    """

    task: JudgeTask
    claims: tuple[str, ...] = ()
    verdict: bool | None = None
    source: str | None = None
    span: str | None = None
    reason: str = ""
    score: float | None = None
    degraded: bool = False
    invented_span: bool = False


@runtime_checkable
class Judge(Protocol):
    """The three judge tasks. All async: :class:`LlmJudge` calls a
    :class:`mmu.generate.base.Generator`, which is network I/O; :class:`LocalJudge`
    implements the same async signatures with no actual await inside, so the two are
    interchangeable behind one Protocol regardless of backend."""

    @property
    def name(self) -> str: ...

    async def decompose_claims(self, answer_text: str) -> Judgment: ...

    async def verify_claim(self, claim: str, chunk_text: str, chunk_id: str) -> Judgment: ...

    async def score_synthesis(
        self,
        *,
        question: str,
        answer_text: str,
        required_doc_texts: Mapping[str, str],
        expected_synthesis_keys: Sequence[str],
    ) -> Judgment: ...


# ------------------------------------------------------------------------- LocalJudge


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: rapidfuzz token_set_ratio is 0-100; the plan's ">= 0.75" threshold on the 0-1 scale.
LOCAL_SUPPORT_THRESHOLD = 75.0


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


class LocalJudge:
    """Deterministic, offline, free — decomposes by sentence, verifies by
    ``rapidfuzz.fuzz.token_set_ratio``.

    **This measures lexical overlap, not entailment.** ``token_set_ratio`` scores how
    much of the claim's vocabulary appears in the chunk's vocabulary, ignoring word
    order and duplicates. Two consequences follow, and only one of them is survivable:

    * **False negative on correct paraphrase.** A claim that restates a chunk's meaning
      in different words scores lower than one that echoes its wording, so this judge
      under-credits faithful paraphrase. Annoying, but safe — it makes AF too
      pessimistic, never too optimistic.
    * **False positive on negation flips — the dangerous direction.** "The controller
      must notify within 72 hours" and "the controller must **not** notify within 72
      hours" share every content word except one, so they score around
      ``token_set_ratio ~= 0.95`` against the same source chunk. In a compliance corpus
      this is exactly the failure that matters: a model that inverts an obligation would
      be scored as *faithful* by this judge. ``test_judge.py`` asserts this as a known,
      permanent limitation rather than a bug to eventually fix — no lexical-overlap
      measure can fix it without becoming an entailment model, at which point it is not
      "local" anymore.

    Conclusion, stated outright so nobody forgets it: **local AF is a CI regression
    tripwire** (did this change measurably worsen lexical grounding), **not a quality
    number to report**. Any AF that goes in a report needs an LLM judge.
    """

    name = "local"

    def __init__(self, threshold: float = LOCAL_SUPPORT_THRESHOLD) -> None:
        self._threshold = threshold

    async def decompose_claims(self, answer_text: str) -> Judgment:
        return Judgment(task="decompose", claims=tuple(_split_sentences(answer_text)))

    async def verify_claim(self, claim: str, chunk_text: str, chunk_id: str) -> Judgment:
        score = fuzz.token_set_ratio(claim, chunk_text)
        verdict = score >= self._threshold
        return Judgment(
            task="verify",
            verdict=verdict,
            source=chunk_id if verdict else None,
            # LocalJudge never claims to have localized a supporting span within the
            # chunk — token_set_ratio scores vocabulary overlap, not a position. Leaving
            # `span=None` here (rather than inventing one, e.g. the whole chunk) keeps
            # it honest: it cannot feed CUR_attr, only AF's lexical tripwire.
            span=None,
            reason=f"token_set_ratio={score:.1f} (threshold {self._threshold:.1f})",
        )

    async def score_synthesis(
        self,
        *,
        question: str,
        answer_text: str,
        required_doc_texts: Mapping[str, str],
        expected_synthesis_keys: Sequence[str],
    ) -> Judgment:
        # question/required_doc_texts are unused here: the offline fallback (see
        # mmu.eval.metrics.synthesis_score_fallback) only ever looks at key presence in
        # the answer text, by design — it cannot assess synthesis, only concatenation.
        score = synthesis_score_fallback(answer_text, expected_synthesis_keys)
        return Judgment(task="syn", score=score, reason="deterministic key-presence fallback")


# --------------------------------------------------------------------------- LlmJudge


_JSON_ONLY_REMINDER = "Respond with JSON only. No prose, no markdown code fences."

_DECOMPOSE_PROMPT = """Break the following answer into a list of atomic, independently \
checkable factual claims. Each claim should be a single self-contained statement. \
Skip hedges and meta-commentary ("I found that...").

Respond with JSON only: {{"claims": ["claim 1", "claim 2", ...]}}

Answer:
{answer}
"""

_VERIFY_PROMPT = """Does the source chunk below support the claim? A claim is supported \
only if the chunk actually states it — pay close attention to polarity: "must notify" \
and "must not notify" are opposite claims, and a chunk supporting one does NOT support \
the other. You must quote the exact verbatim span from the chunk that supports the \
claim; if you cannot quote such a span, the verdict must be false and span must be "".

Respond with JSON only:
{{"verdict": true or false, "source": "{chunk_id}", "span": "<verbatim quote from the \
chunk, or empty string>", "reason": "<one sentence>"}}

Claim: {claim}

Chunk ({chunk_id}):
{chunk}
"""

_SYN_PROMPT = """Score whether the answer's operative conclusion depends on \
synthesizing MULTIPLE of the required documents together, using exactly this rubric:

1   = the operative conclusion follows only from two or more of the required documents \
      taken together — no single document states it alone.
0.5 = all required documents are mentioned or drawn from, but the operative conclusion \
      stands on any one of them alone (the others are decoration, not load-bearing).
0   = the operative conclusion is derivable from a single document; the rest are unused.

Respond with JSON only: {{"score": 0, 0.5, or 1, "reason": "<one sentence>"}}

Question: {question}

Answer:
{answer}

Required documents:
{docs}
"""


def _try_parse_json(text: str) -> dict | None:
    """Best-effort JSON object extraction: strips a leading/trailing markdown fence
    (some small local models wrap JSON in ```json ... ``` despite instructions) before
    parsing. Returns None — never raises — on anything that isn't a JSON object, so
    callers can treat "malformed" uniformly regardless of which way it failed."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = re.sub(r"^json\s*", "", stripped, flags=re.IGNORECASE).strip()
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class LlmJudge:
    """Wraps *any* :class:`mmu.generate.base.Generator` — Ollama or Anthropic, whatever
    is registered — which makes the judge backend pluggable for free: the same class
    serves a fast local judge and a stronger cloud one, and swapping is a Generator
    swap, not a new judge implementation.

    Runs at temperature 0 (a benchmark judge must be as reproducible as the benchmark
    itself). Malformed JSON gets exactly one retry with a "JSON only" reminder appended
    to the conversation; if that also fails to parse, the call falls back to
    :class:`LocalJudge` for that single item and the returned :class:`Judgment` carries
    ``degraded=True`` — see the module docstring for why this is a flag, not a silent
    substitution.
    """

    def __init__(
        self,
        generator: Generator,
        *,
        fallback: Judge | None = None,
        max_tokens: int = 512,
    ) -> None:
        self._generator = generator
        self._fallback = fallback if fallback is not None else LocalJudge()
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return f"llm:{self._generator.name}"

    async def _complete_json(self, prompt: str) -> tuple[dict | None, bool]:
        """Returns ``(parsed_or_None, needed_a_retry_and_still_failed)``."""
        messages = [Message(role="user", content=prompt)]
        first = await self._generator.complete(messages, temperature=0.0, max_tokens=self._max_tokens)
        parsed = _try_parse_json(first.text)
        if parsed is not None:
            return parsed, False

        retry_messages = [
            Message(role="user", content=prompt),
            Message(role="assistant", content=first.text),
            Message(role="user", content=_JSON_ONLY_REMINDER),
        ]
        second = await self._generator.complete(
            retry_messages, temperature=0.0, max_tokens=self._max_tokens
        )
        parsed2 = _try_parse_json(second.text)
        return parsed2, parsed2 is None

    async def decompose_claims(self, answer_text: str) -> Judgment:
        parsed, failed = await self._complete_json(_DECOMPOSE_PROMPT.format(answer=answer_text))
        if failed or not isinstance(parsed.get("claims") if parsed else None, list):
            fallback = await self._fallback.decompose_claims(answer_text)
            return dataclasses.replace(fallback, degraded=True)
        claims = tuple(str(c) for c in parsed["claims"])
        return Judgment(task="decompose", claims=claims)

    async def verify_claim(self, claim: str, chunk_text: str, chunk_id: str) -> Judgment:
        prompt = _VERIFY_PROMPT.format(claim=claim, chunk=chunk_text, chunk_id=chunk_id)
        parsed, failed = await self._complete_json(prompt)
        if failed or parsed is None or "verdict" not in parsed:
            fallback = await self._fallback.verify_claim(claim, chunk_text, chunk_id)
            return dataclasses.replace(fallback, degraded=True)

        verdict = bool(parsed.get("verdict"))
        span = parsed.get("span") or None
        reason = str(parsed.get("reason", ""))
        source = str(parsed.get("source") or chunk_id)
        invented = False

        if verdict:
            if not span or _normalize_ws(str(span)) not in _normalize_ws(chunk_text):
                # The judge said "supported" but either gave no span or gave one that is
                # not actually in the chunk. It does not get to say "supported" — see
                # module docstring. Downgrade and count it.
                verdict = False
                invented = True
                reason = f"judge_invented_span: {reason}".strip(": ")

        return Judgment(
            task="verify",
            verdict=verdict,
            source=source if verdict else None,
            span=span if verdict else None,
            reason=reason,
            invented_span=invented,
        )

    async def score_synthesis(
        self,
        *,
        question: str,
        answer_text: str,
        required_doc_texts: Mapping[str, str],
        expected_synthesis_keys: Sequence[str],
    ) -> Judgment:
        docs_block = "\n\n".join(f"[{doc_id}]\n{text}" for doc_id, text in required_doc_texts.items())
        prompt = _SYN_PROMPT.format(question=question, answer=answer_text, docs=docs_block)
        parsed, failed = await self._complete_json(prompt)

        async def _fallback_result() -> Judgment:
            fb = await self._fallback.score_synthesis(
                question=question,
                answer_text=answer_text,
                required_doc_texts=required_doc_texts,
                expected_synthesis_keys=expected_synthesis_keys,
            )
            return dataclasses.replace(fb, degraded=True)

        if failed or parsed is None or "score" not in parsed:
            return await _fallback_result()
        try:
            score = float(parsed["score"])
        except (TypeError, ValueError):
            return await _fallback_result()
        if score not in (0.0, 0.5, 1.0):
            score = min((0.0, 0.5, 1.0), key=lambda s: abs(s - score))
        return Judgment(task="syn", score=score, reason=str(parsed.get("reason", "")))


# ----------------------------------------------------------------------- CachingJudge


def _judgment_to_dict(j: Judgment) -> dict:
    return {
        "task": j.task,
        "claims": list(j.claims),
        "verdict": j.verdict,
        "source": j.source,
        "span": j.span,
        "reason": j.reason,
        "score": j.score,
        "degraded": j.degraded,
        "invented_span": j.invented_span,
    }


def _judgment_from_dict(d: dict) -> Judgment:
    return Judgment(
        task=d["task"],
        claims=tuple(d.get("claims") or ()),
        verdict=d.get("verdict"),
        source=d.get("source"),
        span=d.get("span"),
        reason=d.get("reason", ""),
        score=d.get("score"),
        degraded=d.get("degraded", False),
        invented_span=d.get("invented_span", False),
    )


class _JsonlJudgmentCache:
    """The on-disk store backing :class:`CachingJudge`. Loaded eagerly from
    ``cache_path`` on construction, appended to on every miss — a new instance pointed
    at the same path picks up everything a previous instance (in this process or a
    previous one) already wrote, which is the persistence guarantee
    :class:`CachingJudge` needs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self._entries[row["key"]] = row["judgment"]

    def get(self, key: str) -> Judgment | None:
        row = self._entries.get(key)
        return _judgment_from_dict(row) if row is not None else None

    def put(self, key: str, judgment: Judgment) -> None:
        self._entries[key] = _judgment_to_dict(judgment)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"key": key, "judgment": _judgment_to_dict(judgment)}, ensure_ascii=False)
                + "\n"
            )


class CachingJudge:
    """Wraps a :class:`Judge` and **persists** every verdict to
    ``data/.cache/judgments.jsonl`` (unlike ``local_infra/bench/judge.py``'s in-memory
    sibling, which only memoizes within one process). An LLM-judge pass over even a
    small eval corpus is minutes of wall clock; re-running ``mmu eval report`` after
    changing report formatting must not re-pay that cost.

    Keyed by ``sha256(question_id | judge_name | prompt_kind | canonical_payload)``, so
    a ``CachingJudge`` is bound to one question (``question_id`` is fixed at
    construction) — this keeps its method signatures identical to :class:`Judge`'s
    (no extra parameter threaded through every call site), at the cost of constructing
    one instance per question. All instances built with the same ``cache_path`` share
    the same file and observe each other's writes across a run, and across separate
    runs of the process.
    """

    def __init__(self, inner: Judge, question_id: str, cache_path: Path) -> None:
        self._inner = inner
        self._question_id = question_id
        self._cache = _JsonlJudgmentCache(cache_path)

    @property
    def name(self) -> str:
        return self._inner.name

    @staticmethod
    def cache_key(question_id: str, judge_name: str, prompt_kind: str, payload: str) -> str:
        canonical = "\x1f".join([question_id, judge_name, prompt_kind, payload])
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def decompose_claims(self, answer_text: str) -> Judgment:
        key = self.cache_key(self._question_id, self._inner.name, "decompose", answer_text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = await self._inner.decompose_claims(answer_text)
        self._cache.put(key, result)
        return result

    async def verify_claim(self, claim: str, chunk_text: str, chunk_id: str) -> Judgment:
        payload = "\x1f".join([chunk_id, claim, chunk_text])
        key = self.cache_key(self._question_id, self._inner.name, "verify", payload)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = await self._inner.verify_claim(claim, chunk_text, chunk_id)
        self._cache.put(key, result)
        return result

    async def score_synthesis(
        self,
        *,
        question: str,
        answer_text: str,
        required_doc_texts: Mapping[str, str],
        expected_synthesis_keys: Sequence[str],
    ) -> Judgment:
        docs_payload = "\x1e".join(f"{k}={v}" for k, v in sorted(required_doc_texts.items()))
        keys_payload = ",".join(expected_synthesis_keys)
        payload = "\x1f".join([question, answer_text, docs_payload, keys_payload])
        key = self.cache_key(self._question_id, self._inner.name, "syn", payload)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = await self._inner.score_synthesis(
            question=question,
            answer_text=answer_text,
            required_doc_texts=required_doc_texts,
            expected_synthesis_keys=expected_synthesis_keys,
        )
        self._cache.put(key, result)
        return result
