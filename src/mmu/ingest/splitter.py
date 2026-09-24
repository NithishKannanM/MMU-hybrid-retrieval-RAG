"""A document splitter: hierarchical, overlapping, and pure.

Why spans, not strings: every chunk is a half-open ``[char_start, char_end)`` slice of
the *original* text, which is what makes ``chunk.text == source[start:end]`` an
invariant a test can assert — and what lets a citation highlight the exact source
region rather than a re-joined approximation of it. A splitter that returned copied
strings would need a second search to relocate each chunk in its document before a
citation could point at anything, and that search is not guaranteed to find a unique
answer when text repeats (a boilerplate clause, a repeated heading).

Token counts are a documented ``len // 4`` heuristic. **Not tiktoken**: it downloads its
BPE files from the network on first use, which would be a network dependency in a
service whose whole point is that offline is structural — an index build must not be
able to fail because a BPE file 404'd.

Defaults (``max_tokens=384, min_tokens=100, overlap_tokens=64``) are retuned for this
project, not inherited: 350/80/48 was sized for nomic-embed-text's 512-token window,
where headroom had to be left inside the window for the instruction prefix. BGE-M3's
window is 8192 tokens, and the corpus here is legal/compliance text, where a defined
term ("controller", "processor", "supervisory authority") is established once and then
governs clauses many paragraphs later — a chunk that is too tight on the token budget
recreates exactly the boundary problem overlap exists to solve, just more often. 384
buys headroom to keep a clause and the definition it depends on in the same chunk far
more often than 350 did, without approaching BGE-M3's window.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

#: Average characters per token for English prose under a BPE tokenizer. Used only to
#: size chunks, never to bill anything — a 25% error here shifts a boundary, nothing more.
CHARS_PER_TOKEN = 4

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
#: Sentence end: terminal punctuation (optionally quoted/bracketed) then whitespace. A
#: single newline also ends a segment — PDF extraction is full of hard-wrapped lines and
#: bullet lists that never see a period.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n")
_WHITESPACE = re.compile(r"\s")


def estimate_tokens(text: str) -> int:
    """Rough token count for sizing. See ``CHARS_PER_TOKEN``."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


@dataclass(frozen=True)
class TextSpan:
    """One retrievable slice of a document, as produced by :func:`split_text`.

    Deliberately not ``mmu.core.models.Chunk``: ``Chunk`` requires ``doc_id``,
    ``filename`` and ``chunk_id``, none of which a pure text splitter can know — it
    never sees a filename, let alone the slugified doc_id assigned by ``corpus.py``.
    Handing this lightweight, splitter-owned type back keeps ``split_text`` a pure
    function of ``text`` alone (so it stays unit-testable with bare strings, no
    ``Chunk`` scaffolding) and puts assembly of the real ``Chunk`` — the step that
    needs document identity — where that identity lives: ``corpus.py`` / the index
    builder.

    ``text`` is exactly ``source[char_start:char_end]``. ``core_start`` marks where this
    chunk's *own* content begins: everything before it is overlap duplicated from the
    previous chunk, which matters when highlighting a hit (you want the new material
    emphasised, not the repeated lead-in).
    """

    index: int
    text: str
    char_start: int
    char_end: int
    core_start: int
    tokens: int

    @property
    def overlap_chars(self) -> int:
        return self.core_start - self.char_start


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink ``[start, end)`` past leading/trailing whitespace, preserving offsets."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _hard_split(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Last resort for a single sentence longer than a whole chunk (minified JSON, a
    table dumped as one line). Breaks at the last whitespace before the limit so words
    stay intact; falls back to a mid-word cut when there is no whitespace at all."""
    spans: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > max_chars:
        window_end = cursor + max_chars
        cut = -1
        for i in range(window_end, cursor, -1):
            if _WHITESPACE.match(text, i - 1):
                cut = i
                break
        if cut <= cursor:
            cut = window_end
        s, e = _trim(text, cursor, cut)
        if e > s:
            spans.append((s, e))
        cursor = cut
    s, e = _trim(text, cursor, end)
    if e > s:
        spans.append((s, e))
    return spans


def _segments(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Atomic units to pack, finest granularity last: paragraphs, then sentences, then
    a hard character split. Every returned span is non-empty, trimmed, in order, and no
    longer than ``max_chars``."""
    segments: list[tuple[int, int]] = []
    for p_start, p_end in _spans_between(text, _PARAGRAPH_BREAK, 0, len(text)):
        if p_end - p_start <= max_chars:
            segments.append((p_start, p_end))
            continue
        for s_start, s_end in _spans_between(text, _SENTENCE_BREAK, p_start, p_end):
            if s_end - s_start <= max_chars:
                segments.append((s_start, s_end))
            else:
                segments.extend(_hard_split(text, s_start, s_end, max_chars))
    return segments


def _spans_between(
    text: str, pattern: re.Pattern[str], start: int, end: int
) -> list[tuple[int, int]]:
    """Trimmed spans of ``text[start:end]`` delimited by ``pattern``, empties dropped."""
    spans: list[tuple[int, int]] = []
    cursor = start
    for m in pattern.finditer(text, start, end):
        s, e = _trim(text, cursor, m.start())
        if e > s:
            spans.append((s, e))
        cursor = m.end()
    s, e = _trim(text, cursor, end)
    if e > s:
        spans.append((s, e))
    return spans


def _overlap_start(text: str, core_start: int, floor: int, overlap_chars: int) -> int:
    """Walk back up to ``overlap_chars`` from ``core_start``, snapping forward to the
    next whitespace so the overlap begins at a word boundary rather than mid-word.
    Never crosses ``floor`` (the previous chunk's own start), so two chunks can never
    become identical."""
    if overlap_chars <= 0:
        return core_start
    target = max(floor, core_start - overlap_chars)
    if target >= core_start:
        return core_start
    m = _WHITESPACE.search(text, target, core_start)
    return m.end() if m is not None else target


def split_text(
    text: str,
    *,
    max_tokens: int = 384,
    min_tokens: int = 100,
    overlap_tokens: int = 64,
) -> list[TextSpan]:
    """Split ``text`` into overlapping chunks sized for BGE-M3's 8192-token window.

    Greedy packing: consecutive segments accumulate until the next one would exceed
    ``max_tokens``. Only the *final* chunk can come out short (everything earlier was
    filled), so ``min_tokens`` is enforced by folding a runt tail back into its
    predecessor — a 12-token trailing chunk embeds to noise and pollutes every recall.
    That fold is the one case where a chunk may exceed ``max_tokens``.

    ``overlap_tokens`` of lead-in is duplicated from the previous chunk so a fact
    straddling a boundary is retrievable from either side.
    """
    if not text or not text.strip():
        return []

    max_chars = max(1, max_tokens * CHARS_PER_TOKEN)
    # Clamped: a `min_tokens` above `max_tokens` would make EVERY chunk a runt, and the
    # fold below would then collapse the whole document into one chunk with no error.
    min_chars = min(max(0, min_tokens * CHARS_PER_TOKEN), max_chars)
    overlap_chars = max(0, overlap_tokens * CHARS_PER_TOKEN)

    segments = _segments(text, max_chars)
    if not segments:
        return []

    # 1. Pack segments into core spans.
    cores: list[tuple[int, int]] = []
    start, end = segments[0]
    for seg_start, seg_end in segments[1:]:
        if seg_end - start <= max_chars:
            end = seg_end
        else:
            cores.append((start, end))
            start, end = seg_start, seg_end
    cores.append((start, end))

    # 2. Fold a runt tail into its predecessor.
    if len(cores) > 1 and cores[-1][1] - cores[-1][0] < min_chars:
        tail = cores.pop()
        cores[-1] = (cores[-1][0], tail[1])

    # 3. Add overlap and materialise.
    chunks: list[TextSpan] = []
    for i, (core_start, core_end) in enumerate(cores):
        floor = cores[i - 1][0] if i else core_start
        char_start = _overlap_start(text, core_start, floor, overlap_chars) if i else core_start
        body = text[char_start:core_end]
        chunks.append(
            TextSpan(
                index=i,
                text=body,
                char_start=char_start,
                char_end=core_end,
                core_start=core_start,
                tokens=estimate_tokens(body),
            )
        )
    return chunks


def page_of(page_starts: list[int], offset: int) -> int | None:
    """1-based page containing ``offset``, given each page's start offset in the joined
    document text. ``None`` when the document has no page structure (TXT/MD).

    Pure and separate from the reader so the offset→page mapping — the part that is
    actually easy to get off by one — is unit-testable without a PDF.
    """
    if not page_starts:
        return None
    return max(1, bisect_right(page_starts, offset))
