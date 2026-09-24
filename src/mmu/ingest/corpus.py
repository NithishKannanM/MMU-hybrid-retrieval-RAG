"""Walk ``data/corpus/`` and decide which documents get indexed.

This is the one module in ``ingest`` that is new to MMU rather than ported: the sibling
project this package borrows ``reader.py``/``splitter.py`` from validates untrusted HTTP
uploads, but MMU's corpus is a directory of files the person running the tool put there
themselves. What that directory needs instead is a stable, reproducible mapping from
"files on disk" to "documents with an id" — because that id, the ``doc_id``, is the join
key between three things built independently: the FAISS row order, ``/index/stats``'s
table, and the ``required_doc_ids``/``relevant_doc_ids`` a person hand-writes into
``data/questions.jsonl`` before ever seeing the index. Get the mapping wrong — two files
colliding on one id, or the same directory producing a different row order on every
rebuild — and every downstream number (per-document XDR coverage, CP's relevance labels)
silently points at the wrong document instead of failing loudly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from mmu.ingest.reader import DocumentKind, ExtractedText

logger = logging.getLogger(__name__)

#: Extension → kind. Anything else is skipped (with a warning), not rejected — a stray
#: .DS_Store or a reference .csv dropped next to the real corpus should not fail the
#: whole `mmu index build`, only be left out of it.
_KIND_BY_EXTENSION: dict[str, DocumentKind] = {
    ".pdf": "pdf",
    ".txt": "txt",
    ".md": "md",
}

#: Below this many extracted characters per page, a PDF is almost certainly scanned or
#: laid out in a way pypdf's text extraction cannot follow (multi-column, form fields).
#: Picked as an order-of-magnitude floor, not a tuned threshold: a genuine text page of
#: legal prose runs well over 1000 chars; 200 only trips on pages that are effectively
#: empty to the extractor.
_SCANNED_PDF_CHARS_PER_PAGE = 200

_SLUG_INVALID = re.compile(r"[^a-z0-9]+")


def slugify(stem: str) -> str:
    """Turn a filename stem into a stable, ASCII, URL/JSON-safe id.

    ``"GDPR (EU) 2016-679.pdf"`` → ``"gdpr-eu-2016-679"``. Deliberately ASCII-only —
    not a transliterating slugifier — because a ``doc_id`` is going to be copied by
    hand into ``data/questions.jsonl``, echoed in ``/index/stats`` JSON, and possibly
    used as a query parameter; every non-ASCII letter is folded to a hyphen rather than
    approximated (no "é" → "e" table to maintain or get subtly wrong), which is a
    stricter but simpler and fully deterministic rule.
    """
    lowered = stem.lower()
    slug = _SLUG_INVALID.sub("-", lowered)
    return slug.strip("-")


class CorpusError(Exception):
    """The corpus directory itself is invalid — currently, only a duplicate doc_id.

    Distinct from :class:`mmu.ingest.reader.ReaderError`, which is about one file's
    bytes; this is about the directory as a whole and is raised by discovery, before any
    file is even opened.
    """


@dataclass(frozen=True)
class CorpusDocument:
    """One file in ``data/corpus/`` that will be read, split, and indexed."""

    doc_id: str
    path: Path
    filename: str
    kind: DocumentKind


def discover(corpus_dir: Path) -> list[CorpusDocument]:
    """Find every indexable file under ``corpus_dir``.

    Recursive (``rglob``), not a flat listing: a real compliance corpus tends to arrive
    pre-organised (``gdpr/``, `national-law/``, one subfolder per regulator), and forcing
    a flat directory would just push that organisation into filenames. The cost is that
    two files with the same name in different subfolders both slugify to the same
    ``doc_id`` — handled below by raising rather than silently merging them.

    Ordering is by each file's path *relative to* ``corpus_dir``, not by the OS's
    directory-iteration order (unspecified) or by absolute path (varies by where the
    repo is checked out). This is what makes FAISS row order — and therefore every
    ``chunk_id``/row index pairing in the store — reproducible across a rebuild on a
    different machine or after `git clone`.

    Skips dotfiles (``.gitkeep``, ``.DS_Store``, ...) and directories entirely. An
    unsupported extension is skipped with a logged warning rather than raising: dropping
    a README or a reference spreadsheet into the corpus folder should not block a build.
    A duplicate ``doc_id`` raises :class:`CorpusError` — two files silently merging their
    chunks under one id would corrupt every per-document metric with no visible symptom
    other than numbers that are quietly wrong.
    """
    candidates = sorted(
        (p for p in corpus_dir.rglob("*") if p.is_file() and not p.name.startswith(".")),
        key=lambda p: p.relative_to(corpus_dir).as_posix(),
    )

    documents: list[CorpusDocument] = []
    seen: dict[str, str] = {}  # doc_id -> filename, for the collision error message
    for path in candidates:
        kind = _KIND_BY_EXTENSION.get(path.suffix.lower())
        if kind is None:
            logger.warning("skipping unsupported file in corpus: %s", path)
            continue

        doc_id = slugify(path.stem)
        if doc_id in seen:
            raise CorpusError(
                f"duplicate doc_id {doc_id!r}: {seen[doc_id]!r} and {path.name!r} "
                "both slugify to the same id — rename one of them"
            )
        seen[doc_id] = path.name

        documents.append(
            CorpusDocument(doc_id=doc_id, path=path, filename=path.name, kind=kind)
        )
    return documents


def scanned_pdf_warning(extracted: ExtractedText, filename: str) -> str | None:
    """``None`` unless ``extracted`` looks like a scanned or unextractable PDF.

    Rationale: a legal PDF is frequently scanned, or laid out multi-column in a way
    pypdf's extraction cannot follow. In either case ``extract_text`` does not raise —
    it returns pages of empty or near-empty strings — so the index build "succeeds" with
    a document contributing zero usable chunks, and every downstream metric involving
    that document then reads as a *model* failure (missing citation, low XDR coverage)
    when the real defect is an empty index. OCR would fix the root cause but is
    explicitly out of scope for MMU; this warning is the whole mitigation, so it must
    fire at build time where a human will actually see it, not get discovered from a
    confusing eval report an hour later.

    Only meaningful for paginated documents (``extracted.pages`` is ``None`` for
    txt/md, which have no such failure mode), so non-paginated input is always quiet.
    """
    pages = extracted.pages
    if pages is None:
        return None
    avg_chars_per_page = len(extracted.text) / pages
    if avg_chars_per_page >= _SCANNED_PDF_CHARS_PER_PAGE:
        return None
    return (
        f"{filename}: only {avg_chars_per_page:.0f} chars/page extracted across "
        f"{pages} page(s) — likely a scanned or multi-column PDF that pypdf cannot "
        "read; the index will build but this document will contribute ~0 chunks. "
        "OCR it first if its content matters to the eval."
    )
