"""Bytes → plain text, with page offsets where the format has pages.

Every function here is **blocking** (pypdf is synchronous, and a 40-page PDF takes real
seconds); callers run it inside ``asyncio.to_thread``. Kept apart from the splitter so
the offset arithmetic stays pure and testable while the format-specific parsing — the
part that needs real files — is isolated here.

pypdf is a deferred import: a service that only ever ingests Markdown should not pay
import time and memory for a PDF parser it never calls.

MMU has no upload validation layer — unlike the sibling project this was ported from,
the corpus is local files dropped in ``data/corpus/`` by the person running the tool,
never an untrusted upload — so there is no magic-byte sniffing here. ``DocumentKind`` is
decided from the file extension by ``corpus.py`` and handed in; this module only reads.

Only PDF/txt/md are handled. DOCX is deliberately absent: adding it back is a two-line
addition to the dispatch in :func:`extract_text` (a ``_read_docx`` calling
``docx.Document``), but python-docx pulls in lxml, and paying that import cost for a
format nothing in ``data/corpus/`` currently uses is not worth it until it is.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Literal

DocumentKind = Literal["pdf", "txt", "md"]


class ReaderError(Exception):
    """A file could not be read into text: unreadable, encrypted, or an unsupported
    kind reached :func:`extract_text` anyway (the caller skipped the extension check
    that ``corpus.py`` normally does first)."""


@dataclass(frozen=True)
class ExtractedText:
    """Document text plus, for paginated formats, each page's start offset in ``text``.

    ``page_starts`` is empty for TXT/MD — those formats have no page structure at all,
    and inventing one would put a wrong page number on a citation, which is worse than
    showing none.
    """

    text: str
    page_starts: list[int] = field(default_factory=list)

    @property
    def pages(self) -> int | None:
        return len(self.page_starts) or None


#: Pages are joined with a blank line so the splitter's paragraph rule treats a page
#: break as a real boundary rather than gluing the last line of one page to the first
#: line of the next.
_PAGE_SEPARATOR = "\n\n"


def _read_pdf(data: bytes) -> ExtractedText:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise ReaderError(f"unreadable PDF: {exc}") from exc

    if reader.is_encrypted:
        # An empty-password decrypt succeeds for the common "owner password only" case;
        # anything else genuinely needs a secret we do not have.
        try:
            if not reader.decrypt(""):
                raise ReaderError("PDF is password-protected")
        except (NotImplementedError, PdfReadError) as exc:
            raise ReaderError(f"PDF uses unsupported encryption: {exc}") from exc

    parts: list[str] = []
    page_starts: list[int] = []
    cursor = 0
    for page in reader.pages:
        try:
            body = page.extract_text() or ""
        except Exception:  # noqa: BLE001 — one malformed page must not lose the document
            body = ""
        page_starts.append(cursor)
        parts.append(body)
        cursor += len(body) + len(_PAGE_SEPARATOR)
    return ExtractedText(text=_PAGE_SEPARATOR.join(parts), page_starts=page_starts)


def _read_text(data: bytes) -> ExtractedText:
    # No upstream validation layer proved this is UTF-8 (see module docstring), so
    # errors="replace" is not belt-and-braces here, it is the only guard: a local corpus
    # file saved in the wrong encoding must degrade to replacement characters, not crash
    # the whole index build over one file. "utf-8-sig" rather than "utf-8": a BOM saved
    # by e.g. Notepad would otherwise survive as a literal U+FEFF at char_start=0 of
    # every such document, silently corrupting the first chunk's citation span; the
    # codec strips it when present and is a no-op otherwise.
    return ExtractedText(text=data.decode("utf-8-sig", errors="replace"))


def extract_text(data: bytes, kind: DocumentKind) -> ExtractedText:
    """Dispatch on ``kind`` (as decided by ``corpus.py`` from the file extension)."""
    if kind == "pdf":
        return _read_pdf(data)
    if kind in ("txt", "md"):
        return _read_text(data)
    raise ReaderError(f"no reader for kind {kind!r}")
