"""Format-specific extraction: decoding edge cases, PDF page offsets, and the failure
modes that must degrade gracefully rather than lose a whole document."""

from __future__ import annotations

import pytest

from mmu.ingest.reader import ExtractedText, ReaderError, extract_text
from tests.pdf_fixture import build_pdf

# --------------------------------------------------------------------------- txt / md


def test_plain_utf8_text_round_trips() -> None:
    extracted = extract_text("Hello, world.".encode("utf-8"), "txt")
    assert extracted.text == "Hello, world."
    assert extracted.page_starts == []
    assert extracted.pages is None


def test_md_kind_uses_the_same_text_path() -> None:
    extracted = extract_text("# Heading\n\nBody.".encode("utf-8"), "md")
    assert extracted.text == "# Heading\n\nBody."


def test_utf8_bom_is_stripped_not_left_as_a_literal_character() -> None:
    data = b"\xef\xbb\xbf" + "Article 33 obligations.".encode("utf-8")
    extracted = extract_text(data, "txt")
    assert extracted.text == "Article 33 obligations."
    assert "﻿" not in extracted.text


def test_invalid_utf8_degrades_to_replacement_characters_instead_of_raising() -> None:
    # A lone continuation byte (0x80) is never valid at the start of a UTF-8 sequence.
    data = b"before " + b"\x80\x80" + b" after"
    extracted = extract_text(data, "txt")
    assert "before" in extracted.text
    assert "after" in extracted.text
    assert "�" in extracted.text  # the replacement character, not a crash


def test_empty_text_file() -> None:
    extracted = extract_text(b"", "txt")
    assert extracted.text == ""
    assert extracted.pages is None


# --------------------------------------------------------------------------------- pdf


def test_pdf_extracts_text_via_build_pdf_fixture() -> None:
    pdf_bytes = build_pdf([["Alpha line one.", "Alpha line two."]])
    extracted = extract_text(pdf_bytes, "pdf")
    assert extracted.pages == 1
    assert "Alpha line one." in extracted.text
    assert "Alpha line two." in extracted.text


def test_pdf_page_offsets_align_with_the_joined_text() -> None:
    pdf_bytes = build_pdf(
        [["FIRSTPAGEMARKER content."], ["SECONDPAGEMARKER content."], ["THIRDPAGEMARKER content."]]
    )
    extracted = extract_text(pdf_bytes, "pdf")
    assert extracted.pages == 3
    assert len(extracted.page_starts) == 3

    markers = ["FIRSTPAGEMARKER", "SECONDPAGEMARKER", "THIRDPAGEMARKER"]
    bounds = [*extracted.page_starts, len(extracted.text)]
    for i, marker in enumerate(markers):
        pos = extracted.text.index(marker)
        # Each marker falls within its own page's declared span, not a neighbour's.
        assert bounds[i] <= pos < bounds[i + 1]
        # And char_start:char_end over the joined text reproduces it, same invariant
        # the splitter relies on downstream.
        assert extracted.text[bounds[i] : bounds[i + 1]].count(marker) >= 1


def test_encrypted_pdf_raises_reader_error() -> None:
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    # A real user password means the reader's empty-password decrypt attempt fails —
    # this is the "genuinely needs a secret we do not have" branch, not the
    # owner-password-only case that _read_pdf silently recovers from.
    writer.encrypt(user_password="secret", owner_password="owner-secret")

    import io

    buf = io.BytesIO()
    writer.write(buf)

    with pytest.raises(ReaderError):
        extract_text(buf.getvalue(), "pdf")


def test_malformed_page_does_not_lose_the_whole_document(monkeypatch: pytest.MonkeyPatch) -> None:
    from pypdf._page import PageObject

    pdf_bytes = build_pdf([["Page one text."], ["Page two text."], ["Page three text."]])

    original_extract_text = PageObject.extract_text
    calls = {"n": 0}

    def flaky_extract_text(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 2:  # sabotage exactly the second page
            raise RuntimeError("simulated malformed page")
        return original_extract_text(self, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", flaky_extract_text)

    extracted = extract_text(pdf_bytes, "pdf")
    assert extracted.pages == 3
    assert "Page one text." in extracted.text
    assert "Page three text." in extracted.text
    # The sabotaged page contributed an empty body rather than aborting the document.
    assert calls["n"] == 3


def test_unreadable_pdf_bytes_raise_reader_error() -> None:
    with pytest.raises(ReaderError):
        extract_text(b"%PDF-1.4\nnot actually a pdf structure", "pdf")


# ---------------------------------------------------------------------------- dispatch


def test_unsupported_kind_raises_reader_error() -> None:
    with pytest.raises(ReaderError):
        extract_text(b"data", "docx")  # type: ignore[arg-type]


def test_extracted_text_pages_property() -> None:
    assert ExtractedText(text="x", page_starts=[]).pages is None
    assert ExtractedText(text="x\n\ny", page_starts=[0, 2]).pages == 2
