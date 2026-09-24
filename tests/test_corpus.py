"""Corpus discovery: the slug that becomes doc_id, stable ordering, and the two
failure modes that must be loud — duplicate ids and a silently-empty scanned PDF."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmu.ingest.corpus import CorpusError, discover, scanned_pdf_warning, slugify
from mmu.ingest.reader import ExtractedText

# ------------------------------------------------------------------------------ slugify


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("GDPR (EU) 2016-679", "gdpr-eu-2016-679"),
        ("simple", "simple"),
        ("Mixed CASE Words", "mixed-case-words"),
        ("multiple   spaces   here", "multiple-spaces-here"),
        ("--leading and trailing--", "leading-and-trailing"),
        ("a___b---c", "a-b-c"),  # repeated separators of different kinds collapse to one
        ("Résumé Café", "r-sum-caf"),  # unicode letters fold to hyphens, not transliterated
        ("already-lowercase-slug", "already-lowercase-slug"),
        ("2024.09.21_report_v2", "2024-09-21-report-v2"),
    ],
)
def test_slugify_cases(stem: str, expected: str) -> None:
    assert slugify(stem) == expected


def test_slugify_never_leaves_leading_or_trailing_hyphens() -> None:
    assert not slugify("!!!weird!!!").startswith("-")
    assert not slugify("!!!weird!!!").endswith("-")


# ------------------------------------------------------------------------------ discover


def _touch(path: Path, content: str = "text") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_discover_maps_extensions_to_kind(tmp_path: Path) -> None:
    _touch(tmp_path / "gdpr.pdf", "%PDF-1.4 fake")
    _touch(tmp_path / "policy.txt")
    _touch(tmp_path / "notes.md")

    docs = discover(tmp_path)
    by_id = {d.doc_id: d for d in docs}
    assert by_id["gdpr"].kind == "pdf"
    assert by_id["policy"].kind == "txt"
    assert by_id["notes"].kind == "md"


def test_discover_stable_sort_order_by_relative_path(tmp_path: Path) -> None:
    _touch(tmp_path / "zebra.txt")
    _touch(tmp_path / "alpha.txt")
    _touch(tmp_path / "sub" / "beta.txt")
    _touch(tmp_path / "middle.md")

    docs = discover(tmp_path)
    actual_order = [d.path.relative_to(tmp_path).as_posix() for d in docs]
    # The one true ordering: relative path, ascending, independent of directory nesting.
    assert actual_order == ["alpha.txt", "middle.md", "sub/beta.txt", "zebra.txt"]

    # Re-run discover to prove the order is reproducible, not merely sorted once here.
    docs_again = discover(tmp_path)
    assert [d.doc_id for d in docs] == [d.doc_id for d in docs_again]


def test_discover_is_recursive(tmp_path: Path) -> None:
    _touch(tmp_path / "top.txt")
    _touch(tmp_path / "nested" / "deep" / "bottom.txt")
    docs = discover(tmp_path)
    assert {d.doc_id for d in docs} == {"top", "bottom"}


def test_discover_raises_on_duplicate_doc_id(tmp_path: Path) -> None:
    _touch(tmp_path / "GDPR.txt")
    _touch(tmp_path / "gdpr.md")  # slugifies to the same "gdpr"
    with pytest.raises(CorpusError):
        discover(tmp_path)


def test_discover_skips_unsupported_extensions(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _touch(tmp_path / "keep.txt")
    _touch(tmp_path / "skip.docx")
    _touch(tmp_path / "skip.csv")

    with caplog.at_level("WARNING"):
        docs = discover(tmp_path)

    assert [d.doc_id for d in docs] == ["keep"]
    assert "skip.docx" in caplog.text or "skip.csv" in caplog.text


def test_discover_skips_dotfiles_and_gitkeep(tmp_path: Path) -> None:
    _touch(tmp_path / ".gitkeep")
    _touch(tmp_path / ".hidden.txt")
    _touch(tmp_path / "visible.txt")

    docs = discover(tmp_path)
    assert [d.doc_id for d in docs] == ["visible"]


def test_discover_empty_directory(tmp_path: Path) -> None:
    assert discover(tmp_path) == []


# ------------------------------------------------------------------------ scanned_pdf_warning


def test_scanned_pdf_warning_fires_below_threshold() -> None:
    # 3 pages, ~150 chars total => 50 chars/page average, well under the 200 floor.
    extracted = ExtractedText(text="x" * 150, page_starts=[0, 50, 100])
    warning = scanned_pdf_warning(extracted, "scanned.pdf")
    assert warning is not None
    assert "scanned.pdf" in warning


def test_scanned_pdf_warning_stays_quiet_above_threshold() -> None:
    # 2 pages, well over 200 chars/page.
    text = "Genuine extracted legal prose. " * 50
    extracted = ExtractedText(text=text, page_starts=[0, len(text) // 2])
    assert scanned_pdf_warning(extracted, "normal.pdf") is None


def test_scanned_pdf_warning_quiet_for_non_paginated_formats() -> None:
    # txt/md never have page_starts, so this must not be mistaken for a scanned PDF.
    extracted = ExtractedText(text="short", page_starts=[])
    assert scanned_pdf_warning(extracted, "notes.txt") is None
