"""Build a minimal, real PDF in memory.

Hand-rolled rather than pulled from a rendering library: the point is to test *our*
reader against a genuine PDF byte stream (xref table, page tree, content streams) with
zero new dependencies and no binary blob checked into the repo. Also used by
`scripts/index_smoke.py` to produce a corpus file for the live index path.
"""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(pages: list[list[str]]) -> bytes:
    """One PDF, ``pages[i]`` being the lines of text drawn on page ``i+1``.

    Objects are emitted in a fixed order and the xref offsets are computed from the real
    byte positions, because pypdf validates them.
    """
    if not pages:
        raise ValueError("a PDF needs at least one page")

    objects: list[bytes] = []
    n_pages = len(pages)
    # 1 = catalog, 2 = page tree, 3 = font; then per page: a Page and its Contents.
    page_ids = [4 + 2 * i for i in range(n_pages)]

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode("latin-1")
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for i, lines in enumerate(pages):
        content_id = page_ids[i] + 1
        body = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for line in lines:
            body.append(f"({_escape(line)}) Tj")
            body.append("T*")
        body.append("ET")
        stream = "\n".join(body).encode("latin-1")
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("latin-1")
        )
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode("latin-1")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: list[int] = []
    for number, payload in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + payload + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)
