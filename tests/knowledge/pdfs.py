"""Small PDFs written byte by byte, so a test knows exactly what each page holds."""

import io
from collections.abc import Sequence

from pypdf import PdfReader, PdfWriter

LEADING = 14


def _literal(line: str) -> bytes:
    escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"({escaped})".encode("latin-1")


def _content(lines: Sequence[str]) -> bytes:
    if not lines:
        return b""
    moves = [b"BT /F1 12 Tf 72 720 Td"]
    for number, line in enumerate(lines):
        if number:
            moves.append(f"0 -{LEADING} Td".encode())
        moves.append(_literal(line) + b" Tj")
    moves.append(b"ET")
    return b"\n".join(moves)


def pdf_of(pages: Sequence[Sequence[str]]) -> bytes:
    """Return a PDF with one page per item, one Helvetica text line per string.

    A page with no lines is an empty page, as a scanned page reads.
    """
    first_page = 4
    kids = " ".join(f"{first_page + 2 * n} 0 R" for n in range(len(pages)))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for n, lines in enumerate(pages):
        stream = _content(lines)
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]"
            b" /Resources << /Font << /F1 3 0 R >> >>"
            + f" /Contents {first_page + 2 * n + 1} 0 R >>".encode()
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def encrypted(data: bytes, password: str = "") -> bytes:
    """Return ``data`` encrypted by pypdf, opened by ``password`` (may be empty)."""
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(data)))
    writer.encrypt(password, "owner")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
