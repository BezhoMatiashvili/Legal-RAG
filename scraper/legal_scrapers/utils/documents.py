"""Extract document bodies from binary downloads (PDF, DOCX) into Markdown/text.

napr serves the actual decision only as a PDF; the courts additionally offer DOCX.
We extract the text so every item carries a ``body_markdown`` like the matsne items.
The heavy parsers are imported lazily so spiders that never touch binaries don't pay
the import cost.

These inputs are UNTRUSTED downloads, so DOCX (a zip) and PDF are guarded against
decompression bombs / pathological page counts before being parsed in memory.
"""

import io
import zipfile

from .markdown import html_to_markdown
from .text import plain_text_to_markdown

MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024  # reject DOCX that decompress beyond this
MAX_COMPRESSION_RATIO = 200                 # ...or with a suspiciously high ratio
MAX_PDF_PAGES = 3000                        # cap pages parsed from a single PDF


def pdf_to_markdown(data: bytes) -> str:
    """Extract text from a PDF byte string (page-capped) and tidy it into Markdown-ish text."""
    import pdfplumber

    parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages[:MAX_PDF_PAGES]:
            parts.append(page.extract_text() or "")
    return plain_text_to_markdown("\n\n".join(parts))


def _guard_docx(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            uncompressed = sum(info.file_size for info in zf.infolist())
    except zipfile.BadZipFile as exc:
        raise ValueError("not a valid DOCX (zip) file") from exc
    compressed = max(len(data), 1)
    if uncompressed > MAX_UNCOMPRESSED_BYTES or uncompressed / compressed > MAX_COMPRESSION_RATIO:
        raise ValueError(
            f"refusing suspicious DOCX: {uncompressed} uncompressed bytes from "
            f"{compressed} compressed (ratio {uncompressed / compressed:.0f})"
        )


def docx_to_markdown(data: bytes) -> str:
    """Convert a DOCX byte string to Markdown by way of HTML (mammoth), after a bomb guard."""
    import mammoth

    _guard_docx(data)
    result = mammoth.convert_to_html(io.BytesIO(data))
    return html_to_markdown(result.value)
