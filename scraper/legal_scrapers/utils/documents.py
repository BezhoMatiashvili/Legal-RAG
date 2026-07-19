"""Extract document bodies from binary downloads (PDF, DOCX) into Markdown/text.

napr serves the actual decision only as a PDF; the courts additionally offer DOCX.
We extract the text so every item carries a ``body_markdown`` like the matsne items.
The heavy parsers are imported lazily so spiders that never touch binaries don't pay
the import cost.

These inputs are UNTRUSTED downloads, so DOCX (a zip) and PDF are guarded against
decompression bombs / pathological page counts before being parsed in memory.
"""

import io
import multiprocessing
import resource
import zipfile
from dataclasses import dataclass
from enum import StrEnum

from .markdown import html_to_markdown
from .text import plain_text_to_markdown

MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024  # reject DOCX that decompress beyond this
MAX_COMPRESSION_RATIO = 200                 # ...or with a suspiciously high ratio
MAX_PDF_PAGES = 3000                        # cap pages parsed from a single PDF
MAX_BINARY_BYTES = 100 * 1024 * 1024
MAX_OUTPUT_CHARS = 20_000_000
WALL_TIME_SECONDS = 120
CPU_TIME_SECONDS = 90
ADDRESS_SPACE_BYTES = 1536 * 1024 * 1024


class ExtractionStatus(StrEnum):
    FULL_TEXT = "full_text"
    SCANNED_NO_TEXT = "scanned_no_text"
    TRUNCATED = "truncated"
    MALFORMED = "malformed"
    RESOURCE_LIMITED = "resource_limited"


@dataclass(frozen=True)
class ExtractionLimits:
    max_input_bytes: int = MAX_BINARY_BYTES
    max_pdf_pages: int = MAX_PDF_PAGES
    max_output_chars: int = MAX_OUTPUT_CHARS
    wall_seconds: int = WALL_TIME_SECONDS
    cpu_seconds: int = CPU_TIME_SECONDS
    address_space_bytes: int = ADDRESS_SPACE_BYTES


@dataclass(frozen=True)
class PageBoundary:
    """One PDF page's exact half-open range in emitted ``body_markdown``.

    Page numbers are one-based physical PDF page numbers.  Character offsets use
    Python/JSON Unicode code-point coordinates.  The two newlines inserted between
    pages are deliberately outside either page's range; substantive page text is
    therefore never attributed to an adjacent page.
    """

    page_number: int
    char_start: int
    char_end: int

    def as_dict(self) -> dict[str, int]:
        return {
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


PAGE_COORDINATE_REASON_EXACT_PDF_TEXT = "exact_pdf_text"


@dataclass(frozen=True)
class ExtractionResult:
    text: str
    status: ExtractionStatus
    content_kind: str = "full_text"
    content_complete: bool = False
    detected_mime: str | None = None
    detail: str = ""
    page_boundaries: tuple[PageBoundary, ...] = ()
    page_coordinate_reason: str | None = None


DEFAULT_EXTRACTION_LIMITS = ExtractionLimits()


def _mime(value: str | bytes | None) -> str | None:
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    if not value:
        return None
    return value.split(";", 1)[0].strip().lower() or None


def _malformed(detail: str, *, mime: str | None = None) -> ExtractionResult:
    return ExtractionResult(
        text="",
        status=ExtractionStatus.MALFORMED,
        detected_mime=mime,
        detail=detail[:500],
    )


def _validate_input(
    data: bytes,
    *,
    kind: str,
    declared_mime: str | bytes | None,
    limits: ExtractionLimits,
) -> ExtractionResult | None:
    mime = _mime(declared_mime)
    if not isinstance(data, bytes) or not data:
        return _malformed("empty or non-byte document", mime=mime)
    if len(data) > limits.max_input_bytes:
        return ExtractionResult(
            text="",
            status=ExtractionStatus.RESOURCE_LIMITED,
            detected_mime=mime,
            detail=f"input exceeds {limits.max_input_bytes} bytes",
        )
    if kind == "pdf":
        allowed = {
            None,
            "application/pdf",
            "application/x-pdf",
            "application/octet-stream",
            "binary/octet-stream",
            "application/download",
        }
        if mime not in allowed:
            return _malformed(f"unexpected PDF MIME type {mime!r}", mime=mime)
        if not data[:1024].lstrip().startswith(b"%PDF-"):
            return _malformed("PDF signature is missing", mime=mime)
    elif kind == "docx":
        allowed = {
            None,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/zip",
            "application/octet-stream",
            "binary/octet-stream",
        }
        if mime not in allowed:
            return _malformed(f"unexpected DOCX MIME type {mime!r}", mime=mime)
        if not data.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            return _malformed("DOCX zip signature is missing", mime=mime)
    else:  # defensive: callers expose only the two supported formats
        return _malformed(f"unsupported extraction kind {kind!r}", mime=mime)
    return None


def _text_result(
    text: str,
    *,
    limits: ExtractionLimits,
    truncated: bool,
    mime: str | None,
) -> ExtractionResult:
    markdown = plain_text_to_markdown(text)
    if not markdown.strip():
        return ExtractionResult(
            text="",
            status=ExtractionStatus.SCANNED_NO_TEXT,
            detected_mime=mime,
            detail="document contains no extractable text",
        )
    if len(markdown) > limits.max_output_chars:
        markdown = markdown[: limits.max_output_chars]
        truncated = True
    status = ExtractionStatus.TRUNCATED if truncated else ExtractionStatus.FULL_TEXT
    return ExtractionResult(
        text=markdown,
        status=status,
        content_complete=status is ExtractionStatus.FULL_TEXT,
        detected_mime=mime,
        detail="configured page/output ceiling reached" if truncated else "",
    )


def _extract_pdf_payload(
    data: bytes, limits: ExtractionLimits, mime: str | None
) -> ExtractionResult:
    import pdfplumber

    parts: list[str] = []
    characters = 0
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        pages = pdf.pages
        truncated = len(pages) > limits.max_pdf_pages
        for page in pages[: limits.max_pdf_pages]:
            text = page.extract_text() or ""
            parts.append(text)
            characters += len(text) + 2
            if characters > limits.max_output_chars:
                truncated = True
                break
    return _pdf_text_result(
        parts, limits=limits, truncated=truncated, mime=mime
    )


def _pdf_text_result(
    pages: list[str],
    *,
    limits: ExtractionLimits,
    truncated: bool,
    mime: str | None,
) -> ExtractionResult:
    """Normalize pages independently and retain exact offsets in the final text."""

    output_parts: list[str] = []
    boundaries: list[PageBoundary] = []
    output_length = 0

    for page_number, raw_text in enumerate(pages, start=1):
        page_text = plain_text_to_markdown(raw_text)
        separator = "" if page_number == 1 else "\n\n"
        remaining = limits.max_output_chars - output_length
        if remaining < len(separator):
            output_parts.append(separator[: max(remaining, 0)])
            output_length += max(remaining, 0)
            truncated = True
            break

        output_parts.append(separator)
        output_length += len(separator)
        char_start = output_length
        remaining = limits.max_output_chars - output_length
        emitted_page_text = page_text[:remaining]
        output_parts.append(emitted_page_text)
        output_length += len(emitted_page_text)
        boundaries.append(
            PageBoundary(
                page_number=page_number,
                char_start=char_start,
                char_end=output_length,
            )
        )
        if len(emitted_page_text) != len(page_text):
            truncated = True
            break

    markdown = "".join(output_parts)
    if not markdown.strip():
        return ExtractionResult(
            text="",
            status=ExtractionStatus.SCANNED_NO_TEXT,
            detected_mime=mime,
            detail="document contains no extractable text",
        )

    status = ExtractionStatus.TRUNCATED if truncated else ExtractionStatus.FULL_TEXT
    return ExtractionResult(
        text=markdown,
        status=status,
        content_complete=status is ExtractionStatus.FULL_TEXT,
        detected_mime=mime,
        detail="configured page/output ceiling reached" if truncated else "",
        page_boundaries=tuple(boundaries),
        page_coordinate_reason=PAGE_COORDINATE_REASON_EXACT_PDF_TEXT,
    )


def _extract_docx_payload(
    data: bytes, limits: ExtractionLimits, mime: str | None
) -> ExtractionResult:
    import mammoth

    _guard_docx(data)
    result = mammoth.convert_to_html(io.BytesIO(data))
    markdown = html_to_markdown(result.value)
    return _text_result(markdown, limits=limits, truncated=False, mime=mime)


def _apply_process_limits(limits: ExtractionLimits) -> None:
    cpu_hard = limits.cpu_seconds + 1
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, cpu_hard))
    resource.setrlimit(
        resource.RLIMIT_AS,
        (limits.address_space_bytes, limits.address_space_bytes),
    )


def _extraction_worker(conn, kind, data, limits, mime) -> None:
    try:
        _apply_process_limits(limits)
        if kind == "pdf":
            result = _extract_pdf_payload(data, limits, mime)
        else:
            result = _extract_docx_payload(data, limits, mime)
    except MemoryError:
        result = ExtractionResult(
            text="",
            status=ExtractionStatus.RESOURCE_LIMITED,
            detected_mime=mime,
            detail="address-space limit reached",
        )
    except Exception as exc:  # untrusted parser/input; convert to an auditable status
        result = _malformed(f"{type(exc).__name__}: {exc}", mime=mime)
    try:
        conn.send(result)
    finally:
        conn.close()


def _run_isolated_extraction(
    kind: str, data: bytes, limits: ExtractionLimits, mime: str | None
) -> ExtractionResult:
    context = multiprocessing.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_extraction_worker,
        args=(child_conn, kind, data, limits, mime),
        daemon=True,
    )
    try:
        process.start()
        child_conn.close()
        if parent_conn.poll(limits.wall_seconds):
            try:
                result = parent_conn.recv()
            except EOFError:
                result = None
            process.join(timeout=1)
            if isinstance(result, ExtractionResult):
                return result
            return ExtractionResult(
                text="",
                status=ExtractionStatus.RESOURCE_LIMITED,
                detected_mime=mime,
                detail=f"extractor exited without a result (exitcode={process.exitcode})",
            )
        process.terminate()
        process.join(timeout=1)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        return ExtractionResult(
            text="",
            status=ExtractionStatus.RESOURCE_LIMITED,
            detected_mime=mime,
            detail=f"wall-time limit of {limits.wall_seconds}s reached",
        )
    except (OSError, RuntimeError) as exc:
        return ExtractionResult(
            text="",
            status=ExtractionStatus.RESOURCE_LIMITED,
            detected_mime=mime,
            detail=f"extractor process failed: {exc}",
        )
    finally:
        parent_conn.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)


def pdf_to_markdown(
    data: bytes,
    *,
    declared_mime: str | bytes | None = None,
    limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
) -> ExtractionResult:
    """Extract a PDF in a resource-limited child process and return a typed result."""
    invalid = _validate_input(
        data, kind="pdf", declared_mime=declared_mime, limits=limits
    )
    if invalid is not None:
        return invalid
    return _run_isolated_extraction("pdf", data, limits, _mime(declared_mime))


def _guard_docx(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
            if not {"[Content_Types].xml", "word/document.xml"} <= names:
                raise ValueError("zip is not a DOCX document")
            uncompressed = sum(info.file_size for info in zf.infolist())
    except zipfile.BadZipFile as exc:
        raise ValueError("not a valid DOCX (zip) file") from exc
    compressed = max(len(data), 1)
    if uncompressed > MAX_UNCOMPRESSED_BYTES or uncompressed / compressed > MAX_COMPRESSION_RATIO:
        raise ValueError(
            f"refusing suspicious DOCX: {uncompressed} uncompressed bytes from "
            f"{compressed} compressed (ratio {uncompressed / compressed:.0f})"
        )


def docx_to_markdown(
    data: bytes,
    *,
    declared_mime: str | bytes | None = None,
    limits: ExtractionLimits = DEFAULT_EXTRACTION_LIMITS,
) -> ExtractionResult:
    """Extract a DOCX in a resource-limited child process and return a typed result."""
    invalid = _validate_input(
        data, kind="docx", declared_mime=declared_mime, limits=limits
    )
    if invalid is not None:
        return invalid
    return _run_isolated_extraction("docx", data, limits, _mime(declared_mime))
