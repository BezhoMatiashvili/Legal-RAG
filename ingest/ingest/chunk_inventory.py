"""Deterministic structural-chunk inventory for immutable corpus snapshots.

The inventory is computed solely from sealed canonical document rows and an injected
token counter.  It deliberately excludes embedding/Qdrant state, so snapshot creation
and embedding can reproduce the same bytes independently before any collection access.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .chunking import (
    EMBED_HEADER_REVISION,
    STRUCTURAL_CHUNKER_REVISION,
    Chunk,
    build_embed_text,
    chunk_document,
)
from .sources import PageBoundary

CHUNK_INVENTORY_SCHEMA_VERSION = 2
CHUNK_INVENTORY_FILENAME = "structural_chunk_inventory.jsonl"
CHUNK_INVENTORY_FORMAT = "canonical-jsonl-header+document-rows-v2"
CHUNK_INVENTORY_KIND = "structural-chunk-inventory"
CHUNK_INVENTORY_DOCUMENT_KIND = "structural-chunk-document"
CHUNK_INVENTORY_STATUS_AVAILABLE = "available"
CHUNK_INVENTORY_STATUS_UNAVAILABLE = "unavailable"
CHUNK_INVENTORY_UNAVAILABLE_REASON = "token_counter_unavailable"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ChunkInventoryError(RuntimeError):
    """The structural chunk inventory cannot be computed or validated exactly."""


@dataclass(frozen=True, slots=True)
class StructuralChunkInventory:
    identity: Mapping[str, Any]
    identity_sha256: str
    artifact_sha256: str
    size_bytes: int
    record_count: int
    document_count: int
    chunk_count: int
    source_counts: Mapping[str, Mapping[str, int]]

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "schema_version": CHUNK_INVENTORY_SCHEMA_VERSION,
            "status": CHUNK_INVENTORY_STATUS_AVAILABLE,
            "reason": None,
            "format": CHUNK_INVENTORY_FORMAT,
            "path": CHUNK_INVENTORY_FILENAME,
            "sha256": self.artifact_sha256,
            "size_bytes": self.size_bytes,
            "identity": dict(self.identity),
            "identity_sha256": self.identity_sha256,
            "record_count": self.record_count,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "source_counts": {
                source: dict(counts) for source, counts in self.source_counts.items()
            },
        }


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ChunkInventoryError(f"chunk inventory value is not canonical JSON: {exc}") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def chunk_inventory_identity(
    *,
    sources: Sequence[str],
    tokenizer_model: str,
    tokenizer_revision: str | None,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    document_header: bool,
) -> dict[str, Any]:
    """Return the complete tokenizer/chunker/header identity for inventory bytes."""

    if (
        not isinstance(tokenizer_model, str)
        or not tokenizer_model
        or tokenizer_revision is not None
        and (not isinstance(tokenizer_revision, str) or not tokenizer_revision)
    ):
        raise ChunkInventoryError("tokenizer identity is invalid")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
        or isinstance(overlap_tokens, bool)
        or not isinstance(overlap_tokens, int)
        or overlap_tokens < 0
        or overlap_tokens >= max_tokens
        or isinstance(min_tokens, bool)
        or not isinstance(min_tokens, int)
        or min_tokens < 1
        or min_tokens > max_tokens
    ):
        raise ChunkInventoryError("chunk budget identity is invalid")
    canonical_sources = list(sources)
    if (
        not canonical_sources
        or len(canonical_sources) != len(set(canonical_sources))
        or any(not isinstance(source, str) or not source for source in canonical_sources)
    ):
        raise ChunkInventoryError("chunk inventory source order is invalid")
    if not isinstance(document_header, bool):
        raise ChunkInventoryError("document_header must be boolean")
    return {
        "sources": canonical_sources,
        "tokenizer": {
            "model": tokenizer_model,
            "revision": tokenizer_revision,
        },
        "chunker": {
            "revision": STRUCTURAL_CHUNKER_REVISION,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "min_tokens": min_tokens,
        },
        "document_header": {
            "enabled": document_header,
            "revision": EMBED_HEADER_REVISION,
        },
    }


def unavailable_manifest_entry(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Represent a deliberately offline noncandidate/preflight snapshot safely."""

    identity_value = dict(identity)
    return {
        "schema_version": CHUNK_INVENTORY_SCHEMA_VERSION,
        "status": CHUNK_INVENTORY_STATUS_UNAVAILABLE,
        "reason": CHUNK_INVENTORY_UNAVAILABLE_REASON,
        "format": CHUNK_INVENTORY_FORMAT,
        "path": None,
        "sha256": None,
        "size_bytes": None,
        "identity": identity_value,
        "identity_sha256": _sha256(identity_value),
        "record_count": None,
        "document_count": None,
        "chunk_count": None,
        "source_counts": None,
    }


def _strict_json(line: str, *, origin: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ChunkInventoryError(f"{origin} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            line,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ChunkInventoryError(f"{origin} contains non-finite number {item!r}")
            ),
        )
    except ChunkInventoryError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ChunkInventoryError(f"cannot parse {origin}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ChunkInventoryError(f"{origin} must be a JSON object")
    return value


def _document_pages(
    record: Mapping[str, Any], body: str
) -> tuple[tuple[PageBoundary, ...], str, list[dict[str, int]], str | None]:
    raw = record.get("page_boundaries")
    reason = record.get("page_coordinate_reason")
    if not isinstance(raw, list):
        raise ChunkInventoryError("snapshot page_boundaries must be an array")
    if not isinstance(reason, str) or not reason:
        raise ChunkInventoryError("snapshot page_coordinate_reason is invalid")
    boundaries: list[PageBoundary] = []
    previous_end = 0
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping) or set(value) not in (
            {"page", "char_start", "char_end"},
            {"page_number", "char_start", "char_end"},
        ):
            raise ChunkInventoryError(f"page_boundaries[{index}] has invalid keys")
        page = value.get("page", value.get("page_number"))
        start = value["char_start"]
        end = value["char_end"]
        if any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in (page, start, end)
        ):
            raise ChunkInventoryError(f"page_boundaries[{index}] values are invalid")
        if page != index + 1 or start < previous_end or end < start or end > len(body):
            raise ChunkInventoryError(f"page_boundaries[{index}] range is invalid")
        boundaries.append(PageBoundary(page=page, char_start=start, char_end=end))
        previous_end = end
    if boundaries and reason != "exact_pdf_text":
        raise ChunkInventoryError("page boundaries require exact_pdf_text coordinates")
    if not boundaries and reason != "source_not_paginated":
        raise ChunkInventoryError(
            "admissible documents without pages require source_not_paginated"
        )
    canonical = [boundary.to_dict() for boundary in boundaries]
    mapping_sha = _sha256(canonical) if canonical else None
    return tuple(boundaries), reason, canonical, mapping_sha


def _structure_value(chunk: Chunk) -> dict[str, Any]:
    return {
        "heading_path": list(chunk.heading_path),
        "article_id": chunk.article_id,
        "article_label": chunk.article_label,
        "article_start": chunk.article_start,
        "clause": chunk.clause,
        "clause_id": chunk.clause_id,
        "subarticle": chunk.subarticle,
        "chapter": chunk.chapter,
        "parent_id": chunk.parent_id,
        "clause_ids": list(chunk.clause_ids),
        "subarticle_ids": list(chunk.subarticle_ids),
        "article_start_chunk_index": chunk.article_start_chunk_index,
        "parent_chunk_index": chunk.parent_chunk_index,
        "chunker_revision": chunk.chunker_revision,
    }


def _chunk_value(
    chunk: Chunk,
    *,
    document_chunk_count: int,
    page_mapping_sha256: str | None,
    embed_text: str,
    count_tokens: Callable[[str], int],
) -> dict[str, Any]:
    passage = chunk.canonical_text if chunk.canonical_text is not None else chunk.text
    passage_bytes = passage.encode("utf-8")
    passage_sha = hashlib.sha256(passage_bytes).hexdigest()
    if chunk.passage_hash is not None and chunk.passage_hash != passage_sha:
        raise ChunkInventoryError("chunk passage hash differs from canonical passage")
    return {
        "chunk_index": chunk.chunk_index,
        "document_chunk_count": document_chunk_count,
        "canonical_passage_sha256": passage_sha,
        "canonical_passage_char_length": len(passage),
        "canonical_passage_utf8_length": len(passage_bytes),
        "token_count": chunk.token_count,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "structure": _structure_value(chunk),
        "page": {
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "page_coordinate_reason": chunk.page_coordinate_reason,
            "page_boundary_mapping_sha256": page_mapping_sha256,
        },
        # The structural passage budget is deliberately distinct from the complete
        # context-enriched encoder input.  Recording both exact bytes and token count
        # exposes (without inventing) any later model-specific truncation risk.
        "embed_input": {
            "sha256": hashlib.sha256(embed_text.encode("utf-8")).hexdigest(),
            "char_length": len(embed_text),
            "utf8_length": len(embed_text.encode("utf-8")),
            "token_count": count_tokens(embed_text),
        },
    }


def _document_value(
    record: Mapping[str, Any],
    *,
    source: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    document_header: bool,
) -> tuple[dict[str, Any], int]:
    document_id = record.get("document_id")
    version_id = record.get("version_id")
    doc_id = record.get("doc_id")
    body = record.get("body_markdown")
    if record.get("source") != source:
        raise ChunkInventoryError("snapshot source/file identity mismatch")
    if (
        not isinstance(document_id, str)
        or not document_id
        or not isinstance(version_id, str)
        or not version_id
        or doc_id != f"{source}:{document_id}:{version_id}"
    ):
        raise ChunkInventoryError("snapshot document/version identity is invalid")
    if not isinstance(body, str) or not body:
        raise ChunkInventoryError("snapshot canonical body is empty")
    if record.get("admissible") is not True:
        raise ChunkInventoryError("chunk inventory refuses inadmissible snapshot documents")
    body_bytes = body.encode("utf-8")
    body_sha = hashlib.sha256(body_bytes).hexdigest()
    if record.get("content_hash") != body_sha or record.get("body_char_len") != len(body):
        raise ChunkInventoryError("snapshot canonical body hash/length mismatch")
    boundaries, reason, canonical_pages, page_mapping_sha = _document_pages(record, body)
    try:
        chunks = chunk_document(
            body,
            max_tokens=max_tokens,
            overlap=overlap_tokens,
            min_tokens=min_tokens,
            count_tokens=count_tokens,
            page_boundaries=boundaries,
            page_coordinate_reason=reason,
        )
    except Exception as exc:
        raise ChunkInventoryError(
            f"cannot structurally chunk {source}/{document_id}/{version_id}: {exc}"
        ) from exc
    if not chunks:
        raise ChunkInventoryError("admitted snapshot document produced no chunks")
    header_v2: dict[str, Any] = {}
    if document_header:
        header_v2 = {
            "document_number": record.get("document_number"),
            "date": record.get("date") or record.get("date_raw"),
            "status": record.get("status"),
            "is_consolidated": record.get("is_consolidated"),
        }
    chunk_rows = []
    for chunk in chunks:
        embed_text = build_embed_text(
            chunk.text,
            title=record.get("title"),
            document_type=record.get("document_type"),
            heading_path=chunk.heading_path,
            **header_v2,
        )
        chunk_rows.append(
            _chunk_value(
                chunk,
                document_chunk_count=len(chunks),
                page_mapping_sha256=page_mapping_sha,
                embed_text=embed_text,
                count_tokens=count_tokens,
            )
        )
    return (
        {
            "schema_version": CHUNK_INVENTORY_SCHEMA_VERSION,
            "kind": CHUNK_INVENTORY_DOCUMENT_KIND,
            "source": source,
            "document_id": document_id,
            "version_id": version_id,
            "doc_id": doc_id,
            "canonical_content_sha256": body_sha,
            "canonical_body_char_length": len(body),
            "canonical_body_utf8_length": len(body_bytes),
            "page_boundaries": canonical_pages,
            "page_boundary_mapping_sha256": page_mapping_sha,
            "page_coordinate_reason": reason,
            "chunk_count": len(chunks),
            "chunks": chunk_rows,
        },
        len(chunks),
    )


def _open_create_only(path: Path) -> BinaryIO:
    if not path.parent.exists() or path.parent.is_symlink() or not path.parent.is_dir():
        raise ChunkInventoryError("chunk inventory output parent must be a real directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ChunkInventoryError(f"cannot create chunk inventory {path}: {exc}") from exc
    try:
        return os.fdopen(descriptor, "wb")
    except Exception:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def compute_structural_chunk_inventory(
    docs_root: str | Path,
    *,
    sources: Sequence[str],
    tokenizer_model: str,
    tokenizer_revision: str,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    document_header: bool,
    count_tokens: Callable[[str], int],
    output: str | Path | None = None,
) -> StructuralChunkInventory:
    """Independently compute canonical inventory bytes, optionally create-only writing them."""

    identity = chunk_inventory_identity(
        sources=sources,
        tokenizer_model=tokenizer_model,
        tokenizer_revision=tokenizer_revision,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
        document_header=document_header,
    )
    identity_sha = _sha256(identity)
    root = Path(docs_root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir():
        raise ChunkInventoryError("snapshot docs root must be a real directory")
    destination = Path(output).expanduser().absolute() if output is not None else None
    handle: BinaryIO | None = None
    created_destination = False
    digest = hashlib.sha256()
    size_bytes = 0
    record_count = 0
    document_count = 0
    chunk_count = 0
    source_counts: dict[str, dict[str, int]] = {
        source: {"document_count": 0, "chunk_count": 0} for source in sources
    }
    seen_documents: set[tuple[str, str, str]] = set()

    def emit(value: object) -> None:
        nonlocal size_bytes, record_count
        encoded = _canonical_bytes(value) + b"\n"
        digest.update(encoded)
        size_bytes += len(encoded)
        record_count += 1
        if handle is not None:
            handle.write(encoded)

    try:
        if destination is not None:
            handle = _open_create_only(destination)
            created_destination = True
        emit(
            {
                "schema_version": CHUNK_INVENTORY_SCHEMA_VERSION,
                "kind": CHUNK_INVENTORY_KIND,
                "format": CHUNK_INVENTORY_FORMAT,
                "identity": identity,
                "identity_sha256": identity_sha,
            }
        )
        for source in sources:
            path = root / f"{source}.jsonl"
            try:
                info = path.lstat()
            except OSError as exc:
                raise ChunkInventoryError(f"cannot inspect snapshot source {path}: {exc}") from exc
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ChunkInventoryError(f"snapshot source is not a regular file: {path}")
            try:
                with path.open(encoding="utf-8") as source_file:
                    for line_number, line in enumerate(source_file, 1):
                        if not line.strip():
                            continue
                        record = _strict_json(line, origin=f"{path}:{line_number}")
                        value, chunks = _document_value(
                            record,
                            source=source,
                            count_tokens=count_tokens,
                            max_tokens=max_tokens,
                            overlap_tokens=overlap_tokens,
                            min_tokens=min_tokens,
                            document_header=document_header,
                        )
                        key = (source, value["document_id"], value["version_id"])
                        if key in seen_documents:
                            raise ChunkInventoryError(f"duplicate snapshot identity: {key!r}")
                        seen_documents.add(key)
                        emit(value)
                        document_count += 1
                        chunk_count += chunks
                        source_counts[source]["document_count"] += 1
                        source_counts[source]["chunk_count"] += chunks
            except (OSError, UnicodeError) as exc:
                raise ChunkInventoryError(f"cannot read snapshot source {path}: {exc}") from exc
        if handle is not None:
            handle.flush()
            os.fsync(handle.fileno())
        if destination is not None:
            # The snapshot sealer later fsyncs the complete tree as well; this makes the
            # standalone create-only API durable at its own directory-entry boundary.
            parent_descriptor = os.open(
                destination.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
    except Exception:
        if handle is not None:
            handle.close()
        if destination is not None and created_destination:
            destination.unlink(missing_ok=True)
        raise
    else:
        if handle is not None:
            handle.close()
    return StructuralChunkInventory(
        identity=identity,
        identity_sha256=identity_sha,
        artifact_sha256=digest.hexdigest(),
        size_bytes=size_bytes,
        record_count=record_count,
        document_count=document_count,
        chunk_count=chunk_count,
        source_counts=source_counts,
    )


def validate_manifest_entry(
    value: object,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate an available/unavailable snapshot manifest entry."""

    if not isinstance(value, Mapping):
        raise ChunkInventoryError("structural_chunk_inventory must be an object")
    expected_keys = {
        "schema_version",
        "status",
        "reason",
        "format",
        "path",
        "sha256",
        "size_bytes",
        "identity",
        "identity_sha256",
        "record_count",
        "document_count",
        "chunk_count",
        "source_counts",
    }
    if set(value) != expected_keys:
        raise ChunkInventoryError("structural_chunk_inventory keys are invalid")
    identity = value.get("identity")
    if (
        value.get("schema_version") != CHUNK_INVENTORY_SCHEMA_VERSION
        or value.get("format") != CHUNK_INVENTORY_FORMAT
        or not isinstance(identity, Mapping)
        or dict(identity) != dict(expected_identity)
        or value.get("identity_sha256") != _sha256(dict(expected_identity))
    ):
        raise ChunkInventoryError("structural chunk inventory identity mismatch")
    status = value.get("status")
    if status == CHUNK_INVENTORY_STATUS_UNAVAILABLE:
        if value != unavailable_manifest_entry(expected_identity):
            raise ChunkInventoryError("unavailable structural chunk inventory is invalid")
        return dict(value)
    if status != CHUNK_INVENTORY_STATUS_AVAILABLE:
        raise ChunkInventoryError("structural chunk inventory status is invalid")
    if value.get("reason") is not None or value.get("path") != CHUNK_INVENTORY_FILENAME:
        raise ChunkInventoryError("available structural chunk inventory path/reason is invalid")
    if not isinstance(value.get("sha256"), str) or not _SHA256_RE.fullmatch(value["sha256"]):
        raise ChunkInventoryError("structural chunk inventory SHA-256 is invalid")
    for field, minimum in (
        ("size_bytes", 1),
        ("record_count", 1),
        ("document_count", 0),
        ("chunk_count", 0),
    ):
        item = value.get(field)
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise ChunkInventoryError(f"structural chunk inventory {field} is invalid")
    if value["record_count"] != value["document_count"] + 1:
        raise ChunkInventoryError("structural chunk inventory record count mismatch")
    counts = value.get("source_counts")
    sources = expected_identity.get("sources")
    if not isinstance(counts, Mapping) or set(counts) != set(sources):
        raise ChunkInventoryError("structural chunk inventory source counts are invalid")
    documents = chunks = 0
    for source in sources:
        source_value = counts[source]
        if not isinstance(source_value, Mapping) or set(source_value) != {
            "document_count",
            "chunk_count",
        }:
            raise ChunkInventoryError("structural chunk source count shape is invalid")
        for field in ("document_count", "chunk_count"):
            item = source_value[field]
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ChunkInventoryError("structural chunk source count is invalid")
        documents += source_value["document_count"]
        chunks += source_value["chunk_count"]
    if documents != value["document_count"] or chunks != value["chunk_count"]:
        raise ChunkInventoryError("structural chunk inventory aggregate count mismatch")
    if chunks < documents:
        raise ChunkInventoryError("structural chunk inventory has fewer chunks than documents")
    return dict(value)


def _inventory_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ChunkInventoryError(f"inventory {field} must be integer >= {minimum}")
    return value


def _inventory_optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _inventory_int(value, field=field)


def _inventory_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ChunkInventoryError(f"inventory {field} must be null or non-empty string")
    return value


def _validated_document_row(
    value: Mapping[str, Any],
    *,
    expected_sources: Sequence[str],
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "kind",
        "source",
        "document_id",
        "version_id",
        "doc_id",
        "canonical_content_sha256",
        "canonical_body_char_length",
        "canonical_body_utf8_length",
        "page_boundaries",
        "page_boundary_mapping_sha256",
        "page_coordinate_reason",
        "chunk_count",
        "chunks",
    }
    if set(value) != expected_keys:
        raise ChunkInventoryError("structural chunk document row keys are invalid")
    source = value.get("source")
    document_id = value.get("document_id")
    version_id = value.get("version_id")
    if (
        value.get("schema_version") != CHUNK_INVENTORY_SCHEMA_VERSION
        or value.get("kind") != CHUNK_INVENTORY_DOCUMENT_KIND
        or source not in expected_sources
        or not isinstance(document_id, str)
        or not document_id
        or not isinstance(version_id, str)
        or not version_id
        or value.get("doc_id") != f"{source}:{document_id}:{version_id}"
    ):
        raise ChunkInventoryError("structural chunk document identity is invalid")
    content_sha = value.get("canonical_content_sha256")
    if not isinstance(content_sha, str) or not _SHA256_RE.fullmatch(content_sha):
        raise ChunkInventoryError("structural chunk document content SHA-256 is invalid")
    body_chars = _inventory_int(
        value.get("canonical_body_char_length"),
        field="canonical_body_char_length",
        minimum=1,
    )
    _inventory_int(
        value.get("canonical_body_utf8_length"),
        field="canonical_body_utf8_length",
        minimum=1,
    )
    boundaries = value.get("page_boundaries")
    reason = value.get("page_coordinate_reason")
    if not isinstance(boundaries, list):
        raise ChunkInventoryError("inventory page_boundaries must be an array")
    previous_end = 0
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping) or set(boundary) != {
            "page",
            "char_start",
            "char_end",
        }:
            raise ChunkInventoryError("inventory page boundary shape is invalid")
        page = _inventory_int(boundary["page"], field="page", minimum=1)
        start = _inventory_int(boundary["char_start"], field="page.char_start")
        end = _inventory_int(boundary["char_end"], field="page.char_end")
        if page != index + 1 or start < previous_end or end < start or end > body_chars:
            raise ChunkInventoryError("inventory page boundary range is invalid")
        previous_end = end
    expected_mapping_sha = _sha256(boundaries) if boundaries else None
    if value.get("page_boundary_mapping_sha256") != expected_mapping_sha:
        raise ChunkInventoryError("inventory page boundary mapping SHA-256 mismatch")
    if (boundaries and reason != "exact_pdf_text") or (
        not boundaries and reason != "source_not_paginated"
    ):
        raise ChunkInventoryError("inventory page coordinate reason is invalid")
    chunks = value.get("chunks")
    chunk_count = _inventory_int(value.get("chunk_count"), field="chunk_count", minimum=1)
    if not isinstance(chunks, list) or len(chunks) != chunk_count:
        raise ChunkInventoryError("inventory document chunk count mismatch")

    structure_keys = {
        "heading_path",
        "article_id",
        "article_label",
        "article_start",
        "clause",
        "clause_id",
        "subarticle",
        "chapter",
        "parent_id",
        "clause_ids",
        "subarticle_ids",
        "article_start_chunk_index",
        "parent_chunk_index",
        "chunker_revision",
    }
    chunk_keys = {
        "chunk_index",
        "document_chunk_count",
        "canonical_passage_sha256",
        "canonical_passage_char_length",
        "canonical_passage_utf8_length",
        "token_count",
        "char_start",
        "char_end",
        "structure",
        "page",
        "embed_input",
    }
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping) or set(chunk) != chunk_keys:
            raise ChunkInventoryError("inventory chunk row shape is invalid")
        if (
            _inventory_int(chunk["chunk_index"], field="chunk_index") != index
            or _inventory_int(
                chunk["document_chunk_count"],
                field="document_chunk_count",
                minimum=1,
            )
            != chunk_count
        ):
            raise ChunkInventoryError("inventory chunk sequence/count is invalid")
        passage_sha = chunk["canonical_passage_sha256"]
        if not isinstance(passage_sha, str) or not _SHA256_RE.fullmatch(passage_sha):
            raise ChunkInventoryError("inventory passage SHA-256 is invalid")
        passage_chars = _inventory_int(
            chunk["canonical_passage_char_length"],
            field="canonical_passage_char_length",
            minimum=1,
        )
        _inventory_int(
            chunk["canonical_passage_utf8_length"],
            field="canonical_passage_utf8_length",
            minimum=1,
        )
        _inventory_int(chunk["token_count"], field="token_count", minimum=1)
        start = _inventory_int(chunk["char_start"], field="char_start")
        end = _inventory_int(chunk["char_end"], field="char_end", minimum=1)
        if end <= start or end > body_chars or end - start != passage_chars:
            raise ChunkInventoryError("inventory passage character range is invalid")

        structure = chunk["structure"]
        if not isinstance(structure, Mapping) or set(structure) != structure_keys:
            raise ChunkInventoryError("inventory chunk structure shape is invalid")
        for field in (
            "article_id",
            "article_label",
            "clause",
            "clause_id",
            "subarticle",
            "chapter",
            "parent_id",
        ):
            _inventory_optional_string(structure[field], field=f"structure.{field}")
        _inventory_optional_int(structure["article_start"], field="structure.article_start")
        _inventory_optional_int(
            structure["article_start_chunk_index"],
            field="structure.article_start_chunk_index",
        )
        _inventory_optional_int(
            structure["parent_chunk_index"], field="structure.parent_chunk_index"
        )
        for field in ("heading_path", "clause_ids", "subarticle_ids"):
            items = structure[field]
            if not isinstance(items, list) or any(
                not isinstance(item, str) or not item for item in items
            ):
                raise ChunkInventoryError(f"inventory structure.{field} is invalid")
        if structure["chunker_revision"] != STRUCTURAL_CHUNKER_REVISION:
            raise ChunkInventoryError("inventory chunker revision mismatch")

        page = chunk["page"]
        if not isinstance(page, Mapping) or set(page) != {
            "page_start",
            "page_end",
            "page_coordinate_reason",
            "page_boundary_mapping_sha256",
        }:
            raise ChunkInventoryError("inventory chunk page shape is invalid")
        expected_pages = [
            boundary["page"]
            for boundary in boundaries
            if start < boundary["char_end"] and end > boundary["char_start"]
        ]
        expected_start = expected_pages[0] if expected_pages else None
        expected_end = expected_pages[-1] if expected_pages else None
        if page != {
            "page_start": expected_start,
            "page_end": expected_end,
            "page_coordinate_reason": reason,
            "page_boundary_mapping_sha256": expected_mapping_sha,
        }:
            raise ChunkInventoryError("inventory chunk page projection mismatch")

        embed_input = chunk["embed_input"]
        if not isinstance(embed_input, Mapping) or set(embed_input) != {
            "sha256",
            "char_length",
            "utf8_length",
            "token_count",
        }:
            raise ChunkInventoryError("inventory embed input shape is invalid")
        embed_sha = embed_input["sha256"]
        if not isinstance(embed_sha, str) or not _SHA256_RE.fullmatch(embed_sha):
            raise ChunkInventoryError("inventory embed input SHA-256 is invalid")
        _inventory_int(embed_input["char_length"], field="embed_input.char_length", minimum=1)
        _inventory_int(embed_input["utf8_length"], field="embed_input.utf8_length", minimum=1)
        _inventory_int(embed_input["token_count"], field="embed_input.token_count", minimum=1)
    return dict(value)


def iter_validated_inventory(
    path: str | Path,
    *,
    manifest_entry: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Stream exact document rows while independently validating the sealed artifact.

    Consumers must exhaust the iterator: the final digest, size, and aggregate count
    comparisons occur after the final yielded row.  Preparation stages inserts in a
    disposable SQLite database and always consumes the iterator completely.
    """

    entry = validate_manifest_entry(manifest_entry, expected_identity=expected_identity)
    if entry["status"] != CHUNK_INVENTORY_STATUS_AVAILABLE:
        raise ChunkInventoryError("cannot stream an unavailable structural chunk inventory")
    artifact = Path(path).expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(artifact, flags)
        info = os.fstat(descriptor)
    except OSError as exc:
        raise ChunkInventoryError(f"cannot open structural chunk inventory {artifact}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise ChunkInventoryError("structural chunk inventory is not a regular file")
    digest = hashlib.sha256()
    size_bytes = record_count = document_count = chunk_count = 0
    source_counts = {
        source: {"document_count": 0, "chunk_count": 0}
        for source in expected_identity["sources"]
    }
    source_positions = {
        source: index for index, source in enumerate(expected_identity["sources"])
    }
    last_source_position = -1
    seen_documents: set[tuple[str, str, str]] = set()
    header_seen = False
    try:
        with os.fdopen(descriptor, "rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                digest.update(raw)
                size_bytes += len(raw)
                record_count += 1
                if not raw or not raw.endswith(b"\n") or not raw.strip():
                    raise ChunkInventoryError(
                        f"structural chunk inventory line {line_number} is not canonical JSONL"
                    )
                try:
                    decoded = raw[:-1].decode("utf-8")
                except UnicodeError as exc:
                    raise ChunkInventoryError(
                        f"structural chunk inventory line {line_number} is not UTF-8"
                    ) from exc
                value = _strict_json(decoded, origin=f"{artifact}:{line_number}")
                if raw != _canonical_bytes(value) + b"\n":
                    raise ChunkInventoryError(
                        f"structural chunk inventory line {line_number} is not canonical"
                    )
                if line_number == 1:
                    expected_header = {
                        "schema_version": CHUNK_INVENTORY_SCHEMA_VERSION,
                        "kind": CHUNK_INVENTORY_KIND,
                        "format": CHUNK_INVENTORY_FORMAT,
                        "identity": dict(expected_identity),
                        "identity_sha256": _sha256(dict(expected_identity)),
                    }
                    if value != expected_header:
                        raise ChunkInventoryError("structural chunk inventory header mismatch")
                    header_seen = True
                    continue
                document = _validated_document_row(
                    value, expected_sources=expected_identity["sources"]
                )
                key = (
                    document["source"],
                    document["document_id"],
                    document["version_id"],
                )
                if key in seen_documents:
                    raise ChunkInventoryError(f"duplicate structural chunk document: {key!r}")
                seen_documents.add(key)
                source_position = source_positions[document["source"]]
                if source_position < last_source_position:
                    raise ChunkInventoryError("structural chunk inventory source order regressed")
                last_source_position = source_position
                document_count += 1
                chunk_count += document["chunk_count"]
                source_counts[document["source"]]["document_count"] += 1
                source_counts[document["source"]]["chunk_count"] += document[
                    "chunk_count"
                ]
                yield document
    except OSError as exc:
        raise ChunkInventoryError(f"cannot read structural chunk inventory: {exc}") from exc
    if not header_seen:
        raise ChunkInventoryError("structural chunk inventory is empty")
    observed = {
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
        "record_count": record_count,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "source_counts": source_counts,
    }
    mismatches = sorted(field for field, value in observed.items() if entry[field] != value)
    if mismatches or info.st_size != size_bytes:
        raise ChunkInventoryError(
            "structural chunk inventory artifact mismatch: "
            + ", ".join(mismatches or ["size_bytes"])
        )


def compare_recomputed_inventory(
    expected: Mapping[str, Any], observed: StructuralChunkInventory
) -> None:
    """Require every recomputed aggregate field to equal the sealed manifest entry."""

    if dict(expected) != observed.manifest_entry():
        fields = sorted(
            key
            for key in set(expected) | set(observed.manifest_entry())
            if expected.get(key) != observed.manifest_entry().get(key)
        )
        raise ChunkInventoryError(
            "recomputed structural chunk inventory mismatch: " + ", ".join(fields)
        )


__all__ = [
    "CHUNK_INVENTORY_FILENAME",
    "CHUNK_INVENTORY_SCHEMA_VERSION",
    "CHUNK_INVENTORY_STATUS_AVAILABLE",
    "CHUNK_INVENTORY_STATUS_UNAVAILABLE",
    "ChunkInventoryError",
    "StructuralChunkInventory",
    "chunk_inventory_identity",
    "compare_recomputed_inventory",
    "compute_structural_chunk_inventory",
    "iter_validated_inventory",
    "unavailable_manifest_entry",
    "validate_manifest_entry",
]
