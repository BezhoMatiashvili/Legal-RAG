"""Derive generation candidate inputs (documents/samples) from an ALREADY-INDEXED, live
Qdrant collection — read-only, no Qdrant writes.

Nothing in this repo previously built these generation inputs from a real corpus:
``ingest/scripts/create_generation.py`` only *consumes* pre-built ``documents.jsonl`` /
``sample_checks.jsonl`` files. This module is that missing producer for the collection
that predates the generation system entirely (every live point lacks a ``generation_id``
and the other identity payload fields ``collection_compatibility.py`` requires). It
reconstructs each document's :class:`~ingest.generation.DocumentRecord` from what is
already stored in Qdrant payloads, reusing the SAME legacy-payload reconstruction
(:func:`ingest.pipeline._document_state_hash`) the watcher already trusts for "did this
doc change since it was indexed".

Deliberately out of scope: the :class:`~ingest.generation.GenerationManifest`'s
``model.*`` revision fields and ``dependency.lock_sha256``/``image_digest``. Those need
real external evidence (a pinned Hugging Face revision, a hash-locked requirements file)
that does not exist in this repo yet (``ingest/serverless/runtime-identity.unconfigured.json``
lists exactly what's missing). Guessing them would violate this repo's own supply-chain
rule ("do not guess model commits, image digests..."), so building a full
``GenerationManifest`` from a scan's output additionally requires that evidence supplied
by the caller — this module only produces the parts that are honestly derivable today.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from . import qdrant_store as store
from .config import Config
from .generation import GENERATION_SCHEMA_VERSION, validate_generation_id
from .pipeline import _document_state_hash

# Every point in the live corpus predates the content_kind/extraction_status/
# content_complete fields (added for the TB Appeals PDF-vs-summary distinction) — the
# whole corpus was, at index time, "the full body we scraped, indexed as-is". Defaulting
# a payload that lacks these fields to "complete full text" describes that history
# honestly; it is not a guess about an unknown, it's the true prior state of this corpus.
_LEGACY_CONTENT_KIND = "full_text"
_LEGACY_EXTRACTION_STATUS = "full_text"


@dataclass(frozen=True)
class ScanIssue:
    """A per-document anomaly found while scanning — surfaced, never silently dropped."""

    source: str
    document_id: str
    reason: str


@dataclass
class _DocAccumulator:
    chunk0_payload: Mapping | None = None
    fallback_payload: Mapping | None = None  # any chunk, used if chunk 0 is itself missing
    seen_count: int = 0
    max_index: int = -1


@dataclass(frozen=True)
class ScanResult:
    documents: list[dict]
    samples: list[dict]
    issues: tuple[ScanIssue, ...] = field(default_factory=tuple)
    chunk_count: int = 0


def _source_identity(source: str, document_id: str, content_hash: str) -> str:
    """Sha256 identity of this document's indexed state (source/id/body identity).

    Distinct from ``content_hash`` (the cleaned-body hash alone): this additionally binds
    the (source, document_id) key, matching the shape ``document.source_identity`` expects
    without depending on raw scrape artifacts that may not exist for very old indexed runs.
    """
    material = json.dumps(
        {"source": source, "document_id": document_id, "content_hash": content_hash},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _document_record_from_accumulator(
    *, generation_id: str, source: str, document_id: str, acc: _DocAccumulator, cfg: Config,
) -> tuple[dict, list[ScanIssue]]:
    issues: list[ScanIssue] = []
    expected_chunk_count = acc.max_index + 1
    if acc.seen_count != expected_chunk_count:
        issues.append(
            ScanIssue(
                source,
                document_id,
                f"non-contiguous or duplicate chunk_index: saw {acc.seen_count} points "
                f"but max chunk_index was {acc.max_index} (expected exactly "
                f"{expected_chunk_count} contiguous 0..{acc.max_index})",
            )
        )
    payload = acc.chunk0_payload or acc.fallback_payload or {}
    if acc.chunk0_payload is None:
        issues.append(ScanIssue(source, document_id, "chunk_index 0 missing; used another chunk's payload"))

    content_hash = payload.get("content_hash")
    if not content_hash:
        issues.append(ScanIssue(source, document_id, "payload missing content_hash"))
        content_hash = hashlib.sha256(b"").hexdigest()

    document_state_hash = payload.get("document_state_hash")
    if not document_state_hash:
        try:
            document_state_hash = _document_state_hash(cfg, payload=payload)
        except Exception as exc:  # noqa: BLE001 - reconstruction must never crash the scan
            issues.append(ScanIssue(source, document_id, f"document_state_hash reconstruction failed: {exc}"))
            document_state_hash = hashlib.sha256(b"").hexdigest()

    content_kind = payload.get("content_kind") or _LEGACY_CONTENT_KIND
    extraction_status = payload.get("extraction_status") or _LEGACY_EXTRACTION_STATUS
    content_complete = payload.get("content_complete")
    if content_complete is None:
        content_complete = extraction_status == "full_text"

    record = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": generation_id,
        "source": source,
        "document_id": document_id,
        "source_identity": _source_identity(source, document_id, content_hash),
        "content_hash": content_hash,
        "document_state_hash": document_state_hash,
        "expected_chunk_count": expected_chunk_count if expected_chunk_count >= 1 else 1,
        "content_kind": content_kind,
        "content_complete": bool(content_complete),
        "extraction_status": extraction_status,
        "article_summary": payload.get("article_summary"),
        "exclusion_reason": None,
        "refresh_deadline": None,
        "source_binary_url": payload.get("source_binary_url"),
    }
    return record, issues


def _sample_check(*, generation_id: str, source: str, document_id: str, payload: Mapping) -> dict:
    text = payload.get("text") or ""
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": generation_id,
        "source": source,
        "document_id": document_id,
        "chunk_index": 0,
        "point_id": store.point_id(source, document_id, 0),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def aggregate_documents(
    points: Iterable[Mapping], *, generation_id: str, cfg: Config,
) -> ScanResult:
    """Group flat per-chunk Qdrant payloads into per-document generation records.

    ``points`` is any iterable of chunk payload dicts (``source``/``document_id``/
    ``chunk_index`` plus the usual payload fields) in ANY order — a real Qdrant ``scroll``
    makes no ordering guarantee. Memory cost is O(documents), not O(chunks): only chunk 0's
    payload (needed for content/state-hash reconstruction) is retained per document, not
    every chunk's payload — the same "one full payload scroll in memory" budget
    ``scripts/verify_all_embedded.py`` already uses safely at this corpus's scale.

    A malformed/missing point (no source, document_id, or chunk_index) is skipped, not
    fatal — mirrors ``verify_all_embedded.embedded_universe``'s tolerance for the same.
    """
    validate_generation_id(generation_id)
    by_doc: dict[tuple[str, str], _DocAccumulator] = {}
    total_chunks = 0
    for payload in points:
        source = payload.get("source")
        document_id = payload.get("document_id")
        chunk_index = payload.get("chunk_index")
        if source is None or document_id is None or chunk_index is None:
            continue
        key = (str(source), str(document_id))
        acc = by_doc.setdefault(key, _DocAccumulator())
        acc.seen_count += 1
        idx = int(chunk_index)
        acc.max_index = max(acc.max_index, idx)
        if idx == 0:
            acc.chunk0_payload = payload
        elif acc.fallback_payload is None:
            acc.fallback_payload = payload
        total_chunks += 1

    documents: list[dict] = []
    samples: list[dict] = []
    issues: list[ScanIssue] = []
    for (source, document_id), acc in by_doc.items():
        record, doc_issues = _document_record_from_accumulator(
            generation_id=generation_id, source=source, document_id=document_id, acc=acc, cfg=cfg,
        )
        documents.append(record)
        issues.extend(doc_issues)
        sample_payload = acc.chunk0_payload or acc.fallback_payload or {}
        samples.append(
            _sample_check(
                generation_id=generation_id, source=source, document_id=document_id,
                payload=sample_payload,
            )
        )
    return ScanResult(documents=documents, samples=samples, issues=tuple(issues), chunk_count=total_chunks)
