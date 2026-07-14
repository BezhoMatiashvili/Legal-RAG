#!/usr/bin/env python
"""Merge a complete delta Qdrant collection into the main collection safely.

The full-corpus GPU embed produced the whole collection and loaded it with a snapshot
*restore* (which replaces the target). An INCREMENTAL embed — the sweep's ~23k newly
scraped docs — instead lands in a small ``georgian_legal_delta`` collection (embedded on
the RunPod GPU, snapshot-transferred back and restored locally under that name). This
script preflights document completeness, upserts the current points, then removes obsolete
high-index tail chunks from shortened documents in the main ``georgian_legal`` collection.

Point ids are deterministic UUIDv5 over ``(source, document_id, chunk_index)``
(``qdrant_store.point_id``), so the upsert is idempotent and additive: re-running never
duplicates, a re-embedded doc's chunks overwrite in place, and only genuinely new chunks
grow the collection. Dense (1024-d) + learned-sparse vectors and the full payload are
copied verbatim. Every current delta point must carry ``document_chunk_count``; legacy or
partial deltas fail closed and must be re-embedded before they can be merged.

Usage (from ingest/):
    .venv/bin/python scripts/merge_delta_collection.py \
        --src georgian_legal_delta_supremecourt_<run> \
        --run-manifest <out>/run_manifest.json --expected-source supremecourt
    .venv/bin/python scripts/merge_delta_collection.py --src georgian_legal_delta
    .venv/bin/python scripts/merge_delta_collection.py --dry-run   # self-test, no main mutation

The production CLI takes ``coordination/locks/qdrant-write.lock``, retains a hashed
pre-merge snapshot under ``.state/qdrant-rollbacks/<run>/``, restores it on any merge or
post-verification failure, and removes only an exact manifest-bound temporary collection.
Legacy ``--src`` remains supported but is audited exactly and is never auto-deleted.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import secrets
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import quote
from urllib.request import Request, urlopen

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # ingest/ root → `import ingest`

from qdrant_client import models  # noqa: E402

from ingest.config import load_config  # noqa: E402
from ingest.operational import refuse_legacy_operation  # noqa: E402
from ingest.qdrant_store import ensure_collection, make_client, point_id  # noqa: E402


INGEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INGEST_ROOT.parent
DEFAULT_LOCK_PATH = REPO_ROOT / "coordination" / "locks" / "qdrant-write.lock"
DEFAULT_ROLLBACK_ROOT = INGEST_ROOT / ".state" / "qdrant-rollbacks"
RUN_SCOPED_PREFIX = "georgian_legal_delta"
_RUN_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")


def _to_struct(point, *, payload: dict | None = None) -> models.PointStruct:
    """Rebuild an upsertable PointStruct from a scrolled Record (named dense[+sparse] vectors).

    A point may legitimately carry NO sparse vector: BGE-M3 yields empty sparse for some
    inputs, and ``_build_doc_points`` stores the ``sparse`` named vector only when it has
    indices. Guard the lookup so a dense-only point doesn't ``KeyError`` and abort the whole
    (non-resumable) merge. Points that DO carry sparse reconstruct byte-identically.
    """
    vec = point.vector
    out_vec = {"dense": vec["dense"]}
    sparse = vec.get("sparse") if isinstance(vec, dict) else None
    if sparse is not None:
        if not isinstance(sparse, models.SparseVector):
            sparse = models.SparseVector(
                indices=sparse["indices"], values=sparse["values"]
            )
        out_vec["sparse"] = sparse
    return models.PointStruct(
        id=point.id,
        vector=out_vec,
        payload=point.payload if payload is None else payload,
    )


@dataclass(frozen=True)
class _DocVersion:
    content_hash: str
    chunk_count: int


@dataclass(frozen=True)
class MergeExpectations:
    """Cryptographic identity/count gate anchored by the GPU run manifest."""

    source: str
    document_ids: frozenset[str]
    document_ids_sha256: str
    documents: int
    chunks: int
    collection: str | None = None
    run_id: str | None = None


@dataclass(frozen=True)
class DeltaAudit:
    manifests: dict[tuple[str, str], _DocVersion]
    document_ids: frozenset[str]
    points_count: int
    current_points: int


@dataclass(frozen=True)
class RollbackArtifact:
    run_id: str
    collection: str
    snapshot_name: str
    path: Path
    sha256: str
    size_bytes: int
    points_count: int
    manifest_path: Path


@dataclass(frozen=True)
class MergeOutcome:
    merged_points: int
    documents: int
    chunks: int
    before_points: int
    after_points: int
    rollback: RollbackArtifact
    preserved_source_documents: int
    source_deleted: bool


def _canonical_document_keys(
    source: str, document_ids: set[str] | frozenset[str]
) -> list[str]:
    return sorted(f"{source}\t{document_id}" for document_id in document_ids)


def _document_ids_sha256(source: str, document_ids: set[str] | frozenset[str]) -> str:
    blob = "\n".join(_canonical_document_keys(source, document_ids)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _positive_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"run manifest {field!r} must be a positive integer")
    return value


def _normalise_document_ids(
    values, source: str, *, require_canonical: bool
) -> frozenset[str]:
    if not isinstance(values, list) or not values:
        raise RuntimeError("expected document IDs must be a non-empty JSON list")
    document_ids: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"invalid expected document identity: {value!r}")
        if "\t" in value:
            item_source, document_id = value.split("\t", 1)
            if item_source != source:
                raise RuntimeError(
                    f"expected identity source {item_source!r} != {source!r}"
                )
        else:
            document_id = value
        if not document_id:
            raise RuntimeError(f"empty document_id in expected identity {value!r}")
        document_ids.append(document_id)
    if len(set(document_ids)) != len(document_ids):
        raise RuntimeError("expected document IDs contain duplicates")
    canonical = _canonical_document_keys(source, set(document_ids))
    qualified = [f"{source}\t{document_id}" for document_id in document_ids]
    if require_canonical and qualified != canonical:
        raise RuntimeError("run manifest document_ids must be sorted canonically")
    return frozenset(document_ids)


def _load_expected_ids(path: Path, source: str) -> frozenset[str]:
    if not path.is_file():
        raise RuntimeError(f"expected IDs file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(value, dict):
        value = value.get("document_ids")
    return _normalise_document_ids(value, source, require_canonical=False)


def load_merge_expectations(
    *,
    src: str,
    run_manifest_path: Path | None = None,
    expected_source: str | None = None,
    expected_ids_path: Path | None = None,
) -> MergeExpectations | None:
    """Load a strict source/identity/count gate, or retain legacy audit mode.

    A current run manifest carries the canonical IDs itself. An older manifest can be
    used only with ``expected_ids_path``; its committed digest must match that companion.
    """
    if (
        run_manifest_path is None
        and expected_source is None
        and expected_ids_path is None
    ):
        return None

    manifest_provided = run_manifest_path is not None
    manifest: dict = {}
    if manifest_provided:
        try:
            manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot read run manifest {run_manifest_path}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("run manifest must contain a JSON object")

    source = expected_source or manifest.get("source")
    if not isinstance(source, str) or not _RUN_COMPONENT.fullmatch(source):
        raise RuntimeError("a safe --expected-source is required for a guarded merge")
    if (
        expected_source is not None
        and manifest_provided
        and manifest.get("source") != expected_source
    ):
        raise RuntimeError(
            f"run manifest source {manifest.get('source')!r} != {expected_source!r}"
        )

    manifest_ids = None
    if "document_ids" in manifest:
        manifest_ids = _normalise_document_ids(
            manifest["document_ids"], source, require_canonical=True
        )
    explicit_ids = (
        _load_expected_ids(expected_ids_path, source)
        if expected_ids_path is not None
        else None
    )
    if (
        manifest_ids is not None
        and explicit_ids is not None
        and manifest_ids != explicit_ids
    ):
        raise RuntimeError("run manifest and --expected-ids disagree")
    document_ids = manifest_ids or explicit_ids
    if not document_ids:
        raise RuntimeError(
            "guarded merge requires run-manifest document_ids or --expected-ids"
        )

    digest = _document_ids_sha256(source, document_ids)
    committed_digest = (
        manifest.get("document_ids_sha256") if manifest_provided else digest
    )
    if committed_digest != digest:
        raise RuntimeError(
            "run manifest document_ids_sha256 does not match document_ids"
        )

    if manifest_provided:
        collection = manifest.get("collection")
        run_id = manifest.get("run_id")
        if collection != src:
            raise RuntimeError(
                f"run manifest collection {collection!r} != source collection {src!r}"
            )
        if not isinstance(run_id, str) or not _RUN_COMPONENT.fullmatch(run_id):
            raise RuntimeError("run manifest has an unsafe or missing run_id")
        expected_collection = f"{RUN_SCOPED_PREFIX}_{source}_{run_id}"
        if collection != expected_collection:
            raise RuntimeError(
                f"run-scoped collection must be {expected_collection!r}, got {collection!r}"
            )
        documents = _positive_int(manifest.get("documents"), "documents")
        expected_documents = _positive_int(
            manifest.get("expected_documents"), "expected_documents"
        )
        chunks = _positive_int(manifest.get("chunks"), "chunks")
        expected_chunks = _positive_int(
            manifest.get("expected_chunks"), "expected_chunks"
        )
        points_count = _positive_int(manifest.get("points_count"), "points_count")
        if manifest.get("skipped") != 0:
            raise RuntimeError("run manifest skipped must be exactly zero")
        if documents != expected_documents or documents != len(document_ids):
            raise RuntimeError("run manifest document counts do not match document_ids")
        if chunks != expected_chunks or chunks != points_count:
            raise RuntimeError("run manifest chunk/point counts disagree")
    else:
        collection = None
        run_id = None
        documents = len(document_ids)
        chunks = 0

    return MergeExpectations(
        source=source,
        document_ids=document_ids,
        document_ids_sha256=digest,
        documents=documents,
        chunks=chunks,
        collection=collection,
        run_id=run_id,
    )


def _point_metadata(point) -> tuple[tuple[str, str], int, _DocVersion | None]:
    """Return ``((source, document_id), chunk_index, version)`` for one delta point.

    Older stale tails may predate the completeness marker and therefore return ``None`` for
    ``version``. Chunk zero may never be legacy: it identifies the current document version.
    """
    payload = point.payload or {}
    source = payload.get("source")
    document_id = payload.get("document_id")
    chunk_index = payload.get("chunk_index")
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"delta point {point.id!r} has no source")
    if not isinstance(document_id, str) or not document_id:
        raise RuntimeError(f"delta point {point.id!r} has no document_id")
    if (
        isinstance(chunk_index, bool)
        or not isinstance(chunk_index, int)
        or chunk_index < 0
    ):
        raise RuntimeError(
            f"delta point {point.id!r} has invalid chunk_index={chunk_index!r}"
        )

    content_hash = payload.get("content_hash")
    chunk_count = payload.get("document_chunk_count")
    version = None
    if (
        isinstance(content_hash, str)
        and content_hash
        and isinstance(chunk_count, int)
        and not isinstance(chunk_count, bool)
        and chunk_count > 0
    ):
        version = _DocVersion(content_hash, chunk_count)
    return (source, document_id), chunk_index, version


def _audit_delta(
    client,
    src: str,
    *,
    batch_size: int,
    expectations: MergeExpectations | None = None,
    exact: bool = False,
) -> DeltaAudit:
    """Identify current docs and prove every current ``0..N-1`` chunk exists.

    ``exact`` is used by the production CLI: it additionally rejects stale/orphan points,
    duplicate logical chunks under alternate IDs, and any non-deterministic point ID. The
    lower-level legacy merge keeps its established stale-tail repair semantics.
    """
    indices: dict[tuple[str, str], dict[_DocVersion, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    current: dict[tuple[str, str], _DocVersion] = {}
    records: list[tuple[object, tuple[str, str], int, _DocVersion | None]] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=src,
            limit=batch_size,
            with_vectors=False,
            with_payload=True,
            offset=offset,
        )
        for point in points:
            key, chunk_index, version = _point_metadata(point)
            records.append((point, key, chunk_index, version))
            if chunk_index == 0:
                if version is None:
                    raise RuntimeError(
                        f"delta {src!r} is legacy/incomplete at {key[0]}:{key[1]} chunk 0; "
                        "re-embed it with document_chunk_count before merging"
                    )
                current[key] = version
            if version is not None:
                indices[key][version].add(chunk_index)
        if offset is None:
            break

    errors: list[str] = []
    for key in indices:
        if key not in current:
            errors.append(f"{key[0]}:{key[1]} has no current chunk 0")
    for key, version in current.items():
        actual = indices[key].get(version, set())
        expected = set(range(version.chunk_count))
        if actual != expected:
            missing = sorted(expected - actual)[:8]
            extra = sorted(actual - expected)[:8]
            errors.append(
                f"{key[0]}:{key[1]} expected 0..{version.chunk_count - 1}; "
                f"missing={missing} extra={extra}"
            )
    if errors:
        detail = "; ".join(errors[:10])
        raise RuntimeError(
            f"delta completeness preflight failed for {len(errors)} document(s): {detail}"
        )
    if not current:
        raise RuntimeError(f"delta collection {src!r} contains no complete documents")

    current_points = 0
    exact_errors: list[str] = []
    for point, key, chunk_index, version in records:
        current_version = current.get(key)
        is_current = (
            current_version is not None
            and version == current_version
            and chunk_index < current_version.chunk_count
        )
        if is_current:
            current_points += 1
        if exact:
            expected_point_id = point_id(key[0], key[1], chunk_index)
            if str(point.id) != expected_point_id:
                exact_errors.append(
                    f"point {point.id!r} != deterministic UUID5 {expected_point_id!r}"
                )
            if not is_current:
                exact_errors.append(
                    f"unexpected stale/orphan point {key[0]}:{key[1]} chunk {chunk_index}"
                )

    document_ids = frozenset(document_id for _, document_id in current)
    if expectations is not None:
        actual_sources = {source for source, _ in current}
        if actual_sources != {expectations.source}:
            exact_errors.append(
                f"delta sources {sorted(actual_sources)!r} != {[expectations.source]!r}"
            )
        actual_ids = frozenset(
            document_id
            for source, document_id in current
            if source == expectations.source
        )
        if actual_ids != expectations.document_ids:
            missing = sorted(expectations.document_ids - actual_ids)[:8]
            extra = sorted(actual_ids - expectations.document_ids)[:8]
            exact_errors.append(f"document IDs differ: missing={missing} extra={extra}")
        actual_digest = _document_ids_sha256(expectations.source, actual_ids)
        if actual_digest != expectations.document_ids_sha256:
            exact_errors.append("document_ids_sha256 differs from expected manifest")
        if len(current) != expectations.documents:
            exact_errors.append(
                f"documents={len(current)} != expected {expectations.documents}"
            )
        if expectations.chunks and current_points != expectations.chunks:
            exact_errors.append(
                f"current chunks={current_points} != expected {expectations.chunks}"
            )
        if expectations.chunks and len(records) != expectations.chunks:
            exact_errors.append(
                f"total points={len(records)} != expected {expectations.chunks}"
            )

    if exact and len(records) != current_points:
        exact_errors.append(
            f"delta contains {len(records)} points but only {current_points} are current"
        )
    expected_current_points = sum(version.chunk_count for version in current.values())
    if exact and current_points != expected_current_points:
        exact_errors.append(
            f"current logical chunks={current_points} != {expected_current_points}; "
            "duplicate logical points are forbidden"
        )
    if exact_errors:
        raise RuntimeError(
            "delta exact preflight failed: " + "; ".join(exact_errors[:12])
        )

    return DeltaAudit(
        manifests=current,
        document_ids=document_ids,
        points_count=len(records),
        current_points=current_points,
    )


def _scan_complete_docs(
    client, src: str, *, batch_size: int
) -> dict[tuple[str, str], _DocVersion]:
    """Compatibility wrapper retaining the production-audit stale-tail behavior."""
    return _audit_delta(client, src, batch_size=batch_size).manifests


def _delete_stale_tails(
    client,
    dst: str,
    manifests: dict[tuple[str, str], _DocVersion],
    *,
    batch_size: int = 128,
) -> None:
    """Delete obsolete destination chunks in bounded batches, after all upserts succeed."""
    items = list(manifests.items())
    for start in range(0, len(items), batch_size):
        should = []
        for (source, document_id), version in items[start : start + batch_size]:
            should.append(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source", match=models.MatchValue(value=source)
                        ),
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        ),
                        models.FieldCondition(
                            key="chunk_index",
                            range=models.Range(gte=version.chunk_count),
                        ),
                    ]
                )
            )
        client.delete(
            collection_name=dst,
            points_selector=models.FilterSelector(filter=models.Filter(should=should)),
            wait=True,
        )


def merge_collection(
    client,
    src: str,
    dst: str,
    *,
    batch_size: int = 256,
    expectations: MergeExpectations | None = None,
    audit: DeltaAudit | None = None,
    exact: bool = False,
) -> int:
    """Merge complete current docs and safely remove destination tails. Fail closed."""
    if src == dst:
        raise RuntimeError("source and destination collections must differ")
    audit = audit or _audit_delta(
        client,
        src,
        batch_size=batch_size,
        expectations=expectations,
        exact=exact,
    )
    manifests = audit.manifests
    offset = None
    total = 0
    scanned = 0
    ineligible: list[str] = []
    merged_indices: dict[tuple[str, str], set[int]] = defaultdict(set)
    while True:
        points, offset = client.scroll(
            collection_name=src,
            limit=batch_size,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        eligible = []
        for point in points:
            scanned += 1
            key, chunk_index, version = _point_metadata(point)
            if (
                version is not None
                and version == manifests.get(key)
                and chunk_index < version.chunk_count
            ):
                if exact:
                    expected_point_id = point_id(key[0], key[1], chunk_index)
                    if str(point.id) != expected_point_id:
                        ineligible.append(
                            f"{key[0]}:{key[1]} chunk {chunk_index} has ID {point.id!r}"
                        )
                        continue
                eligible.append(_to_struct(point))
                merged_indices[key].add(chunk_index)
            elif exact:
                ineligible.append(f"{key[0]}:{key[1]} chunk {chunk_index}")
        if eligible:
            client.upsert(collection_name=dst, points=eligible, wait=True)
            total += len(eligible)
        if offset is None:
            break

    incomplete = [
        f"{source}:{document_id}"
        for (source, document_id), version in manifests.items()
        if merged_indices.get((source, document_id), set())
        != set(range(version.chunk_count))
    ]
    if incomplete:
        raise RuntimeError(
            "delta changed during merge; refusing stale-tail deletion for incomplete docs: "
            + ", ".join(incomplete[:10])
        )
    if exact and (
        ineligible or scanned != audit.points_count or total != audit.current_points
    ):
        raise RuntimeError(
            "delta changed during exact merge: "
            f"scanned={scanned}/{audit.points_count} merged={total}/{audit.current_points} "
            f"ineligible={ineligible[:8]}"
        )

    # Deletion is deliberately last: a partial/failed upsert can be retried idempotently and
    # can never erase legitimate chunks from the prior complete destination version.
    _delete_stale_tails(client, dst, manifests)
    return total


def _source_document_ids(
    client, collection: str, source: str, *, batch_size: int
) -> frozenset[str]:
    """Return source document identities from chunk zero only."""
    found: set[str] = set()
    offset = None
    source_filter = models.Filter(
        must=[
            models.FieldCondition(key="source", match=models.MatchValue(value=source)),
            models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0)),
        ]
    )
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            with_vectors=False,
            with_payload=True,
            offset=offset,
            scroll_filter=source_filter,
        )
        for point in points:
            payload = point.payload or {}
            if payload.get("source") != source or payload.get("chunk_index") != 0:
                continue
            document_id = payload.get("document_id")
            if isinstance(document_id, str) and document_id:
                found.add(document_id)
        if offset is None:
            break
    return frozenset(found)


def verify_destination_coverage(
    client,
    dst: str,
    manifests: dict[tuple[str, str], _DocVersion],
    *,
    batch_size: int = 256,
) -> int:
    """Require exact current chunks for every merged document in the destination."""
    by_source: dict[str, list[str]] = defaultdict(list)
    for source, document_id in manifests:
        by_source[source].append(document_id)

    found: dict[tuple[str, str], set[int]] = defaultdict(set)
    errors: list[str] = []
    total = 0
    for source, document_ids in by_source.items():
        ordered = sorted(document_ids)
        for start in range(0, len(ordered), 128):
            subset = ordered[start : start + 128]
            target_keys = {(source, document_id) for document_id in subset}
            target_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source", match=models.MatchValue(value=source)
                    ),
                    models.FieldCondition(
                        key="document_id", match=models.MatchAny(any=subset)
                    ),
                ]
            )
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=dst,
                    limit=batch_size,
                    with_vectors=False,
                    with_payload=True,
                    offset=offset,
                    scroll_filter=target_filter,
                )
                for point in points:
                    payload = point.payload or {}
                    raw_key = (payload.get("source"), payload.get("document_id"))
                    if raw_key not in target_keys:
                        continue
                    key, chunk_index, version = _point_metadata(point)
                    expected_version = manifests[key]
                    expected_id = point_id(key[0], key[1], chunk_index)
                    if str(point.id) != expected_id:
                        errors.append(
                            f"{key[0]}:{key[1]} chunk {chunk_index} has non-UUID5 ID"
                        )
                        continue
                    if (
                        version != expected_version
                        or chunk_index >= expected_version.chunk_count
                    ):
                        errors.append(
                            f"{key[0]}:{key[1]} chunk {chunk_index} has stale/wrong version"
                        )
                        continue
                    found[key].add(chunk_index)
                    total += 1
                if offset is None:
                    break

    for key, version in manifests.items():
        expected = set(range(version.chunk_count))
        actual = found.get(key, set())
        if actual != expected:
            errors.append(
                f"{key[0]}:{key[1]} destination chunks differ: "
                f"missing={sorted(expected - actual)[:8]} extra={sorted(actual - expected)[:8]}"
            )
    expected_total = sum(version.chunk_count for version in manifests.values())
    if total != expected_total:
        errors.append(f"destination chunks={total} != expected {expected_total}")
    if errors:
        raise RuntimeError("post-merge coverage failed: " + "; ".join(errors[:12]))
    return total


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


@contextmanager
def _qdrant_snapshot_timeout(client, seconds: float = 7200) -> Iterator[None]:
    """Temporarily extend the generated client's snapshot create/upload timeout."""
    snapshots_api = getattr(getattr(client, "http", None), "snapshots_api", None)
    api_client = getattr(snapshots_api, "api_client", None)
    http_client = getattr(api_client, "_client", None)
    previous = getattr(http_client, "timeout", None)
    if previous is not None:
        http_client.timeout = previous.__class__(seconds)
    try:
        yield
    finally:
        if previous is not None:
            http_client.timeout = previous


@contextmanager
def qdrant_write_lock(path: Path, *, owner: str, run_id: str) -> Iterator[Path]:
    """Acquire the repository coordination lock atomically and release only our token."""
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(16)
    payload = (
        f"owner: {owner}\n"
        "why: guarded delta merge with rollback snapshot\n"
        f"since: {datetime.now(UTC).isoformat()}\n"
        "expected: <2h\n"
        f"run: {run_id}\n"
        f"token: {token}\n"
    )
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        try:
            holder = path.read_text(encoding="utf-8").strip()
        except OSError:
            holder = "unreadable existing lock"
        raise RuntimeError(
            f"qdrant write lock already held at {path}: {holder}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        yield path
    finally:
        try:
            current = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            current = ""
        if f"token: {token}\n" in current:
            path.unlink(missing_ok=True)


def download_qdrant_snapshot(
    qdrant_url: str,
    api_key: str | None,
    collection: str,
    snapshot_name: str,
    destination: Path,
) -> None:
    """Atomically stream one Qdrant snapshot out of the service-owned snapshot area."""
    url = (
        f"{qdrant_url.rstrip('/')}/collections/{quote(collection, safe='')}"
        f"/snapshots/{quote(snapshot_name, safe='')}"
    )
    headers = {"api-key": api_key} if api_key else {}
    request = Request(url, headers=headers)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with urlopen(request, timeout=7200) as response, temporary.open("xb") as handle:  # noqa: S310
            while block := response.read(16 * 1024 * 1024):
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def create_rollback_snapshot(
    client,
    dst: str,
    run_id: str,
    rollback_root: Path,
    *,
    snapshot_copy: Callable[[str, str, Path], None],
) -> RollbackArtifact:
    """Create, copy, hash, and durably describe a main-collection rollback snapshot."""
    if not _RUN_COMPONENT.fullmatch(run_id):
        raise RuntimeError(f"unsafe rollback run_id: {run_id!r}")
    run_root = rollback_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    attempt = (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')}_"
        f"{os.getpid()}_{secrets.token_hex(4)}"
    )
    run_dir = run_root / attempt
    run_dir.mkdir(mode=0o700)

    points_count = _count(client, dst)
    with _qdrant_snapshot_timeout(client):
        description = client.create_snapshot(collection_name=dst, wait=True)
    snapshot_name = getattr(description, "name", None)
    if (
        not isinstance(snapshot_name, str)
        or not snapshot_name
        or Path(snapshot_name).name != snapshot_name
    ):
        raise RuntimeError(
            f"Qdrant returned an unsafe snapshot name: {snapshot_name!r}"
        )
    target = run_dir / snapshot_name
    snapshot_copy(dst, snapshot_name, target)
    if not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError(f"rollback snapshot copy is missing/empty: {target}")
    size_bytes = target.stat().st_size
    expected_size = getattr(description, "size", None)
    if (
        isinstance(expected_size, int)
        and expected_size > 0
        and expected_size != size_bytes
    ):
        raise RuntimeError(
            f"rollback snapshot size {size_bytes} != Qdrant descriptor {expected_size}"
        )
    digest = _sha256_path(target)
    expected_digest = getattr(description, "checksum", None)
    if expected_digest and expected_digest != digest:
        raise RuntimeError("rollback snapshot SHA-256 differs from Qdrant descriptor")

    manifest_path = run_dir / "rollback_manifest.json"
    _write_json_atomic(
        manifest_path,
        {
            "schema_version": 1,
            "kind": "qdrant_main_pre_merge_rollback",
            "run_id": run_id,
            "attempt_id": attempt,
            "collection": dst,
            "snapshot": snapshot_name,
            "sha256": digest,
            "size_bytes": size_bytes,
            "points_count": points_count,
            "created_at": datetime.now(UTC).isoformat(),
            "retention": "keep outside publish cleanup until corpus publication is verified",
        },
    )
    return RollbackArtifact(
        run_id=run_id,
        collection=dst,
        snapshot_name=snapshot_name,
        path=target,
        sha256=digest,
        size_bytes=size_bytes,
        points_count=points_count,
        manifest_path=manifest_path,
    )


def restore_rollback_snapshot(client, artifact: RollbackArtifact) -> None:
    """Upload the durable rollback copy and require its exact pre-merge point count."""
    if artifact.path.stat().st_size != artifact.size_bytes:
        raise RuntimeError("rollback snapshot size changed before recovery")
    if _sha256_path(artifact.path) != artifact.sha256:
        raise RuntimeError("rollback snapshot SHA-256 changed before recovery")
    with _qdrant_snapshot_timeout(client), artifact.path.open("rb") as snapshot:
        client.http.snapshots_api.recover_from_uploaded_snapshot(
            collection_name=artifact.collection,
            wait=True,
            priority=models.SnapshotPriority.SNAPSHOT,
            checksum=artifact.sha256,
            snapshot=snapshot,
        )
    restored = _count(client, artifact.collection)
    if restored != artifact.points_count:
        raise RuntimeError(
            f"rollback restored {restored} points, expected {artifact.points_count}"
        )


def _run_scoped_temp_collection(
    expectations: MergeExpectations | None, src: str
) -> bool:
    if expectations is None or expectations.run_id is None:
        return False
    expected = f"{RUN_SCOPED_PREFIX}_{expectations.source}_{expectations.run_id}"
    return src == expectations.collection == expected


def run_merge_workflow(
    client,
    src: str,
    dst: str,
    *,
    run_id: str,
    rollback_root: Path,
    lock_path: Path,
    snapshot_copy: Callable[[str, str, Path], None],
    expectations: MergeExpectations | None = None,
    batch_size: int = 256,
    delete_source: bool = False,
    restore_snapshot: Callable[
        [object, RollbackArtifact], None
    ] = restore_rollback_snapshot,
) -> MergeOutcome:
    """Perform the only mutating production merge path under lock and rollback."""
    if src == dst:
        raise RuntimeError("source and destination collections must differ")
    if delete_source and not _run_scoped_temp_collection(expectations, src):
        raise RuntimeError(
            "source deletion requires the exact manifest-bound run-scoped collection"
        )
    owner = f"merge-delta[{os.getpid()}]"
    with qdrant_write_lock(lock_path, owner=owner, run_id=run_id):
        audit = _audit_delta(
            client,
            src,
            batch_size=batch_size,
            expectations=expectations,
            exact=True,
        )
        preserved = _source_document_ids(
            client, dst, "supremecourt", batch_size=batch_size
        )
        rollback = create_rollback_snapshot(
            client,
            dst,
            run_id,
            rollback_root,
            snapshot_copy=snapshot_copy,
        )
        try:
            merged = merge_collection(
                client,
                src,
                dst,
                batch_size=batch_size,
                expectations=expectations,
                audit=audit,
                exact=True,
            )
            verified_chunks = verify_destination_coverage(
                client, dst, audit.manifests, batch_size=batch_size
            )
            if verified_chunks != audit.current_points:
                raise RuntimeError(
                    f"post-merge chunks {verified_chunks} != audited {audit.current_points}"
                )
            after_supreme = _source_document_ids(
                client, dst, "supremecourt", batch_size=batch_size
            )
            missing_preserved = preserved - after_supreme
            if missing_preserved:
                raise RuntimeError(
                    "post-merge lost existing Supreme Court documents: "
                    + ", ".join(sorted(missing_preserved)[:10])
                )
            after_points = _count(client, dst)
        except BaseException as merge_error:
            try:
                restore_snapshot(client, rollback)
            except BaseException as rollback_error:
                raise RuntimeError(
                    f"merge failed ({merge_error!r}) and rollback failed ({rollback_error!r}); "
                    f"durable snapshot remains at {rollback.path}"
                ) from rollback_error
            raise

        source_deleted = False
        if delete_source:
            # Deliberately outside the rollback block: cleanup failure leaves a verified main
            # collection and a retryable temporary source; it must not undo a valid merge.
            client.delete_collection(src)
            source_deleted = True
        return MergeOutcome(
            merged_points=merged,
            documents=len(audit.manifests),
            chunks=audit.current_points,
            before_points=rollback.points_count,
            after_points=after_points,
            rollback=rollback,
            preserved_source_documents=len(preserved),
            source_deleted=source_deleted,
        )


def _count(client, name: str) -> int:
    return client.count(collection_name=name, exact=True).count


def dry_run(cfg, client) -> None:
    """Prove the merge on real embedded data (the firearms 495 chunks) — main is untouched."""
    refuse_legacy_operation("Qdrant-writing merge dry-run")
    src, dst = "_merge_test_src", "_merge_test_dst"
    for name in (src, dst):
        ensure_collection(
            client, dataclasses.replace(cfg, collection_name=name), recreate=True
        )

    # Seed src with the firearms points copied out of the main collection (real vectors+payload).
    # The live base predates ``document_chunk_count``, so infer and stamp it on this complete
    # sample before exercising the same fail-closed merge path used for new delta collections.
    sample_points, offset = [], None
    while True:
        pts, offset = client.scroll(
            collection_name=cfg.collection_name,
            limit=256,
            with_vectors=True,
            with_payload=True,
            offset=offset,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchAny(any=["18070", "14944"])
                    )
                ]
            ),
        )
        sample_points.extend(pts)
        if offset is None:
            break
    by_doc: dict[tuple[str, str], list] = defaultdict(list)
    for point in sample_points:
        payload = point.payload or {}
        by_doc[(payload.get("source"), payload.get("document_id"))].append(point)
    seeded = 0
    for key, doc_points in by_doc.items():
        indices = {int((point.payload or {})["chunk_index"]) for point in doc_points}
        chunk_count = max(indices, default=-1) + 1
        if indices != set(range(chunk_count)):
            raise RuntimeError(
                f"dry-run source sample {key} is not contiguous: {sorted(indices)}"
            )
        stamped = [
            _to_struct(
                point,
                payload={**(point.payload or {}), "document_chunk_count": chunk_count},
            )
            for point in doc_points
        ]
        client.upsert(collection_name=src, points=stamped, wait=True)
        seeded += len(stamped)
    print(f"  seeded src with {seeded} firearms points")

    merged = merge_collection(client, src, dst)
    after_first = _count(client, dst)
    merge_collection(client, src, dst)  # idempotency: a second merge must not grow dst
    after_second = _count(client, dst)

    sample, _ = client.scroll(
        collection_name=dst,
        limit=1,
        with_vectors=True,
        with_payload=True,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value="14944")
                )
            ]
        ),
    )
    dense_ok = bool(sample) and len(sample[0].vector["dense"]) == cfg.dense_dim
    sparse_ok = bool(sample) and len(sample[0].vector["sparse"].indices) > 0
    payload_ok = bool(sample) and sample[0].payload.get("is_consolidated") is True

    print(
        f"  merged={merged} · dst after 1st={after_first} · after 2nd={after_second} "
        f"· idempotent={after_first == after_second}"
    )
    print(
        f"  round-trip sample: dense_dim={len(sample[0].vector['dense']) if sample else None} "
        f"· sparse_terms={len(sample[0].vector['sparse'].indices) if sample else None} "
        f"· is_consolidated={sample[0].payload.get('is_consolidated') if sample else None}"
    )

    for name in (src, dst):
        client.delete_collection(name)

    if not (
        seeded == after_first == after_second and dense_ok and sparse_ok and payload_ok
    ):
        raise SystemExit("❌ DRY-RUN FAILED — see counts above")
    print(
        f"✅ DRY-RUN PASSED — {seeded} points merged idempotently with dense+sparse+payload "
        f"intact; temp collections dropped, main untouched."
    )


def main() -> None:
    refuse_legacy_operation("direct delta-to-serving collection merge")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src", help="delta collection to merge from (e.g. georgian_legal_delta)"
    )
    ap.add_argument("--dst", help="target collection (default: config collection_name)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument(
        "--run-manifest",
        type=Path,
        help="validated GPU run_manifest.json (enables exact gate and temp cleanup)",
    )
    ap.add_argument(
        "--expected-source",
        help="single allowed source (must agree with --run-manifest when both are set)",
    )
    ap.add_argument(
        "--expected-ids",
        type=Path,
        help="JSON/text expected IDs; companion for an older manifest without document_ids",
    )
    ap.add_argument(
        "--run-id",
        help="rollback run scope (defaults to manifest run_id or a generated merge ID)",
    )
    ap.add_argument(
        "--rollback-root",
        type=Path,
        default=DEFAULT_ROLLBACK_ROOT,
        help=f"durable rollback directory (default: {DEFAULT_ROLLBACK_ROOT})",
    )
    ap.add_argument(
        "--lock-path",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help=f"coordination write lock (default: {DEFAULT_LOCK_PATH})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="self-test the merge on the firearms sample without touching main",
    )
    args = ap.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    cfg = load_config()
    client = make_client(cfg)

    if args.dry_run:
        dry_run_id = args.run_id or (
            f"dry_run_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
        )
        if not _RUN_COMPONENT.fullmatch(dry_run_id):
            raise SystemExit(f"unsafe --run-id: {dry_run_id!r}")
        with qdrant_write_lock(
            args.lock_path,
            owner=f"merge-delta-dry-run[{os.getpid()}]",
            run_id=dry_run_id,
        ):
            dry_run(cfg, client)
        return

    if not args.src:
        raise SystemExit("--src is required (or use --dry-run)")
    dst = args.dst or cfg.collection_name
    expectations = load_merge_expectations(
        src=args.src,
        run_manifest_path=args.run_manifest,
        expected_source=args.expected_source,
        expected_ids_path=args.expected_ids,
    )
    manifest_run_id = expectations.run_id if expectations is not None else None
    if args.run_id and manifest_run_id and args.run_id != manifest_run_id:
        raise SystemExit(
            f"--run-id {args.run_id!r} != run manifest {manifest_run_id!r}"
        )
    run_id = (
        manifest_run_id
        or args.run_id
        or f"merge_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    )
    if not _RUN_COMPONENT.fullmatch(run_id):
        raise SystemExit(f"unsafe --run-id: {run_id!r}")

    def copy_snapshot(collection: str, snapshot_name: str, destination: Path) -> None:
        download_qdrant_snapshot(
            cfg.qdrant_url,
            cfg.qdrant_api_key,
            collection,
            snapshot_name,
            destination,
        )

    outcome = run_merge_workflow(
        client,
        args.src,
        dst,
        run_id=run_id,
        rollback_root=args.rollback_root,
        lock_path=args.lock_path,
        snapshot_copy=copy_snapshot,
        expectations=expectations,
        batch_size=args.batch_size,
        delete_source=_run_scoped_temp_collection(expectations, args.src),
    )
    print(
        f"Merged {outcome.merged_points} points / {outcome.documents} documents "
        f"from {args.src!r} into {dst!r}: {outcome.before_points} → "
        f"{outcome.after_points} "
        f"(+{outcome.after_points - outcome.before_points} net new)."
    )
    print(
        f"Rollback retained: {outcome.rollback.path} "
        f"sha256={outcome.rollback.sha256}; source_deleted={outcome.source_deleted}"
    )


if __name__ == "__main__":
    main()
