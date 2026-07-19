"""Vector-preserving rematerialization into isolated Qdrant collections.

This module deliberately never embeds text.  It proves that a canonical snapshot still
chunks to the passages represented by an immutable/frozen legacy vector source, then
copies the existing dense and learned-sparse values under schema-v2 point identities while
rebuilding payloads through :func:`ingest.qdrant_store.build_payload`.

Two entry points are exposed:

``rematerialize_scratch``
    Read a serving collection and write only a run-scoped delta.  This is validation
    evidence, never production provenance.

``rematerialize_generation``
    Restore a hash-bound legacy Qdrant snapshot into run-scoped staging and create an
    absent immutable physical generation.  The source-evidence format is intentionally
    strict; incomplete historical model identity is a refusal, not something inferred
    from the current Hugging Face cache.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import struct
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qdrant_client import models

from .chunking import build_embed_text, chunk_document
from .config import Config, retrieval_fingerprint_sha256
from .court_extract import EXTRACTOR_REVISION
from .embed_job import (
    binding_path as embed_binding_path,
    iter_snapshot_docs,
    load_checksum_reference,
    prepare_binding,
    snapshot_doc_to_canonical,
    verify_snapshot_docs,
)
from .operational import (
    QDRANT_WRITE_APPROVAL_ENV,
    require_explicit_approval,
    require_run_scoped_delta_collection,
)
from .pipeline import _document_state_hash, _header_v2_kwargs, _prepare_doc_for_index
from .promotion import physical_collection_name
from .qdrant_store import (
    build_payload,
    collection_configuration,
    collection_configuration_sha256,
    ensure_collection,
    point_id,
    refuse_aliased_write_target,
)
from .release_inputs import (
    GENERATION_ID as FROZEN_CANDIDATE_GENERATION_ID,
    PHYSICAL_COLLECTION as FROZEN_CANDIDATE_PHYSICAL_COLLECTION,
)
from .snapshot import SOURCES_PRESENT, verify_sealed_snapshot
from .sources import COURT_CANONICAL_FIELDS, derived_version_id, normalize


REMATERIALIZATION_SCHEMA_VERSION = 1
SOURCE_EVIDENCE_KIND = "legacy-vector-source-evidence"
BINDING_KIND = "vector-preserving-rematerialization-binding"
REPORT_KIND = "vector-preserving-rematerialization-report"
LOGICAL_VECTOR_DIGEST_REVISION = "logical-vector-v1"
SCRATCH_COPY_STRATEGY = "source-payload-court-overlay-v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_COURT_SOURCES = frozenset({"ecd", "supremecourt"})
_SCRATCH_PROMOTED_FIELDS = ("result", "appeal_type")
_SOURCE_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "snapshot_path",
        "snapshot_sha256",
        "source_collection",
        "source_points_count",
        "source_configuration_sha256",
        "legacy_id_scheme",
        "embedding_model",
        "embedding_revision",
        "tokenizer_model",
        "tokenizer_revision",
        "reranker_model",
        "reranker_revision",
        "dense_name",
        "dense_dimension",
        "sparse_name",
        "chunk_tokens",
        "chunk_overlap",
        "chunk_min_tokens",
        "document_header",
        "retrieval_fingerprint",
        "vector_checksum_artifact_sha256",
        "vector_probe_sha256",
    }
)


class RematerializationError(RuntimeError):
    """The requested copy cannot be proven vector- and corpus-preserving."""


def refuse_frozen_candidate_rematerialization(generation_id: str) -> None:
    """Reserve the exact 512-token candidate for its reviewed fresh-embed workflow."""

    if (
        generation_id == FROZEN_CANDIDATE_GENERATION_ID
        and physical_collection_name(generation_id)
        == FROZEN_CANDIDATE_PHYSICAL_COLLECTION
    ):
        raise RematerializationError(
            "the frozen 512-token candidate cannot be rematerialized; "
            "use only its reviewed immutable GPU embed workflow"
        )


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    snapshot_path: Path
    snapshot_sha256: str
    source_collection: str
    source_points_count: int
    source_configuration_sha256: str
    embedding_model: str
    embedding_revision: str
    tokenizer_model: str
    tokenizer_revision: str
    reranker_model: str
    reranker_revision: str
    dense_name: str
    dense_dimension: int
    sparse_name: str
    chunk_tokens: int
    chunk_overlap: int
    chunk_min_tokens: int
    document_header: bool
    retrieval_fingerprint: str
    vector_checksum_artifact_sha256: str
    vector_probe_sha256: str
    canonical: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RematerializationResult:
    target_collection: str
    report_path: Path
    point_count: int
    document_count: int
    source_logical_vector_sha256: str
    target_logical_vector_sha256: str


@dataclass(frozen=True, slots=True)
class _ScratchOverlay:
    source: str
    document_id: str
    version_id: str
    content_hash: str
    source_fingerprint: str
    payload: Mapping[str, Any]


def _canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_absolute_file(path: Path, *, label: str) -> Path:
    candidate = path.expanduser().absolute()
    try:
        mode = candidate.lstat().st_mode
    except OSError as exc:
        raise RematerializationError(f"cannot stat {label}: {exc}") from exc
    if not candidate.is_absolute() or stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RematerializationError(
            f"{label} must be an absolute regular non-symlink file"
        )
    return candidate


def _strict_json_file(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    path = _regular_absolute_file(path, label=label)
    raw = path.read_bytes()

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise RematerializationError(f"{label} has duplicate key {key!r}")
            out[key] = value
        return out

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicate,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RematerializationError(f"{label} contains non-finite {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RematerializationError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise RematerializationError(f"{label} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _require_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise RematerializationError(f"{field} must be a lowercase SHA-256")
    return value


def _require_revision(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
        raise RematerializationError(f"{field} must be an immutable hexadecimal revision")
    return value


def load_source_evidence(path: Path) -> tuple[SourceEvidence, str]:
    """Load and fully validate the create-only production source binding."""

    value, file_sha = _strict_json_file(path, label="source evidence")
    if set(value) != _SOURCE_EVIDENCE_KEYS:
        missing = sorted(_SOURCE_EVIDENCE_KEYS - set(value))
        unknown = sorted(set(value) - _SOURCE_EVIDENCE_KEYS)
        raise RematerializationError(
            f"source evidence keys are invalid; missing={missing}, unknown={unknown}"
        )
    if (
        value["schema_version"] != REMATERIALIZATION_SCHEMA_VERSION
        or value["kind"] != SOURCE_EVIDENCE_KIND
        or value["legacy_id_scheme"] != "uuid5-source-document-chunk-v1"
    ):
        raise RematerializationError("source evidence schema/kind/id scheme is unsupported")
    snapshot_path = _regular_absolute_file(Path(value["snapshot_path"]), label="source snapshot")
    snapshot_sha = _require_sha(value["snapshot_sha256"], field="snapshot_sha256")
    if _file_sha256(snapshot_path) != snapshot_sha:
        raise RematerializationError("source snapshot SHA-256 mismatch")
    positive_ints = (
        "source_points_count",
        "dense_dimension",
        "chunk_tokens",
        "chunk_overlap",
        "chunk_min_tokens",
    )
    for field in positive_ints:
        item = value[field]
        minimum = 0 if field == "chunk_overlap" else 1
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise RematerializationError(f"{field} is invalid")
    for field in (
        "source_collection",
        "embedding_model",
        "tokenizer_model",
        "reranker_model",
        "dense_name",
        "sparse_name",
    ):
        if not isinstance(value[field], str) or not value[field].strip():
            raise RematerializationError(f"{field} must be a non-empty string")
    if not isinstance(value["document_header"], bool):
        raise RematerializationError("document_header must be boolean")
    for field in (
        "source_configuration_sha256",
        "retrieval_fingerprint",
        "vector_checksum_artifact_sha256",
        "vector_probe_sha256",
    ):
        _require_sha(value[field], field=field)
    for field in ("embedding_revision", "tokenizer_revision", "reranker_revision"):
        _require_revision(value[field], field=field)
    evidence = SourceEvidence(
        snapshot_path=snapshot_path,
        snapshot_sha256=snapshot_sha,
        source_collection=value["source_collection"],
        source_points_count=value["source_points_count"],
        source_configuration_sha256=value["source_configuration_sha256"],
        embedding_model=value["embedding_model"],
        embedding_revision=value["embedding_revision"],
        tokenizer_model=value["tokenizer_model"],
        tokenizer_revision=value["tokenizer_revision"],
        reranker_model=value["reranker_model"],
        reranker_revision=value["reranker_revision"],
        dense_name=value["dense_name"],
        dense_dimension=value["dense_dimension"],
        sparse_name=value["sparse_name"],
        chunk_tokens=value["chunk_tokens"],
        chunk_overlap=value["chunk_overlap"],
        chunk_min_tokens=value["chunk_min_tokens"],
        document_header=value["document_header"],
        retrieval_fingerprint=value["retrieval_fingerprint"],
        vector_checksum_artifact_sha256=value["vector_checksum_artifact_sha256"],
        vector_probe_sha256=value["vector_probe_sha256"],
        canonical=value,
    )
    return evidence, file_sha


def _length_prefixed(value: bytes) -> bytes:
    return struct.pack("<Q", len(value)) + value


def _dense_values(value: Any, *, expected_dim: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RematerializationError("dense vector is not a numeric sequence")
    if len(value) != expected_dim:
        raise RematerializationError(
            f"dense vector dimension {len(value)} != expected {expected_dim}"
        )
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RematerializationError("dense vector contains a non-number")
        number = float(item)
        if not math.isfinite(number):
            raise RematerializationError("dense vector contains a non-finite number")
        out.append(number)
    return out


def _sparse_values(value: Any) -> tuple[list[int], list[float]]:
    if value is None:
        return [], []
    indices = getattr(value, "indices", None)
    weights = getattr(value, "values", None)
    if isinstance(value, Mapping):
        indices = value.get("indices")
        weights = value.get("values")
    if (
        not isinstance(indices, Sequence)
        or isinstance(indices, (str, bytes, bytearray))
        or not isinstance(weights, Sequence)
        or isinstance(weights, (str, bytes, bytearray))
        or len(indices) != len(weights)
    ):
        raise RematerializationError("sparse vector indices/values are malformed")
    pairs: list[tuple[int, float]] = []
    for raw_index, raw_weight in zip(indices, weights):
        if (
            isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index < 0
            or isinstance(raw_weight, bool)
            or not isinstance(raw_weight, (int, float))
        ):
            raise RematerializationError("sparse vector contains an invalid entry")
        weight = float(raw_weight)
        if not math.isfinite(weight):
            raise RematerializationError("sparse vector contains a non-finite weight")
        pairs.append((raw_index, weight))
    pairs.sort(key=lambda item: item[0])
    if len({index for index, _ in pairs}) != len(pairs):
        raise RematerializationError("sparse vector contains duplicate indices")
    return [item[0] for item in pairs], [item[1] for item in pairs]


def logical_vector_sha256(
    logical_key: str,
    dense: Sequence[float],
    sparse_indices: Sequence[int],
    sparse_values: Sequence[float],
) -> str:
    """Hash vector bytes independently of Qdrant point ID and payload."""

    digest = hashlib.sha256()
    digest.update((LOGICAL_VECTOR_DIGEST_REVISION + "\0").encode())
    digest.update(_length_prefixed(logical_key.encode("utf-8")))
    digest.update(struct.pack("<Q", len(dense)))
    for value in dense:
        digest.update(struct.pack("<f", float(value)))
    digest.update(struct.pack("<Q", len(sparse_indices)))
    for index, value in zip(sparse_indices, sparse_values):
        digest.update(struct.pack("<Qf", int(index), float(value)))
    return digest.hexdigest()


def rekey_map_sha256(entries: Sequence[tuple[str, str, str]]) -> str:
    digest = hashlib.sha256(b"rekey-map-v1\0")
    for logical_key, old_id, new_id in sorted(entries):
        digest.update(_length_prefixed(logical_key.encode()))
        digest.update(_length_prefixed(old_id.encode()))
        digest.update(_length_prefixed(new_id.encode()))
    return digest.hexdigest()


def _ledger_plan_digests(
    connection: sqlite3.Connection,
) -> tuple[str, str, str, str]:
    """Hash the sealed plan in identity order without retaining millions of keys."""

    vectors = hashlib.sha256(b"logical-vector-collection-v1\0")
    for (vector_sha,) in connection.execute(
        "SELECT vector_sha256 FROM points ORDER BY logical_key"
    ):
        vectors.update(bytes.fromhex(vector_sha))

    old_ids = hashlib.sha256(b"legacy-id-set-v1\0")
    for (old_id,) in connection.execute("SELECT old_id FROM points ORDER BY old_id"):
        old_ids.update(_length_prefixed(old_id.encode()))

    new_ids = hashlib.sha256(b"version-id-set-v1\0")
    for (new_id,) in connection.execute("SELECT new_id FROM points ORDER BY new_id"):
        new_ids.update(_length_prefixed(new_id.encode()))

    rekeys = hashlib.sha256(b"rekey-map-v1\0")
    for logical_key, old_id, new_id in connection.execute(
        "SELECT logical_key,old_id,new_id FROM points ORDER BY logical_key"
    ):
        rekeys.update(_length_prefixed(logical_key.encode()))
        rekeys.update(_length_prefixed(old_id.encode()))
        rekeys.update(_length_prefixed(new_id.encode()))
    return (
        vectors.hexdigest(),
        old_ids.hexdigest(),
        new_ids.hexdigest(),
        rekeys.hexdigest(),
    )


def _point_parts(point: Any, *, dense_dim: int) -> tuple[str, dict[str, Any], list[float], list[int], list[float]]:
    point_value = getattr(point, "id", None)
    payload = getattr(point, "payload", None)
    vector = getattr(point, "vector", None)
    if isinstance(point, Mapping):
        point_value = point.get("id")
        payload = point.get("payload")
        vector = point.get("vector")
    if not isinstance(payload, Mapping) or not isinstance(vector, Mapping):
        raise RematerializationError("Qdrant point lacks payload or named vectors")
    dense = _dense_values(vector.get("dense"), expected_dim=dense_dim)
    sparse_indices, sparse_values = _sparse_values(vector.get("sparse"))
    return str(point_value), dict(payload), dense, sparse_indices, sparse_values


def _atomic_create(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    data = _canonical_bytes(value, pretty=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, value: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(_canonical_bytes(value, pretty=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.close(descriptor)
        descriptor = -1
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        tmp.unlink(missing_ok=True)


def _make_local_token_counter(cfg: Config) -> Callable[[str], int]:
    """Load only the pinned tokenizer from local files; never contact a model hub."""

    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"local_files_only": True}
    if cfg.tokenizer_revision:
        kwargs["revision"] = cfg.tokenizer_revision
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_model, **kwargs)
    except Exception as exc:  # noqa: BLE001 - local cache failures must be actionable
        raise RematerializationError(
            "pinned tokenizer is unavailable locally; rematerialization never downloads it"
        ) from exc

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0

    return count


def _source_filter(sources: Sequence[str]) -> models.Filter | None:
    ordered = sorted(set(sources))
    if set(ordered) == set(SOURCES_PRESENT):
        return None
    return models.Filter(
        should=[
            models.FieldCondition(key="source", match=models.MatchValue(value=source))
            for source in ordered
        ]
    )


def _iter_scratch_docs(source_files: Mapping[str, Path]) -> Iterator[Any]:
    for source in sorted(source_files):
        path = _regular_absolute_file(Path(source_files[source]), label=f"{source} source file")
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RematerializationError(f"{path}:{line_number}: {exc}") from exc
                if not isinstance(value, dict):
                    raise RematerializationError(f"{path}:{line_number}: record is not an object")
                try:
                    doc = (
                        snapshot_doc_to_canonical(value, strict=False)
                        if "snapshot_version" in value
                        else normalize(source, value)
                    )
                    yield _prepare_doc_for_index(doc)
                except Exception as exc:  # noqa: BLE001 - retain source cursor
                    raise RematerializationError(
                        f"{path}:{line_number}: canonicalization failed: {exc}"
                    ) from exc


def _load_scratch_overlays(
    source_files: Mapping[str, Path],
) -> dict[tuple[str, str], _ScratchOverlay]:
    """Build the small document-level payload overlay without invoking a tokenizer."""

    overlays: dict[tuple[str, str], _ScratchOverlay] = {}
    for doc in _iter_scratch_docs(source_files):
        key = (doc.source, doc.document_id)
        if key in overlays:
            raise RematerializationError(
                f"scratch input contains duplicate document {doc.source}:{doc.document_id}"
            )
        source_fingerprint = doc.source_fingerprint
        if not isinstance(source_fingerprint, str) or not _SHA256_RE.fullmatch(
            source_fingerprint
        ):
            raise RematerializationError(
                f"scratch document lacks a source fingerprint: {doc.source}:{doc.document_id}"
            )
        payload: dict[str, Any] = {
            "judges": list(doc.judges),
            "judges_raw": list(doc.judges_raw),
            "reporting_judge": doc.reporting_judge,
            "judge_extraction_confidence": doc.judge_extraction_confidence,
            "disposition": doc.disposition,
            "disposition_source": doc.disposition_source,
            "disposition_confidence": doc.disposition_confidence,
            "disposition_mixed": doc.disposition_mixed,
            "court_extractor_revision": doc.court_extractor_revision,
        }
        if doc.source == "supremecourt":
            for field in _SCRATCH_PROMOTED_FIELDS:
                if field in doc.promoted:
                    payload[field] = doc.promoted[field]
        overlays[key] = _ScratchOverlay(
            source=doc.source,
            document_id=doc.document_id,
            version_id=doc.version_id or derived_version_id(doc),
            content_hash=hashlib.sha256(doc.body_markdown.encode("utf-8")).hexdigest(),
            source_fingerprint=source_fingerprint,
            payload=payload,
        )
    if not overlays:
        raise RematerializationError("scratch input contains no canonical documents")
    return overlays


def _scratch_payload(
    source_payload: Mapping[str, Any], overlay: _ScratchOverlay
) -> dict[str, Any]:
    """Preserve the live point payload and replace only the court allowlist."""

    payload = dict(source_payload)
    for field in COURT_CANONICAL_FIELDS:
        payload.pop(field, None)
    if overlay.source == "supremecourt":
        for field in _SCRATCH_PROMOTED_FIELDS:
            payload.pop(field, None)
    payload.update(overlay.payload)
    return payload


def _iter_production_docs(snapshot_root: Path) -> Iterator[Any]:
    docs_root = snapshot_root / "docs"
    for source in SOURCES_PRESENT:
        for doc in iter_snapshot_docs(source, root=docs_root, strict=True):
            yield _prepare_doc_for_index(doc)


def _chunks(cfg: Config, doc: Any, count_tokens: Callable[[str], int]) -> list[Any]:
    return chunk_document(
        doc.body_markdown,
        max_tokens=cfg.chunk_tokens,
        overlap=cfg.chunk_overlap,
        min_tokens=cfg.chunk_min_tokens,
        count_tokens=count_tokens,
        page_boundaries=doc.page_boundaries,
        page_coordinate_reason=doc.page_coordinate_reason,
    )


def _embed_text(cfg: Config, doc: Any, chunk: Any) -> str:
    return build_embed_text(
        chunk.text,
        title=doc.title,
        document_type=doc.document_type,
        heading_path=chunk.heading_path,
        **_header_v2_kwargs(cfg, doc),
    )


def _create_ledger(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE points (seq INTEGER PRIMARY KEY, logical_key TEXT UNIQUE NOT NULL, "
        "old_id TEXT UNIQUE NOT NULL, new_id TEXT UNIQUE NOT NULL, source TEXT NOT NULL, "
        "document_id TEXT NOT NULL, version_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, "
        "vector_sha256 TEXT NOT NULL, payload_sha256 TEXT NOT NULL, embed_text_sha256 TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE documents (source TEXT NOT NULL, document_id TEXT NOT NULL, "
        "version_id TEXT NOT NULL, chunk_count INTEGER NOT NULL, "
        "PRIMARY KEY(source, document_id, version_id))"
    )
    connection.commit()
    return connection


def _scratch_source_identity(
    point_id_value: str,
    payload: Mapping[str, Any],
    overlays: Mapping[tuple[str, str], _ScratchOverlay],
) -> tuple[_ScratchOverlay, int, int, str]:
    source = payload.get("source")
    document_id = payload.get("document_id")
    chunk_index = payload.get("chunk_index")
    chunk_count = payload.get("document_chunk_count")
    if not isinstance(source, str) or not isinstance(document_id, str):
        raise RematerializationError("scratch source point lacks string document identity")
    if (
        isinstance(chunk_index, bool)
        or not isinstance(chunk_index, int)
        or chunk_index < 0
        or isinstance(chunk_count, bool)
        or not isinstance(chunk_count, int)
        or chunk_count < 1
    ):
        raise RematerializationError(
            f"scratch source has invalid chunk coordinates: {source}:{document_id}"
        )
    overlay = overlays.get((source, document_id))
    if overlay is None:
        raise RematerializationError(
            f"scratch source contains a point absent from raw input: {source}:{document_id}"
        )
    expected_id = point_id(source, document_id, chunk_index)
    if point_id_value != expected_id:
        raise RematerializationError(
            f"scratch source point ID is not legacy-deterministic: {source}:{document_id}:"
            f"{chunk_index}"
        )
    if payload.get("content_hash") != overlay.content_hash:
        raise RematerializationError(
            f"scratch raw/live canonical body differs: {source}:{document_id}"
        )
    canonical_hash = payload.get("canonical_content_hash")
    if canonical_hash is not None and canonical_hash != overlay.content_hash:
        raise RematerializationError(
            f"scratch source canonical-content hash differs: {source}:{document_id}"
        )
    # The current serving Supreme slice predates the source_fingerprint payload field.
    # When a source point carries it, require equality; otherwise the create-only raw-file
    # SHA in the binding plus the canonical body hash are the available live-era proof.
    observed_source_fingerprint = payload.get("source_fingerprint")
    if (
        observed_source_fingerprint is not None
        and observed_source_fingerprint != overlay.source_fingerprint
    ):
        raise RematerializationError(
            f"scratch raw/live source fingerprint differs: {source}:{document_id}"
        )
    raw_version_id = payload.get("version_id")
    version_id = (
        str(raw_version_id).strip()
        if raw_version_id not in (None, "")
        else overlay.version_id
    )
    return overlay, chunk_index, chunk_count, version_id


def _plan_scratch_overlay(
    client: Any,
    source_collection: str,
    cfg: Config,
    overlays: Mapping[tuple[str, str], _ScratchOverlay],
    connection: sqlite3.Connection,
    *,
    sources: Sequence[str],
    batch_size: int,
) -> tuple[int, int, str, str, str, str]:
    """Seal the live chunk layout and court-only target payload before target creation."""

    offset: Any = None
    seq = 0
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=source_collection,
                scroll_filter=_source_filter(sources),
                offset=offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            raise RematerializationError(f"scratch source scroll failed: {exc}") from exc
        if not points and next_offset is not None:
            raise RematerializationError(
                "scratch source scroll returned an empty continuation page"
            )
        for point in points:
            (
                observed_id,
                source_payload,
                dense,
                sparse_indices,
                sparse_values,
            ) = _point_parts(point, dense_dim=cfg.dense_dim)
            overlay, chunk_index, chunk_count, version_id = _scratch_source_identity(
                observed_id, source_payload, overlays
            )
            key = (overlay.source, overlay.document_id)
            document = observed.setdefault(
                key,
                {
                    "chunk_count": chunk_count,
                    "indices": set(),
                    "version_id": version_id,
                },
            )
            if (
                document["chunk_count"] != chunk_count
                or document["version_id"] != version_id
                or chunk_index in document["indices"]
            ):
                raise RematerializationError(
                    f"scratch source chunk declaration is inconsistent: "
                    f"{overlay.source}:{overlay.document_id}"
                )
            document["indices"].add(chunk_index)
            logical_key = (
                f"{overlay.source}\t{overlay.document_id}\t{version_id}\t{chunk_index}"
            )
            target_payload = _scratch_payload(source_payload, overlay)
            vector_sha = logical_vector_sha256(
                logical_key, dense, sparse_indices, sparse_values
            )
            try:
                connection.execute(
                    "INSERT INTO points VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        seq,
                        logical_key,
                        observed_id,
                        observed_id,
                        overlay.source,
                        overlay.document_id,
                        version_id,
                        chunk_index,
                        vector_sha,
                        _canonical_sha256(target_payload),
                        _canonical_sha256(source_payload),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise RematerializationError(
                    f"scratch source point identity is duplicated: {observed_id}"
                ) from exc
            seq += 1
        if next_offset is None:
            break
        if next_offset == offset:
            raise RematerializationError(
                "scratch source scroll continuation did not advance"
            )
        offset = next_offset

    if set(observed) != set(overlays):
        missing = sorted(set(overlays) - set(observed))
        extra = sorted(set(observed) - set(overlays))
        raise RematerializationError(
            "scratch raw/live document coverage differs; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for (source, document_id), document in sorted(observed.items()):
        chunk_count = document["chunk_count"]
        if document["indices"] != set(range(chunk_count)):
            raise RematerializationError(
                f"scratch source chunks are not contiguous: {source}:{document_id}"
            )
        connection.execute(
            "INSERT INTO documents VALUES (?,?,?,?)",
            (source, document_id, document["version_id"], chunk_count),
        )
    connection.commit()
    source_sha, old_ids_sha, new_ids_sha, rekey_sha = _ledger_plan_digests(connection)
    return len(observed), seq, source_sha, old_ids_sha, new_ids_sha, rekey_sha


def _audit_scratch_source(
    client: Any,
    collection: str,
    connection: sqlite3.Connection,
    *,
    sources: Sequence[str],
    dense_dim: int,
    batch_size: int,
) -> int:
    """Re-prove the complete source point, payload, and vector set against the plan."""

    offset: Any = None
    count = 0
    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=_source_filter(sources),
                offset=offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            raise RematerializationError(f"scratch source audit failed: {exc}") from exc
        if not points and next_offset is not None:
            raise RematerializationError(
                "scratch source audit returned an empty continuation page"
            )
        for point in points:
            observed_id, payload, dense, sparse_indices, sparse_values = _point_parts(
                point, dense_dim=dense_dim
            )
            row = connection.execute(
                "SELECT logical_key,vector_sha256,embed_text_sha256 FROM points "
                "WHERE old_id=?",
                (observed_id,),
            ).fetchone()
            if row is None:
                raise RematerializationError(
                    f"scratch source contains an unplanned point: {observed_id}"
                )
            if (
                logical_vector_sha256(row[0], dense, sparse_indices, sparse_values)
                != row[1]
                or _canonical_sha256(payload) != row[2]
            ):
                raise RematerializationError(
                    f"scratch source changed after planning: {observed_id}"
                )
            count += 1
        if next_offset is None:
            break
        if next_offset == offset:
            raise RematerializationError(
                "scratch source audit continuation did not advance"
            )
        offset = next_offset
    expected = connection.execute("SELECT COUNT(*) FROM points").fetchone()[0]
    if count != expected:
        raise RematerializationError(
            f"scratch source point count {count} != planned {expected}"
        )
    return count


def _retrieve_exact(client: Any, collection: str, ids: list[str]) -> dict[str, Any]:
    try:
        records = client.retrieve(
            collection_name=collection,
            ids=ids,
            with_payload=True,
            with_vectors=True,
        )
    except Exception as exc:
        raise RematerializationError(f"source retrieve failed: {exc}") from exc
    out = {str(getattr(record, "id", "")): record for record in records}
    if set(out) != set(ids):
        missing = sorted(set(ids) - set(out))
        extra = sorted(set(out) - set(ids))
        raise RematerializationError(
            f"source retrieve identity mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )
    return out


def _expect_source_payload(payload: Mapping[str, Any], doc: Any, chunk: Any, chunks: int) -> None:
    expected = {
        "source": doc.source,
        "document_id": doc.document_id,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "heading_path": list(chunk.heading_path),
        "title": doc.title,
        "document_type": doc.document_type,
        "content_hash": hashlib.sha256(doc.body_markdown.encode()).hexdigest(),
        "document_chunk_count": chunks,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise RematerializationError(
            f"legacy payload differs from canonical chunk for fields {mismatches}"
        )


def _plan(
    client: Any,
    source_collection: str,
    cfg: Config,
    docs: Callable[[], Iterator[Any]],
    connection: sqlite3.Connection,
    count_tokens: Callable[[str], int],
    *,
    batch_size: int,
) -> tuple[int, int, str, str, str, str]:
    seq = 0
    document_count = 0
    seen_documents: set[tuple[str, str]] = set()
    pending: list[tuple[Any, ...]] = []

    def flush_pending() -> None:
        while pending:
            batch = pending[:batch_size]
            del pending[:batch_size]
            records = _retrieve_exact(
                client, source_collection, [spec[6] for spec in batch]
            )
            for spec in batch:
                (
                    point_seq,
                    doc,
                    chunk,
                    chunk_count,
                    version_id,
                    logical_key,
                    old_id,
                    new_id,
                    target_payload,
                    embed_sha,
                ) = spec
                record = records[old_id]
                (
                    observed_id,
                    source_payload,
                    dense,
                    sparse_indices,
                    sparse_values,
                ) = _point_parts(record, dense_dim=cfg.dense_dim)
                if observed_id != old_id:
                    raise RematerializationError("legacy deterministic point ID mismatch")
                _expect_source_payload(source_payload, doc, chunk, chunk_count)
                vector_sha = logical_vector_sha256(
                    logical_key, dense, sparse_indices, sparse_values
                )
                connection.execute(
                    "INSERT INTO points VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        point_seq,
                        logical_key,
                        old_id,
                        new_id,
                        doc.source,
                        doc.document_id,
                        version_id,
                        chunk.chunk_index,
                        vector_sha,
                        _canonical_sha256(target_payload),
                        embed_sha,
                    ),
                )
            connection.commit()

    for doc in docs():
        key = (doc.source, doc.document_id)
        if key in seen_documents:
            raise RematerializationError(
                f"snapshot contains duplicate legacy document identity {key[0]}:{key[1]}"
            )
        seen_documents.add(key)
        version_id = doc.version_id or derived_version_id(doc)
        chunks = _chunks(cfg, doc, count_tokens)
        if not chunks:
            raise RematerializationError(
                f"canonical document produced no chunks: {doc.source}:{doc.document_id}"
            )
        state_hash = _document_state_hash(cfg, doc=doc)
        connection.execute(
            "INSERT INTO documents VALUES (?,?,?,?)",
            (doc.source, doc.document_id, version_id, len(chunks)),
        )
        document_count += 1
        for chunk in chunks:
            old_id = point_id(doc.source, doc.document_id, chunk.chunk_index)
            logical_key = (
                f"{doc.source}\t{doc.document_id}\t{version_id}\t{chunk.chunk_index}"
            )
            target_payload = build_payload(
                doc,
                chunk,
                document_chunk_count=len(chunks),
                document_state_hash=state_hash,
                cfg=cfg if cfg.generation_id else None,
            )
            # A payload-only scratch copy preserves the legacy ID byte-for-byte.  The
            # explicitly approved schema-v2 generation bridge is the sole exception: its
            # immutable point contract includes version_id and therefore requires the
            # audited old->new mapping captured in this ledger/report.
            new_id = (
                point_id(
                    doc.source,
                    doc.document_id,
                    chunk.chunk_index,
                    version_id=version_id,
                )
                if cfg.generation_id is not None
                else old_id
            )
            embed_sha = hashlib.sha256(_embed_text(cfg, doc, chunk).encode()).hexdigest()
            pending.append(
                (
                    seq,
                    doc,
                    chunk,
                    len(chunks),
                    version_id,
                    logical_key,
                    old_id,
                    new_id,
                    target_payload,
                    embed_sha,
                )
            )
            seq += 1
            if len(pending) >= batch_size:
                flush_pending()
    flush_pending()
    source_sha, old_ids_sha, new_ids_sha, rekey_sha = _ledger_plan_digests(connection)
    return (
        document_count,
        seq,
        source_sha,
        old_ids_sha,
        new_ids_sha,
        rekey_sha,
    )


def _audit_source(
    client: Any,
    collection: str,
    connection: sqlite3.Connection,
    *,
    sources: Sequence[str],
    dense_dim: int,
    batch_size: int,
) -> tuple[int, str]:
    offset: Any = None
    count = 0
    digest = hashlib.sha256(b"logical-vector-collection-v1\0")
    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=collection,
                scroll_filter=_source_filter(sources),
                offset=offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as exc:
            raise RematerializationError(f"source scroll failed: {exc}") from exc
        if not points and next_offset is not None:
            raise RematerializationError("source scroll returned empty continuation page")
        for point in points:
            observed_id, _payload, dense, sparse_indices, sparse_values = _point_parts(
                point, dense_dim=dense_dim
            )
            row = connection.execute(
                "SELECT logical_key, vector_sha256 FROM points WHERE old_id=?", (observed_id,)
            ).fetchone()
            if row is None:
                raise RematerializationError(f"source contains extra point {observed_id}")
            actual = logical_vector_sha256(row[0], dense, sparse_indices, sparse_values)
            if actual != row[1]:
                raise RematerializationError(f"source vector drift for {observed_id}")
            digest.update(bytes.fromhex(actual))
            count += 1
        if next_offset is None:
            break
        if next_offset == offset:
            raise RematerializationError("source scroll continuation did not advance")
        offset = next_offset
    expected = connection.execute("SELECT COUNT(*) FROM points").fetchone()[0]
    if count != expected:
        raise RematerializationError(f"source point count {count} != planned {expected}")
    # Scroll order is point-id order, not logical-key order.  Return an order-independent
    # audit marker; the authoritative logical digest is recomputed below in plan order.
    return count, digest.hexdigest()


def _point_rows(connection: sqlite3.Connection, start: int = 0) -> Iterator[tuple[Any, ...]]:
    yield from connection.execute(
        "SELECT seq, logical_key, old_id, new_id, source, document_id, version_id, "
        "chunk_index, vector_sha256, payload_sha256, embed_text_sha256 "
        "FROM points WHERE seq>=? ORDER BY seq",
        (start,),
    )


def _doc_map(docs: Callable[[], Iterator[Any]]) -> Iterator[tuple[Any, list[Any]]]:
    for doc in docs():
        yield doc, []


def _verify_completed(
    client: Any,
    target: str,
    connection: sqlite3.Connection,
    *,
    completed: int,
    dense_dim: int,
    batch_size: int,
) -> None:
    for start in range(0, completed, batch_size):
        rows = connection.execute(
            "SELECT logical_key,new_id,vector_sha256,payload_sha256 FROM points "
            "WHERE seq>=? AND seq<? ORDER BY seq",
            (start, min(start + batch_size, completed)),
        ).fetchall()
        ids = [row[1] for row in rows]
        records = _retrieve_exact(client, target, ids)
        for logical_key, target_id, vector_sha, payload_sha in rows:
            _id, payload, dense, sparse_indices, sparse_values = _point_parts(
                records[target_id], dense_dim=dense_dim
            )
            if (
                logical_vector_sha256(logical_key, dense, sparse_indices, sparse_values)
                != vector_sha
                or _canonical_sha256(payload) != payload_sha
            ):
                raise RematerializationError(
                    f"completed target prefix is corrupt at point {target_id}"
                )


def _upsert(
    client: Any,
    source_collection: str,
    target_collection: str,
    cfg: Config,
    docs: Callable[[], Iterator[Any]],
    connection: sqlite3.Connection,
    count_tokens: Callable[[str], int],
    checkpoint_path: Path,
    *,
    completed: int,
    batch_size: int,
) -> int:
    pending: list[tuple[Any, ...]] = []
    next_seq = completed

    def flush_pending() -> int:
        if not pending:
            return next_seq
        records = _retrieve_exact(
            client, source_collection, [spec[1] for spec in pending]
        )
        points: list[models.PointStruct] = []
        completed_through = next_seq
        for spec in pending:
            (
                seq,
                old_id,
                new_id,
                logical_key,
                vector_sha,
                payload_sha,
                doc,
                chunk,
                chunk_count,
                payload,
            ) = spec
            record = records[old_id]
            _id, source_payload, dense, sparse_indices, sparse_values = _point_parts(
                record, dense_dim=cfg.dense_dim
            )
            _expect_source_payload(source_payload, doc, chunk, chunk_count)
            if (
                logical_vector_sha256(
                    logical_key, dense, sparse_indices, sparse_values
                )
                != vector_sha
            ):
                raise RematerializationError(
                    f"source vector changed before upsert: {old_id}"
                )
            if _canonical_sha256(payload) != payload_sha:
                raise RematerializationError("target payload changed after preflight")
            vector: dict[str, Any] = {"dense": dense}
            if sparse_indices:
                vector["sparse"] = models.SparseVector(
                    indices=sparse_indices, values=sparse_values
                )
            points.append(models.PointStruct(id=new_id, vector=vector, payload=payload))
            completed_through = seq + 1
        client.upsert(
            collection_name=target_collection,
            points=points,
            wait=True,
        )
        _atomic_replace(checkpoint_path, {"completed": completed_through})
        pending.clear()
        return completed_through

    for doc in docs():
        version_id = doc.version_id or derived_version_id(doc)
        chunks = _chunks(cfg, doc, count_tokens)
        state_hash = _document_state_hash(cfg, doc=doc)
        for chunk in chunks:
            row = connection.execute(
                "SELECT seq,old_id,new_id,logical_key,vector_sha256,payload_sha256,"
                "embed_text_sha256 FROM points WHERE source=? AND document_id=? AND "
                "version_id=? AND chunk_index=?",
                (doc.source, doc.document_id, version_id, chunk.chunk_index),
            ).fetchone()
            if row is None:
                raise RematerializationError("upsert pass differs from sealed plan")
            seq, old_id, new_id, logical_key, vector_sha, payload_sha, embed_sha = row
            if seq < completed:
                continue
            if hashlib.sha256(_embed_text(cfg, doc, chunk).encode()).hexdigest() != embed_sha:
                raise RematerializationError("embedded-text inputs changed after preflight")
            payload = build_payload(
                doc,
                chunk,
                document_chunk_count=len(chunks),
                document_state_hash=state_hash,
                cfg=cfg if cfg.generation_id else None,
            )
            pending.append(
                (
                    seq,
                    old_id,
                    new_id,
                    logical_key,
                    vector_sha,
                    payload_sha,
                    doc,
                    chunk,
                    len(chunks),
                    payload,
                )
            )
            if len(pending) >= batch_size:
                next_seq = flush_pending()
    if pending:
        next_seq = flush_pending()
    return next_seq


def _upsert_scratch_overlay(
    client: Any,
    source_collection: str,
    target_collection: str,
    cfg: Config,
    overlays: Mapping[tuple[str, str], _ScratchOverlay],
    connection: sqlite3.Connection,
    checkpoint_path: Path,
    *,
    completed: int,
    batch_size: int,
) -> int:
    """Copy planned vectors/IDs while deriving only the allowlisted payload overlay."""

    next_seq = completed
    cursor = connection.execute(
        "SELECT seq,logical_key,old_id,new_id,source,document_id,version_id,"
        "chunk_index,vector_sha256,payload_sha256,embed_text_sha256 "
        "FROM points WHERE seq>=? ORDER BY seq",
        (completed,),
    )
    while rows := cursor.fetchmany(batch_size):
        records = _retrieve_exact(
            client, source_collection, [str(row[2]) for row in rows]
        )
        points: list[models.PointStruct] = []
        for row in rows:
            (
                seq,
                logical_key,
                old_id,
                new_id,
                source,
                document_id,
                _version_id,
                chunk_index,
                vector_sha,
                payload_sha,
                source_payload_sha,
            ) = row
            if old_id != new_id:
                raise RematerializationError("scratch copy attempted to rekey a point")
            record = records[old_id]
            (
                observed_id,
                source_payload,
                dense,
                sparse_indices,
                sparse_values,
            ) = _point_parts(record, dense_dim=cfg.dense_dim)
            overlay, observed_chunk, _chunk_count, _observed_version = (
                _scratch_source_identity(observed_id, source_payload, overlays)
            )
            if (
                observed_id != old_id
                or overlay.source != source
                or overlay.document_id != document_id
                or observed_chunk != chunk_index
                or _canonical_sha256(source_payload) != source_payload_sha
                or logical_vector_sha256(
                    logical_key, dense, sparse_indices, sparse_values
                )
                != vector_sha
            ):
                raise RematerializationError(
                    f"scratch source differs from the sealed plan: {old_id}"
                )
            target_payload = _scratch_payload(source_payload, overlay)
            if _canonical_sha256(target_payload) != payload_sha:
                raise RematerializationError(
                    f"scratch target payload changed after planning: {old_id}"
                )
            vector: dict[str, Any] = {"dense": dense}
            if sparse_indices:
                vector["sparse"] = models.SparseVector(
                    indices=sparse_indices, values=sparse_values
                )
            points.append(
                models.PointStruct(id=new_id, vector=vector, payload=target_payload)
            )
            next_seq = seq + 1
        client.upsert(
            collection_name=target_collection,
            points=points,
            wait=True,
        )
        _atomic_replace(checkpoint_path, {"completed": next_seq})
    return next_seq


def _target_digest(
    client: Any,
    target: str,
    connection: sqlite3.Connection,
    *,
    dense_dim: int,
    batch_size: int,
) -> tuple[int, str]:
    expected = connection.execute("SELECT COUNT(*) FROM points").fetchone()[0]
    digest = hashlib.sha256(b"logical-vector-collection-v1\0")
    count = 0
    cursor = connection.execute(
        "SELECT logical_key,new_id,vector_sha256,payload_sha256 "
        "FROM points ORDER BY logical_key"
    )
    while rows := cursor.fetchmany(batch_size):
        records = _retrieve_exact(client, target, [row[1] for row in rows])
        for logical_key, target_id, vector_sha, payload_sha in rows:
            _id, payload, dense, sparse_indices, sparse_values = _point_parts(
                records[target_id], dense_dim=dense_dim
            )
            actual = logical_vector_sha256(logical_key, dense, sparse_indices, sparse_values)
            if actual != vector_sha or _canonical_sha256(payload) != payload_sha:
                raise RematerializationError(f"target verification failed for {target_id}")
            digest.update(bytes.fromhex(actual))
            count += 1
    info = client.get_collection(target)
    advertised = getattr(info, "points_count", None)
    if advertised != expected or count != expected:
        raise RematerializationError("target contains missing or extra points")
    return count, digest.hexdigest()


def _validate_cfg_against_evidence(cfg: Config, evidence: SourceEvidence) -> None:
    expected = {
        "embedding_model": cfg.embed_model,
        "embedding_revision": cfg.embedding_revision,
        "tokenizer_model": cfg.tokenizer_model,
        "tokenizer_revision": cfg.tokenizer_revision,
        "reranker_model": cfg.rerank_model,
        "reranker_revision": cfg.reranker_revision,
        "dense_dimension": cfg.dense_dim,
        "chunk_tokens": cfg.chunk_tokens,
        "chunk_overlap": cfg.chunk_overlap,
        "chunk_min_tokens": cfg.chunk_min_tokens,
        "document_header": cfg.embed_header_v2,
        "retrieval_fingerprint": retrieval_fingerprint_sha256(cfg),
    }
    mismatches = [field for field, actual in expected.items() if getattr(evidence, field) != actual]
    if mismatches:
        raise RematerializationError(
            "target configuration differs from legacy vector evidence: " + ", ".join(mismatches)
        )
    if evidence.dense_name != "dense" or evidence.sparse_name != "sparse":
        raise RematerializationError("legacy evidence must bind dense/sparse named vectors")


def _binding_paths(state_dir: Path, run_id: str) -> tuple[Path, Path, Path, Path]:
    root = state_dir / "rematerialization" / run_id
    return root / "binding.json", root / "plan.sqlite", root / "checkpoint.json", root / "report.json"


def _load_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    value, _sha = _strict_json_file(path, label="rematerialization checkpoint")
    if set(value) != {"completed"} or isinstance(value["completed"], bool) or not isinstance(
        value["completed"], int
    ) or value["completed"] < 0:
        raise RematerializationError("rematerialization checkpoint is invalid")
    return value["completed"]


def _run_copy(
    client: Any,
    cfg: Config,
    *,
    source_collection: str,
    target_collection: str,
    sources: Sequence[str],
    docs: Callable[[], Iterator[Any]],
    binding: dict[str, Any],
    run_id: str,
    state_dir: Path,
    batch_size: int,
    resume: bool,
    apply: bool,
    allow_run_scoped_delta: bool,
    count_tokens: Callable[[str], int] | None,
    environ: Mapping[str, str] | None,
) -> RematerializationResult:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise RematerializationError("run_id is invalid")
    if source_collection == target_collection:
        raise RematerializationError("source and target collections must differ")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise RematerializationError("batch_size must be >= 1")
    require_explicit_approval(
        apply=apply,
        approval_env=QDRANT_WRITE_APPROVAL_ENV,
        operation="vector-preserving rematerialization",
        environ=environ,
    )
    binding_path, ledger_path, checkpoint_path, report_path = _binding_paths(state_dir, run_id)
    binding_sha = _canonical_sha256(binding)
    if resume:
        observed, _file_sha = _strict_json_file(binding_path, label="rematerialization binding")
        requested = dict(binding)
        requested["plan"] = observed.get("plan")
        if observed != requested or not isinstance(observed.get("plan"), Mapping):
            raise RematerializationError("resume binding differs from the original request")
        binding = observed
        binding_sha = _canonical_sha256(binding)
        if not ledger_path.is_file() or not client.collection_exists(target_collection):
            raise RematerializationError("resume requires the existing plan and target")
        connection = sqlite3.connect(ledger_path)
    else:
        if any(os.path.lexists(path) for path in (binding_path, ledger_path, checkpoint_path, report_path)):
            raise RematerializationError("fresh run refuses existing state paths")
        if client.collection_exists(target_collection):
            raise RematerializationError("fresh run refuses an existing target collection")
        _atomic_create(binding_path, binding)
        connection = _create_ledger(ledger_path)
    try:
        token_counter = count_tokens or _make_local_token_counter(cfg)
        if resume:
            planned = connection.execute("SELECT COUNT(*) FROM points").fetchone()[0]
            if planned < 1:
                raise RematerializationError("resume plan is empty")
            document_count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            metadata = binding["plan"]
            source_sha = metadata["source_logical_vector_sha256"]
            old_ids_sha = metadata["old_id_set_sha256"]
            new_ids_sha = metadata["new_id_set_sha256"]
            rekey_sha = metadata["rekey_map_sha256"]
        else:
            (
                document_count,
                planned,
                source_sha,
                old_ids_sha,
                new_ids_sha,
                rekey_sha,
            ) = _plan(
                client,
                source_collection,
                cfg,
                docs,
                connection,
                token_counter,
                batch_size=batch_size,
            )
            _audit_source(
                client,
                source_collection,
                connection,
                sources=sources,
                dense_dim=cfg.dense_dim,
                batch_size=batch_size,
            )
            binding["plan"] = {
                "document_count": document_count,
                "point_count": planned,
                "source_logical_vector_sha256": source_sha,
                "old_id_set_sha256": old_ids_sha,
                "new_id_set_sha256": new_ids_sha,
                "rekey_map_sha256": rekey_sha,
            }
            # Binding was create-only before mutation.  Replace it once, still before
            # target creation, with the complete deterministic plan identity.
            _atomic_replace(binding_path, binding)
            binding_sha = _canonical_sha256(binding)
            refuse_aliased_write_target(client, target_collection)
            ensure_collection(
                client,
                dataclasses.replace(cfg, collection_name=target_collection),
                recreate=False,
                apply=apply,
                allow_run_scoped_delta=allow_run_scoped_delta,
                environ=environ,
            )
        completed = _load_checkpoint(checkpoint_path)
        if completed > planned:
            raise RematerializationError("checkpoint exceeds planned point count")
        _verify_completed(
            client,
            target_collection,
            connection,
            completed=completed,
            dense_dim=cfg.dense_dim,
            batch_size=batch_size,
        )
        completed = _upsert(
            client,
            source_collection,
            target_collection,
            cfg,
            docs,
            connection,
            token_counter,
            checkpoint_path,
            completed=completed,
            batch_size=batch_size,
        )
        if completed != planned:
            raise RematerializationError("upsert ended before the complete plan")
        _audit_source(
            client,
            source_collection,
            connection,
            sources=sources,
            dense_dim=cfg.dense_dim,
            batch_size=batch_size,
        )
        target_count, target_sha = _target_digest(
            client,
            target_collection,
            connection,
            dense_dim=cfg.dense_dim,
            batch_size=batch_size,
        )
        if target_sha != source_sha:
            raise RematerializationError("source and target logical vector digests differ")
        target_configuration = collection_configuration(client.get_collection(target_collection))
        report = {
            "schema_version": REMATERIALIZATION_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "binding_sha256": binding_sha,
            "source_collection": source_collection,
            "target_collection": target_collection,
            "generation_id": cfg.generation_id,
            "document_count": document_count,
            "point_count": target_count,
            "source_logical_vector_sha256": source_sha,
            "target_logical_vector_sha256": target_sha,
            "old_id_set_sha256": old_ids_sha,
            "new_id_set_sha256": new_ids_sha,
            "rekey_map_sha256": rekey_sha,
            "target_configuration_sha256": collection_configuration_sha256(
                target_configuration
            ),
            "retrieval_fingerprint": retrieval_fingerprint_sha256(cfg),
            "court_extractor_revision": binding.get("court_extractor_revision"),
            "provenance_strength": binding["provenance_strength"],
        }
        _atomic_create(report_path, report)
        return RematerializationResult(
            target_collection=target_collection,
            report_path=report_path,
            point_count=target_count,
            document_count=document_count,
            source_logical_vector_sha256=source_sha,
            target_logical_vector_sha256=target_sha,
        )
    finally:
        connection.close()


def _run_scratch_overlay(
    client: Any,
    cfg: Config,
    *,
    source_collection: str,
    target_collection: str,
    source_files: Mapping[str, Path],
    binding: dict[str, Any],
    run_id: str,
    state_dir: Path,
    batch_size: int,
    resume: bool,
    apply: bool,
    environ: Mapping[str, str] | None,
) -> RematerializationResult:
    """Run the live-only payload overlay without reproducing historical chunks."""

    if not _RUN_ID_RE.fullmatch(run_id):
        raise RematerializationError("run_id is invalid")
    if source_collection == target_collection:
        raise RematerializationError("source and target collections must differ")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise RematerializationError("batch_size must be >= 1")
    require_explicit_approval(
        apply=apply,
        approval_env=QDRANT_WRITE_APPROVAL_ENV,
        operation="vector-preserving scratch court-payload overlay",
        environ=environ,
    )
    overlays = _load_scratch_overlays(source_files)
    sources = sorted(source_files)
    binding_path, ledger_path, checkpoint_path, report_path = _binding_paths(
        state_dir, run_id
    )
    if resume:
        observed, _file_sha = _strict_json_file(
            binding_path, label="rematerialization binding"
        )
        requested = dict(binding)
        requested["plan"] = observed.get("plan")
        if observed != requested or not isinstance(observed.get("plan"), Mapping):
            raise RematerializationError("resume binding differs from the original request")
        binding = observed
        if not ledger_path.is_file() or not client.collection_exists(target_collection):
            raise RematerializationError("resume requires the existing plan and target")
        connection = sqlite3.connect(ledger_path)
    else:
        if any(
            os.path.lexists(path)
            for path in (binding_path, ledger_path, checkpoint_path, report_path)
        ):
            raise RematerializationError("fresh run refuses existing state paths")
        if client.collection_exists(target_collection):
            raise RematerializationError(
                "fresh run refuses an existing target collection"
            )
        _atomic_create(binding_path, binding)
        connection = _create_ledger(ledger_path)
    try:
        if resume:
            planned = connection.execute("SELECT COUNT(*) FROM points").fetchone()[0]
            document_count = connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            if planned < 1 or document_count != len(overlays):
                raise RematerializationError("resume scratch plan is incomplete")
            metadata = binding["plan"]
            source_sha = _require_sha(
                metadata.get("source_logical_vector_sha256"),
                field="source_logical_vector_sha256",
            )
            old_ids_sha = _require_sha(
                metadata.get("old_id_set_sha256"), field="old_id_set_sha256"
            )
            new_ids_sha = _require_sha(
                metadata.get("new_id_set_sha256"), field="new_id_set_sha256"
            )
            rekey_sha = _require_sha(
                metadata.get("rekey_map_sha256"), field="rekey_map_sha256"
            )
        else:
            (
                document_count,
                planned,
                source_sha,
                old_ids_sha,
                new_ids_sha,
                rekey_sha,
            ) = _plan_scratch_overlay(
                client,
                source_collection,
                cfg,
                overlays,
                connection,
                sources=sources,
                batch_size=batch_size,
            )
            _audit_scratch_source(
                client,
                source_collection,
                connection,
                sources=sources,
                dense_dim=cfg.dense_dim,
                batch_size=batch_size,
            )
            binding["plan"] = {
                "document_count": document_count,
                "point_count": planned,
                "source_logical_vector_sha256": source_sha,
                "old_id_set_sha256": old_ids_sha,
                "new_id_set_sha256": new_ids_sha,
                "rekey_map_sha256": rekey_sha,
            }
            _atomic_replace(binding_path, binding)
            refuse_aliased_write_target(client, target_collection)
            ensure_collection(
                client,
                dataclasses.replace(cfg, collection_name=target_collection),
                recreate=False,
                apply=apply,
                allow_run_scoped_delta=True,
                environ=environ,
            )
        binding_sha = _canonical_sha256(binding)
        completed = _load_checkpoint(checkpoint_path)
        if completed > planned:
            raise RematerializationError("checkpoint exceeds planned point count")
        _verify_completed(
            client,
            target_collection,
            connection,
            completed=completed,
            dense_dim=cfg.dense_dim,
            batch_size=batch_size,
        )
        completed = _upsert_scratch_overlay(
            client,
            source_collection,
            target_collection,
            cfg,
            overlays,
            connection,
            checkpoint_path,
            completed=completed,
            batch_size=batch_size,
        )
        if completed != planned:
            raise RematerializationError("upsert ended before the complete scratch plan")
        _audit_scratch_source(
            client,
            source_collection,
            connection,
            sources=sources,
            dense_dim=cfg.dense_dim,
            batch_size=batch_size,
        )
        target_count, target_sha = _target_digest(
            client,
            target_collection,
            connection,
            dense_dim=cfg.dense_dim,
            batch_size=batch_size,
        )
        if target_sha != source_sha:
            raise RematerializationError(
                "source and target logical vector digests differ"
            )
        target_configuration = collection_configuration(
            client.get_collection(target_collection)
        )
        report = {
            "schema_version": REMATERIALIZATION_SCHEMA_VERSION,
            "kind": REPORT_KIND,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "binding_sha256": binding_sha,
            "source_collection": source_collection,
            "target_collection": target_collection,
            "generation_id": None,
            "document_count": document_count,
            "point_count": target_count,
            "source_logical_vector_sha256": source_sha,
            "target_logical_vector_sha256": target_sha,
            "old_id_set_sha256": old_ids_sha,
            "new_id_set_sha256": new_ids_sha,
            "rekey_map_sha256": rekey_sha,
            "target_configuration_sha256": collection_configuration_sha256(
                target_configuration
            ),
            "retrieval_fingerprint": retrieval_fingerprint_sha256(cfg),
            "court_extractor_revision": EXTRACTOR_REVISION,
            "provenance_strength": binding["provenance_strength"],
        }
        _atomic_create(report_path, report)
        return RematerializationResult(
            target_collection=target_collection,
            report_path=report_path,
            point_count=target_count,
            document_count=document_count,
            source_logical_vector_sha256=source_sha,
            target_logical_vector_sha256=target_sha,
        )
    finally:
        connection.close()


def rematerialize_scratch(
    client: Any,
    cfg: Config,
    *,
    source_collection: str,
    source_files: Mapping[str, Path],
    run_id: str,
    state_dir: Path,
    batch_size: int = 256,
    resume: bool = False,
    apply: bool = False,
    count_tokens: Callable[[str], int] | None = None,
    environ: Mapping[str, str] | None = None,
) -> RematerializationResult:
    """Copy a bounded live court slice to a run-scoped validation collection."""

    if not source_files or not set(source_files) <= _COURT_SOURCES:
        raise RematerializationError("scratch mode accepts only explicit ECD/Supreme files")
    target = f"georgian_legal_delta_court_{run_id.replace('.', '_')}"
    require_run_scoped_delta_collection(target)
    scratch_cfg = dataclasses.replace(cfg, collection_name=target, generation_id=None)
    file_identity = {
        source: {
            "path": str(_regular_absolute_file(Path(path), label=f"{source} source file")),
            "sha256": _file_sha256(Path(path).expanduser().absolute()),
        }
        for source, path in sorted(source_files.items())
    }
    binding = {
        "schema_version": REMATERIALIZATION_SCHEMA_VERSION,
        "kind": BINDING_KIND,
        "mode": "scratch",
        "run_id": run_id,
        "source_collection": source_collection,
        "target_collection": target,
        "source_files": file_identity,
        "sources": sorted(source_files),
        "retrieval_fingerprint": retrieval_fingerprint_sha256(scratch_cfg),
        "court_extractor_revision": EXTRACTOR_REVISION,
        "copy_strategy": SCRATCH_COPY_STRATEGY,
        "provenance_strength": "live-read-validation-only",
        "plan": None,
    }
    # ``count_tokens`` remains an API-compatibility hook for callers/tests, but this
    # live-only path must not load or call a tokenizer: historical chunk boundaries are
    # copied as evidence instead of being guessed from the current environment.
    del count_tokens
    return _run_scratch_overlay(
        client,
        scratch_cfg,
        source_collection=source_collection,
        target_collection=target,
        source_files=source_files,
        binding=binding,
        run_id=run_id,
        state_dir=state_dir,
        batch_size=batch_size,
        resume=resume,
        apply=apply,
        environ=environ,
    )


def rematerialize_generation(
    client: Any,
    cfg: Config,
    *,
    source_evidence_path: Path,
    snapshot_root: Path,
    generation_id: str,
    vector_checksum_path: Path,
    actor: str,
    run_id: str,
    state_dir: Path,
    batch_size: int = 256,
    resume: bool = False,
    apply: bool = False,
    count_tokens: Callable[[str], int] | None = None,
    environ: Mapping[str, str] | None = None,
) -> RematerializationResult:
    """Restore a frozen legacy source and create an immutable vector-copy generation."""

    # This exact release must be freshly embedded from its sealed structural inventory.
    # Refuse before reading evidence/model artifacts or touching the supplied client.
    refuse_frozen_candidate_rematerialization(generation_id)
    if not actor.strip():
        raise RematerializationError("actor must be non-empty")
    evidence, evidence_sha = load_source_evidence(source_evidence_path)
    checksum_path = _regular_absolute_file(vector_checksum_path, label="vector checksum")
    checksum_sha = _file_sha256(checksum_path)
    if checksum_sha != evidence.vector_checksum_artifact_sha256:
        raise RematerializationError("vector checksum artifact differs from source evidence")
    try:
        checksum_reference = load_checksum_reference(checksum_path)
    except Exception as exc:  # noqa: BLE001 - normalize immutable evidence failures
        raise RematerializationError(f"invalid vector checksum artifact: {exc}") from exc
    if checksum_reference.probe_sha256 != evidence.vector_probe_sha256:
        raise RematerializationError("vector probe digest differs from source evidence")
    snapshot_root = snapshot_root.expanduser().absolute()
    snapshot_manifest = verify_sealed_snapshot(
        snapshot_root, allow_preflight=False, require_all_sources=True
    )
    target = physical_collection_name(generation_id)
    target_cfg = dataclasses.replace(
        cfg,
        generation_id=generation_id,
        collection_name=target,
        state_dir=state_dir,
    )
    _validate_cfg_against_evidence(target_cfg, evidence)
    staging = f"georgian_legal_delta_remat_{run_id.replace('.', '_')}"
    require_run_scoped_delta_collection(staging)
    binding_path, ledger_path, checkpoint_path, report_path = _binding_paths(
        state_dir, run_id
    )
    state_paths = (binding_path, ledger_path, checkpoint_path, report_path)
    if resume:
        if not binding_path.is_file() or not ledger_path.is_file():
            raise RematerializationError("resume requires existing binding and plan state")
        if not client.collection_exists(staging) or not client.collection_exists(target):
            raise RematerializationError("resume requires existing source staging and target")
    else:
        if any(os.path.lexists(path) for path in state_paths):
            raise RematerializationError("fresh run refuses existing state paths")
        if client.collection_exists(staging) or client.collection_exists(target):
            raise RematerializationError("fresh run refuses existing source or target collection")
    require_explicit_approval(
        apply=apply,
        approval_env=QDRANT_WRITE_APPROVAL_ENV,
        operation="legacy snapshot staging restore",
        environ=environ,
    )
    if not resume:
        refuse_aliased_write_target(client, staging)
        # Upload the hash-verified local artifact instead of passing a ``file://`` URL.
        # A Qdrant process usually runs in a container and therefore cannot resolve the
        # host path even when the orchestrator can.  The upload endpoint preserves the
        # same create-only recovery semantics without relying on a shared filesystem.
        with evidence.snapshot_path.open("rb") as snapshot_handle:
            restored = client.http.snapshots_api.recover_from_uploaded_snapshot(
                collection_name=staging,
                snapshot=snapshot_handle,
                checksum=evidence.snapshot_sha256,
                priority=models.SnapshotPriority.SNAPSHOT,
                wait=True,
            )
        if getattr(restored, "result", None) is not True or not client.collection_exists(
            staging
        ):
            raise RematerializationError("Qdrant did not confirm source snapshot restore")
    source_info = client.get_collection(staging)
    if (
        getattr(source_info, "points_count", None) != evidence.source_points_count
        or collection_configuration_sha256(collection_configuration(source_info))
        != evidence.source_configuration_sha256
    ):
        raise RematerializationError("restored legacy source differs from source evidence")
    binding = {
        "schema_version": REMATERIALIZATION_SCHEMA_VERSION,
        "kind": BINDING_KIND,
        "mode": "production",
        "run_id": run_id,
        "actor": actor,
        "generation_id": generation_id,
        "source_collection": staging,
        "target_collection": target,
        "source_evidence_sha256": evidence_sha,
        "source_snapshot_sha256": evidence.snapshot_sha256,
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "snapshot_sha256": snapshot_manifest["snapshot_sha256"],
        "corpus_sha256": snapshot_manifest["corpus_sha256"],
        "vector_checksum_artifact_sha256": checksum_sha,
        "vector_probe_sha256": evidence.vector_probe_sha256,
        "retrieval_fingerprint": retrieval_fingerprint_sha256(target_cfg),
        "court_extractor_revision": EXTRACTOR_REVISION,
        "provenance_strength": "legacy-empirically-attested",
        "sources": list(SOURCES_PRESENT),
        "plan": None,
    }
    result = _run_copy(
        client,
        target_cfg,
        source_collection=staging,
        target_collection=target,
        sources=list(SOURCES_PRESENT),
        docs=lambda: _iter_production_docs(snapshot_root),
        binding=binding,
        run_id=run_id,
        state_dir=state_dir,
        batch_size=batch_size,
        resume=resume,
        apply=apply,
        allow_run_scoped_delta=False,
        count_tokens=count_tokens,
        environ=environ,
    )
    sealed = verify_snapshot_docs(snapshot_root / "docs")
    target_configuration = collection_configuration(client.get_collection(target))
    target_configuration_sha = collection_configuration_sha256(target_configuration)
    expected_binding_path = embed_binding_path(target_cfg)
    prepare_binding(
        target_cfg,
        sealed,
        checksum=checksum_reference,
        worker_count=1,
        collection_configuration=target_configuration,
        collection_configuration_sha256=target_configuration_sha,
        resume=expected_binding_path.exists(),
    )
    return result


def validate_rematerialization_report(
    path: Path,
    *,
    generation_id: str,
    physical_collection: str,
) -> tuple[dict[str, Any], str]:
    """Validate the compact report consumed by generation preparation."""

    value, file_sha = _strict_json_file(path, label="rematerialization report")
    required = {
        "schema_version",
        "kind",
        "created_at",
        "binding_sha256",
        "source_collection",
        "target_collection",
        "generation_id",
        "document_count",
        "point_count",
        "source_logical_vector_sha256",
        "target_logical_vector_sha256",
        "old_id_set_sha256",
        "new_id_set_sha256",
        "rekey_map_sha256",
        "target_configuration_sha256",
        "retrieval_fingerprint",
        "court_extractor_revision",
        "provenance_strength",
    }
    if set(value) != required:
        raise RematerializationError("rematerialization report has invalid keys")
    if (
        value["schema_version"] != REMATERIALIZATION_SCHEMA_VERSION
        or value["kind"] != REPORT_KIND
        or value["generation_id"] != generation_id
        or value["target_collection"] != physical_collection
        or value["source_logical_vector_sha256"]
        != value["target_logical_vector_sha256"]
        or value["court_extractor_revision"] != EXTRACTOR_REVISION
        or value["provenance_strength"] != "legacy-empirically-attested"
    ):
        raise RematerializationError("rematerialization report identity is invalid")
    for field in (
        "binding_sha256",
        "source_logical_vector_sha256",
        "target_logical_vector_sha256",
        "old_id_set_sha256",
        "new_id_set_sha256",
        "rekey_map_sha256",
        "target_configuration_sha256",
        "retrieval_fingerprint",
    ):
        _require_sha(value[field], field=field)
    for field in ("document_count", "point_count"):
        if isinstance(value[field], bool) or not isinstance(value[field], int) or value[field] < 1:
            raise RematerializationError(f"rematerialization report {field} is invalid")
    return value, file_sha
