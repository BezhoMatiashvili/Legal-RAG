"""Streaming, fail-closed verification for immutable Qdrant generations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import struct
import stat
import tempfile
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

from . import chunk_inventory
from .chunking import build_embed_text
from .generation import (
    CANONICAL_PAYLOAD_REQUIRED_FIELDS,
    CANONICAL_PAYLOAD_REVISION,
    GENERATION_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    DocumentRecord,
    GenerationArtifacts,
    GenerationManifest,
    SampleCheck,
    CollectionDigest,
    parse_rfc3339_utc,
)
from .generation_snapshot import (
    PROVENANCE_FILENAME,
    SOURCE_STATE_FILENAME,
    source_state_sha256,
)
from .release_inputs import GENERATION_ID as FROZEN_CANDIDATE_GENERATION_ID
from .snapshot import verify_sealed_snapshot

_SHA256_LENGTH = 64
COLLECTION_DIGEST_DOMAIN = b"georgian-legal-whole-collection-v1\0"
POINT_DIGEST_DOMAIN = b"georgian-legal-point-v1\0"
COLLECTION_PAYLOAD_PROJECTION = "all-payload-fields-canonical-json-v1"
COLLECTION_DENSE_ENCODING = "little-endian-float32"
COLLECTION_SPARSE_ENCODING = "sorted-uint64-index+little-endian-float32-weight"
_PREPARATION_PROVENANCE_FILENAME = "preparation_provenance.json"
_STRUCTURAL_INVENTORY_PROJECTION_FIELDS = (
    "sha256",
    "size_bytes",
    "identity_sha256",
    "record_count",
    "document_count",
    "chunk_count",
)


def _strict_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    if any(not isinstance(key, str) for key in payload):
        raise ValueError("point payload contains a non-string key")
    try:
        return json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"point payload is not canonical JSON: {exc}") from exc


def _f32_bytes(values: Sequence[Any], *, label: str) -> bytes:
    encoded = bytearray()
    for position, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{label}[{position}] is not numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label}[{position}] is not finite")
        try:
            encoded.extend(struct.pack("<f", number))
        except (OverflowError, struct.error) as exc:
            raise ValueError(f"{label}[{position}] is not float32") from exc
    return bytes(encoded)


def point_content_sha256(
    point_id: str,
    payload: Mapping[str, Any],
    vectors: Mapping[str, Any],
    *,
    dense_name: str = "dense",
    sparse_name: str = "sparse",
) -> str:
    """Hash one exact point using canonical payload and IEEE-754 float32 bytes."""

    if not isinstance(point_id, str) or not point_id:
        raise ValueError("point_id must be a non-empty string")
    if not isinstance(payload, Mapping):
        raise ValueError("point payload must be an object")
    if not isinstance(vectors, Mapping) or set(vectors) != {dense_name, sparse_name}:
        raise ValueError("point must have exactly the named dense and sparse vectors")
    dense = vectors[dense_name]
    if not isinstance(dense, Sequence) or isinstance(dense, (str, bytes, bytearray)):
        raise ValueError("point dense vector is invalid")
    sparse = vectors[sparse_name]
    indices = _sparse_field(sparse, "indices")
    weights = _sparse_field(sparse, "values")
    if (
        not isinstance(indices, Sequence)
        or isinstance(indices, (str, bytes, bytearray))
        or not isinstance(weights, Sequence)
        or isinstance(weights, (str, bytes, bytearray))
        or len(indices) != len(weights)
    ):
        raise ValueError("point sparse vector is invalid")
    pairs: list[tuple[int, Any]] = []
    for position, (index, weight) in enumerate(zip(indices, weights, strict=True)):
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index > (2**64 - 1)
        ):
            raise ValueError(f"sparse index {position} is invalid")
        pairs.append((index, weight))
    pairs.sort(key=lambda item: item[0])
    if any(left[0] == right[0] for left, right in zip(pairs, pairs[1:])):
        raise ValueError("sparse indices are not unique")

    identifier = point_id.encode("utf-8")
    payload_bytes = _strict_payload_bytes(payload)
    dense_bytes = _f32_bytes(dense, label="dense")
    digest = hashlib.sha256()
    digest.update(POINT_DIGEST_DOMAIN)
    digest.update(struct.pack("<Q", len(identifier)))
    digest.update(identifier)
    digest.update(struct.pack("<Q", len(payload_bytes)))
    digest.update(payload_bytes)
    digest.update(struct.pack("<Q", len(dense)))
    digest.update(dense_bytes)
    digest.update(struct.pack("<Q", len(pairs)))
    for sparse_index, sparse_weight in pairs:
        digest.update(struct.pack("<Q", sparse_index))
        digest.update(_f32_bytes([sparse_weight], label="sparse"))
    return digest.hexdigest()


def whole_collection_sha256(rows: Iterable[tuple[str, str]]) -> tuple[str, int]:
    """Compose sorted point digests into the sealed whole-collection SHA-256."""

    digest = hashlib.sha256()
    digest.update(COLLECTION_DIGEST_DOMAIN)
    previous: str | None = None
    count = 0
    for point_id, point_digest in rows:
        if previous is not None and point_id <= previous:
            raise ValueError("whole-collection digest rows are not strictly point-ID sorted")
        if not _is_sha256(point_digest):
            raise ValueError("point digest is not SHA-256")
        encoded_id = point_id.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded_id)))
        digest.update(encoded_id)
        digest.update(bytes.fromhex(point_digest))
        previous = point_id
        count += 1
    return digest.hexdigest(), count


@dataclass(frozen=True)
class VerificationOutcome:
    ok: bool
    issue_count: int
    examples: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "issue_count": self.issue_count,
            "examples": list(self.examples),
        }


@dataclass(frozen=True)
class VerificationReport:
    generation_id: str
    manifest_sha256: str
    physical_collection: str | None
    verified_at: str
    covered_runs: tuple[dict[str, str], ...]
    stats: Mapping[str, Any]
    coverage: VerificationOutcome
    integrity: VerificationOutcome
    freshness: VerificationOutcome
    quality: VerificationOutcome
    verification_id: str | None = None
    expected_collection_sha256: str | None = None
    observed_collection_sha256: str | None = None
    expected_collection_configuration_sha256: str | None = None
    observed_collection_configuration_sha256: str | None = None
    vector_checksum_artifact_sha256: str | None = None
    vector_probe_sha256: str | None = None
    schema_version: int = GENERATION_SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        return all(
            outcome.ok
            for outcome in (
                self.coverage,
                self.integrity,
                self.freshness,
                self.quality,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "manifest_sha256": self.manifest_sha256,
            "physical_collection": self.physical_collection,
            "verified_at": self.verified_at,
            "verification_id": self.verification_id,
            "ok": self.ok,
            "covered_runs": [dict(run) for run in self.covered_runs],
            "stats": dict(self.stats),
            "expected_collection_sha256": self.expected_collection_sha256,
            "observed_collection_sha256": self.observed_collection_sha256,
            "expected_collection_configuration_sha256": (
                self.expected_collection_configuration_sha256
            ),
            "observed_collection_configuration_sha256": (
                self.observed_collection_configuration_sha256
            ),
            "vector_checksum_artifact_sha256": (
                self.vector_checksum_artifact_sha256
            ),
            "vector_probe_sha256": self.vector_probe_sha256,
            "coverage": self.coverage.to_dict(),
            "integrity": self.integrity.to_dict(),
            "freshness": self.freshness.to_dict(),
            "quality": self.quality.to_dict(),
        }


@dataclass(frozen=True)
class StructuralInventoryBinding:
    """One independently supplied sealed snapshot bound to generation provenance."""

    root: Path
    artifact: Path
    manifest_entry: Mapping[str, Any]
    expected_identity: Mapping[str, Any]
    proof: Mapping[str, Any]

    def iter_documents(self) -> Iterator[dict[str, Any]]:
        """Return a fresh strict stream whose final checks run only on exhaustion."""

        return chunk_inventory.iter_validated_inventory(
            self.artifact,
            manifest_entry=self.manifest_entry,
            expected_identity=self.expected_identity,
        )


def _strict_json_object(raw: bytes, *, origin: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{origin} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{origin} contains non-finite number {item!r}")
            ),
        )
    except ValueError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"cannot parse {origin}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{origin} must be a JSON object")
    return value


def _load_generation_json(
    artifacts: GenerationArtifacts,
    filename: str,
    *,
    required: bool,
) -> dict[str, Any] | None:
    expected_sha = artifacts.checksums.files.get(filename)
    if expected_sha is None:
        if required:
            raise ValueError(f"generation lacks checksum-bound {filename}")
        return None
    path = artifacts.root / filename
    try:
        mode = path.lstat().st_mode
        raw = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read generation {filename}: {exc}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"generation {filename} must be a regular non-symlink file")
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"generation {filename} changed after checksum validation")
    return _strict_json_object(raw, origin=f"generation {filename}")


def _projection(value: object, *, origin: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        field not in value for field in _STRUCTURAL_INVENTORY_PROJECTION_FIELDS
    ):
        raise ValueError(f"{origin} lacks the structural inventory projection")
    projected = {
        field: value[field] for field in _STRUCTURAL_INVENTORY_PROJECTION_FIELDS
    }
    if any(
        not _is_sha256(projected[field]) for field in ("sha256", "identity_sha256")
    ):
        raise ValueError(f"{origin} structural inventory hashes are invalid")
    for field, minimum in (
        ("size_bytes", 1),
        ("record_count", 1),
        ("document_count", 0),
        ("chunk_count", 0),
    ):
        if not _is_int(projected[field], minimum=minimum):
            raise ValueError(f"{origin} structural inventory {field} is invalid")
    return projected


def _generation_structural_provenance(
    artifacts: GenerationArtifacts,
    *,
    required: bool,
) -> dict[str, Any] | None:
    """Load the published/prepared generation bindings without trusting path hints."""

    source_artifact = _load_generation_json(
        artifacts,
        SOURCE_STATE_FILENAME,
        required=required,
    )
    if source_artifact is None:
        return None
    if "state" in source_artifact:
        if set(source_artifact) != {
            "schema_version",
            "generation_id",
            "source",
            "state_sha256",
            "state",
        }:
            raise ValueError("published generation source_state shape is invalid")
        state = source_artifact["state"]
        if (
            source_artifact["schema_version"] != artifacts.manifest.schema_version
            or source_artifact["generation_id"] != artifacts.manifest.generation_id
            or source_artifact["source"] != artifacts.manifest.source.name
            or source_artifact["state_sha256"]
            != artifacts.manifest.source.state_sha256
        ):
            raise ValueError("published generation source_state identity mismatch")
    else:
        state = source_artifact
    if not isinstance(state, Mapping):
        raise ValueError("generation source state must be an object")
    if source_state_sha256(state) != artifacts.manifest.source.state_sha256:
        raise ValueError("generation source state does not reproduce manifest source hash")
    structural = state.get("structural_chunk_inventory")
    if structural is None:
        if required:
            raise ValueError("generation source state lacks structural inventory provenance")
        return None
    source_projection = _projection(
        structural,
        origin="generation source state",
    )
    snapshot_id = state.get("snapshot_id")
    snapshot_sha256 = state.get("snapshot_sha256")
    corpus_sha256 = state.get("corpus_sha256")
    if (
        snapshot_id != artifacts.manifest.corpus.name
        or snapshot_sha256 != artifacts.manifest.corpus.snapshot_sha256
        or not _is_sha256(corpus_sha256)
    ):
        raise ValueError("generation source state snapshot identity mismatch")

    provenance_name = (
        PROVENANCE_FILENAME
        if PROVENANCE_FILENAME in artifacts.checksums.files
        else _PREPARATION_PROVENANCE_FILENAME
    )
    provenance = _load_generation_json(artifacts, provenance_name, required=True)
    assert provenance is not None
    if provenance_name == PROVENANCE_FILENAME:
        preparation = provenance.get("preparation")
        if not isinstance(preparation, Mapping):
            raise ValueError("published provenance lacks preparation evidence")
        evidence = preparation.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError("published provenance preparation evidence is invalid")
        corpus = provenance.get("corpus")
        if corpus != artifacts.manifest.to_dict()["corpus"]:
            raise ValueError("published provenance corpus differs from manifest")
    else:
        evidence = provenance
    snapshot_evidence = evidence.get("snapshot")
    expected_snapshot_evidence = {
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "corpus_sha256": corpus_sha256,
        "structural_chunk_inventory": source_projection,
    }
    if snapshot_evidence != expected_snapshot_evidence:
        raise ValueError(
            "generation preparation provenance differs from source-state snapshot binding"
        )
    return expected_snapshot_evidence


def load_generation_structural_provenance(
    artifacts: GenerationArtifacts,
) -> dict[str, Any]:
    """Return the checksummed generation-to-snapshot structural binding.

    Release-side verification loaders use this public read-only projection to avoid
    accepting two mutually consistent but forged verification reports.
    """

    if not isinstance(artifacts, GenerationArtifacts):
        raise TypeError("artifacts must be GenerationArtifacts")
    value = _generation_structural_provenance(artifacts, required=True)
    assert value is not None
    return value


def load_structural_inventory_binding(
    artifacts: GenerationArtifacts,
    snapshot_root: str | Path,
) -> StructuralInventoryBinding:
    """Bind an independently supplied sealed snapshot and its exact chunk inventory."""

    if not isinstance(artifacts, GenerationArtifacts):
        raise TypeError("artifacts must be GenerationArtifacts")
    expected = _generation_structural_provenance(artifacts, required=True)
    assert expected is not None
    root = Path(snapshot_root).expanduser().absolute()
    try:
        snapshot_manifest = verify_sealed_snapshot(root)
    except Exception as exc:
        raise ValueError(f"sealed snapshot validation failed: {exc}") from exc
    observed_snapshot = {
        "snapshot_id": snapshot_manifest.get("snapshot_id"),
        "snapshot_sha256": snapshot_manifest.get("snapshot_sha256"),
        "corpus_sha256": snapshot_manifest.get("corpus_sha256"),
    }
    if observed_snapshot != {
        field: expected[field]
        for field in ("snapshot_id", "snapshot_sha256", "corpus_sha256")
    }:
        raise ValueError("sealed snapshot identity differs from generation provenance")
    entry = snapshot_manifest.get("structural_chunk_inventory")
    if not isinstance(entry, Mapping):
        raise ValueError("sealed snapshot lacks structural chunk inventory")
    observed_projection = _projection(entry, origin="sealed snapshot")
    if observed_projection != expected["structural_chunk_inventory"]:
        raise ValueError("sealed snapshot inventory differs from generation provenance")
    if (
        observed_projection["document_count"]
        != artifacts.manifest.indexed_document_count
        or observed_projection["chunk_count"] != artifacts.manifest.chunk_count
    ):
        raise ValueError("sealed structural inventory counts differ from generation manifest")
    identity = entry.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("sealed structural inventory identity is invalid")
    tokenizer = identity.get("tokenizer")
    chunker = identity.get("chunker")
    header = identity.get("document_header")
    sources = identity.get("sources")
    if (
        not isinstance(tokenizer, Mapping)
        or tokenizer.get("model") != artifacts.manifest.model.tokenizer_model
        or tokenizer.get("revision") != artifacts.manifest.model.tokenizer_revision
        or not isinstance(chunker, Mapping)
        or chunker.get("max_tokens") != artifacts.manifest.chunking.max_tokens
        or chunker.get("overlap_tokens") != artifacts.manifest.chunking.overlap_tokens
        or not isinstance(header, Mapping)
        or header.get("enabled") != artifacts.manifest.chunking.document_header
        or not isinstance(sources, list)
        or set(sources) != {run.source for run in artifacts.manifest.covered_runs}
    ):
        raise ValueError(
            "sealed structural inventory tokenizer/chunker/header identity mismatch"
        )
    relative = entry.get("path")
    if relative != chunk_inventory.CHUNK_INVENTORY_FILENAME:
        raise ValueError("sealed structural inventory path is invalid")
    proof = {
        **observed_snapshot,
        "structural_inventory_sha256": observed_projection["sha256"],
        "structural_inventory_size_bytes": observed_projection["size_bytes"],
        "structural_inventory_identity_sha256": observed_projection[
            "identity_sha256"
        ],
        "structural_inventory_record_count": observed_projection["record_count"],
        "expected_document_count": observed_projection["document_count"],
        "expected_chunk_count": observed_projection["chunk_count"],
        "tokenizer_model": tokenizer["model"],
        "tokenizer_revision": tokenizer["revision"],
        "chunker_revision": chunker["revision"],
        "chunk_max_tokens": chunker["max_tokens"],
        "chunk_overlap_tokens": chunker["overlap_tokens"],
        "chunk_min_tokens": chunker["min_tokens"],
        "document_header_enabled": header["enabled"],
        "document_header_revision": header["revision"],
    }
    return StructuralInventoryBinding(
        root=root,
        artifact=root / relative,
        manifest_entry=dict(entry),
        expected_identity=dict(identity),
        proof=proof,
    )


class _Issues:
    def __init__(self, max_examples: int) -> None:
        self.count = 0
        self.examples: list[dict[str, Any]] = []
        self.max_examples = max_examples

    def add(self, code: str, **details: Any) -> None:
        self.count += 1
        if len(self.examples) < self.max_examples:
            self.examples.append({"code": code, **details})

    def outcome(self) -> VerificationOutcome:
        return VerificationOutcome(
            ok=self.count == 0,
            issue_count=self.count,
            examples=tuple(self.examples),
        )


def _is_int(value: Any, *, minimum: int = 0) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _is_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    return all(char in "0123456789abcdef" for char in value)


def _point_field(point: Any, name: str) -> Any:
    if isinstance(point, Mapping):
        return point.get(name)
    return getattr(point, name, None)


def _sparse_field(sparse: Any, name: str) -> Any:
    if isinstance(sparse, Mapping):
        return sparse.get(name)
    return getattr(sparse, name, None)


def _canonical_point_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    try:
        parsed = uuid.UUID(text)
    except ValueError:
        return None
    return text if str(parsed) == text else None


def _validate_vectors(
    vectors: Any,
    manifest: GenerationManifest,
    issues: _Issues,
    *,
    point_id: str,
) -> None:
    if not isinstance(vectors, Mapping):
        issues.add("missing_or_invalid_vectors", point_id=point_id)
        return

    dense_name = manifest.vector_space.dense_name
    sparse_name = manifest.vector_space.sparse_name
    expected_names = {dense_name, sparse_name}
    actual_names = set(vectors)
    for missing in sorted(expected_names - actual_names):
        issues.add("missing_named_vector", point_id=point_id, vector_name=missing)
    for unexplained in sorted(actual_names - expected_names, key=str):
        issues.add(
            "unexpected_named_vector",
            point_id=point_id,
            vector_name=str(unexplained),
        )

    dense = vectors.get(dense_name)
    if dense_name in vectors:
        if not isinstance(dense, Sequence) or isinstance(
            dense, (str, bytes, bytearray)
        ):
            issues.add("invalid_dense_vector", point_id=point_id)
        else:
            if len(dense) != manifest.vector_space.dense_dimension:
                issues.add(
                    "dense_dimension_mismatch",
                    point_id=point_id,
                    expected=manifest.vector_space.dense_dimension,
                    actual=len(dense),
                )
            for index, value in enumerate(dense):
                if isinstance(value, bool) or not isinstance(value, Real):
                    issues.add(
                        "dense_non_numeric",
                        point_id=point_id,
                        vector_index=index,
                    )
                elif not math.isfinite(float(value)):
                    issues.add(
                        "dense_non_finite",
                        point_id=point_id,
                        vector_index=index,
                    )

    sparse = vectors.get(sparse_name)
    if sparse_name not in vectors:
        return
    indices = _sparse_field(sparse, "indices")
    values = _sparse_field(sparse, "values")
    if (
        not isinstance(indices, Sequence)
        or isinstance(indices, (str, bytes, bytearray))
        or not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
    ):
        issues.add("invalid_sparse_vector", point_id=point_id)
        return
    if len(indices) != len(values):
        issues.add(
            "sparse_cardinality_mismatch",
            point_id=point_id,
            index_count=len(indices),
            value_count=len(values),
        )

    valid_indices = True
    normalized_indices: list[int] = []
    for position, value in enumerate(indices):
        if not _is_int(value):
            valid_indices = False
            issues.add(
                "sparse_invalid_index",
                point_id=point_id,
                vector_index=position,
            )
        else:
            normalized_indices.append(value)
    if valid_indices and any(
        left >= right
        for left, right in zip(normalized_indices, normalized_indices[1:], strict=False)
    ):
        issues.add("sparse_indexes_not_sorted_unique", point_id=point_id)

    for position, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            issues.add(
                "sparse_non_numeric",
                point_id=point_id,
                vector_index=position,
            )
        elif not math.isfinite(float(value)):
            issues.add(
                "sparse_non_finite",
                point_id=point_id,
                vector_index=position,
            )
        elif float(value) < 0:
            issues.add(
                "sparse_negative_value",
                point_id=point_id,
                vector_index=position,
            )


def _expect_payload_value(
    payload: Mapping[str, Any],
    field: str,
    expected: Any,
    issues: _Issues,
    *,
    point_id: str,
) -> None:
    if field not in payload:
        issues.add("missing_payload_field", point_id=point_id, field=field)
    elif payload[field] != expected:
        issues.add(
            "payload_identity_mismatch",
            point_id=point_id,
            field=field,
            expected=expected,
            actual=payload[field],
        )


def _validate_canonical_payload(
    payload: Mapping[str, Any],
    manifest: GenerationManifest,
    issues: _Issues,
    *,
    point_id: str,
    text: str | None,
) -> None:
    """Validate the schema-v2 fields that make a point admissible as legal evidence."""
    identity_fields = {
        "canonical_payload_revision",
        "canonical_text_exact",
        "model_revision",
    }
    for field in sorted(CANONICAL_PAYLOAD_REQUIRED_FIELDS - identity_fields):
        if field not in payload:
            issues.add("missing_payload_field", point_id=point_id, field=field)

    _expect_payload_value(
        payload,
        "canonical_payload_revision",
        CANONICAL_PAYLOAD_REVISION,
        issues,
        point_id=point_id,
    )
    _expect_payload_value(
        payload,
        "canonical_text_exact",
        True,
        issues,
        point_id=point_id,
    )
    _expect_payload_value(
        payload,
        "model_revision",
        manifest.model.embedding_revision,
        issues,
        point_id=point_id,
    )

    for field in ("canonical_content_hash", "passage_hash", "source_fingerprint"):
        if field in payload and not _is_sha256(payload[field]):
            issues.add("invalid_payload_hash", point_id=point_id, field=field)
    if (
        _is_sha256(payload.get("canonical_content_hash"))
        and payload.get("canonical_content_hash") != payload.get("content_hash")
    ):
        issues.add(
            "canonical_content_hash_mismatch",
            point_id=point_id,
        )
    if text is not None and _is_sha256(payload.get("passage_hash")):
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if payload["passage_hash"] != actual:
            issues.add("passage_hash_mismatch", point_id=point_id)

    start, end = payload.get("char_start"), payload.get("char_end")
    if not _is_int(start):
        issues.add("invalid_payload_field", point_id=point_id, field="char_start")
    if not _is_int(end, minimum=1):
        issues.add("invalid_payload_field", point_id=point_id, field="char_end")
    if _is_int(start) and _is_int(end, minimum=1):
        if end <= start or (text is not None and end - start != len(text)):
            issues.add("invalid_canonical_offsets", point_id=point_id)

    for field in (
        "passage_id",
        "normalizer_revision",
        "chunker_revision",
        "version_id",
        "version_lineage_status",
        "official_url",
    ):
        if field in payload and not _is_string(payload[field]):
            issues.add("invalid_payload_field", point_id=point_id, field=field)
    authority = payload.get("source_authority")
    if authority not in {"official", "primary_official"}:
        issues.add("invalid_payload_field", point_id=point_id, field="source_authority")
    if payload.get("admissible") is not True:
        issues.add("invalid_payload_field", point_id=point_id, field="admissible")

    for field in (
        "article_id",
        "clause_id",
        "subarticle",
        "chapter",
        "parent_id",
        "consolidation_status",
        "official_binary_url",
    ):
        if field in payload and payload[field] is not None and not _is_string(payload[field]):
            issues.add("invalid_payload_field", point_id=point_id, field=field)
    heading_path = payload.get("heading_path")
    if "heading_path" in payload and (
        not isinstance(heading_path, list)
        or any(not _is_string(item) for item in heading_path)
    ):
        issues.add("invalid_payload_field", point_id=point_id, field="heading_path")
    supersedes = payload.get("supersedes")
    if "supersedes" in payload and (
        not isinstance(supersedes, list)
        or any(not _is_string(item) for item in supersedes)
    ):
        issues.add("invalid_payload_field", point_id=point_id, field="supersedes")

    for field in (
        "article_start_chunk_index",
        "parent_chunk_index",
        "page_start",
        "page_end",
    ):
        value = payload.get(field)
        if field in payload and value is not None and not _is_int(value):
            issues.add("invalid_payload_field", point_id=point_id, field=field)
    page_reason = payload.get("page_coordinate_reason")
    page_start = payload.get("page_start")
    page_end = payload.get("page_end")
    mapping_sha = payload.get("page_boundary_mapping_sha256")
    page_boundaries = payload.get("page_boundaries")
    chunk_index = payload.get("chunk_index")
    if page_reason not in {"exact_pdf_text", "source_not_paginated"}:
        issues.add(
            "invalid_payload_field", point_id=point_id, field="page_coordinate_reason"
        )
    if page_reason == "source_not_paginated" and (
        page_start is not None or page_end is not None or mapping_sha is not None
    ):
        issues.add("invalid_page_coordinates", point_id=point_id)
    if page_reason == "exact_pdf_text" and (
        not _is_int(page_start, minimum=1)
        or not _is_int(page_end, minimum=1)
        or page_end < page_start
        or not _is_sha256(mapping_sha)
    ):
        issues.add("invalid_page_coordinates", point_id=point_id)
    if chunk_index == 0:
        if not isinstance(page_boundaries, list):
            issues.add("invalid_page_boundary_mapping", point_id=point_id)
        else:
            previous_end = 0
            valid_mapping = True
            for index, boundary in enumerate(page_boundaries):
                if (
                    not isinstance(boundary, dict)
                    or set(boundary) != {"page", "char_start", "char_end"}
                    or boundary.get("page") != index + 1
                    or not _is_int(boundary.get("char_start"))
                    or not _is_int(boundary.get("char_end"))
                    or boundary["char_start"] < previous_end
                    or boundary["char_end"] < boundary["char_start"]
                ):
                    valid_mapping = False
                    break
                previous_end = boundary["char_end"]
            if not valid_mapping:
                issues.add("invalid_page_boundary_mapping", point_id=point_id)
            else:
                observed_mapping_sha = (
                    hashlib.sha256(
                        json.dumps(
                            page_boundaries,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    if page_boundaries
                    else None
                )
                if observed_mapping_sha != mapping_sha:
                    issues.add("page_boundary_mapping_hash_mismatch", point_id=point_id)
    elif page_boundaries is not None:
        issues.add("invalid_page_boundary_mapping", point_id=point_id)
    if "version_lineage_complete" in payload and not isinstance(
        payload["version_lineage_complete"], bool
    ):
        issues.add(
            "invalid_payload_field",
            point_id=point_id,
            field="version_lineage_complete",
        )
    freshness = payload.get("freshness_sla_met")
    if "freshness_sla_met" in payload and freshness is not None and not isinstance(
        freshness, bool
    ):
        issues.add("invalid_payload_field", point_id=point_id, field="freshness_sla_met")

    for field in ("effective_from", "effective_to", "repeal_date"):
        value = payload.get(field)
        if field not in payload or value is None:
            continue
        try:
            parse_rfc3339_utc(value, field=f"payload.{field}")
        except ValueError:
            issues.add("invalid_payload_field", point_id=point_id, field=field)


def _canonical_json_text(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"structural inventory row is not canonical JSON: {exc}") from exc


def _load_structural_inventory_rows(
    connection: sqlite3.Connection,
    rows: Iterable[Mapping[str, Any]],
    manifest: GenerationManifest,
    integrity: _Issues,
) -> tuple[int, int]:
    """Exhaust a validated inventory stream into the verifier's disposable index."""

    documents = chunks = 0
    for raw_document in rows:
        if not isinstance(raw_document, Mapping):
            raise TypeError("structural inventory must yield mapping document rows")
        try:
            source = raw_document["source"]
            document_id = raw_document["document_id"]
            version_id = raw_document["version_id"]
            content_hash = raw_document["canonical_content_sha256"]
            document_chunks = raw_document["chunks"]
            declared_count = raw_document["chunk_count"]
        except KeyError as exc:
            raise ValueError(
                f"structural inventory document lacks {exc.args[0]}"
            ) from exc
        if (
            not _is_string(source)
            or not _is_string(document_id)
            or not _is_string(version_id)
            or not _is_sha256(content_hash)
            or not _is_int(declared_count, minimum=1)
            or not isinstance(document_chunks, list)
            or len(document_chunks) != declared_count
        ):
            raise ValueError("structural inventory document identity/count is invalid")
        expected = connection.execute(
            "SELECT expected_chunks, content_hash, excluded FROM expected "
            "WHERE source=? AND document_id=? AND version_id=?",
            (source, document_id, version_id),
        ).fetchone()
        if expected is None:
            integrity.add(
                "structural_inventory_document_outside_generation",
                source=source,
                document_id=document_id,
                version_id=version_id,
            )
        else:
            expected_chunks, expected_content_hash, excluded = expected
            if excluded:
                integrity.add(
                    "excluded_document_in_structural_inventory",
                    source=source,
                    document_id=document_id,
                    version_id=version_id,
                )
            if expected_chunks != declared_count:
                integrity.add(
                    "structural_inventory_document_chunk_count_mismatch",
                    source=source,
                    document_id=document_id,
                    version_id=version_id,
                    expected=expected_chunks,
                    actual=declared_count,
                )
            if expected_content_hash != content_hash:
                integrity.add(
                    "structural_inventory_document_content_mismatch",
                    source=source,
                    document_id=document_id,
                    version_id=version_id,
                )
        try:
            connection.execute(
                "INSERT INTO inventory_documents VALUES (?, ?, ?, ?, ?, ?)",
                (
                    source,
                    document_id,
                    version_id,
                    content_hash,
                    declared_count,
                    _canonical_json_text(raw_document),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError("duplicate structural inventory document identity") from exc
        for chunk in document_chunks:
            if not isinstance(chunk, Mapping):
                raise ValueError("structural inventory chunk must be an object")
            chunk_index = chunk.get("chunk_index")
            if not _is_int(chunk_index):
                raise ValueError("structural inventory chunk index is invalid")
            try:
                connection.execute(
                    "INSERT INTO inventory_chunks VALUES (?, ?, ?, ?, ?)",
                    (
                        source,
                        document_id,
                        version_id,
                        chunk_index,
                        _canonical_json_text(chunk),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("duplicate structural inventory chunk identity") from exc
            chunks += 1
        documents += 1
        if documents % 1000 == 0:
            connection.commit()
    connection.commit()

    missing = connection.execute(
        """
        SELECT e.source, e.document_id, e.version_id
        FROM expected AS e
        LEFT JOIN inventory_documents AS i
          ON i.source=e.source AND i.document_id=e.document_id
         AND i.version_id=e.version_id
        WHERE e.excluded=0 AND i.source IS NULL
        ORDER BY e.source, e.document_id, e.version_id
        """
    )
    for source, document_id, version_id in missing:
        integrity.add(
            "generation_document_missing_structural_inventory",
            source=source,
            document_id=document_id,
            version_id=version_id,
        )
    if documents != manifest.indexed_document_count:
        integrity.add(
            "structural_inventory_document_count_mismatch",
            expected=manifest.indexed_document_count,
            actual=documents,
        )
    if chunks != manifest.chunk_count:
        integrity.add(
            "structural_inventory_chunk_count_mismatch",
            expected=manifest.chunk_count,
            actual=chunks,
        )
    return documents, chunks


def _inventory_field_matches(
    payload: Mapping[str, Any],
    field: str,
    expected: Any,
    issues: _Issues,
    *,
    point_id: str,
) -> bool:
    if field in payload and payload[field] == expected:
        return True
    issues.add(
        "structural_inventory_payload_mismatch",
        point_id=point_id,
        field=field,
        expected=expected,
        actual=payload.get(field),
    )
    return False


def _validate_structural_inventory_point(
    payload: Mapping[str, Any],
    manifest: GenerationManifest,
    document: Mapping[str, Any],
    chunk: Mapping[str, Any],
    count_tokens: Callable[[str], int],
    issues: _Issues,
    *,
    point_id: str,
) -> tuple[bool, bool]:
    """Compare a physical point to every sealed structural and encoder-input field."""

    source = document["source"]
    document_id = document["document_id"]
    version_id = document["version_id"]
    structure = chunk["structure"]
    page = chunk["page"]
    local_parent_id = structure["parent_id"]
    parent_id = (
        f"{source}:{document_id}:{version_id}:{local_parent_id}:"
        f"{structure['parent_chunk_index']}"
        if local_parent_id
        else None
    )
    passage_sha = chunk["canonical_passage_sha256"]
    start = chunk["char_start"]
    end = chunk["char_end"]
    expected_passage_id = "passage:" + hashlib.sha256(
        (
            f"{source}\0{document_id}\0{version_id}\0{start}\0{end}\0{passage_sha}"
        ).encode("utf-8")
    ).hexdigest()
    expected_payload = {
        "source": source,
        "document_id": document_id,
        "version_id": version_id,
        "chunk_index": chunk["chunk_index"],
        "document_chunk_count": chunk["document_chunk_count"],
        "content_hash": document["canonical_content_sha256"],
        "canonical_content_hash": document["canonical_content_sha256"],
        "passage_id": expected_passage_id,
        "passage_hash": passage_sha,
        "passage_content_hash": passage_sha,
        "token_count": chunk["token_count"],
        "char_start": start,
        "char_end": end,
        "offset_unit": "unicode_codepoint",
        "heading_path": structure["heading_path"],
        "heading": " > ".join(structure["heading_path"]) or None,
        "article_id": structure["article_id"],
        "article_label": structure["article_label"],
        "article_start": structure["article_start"],
        "clause": structure["clause"],
        "clause_id": structure["clause_id"] or structure["clause"],
        "clause_ids": structure["clause_ids"],
        "subarticle": structure["subarticle"],
        "subarticle_ids": structure["subarticle_ids"],
        "chapter": structure["chapter"],
        "structural_parent_id": local_parent_id,
        "parent_id": parent_id,
        "article_start_chunk_index": structure["article_start_chunk_index"],
        "parent_chunk_index": structure["parent_chunk_index"],
        "chunker_revision": structure["chunker_revision"],
        "page_start": page["page_start"],
        "page_end": page["page_end"],
        "page_coordinate_reason": page["page_coordinate_reason"],
        "page_boundary_mapping_sha256": page["page_boundary_mapping_sha256"],
        "page_boundaries": (
            document["page_boundaries"] if chunk["chunk_index"] == 0 else None
        ),
    }
    exact = True
    for field, expected in expected_payload.items():
        field_matches = _inventory_field_matches(
            payload,
            field,
            expected,
            issues,
            point_id=point_id,
        )
        exact = exact and field_matches
    text = payload.get("text")
    if not isinstance(text, str):
        return False, False
    passage_projection = {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "char_length": len(text),
        "utf8_length": len(text.encode("utf-8")),
    }
    expected_passage_projection = {
        "sha256": passage_sha,
        "char_length": chunk["canonical_passage_char_length"],
        "utf8_length": chunk["canonical_passage_utf8_length"],
    }
    if passage_projection != expected_passage_projection:
        exact = False
        issues.add(
            "structural_inventory_passage_bytes_mismatch",
            point_id=point_id,
            expected=expected_passage_projection,
            actual=passage_projection,
        )

    header_fields = ("title", "document_type", "document_number", "status")
    invalid_header_fields = [
        field
        for field in header_fields
        if payload.get(field) is not None and not isinstance(payload.get(field), str)
    ]
    if payload.get("date") is not None and not isinstance(payload.get("date"), str):
        invalid_header_fields.append("date")
    if payload.get("date_raw") is not None and not isinstance(
        payload.get("date_raw"), str
    ):
        invalid_header_fields.append("date_raw")
    if payload.get("is_consolidated") is not None and not isinstance(
        payload.get("is_consolidated"), bool
    ):
        invalid_header_fields.append("is_consolidated")
    if invalid_header_fields:
        for field in invalid_header_fields:
            issues.add(
                "structural_inventory_embed_input_field_invalid",
                point_id=point_id,
                field=field,
            )
        return False, False
    header_kwargs: dict[str, Any] = {}
    if manifest.chunking.document_header:
        header_kwargs = {
            "document_number": payload.get("document_number"),
            "date": payload.get("date") or payload.get("date_raw"),
            "status": payload.get("status"),
            "is_consolidated": payload.get("is_consolidated"),
        }
    embed_text = build_embed_text(
        text,
        title=payload.get("title"),
        document_type=payload.get("document_type"),
        heading_path=list(structure["heading_path"]),
        **header_kwargs,
    )
    observed_tokens = count_tokens(embed_text)
    if not _is_int(observed_tokens, minimum=1):
        raise ValueError("pinned tokenizer counter returned an invalid token count")
    embed_projection = {
        "sha256": hashlib.sha256(embed_text.encode("utf-8")).hexdigest(),
        "char_length": len(embed_text),
        "utf8_length": len(embed_text.encode("utf-8")),
        "token_count": observed_tokens,
    }
    expected_embed_projection = dict(chunk["embed_input"])
    embed_exact = embed_projection == expected_embed_projection
    if not embed_exact:
        exact = False
        issues.add(
            "structural_inventory_embed_input_mismatch",
            point_id=point_id,
            expected=expected_embed_projection,
            actual=embed_projection,
        )
    return exact, embed_exact


def _create_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE expected (
            source TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            expected_chunks INTEGER NOT NULL,
            state_hash TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            excluded INTEGER NOT NULL,
            complete INTEGER NOT NULL,
            content_kind TEXT NOT NULL,
            extraction_status TEXT NOT NULL,
            article_summary TEXT,
            source_binary_url TEXT,
            refresh_deadline TEXT,
            PRIMARY KEY (source, document_id, version_id)
        );
        CREATE TABLE observed (
            source TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            point_id TEXT NOT NULL UNIQUE,
            text_sha256 TEXT,
            point_sha256 TEXT,
            PRIMARY KEY (source, document_id, version_id, chunk_index)
        );
        CREATE TABLE samples (
            source TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            PRIMARY KEY (source, document_id, version_id, chunk_index)
        );
        CREATE TABLE inventory_documents (
            source TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            document_json TEXT NOT NULL,
            PRIMARY KEY (source, document_id, version_id)
        );
        CREATE TABLE inventory_chunks (
            source TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_json TEXT NOT NULL,
            PRIMARY KEY (source, document_id, version_id, chunk_index)
        );
        """
    )


def _normalize_now(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def verify_generation_points(
    manifest: GenerationManifest,
    manifest_sha256: str,
    documents: Iterable[DocumentRecord],
    sample_checks: Iterable[SampleCheck],
    points: Iterable[Any],
    *,
    now: datetime | None = None,
    max_examples: int = 20,
    temp_dir: str | Path | None = None,
    physical_collection: str | None = None,
    expected_collection_digest: CollectionDigest | None = None,
    observed_collection_configuration_sha256: str | None = None,
    verification_id: str | None = None,
    structural_inventory: Iterable[Mapping[str, Any]] | None = None,
    structural_inventory_proof: Mapping[str, Any] | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> VerificationReport:
    """Verify a point stream against a generation ledger using bounded memory."""
    if not isinstance(manifest, GenerationManifest):
        raise TypeError("manifest must be a GenerationManifest")
    manifest = GenerationManifest.from_dict(json.loads(json.dumps(manifest.to_dict())))
    if physical_collection is not None:
        expected_collection = f"georgian_legal__gen_{manifest.generation_id}"
        if physical_collection != expected_collection:
            raise ValueError(
                "verification must target the exact physical collection "
                f"{expected_collection!r}; got {physical_collection!r}"
            )
    if expected_collection_digest is not None:
        expected_collection_digest = CollectionDigest.from_dict(
            expected_collection_digest.to_dict()
        )
        if (
            expected_collection_digest.point_count != manifest.chunk_count
            or expected_collection_digest.physical_collection
            != f"georgian_legal__gen_{manifest.generation_id}"
            or expected_collection_digest.payload_projection
            != COLLECTION_PAYLOAD_PROJECTION
            or expected_collection_digest.dense_encoding
            != COLLECTION_DENSE_ENCODING
            or expected_collection_digest.sparse_encoding
            != COLLECTION_SPARSE_ENCODING
        ):
            raise ValueError("collection digest is incompatible with generation verifier")
    if not _is_sha256(manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
    if not _is_int(max_examples, minimum=1):
        raise ValueError("max_examples must be an integer >= 1")
    if (structural_inventory is None) != (structural_inventory_proof is None):
        raise ValueError(
            "structural inventory rows and proof must be supplied together"
        )
    if structural_inventory is not None and count_tokens is None:
        raise ValueError(
            "structural inventory verification requires the exact pinned token counter"
        )
    current_time = _normalize_now(now)
    coverage = _Issues(max_examples)
    integrity = _Issues(max_examples)
    freshness = _Issues(max_examples)
    quality = _Issues(max_examples)

    sqlite_directory = None if temp_dir is None else str(Path(temp_dir))
    fd, sqlite_path = tempfile.mkstemp(
        prefix="generation-integrity-",
        suffix=".sqlite3",
        dir=sqlite_directory,
    )
    os.fchmod(fd, 0o600)
    os.close(fd)

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(sqlite_path)
        connection.execute("PRAGMA journal_mode=MEMORY")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA synchronous=OFF")
        _create_tables(connection)

        covered_sources = {run.source for run in manifest.covered_runs}
        ledger_records = 0
        indexed_documents = 0
        excluded_documents = 0
        expected_chunks = 0

        for record in documents:
            if not isinstance(record, DocumentRecord):
                raise TypeError("documents must yield DocumentRecord instances")
            record = DocumentRecord.from_dict(
                record.to_dict(),
                expected_generation_id=manifest.generation_id,
            )
            ledger_records += 1
            if record.source not in covered_sources:
                integrity.add(
                    "document_source_without_covered_run",
                    source=record.source,
                    document_id=record.document_id,
                )
            try:
                connection.execute(
                    """
                    INSERT INTO expected (
                        source, document_id, version_id, expected_chunks, state_hash,
                        content_hash, excluded, complete, content_kind,
                        extraction_status, article_summary, source_binary_url,
                        refresh_deadline
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.source,
                        record.document_id,
                        record.version_id,
                        record.expected_chunk_count,
                        record.document_state_hash,
                        record.content_hash,
                        int(not record.indexed),
                        int(record.content_complete),
                        record.content_kind,
                        record.extraction_status,
                        record.article_summary,
                        record.source_binary_url,
                        record.refresh_deadline,
                    ),
                )
            except sqlite3.IntegrityError:
                integrity.add(
                    "duplicate_document_record",
                    source=record.source,
                    document_id=record.document_id,
                )
                continue

            if record.indexed:
                indexed_documents += 1
                expected_chunks += record.expected_chunk_count
            else:
                excluded_documents += 1
            if not record.content_complete:
                quality.add(
                    "incomplete_document",
                    source=record.source,
                    document_id=record.document_id,
                    exclusion_reason=record.exclusion_reason,
                )
            if record.refresh_deadline is not None:
                deadline = parse_rfc3339_utc(
                    record.refresh_deadline,
                    field="document.refresh_deadline",
                )
                if deadline <= current_time:
                    freshness.add(
                        "refresh_deadline_expired",
                        source=record.source,
                        document_id=record.document_id,
                        refresh_deadline=record.refresh_deadline,
                    )
            if ledger_records % 1000 == 0:
                connection.commit()
        connection.commit()

        if ledger_records != manifest.document_count:
            integrity.add(
                "manifest_document_count_mismatch",
                expected=manifest.document_count,
                actual=ledger_records,
            )
        if indexed_documents != manifest.indexed_document_count:
            integrity.add(
                "manifest_indexed_document_count_mismatch",
                expected=manifest.indexed_document_count,
                actual=indexed_documents,
            )
        if excluded_documents != manifest.excluded_document_count:
            integrity.add(
                "manifest_excluded_document_count_mismatch",
                expected=manifest.excluded_document_count,
                actual=excluded_documents,
            )
        if expected_chunks != manifest.chunk_count:
            integrity.add(
                "manifest_chunk_count_mismatch",
                expected=manifest.chunk_count,
                actual=expected_chunks,
            )

        inventory_documents = inventory_chunks = 0
        if structural_inventory is not None:
            assert structural_inventory_proof is not None
            inventory_documents, inventory_chunks = _load_structural_inventory_rows(
                connection,
                structural_inventory,
                manifest,
                integrity,
            )
            if (
                structural_inventory_proof.get("expected_document_count")
                != inventory_documents
                or structural_inventory_proof.get("expected_chunk_count")
                != inventory_chunks
                or not _is_sha256(
                    structural_inventory_proof.get("structural_inventory_sha256")
                )
                or not _is_sha256(
                    structural_inventory_proof.get(
                        "structural_inventory_identity_sha256"
                    )
                )
            ):
                raise ValueError(
                    "exhausted structural inventory differs from its bound proof"
                )

        from .qdrant_store import point_id as expected_point_id

        observed_points = 0
        inventory_rows_found = 0
        inventory_exact_matches = 0
        embed_inputs_verified = 0
        for point in points:
            observed_points += 1
            raw_point_id = _point_field(point, "id")
            point_id = _canonical_point_id(raw_point_id)
            if point_id is None:
                integrity.add(
                    "invalid_point_id",
                    point_id=str(raw_point_id),
                )
                storage_point_id = f"invalid:{observed_points}"
                issue_point_id = str(raw_point_id)
            else:
                storage_point_id = point_id
                issue_point_id = point_id

            payload = _point_field(point, "payload")
            vectors = _point_field(point, "vector")
            _validate_vectors(
                vectors,
                manifest,
                integrity,
                point_id=issue_point_id,
            )
            if not isinstance(payload, Mapping):
                integrity.add(
                    "missing_or_invalid_payload",
                    point_id=issue_point_id,
                )
                continue
            point_sha256 = None
            if point_id is not None:
                try:
                    point_sha256 = point_content_sha256(
                        point_id,
                        payload,
                        vectors,
                        dense_name=manifest.vector_space.dense_name,
                        sparse_name=manifest.vector_space.sparse_name,
                    )
                except ValueError as exc:
                    integrity.add(
                        "point_digest_invalid",
                        point_id=issue_point_id,
                        error=str(exc),
                    )

            source = payload.get("source")
            document_id = payload.get("document_id")
            version_id = payload.get("version_id")
            chunk_index = payload.get("chunk_index")
            key_is_valid = True
            if not _is_string(source):
                integrity.add(
                    "invalid_payload_field",
                    point_id=issue_point_id,
                    field="source",
                )
                key_is_valid = False
            if not _is_string(document_id):
                integrity.add(
                    "invalid_payload_field",
                    point_id=issue_point_id,
                    field="document_id",
                )
                key_is_valid = False
            if not _is_string(version_id):
                integrity.add(
                    "invalid_payload_field",
                    point_id=issue_point_id,
                    field="version_id",
                )
                key_is_valid = False
            if not _is_int(chunk_index):
                integrity.add(
                    "invalid_payload_field",
                    point_id=issue_point_id,
                    field="chunk_index",
                )
                key_is_valid = False

            _expect_payload_value(
                payload,
                "schema_version",
                manifest.schema_version,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "generation_id",
                manifest.generation_id,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "embedding_model",
                manifest.model.embedding_model,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "embedding_revision",
                manifest.model.embedding_revision,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "tokenizer_model",
                manifest.model.tokenizer_model,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "tokenizer_revision",
                manifest.model.tokenizer_revision,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "reranker_model",
                manifest.model.reranker_model,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "reranker_revision",
                manifest.model.reranker_revision,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "vector_space_id",
                manifest.vector_space.id,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "chunking_fingerprint",
                manifest.chunking.fingerprint,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "document_header",
                manifest.chunking.document_header,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "retrieval_fingerprint_revision",
                manifest.retrieval_fingerprint_revision,
                integrity,
                point_id=issue_point_id,
            )
            _expect_payload_value(
                payload,
                "retrieval_fingerprint",
                manifest.retrieval_fingerprint,
                integrity,
                point_id=issue_point_id,
            )
            if "schema_version" in payload and not _is_int(
                payload["schema_version"], minimum=1
            ):
                integrity.add(
                    "invalid_payload_field",
                    point_id=issue_point_id,
                    field="schema_version",
                )
            for string_field in (
                "generation_id",
                "embedding_model",
                "embedding_revision",
                "tokenizer_model",
                "tokenizer_revision",
                "reranker_model",
                "reranker_revision",
                "vector_space_id",
                "chunking_fingerprint",
                "retrieval_fingerprint",
            ):
                if string_field in payload and not _is_string(payload[string_field]):
                    integrity.add(
                        "invalid_payload_field",
                        point_id=issue_point_id,
                        field=string_field,
                    )
            if "document_header" in payload and not isinstance(
                payload["document_header"], bool
            ):
                integrity.add(
                    "invalid_payload_field",
                    point_id=issue_point_id,
                    field="document_header",
                )
            for hash_field in ("content_hash", "document_state_hash"):
                if hash_field not in payload:
                    integrity.add(
                        "missing_payload_field",
                        point_id=issue_point_id,
                        field=hash_field,
                    )
                elif not _is_sha256(payload[hash_field]):
                    integrity.add(
                        "invalid_payload_hash",
                        point_id=issue_point_id,
                        field=hash_field,
                    )
            for lineage_field in ("content_kind", "extraction_status"):
                if lineage_field not in payload:
                    integrity.add(
                        "missing_payload_field",
                        point_id=issue_point_id,
                        field=lineage_field,
                    )
                elif not _is_string(payload[lineage_field]):
                    integrity.add(
                        "invalid_payload_field",
                        point_id=issue_point_id,
                        field=lineage_field,
                    )
            if "content_complete" not in payload:
                integrity.add(
                    "missing_payload_field",
                    point_id=issue_point_id,
                    field="content_complete",
                )
            elif not isinstance(payload["content_complete"], bool):
                integrity.add(
                    "invalid_payload_field",
                    point_id=issue_point_id,
                    field="content_complete",
                )
            for optional_lineage_field in ("article_summary", "source_binary_url"):
                if optional_lineage_field not in payload:
                    integrity.add(
                        "missing_payload_field",
                        point_id=issue_point_id,
                        field=optional_lineage_field,
                    )
                elif payload[optional_lineage_field] is not None and not _is_string(
                    payload[optional_lineage_field]
                ):
                    integrity.add(
                        "invalid_payload_field",
                        point_id=issue_point_id,
                        field=optional_lineage_field,
                    )
            if "document_chunk_count" not in payload:
                integrity.add(
                    "missing_payload_field",
                    point_id=issue_point_id,
                    field="document_chunk_count",
                )
            elif not _is_int(payload["document_chunk_count"], minimum=1):
                integrity.add(
                    "invalid_payload_field",
                    point_id=issue_point_id,
                    field="document_chunk_count",
                )
            text = payload.get("text")
            if not isinstance(text, str):
                integrity.add(
                    "invalid_payload_field",
                    point_id=issue_point_id,
                    field="text",
                )
                text_sha256 = None
            else:
                text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

            _validate_canonical_payload(
                payload,
                manifest,
                integrity,
                point_id=issue_point_id,
                text=text if isinstance(text, str) else None,
            )

            if not key_is_valid:
                continue
            assert isinstance(source, str)
            assert isinstance(document_id, str)
            assert isinstance(version_id, str)
            assert isinstance(chunk_index, int)
            deterministic_point_id = expected_point_id(
                source,
                document_id,
                chunk_index,
                version_id=version_id,
            )
            point_identity_exact = point_id == deterministic_point_id
            if point_id is not None and not point_identity_exact:
                integrity.add(
                    "version_scoped_point_id_mismatch",
                    source=source,
                    document_id=document_id,
                    version_id=version_id,
                    chunk_index=chunk_index,
                    expected=deterministic_point_id,
                    actual=point_id,
                )
            if structural_inventory is not None:
                inventory_row = connection.execute(
                    """
                    SELECT c.chunk_json, d.document_json
                    FROM inventory_chunks AS c
                    JOIN inventory_documents AS d
                      ON d.source=c.source AND d.document_id=c.document_id
                     AND d.version_id=c.version_id
                    WHERE c.source=? AND c.document_id=? AND c.version_id=?
                      AND c.chunk_index=?
                    """,
                    (source, document_id, version_id, chunk_index),
                ).fetchone()
                if inventory_row is None:
                    integrity.add(
                        "point_outside_structural_inventory",
                        point_id=issue_point_id,
                        source=source,
                        document_id=document_id,
                        version_id=version_id,
                        chunk_index=chunk_index,
                    )
                else:
                    inventory_rows_found += 1
                    chunk_value = json.loads(inventory_row[0])
                    document_value = json.loads(inventory_row[1])
                    assert count_tokens is not None
                    exact_match, embed_match = _validate_structural_inventory_point(
                        payload,
                        manifest,
                        document_value,
                        chunk_value,
                        count_tokens,
                        integrity,
                        point_id=issue_point_id,
                    )
                    inventory_exact_matches += int(
                        exact_match and point_identity_exact
                    )
                    embed_inputs_verified += int(embed_match)
            expected = connection.execute(
                """
                SELECT expected_chunks, state_hash, content_hash, excluded,
                       complete, content_kind, extraction_status,
                       article_summary, source_binary_url
                FROM expected
                WHERE source = ? AND document_id = ? AND version_id = ?
                """,
                (source, document_id, version_id),
            ).fetchone()
            if expected is None:
                integrity.add(
                    "unexpected_point",
                    source=source,
                    document_id=document_id,
                    chunk_index=chunk_index,
                )
            else:
                (
                    expected_count,
                    state_hash,
                    content_hash,
                    excluded,
                    complete,
                    content_kind,
                    extraction_status,
                    article_summary,
                    source_binary_url,
                ) = expected
                if excluded:
                    integrity.add(
                        "point_for_excluded_document",
                        source=source,
                        document_id=document_id,
                        chunk_index=chunk_index,
                    )
                if payload.get("document_chunk_count") != expected_count:
                    integrity.add(
                        "document_chunk_count_mismatch",
                        source=source,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        expected=expected_count,
                        actual=payload.get("document_chunk_count"),
                    )
                if payload.get("document_state_hash") != state_hash:
                    integrity.add(
                        "document_state_hash_mismatch",
                        source=source,
                        document_id=document_id,
                        chunk_index=chunk_index,
                    )
                if payload.get("content_hash") != content_hash:
                    integrity.add(
                        "content_hash_mismatch",
                        source=source,
                        document_id=document_id,
                        chunk_index=chunk_index,
                    )
                for field, value in (
                    ("content_complete", bool(complete)),
                    ("content_kind", content_kind),
                    ("extraction_status", extraction_status),
                    ("article_summary", article_summary),
                    ("source_binary_url", source_binary_url),
                ):
                    if payload.get(field) != value:
                        integrity.add(
                            "document_lineage_mismatch",
                            source=source,
                            document_id=document_id,
                            chunk_index=chunk_index,
                            field=field,
                            expected=value,
                            actual=payload.get(field),
                        )

            try:
                connection.execute(
                    """
                    INSERT INTO observed (
                        source, document_id, version_id, chunk_index, point_id,
                        text_sha256, point_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source,
                        document_id,
                        version_id,
                        chunk_index,
                        storage_point_id,
                        text_sha256,
                        point_sha256,
                    ),
                )
            except sqlite3.IntegrityError:
                existing_key = connection.execute(
                    """
                    SELECT 1 FROM observed
                    WHERE source = ? AND document_id = ? AND version_id = ?
                      AND chunk_index = ?
                    """,
                    (source, document_id, version_id, chunk_index),
                ).fetchone()
                code = (
                    "duplicate_logical_chunk"
                    if existing_key is not None
                    else "point_id_collision"
                )
                integrity.add(
                    code,
                    source=source,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    point_id=issue_point_id,
                )
            if observed_points % 1000 == 0:
                connection.commit()
        connection.commit()

        rows = connection.execute(
            """
            SELECT
                e.source,
                e.document_id,
                e.version_id,
                e.expected_chunks,
                e.excluded,
                COUNT(o.chunk_index),
                MIN(o.chunk_index),
                MAX(o.chunk_index)
            FROM expected AS e
            LEFT JOIN observed AS o
              ON o.source = e.source AND o.document_id = e.document_id
             AND o.version_id = e.version_id
            GROUP BY e.source, e.document_id, e.version_id
            ORDER BY e.source, e.document_id, e.version_id
            """
        )
        for (
            source,
            document_id,
            version_id,
            expected_count,
            excluded,
            actual_count,
            minimum_index,
            maximum_index,
        ) in rows:
            if excluded:
                if actual_count:
                    integrity.add(
                        "excluded_document_has_points",
                        source=source,
                        document_id=document_id,
                        actual=actual_count,
                    )
                continue
            if actual_count != expected_count:
                coverage.add(
                    "missing_document" if actual_count == 0 else "chunk_count_mismatch",
                    source=source,
                    document_id=document_id,
                    expected=expected_count,
                    actual=actual_count,
                )
            if actual_count and (
                minimum_index != 0
                or maximum_index != expected_count - 1
                or actual_count != expected_count
            ):
                integrity.add(
                    "non_contiguous_chunk_indexes",
                    source=source,
                    document_id=document_id,
                    expected=expected_count,
                    actual=actual_count,
                    minimum=minimum_index,
                    maximum=maximum_index,
                )

        unique_observed = connection.execute(
            "SELECT COUNT(*) FROM observed"
        ).fetchone()[0]
        observed_indexed_documents = connection.execute(
            """
            SELECT COUNT(DISTINCT o.source || char(0) || o.document_id || char(0) || o.version_id)
            FROM observed AS o
            JOIN expected AS e
              ON e.source = o.source AND e.document_id = o.document_id
             AND e.version_id = o.version_id
            WHERE e.excluded = 0
            """
        ).fetchone()[0]
        unexpected_points = connection.execute(
            """
            SELECT COUNT(*)
            FROM observed AS o
            LEFT JOIN expected AS e
              ON e.source = o.source AND e.document_id = o.document_id
             AND e.version_id = o.version_id
            WHERE e.source IS NULL
            """
        ).fetchone()[0]
        if unique_observed != manifest.chunk_count:
            coverage.add(
                "generation_chunk_count_mismatch",
                expected=manifest.chunk_count,
                actual=unique_observed,
            )
        if observed_indexed_documents != manifest.indexed_document_count:
            coverage.add(
                "generation_indexed_document_count_mismatch",
                expected=manifest.indexed_document_count,
                actual=observed_indexed_documents,
            )
        if structural_inventory is not None:
            missing_inventory_points = connection.execute(
                """
                SELECT i.source, i.document_id, i.version_id, i.chunk_index
                FROM inventory_chunks AS i
                LEFT JOIN observed AS o
                  ON o.source=i.source AND o.document_id=i.document_id
                 AND o.version_id=i.version_id AND o.chunk_index=i.chunk_index
                WHERE o.source IS NULL
                ORDER BY i.source, i.document_id, i.version_id, i.chunk_index
                """
            )
            for source, document_id, version_id, chunk_index in missing_inventory_points:
                coverage.add(
                    "structural_inventory_point_missing",
                    source=source,
                    document_id=document_id,
                    version_id=version_id,
                    chunk_index=chunk_index,
                )

        sample_count = 0
        for sample in sample_checks:
            if not isinstance(sample, SampleCheck):
                raise TypeError("sample_checks must yield SampleCheck instances")
            sample = SampleCheck.from_dict(
                sample.to_dict(),
                expected_generation_id=manifest.generation_id,
            )
            sample_count += 1
            try:
                connection.execute(
                    """
                    INSERT INTO samples (source, document_id, version_id, chunk_index)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        sample.source,
                        sample.document_id,
                        sample.version_id,
                        sample.chunk_index,
                    ),
                )
            except sqlite3.IntegrityError:
                integrity.add(
                    "duplicate_sample_check",
                    source=sample.source,
                    document_id=sample.document_id,
                    chunk_index=sample.chunk_index,
                )
                continue
            expected = connection.execute(
                """
                SELECT expected_chunks, excluded
                FROM expected
                WHERE source = ? AND document_id = ? AND version_id = ?
                """,
                (sample.source, sample.document_id, sample.version_id),
            ).fetchone()
            if expected is None or expected[1] or sample.chunk_index >= expected[0]:
                integrity.add(
                    "sample_outside_ledger",
                    source=sample.source,
                    document_id=sample.document_id,
                    chunk_index=sample.chunk_index,
                )
                continue
            observed = connection.execute(
                """
                SELECT point_id, text_sha256
                FROM observed
                WHERE source = ? AND document_id = ? AND version_id = ?
                  AND chunk_index = ?
                """,
                (
                    sample.source,
                    sample.document_id,
                    sample.version_id,
                    sample.chunk_index,
                ),
            ).fetchone()
            if observed is None:
                coverage.add(
                    "sample_point_missing",
                    source=sample.source,
                    document_id=sample.document_id,
                    chunk_index=sample.chunk_index,
                )
                continue
            observed_point_id, observed_text_sha256 = observed
            if observed_point_id != sample.point_id:
                integrity.add(
                    "sample_point_id_mismatch",
                    source=sample.source,
                    document_id=sample.document_id,
                    chunk_index=sample.chunk_index,
                    expected=sample.point_id,
                    actual=observed_point_id,
                )
            if observed_text_sha256 != sample.text_sha256:
                integrity.add(
                    "sample_text_hash_mismatch",
                    source=sample.source,
                    document_id=sample.document_id,
                    chunk_index=sample.chunk_index,
                )
        connection.commit()
        if sample_count != manifest.sample_count:
            integrity.add(
                "manifest_sample_count_mismatch",
                expected=manifest.sample_count,
                actual=sample_count,
            )

        observed_collection_sha256: str | None = None
        digested_points = 0
        try:
            observed_collection_sha256, digested_points = whole_collection_sha256(
                (str(point_id), str(point_sha))
                for point_id, point_sha in connection.execute(
                    "SELECT point_id, point_sha256 FROM observed "
                    "WHERE point_sha256 IS NOT NULL ORDER BY point_id"
                )
            )
        except ValueError as exc:
            integrity.add("whole_collection_digest_invalid", error=str(exc))
        if digested_points != observed_points:
            integrity.add(
                "whole_collection_digest_point_count_mismatch",
                expected=observed_points,
                actual=digested_points,
            )
        if expected_collection_digest is not None:
            if (
                observed_collection_sha256
                != expected_collection_digest.collection_sha256
            ):
                integrity.add(
                    "whole_collection_digest_mismatch",
                    expected=expected_collection_digest.collection_sha256,
                    actual=observed_collection_sha256,
                )
            if observed_collection_configuration_sha256 is None:
                integrity.add("collection_configuration_digest_unavailable")
            elif (
                observed_collection_configuration_sha256
                != expected_collection_digest.collection_configuration_sha256
            ):
                integrity.add(
                    "collection_configuration_digest_mismatch",
                    expected=(
                        expected_collection_digest.collection_configuration_sha256
                    ),
                    actual=observed_collection_configuration_sha256,
                )

        stats = {
            "ledger_documents": ledger_records,
            "ledger_indexed_documents": indexed_documents,
            "ledger_excluded_documents": excluded_documents,
            "expected_chunks": expected_chunks,
            "observed_points": observed_points,
            "observed_unique_chunks": unique_observed,
            "observed_indexed_documents": observed_indexed_documents,
            "unaccounted_points": unexpected_points,
            "sample_checks": sample_count,
            "digested_points": digested_points,
        }
        if structural_inventory is not None:
            assert structural_inventory_proof is not None
            stats["structural_inventory_proof"] = {
                **dict(structural_inventory_proof),
                "artifact_exhausted": True,
                "document_rows": inventory_documents,
                "chunk_rows": inventory_chunks,
                "point_rows_found": inventory_rows_found,
                "exact_point_matches": inventory_exact_matches,
                "embed_inputs_verified": embed_inputs_verified,
            }
        covered_runs = tuple(
            {"source": run.source, "run_id": run.run_id}
            for run in manifest.covered_runs
        )
        return VerificationReport(
            generation_id=manifest.generation_id,
            manifest_sha256=manifest_sha256,
            physical_collection=physical_collection,
            verified_at=_format_utc(current_time),
            verification_id=verification_id,
            covered_runs=covered_runs,
            stats=stats,
            coverage=coverage.outcome(),
            integrity=integrity.outcome(),
            freshness=freshness.outcome(),
            quality=quality.outcome(),
            expected_collection_sha256=(
                expected_collection_digest.collection_sha256
                if expected_collection_digest is not None
                else None
            ),
            observed_collection_sha256=observed_collection_sha256,
            expected_collection_configuration_sha256=(
                expected_collection_digest.collection_configuration_sha256
                if expected_collection_digest is not None
                else None
            ),
            observed_collection_configuration_sha256=(
                observed_collection_configuration_sha256
            ),
            vector_checksum_artifact_sha256=(
                expected_collection_digest.vector_checksum_artifact_sha256
                if expected_collection_digest is not None
                else None
            ),
            vector_probe_sha256=(
                expected_collection_digest.vector_probe_sha256
                if expected_collection_digest is not None
                else None
            ),
        )
    finally:
        if connection is not None:
            connection.close()
        try:
            os.unlink(sqlite_path)
        except FileNotFoundError:
            pass


def verify_generation_artifacts(
    artifacts: GenerationArtifacts,
    points: Iterable[Any],
    *,
    snapshot_root: str | Path | None = None,
    count_tokens: Callable[[str], int] | None = None,
    now: datetime | None = None,
    max_examples: int = 20,
    temp_dir: str | Path | None = None,
    physical_collection: str | None = None,
    observed_collection_configuration_sha256: str | None = None,
    verification_id: str | None = None,
) -> VerificationReport:
    """Verify a checksum-validated artifact bundle against a point stream."""
    if not isinstance(artifacts, GenerationArtifacts):
        raise TypeError("artifacts must be GenerationArtifacts")
    if artifacts.collection_digest is None:
        raise ValueError(
            "generation lacks sealed whole-collection digest artifact"
        )
    binding: StructuralInventoryBinding | None = None
    if snapshot_root is None:
        structural_provenance = _generation_structural_provenance(
            artifacts,
            required=artifacts.manifest.generation_id
            == FROZEN_CANDIDATE_GENERATION_ID,
        )
        if structural_provenance is not None:
            raise ValueError(
                "generation with structural inventory provenance requires explicit snapshot_root"
            )
    else:
        if count_tokens is None:
            raise ValueError(
                "physical structural verification requires an injected offline pinned "
                "token counter"
            )
        binding = load_structural_inventory_binding(artifacts, snapshot_root)
    return verify_generation_points(
        artifacts.manifest,
        artifacts.checksums.files[MANIFEST_FILENAME],
        artifacts.iter_documents(),
        artifacts.iter_samples(),
        points,
        now=now,
        max_examples=max_examples,
        temp_dir=temp_dir,
        physical_collection=physical_collection,
        expected_collection_digest=artifacts.collection_digest,
        observed_collection_configuration_sha256=(
            observed_collection_configuration_sha256
        ),
        verification_id=verification_id,
        structural_inventory=(binding.iter_documents() if binding is not None else None),
        structural_inventory_proof=(binding.proof if binding is not None else None),
        count_tokens=count_tokens,
    )


def write_verification_report(
    path: str | Path,
    report: VerificationReport,
) -> Path:
    """Create one owner-only verification report; an existing path is immutable."""
    if not isinstance(report, VerificationReport):
        raise TypeError("report must be a VerificationReport")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = (
        json.dumps(
            report.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(destination, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(
            f"verification report already exists and will not be replaced: {destination}"
        ) from exc
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(destination)
        except FileNotFoundError:
            pass
        raise
    return destination
