"""Streaming, fail-closed verification for immutable Qdrant generations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

from .generation import (
    CANONICAL_PAYLOAD_REQUIRED_FIELDS,
    CANONICAL_PAYLOAD_REVISION,
    GENERATION_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    DocumentRecord,
    GenerationArtifacts,
    GenerationManifest,
    SampleCheck,
    parse_rfc3339_utc,
)

_SHA256_LENGTH = 64


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
    stats: Mapping[str, int]
    coverage: VerificationOutcome
    integrity: VerificationOutcome
    freshness: VerificationOutcome
    quality: VerificationOutcome
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
            "ok": self.ok,
            "covered_runs": [dict(run) for run in self.covered_runs],
            "stats": dict(self.stats),
            "coverage": self.coverage.to_dict(),
            "integrity": self.integrity.to_dict(),
            "freshness": self.freshness.to_dict(),
            "quality": self.quality.to_dict(),
        }


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
            PRIMARY KEY (source, document_id, version_id, chunk_index)
        );
        CREATE TABLE samples (
            source TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
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
    if not _is_sha256(manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
    if not _is_int(max_examples, minimum=1):
        raise ValueError("max_examples must be an integer >= 1")
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

        from .qdrant_store import point_id as expected_point_id

        observed_points = 0
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
            if point_id is not None and point_id != deterministic_point_id:
                integrity.add(
                    "version_scoped_point_id_mismatch",
                    source=source,
                    document_id=document_id,
                    version_id=version_id,
                    chunk_index=chunk_index,
                    expected=deterministic_point_id,
                    actual=point_id,
                )
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
                        source, document_id, version_id, chunk_index, point_id, text_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source,
                        document_id,
                        version_id,
                        chunk_index,
                        storage_point_id,
                        text_sha256,
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
            covered_runs=covered_runs,
            stats=stats,
            coverage=coverage.outcome(),
            integrity=integrity.outcome(),
            freshness=freshness.outcome(),
            quality=quality.outcome(),
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
    now: datetime | None = None,
    max_examples: int = 20,
    temp_dir: str | Path | None = None,
    physical_collection: str | None = None,
) -> VerificationReport:
    """Verify a checksum-validated artifact bundle against a point stream."""
    if not isinstance(artifacts, GenerationArtifacts):
        raise TypeError("artifacts must be GenerationArtifacts")
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
    )


def write_verification_report(
    path: str | Path,
    report: VerificationReport,
) -> Path:
    """Atomically write an owner-only verification report beside a generation."""
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

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        os.chmod(destination, 0o600)
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
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination
