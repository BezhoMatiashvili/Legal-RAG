"""Strict, versioned generation artifacts for immutable legal-search corpora.

The format deliberately uses small JSON/JSONL records and the Python standard
library. Loading is fail-closed: unknown keys, duplicate JSON keys, mutable
model revisions, unsafe paths, and checksum drift are errors.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .config import RETRIEVAL_FINGERPRINT_REVISION

GENERATION_SCHEMA_VERSION = 2
# Schema v2 is the first generation format whose Qdrant points are valid canonical legal
# evidence rather than retrieval-only chunks.  The revision marker is repeated on every
# point and included in startup compatibility counts.
CANONICAL_PAYLOAD_REVISION = "canonical-evidence-v2"
CANONICAL_PAYLOAD_REQUIRED_FIELDS = frozenset(
    {
        "canonical_payload_revision",
        "canonical_content_hash",
        "canonical_text_exact",
        "passage_id",
        "passage_hash",
        "source_fingerprint",
        "normalizer_revision",
        "chunker_revision",
        "model_revision",
        "article_id",
        "clause_id",
        "subarticle",
        "chapter",
        "heading_path",
        "parent_id",
        "article_start_chunk_index",
        "parent_chunk_index",
        "char_start",
        "char_end",
        "page_start",
        "page_end",
        "version_id",
        "supersedes",
        "effective_from",
        "effective_to",
        "repeal_date",
        "consolidation_status",
        "version_lineage_status",
        "version_lineage_complete",
        "official_url",
        "official_binary_url",
        "source_authority",
        "freshness_sla_met",
    }
)
# These fields are never legitimately null/empty on an indexed canonical passage.  Startup
# compatibility uses Qdrant ``is_empty`` exclusions so deleting any one makes the exact
# identity count fall below the manifest chunk count before the collection can serve.
CANONICAL_PAYLOAD_REQUIRED_NONEMPTY_FIELDS = frozenset(
    {
        "canonical_payload_revision",
        "canonical_content_hash",
        "passage_id",
        "passage_hash",
        "source_fingerprint",
        "normalizer_revision",
        "chunker_revision",
        "model_revision",
        "char_start",
        "char_end",
        "version_id",
        "official_url",
        "source_authority",
    }
)
CHECKSUM_ALGORITHM = "sha256"
CHECKSUM_FILENAME = "checksums.json"
MANIFEST_FILENAME = "manifest.json"
DOCUMENTS_FILENAME = "documents.jsonl"
SAMPLE_CHECKS_FILENAME = "sample_checks.jsonl"
MAX_METADATA_LINE_BYTES = 1_000_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_GENERATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{7,127}$")
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DISTANCES = frozenset({"cosine", "dot", "euclid", "manhattan"})
_EXTRACTION_STATUSES = frozenset(
    {
        "full_text",
        "scanned_no_text",
        "truncated",
        "malformed",
        "resource_limited",
    }
)


class GenerationFormatError(ValueError):
    """A generation artifact is malformed, ambiguous, or incompatible."""


class ChecksumMismatchError(GenerationFormatError):
    """A generation artifact inventory or digest does not match the files."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GenerationFormatError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(text: str, *, origin: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except GenerationFormatError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise GenerationFormatError(f"{origin}: invalid JSON: {exc}") from exc


def _regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise GenerationFormatError(f"cannot stat {path}: {exc}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise GenerationFormatError(f"expected a regular non-symlink file: {path}")


def _load_json(path: str | Path) -> Any:
    file_path = Path(path)
    _regular_file(file_path)
    try:
        raw = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GenerationFormatError(f"cannot read {file_path}: {exc}") from exc
    return _parse_json(raw, origin=str(file_path))


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GenerationFormatError(f"{field} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"unknown={extra}")
        raise GenerationFormatError(f"{field} has invalid keys ({', '.join(parts)})")


def _string(value: Any, *, field: str, max_length: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value) > max_length:
        raise GenerationFormatError(
            f"{field} must be a non-empty string of at most {max_length} characters"
        )
    if any(ord(char) < 32 for char in value):
        raise GenerationFormatError(f"{field} contains control characters")
    return value


def _model_name(value: Any, *, field: str) -> str:
    model = _string(value, field=field)
    if not model.strip():
        raise GenerationFormatError(f"{field} must be a non-empty model identity")
    return model


def _optional_string(value: Any, *, field: str, max_length: int = 4096) -> str | None:
    if value is None:
        return None
    return _string(value, field=field, max_length=max_length)


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GenerationFormatError(f"{field} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise GenerationFormatError(f"{field} must be a boolean")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise GenerationFormatError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _optional_sha256(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, field=field)


def _revision(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
        raise GenerationFormatError(
            f"{field} must be an immutable lowercase hexadecimal revision"
        )
    return value


def validate_generation_id(value: Any) -> str:
    """Validate a generation ID before it is used in a path or collection name."""
    if not isinstance(value, str) or not _GENERATION_ID_RE.fullmatch(value):
        raise GenerationFormatError(
            "generation_id must be 8-128 lowercase letters, digits, underscores, or hyphens"
        )
    if value in {"legacy", "snapshot_v1"} or value.startswith("v1"):
        raise GenerationFormatError(
            "legacy/v1 identifiers are not valid generation IDs"
        )
    return value


def parse_rfc3339_utc(value: Any, *, field: str) -> datetime:
    """Parse the canonical UTC timestamp form used by generation artifacts."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise GenerationFormatError(
            f"{field} must be an RFC 3339 UTC timestamp ending in Z"
        )
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise GenerationFormatError(
            f"{field} is not a valid RFC 3339 timestamp"
        ) from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise GenerationFormatError(f"{field} must use UTC")
    return parsed


@dataclass(frozen=True)
class CorpusIdentity:
    name: str
    snapshot_sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> CorpusIdentity:
        data = _object(value, field="corpus")
        _exact_keys(data, {"name", "snapshot_sha256"}, field="corpus")
        return cls(
            name=_string(data["name"], field="corpus.name"),
            snapshot_sha256=_sha256(
                data["snapshot_sha256"], field="corpus.snapshot_sha256"
            ),
        )


@dataclass(frozen=True)
class SourceIdentity:
    name: str
    state_sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> SourceIdentity:
        data = _object(value, field="source")
        _exact_keys(data, {"name", "state_sha256"}, field="source")
        return cls(
            name=_string(data["name"], field="source.name"),
            state_sha256=_sha256(data["state_sha256"], field="source.state_sha256"),
        )


@dataclass(frozen=True)
class ModelIdentity:
    embedding_model: str
    embedding_revision: str
    tokenizer_model: str
    tokenizer_revision: str
    reranker_model: str
    reranker_revision: str

    @classmethod
    def from_dict(cls, value: Any) -> ModelIdentity:
        data = _object(value, field="model")
        _exact_keys(
            data,
            {
                "embedding_model",
                "embedding_revision",
                "tokenizer_model",
                "tokenizer_revision",
                "reranker_model",
                "reranker_revision",
            },
            field="model",
        )
        return cls(
            embedding_model=_model_name(
                data["embedding_model"], field="model.embedding_model"
            ),
            embedding_revision=_revision(
                data["embedding_revision"], field="model.embedding_revision"
            ),
            tokenizer_model=_model_name(
                data["tokenizer_model"], field="model.tokenizer_model"
            ),
            tokenizer_revision=_revision(
                data["tokenizer_revision"], field="model.tokenizer_revision"
            ),
            reranker_model=_model_name(
                data["reranker_model"], field="model.reranker_model"
            ),
            reranker_revision=_revision(
                data["reranker_revision"], field="model.reranker_revision"
            ),
        )


@dataclass(frozen=True)
class VectorSpaceIdentity:
    id: str
    dense_name: str
    dense_dimension: int
    distance: str
    sparse_name: str

    @classmethod
    def from_dict(cls, value: Any) -> VectorSpaceIdentity:
        data = _object(value, field="vector_space")
        _exact_keys(
            data,
            {"id", "dense_name", "dense_dimension", "distance", "sparse_name"},
            field="vector_space",
        )
        distance = _string(data["distance"], field="vector_space.distance").lower()
        if distance not in _DISTANCES:
            raise GenerationFormatError(
                f"vector_space.distance must be one of {sorted(_DISTANCES)}"
            )
        dense_name = _string(data["dense_name"], field="vector_space.dense_name")
        sparse_name = _string(data["sparse_name"], field="vector_space.sparse_name")
        if dense_name == sparse_name:
            raise GenerationFormatError("dense and sparse vector names must differ")
        return cls(
            id=_sha256(data["id"], field="vector_space.id"),
            dense_name=dense_name,
            dense_dimension=_integer(
                data["dense_dimension"], field="vector_space.dense_dimension", minimum=1
            ),
            distance=distance,
            sparse_name=sparse_name,
        )


@dataclass(frozen=True)
class ChunkingIdentity:
    fingerprint: str
    max_tokens: int
    overlap_tokens: int
    document_header: bool

    @classmethod
    def from_dict(cls, value: Any) -> ChunkingIdentity:
        data = _object(value, field="chunking")
        _exact_keys(
            data,
            {"fingerprint", "max_tokens", "overlap_tokens", "document_header"},
            field="chunking",
        )
        max_tokens = _integer(
            data["max_tokens"], field="chunking.max_tokens", minimum=1
        )
        overlap_tokens = _integer(
            data["overlap_tokens"], field="chunking.overlap_tokens"
        )
        if overlap_tokens >= max_tokens:
            raise GenerationFormatError(
                "chunking.overlap_tokens must be less than max_tokens"
            )
        return cls(
            fingerprint=_sha256(data["fingerprint"], field="chunking.fingerprint"),
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            document_header=_boolean(
                data["document_header"], field="chunking.document_header"
            ),
        )


@dataclass(frozen=True, order=True)
class CoveredRun:
    source: str
    run_id: str

    @classmethod
    def from_dict(cls, value: Any, *, index: int) -> CoveredRun:
        data = _object(value, field=f"covered_runs[{index}]")
        _exact_keys(data, {"source", "run_id"}, field=f"covered_runs[{index}]")
        return cls(
            source=_string(data["source"], field=f"covered_runs[{index}].source"),
            run_id=_string(data["run_id"], field=f"covered_runs[{index}].run_id"),
        )


@dataclass(frozen=True)
class CodeIdentity:
    git_sha: str
    dirty_patch_sha256: str | None

    @classmethod
    def from_dict(cls, value: Any) -> CodeIdentity:
        data = _object(value, field="code")
        _exact_keys(data, {"git_sha", "dirty_patch_sha256"}, field="code")
        git_sha = data["git_sha"]
        if not isinstance(git_sha, str) or not _GIT_SHA_RE.fullmatch(git_sha):
            raise GenerationFormatError(
                "code.git_sha must be a lowercase Git object ID"
            )
        return cls(
            git_sha=git_sha,
            dirty_patch_sha256=_optional_sha256(
                data["dirty_patch_sha256"], field="code.dirty_patch_sha256"
            ),
        )


@dataclass(frozen=True)
class DependencyIdentity:
    lock_sha256: str
    image_digest: str | None

    @classmethod
    def from_dict(cls, value: Any) -> DependencyIdentity:
        data = _object(value, field="dependency")
        _exact_keys(data, {"lock_sha256", "image_digest"}, field="dependency")
        image_digest = data["image_digest"]
        if image_digest is not None and (
            not isinstance(image_digest, str)
            or not _IMAGE_DIGEST_RE.fullmatch(image_digest)
        ):
            raise GenerationFormatError(
                "dependency.image_digest must be null or a sha256: image digest"
            )
        return cls(
            lock_sha256=_sha256(data["lock_sha256"], field="dependency.lock_sha256"),
            image_digest=image_digest,
        )


@dataclass(frozen=True)
class CreationIdentity:
    created_at: str
    run_id: str
    actor: str

    @classmethod
    def from_dict(cls, value: Any) -> CreationIdentity:
        data = _object(value, field="creation")
        _exact_keys(data, {"created_at", "run_id", "actor"}, field="creation")
        created_at = data["created_at"]
        parse_rfc3339_utc(created_at, field="creation.created_at")
        return cls(
            created_at=created_at,
            run_id=_string(data["run_id"], field="creation.run_id"),
            actor=_string(data["actor"], field="creation.actor"),
        )


@dataclass(frozen=True)
class GenerationManifest:
    schema_version: int
    generation_id: str
    document_count: int
    indexed_document_count: int
    excluded_document_count: int
    chunk_count: int
    sample_count: int
    corpus: CorpusIdentity
    source: SourceIdentity
    model: ModelIdentity
    vector_space: VectorSpaceIdentity
    chunking: ChunkingIdentity
    covered_runs: tuple[CoveredRun, ...]
    retrieval_fingerprint_revision: int
    retrieval_fingerprint: str
    code: CodeIdentity
    dependency: DependencyIdentity
    creation: CreationIdentity

    @classmethod
    def from_dict(cls, value: Any) -> GenerationManifest:
        data = _object(value, field="manifest")
        expected = {
            "schema_version",
            "generation_id",
            "document_count",
            "indexed_document_count",
            "excluded_document_count",
            "chunk_count",
            "sample_count",
            "corpus",
            "source",
            "model",
            "vector_space",
            "chunking",
            "covered_runs",
            "retrieval_fingerprint_revision",
            "retrieval_fingerprint",
            "code",
            "dependency",
            "creation",
        }
        _exact_keys(data, expected, field="manifest")
        schema_version = _integer(
            data["schema_version"], field="schema_version", minimum=1
        )
        if schema_version != GENERATION_SCHEMA_VERSION:
            raise GenerationFormatError(
                f"unsupported generation schema_version {schema_version}; "
                f"expected {GENERATION_SCHEMA_VERSION}"
            )
        document_count = _integer(data["document_count"], field="document_count")
        indexed_count = _integer(
            data["indexed_document_count"], field="indexed_document_count"
        )
        excluded_count = _integer(
            data["excluded_document_count"], field="excluded_document_count"
        )
        chunk_count = _integer(data["chunk_count"], field="chunk_count")
        sample_count = _integer(data["sample_count"], field="sample_count")
        if document_count != indexed_count + excluded_count:
            raise GenerationFormatError(
                "document_count must equal indexed_document_count + excluded_document_count"
            )
        if indexed_count == 0 and (chunk_count != 0 or sample_count != 0):
            raise GenerationFormatError(
                "a generation without indexed documents cannot have chunks or samples"
            )
        if indexed_count > 0:
            if chunk_count < indexed_count:
                raise GenerationFormatError(
                    "chunk_count cannot be smaller than indexed_document_count"
                )
            if sample_count < 1 or sample_count > chunk_count:
                raise GenerationFormatError(
                    "sample_count must be between 1 and chunk_count"
                )
        raw_covered_runs = data["covered_runs"]
        if not isinstance(raw_covered_runs, list):
            raise GenerationFormatError("covered_runs must be a JSON array")
        covered_runs = tuple(
            CoveredRun.from_dict(item, index=index)
            for index, item in enumerate(raw_covered_runs)
        )
        if len(set(covered_runs)) != len(covered_runs):
            raise GenerationFormatError("covered_runs contains duplicates")
        if tuple(sorted(covered_runs)) != covered_runs:
            raise GenerationFormatError(
                "covered_runs must be sorted by source and run_id"
            )
        if document_count and not covered_runs:
            raise GenerationFormatError(
                "non-empty generations require at least one covered raw run"
            )
        fingerprint_revision = _integer(
            data["retrieval_fingerprint_revision"],
            field="retrieval_fingerprint_revision",
            minimum=1,
        )
        if fingerprint_revision != RETRIEVAL_FINGERPRINT_REVISION:
            raise GenerationFormatError(
                "unsupported retrieval_fingerprint_revision "
                f"{fingerprint_revision}; expected {RETRIEVAL_FINGERPRINT_REVISION}"
            )
        return cls(
            schema_version=schema_version,
            generation_id=validate_generation_id(data["generation_id"]),
            document_count=document_count,
            indexed_document_count=indexed_count,
            excluded_document_count=excluded_count,
            chunk_count=chunk_count,
            sample_count=sample_count,
            corpus=CorpusIdentity.from_dict(data["corpus"]),
            source=SourceIdentity.from_dict(data["source"]),
            model=ModelIdentity.from_dict(data["model"]),
            vector_space=VectorSpaceIdentity.from_dict(data["vector_space"]),
            chunking=ChunkingIdentity.from_dict(data["chunking"]),
            covered_runs=covered_runs,
            retrieval_fingerprint_revision=fingerprint_revision,
            retrieval_fingerprint=_sha256(
                data["retrieval_fingerprint"], field="retrieval_fingerprint"
            ),
            code=CodeIdentity.from_dict(data["code"]),
            dependency=DependencyIdentity.from_dict(data["dependency"]),
            creation=CreationIdentity.from_dict(data["creation"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentRecord:
    schema_version: int
    generation_id: str
    source: str
    document_id: str
    version_id: str
    source_identity: str
    content_hash: str
    document_state_hash: str
    expected_chunk_count: int
    content_kind: str
    content_complete: bool
    extraction_status: str
    article_summary: str | None
    exclusion_reason: str | None
    refresh_deadline: str | None
    source_binary_url: str | None

    @classmethod
    def from_dict(
        cls, value: Any, *, expected_generation_id: str | None = None
    ) -> DocumentRecord:
        data = _object(value, field="document")
        expected = {
            "schema_version",
            "generation_id",
            "source",
            "document_id",
            "version_id",
            "source_identity",
            "content_hash",
            "document_state_hash",
            "expected_chunk_count",
            "content_kind",
            "content_complete",
            "extraction_status",
            "article_summary",
            "exclusion_reason",
            "refresh_deadline",
            "source_binary_url",
        }
        _exact_keys(data, expected, field="document")
        schema_version = _integer(
            data["schema_version"], field="document.schema_version", minimum=1
        )
        if schema_version != GENERATION_SCHEMA_VERSION:
            raise GenerationFormatError(
                f"unsupported document schema_version {schema_version}"
            )
        generation_id = validate_generation_id(data["generation_id"])
        if (
            expected_generation_id is not None
            and generation_id != expected_generation_id
        ):
            raise GenerationFormatError(
                f"document generation_id {generation_id!r} does not match "
                f"{expected_generation_id!r}"
            )
        expected_chunks = _integer(
            data["expected_chunk_count"], field="document.expected_chunk_count"
        )
        exclusion_reason = _optional_string(
            data["exclusion_reason"], field="document.exclusion_reason"
        )
        if exclusion_reason is None and expected_chunks < 1:
            raise GenerationFormatError(
                "indexed documents must have expected_chunk_count >= 1"
            )
        if exclusion_reason is not None and expected_chunks != 0:
            raise GenerationFormatError(
                "excluded documents must have expected_chunk_count == 0"
            )
        content_kind = _string(
            data["content_kind"], field="document.content_kind"
        )
        content_complete = _boolean(
            data["content_complete"], field="document.content_complete"
        )
        extraction_status = _string(
            data["extraction_status"], field="document.extraction_status"
        )
        if extraction_status not in _EXTRACTION_STATUSES:
            raise GenerationFormatError(
                "document.extraction_status must be one of "
                f"{sorted(_EXTRACTION_STATUSES)}"
            )
        if content_complete != (extraction_status == "full_text"):
            raise GenerationFormatError(
                "document.content_complete must be true exactly when "
                "document.extraction_status is 'full_text'"
            )
        if content_kind == "article_summary" and content_complete:
            raise GenerationFormatError(
                "article_summary documents cannot be marked content_complete"
            )
        refresh_deadline = data["refresh_deadline"]
        if refresh_deadline is not None:
            parse_rfc3339_utc(refresh_deadline, field="document.refresh_deadline")
        return cls(
            schema_version=schema_version,
            generation_id=generation_id,
            source=_string(data["source"], field="document.source"),
            document_id=_string(
                data["document_id"], field="document.document_id", max_length=2048
            ),
            version_id=_string(
                data["version_id"], field="document.version_id", max_length=2048
            ),
            source_identity=_sha256(
                data["source_identity"], field="document.source_identity"
            ),
            content_hash=_sha256(data["content_hash"], field="document.content_hash"),
            document_state_hash=_sha256(
                data["document_state_hash"], field="document.document_state_hash"
            ),
            expected_chunk_count=expected_chunks,
            content_kind=content_kind,
            content_complete=content_complete,
            extraction_status=extraction_status,
            article_summary=_optional_string(
                data["article_summary"],
                field="document.article_summary",
                max_length=1_000_000,
            ),
            exclusion_reason=exclusion_reason,
            refresh_deadline=refresh_deadline,
            source_binary_url=_optional_string(
                data["source_binary_url"],
                field="document.source_binary_url",
                max_length=8192,
            ),
        )

    @property
    def indexed(self) -> bool:
        return self.exclusion_reason is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SampleCheck:
    schema_version: int
    generation_id: str
    source: str
    document_id: str
    version_id: str
    chunk_index: int
    point_id: str
    text_sha256: str

    @classmethod
    def from_dict(
        cls, value: Any, *, expected_generation_id: str | None = None
    ) -> SampleCheck:
        data = _object(value, field="sample_check")
        expected = {
            "schema_version",
            "generation_id",
            "source",
            "document_id",
            "version_id",
            "chunk_index",
            "point_id",
            "text_sha256",
        }
        _exact_keys(data, expected, field="sample_check")
        schema_version = _integer(
            data["schema_version"], field="sample_check.schema_version", minimum=1
        )
        if schema_version != GENERATION_SCHEMA_VERSION:
            raise GenerationFormatError(
                f"unsupported sample_check schema_version {schema_version}"
            )
        generation_id = validate_generation_id(data["generation_id"])
        if (
            expected_generation_id is not None
            and generation_id != expected_generation_id
        ):
            raise GenerationFormatError(
                f"sample generation_id {generation_id!r} does not match "
                f"{expected_generation_id!r}"
            )
        point_id = _string(data["point_id"], field="sample_check.point_id")
        try:
            parsed_point_id = uuid.UUID(point_id)
        except ValueError as exc:
            raise GenerationFormatError("sample_check.point_id must be a UUID") from exc
        if str(parsed_point_id) != point_id:
            raise GenerationFormatError(
                "sample_check.point_id must use canonical lowercase UUID form"
            )
        return cls(
            schema_version=schema_version,
            generation_id=generation_id,
            source=_string(data["source"], field="sample_check.source"),
            document_id=_string(
                data["document_id"],
                field="sample_check.document_id",
                max_length=2048,
            ),
            version_id=_string(
                data["version_id"],
                field="sample_check.version_id",
                max_length=2048,
            ),
            chunk_index=_integer(data["chunk_index"], field="sample_check.chunk_index"),
            point_id=point_id,
            text_sha256=_sha256(data["text_sha256"], field="sample_check.text_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChecksumInventory:
    schema_version: int
    algorithm: str
    files: Mapping[str, str]

    @classmethod
    def from_dict(cls, value: Any) -> ChecksumInventory:
        data = _object(value, field="checksums")
        _exact_keys(data, {"schema_version", "algorithm", "files"}, field="checksums")
        schema_version = _integer(
            data["schema_version"], field="checksums.schema_version", minimum=1
        )
        if schema_version != GENERATION_SCHEMA_VERSION:
            raise GenerationFormatError(
                f"unsupported checksums schema_version {schema_version}"
            )
        if data["algorithm"] != CHECKSUM_ALGORITHM:
            raise GenerationFormatError(
                f"checksums.algorithm must be {CHECKSUM_ALGORITHM!r}"
            )
        raw_files = _object(data["files"], field="checksums.files")
        files: dict[str, str] = {}
        for raw_name, digest in raw_files.items():
            name = _safe_artifact_name(raw_name)
            if name == CHECKSUM_FILENAME:
                raise GenerationFormatError(
                    "checksums.json cannot include its own circular checksum"
                )
            files[name] = _sha256(digest, field=f"checksums.files[{name!r}]")
        required = {MANIFEST_FILENAME, DOCUMENTS_FILENAME, SAMPLE_CHECKS_FILENAME}
        missing = sorted(required - files.keys())
        if missing:
            raise GenerationFormatError(
                f"checksum inventory is missing required artifacts: {missing}"
            )
        return cls(
            schema_version=schema_version,
            algorithm=CHECKSUM_ALGORITHM,
            files=files,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "files": dict(self.files),
        }


def _safe_artifact_name(value: Any) -> str:
    name = _string(value, field="artifact path", max_length=4096)
    if "\\" in name or name.startswith("/") or "//" in name:
        raise GenerationFormatError(f"unsafe artifact path: {name!r}")
    raw_parts = name.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise GenerationFormatError(f"unsafe artifact path: {name!r}")
    parsed = PurePosixPath(name)
    if parsed.is_absolute() or parsed.as_posix() != name:
        raise GenerationFormatError(f"unsafe artifact path: {name!r}")
    return name


def load_manifest(path: str | Path) -> GenerationManifest:
    return GenerationManifest.from_dict(_load_json(path))


def load_checksums(path: str | Path) -> ChecksumInventory:
    return ChecksumInventory.from_dict(_load_json(path))


def _iter_jsonl(path: str | Path) -> Iterator[tuple[int, Any]]:
    file_path = Path(path)
    _regular_file(file_path)
    try:
        handle = file_path.open("rb")
    except OSError as exc:
        raise GenerationFormatError(f"cannot read {file_path}: {exc}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if len(raw_line) > MAX_METADATA_LINE_BYTES:
                raise GenerationFormatError(
                    f"{file_path}:{line_number}: metadata line exceeds "
                    f"{MAX_METADATA_LINE_BYTES} bytes"
                )
            if not raw_line.strip():
                raise GenerationFormatError(
                    f"{file_path}:{line_number}: blank JSONL lines are not allowed"
                )
            try:
                text = raw_line.decode("utf-8")
            except UnicodeError as exc:
                raise GenerationFormatError(
                    f"{file_path}:{line_number}: invalid UTF-8"
                ) from exc
            yield line_number, _parse_json(text, origin=f"{file_path}:{line_number}")


def iter_document_records(
    path: str | Path, *, expected_generation_id: str | None = None
) -> Iterator[DocumentRecord]:
    for line_number, value in _iter_jsonl(path):
        try:
            yield DocumentRecord.from_dict(
                value, expected_generation_id=expected_generation_id
            )
        except GenerationFormatError as exc:
            raise GenerationFormatError(f"{path}:{line_number}: {exc}") from exc


def iter_sample_checks(
    path: str | Path, *, expected_generation_id: str | None = None
) -> Iterator[SampleCheck]:
    for line_number, value in _iter_jsonl(path):
        try:
            yield SampleCheck.from_dict(
                value, expected_generation_id=expected_generation_id
            )
        except GenerationFormatError as exc:
            raise GenerationFormatError(f"{path}:{line_number}: {exc}") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ChecksumMismatchError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def verify_artifact_checksums(root: str | Path, inventory: ChecksumInventory) -> None:
    """Verify exact inventory membership and SHA-256 values below root."""
    directory = Path(root)
    try:
        root_mode = directory.lstat().st_mode
    except OSError as exc:
        raise ChecksumMismatchError(
            f"cannot stat generation root {directory}: {exc}"
        ) from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ChecksumMismatchError(
            f"generation root must be a non-symlink directory: {directory}"
        )

    actual: set[str] = set()
    try:
        descendants = directory.rglob("*")
        for path in descendants:
            mode = path.lstat().st_mode
            relative = path.relative_to(directory).as_posix()
            if stat.S_ISLNK(mode):
                raise ChecksumMismatchError(
                    f"generation artifacts cannot contain symlinks: {relative}"
                )
            if stat.S_ISREG(mode) and relative != CHECKSUM_FILENAME:
                actual.add(relative)
    except OSError as exc:
        raise ChecksumMismatchError(f"cannot inventory generation root: {exc}") from exc

    expected = set(inventory.files)
    if actual != expected:
        missing = sorted(expected - actual)
        unexplained = sorted(actual - expected)
        raise ChecksumMismatchError(
            f"artifact inventory mismatch: missing={missing}, unexplained={unexplained}"
        )

    for name, expected_digest in inventory.files.items():
        path = directory.joinpath(*PurePosixPath(name).parts)
        _regular_file(path)
        actual_digest = _file_sha256(path)
        if actual_digest != expected_digest:
            raise ChecksumMismatchError(
                f"checksum mismatch for {name}: "
                f"expected {expected_digest}, got {actual_digest}"
            )


@dataclass(frozen=True)
class GenerationArtifacts:
    root: Path
    manifest: GenerationManifest
    checksums: ChecksumInventory

    def iter_documents(self) -> Iterator[DocumentRecord]:
        return iter_document_records(
            self.root / DOCUMENTS_FILENAME,
            expected_generation_id=self.manifest.generation_id,
        )

    def iter_samples(self) -> Iterator[SampleCheck]:
        return iter_sample_checks(
            self.root / SAMPLE_CHECKS_FILENAME,
            expected_generation_id=self.manifest.generation_id,
        )


def load_generation(root: str | Path) -> GenerationArtifacts:
    """Checksum and open a generation without materializing its ledgers."""
    directory = Path(root)
    checksums = load_checksums(directory / CHECKSUM_FILENAME)
    verify_artifact_checksums(directory, checksums)
    manifest = load_manifest(directory / MANIFEST_FILENAME)
    return GenerationArtifacts(
        root=directory,
        manifest=manifest,
        checksums=checksums,
    )
