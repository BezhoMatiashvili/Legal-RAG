"""Qdrant collection management, deterministic point IDs, payloads, upsert/delete."""

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from datetime import date as _date
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models

from .chunking import Chunk
from .config import (
    RETRIEVAL_FINGERPRINT_REVISION,
    ConfigurationError,
    Config,
    retrieval_fingerprint_sha256,
)
from .dedup import content_hash
from .embedding import Sparse
from .generation import CANONICAL_PAYLOAD_REVISION, GENERATION_SCHEMA_VERSION
from .operational import (
    QDRANT_RECREATE_APPROVAL_ENV,
    QDRANT_WRITE_APPROVAL_ENV,
    RUNPOD_EPHEMERAL_QDRANT_ENV,
    require_run_scoped_delta_collection,
)
from .promotion import physical_collection_name
from .sources import (
    COURT_EXTRACTION_SOURCES,
    PROMOTED_KEYWORD_FIELDS,
    PROMOTED_TEXT_FIELDS,
    CanonicalDoc,
    derived_version_id,
)

# Fixed namespace so point IDs are stable across runs/machines.
NAMESPACE = uuid.UUID("8b1d3c9e-7a2f-4c0b-9e6a-2f1d4c5b6a70")

KEYWORD_FIELDS = (
    "source",
    "document_type",
    "language",
    "court",
    "judges",
    "reporting_judge",
    "judge_extraction_confidence",
    "disposition",
    "disposition_source",
    "disposition_confidence",
    "court_extractor_revision",
    "document_id",
    "document_number",
    "registration_code",
    "status",
    # Doc-body identity (SHA-256 of the cleaned body). Replicated on every chunk of a doc.
    # Indexed so watch's change-detection and cross-source exact-dup version grouping can
    # filter by it without a full scroll.
    "content_hash",
    "canonical_content_hash",
    "passage_hash",
    "passage_id",
    "source_fingerprint",
    "normalizer_revision",
    "chunker_revision",
    "model_revision",
    "article_id",
    "clause_id",
    "subarticle",
    "parent_id",
    "version_id",
    "supersedes",
    "consolidation_status",
    "version_lineage_status",
    "source_authority",
    "canonical_payload_revision",
    # Immutable-generation identity. Legacy collections intentionally lack these fields
    # and therefore cannot pass the generation integrity gate.
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
    "content_kind",
    "extraction_status",
    "page_coordinate_reason",
    "page_boundary_mapping_sha256",
)
# Boolean payload indexes (matsne consolidation flag). Filter with MatchValue(value=True/False).
BOOL_FIELDS = (
    "is_consolidated",
    "content_complete",
    "canonical_text_exact",
    "version_lineage_complete",
    "freshness_sla_met",
    "admissible",
    "disposition_mixed",
)
# Integer payload indexes (e.g. number of consolidated versions).
INTEGER_FIELDS = ("consolidated_count", "retrieval_fingerprint_revision")
# Datetime range indexes (in addition to the primary "date" index created below).
DATETIME_FIELDS = (
    "date",
    "in_force_date",
    "expiry_date",
    "effective_from",
    "effective_to",
    "repeal_date",
)
# Full-text (MatchText) indexes for exact keyword / phrase lookup by lawyers.
TEXT_FIELDS = (
    "text",
    "title",
    "parties",
    "article_summary",
    "article_label",
    "chapter",
)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_GENERATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{7,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_READ_ONLY_SERVING_COLLECTIONS = frozenset(
    {"georgian_legal", "georgian_legal_delta"}
)
COLLECTION_CONFIGURATION_REVISION = 1
EMBED_COLLECTION_PROFILE_REVISION = 1

_EMBED_HNSW = {
    "m": 16,
    "ef_construct": 100,
    "full_scan_threshold": 10_000,
    "max_indexing_threads": 0,
    "on_disk": False,
    "payload_m": None,
    "inline_storage": None,
}
_EMBED_OPTIMIZERS = {
    "deleted_threshold": 0.2,
    "vacuum_min_vector_number": 1_000,
    "default_segment_number": 0,
    "max_segment_size": None,
    "memmap_threshold": None,
    "indexing_threshold": 10_000,
    "flush_interval_sec": 5,
    "max_optimization_threads": None,
    "prevent_unoptimized": None,
}
_EMBED_WAL = {
    "wal_capacity_mb": 32,
    "wal_segments_ahead": 0,
    "wal_retain_closed": 1,
}
_REVIEWED_MUTATION_AUTHORITY = object()
_REVIEWED_MUTATION_OPERATIONS = frozenset({"create", "recover", "upsert"})


@dataclass(frozen=True, slots=True)
class ReviewedMutationCapability:
    """Opaque, launch-scoped authority for frozen-candidate Qdrant mutations."""

    generation_id: str
    collection_name: str
    reviewed_plan_sha256: str
    launch_evidence_sha256: str
    collection_configuration_sha256: str
    storage_identity_sha256: str
    allowed_operations: tuple[str, ...]
    _authority: object = dataclass_field(repr=False, compare=False)


def _frozen_candidate_identity() -> tuple[str, str]:
    from .release_inputs import GENERATION_ID, PHYSICAL_COLLECTION

    return GENERATION_ID, PHYSICAL_COLLECTION


def _issue_reviewed_mutation_capability(
    *,
    generation_id: str,
    collection_name: str,
    reviewed_plan_sha256: str,
    launch_evidence_sha256: str,
    collection_configuration_digest: str,
    storage_identity_sha256: str,
    allowed_operations: Sequence[str],
) -> ReviewedMutationCapability:
    """Issue authority only after the GPU workflow independently validates a launch."""

    frozen_generation, frozen_collection = _frozen_candidate_identity()
    if (
        generation_id != frozen_generation
        or collection_name != frozen_collection
    ):
        raise ConfigurationError(
            "reviewed mutation capability is reserved for the exact frozen target"
        )
    digests = (
        reviewed_plan_sha256,
        launch_evidence_sha256,
        collection_configuration_digest,
        storage_identity_sha256,
    )
    if any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in digests):
        raise ConfigurationError("reviewed mutation capability has an invalid SHA-256")
    frozen_configuration_sha = collection_configuration_sha256(
        expected_embed_collection_configuration(dense_dim=1024)
    )
    if collection_configuration_digest != frozen_configuration_sha:
        raise ConfigurationError(
            "reviewed mutation capability collection configuration is invalid"
        )
    operations = tuple(sorted(set(allowed_operations)))
    if not operations or not set(operations) <= _REVIEWED_MUTATION_OPERATIONS:
        raise ConfigurationError("reviewed mutation capability operations are invalid")
    return ReviewedMutationCapability(
        generation_id=generation_id,
        collection_name=collection_name,
        reviewed_plan_sha256=reviewed_plan_sha256,
        launch_evidence_sha256=launch_evidence_sha256,
        collection_configuration_sha256=collection_configuration_digest,
        storage_identity_sha256=storage_identity_sha256,
        allowed_operations=operations,
        _authority=_REVIEWED_MUTATION_AUTHORITY,
    )


def require_reviewed_mutation_capability(
    cfg: Config,
    capability: ReviewedMutationCapability | None,
    *,
    operation: str,
    reviewed_plan_sha256: str | None = None,
    launch_evidence_sha256: str | None = None,
    storage_identity_sha256: str | None = None,
) -> None:
    """Fail closed when a frozen mutation lacks exact reviewed-launch authority."""

    frozen_generation, frozen_collection = _frozen_candidate_identity()
    in_frozen_scope = (
        cfg.generation_id == frozen_generation
        or cfg.collection_name == frozen_collection
    )
    if not in_frozen_scope:
        return
    expected_configuration_sha = collection_configuration_sha256(
        expected_embed_collection_configuration(dense_dim=cfg.dense_dim)
    )
    if (
        cfg.generation_id != frozen_generation
        or cfg.collection_name != frozen_collection
        or not isinstance(capability, ReviewedMutationCapability)
        or capability._authority is not _REVIEWED_MUTATION_AUTHORITY
        or capability.generation_id != frozen_generation
        or capability.collection_name != frozen_collection
        or capability.collection_configuration_sha256
        != expected_configuration_sha
        or capability.reviewed_plan_sha256 != reviewed_plan_sha256
        or capability.launch_evidence_sha256 != launch_evidence_sha256
        or capability.storage_identity_sha256 != storage_identity_sha256
        or operation not in capability.allowed_operations
    ):
        raise ConfigurationError(
            f"frozen candidate {operation} requires exact reviewed-workflow authority"
        )


def _require_reviewed_collection_capability(
    name: str,
    capability: ReviewedMutationCapability | None,
    *,
    operation: str,
    reviewed_plan_sha256: str | None,
    launch_evidence_sha256: str | None,
    storage_identity_sha256: str | None,
) -> None:
    _frozen_generation, frozen_collection = _frozen_candidate_identity()
    if name != frozen_collection:
        return
    if (
        not isinstance(capability, ReviewedMutationCapability)
        or capability._authority is not _REVIEWED_MUTATION_AUTHORITY
        or capability.generation_id != _frozen_generation
        or capability.collection_name != frozen_collection
        or capability.collection_configuration_sha256
        != collection_configuration_sha256(
            expected_embed_collection_configuration(dense_dim=1024)
        )
        or capability.reviewed_plan_sha256 != reviewed_plan_sha256
        or capability.launch_evidence_sha256 != launch_evidence_sha256
        or capability.storage_identity_sha256 != storage_identity_sha256
        or operation not in capability.allowed_operations
    ):
        raise ConfigurationError(
            f"frozen candidate {operation} requires exact reviewed-workflow authority"
        )


def _identity_sha256(material: dict) -> str:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _configuration_json_value(value):
    """Convert Qdrant model/config values to strict, stable JSON data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError("Qdrant configuration contains a non-finite number")
        return value
    raw_enum = getattr(value, "value", None)
    if isinstance(raw_enum, (str, int, float, bool)):
        return _configuration_json_value(raw_enum)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ConfigurationError("Qdrant configuration contains a non-string key")
        return {
            key: _configuration_json_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_configuration_json_value(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _configuration_json_value(dump(mode="json", exclude_none=False))
        except TypeError:
            return _configuration_json_value(dump())
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        return _configuration_json_value(
            {key: item for key, item in attributes.items() if not key.startswith("_")}
        )
    raise ConfigurationError(
        f"unsupported Qdrant configuration value {type(value).__name__}"
    )


def collection_configuration(info) -> dict:
    """Return the complete deterministic collection configuration + payload schema.

    Collection status/count/optimizer progress are deliberately excluded.  Everything that
    controls storage, vector behavior, quantization, optimizers, WAL, strict mode, and payload
    indexes is retained exactly as returned by Qdrant.
    """

    config = getattr(info, "config", None)
    payload_schema = getattr(info, "payload_schema", None)
    if isinstance(info, Mapping):
        config = info.get("config")
        payload_schema = info.get("payload_schema")
    if config is None:
        raise ConfigurationError("Qdrant collection info lacks config")
    payload_value = _configuration_json_value(payload_schema or {})
    if not isinstance(payload_value, dict):
        raise ConfigurationError("Qdrant payload schema is not an object")
    # PayloadIndexInfo contains a live ``points`` counter.  That field is collection
    # state, not configuration, and changes on each insert.  Bind every index type and
    # parameter while excluding only that mutable count so the pre-embed configuration
    # can be reproduced exactly after embedding.
    stable_payload_schema: dict[str, object] = {}
    for field, index_info in sorted(payload_value.items()):
        if not isinstance(index_info, dict):
            raise ConfigurationError(
                f"Qdrant payload schema entry {field!r} is not an object"
            )
        stable_payload_schema[field] = {
            key: item for key, item in index_info.items() if key != "points"
        }
    return {
        "revision": COLLECTION_CONFIGURATION_REVISION,
        "profile_revision": EMBED_COLLECTION_PROFILE_REVISION,
        "config": _configuration_json_value(config),
        "payload_schema": stable_payload_schema,
    }


def collection_configuration_sha256(value: Mapping[str, object]) -> str:
    if not isinstance(value, Mapping):
        raise ConfigurationError("collection configuration must be an object")
    blob = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _expected_payload_schema() -> dict[str, object]:
    text_params = models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.MULTILINGUAL,
        min_token_len=2,
        max_token_len=30,
        lowercase=True,
    )
    output: dict[str, object] = {}
    for field, schema in _expected_payload_indexes().items():
        if isinstance(schema, models.TextIndexParams):
            data_type = models.PayloadSchemaType.TEXT
            params = text_params
        else:
            data_type = schema
            params = None
        value = _configuration_json_value(
            models.PayloadIndexInfo(data_type=data_type, params=params, points=0)
        )
        assert isinstance(value, dict)
        value.pop("points", None)
        output[field] = value
    return dict(sorted(output.items()))


def expected_embed_collection_configuration(*, dense_dim: int) -> dict[str, object]:
    """Return the reviewed, complete normalized configuration before creation.

    Every server-default-sensitive field is explicit.  This makes a marker-absent
    recovery compare the live empty target byte-for-byte instead of adopting a foreign
    collection that merely has compatible vector names.
    """

    if isinstance(dense_dim, bool) or not isinstance(dense_dim, int) or dense_dim < 1:
        raise ConfigurationError("expected embed dense dimension must be positive")
    config = models.CollectionConfig(
        params=models.CollectionParams(
            vectors={
                "dense": models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            },
            sparse_vectors={"sparse": models.SparseVectorParams()},
            shard_number=1,
            sharding_method=None,
            replication_factor=1,
            write_consistency_factor=1,
            read_fan_out_factor=None,
            read_fan_out_delay_ms=None,
            on_disk_payload=True,
        ),
        hnsw_config=models.HnswConfig(**_EMBED_HNSW),
        optimizer_config=models.OptimizersConfig(**_EMBED_OPTIMIZERS),
        wal_config=models.WalConfig(**_EMBED_WAL),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=None,
                always_ram=True,
            )
        ),
        strict_mode_config=None,
        metadata=None,
    )
    normalized = _configuration_json_value(config)
    assert isinstance(normalized, dict)
    return {
        "revision": COLLECTION_CONFIGURATION_REVISION,
        "profile_revision": EMBED_COLLECTION_PROFILE_REVISION,
        "config": normalized,
        "payload_schema": _expected_payload_schema(),
    }


def vector_space_id(cfg: Config) -> str:
    """Cryptographic identity of the dense+sparse vector coordinate system."""
    return _identity_sha256(
        {
            "embedding_model": cfg.embed_model,
            "embedding_revision": cfg.embedding_revision,
            "tokenizer_model": cfg.tokenizer_model,
            "tokenizer_revision": cfg.tokenizer_revision,
            "dense_name": "dense",
            "dense_dimension": cfg.dense_dim,
            "distance": "cosine",
            "sparse_name": "sparse",
        }
    )


def chunking_fingerprint(cfg: Config) -> str:
    """Cryptographic identity of tokenization, chunk budgets, and embedded headers."""
    return _identity_sha256(
        {
            "tokenizer_model": cfg.tokenizer_model,
            "tokenizer_revision": cfg.tokenizer_revision,
            "max_tokens": cfg.chunk_tokens,
            "overlap_tokens": cfg.chunk_overlap,
            "min_tokens": cfg.chunk_min_tokens,
            "document_header": cfg.embed_header_v2,
        }
    )


@dataclass(frozen=True)
class GenerationPointIdentity:
    schema_version: int
    generation_id: str
    embedding_model: str
    embedding_revision: str
    tokenizer_model: str
    tokenizer_revision: str
    reranker_model: str
    reranker_revision: str
    vector_space_id: str
    chunking_fingerprint: str
    document_header: bool
    retrieval_fingerprint: str
    retrieval_fingerprint_revision: int

    def as_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "canonical_payload_revision": CANONICAL_PAYLOAD_REVISION,
            "generation_id": self.generation_id,
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            # Generic alias consumed by the canonical evidence contract.  The explicit
            # embedding/tokenizer/reranker revisions remain the authoritative tuple.
            "model_revision": self.embedding_revision,
            "tokenizer_model": self.tokenizer_model,
            "tokenizer_revision": self.tokenizer_revision,
            "reranker_model": self.reranker_model,
            "reranker_revision": self.reranker_revision,
            "vector_space_id": self.vector_space_id,
            "chunking_fingerprint": self.chunking_fingerprint,
            "document_header": self.document_header,
            "retrieval_fingerprint": self.retrieval_fingerprint,
            "retrieval_fingerprint_revision": self.retrieval_fingerprint_revision,
        }


def generation_point_identity(cfg: Config) -> GenerationPointIdentity | None:
    """Return a complete point identity, or reject a partially identified generation."""
    if cfg.generation_id is None:
        return None
    missing = [
        name
        for name, value in (
            ("EMBED_REVISION", cfg.embedding_revision),
            ("TOKENIZER_REVISION", cfg.tokenizer_revision),
            ("RERANK_REVISION", cfg.reranker_revision),
        )
        if value is None
    ]
    if missing:
        raise ConfigurationError(
            "generation point identity is incomplete; missing " + ", ".join(missing)
        )
    model_names = (
        ("EMBED_MODEL", cfg.embed_model),
        ("TOKENIZER_MODEL", cfg.tokenizer_model),
        ("RERANK_MODEL", cfg.rerank_model),
    )
    invalid_models = [
        name
        for name, value in model_names
        if not isinstance(value, str) or not value.strip()
    ]
    if invalid_models:
        raise ConfigurationError(
            "generation point identity has empty models: "
            + ", ".join(invalid_models)
        )
    assert cfg.embedding_revision is not None
    assert cfg.tokenizer_revision is not None
    assert cfg.reranker_revision is not None
    revisions = (
        ("EMBED_REVISION", cfg.embedding_revision),
        ("TOKENIZER_REVISION", cfg.tokenizer_revision),
        ("RERANK_REVISION", cfg.reranker_revision),
    )
    mutable = [
        name for name, value in revisions if not _REVISION_RE.fullmatch(value)
    ]
    if mutable:
        raise ConfigurationError(
            "generation point identity requires immutable revisions: "
            + ", ".join(mutable)
        )
    return GenerationPointIdentity(
        schema_version=GENERATION_SCHEMA_VERSION,
        generation_id=cfg.generation_id,
        embedding_model=cfg.embed_model,
        embedding_revision=cfg.embedding_revision,
        tokenizer_model=cfg.tokenizer_model,
        tokenizer_revision=cfg.tokenizer_revision,
        reranker_model=cfg.rerank_model,
        reranker_revision=cfg.reranker_revision,
        vector_space_id=vector_space_id(cfg),
        chunking_fingerprint=chunking_fingerprint(cfg),
        document_header=cfg.embed_header_v2,
        retrieval_fingerprint=retrieval_fingerprint_sha256(cfg),
        retrieval_fingerprint_revision=RETRIEVAL_FINGERPRINT_REVISION,
    )


def validate_generation_identity(cfg: Config) -> GenerationPointIdentity:
    """Validate the complete immutable generation/model/physical-target tuple.

    This helper performs no client access and is therefore safe to call before importing
    or loading heavyweight models.  Mutation authorization remains a separate concern in
    :func:`validate_generation_write_target`.
    """

    if cfg.generation_id is None:
        raise ConfigurationError(
            "Qdrant writes require an explicit non-legacy GENERATION_ID before any "
            "client, model, checkpoint, or collection access"
        )
    try:
        expected_name = physical_collection_name(cfg.generation_id)
    except ValueError as exc:
        raise ConfigurationError(f"invalid GENERATION_ID: {exc}") from exc
    if cfg.collection_name != expected_name:
        raise ConfigurationError(
            "generation writes require the exact physical collection "
            f"{expected_name!r}; refusing target {cfg.collection_name!r}"
        )
    identity = generation_point_identity(cfg)
    if identity is None:  # pragma: no cover - guarded above, defensive for type narrowing
        raise ConfigurationError("generation point identity is missing")
    return identity


def validate_generation_write_target(
    cfg: Config,
    *,
    apply: bool = False,
    recreate: bool = False,
    allow_run_scoped_delta: bool = False,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Authorize a physical generation or explicitly run-scoped staging mutation."""
    generation_id = cfg.generation_id
    if allow_run_scoped_delta:
        try:
            require_run_scoped_delta_collection(cfg.collection_name)
        except SystemExit as exc:
            raise ConfigurationError(str(exc)) from exc
        if generation_id is not None:
            raise ConfigurationError(
                "run-scoped delta staging must not claim an immutable GENERATION_ID"
            )
    else:
        validate_generation_identity(cfg)
        if recreate:
            raise ConfigurationError(
                "immutable generation candidates are create-only; --recreate is forbidden"
            )
    environment = os.environ if environ is None else environ
    attested_ephemeral = environment.get(RUNPOD_EPHEMERAL_QDRANT_ENV) == "1"
    approved_write = environment.get(QDRANT_WRITE_APPROVAL_ENV) == "1"
    if not apply or not (approved_write or attested_ephemeral):
        raise ConfigurationError(
            "Qdrant mutation requires --apply and either "
            f"{QDRANT_WRITE_APPROVAL_ENV}=1 or {RUNPOD_EPHEMERAL_QDRANT_ENV}=1"
        )
    if recreate and environment.get(QDRANT_RECREATE_APPROVAL_ENV) != "1":
        raise ConfigurationError(
            "destructive collection recreation additionally requires "
            f"{QDRANT_RECREATE_APPROVAL_ENV}=1"
        )


def _refuse_serving_collection_mutation(name: str) -> None:
    if name in _READ_ONLY_SERVING_COLLECTIONS:
        raise ConfigurationError(
            f"serving collection {name!r} is strictly read-only; write an independently "
            "versioned physical generation or run-scoped staging collection"
        )


def make_client(cfg: Config) -> QdrantClient:
    parsed = urlparse(cfg.qdrant_url)
    is_local = (parsed.hostname or "") in _LOCAL_HOSTS
    if not is_local:
        if not cfg.qdrant_api_key:
            raise RuntimeError(
                f"QDRANT_API_KEY is required for a non-local QDRANT_URL ({cfg.qdrant_url})."
            )
        if parsed.scheme != "https":
            raise RuntimeError(
                f"Refusing to send QDRANT_API_KEY over non-HTTPS to a remote host ({cfg.qdrant_url})."
            )
    return QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key, timeout=120)


def refuse_aliased_write_target(client: QdrantClient, collection_name: str) -> None:
    """Fail if any alias currently exposes the candidate being embedded/restored."""

    try:
        response = client.get_aliases()
        aliases = getattr(response, "aliases", None)
        if aliases is None and isinstance(response, Mapping):
            aliases = response.get("aliases")
    except Exception as exc:  # noqa: BLE001 - alias uncertainty must fail closed
        raise RuntimeError(f"cannot inspect Qdrant aliases before candidate write: {exc}") from exc
    if aliases is None:
        raise RuntimeError("Qdrant alias inventory is unavailable")
    referencing: list[str] = []
    for alias in aliases:
        target = getattr(alias, "collection_name", None)
        name = getattr(alias, "alias_name", None)
        if isinstance(alias, Mapping):
            target = alias.get("collection_name")
            name = alias.get("alias_name")
        if target == collection_name:
            referencing.append(str(name))
    if referencing:
        raise RuntimeError(
            f"candidate {collection_name!r} is referenced by aliases and is read-only: "
            + ", ".join(sorted(referencing))
        )


def point_id(
    source: str,
    document_id: str,
    chunk_index: int,
    *,
    version_id: str | None = None,
) -> str:
    """Return a deterministic UUIDv5 for a legacy document or canonical version.

    Legacy mutable/non-generation collections intentionally retain the historical
    ``source:document:chunk`` identity. Immutable schema-v2 generations must pass
    ``version_id`` so current and repealed versions cannot overwrite each other.
    """

    material = (
        f"{source}:{document_id}:{version_id}:{chunk_index}"
        if version_id is not None
        else f"{source}:{document_id}:{chunk_index}"
    )
    return str(uuid.uuid5(NAMESPACE, material))


def _assert_dense_dim(client: QdrantClient, name: str, expected: int) -> None:
    """Guard against silently upserting wrong-sized vectors into an existing collection."""
    try:
        vectors = client.get_collection(name).config.params.vectors
        size = vectors["dense"].size if isinstance(vectors, dict) else getattr(vectors, "size", None)
    except Exception:  # noqa: BLE001 - best-effort guard; never block on introspection
        return
    if size is not None and size != expected:
        raise RuntimeError(
            f"Collection {name!r} has dense dim {size} but DENSE_DIM={expected}. "
            f"Use --recreate (data loss) or set DENSE_DIM/EMBED_MODEL to match."
        )


def inspect_embed_collection(client: QdrantClient, cfg: Config) -> int:
    """Strictly validate a candidate collection's physical vector shape.

    Unlike the legacy best-effort dimension guard, immutable candidate startup must fail
    if collection metadata cannot be read or is ambiguous.  The complete model/chunk
    identity is held by the local create-only binding; this check proves the live physical
    storage still has the dense+sparse shape that binding names.
    """

    try:
        info = client.get_collection(cfg.collection_name)
        params = info.config.params
        vectors = params.vectors
        dense = vectors.get("dense") if isinstance(vectors, dict) else None
        if dense is None or set(vectors) != {"dense"}:
            raise RuntimeError("named dense vector 'dense' is missing")
        size = getattr(dense, "size", None)
        distance = getattr(dense, "distance", None)
        distance_value = getattr(distance, "value", distance)
        sparse_vectors = getattr(params, "sparse_vectors", None)
        if sparse_vectors is None:
            sparse_vectors = getattr(params, "sparse_vectors_config", None)
        has_sparse = isinstance(sparse_vectors, dict) and set(sparse_vectors) == {
            "sparse"
        }
        points_count = info.points_count
    except Exception as exc:  # noqa: BLE001 - metadata ambiguity is fatal here
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            f"cannot validate immutable candidate collection {cfg.collection_name!r}: {exc}"
        ) from exc

    if size != cfg.dense_dim:
        raise RuntimeError(
            f"candidate collection {cfg.collection_name!r} has dense dimension {size!r}; "
            f"expected {cfg.dense_dim}"
        )
    if str(distance_value).lower() != "cosine":
        raise RuntimeError(
            f"candidate collection {cfg.collection_name!r} has distance "
            f"{distance_value!r}; expected cosine"
        )
    if not has_sparse:
        raise RuntimeError(
            f"candidate collection {cfg.collection_name!r} lacks named sparse vector 'sparse'"
        )
    if not isinstance(points_count, int) or isinstance(points_count, bool) or points_count < 0:
        raise RuntimeError(
            f"candidate collection {cfg.collection_name!r} has invalid points_count "
            f"{points_count!r}"
        )
    return points_count


def verify_embed_collection_identity(
    client: QdrantClient,
    cfg: Config,
    *,
    points_count: int,
) -> None:
    """Prove every existing point matches the binding identity before a resume."""

    identity = validate_generation_identity(cfg)
    expected = {
        "schema_version": identity.schema_version,
        "canonical_payload_revision": CANONICAL_PAYLOAD_REVISION,
        "generation_id": identity.generation_id,
        "embedding_model": identity.embedding_model,
        "embedding_revision": identity.embedding_revision,
        "model_revision": identity.embedding_revision,
        "tokenizer_model": identity.tokenizer_model,
        "tokenizer_revision": identity.tokenizer_revision,
        "reranker_model": identity.reranker_model,
        "reranker_revision": identity.reranker_revision,
        "vector_space_id": identity.vector_space_id,
        "chunking_fingerprint": identity.chunking_fingerprint,
        "document_header": identity.document_header,
        "retrieval_fingerprint": identity.retrieval_fingerprint,
        "retrieval_fingerprint_revision": identity.retrieval_fingerprint_revision,
        "admissible": True,
    }
    count_filter = models.Filter(
        must=[
            models.FieldCondition(key=field, match=models.MatchValue(value=value))
            for field, value in expected.items()
        ]
    )
    try:
        result = client.count(
            collection_name=cfg.collection_name,
            count_filter=count_filter,
            exact=True,
        )
        matching_count = result.count
    except Exception as exc:  # noqa: BLE001 - an unavailable proof must fail closed
        raise RuntimeError(
            f"cannot verify existing point identity for {cfg.collection_name!r}: {exc}"
        ) from exc
    if (
        not isinstance(matching_count, int)
        or isinstance(matching_count, bool)
        or matching_count != points_count
    ):
        raise RuntimeError(
            f"existing candidate identity mismatch for {cfg.collection_name!r}: "
            f"{matching_count!r} of {points_count} points match the immutable binding"
        )


def prepare_embed_collection(
    client: QdrantClient,
    cfg: Config,
    *,
    resume: bool,
    recreate: bool,
    apply: bool,
    minimum_points: int = 0,
    environ: Mapping[str, str] | None = None,
    mutation_capability: ReviewedMutationCapability | None = None,
    reviewed_plan_sha256: str | None = None,
    launch_evidence_sha256: str | None = None,
    storage_identity_sha256: str | None = None,
) -> int:
    """Create or validate the collection lifecycle for an immutable embed run."""

    if recreate:
        raise ConfigurationError(
            "immutable generation candidates are create-only; --recreate is forbidden"
        )
    if (
        not isinstance(minimum_points, int)
        or isinstance(minimum_points, bool)
        or minimum_points < 0
    ):
        raise ConfigurationError("minimum_points must be a non-negative integer")
    validate_generation_write_target(
        cfg, apply=apply, recreate=recreate, environ=environ
    )
    if not resume:
        require_reviewed_mutation_capability(
            cfg,
            mutation_capability,
            operation="create",
            reviewed_plan_sha256=reviewed_plan_sha256,
            launch_evidence_sha256=launch_evidence_sha256,
            storage_identity_sha256=storage_identity_sha256,
        )
    exists = client.collection_exists(cfg.collection_name)
    if resume:
        if not exists:
            raise RuntimeError(
                f"--resume requires existing physical collection {cfg.collection_name!r}"
            )
        points_count = inspect_embed_collection(client, cfg)
        verify_embed_collection_identity(client, cfg, points_count=points_count)
        if points_count < minimum_points:
            raise RuntimeError(
                f"existing candidate {cfg.collection_name!r} has {points_count} points, "
                f"below the {minimum_points} chunks acknowledged by resume checkpoints"
            )
        return points_count

    if exists:
        raise RuntimeError(
            "fresh immutable embed refuses any pre-existing physical collection "
            f"{cfg.collection_name!r}, including an empty one"
        )

    ensure_collection(
        client,
        cfg,
        recreate=False,
        apply=apply,
        environ=environ,
        mutation_capability=mutation_capability,
        reviewed_plan_sha256=reviewed_plan_sha256,
        launch_evidence_sha256=launch_evidence_sha256,
        storage_identity_sha256=storage_identity_sha256,
    )
    points_count = inspect_embed_collection(client, cfg)
    if points_count:
        raise RuntimeError(
            f"new candidate collection {cfg.collection_name!r} is unexpectedly non-empty"
        )
    return points_count


def ensure_collection(
    client: QdrantClient,
    cfg: Config,
    *,
    recreate: bool = False,
    apply: bool = False,
    allow_run_scoped_delta: bool = False,
    environ: Mapping[str, str] | None = None,
    mutation_capability: ReviewedMutationCapability | None = None,
    reviewed_plan_sha256: str | None = None,
    launch_evidence_sha256: str | None = None,
    storage_identity_sha256: str | None = None,
) -> bool:
    """Create the collection if needed. Returns True if it was (re)created (i.e. empty)."""
    validate_generation_write_target(
        cfg,
        apply=apply,
        recreate=recreate,
        allow_run_scoped_delta=allow_run_scoped_delta,
        environ=environ,
    )
    require_reviewed_mutation_capability(
        cfg,
        mutation_capability,
        operation="create",
        reviewed_plan_sha256=reviewed_plan_sha256,
        launch_evidence_sha256=launch_evidence_sha256,
        storage_identity_sha256=storage_identity_sha256,
    )
    name = cfg.collection_name
    if client.collection_exists(name):
        if not recreate:
            _assert_dense_dim(client, name, cfg.dense_dim)
            return False
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=cfg.dense_dim, distance=models.Distance.COSINE, on_disk=True
            )
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
        shard_number=1,
        sharding_method=None,
        replication_factor=1,
        write_consistency_factor=1,
        on_disk_payload=True,
        hnsw_config=models.HnswConfigDiff(**_EMBED_HNSW),
        optimizers_config=models.OptimizersConfigDiff(**_EMBED_OPTIMIZERS),
        wal_config=models.WalConfigDiff(**_EMBED_WAL),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                quantile=None,
                always_ram=True,
            )
        ),
        strict_mode_config=None,
        metadata=None,
    )
    _create_missing_payload_indexes(client, name, missing=_expected_payload_indexes())
    return True


def _expected_payload_indexes() -> dict[str, object]:
    indexes: dict[str, object] = {
        field: models.PayloadSchemaType.KEYWORD
        for field in (*KEYWORD_FIELDS, *PROMOTED_KEYWORD_FIELDS)
    }
    indexes.update(
        {
            field: models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                # MULTILINGUAL handles Georgian word boundaries; the qdrant/qdrant image
                # ships the tokenizer. Georgian is caseless, so lowercase is a no-op safety.
                tokenizer=models.TokenizerType.MULTILINGUAL,
                min_token_len=2,
                max_token_len=30,
                lowercase=True,
            )
            for field in (*TEXT_FIELDS, *PROMOTED_TEXT_FIELDS)
        }
    )
    indexes.update({field: models.PayloadSchemaType.DATETIME for field in DATETIME_FIELDS})
    indexes.update({field: models.PayloadSchemaType.BOOL for field in BOOL_FIELDS})
    indexes.update(
        {
            field: models.PayloadSchemaType.INTEGER
            for field in (
                "chunk_index",
                "page_start",
                "page_end",
                "article_start_chunk_index",
                "article_start",
                "parent_chunk_index",
                *INTEGER_FIELDS,
            )
        }
    )
    return indexes


def _create_missing_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    *,
    missing: Mapping[str, object],
) -> None:
    for field, schema in missing.items():
        client.create_payload_index(
            collection_name,
            field_name=field,
            field_schema=schema,
        )


def recover_embed_collection_initialization(
    client: QdrantClient,
    cfg: Config,
    *,
    expected_configuration: Mapping[str, object],
    apply: bool,
    environ: Mapping[str, str] | None = None,
    mutation_capability: ReviewedMutationCapability | None = None,
    reviewed_plan_sha256: str | None = None,
    launch_evidence_sha256: str | None = None,
    storage_identity_sha256: str | None = None,
) -> int:
    """Recover only an intent-authorized, still-empty candidate initialization.

    The caller must first validate the immutable local initialization intent.  This
    function never deletes/recreates a target: it creates an absent collection or fills
    only missing indexes on an empty collection left by a crashed create-only attempt.
    Unknown indexes or any point make recovery unsafe and fail closed.
    """

    reviewed_configuration = expected_embed_collection_configuration(
        dense_dim=cfg.dense_dim
    )
    if dict(expected_configuration) != reviewed_configuration:
        raise RuntimeError(
            "initialization recovery configuration differs from the reviewed "
            "complete collection profile"
        )
    require_reviewed_mutation_capability(
        cfg,
        mutation_capability,
        operation="recover",
        reviewed_plan_sha256=reviewed_plan_sha256,
        launch_evidence_sha256=launch_evidence_sha256,
        storage_identity_sha256=storage_identity_sha256,
    )
    validate_generation_write_target(cfg, apply=apply, environ=environ)
    if not client.collection_exists(cfg.collection_name):
        points_count = prepare_embed_collection(
            client,
            cfg,
            resume=False,
            recreate=False,
            apply=apply,
            environ=environ,
            mutation_capability=mutation_capability,
            reviewed_plan_sha256=reviewed_plan_sha256,
            launch_evidence_sha256=launch_evidence_sha256,
            storage_identity_sha256=storage_identity_sha256,
        )
        created_configuration = collection_configuration(
            client.get_collection(cfg.collection_name)
        )
        if created_configuration != reviewed_configuration:
            raise RuntimeError(
                "created collection does not reproduce the reviewed complete "
                "collection configuration"
            )
        return points_count
    points_count = inspect_embed_collection(client, cfg)
    if points_count != 0:
        raise RuntimeError(
            "marker-absent initialization recovery requires an empty physical collection"
        )
    info = client.get_collection(cfg.collection_name)
    observed_configuration = collection_configuration(info)
    observed_payload = observed_configuration.get("payload_schema")
    expected_payload = reviewed_configuration["payload_schema"]
    if not isinstance(observed_payload, Mapping) or not isinstance(
        expected_payload, Mapping
    ):
        raise RuntimeError("cannot inspect payload indexes during initialization recovery")
    observed_non_payload = {
        key: value
        for key, value in observed_configuration.items()
        if key != "payload_schema"
    }
    expected_non_payload = {
        key: value
        for key, value in reviewed_configuration.items()
        if key != "payload_schema"
    }
    if observed_non_payload != expected_non_payload:
        raise RuntimeError(
            "initialization recovery found a foreign collection configuration"
        )
    unknown = sorted(set(observed_payload) - set(expected_payload))
    if unknown:
        raise RuntimeError(
            "initialization recovery found unknown payload indexes: " + ", ".join(unknown)
        )
    mismatched = sorted(
        field
        for field, value in observed_payload.items()
        if value != expected_payload[field]
    )
    if mismatched:
        raise RuntimeError(
            "initialization recovery found mismatched payload index configuration: "
            + ", ".join(mismatched)
        )
    index_models = _expected_payload_indexes()
    missing = {
        field: index_models[field]
        for field in expected_payload
        if field not in observed_payload
    }
    _create_missing_payload_indexes(client, cfg.collection_name, missing=missing)
    final_info = client.get_collection(cfg.collection_name)
    final_configuration = collection_configuration(final_info)
    if final_configuration != reviewed_configuration:
        raise RuntimeError(
            "collection configuration remains incomplete or changed after recovery"
        )
    return inspect_embed_collection(client, cfg)


def _rfc3339(date: str | None) -> str | None:
    """Qdrant datetime index wants RFC 3339; promote a valid YYYY-MM-DD to midnight UTC.

    Returns None for anything that isn't a calendar-valid date (the raw value is kept
    separately in ``date_raw``), so a malformed scraped date can't poison the index.
    """
    if not date or len(date) != 10:
        return None
    try:
        _date.fromisoformat(date)
    except ValueError:
        return None
    return f"{date}T00:00:00Z"


def build_payload(
    doc: CanonicalDoc,
    chunk: Chunk,
    *,
    document_chunk_count: int | None = None,
    document_state_hash: str | None = None,
    cfg: Config | None = None,
) -> dict:
    canonical_text = chunk.canonical_text if chunk.canonical_text is not None else chunk.text
    has_exact_offsets = (
        isinstance(chunk.char_start, int)
        and isinstance(chunk.char_end, int)
        and 0 <= chunk.char_start <= chunk.char_end <= len(doc.body_markdown or "")
    )
    canonical_text_exact = bool(
        has_exact_offsets
        and (doc.body_markdown or "")[chunk.char_start : chunk.char_end] == canonical_text
    )
    if chunk.canonical_text is not None and has_exact_offsets and not canonical_text_exact:
        raise ValueError(
            "chunk canonical_text does not equal the document slice at its declared offsets"
        )
    passage_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    if chunk.passage_hash is not None and chunk.passage_hash != passage_hash:
        raise ValueError("chunk passage_hash does not match its exact canonical text")
    canonical_content_hash = content_hash(doc.body_markdown or "")
    version_id = doc.version_id or derived_version_id(doc)
    local_parent_id = chunk.parent_id
    parent_id = (
        f"{doc.source}:{doc.document_id}:{version_id}:{local_parent_id}:"
        f"{chunk.parent_chunk_index}"
        if local_parent_id
        else None
    )
    passage_identity_material = (
        f"{doc.source}\0{doc.document_id}\0{version_id}\0"
        f"{chunk.char_start}\0{chunk.char_end}\0{passage_hash}"
    )
    passage_id = "passage:" + hashlib.sha256(
        passage_identity_material.encode("utf-8")
    ).hexdigest()
    page_boundaries = [boundary.to_dict() for boundary in doc.page_boundaries]
    page_boundary_mapping_sha256 = (
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
    payload = {
        "source": doc.source,
        "document_id": doc.document_id,
        "chunk_index": chunk.chunk_index,
        "title": doc.title,
        "date": _rfc3339(doc.date),
        "date_raw": doc.date_raw,
        "language": doc.language,
        "document_type": doc.document_type,
        "court": doc.court,
        "source_url": doc.source_url,
        "source_binary_url": doc.source_binary_url,
        "official_url": doc.official_url or doc.source_url,
        "official_binary_url": doc.official_binary_url or doc.source_binary_url,
        "binary_url": doc.official_binary_url or doc.source_binary_url,
        "official_html_url": doc.official_html_url or doc.official_url or doc.source_url,
        "official_pdf_url": doc.official_pdf_url,
        "source_authority": doc.source_authority,
        "freshness_sla_met": doc.freshness_sla_met,
        "document_number": doc.document_number,
        "registration_code": doc.registration_code,
        "parties": doc.parties,
        "status": doc.status,
        "is_consolidated": doc.is_consolidated,
        "consolidated_count": doc.consolidated_count,
        "in_force_date": _rfc3339(doc.in_force_date),
        "expiry_date": _rfc3339(doc.expiry_date),
        "heading": " > ".join(chunk.heading_path) or None,
        "heading_path": list(chunk.heading_path),
        "article_id": chunk.article_id,
        "article_label": chunk.article_label,
        "article_start": chunk.article_start,
        "clause": chunk.clause,
        "clause_id": chunk.clause_id or chunk.clause,
        "clause_ids": list(chunk.clause_ids),
        "subarticle": chunk.subarticle,
        "subarticle_ids": list(chunk.subarticle_ids),
        "chapter": chunk.chapter,
        "parent_id": parent_id,
        "structural_parent_id": local_parent_id,
        "article_start_chunk_index": chunk.article_start_chunk_index,
        "parent_chunk_index": chunk.parent_chunk_index,
        "token_count": chunk.token_count,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "page_coordinate_reason": chunk.page_coordinate_reason,
        # Preserve the exact canonical map once per document without multiplying a
        # potentially hundreds-page array by every chunk.  The deterministic map hash is
        # repeated on every point and therefore remains part of every point identity.
        "page_boundaries": page_boundaries if chunk.chunk_index == 0 else None,
        "page_boundary_mapping_sha256": page_boundary_mapping_sha256,
        "admissible": doc.admissible,
        "offset_unit": "unicode_codepoint",
        "text": chunk.text,
        "canonical_text_exact": canonical_text_exact,
        "passage_hash": passage_hash,
        "passage_content_hash": passage_hash,
        "passage_id": passage_id,
        "article_summary": doc.article_summary,
        "content_kind": doc.content_kind,
        "content_complete": doc.content_complete,
        "extraction_status": doc.extraction_status,
        # Doc-level identity replicated on each chunk: lets watch skip re-embedding an
        # unchanged doc (compare chunk-0's hash) without re-reading the whole body from Qdrant.
        "content_hash": canonical_content_hash,
        "canonical_content_hash": canonical_content_hash,
        "source_fingerprint": doc.source_fingerprint,
        "normalizer_revision": doc.normalizer_revision,
        "chunker_revision": chunk.chunker_revision,
        "version_id": version_id,
        "version_id_kind": doc.version_id_kind,
        "supersedes": list(doc.supersedes),
        "effective_from": _rfc3339(doc.effective_from),
        "effective_to": _rfc3339(doc.effective_to),
        "repeal_date": _rfc3339(doc.repeal_date),
        "consolidation_status": doc.consolidation_status,
        "consolidated_dates": list(doc.consolidated_dates),
        "version_lineage_status": doc.version_lineage_status,
        "version_lineage_complete": doc.version_lineage_complete,
    }
    if doc.source in COURT_EXTRACTION_SOURCES:
        payload.update(
            {
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
        )
    if document_chunk_count is not None:
        # Completeness marker for safe delta merges. Search does not consume/index it; the
        # merge preflight uses it to prove chunks 0..N-1 are present before deleting an older
        # document's stale tail from the destination collection.
        payload["document_chunk_count"] = document_chunk_count
    if document_state_hash is not None:
        # Full writer identity (body + metadata + embedding/chunk config), used by watch to
        # detect same-body legal status/title/date/consolidation changes.
        payload["document_state_hash"] = document_state_hash
    identity = generation_point_identity(cfg) if cfg is not None else None
    if identity is not None:
        if document_chunk_count is None or document_state_hash is None:
            raise ConfigurationError(
                "generation points require document_chunk_count and document_state_hash"
            )
        if not canonical_text_exact or not canonical_text or chunk.char_end <= chunk.char_start:
            raise ConfigurationError(
                "generation points require non-empty text proven equal to the canonical "
                "document slice at exact character offsets"
            )
        if not isinstance(doc.source_fingerprint, str) or not re.fullmatch(
            r"[0-9a-f]{64}", doc.source_fingerprint
        ):
            raise ConfigurationError(
                "generation points require a canonical source_fingerprint"
            )
        if not (doc.official_url or doc.source_url):
            raise ConfigurationError("generation points require an official source URL")
        if doc.source_authority not in {"official", "primary_official"}:
            raise ConfigurationError(
                "generation points require official source authority"
            )
        if not doc.content_complete or doc.extraction_status != "full_text":
            raise ConfigurationError(
                "generation points require complete, full-text source extraction"
            )
        if doc.admissible is not True:
            raise ConfigurationError("generation points require admissible canonical evidence")
        if chunk.page_coordinate_reason != doc.page_coordinate_reason:
            raise ConfigurationError(
                "chunk page-coordinate reason differs from its canonical document"
            )
        if doc.page_boundaries:
            if (
                doc.page_coordinate_reason != "exact_pdf_text"
                or chunk.page_start is None
                or chunk.page_end is None
                or page_boundary_mapping_sha256 is None
            ):
                raise ConfigurationError(
                    "paginated generation points require exact page mapping and coordinates"
                )
        elif (
            doc.page_coordinate_reason != "source_not_paginated"
            or chunk.page_start is not None
            or chunk.page_end is not None
            or page_boundary_mapping_sha256 is not None
        ):
            raise ConfigurationError(
                "non-paginated generation points require explicit null page coordinates"
            )
        payload.update(identity.as_payload())
    # Promoted structured fields (e.g. tas applicant IDs/phones). setdefault so a promoted
    # key can never overwrite a canonical payload field.
    for key, value in doc.promoted.items():
        payload.setdefault(key, value)
    return payload


def sparse_vector(sparse: Sparse) -> models.SparseVector:
    return models.SparseVector(indices=sparse.indices, values=sparse.values)


def point_struct(pid: str, vector: dict, payload: dict) -> models.PointStruct:
    return models.PointStruct(id=pid, vector=vector, payload=payload)


def upsert_points(
    client: QdrantClient,
    name: str,
    points,
    *,
    wait: bool = False,
    mutation_capability: ReviewedMutationCapability | None = None,
    reviewed_plan_sha256: str | None = None,
    launch_evidence_sha256: str | None = None,
    storage_identity_sha256: str | None = None,
) -> None:
    _refuse_serving_collection_mutation(name)
    _require_reviewed_collection_capability(
        name,
        mutation_capability,
        operation="upsert",
        reviewed_plan_sha256=reviewed_plan_sha256,
        launch_evidence_sha256=launch_evidence_sha256,
        storage_identity_sha256=storage_identity_sha256,
    )
    if points:
        client.upsert(collection_name=name, points=points, wait=wait)


def delete_doc_chunks_from(
    client: QdrantClient,
    name: str,
    source: str,
    document_id: str,
    from_index: int,
    *,
    version_id: str | None = None,
    mutation_capability: ReviewedMutationCapability | None = None,
    reviewed_plan_sha256: str | None = None,
    launch_evidence_sha256: str | None = None,
    storage_identity_sha256: str | None = None,
) -> None:
    """Durably drop only the stale tail of the replaced document version."""
    _refuse_serving_collection_mutation(name)
    _require_reviewed_collection_capability(
        name,
        mutation_capability,
        operation="delete",
        reviewed_plan_sha256=reviewed_plan_sha256,
        launch_evidence_sha256=launch_evidence_sha256,
        storage_identity_sha256=storage_identity_sha256,
    )
    conditions = [
        models.FieldCondition(key="source", match=models.MatchValue(value=source)),
        models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
        models.FieldCondition(key="chunk_index", range=models.Range(gte=from_index)),
    ]
    if version_id is not None:
        conditions.append(
            models.FieldCondition(
                key="version_id", match=models.MatchValue(value=version_id)
            )
        )
    client.delete(
        collection_name=name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=conditions
            )
        ),
        wait=True,
    )
