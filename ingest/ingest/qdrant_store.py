"""Qdrant collection management, deterministic point IDs, payloads, upsert/delete."""

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date as _date
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models

from .chunking import Chunk
from .config import ConfigurationError, Config, retrieval_fingerprint_sha256
from .dedup import content_hash
from .embedding import Sparse
from .generation import GENERATION_SCHEMA_VERSION
from .operational import (
    QDRANT_RECREATE_APPROVAL_ENV,
    QDRANT_WRITE_APPROVAL_ENV,
    RUNPOD_EPHEMERAL_QDRANT_ENV,
    require_run_scoped_delta_collection,
)
from .sources import (
    PROMOTED_KEYWORD_FIELDS,
    PROMOTED_TEXT_FIELDS,
    CanonicalDoc,
)

# Fixed namespace so point IDs are stable across runs/machines.
NAMESPACE = uuid.UUID("8b1d3c9e-7a2f-4c0b-9e6a-2f1d4c5b6a70")

KEYWORD_FIELDS = (
    "source",
    "document_type",
    "language",
    "court",
    "document_id",
    "document_number",
    "registration_code",
    "status",
    # Doc-body identity (SHA-256 of the cleaned body). Replicated on every chunk of a doc.
    # Indexed so watch's change-detection and cross-source exact-dup version grouping can
    # filter by it without a full scroll.
    "content_hash",
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
)
# Boolean payload indexes (matsne consolidation flag). Filter with MatchValue(value=True/False).
BOOL_FIELDS = ("is_consolidated", "content_complete")
# Integer payload indexes (e.g. number of consolidated versions).
INTEGER_FIELDS = ("consolidated_count",)
# Datetime range indexes (in addition to the primary "date" index created below).
DATETIME_FIELDS = ("date", "in_force_date", "expiry_date")
# Full-text (MatchText) indexes for exact keyword / phrase lookup by lawyers.
TEXT_FIELDS = ("text", "title", "parties", "article_summary")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_GENERATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{7,127}$")
_READ_ONLY_SERVING_COLLECTIONS = frozenset(
    {"georgian_legal", "georgian_legal_delta"}
)


def _identity_sha256(material: dict) -> str:
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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

    def as_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "tokenizer_model": self.tokenizer_model,
            "tokenizer_revision": self.tokenizer_revision,
            "reranker_model": self.reranker_model,
            "reranker_revision": self.reranker_revision,
            "vector_space_id": self.vector_space_id,
            "chunking_fingerprint": self.chunking_fingerprint,
            "document_header": self.document_header,
            "retrieval_fingerprint": self.retrieval_fingerprint,
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
    )


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
        if generation_id is None or not _GENERATION_ID_RE.fullmatch(generation_id):
            raise ConfigurationError(
                "Qdrant writes require an explicit non-legacy GENERATION_ID before any "
                "client, model, checkpoint, or collection access"
            )
        expected_name = f"georgian_legal__gen_{generation_id}"
        if cfg.collection_name != expected_name:
            raise ConfigurationError(
                "generation writes require the exact physical collection "
                f"{expected_name!r}; refusing target {cfg.collection_name!r}"
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


def point_id(source: str, document_id: str, chunk_index: int) -> str:
    """Deterministic UUIDv5 → idempotent upserts (re-runs overwrite, never duplicate)."""
    return str(uuid.uuid5(NAMESPACE, f"{source}:{document_id}:{chunk_index}"))


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


def ensure_collection(
    client: QdrantClient,
    cfg: Config,
    *,
    recreate: bool = False,
    apply: bool = False,
    allow_run_scoped_delta: bool = False,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Create the collection if needed. Returns True if it was (re)created (i.e. empty)."""
    validate_generation_write_target(
        cfg,
        apply=apply,
        recreate=recreate,
        allow_run_scoped_delta=allow_run_scoped_delta,
        environ=environ,
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
        on_disk_payload=True,
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, always_ram=True
            )
        ),
    )
    for field in (*KEYWORD_FIELDS, *PROMOTED_KEYWORD_FIELDS):
        client.create_payload_index(
            name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
        )
    for field in (*TEXT_FIELDS, *PROMOTED_TEXT_FIELDS):
        client.create_payload_index(
            name,
            field_name=field,
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                # MULTILINGUAL handles Georgian word boundaries; the qdrant/qdrant image
                # ships the tokenizer. Georgian is caseless, so lowercase is a no-op safety.
                tokenizer=models.TokenizerType.MULTILINGUAL,
                min_token_len=2,
                max_token_len=30,
                lowercase=True,
            ),
        )
    for field in DATETIME_FIELDS:
        client.create_payload_index(
            name, field_name=field, field_schema=models.PayloadSchemaType.DATETIME
        )
    for field in BOOL_FIELDS:
        client.create_payload_index(
            name, field_name=field, field_schema=models.PayloadSchemaType.BOOL
        )
    for field in ("chunk_index", *INTEGER_FIELDS):
        client.create_payload_index(
            name, field_name=field, field_schema=models.PayloadSchemaType.INTEGER
        )
    return True


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
        "document_number": doc.document_number,
        "registration_code": doc.registration_code,
        "parties": doc.parties,
        "status": doc.status,
        "is_consolidated": doc.is_consolidated,
        "consolidated_count": doc.consolidated_count,
        "in_force_date": _rfc3339(doc.in_force_date),
        "expiry_date": _rfc3339(doc.expiry_date),
        "heading": " > ".join(chunk.heading_path) or None,
        "token_count": chunk.token_count,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "text": chunk.text,
        "article_summary": doc.article_summary,
        "content_kind": doc.content_kind,
        "content_complete": doc.content_complete,
        "extraction_status": doc.extraction_status,
        # Doc-level identity replicated on each chunk: lets watch skip re-embedding an
        # unchanged doc (compare chunk-0's hash) without re-reading the whole body from Qdrant.
        "content_hash": content_hash(doc.body_markdown or ""),
    }
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


def upsert_points(client: QdrantClient, name: str, points, *, wait: bool = False) -> None:
    _refuse_serving_collection_mutation(name)
    if points:
        client.upsert(collection_name=name, points=points, wait=wait)


def delete_doc_chunks_from(
    client: QdrantClient, name: str, source: str, document_id: str, from_index: int
) -> None:
    """Durably drop stale chunks of a re-ingested doc after its replacement is upserted."""
    _refuse_serving_collection_mutation(name)
    client.delete(
        collection_name=name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="source", match=models.MatchValue(value=source)),
                    models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
                    models.FieldCondition(key="chunk_index", range=models.Range(gte=from_index)),
                ]
            )
        ),
        wait=True,
    )
