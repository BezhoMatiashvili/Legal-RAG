"""Qdrant collection management, deterministic point IDs, payloads, upsert/delete."""

import uuid
from datetime import date as _date
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models

from .chunking import Chunk
from .config import Config
from .embedding import Sparse
from .sources import CanonicalDoc

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
)
# Datetime range indexes (in addition to the primary "date" index created below).
DATETIME_FIELDS = ("date", "in_force_date", "expiry_date")
# Full-text (MatchText) indexes for exact keyword / phrase lookup by lawyers.
TEXT_FIELDS = ("text", "title", "parties")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}


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


def ensure_collection(client: QdrantClient, cfg: Config, *, recreate: bool = False) -> bool:
    """Create the collection if needed. Returns True if it was (re)created (i.e. empty)."""
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
    for field in KEYWORD_FIELDS:
        client.create_payload_index(
            name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
        )
    for field in TEXT_FIELDS:
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
    client.create_payload_index(
        name, field_name="chunk_index", field_schema=models.PayloadSchemaType.INTEGER
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


def build_payload(doc: CanonicalDoc, chunk: Chunk) -> dict:
    return {
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
        "document_number": doc.document_number,
        "registration_code": doc.registration_code,
        "parties": doc.parties,
        "status": doc.status,
        "in_force_date": _rfc3339(doc.in_force_date),
        "expiry_date": _rfc3339(doc.expiry_date),
        "heading": " > ".join(chunk.heading_path) or None,
        "token_count": chunk.token_count,
        "text": chunk.text,
    }


def sparse_vector(sparse: Sparse) -> models.SparseVector:
    return models.SparseVector(indices=sparse.indices, values=sparse.values)


def point_struct(pid: str, vector: dict, payload: dict) -> models.PointStruct:
    return models.PointStruct(id=pid, vector=vector, payload=payload)


def upsert_points(client: QdrantClient, name: str, points, *, wait: bool = False) -> None:
    if points:
        client.upsert(collection_name=name, points=points, wait=wait)


def delete_doc_chunks_from(
    client: QdrantClient, name: str, source: str, document_id: str, from_index: int
) -> None:
    """Drop stale chunks of a re-ingested doc (chunk_index >= the new chunk count)."""
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
    )
