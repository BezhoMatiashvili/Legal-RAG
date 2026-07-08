"""Full-corpus embedding job — one code path, two profiles (CPU pilot / RunPod GPU prod).

Reads the **clean snapshot** (``snapshots/v1/docs/<source>.jsonl``, the versioned corpus
every downstream Part consumes), chunks + context-enriches + embeds each doc with BGE-M3
(dense + learned-sparse), and upserts Qdrant-ready points. The profile is env-driven via
``Config`` (``EMBED_DEVICE`` / ``EMBED_USE_FP16`` / ``EMBED_BATCH_SIZE``): the CPU pilot
(device cpu, fp32, small batch) proves correctness; the RunPod production run (cuda, fp16,
batch 256) does the full ~184k-doc corpus. Resumable per-source so a crash/`--resume`
never re-embeds finished work.

**One vector space (guardrail G2):** the *same* BGE-M3 version must embed corpus (GPU) and
queries (CPU). :func:`dense_checksum` embeds a fixed sentence; run it in both environments
and assert the vectors match before trusting a GPU-embedded index.
"""

import hashlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path

from . import qdrant_store as store
from .config import Config
from .pipeline import _build_doc_points
from .sources import CanonicalDoc

logger = logging.getLogger("ingest.embed_job")

SNAPSHOT_DOCS = Path(__file__).resolve().parents[1] / "snapshots" / "v1" / "docs"
SOURCES = ("matsne", "napr", "ecd", "constcourt", "tas", "tbappeal")

# Fixed KA+EN checksum sentence for the CPU-vs-GPU vector-space identity check.
CHECKSUM_SENTENCE = "საქართველოს კანონი — Article 1: this sentence pins the BGE-M3 vector space."


def snapshot_doc_to_canonical(d: dict) -> CanonicalDoc:
    """Rebuild a CanonicalDoc from a snapshot record (``extra`` isn't persisted → ``{}``)."""
    return CanonicalDoc(
        source=d["source"],
        document_id=d["document_id"],
        title=d.get("title"),
        date=d.get("date"),
        date_raw=d.get("date_raw"),
        language=d.get("language") or "ka",
        document_type=d.get("document_type"),
        court=d.get("court"),
        source_url=d.get("source_url"),
        document_number=d.get("document_number"),
        registration_code=d.get("registration_code"),
        parties=d.get("parties"),
        status=d.get("status"),
        status_raw=d.get("status_raw"),
        in_force_date=d.get("in_force_date"),
        expiry_date=d.get("expiry_date"),
        body_markdown=d["body_markdown"],
        extra={},
        promoted=d.get("promoted") or {},
    )


def iter_snapshot_docs(
    source: str, *, root: Path = SNAPSHOT_DOCS, limit: int | None = None
) -> Iterator[CanonicalDoc]:
    path = root / f"{source}.jsonl"
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield snapshot_doc_to_canonical(json.loads(line))
            n += 1
            if limit and n >= limit:
                return


def load_snapshot_docs(
    source: str, ids: set[str], *, root: Path = SNAPSHOT_DOCS
) -> list[CanonicalDoc]:
    """Load a specific set of document_ids from a source's snapshot."""
    out: list[CanonicalDoc] = []
    remaining = set(ids)
    for doc in iter_snapshot_docs(source, root=root):
        if doc.document_id in remaining:
            out.append(doc)
            remaining.discard(doc.document_id)
            if not remaining:
                break
    return out


def dense_checksum(embedder) -> tuple[str, list[float]]:
    """Return ``(sha256_prefix, full_dense_vector)`` for the checksum sentence.

    The sha (rounded to 3 dp) is a quick identical-environment fingerprint; the raw vector
    is for the **tolerant** CPU-vs-GPU comparison (:func:`checksum_cosine`) that actually
    gates the GPU embed — FP16 (corpus, GPU) vs FP32 (queries, CPU) are never bit-identical,
    so the guardrail is "same vector space", i.e. cosine ≈ 1, not sha equality.
    """
    vec = [float(x) for x in embedder.encode_query(CHECKSUM_SENTENCE).dense]
    digest = hashlib.sha256(
        json.dumps([round(x, 3) for x in vec], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return digest, vec


def checksum_cosine(cpu_vec: list[float], gpu_vec: list[float]) -> float:
    """Cosine similarity between two checksum vectors (≈1.0 ⇒ same BGE-M3 vector space)."""
    import math

    dot = sum(a * b for a, b in zip(cpu_vec, gpu_vec))
    na = math.sqrt(sum(a * a for a in cpu_vec)) or 1.0
    nb = math.sqrt(sum(b * b for b in gpu_vec)) or 1.0
    return dot / (na * nb)


def save_checksum_reference(embedder, path: Path) -> str:
    """Persist the CPU checksum vector so the GPU run can be verified against it later."""
    digest, vec = dense_checksum(embedder)
    path.write_text(json.dumps({"sha": digest, "sentence": CHECKSUM_SENTENCE, "dense": vec}),
                    encoding="utf-8")
    return digest


def _checkpoint_path(cfg: Config, source: str) -> Path:
    return cfg.state_dir / f"{source}.embed.json"


def _load_ckpt(cfg: Config, source: str) -> dict:
    p = _checkpoint_path(cfg, source)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_ckpt(cfg: Config, source: str, payload: dict) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    _checkpoint_path(cfg, source).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def embed_docs(
    cfg: Config, client, embedder, count_tokens, docs: list[CanonicalDoc], *, batch_size: int = 256
) -> tuple[int, int, int]:
    """Chunk + enrich + embed + upsert an explicit list of docs. Returns (docs, chunks, skipped)."""
    pending: list = []
    n_docs = n_chunks = skipped = 0
    for doc in docs:
        try:
            points, k = _build_doc_points(cfg, embedder, count_tokens, doc)
        except Exception as exc:  # one bad doc must not abort the run
            skipped += 1
            logger.warning("skip %s:%s (%s)", doc.source, doc.document_id, exc)
            continue
        if not points:
            continue
        pending.extend(points)
        n_docs += 1
        n_chunks += k
        if len(pending) >= batch_size:
            store.upsert_points(client, cfg.collection_name, pending, wait=True)
            pending.clear()
    if pending:
        store.upsert_points(client, cfg.collection_name, pending, wait=True)
    return n_docs, n_chunks, skipped


def embed_source_resumable(
    cfg: Config, client, embedder, count_tokens, source: str, *,
    batch_size: int = 256, limit: int | None = None,
) -> tuple[int, int, int]:
    """Embed an entire source from the snapshot, resumable via a per-source checkpoint.

    The checkpoint's ``last_document_id`` only advances over an acknowledged (wait=True)
    upsert, so a crash + re-run never skips un-embedded docs (same durability contract as
    the ingest pipeline).
    """
    ckpt = _load_ckpt(cfg, source)
    resume_after = ckpt.get("last_document_id")
    skipping = resume_after is not None
    done_docs = ckpt.get("docs", 0)
    done_chunks = ckpt.get("chunks", 0)

    pending: list = []
    last_doc: str | None = None
    docs = chunks = skipped = 0

    def flush():
        nonlocal last_doc
        if not pending:
            return
        store.upsert_points(client, cfg.collection_name, pending, wait=True)
        pending.clear()
        if last_doc is not None:
            _save_ckpt(cfg, source, {
                "last_document_id": last_doc,
                "docs": done_docs + docs, "chunks": done_chunks + chunks,
            })

    for doc in iter_snapshot_docs(source):
        if skipping:
            if doc.document_id == resume_after:
                skipping = False
            continue
        try:
            points, k = _build_doc_points(cfg, embedder, count_tokens, doc)
        except Exception as exc:
            skipped += 1
            logger.warning("skip %s:%s (%s)", source, doc.document_id, exc)
            continue
        if points:
            pending.extend(points)
        last_doc = doc.document_id
        docs += 1
        chunks += k
        if len(pending) >= batch_size:
            flush()
        if limit and docs >= limit:
            break
    flush()
    if skipping:
        raise RuntimeError(
            f"--resume id {resume_after!r} for {source!r} never found; delete "
            f"{_checkpoint_path(cfg, source)} to restart."
        )
    return docs, chunks, skipped
