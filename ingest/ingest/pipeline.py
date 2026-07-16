"""Batch ingestion: read JSONL artifacts -> normalize -> chunk -> embed -> upsert.

Decoupled from the scraper (reads its output files), idempotent (deterministic point
IDs), incremental (drops stale chunks of re-ingested docs), and resumable.

Durability contract: the per-source checkpoint only ever advances to a document whose
points are in an *acknowledged* (wait=True) upsert, so a crash + ``--resume`` cannot
silently skip un-written documents. Malformed records are skipped and counted, not fatal.
"""

import hashlib
import json
import logging
import os
import signal
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from tqdm import tqdm

from . import qdrant_store as store
from .chunking import build_embed_text, chunk_document
from .config import Config
from .dedup import content_hash
from .hygiene import assess, clean_text
from .sources import (
    SOURCES,
    derived_version_id,
    finalize_canonical_text,
    normalize,
    schema_drift,
)

logger = logging.getLogger("ingest.pipeline")

# Every authoritative source is first-class in full, snapshot and delta builds.  Incomplete
# TAS/Tbilisi Appeal records are still rejected by ``_prepare_doc_for_index`` below.
CORPUS_SOURCES = (
    "matsne",
    "ecd",
    "constcourt",
    "napr",
    "supremecourt",
    "tas",
    "tbappeal",
)


class DocumentQuarantined(ValueError):
    """A normalized document is too damaged to enter the retrieval index."""

    def __init__(self, source: str, document_id: str, reason: str):
        self.source = source
        self.document_id = document_id
        self.reason = reason
        super().__init__(f"{source}:{document_id} quarantined ({reason})")


def _prepare_doc_for_index(doc):
    """Return ``doc`` with the exact cleaned body used by the snapshot pipeline.

    Every writer must hash, chunk, embed, and calculate offsets over this same text. Keeping
    the hygiene gate here prevents continuous/delta ingestion from silently replacing clean
    base vectors with raw control-character text or indexing documents the snapshot quarantines.
    """
    report = assess(doc.body_markdown)
    if not report.is_usable:
        raise DocumentQuarantined(doc.source, doc.document_id, report.quarantine_reason)
    if not doc.content_complete:
        raise DocumentQuarantined(
            doc.source,
            doc.document_id,
            f"incomplete_content:{doc.content_kind}:{doc.extraction_status}",
        )
    return finalize_canonical_text(doc, clean_text(doc.body_markdown))


def _generation_version_id(cfg: Config, doc) -> str | None:
    """Use exactly the canonical version identity written by ``build_payload``."""

    if not cfg.generation_id:
        return None
    return doc.version_id or derived_version_id(doc)


def _header_v2_kwargs(cfg: Config, doc) -> dict:
    """The I6 v2-header fields for ``build_embed_text`` — empty dict when the knob is off
    (v1 embed text stays byte-identical)."""
    if not getattr(cfg, "embed_header_v2", False):
        return {}
    return {
        "document_number": doc.document_number,
        "date": doc.date or doc.date_raw,  # ISO first; str()[:10] in build_embed_text
        "status": doc.status,
        "is_consolidated": doc.is_consolidated,
    }


def _validated_embeddings(cfg: Config, chunks, embedded):
    """Fail loudly on partial/malformed encoder output before any index mutation."""
    embedded = list(embedded)
    if len(embedded) != len(chunks):
        raise ValueError(
            f"embedder returned {len(embedded)} vectors for {len(chunks)} chunks"
        )
    for index, emb in enumerate(embedded):
        if len(emb.dense) != cfg.dense_dim:
            raise ValueError(
                f"chunk {index}: dense dimension {len(emb.dense)} != configured {cfg.dense_dim}"
            )
        if len(emb.sparse.indices) != len(emb.sparse.values):
            raise ValueError(
                f"chunk {index}: sparse index/value cardinality mismatch "
                f"({len(emb.sparse.indices)} != {len(emb.sparse.values)})"
            )
    return embedded


_STATE_FIELDS = (
    "source",
    "document_id",
    "title",
    "date",
    "date_raw",
    "language",
    "document_type",
    "court",
    "source_url",
    "document_number",
    "registration_code",
    "parties",
    "status",
    "is_consolidated",
    "consolidated_count",
    "in_force_date",
    "expiry_date",
    "content_kind",
    "content_complete",
    "extraction_status",
    "source_binary_url",
    "article_summary",
    "source_fingerprint",
    "normalizer_revision",
    "version_id",
    "version_id_kind",
    "supersedes",
    "effective_from",
    "effective_to",
    "repeal_date",
    "consolidation_status",
    "version_lineage_status",
    "version_lineage_complete",
    "consolidated_dates",
    "official_url",
    "official_binary_url",
    "official_html_url",
    "official_pdf_url",
    "source_authority",
    "freshness_sla_met",
)
_PROMOTED_FIELDS = tuple(sorted({
    field for spec in SOURCES.values() for field in spec.promote_fields
}))


def _canonical_date(value) -> str | None:
    return str(value)[:10] if value else None


def _document_state_hash(cfg: Config, doc=None, payload: dict | None = None) -> str:
    """Hash every field that changes stored payloads, chunks, or passage embeddings."""
    if (doc is None) == (payload is None):
        raise ValueError("pass exactly one of doc or payload")
    if doc is not None:
        fields = {name: getattr(doc, name) for name in _STATE_FIELDS}
        fields["date"] = _canonical_date(doc.date)
        fields["in_force_date"] = _canonical_date(doc.in_force_date)
        fields["expiry_date"] = _canonical_date(doc.expiry_date)
        fields["effective_from"] = _canonical_date(doc.effective_from)
        fields["effective_to"] = _canonical_date(doc.effective_to)
        fields["repeal_date"] = _canonical_date(doc.repeal_date)
        promoted = dict(sorted((doc.promoted or {}).items()))
        body_hash = content_hash(doc.body_markdown or "")
    else:
        fields = {name: payload.get(name) for name in _STATE_FIELDS}
        fields["date"] = _canonical_date(payload.get("date"))
        fields["in_force_date"] = _canonical_date(payload.get("in_force_date"))
        fields["expiry_date"] = _canonical_date(payload.get("expiry_date"))
        fields["effective_from"] = _canonical_date(payload.get("effective_from"))
        fields["effective_to"] = _canonical_date(payload.get("effective_to"))
        fields["repeal_date"] = _canonical_date(payload.get("repeal_date"))
        promoted = {
            name: payload.get(name) for name in _PROMOTED_FIELDS
            if payload.get(name) not in (None, "")
        }
        body_hash = payload.get("content_hash")
    material = {
        "body_hash": body_hash,
        "fields": fields,
        "promoted": promoted,
        "index_config": {
            "embed_model": cfg.embed_model,
            "dense_dim": cfg.dense_dim,
            "chunk_tokens": cfg.chunk_tokens,
            "chunk_overlap": cfg.chunk_overlap,
            "chunk_min_tokens": cfg.chunk_min_tokens,
            "embed_header_v2": cfg.embed_header_v2,
        },
    }
    # Preserve legacy/default state hashes exactly; only explicit model identity alters
    # the hash and therefore triggers the required re-embed.
    if cfg.embedding_revision:
        material["index_config"]["embedding_revision"] = cfg.embedding_revision
    if cfg.tokenizer_model != cfg.embed_model:
        material["index_config"]["tokenizer_model"] = cfg.tokenizer_model
    if cfg.tokenizer_revision:
        material["index_config"]["tokenizer_revision"] = cfg.tokenizer_revision
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _indexed_document_state(
    client,
    cfg: Config,
    source: str,
    document_id: str,
    version_id: str | None = None,
) -> str | None:
    """State hash of the indexed chunk zero, with a legacy-payload fallback.

    A single O(1) point lookup by deterministic id — lets watch skip re-embedding a doc the
    scraper re-emitted unchanged, while metadata-only changes still update the index."""
    try:
        recs = client.retrieve(
            collection_name=cfg.collection_name,
            ids=[store.point_id(
                source,
                document_id,
                0,
                version_id=version_id if cfg.generation_id else None,
            )],
            with_payload=[*_STATE_FIELDS, *_PROMOTED_FIELDS, "content_hash", "document_state_hash"],
        )
    except Exception:  # noqa: BLE001 - a lookup failure must never block ingestion
        return None
    if not recs:
        return None
    payload = recs[0].payload or {}
    return payload.get("document_state_hash") or _document_state_hash(cfg, payload=payload)


def _record_schema_drift(source: str, item: dict, state: dict) -> None:
    """Seed / update the per-source key baseline in ``state`` and warn on newly-seen keys."""
    seen = set(state.get("schema_keys") or [])
    if not seen:  # first item establishes the baseline (no alert)
        state["schema_keys"] = sorted(set(item))
        return
    new_keys, undeclared = schema_drift(source, item, seen)
    if new_keys:
        logger.warning("%s: schema drift — new key(s) %s (undeclared: %s)",
                       source, sorted(new_keys), sorted(undeclared) or "none")
        drift = state.setdefault("schema_drift", {})
        for kk in new_keys:
            drift[kk] = drift.get(kk, 0) + 1
        state["schema_keys"] = sorted(seen | new_keys)


def write_ingest_report(cfg: Config, states: dict, *, kind: str = "watch") -> Path:
    """Write a per-source ingestion report (docs/added/updated/unchanged/skipped/chunks +
    schema-drift) to ``.state/reports/ingest-<date>.json``; returns the path."""
    reports_dir = cfg.state_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).replace(microsecond=0)
    fields = (
        "docs",
        "chunks",
        "added",
        "updated",
        "unchanged",
        "skipped",
        "pending_retries",
        "dead_letters",
    )
    per_source: dict = {}
    totals = dict.fromkeys(fields, 0)
    drift_total: dict = {}
    for source, st in states.items():
        row = {f: int(st.get(f, 0)) for f in fields[:-2]}
        row["pending_retries"] = len(st.get("embed_retry") or {})
        row["dead_letters"] = len(st.get("dead_letter") or [])
        row["dead_letter_sample"] = list(st.get("dead_letter") or [])[:20]
        row["schema_drift"] = dict(st.get("schema_drift", {}))
        row["updated_at"] = st.get("updated_at")
        per_source[source] = row
        for f in fields:
            totals[f] += row[f]
        for kk, n in row["schema_drift"].items():
            drift_total[kk] = drift_total.get(kk, 0) + n
    report = {
        "generated_at": now.isoformat(), "kind": kind, "collection": cfg.collection_name,
        "totals": totals, "schema_drift": drift_total, "per_source": per_source,
    }
    path = reports_dir / f"ingest-{now:%Y%m%d}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def items_path(cfg: Config, source: str, run: str = "latest") -> Path:
    return cfg.artifacts_root / source / run / "items.jsonl"


def _iter_lines(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield line


def _checkpoint_path(cfg: Config, source: str) -> Path:
    return cfg.state_dir / f"{source}.json"


def _load_checkpoint(cfg: Config, source: str) -> dict:
    path = _checkpoint_path(cfg, source)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A torn checkpoint (crash mid-write) must not abort --resume with an unhandled
            # JSONDecodeError; treat it as "no checkpoint" and restart (idempotent uuid5 upserts).
            logger.warning("%s: ignoring corrupt checkpoint %s (%s) — starting fresh",
                           source, path, exc)
    return {}


def _save_checkpoint(cfg: Config, source: str, payload: dict) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(cfg, source)
    # Atomic write (tmp then os.replace), mirroring _save_watch_state: this runs inside flush()
    # after every acknowledged upsert, so a crash mid-write must not truncate the checkpoint.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def delete_checkpoint(cfg: Config, source: str) -> None:
    _checkpoint_path(cfg, source).unlink(missing_ok=True)


def ingest_source(
    cfg: Config,
    client,
    embedder,
    count_tokens,
    source: str,
    *,
    run: str = "latest",
    limit: int | None = None,
    batch_size: int = 256,
    resume: bool = False,
    skip_stale_delete: bool = False,
    progress=None,
) -> tuple[int, int, int]:
    """Returns (docs, chunks, skipped).

    ``progress`` is an optional :class:`ingest.progress.IngestProgress`. When it is
    enabled (interactive TTY) the per-doc updates feed its live panel; otherwise the
    loop falls back to a plain tqdm bar, preserving the original non-interactive output.
    """
    path = items_path(cfg, source, run)
    if not path.exists():
        raise FileNotFoundError(f"No artifacts for {source!r}: {path}")

    use_panel = progress is not None and progress.enabled

    checkpoint = _load_checkpoint(cfg, source) if resume else {}
    resume_after = checkpoint.get("last_document_id")
    resume_after_version = checkpoint.get("last_version_id")
    if cfg.generation_id and resume_after is not None and not resume_after_version:
        raise RuntimeError(
            "generation resume checkpoint lacks last_version_id; delete the legacy "
            "checkpoint and restart this immutable generation"
        )
    skipping = resume_after is not None

    pending: list = []
    pending_deletes: list[tuple[str, str, str | None, int]] = []
    last_buffered_doc: str | None = None
    last_buffered_version: str | None = None
    docs = chunks_total = skipped = processed = 0

    def report():
        if use_panel:
            progress.update(source, docs=docs, chunks=chunks_total, skipped=skipped)

    def flush():
        nonlocal last_buffered_doc, last_buffered_version
        if not pending:
            return
        if use_panel:
            progress.set_phase(source, "upserting")
        # wait=True so the checkpoint below only advances over durably-written points.
        store.upsert_points(client, cfg.collection_name, pending, wait=True)
        # Only after every replacement point is acknowledged may an older tail be removed.
        # A delete failure leaves the checkpoint behind, so the idempotent upsert+delete pair
        # is retried rather than recording a partial document as complete.
        for delete_source, delete_document_id, delete_version_id, from_index in pending_deletes:
            store.delete_doc_chunks_from(
                client,
                cfg.collection_name,
                delete_source,
                delete_document_id,
                from_index,
                version_id=delete_version_id,
            )
        if last_buffered_doc is not None:
            checkpoint_payload = {
                "last_document_id": last_buffered_doc,
                "docs": docs,
                "chunks": chunks_total,
            }
            if cfg.generation_id:
                checkpoint_payload["last_version_id"] = last_buffered_version
            _save_checkpoint(
                cfg, source, checkpoint_payload,
            )
        pending.clear()
        pending_deletes.clear()

    if use_panel:
        # One cheap pass to count candidate docs so the panel can show docs/total.
        total = sum(1 for _ in _iter_lines(path))
        progress.start_source(source, total=min(total, limit) if limit else total)
        lines = _iter_lines(path)
    else:
        lines = tqdm(_iter_lines(path), desc=f"ingest:{source}", unit="doc")

    for raw in lines:
        try:
            item = json.loads(raw)
            doc = normalize(source, item)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            skipped += 1
            logger.warning("%s: skipping malformed record: %s", source, exc)
            report()
            continue

        if skipping:
            # Fast-forward past the exact immutable document version, then resume on the
            # next row. Legacy collections retain the historical document-only checkpoint.
            identity_matches = doc.document_id == resume_after
            if cfg.generation_id:
                identity_matches = identity_matches and (
                    _generation_version_id(cfg, doc) == resume_after_version
                )
            if identity_matches:
                skipping = False
            continue

        try:
            doc = _prepare_doc_for_index(doc)
            chunks = chunk_document(
                doc.body_markdown,
                max_tokens=cfg.chunk_tokens,
                overlap=cfg.chunk_overlap,
                min_tokens=cfg.chunk_min_tokens,
                count_tokens=count_tokens,
            )
            if not chunks:
                continue
            if use_panel:
                progress.set_phase(source, "embedding")
            embedded = _validated_embeddings(cfg, chunks, embedder.encode_passages([
                build_embed_text(
                    c.text, title=doc.title, document_type=doc.document_type,
                    heading_path=c.heading_path, **_header_v2_kwargs(cfg, doc),
                )
                for c in chunks
            ]))
        except DocumentQuarantined as exc:
            skipped += 1
            logger.warning("%s", exc)
            report()
            continue
        except Exception as exc:  # one bad doc must not abort the whole source
            skipped += 1
            logger.warning("%s: skipping doc %s (chunk/embed failed): %s", source, doc.document_id, exc)
            report()
            continue

        state_hash = _document_state_hash(cfg, doc=doc)
        for chunk, emb in zip(chunks, embedded):
            vector = {"dense": emb.dense}
            if emb.sparse.indices:
                vector["sparse"] = store.sparse_vector(emb.sparse)
            pid = store.point_id(
                doc.source,
                doc.document_id,
                chunk.chunk_index,
                version_id=_generation_version_id(cfg, doc),
            )
            pending.append(store.point_struct(
                pid,
                vector,
                store.build_payload(
                    doc,
                    chunk,
                    document_chunk_count=len(chunks),
                    document_state_hash=state_hash,
                    cfg=cfg,
                ),
            ))

        # Incremental: remove chunks left over from a previously longer version. Skipped
        # on a freshly (re)created/empty collection where there is nothing to delete.
        if not skip_stale_delete:
            pending_deletes.append((
                doc.source,
                doc.document_id,
                _generation_version_id(cfg, doc),
                len(chunks),
            ))

        last_buffered_doc = doc.document_id
        last_buffered_version = _generation_version_id(cfg, doc)
        docs += 1
        chunks_total += len(chunks)
        processed += 1
        report()
        if len(pending) >= batch_size:
            flush()
        if limit and processed >= limit:
            break

    flush()

    if skipping:
        checkpoint_identity = (
            f"{resume_after}@{resume_after_version}"
            if resume_after_version
            else repr(resume_after)
        )
        raise RuntimeError(
            f"--resume checkpoint identity {checkpoint_identity} for source {source!r} "
            "was never found "
            f"in {path}. Refusing to silently ingest nothing; delete the checkpoint to start over."
        )
    if skipped:
        logger.warning("%s: skipped %s malformed/failed records", source, skipped)
    return docs, chunks_total, skipped


def resolve_sources(source: str) -> list[str]:
    if source == "all":
        return list(CORPUS_SOURCES)
    requested = [part.strip() for part in source.split(",") if part.strip()]
    unknown = [part for part in requested if part not in SOURCES]
    if not requested or unknown:
        bad = unknown or [source]
        raise SystemExit(f"Unknown source(s) {bad!r}; known: {', '.join(SOURCES)} (or 'all')")
    return list(dict.fromkeys(requested))


# --- continuous watch mode -------------------------------------------------
#
# ``watch`` backfills every already-scraped document oldest->newest, then keeps
# running and ingests new documents as the scraper produces them. It reads across
# *all* run dirs (``artifacts/<source>/runs/*/items.jsonl``) instead of the single
# ``latest`` file the plain ``ingest`` command uses, and tracks a per-source byte
# offset into each run file in its own state file (``<source>.watch.json``) so it
# never re-embeds work it has already done. The same durability contract applies:
# an offset only advances over an acknowledged (wait=True) upsert.


def discover_runs(cfg: Config, source: str) -> list[tuple[str, Path]]:
    """Return ``[(run_id, items.jsonl path), ...]`` for a source, ascending by run_id.

    ``run_id`` carries a fixed-width UTC-timestamp prefix, so a plain ascending sort
    is chronological (oldest first). ``latest/`` is ignored: it is a duplicate of the
    newest run dir, so reading only ``runs/`` avoids ingesting the same docs twice.
    """
    runs_dir = cfg.artifacts_root / source / "runs"
    if not runs_dir.exists():
        return []
    out: list[tuple[str, Path]] = []
    for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        items = run_dir / "items.jsonl"
        if items.exists():
            out.append((run_dir.name, items))
    return out


def _read_complete_lines(path: Path, offset: int) -> tuple[list[tuple[str, int]], int]:
    """Read newline-terminated lines from ``offset`` onward.

    Returns ``([(line_text, end_byte_offset), ...], new_offset)``. Only lines ending
    in ``\\n`` are returned; a partial trailing line still being written by an
    in-progress crawl is left unconsumed (``new_offset`` stays before it) so it is
    picked up once complete. Offsets are byte counts (UTF-8 safe for Georgian text).
    Self-heals if the file shrank or was rewritten (``offset`` past EOF -> restart).
    """
    size = path.stat().st_size
    if offset > size:  # file shrank / was overwritten -> re-read from the start
        offset = 0
    with open(path, "rb") as fh:
        fh.seek(offset)
        data = fh.read()
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return [], offset  # no complete line available yet
    consumed = data[: last_nl + 1]
    new_offset = offset + len(consumed)
    out: list[tuple[str, int]] = []
    acc = offset
    for part in consumed.split(b"\n")[:-1]:  # drop the empty element after the final \n
        acc += len(part) + 1  # +1 for the newline that split() stripped
        text = part.decode("utf-8", "replace").strip()
        if text:
            out.append((text, acc))
    return out, new_offset


def _watch_state_path(cfg: Config, source: str) -> Path:
    return cfg.state_dir / f"{source}.watch.json"


def _load_watch_state(cfg: Config, source: str) -> dict:
    path = _watch_state_path(cfg, source)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        state.setdefault("run_offsets", {})
        return state
    return {"run_offsets": {}, "docs": 0, "chunks": 0, "skipped": 0}


def _save_watch_state(cfg: Config, source: str, state: dict) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    path = _watch_state_path(cfg, source)
    tmp = path.with_name(path.name + ".tmp")  # atomic write: tmp then os.replace
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def delete_watch_state(cfg: Config, source: str) -> None:
    _watch_state_path(cfg, source).unlink(missing_ok=True)


def _build_doc_points(cfg: Config, embedder, count_tokens, doc) -> tuple[list, int]:
    """Chunk + embed one canonical doc into Qdrant points. Returns ``([], 0)`` when the
    doc yields no chunks (e.g. empty body). Mirrors the per-doc core of ``ingest_source``."""
    doc = _prepare_doc_for_index(doc)
    chunks = chunk_document(
        doc.body_markdown,
        max_tokens=cfg.chunk_tokens,
        overlap=cfg.chunk_overlap,
        min_tokens=cfg.chunk_min_tokens,
        count_tokens=count_tokens,
    )
    if not chunks:
        return [], 0
    embedded = _validated_embeddings(cfg, chunks, embedder.encode_passages([
        build_embed_text(
            c.text, title=doc.title, document_type=doc.document_type,
            heading_path=c.heading_path, **_header_v2_kwargs(cfg, doc),
        )
        for c in chunks
    ]))
    points = []
    state_hash = _document_state_hash(cfg, doc=doc)
    for chunk, emb in zip(chunks, embedded):
        vector = {"dense": emb.dense}
        if emb.sparse.indices:
            vector["sparse"] = store.sparse_vector(emb.sparse)
        pid = store.point_id(
            doc.source,
            doc.document_id,
            chunk.chunk_index,
            version_id=_generation_version_id(cfg, doc),
        )
        points.append(store.point_struct(
            pid,
            vector,
            store.build_payload(
                doc,
                chunk,
                document_chunk_count=len(chunks),
                document_state_hash=state_hash,
                cfg=cfg,
            ),
        ))
    return points, len(chunks)


# How many passes a doc may fail to embed before it is dead-lettered instead of retried.
# Bounds the retry so a deterministically-poison doc cannot head-of-line-block the drain,
# while a transient CUDA/OOM fault still gets several attempts before being given up on.
_MAX_EMBED_RETRIES = 5


def watch_drain_source(
    cfg: Config,
    client,
    embedder,
    count_tokens,
    source: str,
    state: dict,
    *,
    batch_size: int = 256,
    skip_stale_delete: bool = False,
    skip_unchanged: bool = True,
    limit: int | None = None,
    stop_event: threading.Event | None = None,
) -> tuple[int, int, int]:
    """Process every currently-available complete line for ``source``, oldest run first.

    Mutates and persists ``state`` after each acknowledged upsert (durability contract:
    a run's offset only advances over points that are durably written, so a crash or a
    Qdrant outage can never skip a document). A doc whose embed *transiently* fails holds
    the run offset at its own start (``retry_floor``) so it is re-read next pass, bounded
    by ``_MAX_EMBED_RETRIES``; only after that many failures is it dead-lettered (recorded
    in ``state['dead_letter']``) and the offset allowed past it. Returns
    ``(docs, chunks, skipped)`` for this pass. Stops early (leaving offsets at the last
    flushed doc) when ``stop_event`` is set or ``limit`` is reached.
    """
    run_offsets = state["run_offsets"]
    docs = chunks_total = skipped = 0
    pending: list = []
    pending_deletes: list[tuple[str, str, str | None, int]] = []
    last_off: int | None = None
    # Lowest byte offset of a doc whose embed failed transiently this pass; the run offset
    # must never advance past it (else a later successful doc's offset strands the failed one).
    retry_floor: int | None = None
    current_run_id: str | None = None

    def flush():
        nonlocal last_off
        if not pending:
            return
        # wait=True so the offset below only advances over durably-written points.
        store.upsert_points(client, cfg.collection_name, pending, wait=True)
        for delete_source, delete_document_id, delete_version_id, from_index in pending_deletes:
            store.delete_doc_chunks_from(
                client,
                cfg.collection_name,
                delete_source,
                delete_document_id,
                from_index,
                version_id=delete_version_id,
            )
        if last_off is not None and current_run_id is not None:
            # Cap at retry_floor: never persist an offset past a doc awaiting an embed retry.
            off = last_off if retry_floor is None else min(last_off, retry_floor)
            run_offsets[current_run_id] = off
        _save_watch_state(cfg, source, state)
        pending.clear()
        pending_deletes.clear()

    for run_id, path in discover_runs(cfg, source):
        if stop_event is not None and stop_event.is_set():
            break
        current_run_id = run_id
        last_off = None
        retry_floor = None
        offset = run_offsets.get(run_id, 0)
        lines, new_offset = _read_complete_lines(path, offset)
        if not lines:
            if run_id not in run_offsets:  # record a newly-seen (still-empty) run
                run_offsets[run_id] = offset
                _save_watch_state(cfg, source, state)
            continue

        prev_off = offset  # running byte cursor: the start of a line == the end of the previous
        completed = True
        for raw, end_off in lines:
            if stop_event is not None and stop_event.is_set():
                completed = False
                break
            if limit is not None and docs >= limit:
                completed = False
                break
            start_off = prev_off  # byte offset of THIS doc's line (for retry_floor)
            prev_off = end_off
            try:
                item = json.loads(raw)
                _record_schema_drift(source, item, state)
                doc = normalize(source, item)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                # Permanently malformed record (not a transient fault) — safe to advance past.
                skipped += 1
                state["skipped"] = state.get("skipped", 0) + 1
                logger.warning("%s: skipping malformed record: %s", source, exc)
                continue

            try:
                doc = _prepare_doc_for_index(doc)
            except DocumentQuarantined as exc:
                skipped += 1
                state["skipped"] = state.get("skipped", 0) + 1
                quarantine = state.setdefault("quarantined", {})
                quarantine[exc.reason] = quarantine.get(exc.reason, 0) + 1
                last_off = end_off  # permanent hygiene decision; never retry this record
                logger.warning("%s", exc)
                continue

            # Change detection: skip re-embedding a doc the scraper re-emitted unchanged.
            indexed_prev_state = None
            if skip_unchanged:
                indexed_prev_state = _indexed_document_state(
                    client, cfg, source, doc.document_id,
                    _generation_version_id(cfg, doc),
                )
                if (indexed_prev_state is not None
                        and indexed_prev_state == _document_state_hash(cfg, doc=doc)):
                    state["unchanged"] = state.get("unchanged", 0) + 1
                    last_off = end_off  # already current in the index; advance past it
                    continue

            try:
                points, n_chunks = _build_doc_points(cfg, embedder, count_tokens, doc)
            except Exception as exc:  # a chunk/embed failure for one doc must not abort the source
                # A transient embed fault (CUDA/OOM, momentary model glitch) is NOT permanent, so
                # the run offset must NOT advance past this doc or it is silently lost forever
                # (undermining the 'verify_all_embedded: 0 missing' guarantee). Hold the offset at
                # the doc's start (retry_floor) so the next pass re-reads it, bounded by
                # _MAX_EMBED_RETRIES; after that, dead-letter it so a deterministically-poison doc
                # cannot head-of-line-block the whole drain.
                retries = state.setdefault("embed_retry", {})
                n = retries.get(doc.document_id, 0) + 1
                skipped += 1
                if n < _MAX_EMBED_RETRIES:
                    retries[doc.document_id] = n
                    if retry_floor is None or start_off < retry_floor:
                        retry_floor = start_off
                    logger.warning("%s: transient chunk/embed failure for doc %s (attempt %d/%d) "
                                   "— will retry next pass: %s",
                                   source, doc.document_id, n, _MAX_EMBED_RETRIES, exc)
                else:
                    retries.pop(doc.document_id, None)
                    state["skipped"] = state.get("skipped", 0) + 1
                    state.setdefault("dead_letter", []).append(
                        {"document_id": doc.document_id, "run": run_id, "error": str(exc)[:200]})
                    last_off = end_off  # give up on the poison doc; let the drain progress past it
                    logger.error("%s: doc %s failed embed %d times — DEAD-LETTERING (offset "
                                 "advances past it): %s",
                                 source, doc.document_id, _MAX_EMBED_RETRIES, exc)
                continue
            if not points:  # empty body -> no chunks; offset still advances at run end
                continue
            # This doc embedded cleanly — clear any prior transient-failure count for it.
            if state.get("embed_retry"):
                state["embed_retry"].pop(doc.document_id, None)
            pending.extend(points)
            # Incremental: drop chunks left over from a previously longer version.
            if not skip_stale_delete:
                pending_deletes.append((
                    doc.source,
                    doc.document_id,
                    _generation_version_id(cfg, doc),
                    n_chunks,
                ))
            if indexed_prev_state is not None:
                state["updated"] = state.get("updated", 0) + 1   # re-embedded a changed doc
            elif skip_unchanged:
                state["added"] = state.get("added", 0) + 1       # newly-seen doc
            last_off = end_off
            docs += 1
            chunks_total += n_chunks
            state["docs"] = state.get("docs", 0) + 1
            state["chunks"] = state.get("chunks", 0) + n_chunks
            if len(pending) >= batch_size:
                flush()

        flush()
        if completed:
            # Advance past trailing skipped/blank/empty-body lines (they carry no points), but
            # never past a doc still awaiting an embed retry this pass (retry_floor).
            target = new_offset if retry_floor is None else min(new_offset, retry_floor)
            if run_offsets.get(run_id) != target:
                run_offsets[run_id] = target
                _save_watch_state(cfg, source, state)
        else:
            break

    return docs, chunks_total, skipped


def watch_loop(
    cfg: Config,
    client,
    embedder,
    count_tokens,
    sources: list[str],
    *,
    batch_size: int = 256,
    poll_interval: float = 5.0,
    once: bool = False,
    limit: int | None = None,
    skip_stale_delete: bool = False,
) -> None:
    """Backfill all scraped docs oldest->newest, then poll for new ones until stopped.

    ``once`` drains everything not yet ingested and exits (no waiting). Otherwise runs as
    a daemon: SIGINT/SIGTERM trigger a graceful stop that finishes the in-flight batch and
    does one final drain so nothing the scraper just wrote is missed. Qdrant errors are
    retried with exponential backoff; offsets never advance on a failed pass.
    """
    stop_event = threading.Event()

    def _handle_signal(signum, _frame):
        logger.info("received signal %s; finishing current batch, then stopping…", signum)
        stop_event.set()

    installed: dict = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            installed[sig] = signal.signal(sig, _handle_signal)
        except ValueError:  # not on the main thread -> can't install handlers
            installed.clear()
            break

    states = {s: _load_watch_state(cfg, s) for s in sources}
    base_interval = poll_interval if poll_interval > 0 else 1.0
    backoff = base_interval
    was_idle = False
    first_pass = True

    try:
        while not stop_event.is_set():
            try:
                total = 0
                for source in sources:
                    d, c, k = watch_drain_source(
                        cfg, client, embedder, count_tokens, source, states[source],
                        batch_size=batch_size,
                        skip_stale_delete=skip_stale_delete and first_pass,
                        limit=limit, stop_event=stop_event,
                    )
                    if d or k:
                        logger.info("%s: ingested %d docs (%d chunks, %d skipped) this pass",
                                    source, d, c, k)
                    total += d
                first_pass = False
                backoff = base_interval  # success resets the backoff
            except Exception as exc:  # Qdrant down, transient I/O, etc.
                logger.warning("ingest pass failed, retrying in %.0fs: %s", backoff, exc)
                if stop_event.wait(backoff):
                    break
                backoff = min(backoff * 2, 60.0)
                continue

            if stop_event.is_set():
                break
            if once:
                if total == 0:
                    pending = sum(len(state.get("embed_retry") or {}) for state in states.values())
                    dead = sum(len(state.get("dead_letter") or []) for state in states.values())
                    if dead:
                        raise RuntimeError(
                            f"watch completed with {dead} dead-lettered document(s); "
                            "inspect the ingest report and repair/retry before verification"
                        )
                    if pending:
                        logger.warning(
                            "%d document(s) still await bounded embed retry; draining again",
                            pending,
                        )
                        continue
                    return  # backfill complete
                continue  # keep draining until a pass yields nothing new
            if total == 0:
                if not was_idle:
                    logger.info("caught up; waiting for new documents (poll every %.0fs)…",
                                poll_interval)
                    was_idle = True
            else:
                was_idle = False
            if stop_event.wait(poll_interval):
                break

        # Graceful shutdown: one final drain so docs written just before the signal land.
        if stop_event.is_set():
            logger.info("shutting down; final drain…")
            for source in sources:
                try:
                    watch_drain_source(
                        cfg, client, embedder, count_tokens, source, states[source],
                        batch_size=batch_size, skip_stale_delete=False, limit=limit,
                    )
                except Exception as exc:
                    logger.warning("%s: final drain failed: %s", source, exc)
    finally:
        for sig, handler in installed.items():
            signal.signal(sig, handler)
        try:
            path = write_ingest_report(cfg, states, kind="watch")
            logger.info("ingest report: %s", path)
        except Exception as exc:  # noqa: BLE001 - a report failure must not mask shutdown
            logger.warning("failed to write ingest report: %s", exc)
