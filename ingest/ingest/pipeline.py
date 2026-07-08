"""Batch ingestion: read JSONL artifacts -> normalize -> chunk -> embed -> upsert.

Decoupled from the scraper (reads its output files), idempotent (deterministic point
IDs), incremental (drops stale chunks of re-ingested docs), and resumable.

Durability contract: the per-source checkpoint only ever advances to a document whose
points are in an *acknowledged* (wait=True) upsert, so a crash + ``--resume`` cannot
silently skip un-written documents. Malformed records are skipped and counted, not fatal.
"""

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
from .sources import SOURCES, normalize, schema_drift

logger = logging.getLogger("ingest.pipeline")


def _indexed_content_hash(client, cfg: Config, source: str, document_id: str) -> str | None:
    """``content_hash`` of the doc's chunk-0 already in Qdrant, or None if not indexed.

    A single O(1) point lookup by deterministic id — lets watch skip re-embedding a doc the
    scraper re-emitted unchanged, without re-reading the whole body from the index."""
    try:
        recs = client.retrieve(
            collection_name=cfg.collection_name,
            ids=[store.point_id(source, document_id, 0)],
            with_payload=["content_hash"],
        )
    except Exception:  # noqa: BLE001 - a lookup failure must never block ingestion
        return None
    return (recs[0].payload or {}).get("content_hash") if recs else None


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
    fields = ("docs", "chunks", "added", "updated", "unchanged", "skipped")
    per_source: dict = {}
    totals = dict.fromkeys(fields, 0)
    drift_total: dict = {}
    for source, st in states.items():
        row = {f: int(st.get(f, 0)) for f in fields}
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
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_checkpoint(cfg: Config, source: str, payload: dict) -> None:
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    _checkpoint_path(cfg, source).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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

    resume_after = _load_checkpoint(cfg, source).get("last_document_id") if resume else None
    skipping = resume_after is not None

    pending: list = []
    last_buffered_doc: str | None = None
    docs = chunks_total = skipped = processed = 0

    def report():
        if use_panel:
            progress.update(source, docs=docs, chunks=chunks_total, skipped=skipped)

    def flush():
        nonlocal last_buffered_doc
        if not pending:
            return
        if use_panel:
            progress.set_phase(source, "upserting")
        # wait=True so the checkpoint below only advances over durably-written points.
        store.upsert_points(client, cfg.collection_name, pending, wait=True)
        pending.clear()
        if last_buffered_doc is not None:
            _save_checkpoint(
                cfg, source,
                {"last_document_id": last_buffered_doc, "docs": docs, "chunks": chunks_total},
            )

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
            # Fast-forward past already-ingested docs, then resume on the next one.
            if doc.document_id == resume_after:
                skipping = False
            continue

        try:
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
            embedded = embedder.encode_passages([
                build_embed_text(
                    c.text, title=doc.title, document_type=doc.document_type,
                    heading_path=c.heading_path,
                )
                for c in chunks
            ])
        except Exception as exc:  # one bad doc must not abort the whole source
            skipped += 1
            logger.warning("%s: skipping doc %s (chunk/embed failed): %s", source, doc.document_id, exc)
            report()
            continue

        for chunk, emb in zip(chunks, embedded):
            vector = {"dense": emb.dense}
            if emb.sparse.indices:
                vector["sparse"] = store.sparse_vector(emb.sparse)
            pid = store.point_id(doc.source, doc.document_id, chunk.chunk_index)
            pending.append(store.point_struct(pid, vector, store.build_payload(doc, chunk)))

        # Incremental: remove chunks left over from a previously longer version. Skipped
        # on a freshly (re)created/empty collection where there is nothing to delete.
        if not skip_stale_delete:
            store.delete_doc_chunks_from(
                client, cfg.collection_name, doc.source, doc.document_id, len(chunks)
            )

        last_buffered_doc = doc.document_id
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
        raise RuntimeError(
            f"--resume checkpoint id {resume_after!r} for source {source!r} was never found "
            f"in {path}. Refusing to silently ingest nothing; delete the checkpoint to start over."
        )
    if skipped:
        logger.warning("%s: skipped %s malformed/failed records", source, skipped)
    return docs, chunks_total, skipped


def resolve_sources(source: str) -> list[str]:
    if source == "all":
        return list(SOURCES)
    if source not in SOURCES:
        raise SystemExit(f"Unknown source {source!r}; known: {', '.join(SOURCES)} (or 'all')")
    return [source]


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
    chunks = chunk_document(
        doc.body_markdown,
        max_tokens=cfg.chunk_tokens,
        overlap=cfg.chunk_overlap,
        min_tokens=cfg.chunk_min_tokens,
        count_tokens=count_tokens,
    )
    if not chunks:
        return [], 0
    embedded = embedder.encode_passages([
        build_embed_text(
            c.text, title=doc.title, document_type=doc.document_type,
            heading_path=c.heading_path,
        )
        for c in chunks
    ])
    points = []
    for chunk, emb in zip(chunks, embedded):
        vector = {"dense": emb.dense}
        if emb.sparse.indices:
            vector["sparse"] = store.sparse_vector(emb.sparse)
        pid = store.point_id(doc.source, doc.document_id, chunk.chunk_index)
        points.append(store.point_struct(pid, vector, store.build_payload(doc, chunk)))
    return points, len(chunks)


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
    Qdrant outage can never skip a document). Returns ``(docs, chunks, skipped)`` for
    this pass. Stops early (leaving offsets at the last flushed doc) when ``stop_event``
    is set or ``limit`` is reached.
    """
    run_offsets = state["run_offsets"]
    docs = chunks_total = skipped = 0
    pending: list = []
    last_off: int | None = None
    current_run_id: str | None = None

    def flush():
        nonlocal last_off
        if not pending:
            return
        # wait=True so the offset below only advances over durably-written points.
        store.upsert_points(client, cfg.collection_name, pending, wait=True)
        pending.clear()
        if last_off is not None and current_run_id is not None:
            run_offsets[current_run_id] = last_off
        _save_watch_state(cfg, source, state)

    for run_id, path in discover_runs(cfg, source):
        if stop_event is not None and stop_event.is_set():
            break
        current_run_id = run_id
        last_off = None
        offset = run_offsets.get(run_id, 0)
        lines, new_offset = _read_complete_lines(path, offset)
        if not lines:
            if run_id not in run_offsets:  # record a newly-seen (still-empty) run
                run_offsets[run_id] = offset
                _save_watch_state(cfg, source, state)
            continue

        completed = True
        for raw, end_off in lines:
            if stop_event is not None and stop_event.is_set():
                completed = False
                break
            if limit is not None and docs >= limit:
                completed = False
                break
            try:
                item = json.loads(raw)
                _record_schema_drift(source, item, state)
                doc = normalize(source, item)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                skipped += 1
                state["skipped"] = state.get("skipped", 0) + 1
                logger.warning("%s: skipping malformed record: %s", source, exc)
                continue

            # Change detection: skip re-embedding a doc the scraper re-emitted unchanged.
            embedded_prev = None
            if skip_unchanged:
                embedded_prev = _indexed_content_hash(client, cfg, source, doc.document_id)
                if embedded_prev is not None and embedded_prev == content_hash(doc.body_markdown or ""):
                    state["unchanged"] = state.get("unchanged", 0) + 1
                    last_off = end_off  # already current in the index; advance past it
                    continue

            try:
                points, n_chunks = _build_doc_points(cfg, embedder, count_tokens, doc)
            except Exception as exc:  # one bad doc must not abort the source
                skipped += 1
                state["skipped"] = state.get("skipped", 0) + 1
                logger.warning("%s: skipping doc %s (chunk/embed failed): %s", source, doc.document_id, exc)
                continue
            if not points:  # empty body -> no chunks; offset still advances at run end
                continue
            pending.extend(points)
            # Incremental: drop chunks left over from a previously longer version.
            if not skip_stale_delete:
                store.delete_doc_chunks_from(
                    client, cfg.collection_name, doc.source, doc.document_id, n_chunks
                )
            if embedded_prev is not None:
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
            # Advance past trailing skipped/blank/empty-body lines (they carry no points).
            if run_offsets.get(run_id) != new_offset:
                run_offsets[run_id] = new_offset
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
