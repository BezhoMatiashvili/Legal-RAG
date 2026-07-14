#!/usr/bin/env python
"""Check document-ID coverage across every source in the main collection.

Universe = every unique ``document_id`` per source found in
``artifacts/<source>/{latest,runs/*}/items.jsonl``. Embedded set = one full scroll of the
collection reading only the ``source`` + ``document_id`` payload fields (no per-id count
queries — one scroll covers the whole corpus in minutes). The diff is written to:

    ingest/.state/embed_coverage.json   — per-source summary (rendered by session_monitor)
    ingest/.state/embed_missing.txt     — "<source>\t<document_id>" per missing doc

Exit 0 = every scraped doc has at least one point in the collection; exit 1 otherwise.
This is intentionally a coverage check, not an index-integrity proof: it does not validate
the current cleaned document-state hash, a complete contiguous chunk generation, payload
schema, or vector identity/dimensions. Do not use it alone as a production promotion gate.

Usage (from ingest/):
    .venv/bin/python scripts/verify_all_embedded.py --coverage-only
    .venv/bin/python scripts/verify_all_embedded.py --coverage-only --collection georgian_legal --sources matsne,ecd
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ root → `import ingest`

from ingest.config import load_config  # noqa: E402
from ingest.qdrant_store import make_client  # noqa: E402
from ingest.sources import SOURCES  # noqa: E402

STATE_DIR = Path(__file__).resolve().parents[1] / ".state"
QUARANTINE = Path(__file__).resolve().parents[1] / "snapshots" / "v1" / "quarantine.jsonl"


def quarantined_ids() -> dict[str, set[str]]:
    """{source: {document_id}} the snapshot hygiene pass excluded by design (mojibake,
    empty/near-empty bodies). These are intentionally NOT in the collection — counting
    them as missing would make a clean exit 0 unreachable forever."""
    out: dict[str, set[str]] = {}
    try:
        for ln in QUARANTINE.open(encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            src, did = d.get("source"), d.get("document_id")
            if src and did:
                out.setdefault(str(src), set()).add(str(did))
    except OSError:
        pass
    return out


def scraped_universe(artifacts_root: Path, only: set[str] | None) -> dict[str, set[str]]:
    """{source: {document_id}} across latest/ + every runs/*/items.jsonl.

    The id is derived per source exactly the way ingest does it —
    ``":".join(SourceSpec.id_fields values)`` — so ecd counts ``decision_document_id``,
    constcourt ``legal_id``, tbappeal ``slug``, etc. A raw ``document_id`` field alone
    would mismatch the embedded payloads (ecd items carry both).
    """
    out: dict[str, set[str]] = {}
    for src_dir in sorted(artifacts_root.iterdir()):
        if not src_dir.is_dir() or (only and src_dir.name not in only):
            continue
        spec = SOURCES.get(src_dir.name)
        if spec is None:
            print(f"  ! skipping {src_dir.name}: no SourceSpec")
            continue
        files = sorted(src_dir.glob("runs/*/items.jsonl"))
        latest = src_dir / "latest" / "items.jsonl"
        if latest.exists():
            files.insert(0, latest)
        if not files:
            continue
        ids: set[str] = set()
        unidentifiable = 0
        for f in files:
            try:
                with f.open(encoding="utf-8") as fh:
                    for ln in fh:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            item = json.loads(ln)
                        except json.JSONDecodeError:
                            unidentifiable += 1
                            continue
                        parts = [str(item[fld]) for fld in spec.id_fields
                                 if item.get(fld) not in (None, "")]
                        if parts:
                            ids.add(":".join(parts))
                        else:
                            unidentifiable += 1
            except OSError as e:
                print(f"  ! unreadable {f}: {e}")
        out[src_dir.name] = ids
        extra = f", {unidentifiable} lines without id" if unidentifiable else ""
        print(f"  scraped {src_dir.name}: {len(ids)} unique docs ({len(files)} files{extra})")
    return out


def no_text_ids(artifacts_root: Path, missing: dict[str, set[str]]) -> dict[str, set[str]]:
    """Which missing docs have an empty body in their latest scrape → zero chunks.

    ``chunk_document("")`` yields no chunks, so the embed job never upserts these — they
    are unembeddable as scraped, not pipeline losses. Latest occurrence wins, matching the
    embed jobs' last-wins dedup.
    """
    out: dict[str, set[str]] = {}
    for src, want in missing.items():
        if not want:
            continue
        spec = SOURCES[src]
        files = sorted(artifacts_root.glob(f"{src}/runs/*/items.jsonl"))
        latest = artifacts_root / src / "latest" / "items.jsonl"
        if latest.exists():
            files.insert(0, latest)
        body_len: dict[str, int] = {}
        for f in files:
            try:
                with f.open(encoding="utf-8") as fh:
                    for ln in fh:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            item = json.loads(ln)
                        except json.JSONDecodeError:
                            continue
                        parts = [str(item[fld]) for fld in spec.id_fields
                                 if item.get(fld) not in (None, "")]
                        did = ":".join(parts) if parts else None
                        if did in want:
                            body_len[did] = len((item.get("body_markdown") or "").strip())
            except OSError:
                continue
        out[src] = {d for d, n in body_len.items() if n == 0}
    return out


def embedded_universe(client, collection: str,
                      only: set[str] | None = None) -> dict[str, set[str]]:
    """{source: {document_id}} via one full payload-only scroll of the collection."""
    out: dict[str, set[str]] = {}
    offset = None
    seen = 0
    t0 = time.time()
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            with_payload=["source", "document_id"],
            with_vectors=False,
            limit=10_000,
            offset=offset,
        )
        for p in points:
            pl = p.payload or {}
            src, did = pl.get("source"), pl.get("document_id")
            if src and did and (only is None or str(src) in only):
                out.setdefault(str(src), set()).add(str(did))
        seen += len(points)
        if seen % 200_000 < 10_000:
            print(f"  ...scrolled {seen} points ({time.time() - t0:.0f}s)")
        if offset is None:
            break
    print(f"  scrolled {seen} points total in {time.time() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--collection", default=None)
    ap.add_argument("--sources", default=None, help="comma list; default = every artifacts subdir")
    ap.add_argument(
        "--coverage-only",
        action="store_true",
        help="explicitly acknowledge this does not verify generation integrity",
    )
    args = ap.parse_args()
    if not args.coverage_only:
        ap.error("ID coverage only; pass --coverage-only, or use scripts/verify_generation.py")

    cfg = load_config()
    collection = args.collection or cfg.collection_name
    only = None
    if args.sources:
        only = {s.strip() for s in args.sources.split(",") if s.strip()}
        unknown = {s for s in only if not (cfg.artifacts_root / s).is_dir()}
        if unknown:
            raise SystemExit(f"--sources names without an artifacts dir: {sorted(unknown)}")
    client = make_client(cfg)

    info = client.get_collection(collection)
    print(f"collection {collection!r}: {info.points_count} points, status={info.status}")

    print("scanning scraped artifacts...")
    scraped = scraped_universe(cfg.artifacts_root, only)
    print("scrolling embedded ids...")
    embedded = embedded_universe(client, collection, only)

    raw_missing = {src: scraped.get(src, set()) - embedded.get(src, set())
                   for src in scraped}
    print("classifying missing docs (empty body / quarantined)...")
    no_text = no_text_ids(cfg.artifacts_root, raw_missing)
    quarantine = quarantined_ids()

    per_source: dict[str, dict] = {}
    missing_lines: list[str] = []
    for src in sorted(set(scraped) | set(embedded)):
        s, e = scraped.get(src, set()), embedded.get(src, set())
        nt = no_text.get(src, set())
        # live empty-body first (can never chunk), then the snapshot hygiene quarantine
        # (mojibake/near-empty text excluded by design) — the rest is genuinely missing.
        q = (s - e - nt) & quarantine.get(src, set())
        missing = sorted(s - e - nt - q)
        per_source[src] = {
            "scraped_docs": len(s),
            "embedded_docs": len(s & e),
            "missing": len(missing),
            "missing_sample": missing[:20],
            "no_text": len(nt),                    # empty body → zero chunks → unembeddable
            "no_text_ids": sorted(nt)[:50],
            "quarantined": len(q),                 # excluded by snapshot hygiene, by design
            "quarantined_ids": sorted(q)[:50],
            "extra_embedded": len(e - s),  # in Qdrant but not in artifacts (informational)
        }
        missing_lines += [f"{src}\t{d}" for d in missing]

    total_scraped = sum(v["scraped_docs"] for v in per_source.values())
    total_missing = sum(v["missing"] for v in per_source.values())
    total_no_text = sum(v["no_text"] for v in per_source.values())
    total_quarantined = sum(v["quarantined"] for v in per_source.values())
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collection": collection,
        "points_count": info.points_count,
        "collection_status": str(info.status),
        "sources_filter": sorted(only) if only else None,
        "total_scraped_docs": total_scraped,
        "total_missing": total_missing,
        "total_no_text": total_no_text,
        "total_quarantined": total_quarantined,
        "per_source": per_source,
    }
    STATE_DIR.mkdir(exist_ok=True)
    # A --sources run is a partial view: never overwrite the global report the monitor
    # renders — a filtered run would show unscanned sources as falsely green.
    suffix = ".partial" if only else ""
    tmp = STATE_DIR / f"embed_coverage{suffix}.json.tmp"
    tmp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    tmp.replace(STATE_DIR / f"embed_coverage{suffix}.json")
    miss_file = STATE_DIR / f"embed_missing{suffix}.txt"
    miss_file.write_text("\n".join(missing_lines) + ("\n" if missing_lines else ""),
                         encoding="utf-8")

    print(f"\n{'source':<12} {'scraped':>9} {'embedded':>9} {'missing':>8} {'no-text':>8} "
          f"{'quarant.':>8} {'extra':>7}")
    for src, v in per_source.items():
        print(f"{src:<12} {v['scraped_docs']:>9} {v['embedded_docs']:>9} "
              f"{v['missing']:>8} {v['no_text']:>8} {v['quarantined']:>8} {v['extra_embedded']:>7}")
    print(f"\n{total_scraped - total_missing - total_no_text - total_quarantined}/{total_scraped} "
          f"scraped docs embedded · {total_missing} MISSING → "
          f"{miss_file if total_missing else '(none)'} · {total_no_text} empty-body "
          f"· {total_quarantined} quarantined (excluded by design)")
    print(f"report → {STATE_DIR / f'embed_coverage{suffix}.json'}")
    if total_missing:
        raise SystemExit(1)
    scope = f"sources {sorted(only)}" if only else "every source"
    print(f"✅ every embeddable scraped document in {scope} has at least one index point "
          "(ID coverage only; integrity was not verified).")


if __name__ == "__main__":
    main()
