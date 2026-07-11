#!/usr/bin/env python
"""Build an ADDITIVE snapshot delta (``snapshots/v2-delta/``) for specific document ids.

Snapshot v1 is frozen (its bodies anchor every golden-set span), but ~24k documents were
scraped/embedded after it was built and exist only in Qdrant — whose chunk payloads cannot
reproduce the exact ``body_markdown`` a gold span must anchor into. This script materializes
those bodies for a chosen id set by replaying the IDENTICAL v1 snapshot path over the raw
artifacts (``sources.normalize`` → ``hygiene.assess``/``clean_text`` → ``structure.detect``
→ ``snapshot._snapshot_record``), so ``eval.goldset.SnapshotBodies(extra_roots=…)`` can
re-ground v2 spans byte-exactly. Never touches ``snapshots/v1/`` (read-only guard).

Per requested doc, newest artifact run first:
  * ids already present in v1 are skipped (the multi-root loader would shadow them anyway);
  * with index verification (default ON) the doc's chunk-0 payload ``content_hash`` must
    match the artifact body — raw-space (pipeline-ingested lineage) or cleaned-space
    (snapshot-embedded lineage). A newer scrape that diverges from the embedded revision
    falls back through OLDER runs until one matches, so spans anchor into the text that is
    actually indexed; no match ⇒ hard fail (never guess);
  * docs the index lacks entirely, or that hygiene would quarantine, also hard-fail —
    a gold doc must be servable;
  * pipeline-ingested docs whose body changes under ``clean_text`` are written but flagged
    (``chunk_alignment_risk_ids``): their live chunk boundaries were computed over the RAW
    body while eval spans map over the CLEANED one — avoid them as gold docs.

Usage (from ingest/):
    .venv/bin/python scripts/build_snapshot_delta.py --ids delta_ids.json
    .venv/bin/python scripts/build_snapshot_delta.py --ids delta_ids.json --no-verify-index

``delta_ids.json`` = {"matsne": ["90052", ...], "ecd": [...], ...}. Output is a full
rewrite of ``snapshots/<version>/docs/<source>.jsonl`` (atomic tmp+rename, re-runnable)
plus ``manifest.json``. Exit 0 only if every requested id was written cleanly.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ root → `import ingest`

from ingest import dedup, hygiene, structure  # noqa: E402
from ingest.config import load_config  # noqa: E402
from ingest.snapshot import (  # noqa: E402
    DEFAULT_SNAPSHOT_ROOT,
    _run_files_desc,
    _snapshot_record,
)
from ingest.sources import SOURCES, normalize  # noqa: E402

DELTA_VERSION = "v2-delta"
_DOC_ID_RE = re.compile(r'"document_id": "([^"]*)"')


def load_id_map(path: Path) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    bad = sorted(set(data) - set(SOURCES))
    if bad:
        raise SystemExit(f"unknown sources in id map: {bad}")
    return {src: [str(i) for i in ids] for src, ids in data.items()}


def already_in_v1(source: str, ids: set[str], v1_docs: Path) -> set[str]:
    """Which of ``ids`` the frozen v1 snapshot already has (v1 opened READ-ONLY)."""
    path = v1_docs / f"{source}.jsonl"
    if not path.exists() or not ids:
        return set()
    found: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            # fast path: document_id is the 4th key _snapshot_record writes, so it sits in
            # the line's first few hundred bytes; fall back to full parse if the shape drifts
            m = _DOC_ID_RE.search(line[:600])
            did = m.group(1) if m else (json.loads(line)["document_id"] if line.strip() else None)
            if did in ids:
                found.add(did)
                if found == ids:
                    break
    return found


def fetch_payload_hashes(client, collection: str, id_map: dict[str, list[str]]) -> dict[tuple[str, str], str]:
    """{(source, document_id): chunk-0 payload content_hash} for ids present in the index."""
    from ingest.qdrant_store import point_id

    out: dict[tuple[str, str], str] = {}
    for source, ids in id_map.items():
        pids = {point_id(source, did, 0): did for did in ids}
        for batch_start in range(0, len(pids), 256):
            batch = list(pids)[batch_start : batch_start + 256]
            for pt in client.retrieve(collection_name=collection, ids=batch, with_payload=["content_hash"]):
                did = pids[str(pt.id)]
                ch = (pt.payload or {}).get("content_hash")
                if ch:
                    out[(source, did)] = ch
    return out


def build_delta(
    id_map: dict[str, list[str]],
    *,
    artifacts_root: Path,
    out_root: Path = DEFAULT_SNAPSHOT_ROOT,
    version: str = DELTA_VERSION,
    payload_hashes: dict[tuple[str, str], str] | None = None,
) -> dict:
    """Write the delta snapshot; returns the manifest. ``payload_hashes=None`` skips
    index verification (accept the newest artifact revision)."""
    if not version.startswith("v2"):
        raise ValueError(f"delta version must start with 'v2', got {version!r} — v1 is frozen")
    out_root = Path(out_root)
    out_dir = out_root / version
    if out_dir.resolve() == (out_root / "v1").resolve():
        raise ValueError("refusing to write into snapshots/v1 — it is frozen")
    v1_docs = out_root / "v1" / "docs"
    (out_dir / "docs").mkdir(parents=True, exist_ok=True)

    verify = payload_hashes is not None
    manifest: dict = {
        "delta_version": version,
        "base_snapshot": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verified_against_index": verify,
        "sources": {},
    }
    failed = False

    for source in sorted(id_map):
        requested = {str(i) for i in id_map[source]}
        skipped_v1 = already_in_v1(source, set(requested), v1_docs)
        want = requested - skipped_v1
        stats = {
            "requested": len(requested),
            "skipped_in_v1": sorted(skipped_v1),
            "written": 0,
            "missing_from_artifacts": [],
            "missing_from_index": [],
            "mismatch_unresolved": [],
            "quarantined": {},
            "recovered_from_older_run": {},
            "chunk_alignment_risk_ids": [],
            "malformed_lines": 0,
        }
        manifest["sources"][source] = stats

        if verify:
            missing_idx = sorted(d for d in want if (source, d) not in payload_hashes)
            stats["missing_from_index"] = missing_idx
            want -= set(missing_idx)

        records: dict[str, dict] = {}
        pending = set(want)  # ids still needing an (index-matching) artifact revision
        rejected_newer: set[str] = set()  # ids whose newer revision(s) didn't match the index
        for run_label, run_path in _run_files_desc(source, artifacts_root):
            if not pending:
                break
            with open(run_path, encoding="utf-8") as f:
                for line in f:
                    if not pending:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = normalize(source, json.loads(line))
                    except Exception:
                        stats["malformed_lines"] += 1
                        continue
                    did = doc.document_id
                    if did not in pending:
                        continue
                    raw = doc.body_markdown or ""
                    clean_body = hygiene.clean_text(raw)
                    lineage = None
                    if verify:
                        ph = payload_hashes[(source, did)]
                        if dedup.content_hash(raw) == ph:
                            lineage = "raw"
                        elif dedup.content_hash(clean_body) == ph:
                            lineage = "clean"
                        else:
                            rejected_newer.add(did)
                            continue  # not the indexed revision — keep looking in older runs
                    report = hygiene.assess(raw)
                    if not report.is_usable:
                        stats["quarantined"][did] = report.quarantine_reason
                        pending.discard(did)
                        continue
                    chash = dedup.content_hash(clean_body)
                    sinfo = structure.detect(clean_body)
                    rec = _snapshot_record(source, doc, clean_body, chash, sinfo, run_label)
                    rec["snapshot_version"] = version
                    records[did] = rec
                    pending.discard(did)
                    if did in rejected_newer:
                        stats["recovered_from_older_run"][did] = run_label
                    if lineage == "raw" and clean_body != raw:
                        # live chunks were cut over the RAW body; eval spans map over the
                        # CLEANED one — chunk_index may diverge. Avoid as a gold doc.
                        stats["chunk_alignment_risk_ids"].append(did)

        leftover = sorted(pending)
        if verify:
            stats["mismatch_unresolved"] = [d for d in leftover if (source, d) in payload_hashes]
            stats["missing_from_artifacts"] = [d for d in leftover if (source, d) not in payload_hashes]
        else:
            stats["missing_from_artifacts"] = leftover

        out_path = out_dir / "docs" / f"{source}.jsonl"
        tmp = out_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for did in sorted(records):
                f.write(json.dumps(records[did], ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)
        stats["written"] = len(records)

        if (
            stats["missing_from_artifacts"]
            or stats["missing_from_index"]
            or stats["mismatch_unresolved"]
            or stats["quarantined"]
        ):
            failed = True

    manifest["ok"] = not failed
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ids", required=True, metavar="PATH", help='JSON {"source": ["document_id", ...]}')
    ap.add_argument("--out-version", default=DELTA_VERSION)
    ap.add_argument("--collection", default=None, help="index to verify against (default: config)")
    ap.add_argument("--no-verify-index", action="store_true",
                    help="accept the newest artifact revision without checking the live index")
    args = ap.parse_args()

    cfg = load_config()
    id_map = load_id_map(Path(args.ids))

    payload_hashes = None
    if not args.no_verify_index:
        from ingest.qdrant_store import make_client

        client = make_client(cfg)
        payload_hashes = fetch_payload_hashes(client, args.collection or cfg.collection_name, id_map)

    manifest = build_delta(
        id_map,
        artifacts_root=Path(cfg.artifacts_root),
        version=args.out_version,
        payload_hashes=payload_hashes,
    )

    for source, stats in manifest["sources"].items():
        print(
            f"{source}: requested={stats['requested']} written={stats['written']} "
            f"in_v1={len(stats['skipped_in_v1'])} "
            f"missing_idx={len(stats['missing_from_index'])} "
            f"missing_art={len(stats['missing_from_artifacts'])} "
            f"mismatch={len(stats['mismatch_unresolved'])} "
            f"quarantined={len(stats['quarantined'])} "
            f"align_risk={len(stats['chunk_alignment_risk_ids'])}"
        )
        for did in stats["chunk_alignment_risk_ids"]:
            print(f"  WARNING chunk-alignment risk: {source}:{did} — do not use as a gold doc")
    if not manifest["ok"]:
        print("FAILED: some requested docs are missing/mismatched/quarantined (see manifest.json)")
        raise SystemExit(1)
    print("OK: delta snapshot complete")


if __name__ == "__main__":
    main()
