#!/usr/bin/env python
"""Embed the sweep's newly-scraped matsne docs (the delta) into a Qdrant collection.

Reads RAW scraped ``items.jsonl`` (which carry ``is_consolidated`` / ``consolidated_count``
/ ``status``) and normalizes each with ``sources.normalize`` — deliberately NOT the snapshot
loader (``embed_job.snapshot_doc_to_canonical`` drops consolidation), so the new docs keep
their consolidation metadata. Then chunks + BGE-M3 embeds (dense + learned-sparse) and
upserts into ``--collection``.

CPU here is far too slow for a ~300k-chunk delta (~15-33 s/chunk), so the real run is on a
RunPod GPU:
    EMBED_DEVICE=cuda ... python scripts/embed_delta.py --runs-since <id> --collection georgian_legal_delta
then snapshot ``georgian_legal_delta`` back and merge into main with
``merge_delta_collection.py``. Point ids are deterministic, so re-running is idempotent.

Usage (from ingest/):
    .venv/bin/python scripts/embed_delta.py --runs-since 20260709T080007Z --collection georgian_legal_delta
    .venv/bin/python scripts/embed_delta.py --items <path/items.jsonl> ... [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ root → `import ingest`

from ingest.config import load_config  # noqa: E402
from ingest.sources import normalize  # noqa: E402


def _iter_items(paths):
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _resolve_paths(cfg, args) -> list[Path]:
    if args.items:
        return [Path(x) for x in args.items]
    runs_dir = cfg.artifacts_root / "matsne" / "runs"
    paths = sorted(runs_dir.glob("*/items.jsonl"))
    if args.runs_since:
        paths = [p for p in paths if p.parent.name >= args.runs_since]
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", nargs="*", help="explicit items.jsonl path(s)")
    ap.add_argument("--source", default="matsne",
                    help="source name for sources.normalize (default: matsne)")
    ap.add_argument("--runs-since", help="include run dirs whose id >= this (e.g. 20260709T080007Z)")
    ap.add_argument("--collection", help="target collection (on the pod: georgian_legal_delta)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true",
                    help="normalize + report only; no model load, no embed")
    args = ap.parse_args()

    cfg = load_config()
    if args.collection:
        cfg = dataclasses.replace(cfg, collection_name=args.collection)

    paths = _resolve_paths(cfg, args)
    if not paths:
        raise SystemExit("no items.jsonl found (pass --items or --runs-since)")

    docs: dict[str, object] = {}  # dedup by document_id, last wins
    malformed = 0
    for item in _iter_items(paths):
        try:
            doc = normalize(args.source, item)
        except Exception:  # noqa: BLE001 - a malformed record must not abort the delta
            malformed += 1
            continue
        docs[doc.document_id] = doc
    docs = list(docs.values())
    n_cons = sum(1 for d in docs if d.is_consolidated)
    print(f"delta: {len(docs)} unique docs from {len(paths)} file(s) "
          f"({n_cons} consolidated, {malformed} malformed skipped) → {cfg.collection_name!r}")

    if args.dry_run:
        for d in docs[:6]:
            print(f"  {d.document_id} | status={d.status} | is_consolidated={d.is_consolidated} "
                  f"| n={d.consolidated_count} | {(d.title or '')[:40]}")
        return

    from ingest.embed_job import embed_docs
    from ingest.embedding import BGEM3Embedder, make_token_counter
    from ingest.qdrant_store import ensure_collection, make_client

    client = make_client(cfg)
    ensure_collection(client, cfg)
    print(f"loading BGE-M3 (device={cfg.embed_device or 'cpu'}, fp16={cfg.embed_use_fp16})...")
    embedder = BGEM3Embedder(cfg)
    count_tokens = make_token_counter(cfg.embed_model)

    d, c, k = embed_docs(cfg, client, embedder, count_tokens, docs, batch_size=args.batch_size)
    info = client.get_collection(cfg.collection_name)
    print(f"embedded {d} docs → {c} chunks ({k} skipped) into {cfg.collection_name!r}; "
          f"collection now {info.points_count} points.")


if __name__ == "__main__":
    main()
