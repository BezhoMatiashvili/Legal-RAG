#!/usr/bin/env python
"""Prove every doc scraped in the delta runs is embedded in the main collection.

Collects every unique ``document_id`` across the given matsne run items files (default:
all runs whose id >= --runs-since) and exact-counts its chunks in ``georgian_legal``
(per-id, batched). Prints totals and writes any missing ids (docs with zero chunks) to
``--out`` for a follow-up embed:
    .venv/bin/python scripts/embed_delta.py --items <run>/items.jsonl   # or seed re-fetch

Usage (from ingest/):
    .venv/bin/python scripts/verify_delta_embedded.py --runs-since 20260709T080007Z
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ root → `import ingest`

from qdrant_client import models  # noqa: E402

from ingest.config import load_config  # noqa: E402
from ingest.qdrant_store import make_client  # noqa: E402


def delta_doc_ids(runs_dir: Path, since: str | None) -> dict[str, str]:
    """{document_id: title} for every unique doc in the selected run files (last wins)."""
    docs: dict[str, str] = {}
    for items in sorted(runs_dir.glob("*/items.jsonl")):
        if since and items.parent.name < since:
            continue
        for ln in items.open(encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            did = str(d.get("document_id") or "")
            if did:
                title = d.get("title") or ""
                if isinstance(title, list):
                    title = title[0] if title else ""
                docs[did] = str(title)[:80]
    return docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-since", default="20260709T080007Z")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--out", default="/tmp/missing_delta_ids.txt")
    args = ap.parse_args()

    cfg = load_config()
    collection = args.collection or cfg.collection_name
    client = make_client(cfg)

    runs_dir = cfg.artifacts_root / "matsne" / "runs"
    docs = delta_doc_ids(runs_dir, args.runs_since)
    print(f"verifying {len(docs)} unique delta docs against {collection!r}...")

    missing: list[str] = []
    total_chunks = 0
    for i, did in enumerate(sorted(docs), 1):
        n = client.count(
            collection_name=collection,
            count_filter=models.Filter(must=[
                models.FieldCondition(key="source", match=models.MatchValue(value="matsne")),
                models.FieldCondition(key="document_id", match=models.MatchValue(value=did)),
            ]),
            exact=True,
        ).count
        total_chunks += n
        if n == 0:
            missing.append(did)
        if i % 2000 == 0:
            print(f"  ...{i}/{len(docs)} checked, {len(missing)} missing so far")

    print(f"\n{len(docs) - len(missing)}/{len(docs)} docs embedded · {total_chunks} chunks total "
          f"· {len(missing)} MISSING")
    if missing:
        Path(args.out).write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"missing ids → {args.out}")
        for did in missing[:10]:
            print(f"  {did} | {docs[did]}")
        raise SystemExit(1)
    print("✅ every scraped delta doc is embedded and queryable.")


if __name__ == "__main__":
    main()
