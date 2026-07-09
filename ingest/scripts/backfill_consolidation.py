#!/usr/bin/env python
"""Backfill the ``is_consolidated`` flag onto matsne chunks already in Qdrant — no re-embed.

Consolidation was added to the ingest payload after the corpus was embedded, so the
existing ~1.85M matsne chunks carry no ``is_consolidated``. This script derives the flag
for every already-scraped matsne doc from its stored ``consolidated_publications`` field
(non-empty ⇔ the detail page had a #publication-switcher ⇔ the act has consolidated
versions) and writes it with a payload-only ``set_payload`` (content unchanged, so the
vectors are untouched). Newly-(re)scraped docs already get the flag through normal ingest.

Note: only the boolean is backfilled here; the exact ``consolidated_count`` is unknown for
pre-existing docs (the old scrape kept just the first switcher option) and is filled in
only when a doc is re-scraped with the current spider.

Usage (from ingest/):  .venv/bin/python scripts/backfill_consolidation.py [--run latest] [--batch 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ root → `import ingest`

from qdrant_client import models  # noqa: E402

from ingest.config import load_config  # noqa: E402
from ingest.pipeline import items_path  # noqa: E402


def _flush(client, collection: str, ids: list[str], value: bool) -> int:
    """set_payload is_consolidated=value on every chunk of the given matsne doc ids."""
    if not ids:
        return 0
    client.set_payload(
        collection_name=collection,
        payload={"is_consolidated": value},
        points=models.Filter(
            must=[
                models.FieldCondition(key="source", match=models.MatchValue(value="matsne")),
                models.FieldCondition(key="document_id", match=models.MatchAny(any=ids)),
            ]
        ),
        wait=True,
    )
    return len(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="latest", help="matsne run to read items from (default: latest)")
    ap.add_argument("--batch", type=int, default=300,
                    help="doc ids per set_payload call (smaller = gentler on Qdrant)")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="seconds to pause between set_payload calls (throttle Qdrant load)")
    ap.add_argument("--only-consolidated", action="store_true",
                    help="only set is_consolidated=true (skip the heavy false pass); the "
                    "high-value flag, and far lighter on Qdrant")
    ap.add_argument("--collection", help="override collection name")
    args = ap.parse_args()

    cfg = load_config()
    collection = args.collection or cfg.collection_name
    from ingest.qdrant_store import make_client

    client = make_client(cfg)

    # Ensure the new payload indexes exist on the pre-existing collection (ensure_collection
    # only creates them for a fresh collection). Idempotent — ignore "already exists".
    for field_name, schema in (
        ("is_consolidated", models.PayloadSchemaType.BOOL),
        ("consolidated_count", models.PayloadSchemaType.INTEGER),
    ):
        try:
            client.create_payload_index(collection, field_name=field_name, field_schema=schema)
            print(f"  created payload index: {field_name}")
        except Exception as exc:  # noqa: BLE001 - index likely already exists
            print(f"  payload index {field_name}: {exc}")

    path = items_path(cfg, "matsne", args.run)
    if not path.exists():
        raise SystemExit(f"No matsne items at {path}")

    seen: set[str] = set()
    true_ids: list[str] = []
    false_ids: list[str] = []
    n_true = n_false = n_written = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = doc.get("document_id")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            is_consolidated = bool((doc.get("consolidated_publications") or "").strip())
            if is_consolidated:
                true_ids.append(doc_id)
                n_true += 1
                if len(true_ids) >= args.batch:
                    n_written += _flush(client, collection, true_ids, True)
                    true_ids = []
                    time.sleep(args.sleep)  # throttle so a long run can't overwhelm Qdrant
            elif not args.only_consolidated:
                false_ids.append(doc_id)
                n_false += 1
                if len(false_ids) >= args.batch:
                    n_written += _flush(client, collection, false_ids, False)
                    false_ids = []
                    time.sleep(args.sleep)

    n_written += _flush(client, collection, true_ids, True)
    if not args.only_consolidated:
        n_written += _flush(client, collection, false_ids, False)

    print(
        f"Backfill done: {len(seen)} matsne docs "
        f"({n_true} consolidated, {n_false} not) → is_consolidated set on {n_written} docs."
    )


if __name__ == "__main__":
    main()
