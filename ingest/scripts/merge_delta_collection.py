#!/usr/bin/env python
"""Merge a delta Qdrant collection into the main collection (upsert, never replace).

The full-corpus GPU embed produced the whole collection and loaded it with a snapshot
*restore* (which replaces the target). An INCREMENTAL embed — the sweep's ~23k newly
scraped docs — instead lands in a small ``georgian_legal_delta`` collection (embedded on
the RunPod GPU, snapshot-transferred back and restored locally under that name). This
script upserts every point of the delta collection into the main ``georgian_legal``.

Point ids are deterministic UUIDv5 over ``(source, document_id, chunk_index)``
(``qdrant_store.point_id``), so the upsert is idempotent and additive: re-running never
duplicates, a re-embedded doc's chunks overwrite in place, and only genuinely new chunks
grow the collection. Dense (1024-d) + learned-sparse vectors and the full payload are
copied verbatim.

Usage (from ingest/):
    .venv/bin/python scripts/merge_delta_collection.py --src georgian_legal_delta
    .venv/bin/python scripts/merge_delta_collection.py --dry-run   # self-test, no main mutation
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ root → `import ingest`

from qdrant_client import models  # noqa: E402

from ingest.config import load_config  # noqa: E402
from ingest.qdrant_store import ensure_collection, make_client  # noqa: E402


def _to_struct(point) -> models.PointStruct:
    """Rebuild an upsertable PointStruct from a scrolled Record (named dense+sparse vectors)."""
    vec = point.vector
    sparse = vec["sparse"]
    if not isinstance(sparse, models.SparseVector):
        sparse = models.SparseVector(indices=sparse["indices"], values=sparse["values"])
    return models.PointStruct(
        id=point.id,
        vector={"dense": vec["dense"], "sparse": sparse},
        payload=point.payload,
    )


def merge_collection(client, src: str, dst: str, *, batch_size: int = 256) -> int:
    """Upsert every point of ``src`` into ``dst``. Returns the number of points merged."""
    offset = None
    total = 0
    while True:
        points, offset = client.scroll(
            collection_name=src,
            limit=batch_size,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        if points:
            client.upsert(collection_name=dst, points=[_to_struct(p) for p in points], wait=True)
            total += len(points)
        if offset is None:
            break
    return total


def _count(client, name: str) -> int:
    return client.count(collection_name=name, exact=True).count


def dry_run(cfg, client) -> None:
    """Prove the merge on real embedded data (the firearms 495 chunks) — main is untouched."""
    src, dst = "_merge_test_src", "_merge_test_dst"
    for name in (src, dst):
        ensure_collection(client, dataclasses.replace(cfg, collection_name=name), recreate=True)

    # Seed src with the firearms points copied out of the main collection (real vectors+payload).
    seeded, offset = 0, None
    while True:
        pts, offset = client.scroll(
            collection_name=cfg.collection_name,
            limit=256, with_vectors=True, with_payload=True, offset=offset,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="document_id", match=models.MatchAny(any=["18070", "14944"]))
            ]),
        )
        if pts:
            client.upsert(collection_name=src, points=[_to_struct(p) for p in pts], wait=True)
            seeded += len(pts)
        if offset is None:
            break
    print(f"  seeded src with {seeded} firearms points")

    merged = merge_collection(client, src, dst)
    after_first = _count(client, dst)
    merge_collection(client, src, dst)  # idempotency: a second merge must not grow dst
    after_second = _count(client, dst)

    sample, _ = client.scroll(
        collection_name=dst, limit=1, with_vectors=True, with_payload=True,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="document_id", match=models.MatchValue(value="14944"))
        ]),
    )
    dense_ok = bool(sample) and len(sample[0].vector["dense"]) == cfg.dense_dim
    sparse_ok = bool(sample) and len(sample[0].vector["sparse"].indices) > 0
    payload_ok = bool(sample) and sample[0].payload.get("is_consolidated") is True

    print(f"  merged={merged} · dst after 1st={after_first} · after 2nd={after_second} "
          f"· idempotent={after_first == after_second}")
    print(f"  round-trip sample: dense_dim={len(sample[0].vector['dense']) if sample else None} "
          f"· sparse_terms={len(sample[0].vector['sparse'].indices) if sample else None} "
          f"· is_consolidated={sample[0].payload.get('is_consolidated') if sample else None}")

    for name in (src, dst):
        client.delete_collection(name)

    if not (seeded == after_first == after_second and dense_ok and sparse_ok and payload_ok):
        raise SystemExit("❌ DRY-RUN FAILED — see counts above")
    print(f"✅ DRY-RUN PASSED — {seeded} points merged idempotently with dense+sparse+payload "
          f"intact; temp collections dropped, main untouched.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", help="delta collection to merge from (e.g. georgian_legal_delta)")
    ap.add_argument("--dst", help="target collection (default: config collection_name)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true",
                    help="self-test the merge on the firearms sample without touching main")
    args = ap.parse_args()

    cfg = load_config()
    client = make_client(cfg)

    if args.dry_run:
        dry_run(cfg, client)
        return

    if not args.src:
        raise SystemExit("--src is required (or use --dry-run)")
    dst = args.dst or cfg.collection_name
    before = _count(client, dst)
    merged = merge_collection(client, args.src, dst, batch_size=args.batch_size)
    after = _count(client, dst)
    print(f"Merged {merged} points from {args.src!r} into {dst!r}: "
          f"{before} → {after} (+{after - before} net new).")


if __name__ == "__main__":
    main()
