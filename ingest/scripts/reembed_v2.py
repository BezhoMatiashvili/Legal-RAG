#!/usr/bin/env python
"""Re-embed exported rows with I6 v2 headers into a Qdrant collection (pod or local).

Reads the ``reembed_export.py`` row shards, rebuilds the embed text from each payload —
v1 text is byte-reconstructible (title > document_type > heading), v2 inserts the
``№ · date · status · consolidation`` segment — encodes with BGE-M3 (dense + learned
sparse) and upserts under the SAME point id with the SAME payload. Idempotent; resumable
via a per-shard checkpoint; shardable by row index for multi-GPU pods.

    EMBED_DEVICE=cuda EMBED_USE_FP16=true python scripts/reembed_v2.py \
        --rows /workspace/rows --collection georgian_legal_v2 \
        --shard 0 --num-shards 4 [--header v2|v1] [--limit N]
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import models  # noqa: E402

from ingest.chunking import build_embed_text  # noqa: E402
import os  # noqa: E402

from ingest.config import load_config  # noqa: E402
from ingest.embedding import BGEM3Embedder  # noqa: E402
from ingest.qdrant_store import ensure_collection, make_client, sparse_vector  # noqa: E402

# Batch fed to the GPU per encode+upsert. 64 badly underutilizes a 4090 (measured ~47/s;
# batch 256 matches the production embed's ~128/s per GPU). Read from EMBED_BATCH_SIZE
# (the pod sets it to 256) so it's tunable without a code edit.
BATCH = int(os.getenv("EMBED_BATCH_SIZE") or 256)


def embed_text_from_payload(p: dict, *, v2: bool) -> str:
    """Rebuild the embed text from a stored payload (v1-exact, or v2 with the meta segment)."""
    kwargs = {}
    if v2:
        kwargs = {
            "document_number": p.get("document_number"),
            "date": p.get("date") or p.get("date_raw"),
            "status": p.get("status"),
            "is_consolidated": p.get("is_consolidated"),
        }
    heading = p.get("heading")
    return build_embed_text(
        p.get("text") or "",
        title=p.get("title"),
        document_type=p.get("document_type"),
        heading_path=[heading] if heading else None,
        **kwargs,
    )


def iter_rows(rows_dir: Path):
    for path in sorted(rows_dir.glob("rows-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--collection", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--header", choices=("v1", "v2"), default="v2")
    ap.add_argument("--limit", type=int, default=None, help="debug: embed at most N rows")
    args = ap.parse_args()

    cfg = load_config()
    client = make_client(cfg)
    if args.shard == 0:
        ensure_collection(client, dataclasses.replace(cfg, collection_name=args.collection))
    embedder = BGEM3Embedder(cfg)

    ckpt = Path(args.rows) / f".ckpt-{args.collection}-{args.shard}"
    done = int(ckpt.read_text()) if ckpt.exists() else 0

    texts: list[str] = []
    metas: list[tuple[str, dict]] = []
    seen = 0
    upserted = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal upserted
        if not texts:
            return
        embs = embedder.encode_passages(texts)
        points = []
        for (pid, payload), emb in zip(metas, embs):
            vec = {"dense": emb.dense}
            if emb.sparse.indices:
                vec["sparse"] = sparse_vector(emb.sparse)
            points.append(models.PointStruct(id=pid, vector=vec, payload=payload))
        client.upsert(collection_name=args.collection, points=points, wait=False)
        upserted += len(points)
        texts.clear()
        metas.clear()
        ckpt.write_text(str(seen))

    for i, row in enumerate(iter_rows(Path(args.rows))):
        if i % args.num_shards != args.shard:
            continue
        seen += 1
        if seen <= done:
            continue  # resume past the checkpoint
        texts.append(embed_text_from_payload(row["payload"], v2=args.header == "v2"))
        metas.append((row["id"], row["payload"]))
        if len(texts) >= BATCH:
            flush()
            if upserted % (BATCH * 100) == 0:
                rate = upserted / max(time.time() - t0, 1)
                print(f"shard {args.shard}: {upserted:,} upserted ({rate:.0f}/s)", flush=True)
        if args.limit and seen - done >= args.limit:
            break
    flush()
    print(f"shard {args.shard}: DONE — {upserted:,} upserted (seen {seen:,})")


if __name__ == "__main__":
    main()
