#!/usr/bin/env python
"""Export every point of a collection as re-embeddable rows (improvement I6).

The I6 re-embed changes ONLY the embedded prefix (v2 headers) — chunks, point ids and
payloads must stay identical so the old and new indexes differ in exactly one variable.
The stored payload already carries everything ``build_embed_text`` needs (``heading`` is
the full heading_path joined with the same ``" > "`` separator), so re-embedding from
payloads is drift-free by construction: no re-chunking, no hygiene re-run.

Writes gzipped JSONL shards of {"id", "payload"} rows (~rows-per-file bounded so a flaky
uplink resumes per-file), plus a manifest with counts for the pod-side sanity check.

    .venv/bin/python scripts/reembed_export.py --out .state/reembed_v2/rows [--limit N]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.config import load_config  # noqa: E402
from ingest.qdrant_store import make_client  # noqa: E402

ROWS_PER_FILE = 200_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".state/reembed_v2/rows")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: stop after N rows")
    args = ap.parse_args()

    cfg = load_config()
    coll = args.collection or cfg.collection_name
    client = make_client(cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    file_idx = 0
    fh = None
    offset = None
    t0 = time.time()
    try:
        while True:
            points, offset = client.scroll(
                collection_name=coll, with_payload=True, with_vectors=False,
                limit=1024, offset=offset,
            )
            for pt in points:
                if total % ROWS_PER_FILE == 0:
                    if fh:
                        fh.close()
                    fh = gzip.open(out_dir / f"rows-{file_idx:04d}.jsonl.gz", "wt",
                                   encoding="utf-8", compresslevel=6)
                    file_idx += 1
                fh.write(json.dumps({"id": str(pt.id), "payload": pt.payload},
                                    ensure_ascii=False) + "\n")
                total += 1
                if args.limit and total >= args.limit:
                    offset = None
                    break
            if total % 102_400 < 1024:
                rate = total / max(time.time() - t0, 1)
                print(f"  {total:,} rows ({rate:,.0f}/s)", flush=True)
            if offset is None:
                break
    finally:
        if fh:
            fh.close()

    manifest = {"collection": coll, "rows": total, "files": file_idx,
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"exported {total:,} rows into {file_idx} file(s) at {out_dir}")


if __name__ == "__main__":
    main()
