#!/usr/bin/env python
"""Measure CPU cross-encoder rerank latency per candidate depth — safely.

The full `--mode rerank` eval loads BGE-M3 + the reranker together (~4.6 GB) alongside Qdrant
(~21 GB) and OOM-crashes on this box. Rerank latency is what we actually need from CPU (the
serving target), and it depends only on (#candidates × passage length), not on retrieval. So we
measure it in isolation: load ONLY the reranker (~2.3 GB → no OOM), grab a realistic pool of 80
real chunk texts once, and time reranker.score(query, texts[:depth]) over the golden queries for
depth ∈ {10,30,50,80}. Also writes a small (query, texts, cpu_scores) fixture so the GPU pod's
scores can be numerically cross-validated against CPU.

Output: prints a per-depth p50/p95 ms/query table; writes <scratch>/rerank_xval.json.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from ingest.config import load_config
from ingest.qdrant_store import make_client
from ingest.rerank import BGEReranker

DEPTHS = (10, 30, 50, 80)
N_QUERIES = 15
POOL = 80
XVAL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/rerank_xval.json")


def _golden_queries(n: int) -> list[str]:
    path = Path("eval/golden_set_v1.jsonl")
    qs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        qs.append(json.loads(line)["query"])
        if len(qs) >= n:
            break
    return qs


def main() -> int:
    cfg = load_config()
    client = make_client(cfg)
    # one scroll for a realistic candidate pool (real chunk texts, ~512 tokens each)
    pts, _ = client.scroll(cfg.collection_name, limit=POOL, with_payload=True, with_vectors=False)
    texts = [(p.payload or {}).get("text") or "" for p in pts]
    queries = _golden_queries(N_QUERIES)
    print(f"probe: {len(queries)} queries × pool {len(texts)} chunks "
          f"(avg {statistics.mean(len(t) for t in texts):.0f} chars)")

    reranker = BGEReranker(cfg)  # CPU, ~2.3 GB — no embedder, safe from OOM
    reranker.score(queries[0], texts[:8])  # warm up

    print(f"\n{'depth':>6} {'p50 ms/query':>14} {'p95 ms/query':>14} {'mean':>10}")
    results = {}
    for d in DEPTHS:
        per = []
        for q in queries:
            t0 = time.perf_counter()
            reranker.score(q, texts[:d])
            per.append((time.perf_counter() - t0) * 1000)
        per.sort()
        p50 = statistics.median(per)
        p95 = per[min(len(per) - 1, int(0.95 * len(per)))]
        results[d] = {"p50_ms": p50, "p95_ms": p95, "mean_ms": statistics.mean(per)}
        print(f"{d:>6} {p50:>14.0f} {p95:>14.0f} {statistics.mean(per):>10.0f}")

    # cross-validation fixture: exact CPU scores for a few (query, texts) to check GPU parity
    xval = []
    for q in queries[:3]:
        xval.append({"query": q, "texts": texts[:20], "cpu_scores": reranker.score(q, texts[:20])})
    XVAL.write_text(json.dumps({"depths": results, "xval": xval}, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {XVAL} (per-depth latency + GPU cross-val fixture)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
