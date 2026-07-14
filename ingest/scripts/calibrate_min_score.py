#!/usr/bin/env python
"""Calibrate the abstention threshold for `legal_search` (improvement I4).

Runs the golden queries through the production retrieval mode (rerank) and collects the
top-1 reranker score per query, split by whether the query's top-10 actually contains a
gold chunk. The abstention contract in the `legal_search` docstring ("below score X,
report not-found instead of citing the nearest match") gets its X from this data: the
threshold that keeps >=99% of gold-hitting queries.

Also informs whether serving RERANK_MIN_SCORE (default 0.3) should move.

Usage (GPU rerank via tunnel recommended — CPU is ~2h):
    RERANK_ENABLED=true RERANK_REMOTE_URL=http://localhost:8900 \
      .venv/bin/python scripts/calibrate_min_score.py \
      --translate-queries eval/query_translations_v1.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import goldset  # noqa: E402
from eval.evaluate import build_query_relevance, qdrant_deps  # noqa: E402
from ingest.config import load_config  # noqa: E402


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, max(0, round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-candidates", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--translate-queries", metavar="PATH", default=None)
    ap.add_argument("--out", default=".state/min_score_calibration.json")
    args = ap.parse_args()

    cfg = load_config()
    chunk_cfg = {"max_tokens": cfg.chunk_tokens, "overlap": cfg.chunk_overlap,
                 "min_tokens": cfg.chunk_min_tokens}
    from ingest.embedding import make_token_counter
    count_tokens = make_token_counter(cfg.tokenizer_model, cfg.tokenizer_revision)

    gold = goldset.load_golden_set()
    bodies = goldset.SnapshotBodies(needed=goldset.gold_docs(gold))
    goldset.reground(gold, bodies)
    rel = build_query_relevance(gold, bodies, chunk_cfg, count_tokens)

    translations = {}
    if args.translate_queries:
        from eval.translations import load_query_translations
        translations, _ = load_query_translations(Path(args.translate_queries), gold)

    client, embedder, reranker, info = qdrant_deps(cfg)
    if reranker is None:
        raise SystemExit("RERANK_ENABLED must be true (use the GPU tunnel for speed)")
    from eval.backend import QdrantBackend
    backend = QdrantBackend(cfg, client, embedder, reranker=reranker,
                            rerank_candidates=args.rerank_candidates,
                            translations=translations or None)
    print(f"backend: {info}  rc={args.rerank_candidates}  translated={bool(translations)}")

    rows = []
    for i, q in enumerate(gold, 1):
        hits, _ = backend.search(q.query, "rerank", args.top_k)
        gold_keys = set(rel[q.id]["chunk"])
        top1 = hits[0] if hits else None
        rows.append({
            "id": q.id, "query_type": q.query_type, "lang": q.query_language,
            "top1_score": top1.score if top1 else None,
            "top1_is_gold": bool(top1) and (top1.source, top1.document_id, top1.chunk_index) in gold_keys,
            "any_top10_gold": any((h.source, h.document_id, h.chunk_index) in gold_keys for h in hits),
        })
        if i % 20 == 0:
            print(f"  {i}/{len(gold)}")

    hit = sorted(r["top1_score"] for r in rows if r["any_top10_gold"] and r["top1_score"] is not None)
    miss = sorted(r["top1_score"] for r in rows if not r["any_top10_gold"] and r["top1_score"] is not None)
    print(f"\nqueries: {len(rows)}  gold-hit@10: {len(hit)}  miss: {len(miss)}")
    print(f"{'pct':>5} {'hit top1':>9} {'miss top1':>10}")
    for p in (1, 5, 10, 25, 50, 75, 90):
        print(f"{p:>5} {percentile(hit, p):>9.3f} {percentile(miss, p):>10.3f}")
    thr = percentile(hit, 1)
    kept = sum(1 for s in hit if s >= thr) / (len(hit) or 1)
    rejected = sum(1 for s in miss if s < thr) / (len(miss) or 1)
    print(f"\nthreshold keeping >=99% of gold hits: {thr:.3f} "
          f"(keeps {kept:.1%} of hits, abstains on {rejected:.1%} of misses)")
    print(f"current serving RERANK_MIN_SCORE: {cfg.rerank_min_score}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": info, "rerank_candidates": args.rerank_candidates,
                               "translated": bool(translations), "threshold_99": thr,
                               "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
