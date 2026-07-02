"""Retrieval evaluation harness for the Georgian legal RAG index.

Measures how well ``legal_search`` finds the right *document* for a set of labelled
queries, and compares the hybrid pipeline with the cross-encoder reranker ON vs OFF.
This is how we prove a change actually improves search (and how we tune
``RERANK_MIN_SCORE`` / candidate depth) instead of guessing.

Ground truth without hand-curated document_ids: each query in ``queries.jsonl`` names
its target by an *exact identifier* (``gold``, e.g. {"source": "matsne",
"document_number": "55"}). At eval time we resolve that to the concrete set of
(source, document_id) docs by filtering the live index — so the gold set stays correct
as the corpus is re-ingested. The query string itself is natural language, so dense,
sparse and rerank are all genuinely exercised.

Usage (from the ingest/ project root):

    uv run python -m eval.evaluate                 # rerank off vs on, default queries.jsonl
    uv run python -m eval.evaluate --queries my.jsonl --top-k 10
    uv run python -m eval.evaluate --only-rerank   # just the reranked config

Metrics (document-level, binary relevance):
  recall@k  fraction of queries with at least one gold doc in the top-k
  MRR       mean reciprocal rank of the first gold doc (0 if not found)
  nDCG@k    normalised discounted cumulative gain
"""

import argparse
import json
import math
from pathlib import Path

from ingest.config import load_config
from ingest.qdrant_store import make_client
from ingest.search import build_filter, hybrid_search

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES = EVAL_DIR / "queries.jsonl"


def _load_queries(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def _scroll_doc_ids(client, collection, gold: dict) -> set[tuple]:
    """Resolve a gold identifier spec to the set of (source, document_id) it matches."""
    flt = build_filter(**gold)
    ids: set[tuple] = set()
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection, scroll_filter=flt,
            with_payload=True, with_vectors=False, limit=256, offset=offset,
        )
        for pt in batch:
            p = pt.payload or {}
            ids.add((p.get("source"), p.get("document_id")))
        if offset is None:
            break
    return ids


def _doc_rank_list(hits) -> list[tuple]:
    """Collapse chunk hits to a de-duplicated, rank-ordered list of (source, document_id)."""
    seen: set[tuple] = set()
    order: list[tuple] = []
    for h in hits:
        p = h.payload or {}
        key = (p.get("source"), p.get("document_id"))
        if key not in seen:
            seen.add(key)
            order.append(key)
    return order


def _metrics(ranked: list[tuple], gold: set[tuple], k: int) -> dict:
    top = ranked[:k]
    hit_rank = next((i + 1 for i, d in enumerate(top) if d in gold), None)
    recall = 1.0 if hit_rank else 0.0
    mrr = 1.0 / hit_rank if hit_rank else 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(top) if d in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    ndcg = dcg / ideal if ideal else 0.0
    return {"recall": recall, "mrr": mrr, "ndcg": ndcg, "rank": hit_rank}


def _build_reranker(cfg):
    from ingest.rerank import BGEReranker

    print(f"Loading reranker {cfg.rerank_model}...")
    return BGEReranker(cfg)


def _run_config(cfg, client, embedder, reranker, queries, golds, k):
    agg = {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    rows = []
    for q, gold in zip(queries, golds):
        hits = hybrid_search(
            cfg, client, embedder, q["query"], top_k=k,
            reranker=reranker,
            rerank_candidates=cfg.rerank_candidates,
            rerank_min_score=None,  # measure ranking quality; the gate is tuned separately
            **q.get("filters", {}),
        )
        m = _metrics(_doc_rank_list(hits), gold, k)
        rows.append((q, m))
        for key in agg:
            agg[key] += m[key]
    n = len(queries) or 1
    return {key: v / n for key, v in agg.items()}, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate legal RAG retrieval quality.")
    ap.add_argument("--queries", default=str(DEFAULT_QUERIES))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--only-rerank", action="store_true", help="skip the rerank-off baseline")
    ap.add_argument("--verbose", action="store_true", help="print per-query ranks")
    args = ap.parse_args()

    cfg = load_config()
    client = make_client(cfg)
    from ingest.embedding import BGEM3Embedder

    print(f"Loading embedder {cfg.embed_model}...")
    embedder = BGEM3Embedder(cfg)

    queries = _load_queries(Path(args.queries))
    golds = [_scroll_doc_ids(client, cfg.collection_name, q["gold"]) for q in queries]

    unresolved = [q["query"] for q, g in zip(queries, golds) if not g]
    if unresolved:
        print(f"\n⚠ {len(unresolved)} queries have no matching gold doc in the index "
              f"(not ingested yet?):")
        for u in unresolved:
            print(f"   - {u}")
    usable = [(q, g) for q, g in zip(queries, golds) if g]
    if not usable:
        print("\nNo evaluable queries — run the ingest first, then re-run.")
        return
    queries, golds = [q for q, _ in usable], [g for _, g in usable]
    print(f"\nEvaluating {len(queries)} queries (top_k={args.top_k}) "
          f"against collection '{cfg.collection_name}'.\n")

    configs = []
    if not args.only_rerank:
        configs.append(("rerank OFF (RRF)", None))
    configs.append((f"rerank ON ({cfg.rerank_model.split('/')[-1]})", _build_reranker(cfg)))

    print(f"{'config':<34} {'recall@k':>9} {'MRR':>7} {'nDCG@k':>8}")
    print("-" * 62)
    detail = {}
    for label, reranker in configs:
        agg, rows = _run_config(cfg, client, embedder, reranker, queries, golds, args.top_k)
        detail[label] = rows
        print(f"{label:<34} {agg['recall']:>9.3f} {agg['mrr']:>7.3f} {agg['ndcg']:>8.3f}")

    if args.verbose:
        for label, rows in detail.items():
            print(f"\n## {label}")
            for q, m in rows:
                r = m["rank"] if m["rank"] else "—"
                print(f"  rank={str(r):>3}  {q['query']}")


if __name__ == "__main__":
    main()
