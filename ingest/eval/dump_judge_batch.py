"""Produce a judging batch for the L2 faithfulness layer (see ``eval/judge_eval.py``).

Retrieves the top-k context for a (stratified, seeded) SAMPLE of golden queries and writes a
JSONL batch of ``(query, retrieved context chunk texts, gold answer)`` triples. An approved
manual or automated reviewer then reads the batch and emits one verdict per row (faithful /
correct / complete / abstained); ``judge_eval.py`` aggregates.

The retrieved chunk **texts are included in full and unmasked** (incl. PII) — the exact
context an answer would be composed from, matching answer-time exposure. This script is the
reviewer-agnostic producer: who judges the batch is a separate choice and does not change
this output.

    # sample 30 queries, dump their reranked context (production-faithful):
    RERANK_ENABLED=true python -m eval.dump_judge_batch --mode rerank --sample 30 \
        --golden-set v2 --translate-queries eval/query_translations_v2.json
    # cheaper recall-stage context (no reranker RAM):
    RERANK_ENABLED=false python -m eval.dump_judge_batch --mode hybrid --sample 30
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from ingest.config import load_config
from ingest.qdrant_store import point_id

from . import goldset
from .evaluate import make_backend


def _token_counter(kind: str, tokenizer_model: str, tokenizer_revision: str | None = None):
    if kind == "word":
        from ingest.chunking import default_token_counter

        return default_token_counter
    from ingest.embedding import make_token_counter

    return make_token_counter(tokenizer_model, tokenizer_revision)


def _stratified_sample(gold, n, seed):
    """Deterministic sample of ~n queries with proportional query_type coverage."""
    if n >= len(gold):
        return list(gold)
    by_type: dict[str, list] = {}
    for q in gold:
        by_type.setdefault(q.query_type, []).append(q)
    rng = random.Random(seed)
    picked: list = []
    for _t, qs in sorted(by_type.items()):
        k = max(1, round(n * len(qs) / len(gold)))
        picked.extend(rng.sample(qs, min(k, len(qs))))
    rng.shuffle(picked)
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump a judging batch for the L2 faithfulness layer.")
    ap.add_argument("--backend", choices=("qdrant",), default="qdrant")  # needs real chunk text
    ap.add_argument("--mode", choices=("hybrid", "rerank", "routed", "dense"), default="hybrid")
    ap.add_argument("--golden-set", choices=sorted(goldset.EVAL_SETS), default="v2")
    ap.add_argument("--sample", type=int, default=30, help="number of queries to judge")
    ap.add_argument("--k", type=int, default=8, help="context chunks per query")
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--tokenizer", choices=("word", "bge"), default="bge")
    ap.add_argument("--rerank-candidates", type=int, default=None)
    ap.add_argument("--citation-route", choices=("ids", "full"), default=None)
    ap.add_argument("--translate-queries", metavar="PATH", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    chunk_cfg = {"max_tokens": cfg.chunk_tokens, "overlap": cfg.chunk_overlap,
                 "min_tokens": cfg.chunk_min_tokens}
    count_tokens = _token_counter(
        args.tokenizer, cfg.tokenizer_model, cfg.tokenizer_revision
    )

    spec = goldset.EVAL_SETS[args.golden_set]
    gold = goldset.load_golden_set(spec.gold)
    sample = _stratified_sample(gold, args.sample, args.seed)
    bodies = goldset.SnapshotBodies(
        root=spec.roots[0], needed=goldset.gold_docs(sample), extra_roots=spec.roots[1:])

    translations = None
    if args.translate_queries:
        from .translations import load_query_translations

        translations, _ = load_query_translations(Path(args.translate_queries), gold)

    knobs = {"rerank_candidates": args.rerank_candidates, "translations": translations,
             "citation_route": args.citation_route}
    backend, info = make_backend("qdrant", cfg, sample, bodies, chunk_cfg, count_tokens, knobs=knobs)
    client = backend.client
    print(f"Sampled {len(sample)}/{len(gold)} queries (seed={args.seed}); backend={info}; mode={args.mode}")

    out_path = Path(args.out) if args.out else (
        cfg.state_dir / "answer_eval"
        / f"judge_batch_{spec.version}_{args.mode}_n{len(sample)}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for q in sample:
            hits, _lat = backend.search(q.query, args.mode, args.k)
            ids = [point_id(h.source, h.document_id, h.chunk_index) for h in hits]
            text_by_id: dict[str, str] = {}
            if ids:
                for r in client.retrieve(cfg.collection_name, ids=ids, with_payload=True):
                    text_by_id[str(r.id)] = (r.payload or {}).get("text", "")
            context = [
                {"rank": i + 1, "source": h.source, "document_id": h.document_id,
                 "chunk_index": h.chunk_index, "score": round(h.score, 4),
                 "text": text_by_id.get(pid, "")}
                for i, (h, pid) in enumerate(zip(hits, ids))
            ]
            row = {
                "id": q.id,
                "query": q.query,
                "query_type": q.query_type,
                "language": q.query_language,
                "gold_doc": [q.gold_source, q.gold_document_id],
                "gold_doc_title": q.doc_title,
                "gold_answer": q.answer,
                "context": context,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path} ({len(sample)} rows). Judge → verdicts JSONL → `python -m eval.score_judge`.")


if __name__ == "__main__":
    main()
