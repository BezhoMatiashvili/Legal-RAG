#!/usr/bin/env python
"""In-domain fine-tune of the cross-encoder reranker (improvement.md P5 / plan ⑥).

Builds a ``(query, positive, negative)`` training set from validated golden pairs that are
DISJOINT from the current test set, mines hard negatives from live retrieval, and fine-tunes
``BAAI/bge-reranker-v2-m3`` (same classification head → the result is a pure ``RERANK_MODEL``
drop-in). A/B the output on golden_set_v2 rerank mode and KEEP only on a gated win.

⚠ CONTAMINATION GUARD (the P5 trap): NEVER train on a gold doc that appears in the eval TEST
set. This script hard-excludes every ``(source, document_id)`` in the test golden set AND in
``holdout_doc_ids_v2.json`` from BOTH positives and mined negatives, and refuses to run if too
few clean pairs remain (default floor 200 — small-data fine-tunes overfit/degrade; measured
2026-07-13: only 48 of the 100 v3-pending pairs are disjoint from v2 test, so this floor
intentionally BLOCKS until v3 authoring adds more clean pairs).

    # once v3 (or more clean pairs) exists:
    python scripts/finetune_reranker.py --pairs eval/.golden_v2_pending/batches/*/batch.jsonl \
        --test-golden eval/golden_set_v2.jsonl --holdout eval/holdout_doc_ids_v2.json \
        --out .state/reranker_ft --epochs 2 --min-clean-pairs 200
    # then A/B (KEEP only on a gate pass; rerank_model is in the retrieval fingerprint):
    RERANK_MODEL=.state/reranker_ft .venv/bin/python -m eval.eval_answer_quality \
        --backend qdrant --mode rerank --golden-set v2 --translate-queries eval/query_translations_v2.json
"""

import argparse
import glob
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # ingest/ on path for `ingest`/`eval`


def _load_pairs(patterns: list[str]) -> list[dict]:
    out = []
    for pat in patterns:
        for path in glob.glob(pat):
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if line and not line.startswith("#"):
                    out.append(json.loads(line))
    return out


def _excluded_docs(test_golden: Path, holdout: Path) -> set[tuple[str, str]]:
    """The (source, document_id) set that must NEVER be trained on (test + reserved)."""
    ex: set[tuple[str, str]] = set()
    for line in open(test_golden, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            d = json.loads(line)
            ex.add((d["gold"]["source"], d["gold"]["document_id"]))
    if holdout.exists():
        for x in json.load(open(holdout, encoding="utf-8")):
            ex.add((x["source"], x["document_id"]))
    return ex


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the bge cross-encoder reranker (P5).")
    ap.add_argument("--pairs", nargs="+", required=True, help="golden-pair JSONL glob(s) for training")
    ap.add_argument("--test-golden", type=Path, default=Path("eval/golden_set_v2.jsonl"))
    ap.add_argument("--holdout", type=Path, default=Path("eval/holdout_doc_ids_v2.json"))
    ap.add_argument("--out", type=Path, default=Path(".state/reranker_ft"))
    ap.add_argument("--negs-per-pos", type=int, default=4, help="hard negatives mined per positive")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-clean-pairs", type=int, default=200,
                    help="refuse to train below this many clean pairs (overfit guard)")
    ap.add_argument("--seed", type=int, default=20260713)
    ap.add_argument("--dry-run", action="store_true", help="build+report the training set, don't train")
    args = ap.parse_args()

    from ingest.config import load_config
    from ingest.embedding import BGEM3Embedder
    from ingest.qdrant_store import make_client, point_id
    from ingest.search import hybrid_search

    cfg = load_config()
    excluded = _excluded_docs(args.test_golden, args.holdout)
    pairs = _load_pairs(args.pairs)
    clean = [p for p in pairs if (p["gold"]["source"], p["gold"]["document_id"]) not in excluded]
    print(f"pairs={len(pairs)}  excluded_by_contamination_guard={len(pairs) - len(clean)}  clean={len(clean)}")
    if len(clean) < args.min_clean_pairs:
        raise SystemExit(
            f"REFUSING: only {len(clean)} clean pairs < floor {args.min_clean_pairs}. Small-data "
            f"fine-tunes of a 568M cross-encoder overfit/degrade. Author more v3 pairs (disjoint "
            f"from the v2 test set) first, then re-run. (Lower --min-clean-pairs only to test the "
            f"pipeline, never to produce a serving model.)"
        )

    client = make_client(cfg)
    embedder = BGEM3Embedder(cfg)
    rng = random.Random(args.seed)

    # positive = gold chunk-0 text; hard negatives = top retrieved chunks from NON-gold, non-excluded docs
    from sentence_transformers import InputExample

    def chunk_text(source: str, doc: str, ci: int) -> str | None:
        recs = client.retrieve(cfg.collection_name, ids=[point_id(source, doc, ci)], with_payload=True)
        return (recs[0].payload or {}).get("text") if recs else None

    examples: list = []
    for p in clean:
        gs, gd = p["gold"]["source"], p["gold"]["document_id"]
        pos = chunk_text(gs, gd, 0)
        if not pos:
            continue
        try:
            pts = hybrid_search(cfg, client, embedder, p["query"], top_k=30,
                                reranker=None, rerank_candidates=30)
        except Exception:  # noqa: BLE001
            continue
        negs = []
        for pt in pts:
            pl = pt.payload or {}
            key = (pl.get("source"), pl.get("document_id"))
            if key != (gs, gd) and key not in excluded and pl.get("text"):
                negs.append(pl["text"])
            if len(negs) >= args.negs_per_pos:
                break
        examples.append(InputExample(texts=[p["query"], pos], label=1.0))
        for n in negs:
            examples.append(InputExample(texts=[p["query"], n], label=0.0))
    rng.shuffle(examples)
    print(f"training examples: {len(examples)} (pos+neg) from {len(clean)} clean queries")

    if args.dry_run:
        print("dry-run: not training. Training set is ready.")
        return

    from torch.utils.data import DataLoader

    from sentence_transformers.cross_encoder import CrossEncoder

    model = CrossEncoder(cfg.rerank_model, num_labels=1, max_length=512)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    args.out.mkdir(parents=True, exist_ok=True)
    model.fit(train_dataloader=loader, epochs=args.epochs, warmup_steps=max(10, len(loader) // 10),
              output_path=str(args.out), show_progress_bar=True)
    print(f"Wrote fine-tuned reranker to {args.out}. A/B it with RERANK_MODEL={args.out} "
          f"(rerank_model is in retrieval_fingerprint → re-baseline; KEEP only on a gated win).")


if __name__ == "__main__":
    main()
