#!/usr/bin/env python
"""In-domain fine-tune of the cross-encoder reranker (improvement.md P5 / plan ⑥).

Builds a ``(query, positive, negative)`` training set from validated golden pairs that are
DISJOINT from the current test set, mines hard negatives from live retrieval, and fine-tunes
``BAAI/bge-reranker-v2-m3`` (same classification head → the result is a pure ``RERANK_MODEL``
drop-in). A/B the output on golden_set_v2 rerank mode and KEEP only on a gated win.

⚠ CONTAMINATION GUARD (the P5 trap): NEVER train on a document/version lineage that appears
in the blind/test set. This script hard-excludes every declared ``version_family`` (falling
back to ``source:document_id`` for legacy data) from both positives and unrelated negatives.
Wrong passages, neighboring articles, and alternate versions of a *training* authority remain
eligible hard negatives. The script refuses to run if too few clean pairs remain (default
floor 200 — small-data fine-tunes overfit/degrade; measured
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
from typing import Any, Iterable

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


HARD_NEGATIVE_KINDS = frozenset({
    "wrong_passage_same_document",
    "similar_law_or_case",
    "neighboring_article",
    "current_or_repealed_version",
})
_HARD_NEGATIVE_PRIORITY = (
    "wrong_passage_same_document",
    "neighboring_article",
    "current_or_repealed_version",
    "similar_law_or_case",
)


def _record_family(record: dict[str, Any]) -> str:
    """Return an explicit lineage family, with a conservative legacy fallback."""

    gold = record.get("gold") if isinstance(record.get("gold"), dict) else record
    family = (
        gold.get("version_family")
        or record.get("version_family")
        or record.get("document_version_family")
    )
    if family:
        return str(family)
    source, document_id = gold.get("source"), gold.get("document_id")
    if not source or not document_id:
        raise ValueError("training/holdout row lacks version_family and document identity")
    return f"{source}:{document_id}"


def _excluded_families(test_golden: Path, holdout: Path) -> set[str]:
    """Document/version families that must never enter training."""

    excluded: set[str] = set()
    for line in open(test_golden, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            excluded.add(_record_family(json.loads(line)))
    if holdout.exists():
        for x in json.load(open(holdout, encoding="utf-8")):
            excluded.add(_record_family(x))
    return excluded


def _payload_family(payload: dict[str, Any]) -> str | None:
    family = (
        payload.get("version_family")
        or payload.get("document_version_family")
        or payload.get("version_lineage_id")
    )
    if family:
        return str(family)
    source, document_id = payload.get("source"), payload.get("document_id")
    return f"{source}:{document_id}" if source and document_id else None


def _negative_kind(payload: dict[str, Any], gold: dict[str, Any]) -> str | None:
    """Classify a candidate, excluding the annotated positive itself."""

    same_document = (
        payload.get("source") == gold.get("source")
        and payload.get("document_id") == gold.get("document_id")
    )
    if not same_document:
        return "similar_law_or_case"
    same_version = payload.get("version_id") == gold.get("version_id")
    same_chunk = payload.get("chunk_index") == int(gold["chunk_index"])
    if same_version and same_chunk:
        return None
    if not same_version:
        return "current_or_repealed_version"
    gold_article = gold.get("article_id")
    candidate_article = payload.get("article_id")
    if gold_article and candidate_article and str(gold_article) != str(candidate_article):
        return "neighboring_article"
    return "wrong_passage_same_document"


def _select_hard_negatives(
    points: Iterable[Any],
    *,
    gold: dict[str, Any],
    positive_family: str,
    excluded_families: set[str],
    limit: int,
) -> list[tuple[str, str]]:
    """Select a balanced, leakage-free set of hard-negative texts and classes."""

    by_kind: dict[str, list[str]] = {kind: [] for kind in HARD_NEGATIVE_KINDS}
    seen_text: set[str] = set()
    for point in points:
        payload = getattr(point, "payload", None) or {}
        text = payload.get("text")
        family = _payload_family(payload)
        if (
            not isinstance(text, str)
            or not text.strip()
            or text in seen_text
            or family is None
            or (family in excluded_families and family != positive_family)
        ):
            continue
        kind = _negative_kind(payload, gold)
        if kind is None:
            continue
        seen_text.add(text)
        by_kind[kind].append(text)

    selected: list[tuple[str, str]] = []
    for kind in _HARD_NEGATIVE_PRIORITY:
        if by_kind[kind] and len(selected) < limit:
            selected.append((by_kind[kind].pop(0), kind))
    for kind in _HARD_NEGATIVE_PRIORITY:
        for text in by_kind[kind]:
            if len(selected) >= limit:
                return selected
            selected.append((text, kind))
    return selected


def _load_cross_encoder(model_name: str, revision: str | None):
    from sentence_transformers.cross_encoder import CrossEncoder

    revision_kwargs = {"revision": revision} if revision is not None else {}
    return CrossEncoder(
        model_name, num_labels=1, max_length=512, **revision_kwargs
    )


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
    if args.negs_per_pos < len(HARD_NEGATIVE_KINDS):
        raise SystemExit(
            f"REFUSING: --negs-per-pos must be at least {len(HARD_NEGATIVE_KINDS)} "
            "to represent every required hard-negative class"
        )
    excluded = _excluded_families(args.test_golden, args.holdout)
    pairs = _load_pairs(args.pairs)
    clean = [p for p in pairs if _record_family(p) not in excluded]
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

    # Positive = the annotated evidence chunk (including immutable version identity), never
    # an assumed chunk zero. Legacy v2 pairs may omit these fields only on legacy collections.
    from sentence_transformers import InputExample

    def chunk_text(
        source: str, doc: str, ci: int, *, version_id: str | None
    ) -> str | None:
        if cfg.generation_id and not version_id:
            raise ValueError(
                "generation reranker pairs require an annotated gold.version_id"
            )
        recs = client.retrieve(
            cfg.collection_name,
            ids=[point_id(
                source,
                doc,
                ci,
                version_id=version_id if cfg.generation_id else None,
            )],
            with_payload=True,
        )
        return (recs[0].payload or {}).get("text") if recs else None

    examples: list = []
    mined_kinds: set[str] = set()
    for p in clean:
        gs, gd = p["gold"]["source"], p["gold"]["document_id"]
        if "chunk_index" not in p["gold"]:
            raise ValueError(
                "reranker positives require an annotated gold.chunk_index"
            )
        gold_chunk_index = int(p["gold"].get("chunk_index", 0))
        pos = chunk_text(
            gs,
            gd,
            gold_chunk_index,
            version_id=p["gold"].get("version_id"),
        )
        if not pos:
            continue
        try:
            global_points = hybrid_search(
                cfg, client, embedder, p["query"], top_k=80,
                reranker=None, rerank_candidates=80,
            )
            document_points = hybrid_search(
                cfg, client, embedder, p["query"], top_k=80,
                reranker=None, rerank_candidates=80,
                source=gs, document_id=gd,
            )
        except Exception:  # noqa: BLE001
            continue
        negs = _select_hard_negatives(
            (*document_points, *global_points),
            gold=p["gold"],
            positive_family=_record_family(p),
            excluded_families=excluded,
            limit=args.negs_per_pos,
        )
        if not negs:
            continue
        examples.append(InputExample(texts=[p["query"], pos], label=1.0))
        for n, kind in negs:
            examples.append(InputExample(texts=[p["query"], n], label=0.0))
            mined_kinds.add(kind)
    missing_kinds = HARD_NEGATIVE_KINDS - mined_kinds
    if missing_kinds:
        raise SystemExit(
            "REFUSING: mined training set lacks required hard-negative classes: "
            + ", ".join(sorted(missing_kinds))
        )
    rng.shuffle(examples)
    print(f"training examples: {len(examples)} (pos+neg) from {len(clean)} clean queries")

    if args.dry_run:
        print("dry-run: not training. Training set is ready.")
        return

    from torch.utils.data import DataLoader

    model = _load_cross_encoder(cfg.rerank_model, cfg.reranker_revision)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    args.out.mkdir(parents=True, exist_ok=True)
    model.fit(train_dataloader=loader, epochs=args.epochs, warmup_steps=max(10, len(loader) // 10),
              output_path=str(args.out), show_progress_bar=True)
    print(f"Wrote fine-tuned reranker to {args.out}. A/B it with RERANK_MODEL={args.out} "
          f"(rerank_model is in retrieval_fingerprint → re-baseline; KEEP only on a gated win).")


if __name__ == "__main__":
    main()
