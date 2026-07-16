"""One-off answer-quality eval runner that BYPASSES eval.evaluate.qdrant_deps()'s
GENERATION_DIR production-parity manifest/provenance check.

Why this exists: eval.evaluate.qdrant_deps() (the function eval.eval_answer_quality normally
uses via make_backend(...)) unconditionally raises RuntimeError without a verified
GENERATION_DIR manifest. No manifest producer exists yet for the live corpus (repo-wide,
2026-07-14) — see coordination/messages.md. This script builds the same (client, embedder,
reranker, info) tuple qdrant_deps() would, MINUS the manifest/provenance verification, and
injects it via eval.backend.make_backend(..., deps=...) — a parameter that already exists for
exactly this purpose. Read-only: no Qdrant writes, no RunPod spend, no model re-embed. It only
runs retrieval + scoring queries against the already-embedded live index.

This intentionally skips: generation manifest identity, collection-compatibility proof,
evaluation-provenance completeness. The retrieval_fingerprint stamped in the output JSON still
reflects the exact live Config, so results remain comparable across runs of THIS script (that is
what the reranker knob A/B experiment actually needs — internal comparability, not the full
production-parity provenance chain).

Usage mirrors eval.eval_answer_quality:
    RERANK_ENABLED=true .venv/bin/python scripts/eval_local_bypass_run.py \
        --mode rerank --golden-set v2 \
        --translate-queries eval/query_translations_v2.json --citation-route ids \
        --out .state/answer_eval/answer_eval_v2_rerank_knobA_512.json

Run from ingest/. RERANK_REMOTE_URL must be UNSET so max_length (knob B) actually takes effect
locally (a remote pod reranker ignores RERANK_MAX_LENGTH set in this process's env).
"""

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ root → `import eval`/`import ingest`

from eval import goldset
from eval.answer_eval import (
    DEFAULT_ABSTENTION_THRESHOLD,
    aggregate_answer_scores,
    breakdown_by,
    identity_failures,
    score_answer,
)
from eval.backend import MODES
from eval.evaluate import build_query_relevance, make_backend
from ingest.config import load_config, retrieval_fingerprint
from ingest.qdrant_store import make_client


def _token_counter(kind: str, tokenizer_model: str, tokenizer_revision: str | None = None):
    if kind == "word":
        from ingest.chunking import default_token_counter

        return default_token_counter
    from ingest.embedding import make_token_counter

    return make_token_counter(tokenizer_model, tokenizer_revision)


def _chunkable_gold(gold, bodies, chunk_cfg, count_tokens):
    from ingest.chunking import chunk_document

    verdict: dict[tuple[str, str], bool] = {}
    ok, dropped = [], []
    for q in gold:
        key = (q.gold_source, q.gold_document_id)
        if key not in verdict:
            try:
                chunk_document(bodies.body(*key), count_tokens=count_tokens, **chunk_cfg)
                verdict[key] = True
            except Exception:  # noqa: BLE001 — un-chunkable body; skip rather than crash
                verdict[key] = False
        (ok if verdict[key] else dropped).append(q)
    return ok, [q.id for q in dropped]


def _build_deps_no_manifest_gate(cfg):
    """Same construction as eval.evaluate.qdrant_deps(), minus the GENERATION_DIR
    manifest/provenance block — see module docstring for why this is safe here."""
    client = make_client(cfg)
    count = client.count(cfg.collection_name, exact=False)
    info = {"kind": "qdrant", "collection": cfg.collection_name, "n_points": count.count}

    from ingest.embedding import BGEM3Embedder

    embedder = BGEM3Embedder(cfg)
    reranker = None
    if cfg.rerank_enabled:
        remote = os.environ.get("RERANK_REMOTE_URL")
        if remote:
            from ingest.rerank import RemoteBGEReranker

            reranker = RemoteBGEReranker(remote)
        else:
            from ingest.rerank import make_reranker

            reranker = make_reranker(cfg)
    return client, embedder, reranker, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=MODES, default="rerank")
    ap.add_argument("--golden-set", choices=sorted(goldset.EVAL_SETS), default="v2")
    ap.add_argument("--query-type", metavar="TYPE[,TYPE...]", default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--tokenizer", choices=("word", "bge"), default=None)
    ap.add_argument("--rerank-candidates", type=int, default=None)
    ap.add_argument("--citation-route", choices=("ids", "full"), default=None)
    ap.add_argument("--translate-queries", metavar="PATH", default=None)
    ap.add_argument("--abstention-threshold", type=float, default=DEFAULT_ABSTENTION_THRESHOLD)
    ap.add_argument("--out", required=True, help="results JSON path")
    ap.add_argument("--variant-label", default=None,
                     help="free-text label stamped into the output JSON (e.g. 'knobA_enrich_512')")
    args = ap.parse_args()

    cfg = load_config()
    if cfg.generation_dir is not None:
        raise SystemExit(
            "GENERATION_DIR is set — the real eval.eval_answer_quality path should work now; "
            "use that instead of this bypass script."
        )
    if os.environ.get("RERANK_REMOTE_URL"):
        raise SystemExit(
            "RERANK_REMOTE_URL is set — unset it so RERANK_MAX_LENGTH/RERANK_CONTEXT_ENRICHED "
            "take effect in this process (a remote pod reranker ignores local env overrides)."
        )

    chunk_cfg = {"max_tokens": cfg.chunk_tokens, "overlap": cfg.chunk_overlap,
                 "min_tokens": cfg.chunk_min_tokens}
    tok_kind = args.tokenizer or "bge"
    count_tokens = _token_counter(tok_kind, cfg.tokenizer_model, cfg.tokenizer_revision)

    spec = goldset.EVAL_SETS[args.golden_set]
    gold_all = goldset.load_golden_set(spec.gold)
    holdout = goldset.load_holdout(spec.holdout)
    bodies = goldset.SnapshotBodies(
        root=spec.roots[0], needed=goldset.gold_docs(gold_all), extra_roots=spec.roots[1:])
    n_spans = goldset.reground(gold_all, bodies)
    goldset.enforce_holdout(gold_all, holdout)
    gold, dropped = _chunkable_gold(gold_all, bodies, chunk_cfg, count_tokens)
    if dropped:
        print(f"WARNING: skipped {len(dropped)} query(ies) with an un-chunkable gold doc: {dropped}")
    if args.query_type:
        wanted = {t.strip() for t in args.query_type.split(",") if t.strip()}
        gold = [q for q in gold if q.query_type in wanted]
        print(f"Restricted to query_type in {sorted(wanted)}: {len(gold)} queries")
    goldset.lint_span_coverage(gold, bodies, count_tokens=count_tokens, **chunk_cfg)
    rel = build_query_relevance(gold, bodies, chunk_cfg, count_tokens)

    translations = None
    if args.translate_queries:
        from eval.translations import load_query_translations

        translations, _ = load_query_translations(Path(args.translate_queries), gold_all)

    knobs = {"rerank_candidates": args.rerank_candidates, "translations": translations,
             "citation_route": args.citation_route}
    deps = _build_deps_no_manifest_gate(cfg)
    backend, index_info = make_backend(
        "qdrant", cfg, gold, bodies, chunk_cfg, count_tokens, knobs=knobs, deps=deps)
    fp = retrieval_fingerprint(cfg)
    print(f"[BYPASS RUN — no GENERATION_DIR provenance check] Loaded {len(gold)} gold queries "
          f"(eval_set={spec.version}, spans={n_spans}); backend={index_info}; mode={args.mode}; "
          f"retrieval_fingerprint={fp}; rerank_context_enriched={cfg.rerank_context_enriched}; "
          f"rerank_max_length={cfg.rerank_max_length}")

    scores = []
    answerable_top1: list[float] = []
    for q in gold:
        hits, _lat = backend.search(q.query, args.mode, args.top_k)
        scores.append(score_answer(q, hits, set(rel[q.id]["chunk"]), threshold=args.abstention_threshold))
        if hits:
            answerable_top1.append(hits[0].score)

    overall = aggregate_answer_scores(scores)
    by_type = breakdown_by(scores, "query_type")
    by_lang = breakdown_by(scores, "language")
    failures = identity_failures(scores)

    print("\n== Answer-quality (deterministic, BYPASS run) ==")
    print(f"  identity@1 (all):        {overall['identity_at_1']:.3f}")
    print(f"  identity@1 (known-item): {overall['known_item_identity_at_1']:.3f} (n={overall['n_known_item']})")
    print(f"  identity@10 (all):       {overall['identity_at_10']:.3f}")
    print(f"  span-coverage@10:        {overall['span_coverage_at_10']:.3f}")
    print(f"  fully-grounded@10:       {overall['fully_grounded_at_10']:.3f}")
    print(f"  confident-wrong (known): {overall['known_item_confident_wrong']:.3f}")
    print(f"  mean top-1 score:        {overall['mean_top1_score']:.3f}")
    print("\n  identity@1 by query_type:")
    for t, agg in by_type.items():
        print(f"    {t:<18} n={agg['n']:<3} id@1={agg['identity_at_1']:.3f} id@10={agg['identity_at_10']:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": "BYPASS — GENERATION_DIR manifest/provenance check skipped, see script docstring",
        "variant_label": args.variant_label,
        "eval_set_version": spec.version,
        "eval_set_hash": goldset.eval_set_hash(spec.gold),
        "retrieval_fingerprint": fp,
        "rerank_context_enriched": cfg.rerank_context_enriched,
        "rerank_max_length": cfg.rerank_max_length,
        "mode": args.mode,
        "top_k": args.top_k,
        "backend": index_info,
        "knobs": {"rerank_candidates": args.rerank_candidates, "citation_route": args.citation_route,
                  "translate_queries": bool(args.translate_queries)},
        "overall": overall,
        "per_query_type": by_type,
        "per_language": by_lang,
        "identity_failures": failures,
        # Per-query records (needed for a paired significance test — aggregates alone
        # can't be re-paired by query id across two separate runs).
        "per_query": [dataclasses.asdict(s) for s in scores],
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
