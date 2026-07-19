"""CLI runner for the deterministic answer-quality eval (see ``eval/answer_eval.py``).

Runs retrieval over a golden set (fake or qdrant backend), computes the deterministic
answer-correctness metrics — citation-identity correctness, context-sufficiency, and (when a
known-unanswerable set is supplied) abstention correctness — prints a report, and writes a
results JSON under ``.state/answer_eval/``.

This is **import-only** over the retrieval harness (``eval/evaluate.py``, ``eval/goldset.py``,
``eval/backend.py``): it reuses their loaders/relevance/backend construction, adds no row to
``experiments.jsonl`` (answer-eval results live in their own JSON), and edits no claimed file.

    # offline smoke — no torch/Qdrant, proves the wiring end-to-end:
    python -m eval.eval_answer_quality --backend fake --mode hybrid

    # real index (DEFERRED to a quiet box; take eval-run.lock + reranker-ram.lock first,
    # and use --mode rerank so top-1 scores are calibrated 0..1 for the abstention metric):
    RERANK_ENABLED=true python -m eval.eval_answer_quality --backend qdrant --mode rerank \
        --golden-set v2 --translate-queries eval/query_translations_v2.json \
        --unanswerable eval/unanswerable_v1.json
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ingest.config import load_config, retrieval_fingerprint

from . import goldset
from .answer_eval import (
    DEFAULT_ABSTENTION_THRESHOLD,
    abstention_correctness,
    aggregate_answer_scores,
    breakdown_by,
    identity_failures,
    score_answer,
)
from .backend import MODES
from .evaluate import build_query_relevance, make_backend


def _token_counter(kind: str, tokenizer_model: str, tokenizer_revision: str | None = None):
    if kind == "word":
        from ingest.chunking import default_token_counter

        return default_token_counter
    from ingest.embedding import make_token_counter

    return make_token_counter(tokenizer_model, tokenizer_revision)


def _load_unanswerable(path: Path) -> list[str]:
    """A JSON list of known-unanswerable queries (each a string or ``{"query": ...}``)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [x["query"] if isinstance(x, dict) else x for x in data]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunkable_gold(gold, bodies, chunk_cfg, count_tokens):
    """Identify known chunk-budget failures without swallowing unrelated implementation bugs."""
    from ingest.chunking import chunk_document

    verdict: dict[tuple[str, str], bool] = {}
    ok, dropped = [], {}
    for q in gold:
        key = (q.gold_source, q.gold_document_id)
        if key not in verdict:
            try:
                chunk_document(bodies.body(*key), count_tokens=count_tokens, **chunk_cfg)
                verdict[key] = True
            except (ValueError, AssertionError) as exc:
                verdict[key] = False
                dropped[q.id] = f"{type(exc).__name__}: {exc}"
        if verdict[key]:
            ok.append(q)
        else:
            dropped.setdefault(q.id, "document previously identified as unchunkable")
    return ok, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic answer-quality eval for the legal RAG.")
    ap.add_argument("--backend", choices=("fake", "qdrant"), default="qdrant")
    ap.add_argument("--mode", choices=MODES, default="rerank",
                    help="retrieval mode; use 'rerank' for calibrated 0..1 top-1 scores (abstention)")
    ap.add_argument("--golden-set", choices=sorted(goldset.EVAL_SETS), default="v2")
    ap.add_argument("--query-type", metavar="TYPE[,TYPE...]", default=None,
                    help="restrict the run to these comma-separated query_type values "
                         "(e.g. legal_citation,keyword) — cuts runtime for a slice-only run")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--tokenizer", choices=("word", "bge"), default=None)
    ap.add_argument("--rerank-candidates", type=int, default=None)
    ap.add_argument("--citation-route", choices=("ids", "full"), default=None)
    ap.add_argument("--translate-queries", metavar="PATH", default=None,
                    help="authored EN→KA query-translation JSON (embeds the KA text)")
    ap.add_argument("--unanswerable", metavar="PATH", default=None,
                    help="JSON list of KNOWN-UNANSWERABLE queries → abstention metric")
    ap.add_argument("--abstention-threshold", type=float, default=DEFAULT_ABSTENTION_THRESHOLD)
    ap.add_argument("--out", default=None, help="results JSON path (default under .state/answer_eval/)")
    args = ap.parse_args()
    if args.top_k != 10:
        raise SystemExit("--top-k must be 10 until answer metric field names are depth-dynamic")

    cfg = load_config()
    chunk_cfg = {"max_tokens": cfg.chunk_tokens, "overlap": cfg.chunk_overlap,
                 "min_tokens": cfg.chunk_min_tokens}
    tok_kind = args.tokenizer or ("word" if args.backend == "fake" else "bge")
    count_tokens = _token_counter(tok_kind, cfg.tokenizer_model, cfg.tokenizer_revision)

    spec = goldset.EVAL_SETS[args.golden_set]
    gold_all = goldset.load_golden_set(spec.gold)          # full set (translations validate vs this)
    holdout = goldset.load_holdout(spec.holdout)
    bodies = goldset.SnapshotBodies(
        root=spec.roots[0], needed=goldset.gold_docs(gold_all), extra_roots=spec.roots[1:])
    n_spans = goldset.reground(gold_all, bodies)           # fail loud on hygiene drift
    goldset.enforce_holdout(gold_all, holdout)             # fail loud on contamination
    _chunkable, dropped = _chunkable_gold(gold_all, bodies, chunk_cfg, count_tokens)
    gold = list(gold_all)
    if dropped:
        print(f"WARNING: {len(dropped)} query(ies) have an un-chunkable gold doc and will "
              f"count as failures: {sorted(dropped)}")
    wanted = None
    if args.query_type:
        wanted = {t.strip() for t in args.query_type.split(",") if t.strip()}
        gold = [q for q in gold if q.query_type in wanted]
        selected_ids = {q.id for q in gold}
        dropped = {query_id: reason for query_id, reason in dropped.items()
                   if query_id in selected_ids}
        print(f"Restricted to query_type in {sorted(wanted)}: {len(gold)} queries")
    rel = build_query_relevance(gold, bodies, chunk_cfg, count_tokens)

    translations = None
    translations_file_hash = None
    if args.translate_queries:
        from .translations import load_query_translations

        # Validate against the FULL set: the guard may drop cross-lingual queries whose
        # translation entries would otherwise look "unknown". Extra entries are harmless.
        translations_path = Path(args.translate_queries)
        translations, _ = load_query_translations(translations_path, gold_all)
        translations_file_hash = _sha256(translations_path)

    knobs = {"rerank_candidates": args.rerank_candidates, "translations": translations,
             "citation_route": args.citation_route}
    backend, index_info = make_backend(
        args.backend, cfg, gold, bodies, chunk_cfg, count_tokens, knobs=knobs)
    print(f"Loaded {len(gold)} gold queries (eval_set={spec.version}, spans={n_spans}); "
          f"backend={index_info}; mode={args.mode}")

    scores = []
    answerable_top1: list[float] = []
    per_query = []
    for q in gold:
        failure_reason = rel[q.id].get("failure_reason")
        if failure_reason:
            hits, latency = [], {"embed": 0.0, "search": 0.0, "rerank": 0.0}
            status = "unscorable"
        else:
            hits, latency = backend.search(q.query, args.mode, args.top_k)
            outcome = getattr(backend, "last_outcome", None)
            if outcome is not None and outcome.degraded:
                failure_reason = outcome.degraded_reason
                hits = []
                status = "degraded"
            elif outcome is not None and outcome.service_abstention:
                status = "abstain"
            else:
                status = "ok"
        score = score_answer(
            q,
            hits,
            rel[q.id]["evidence_groups"],
            threshold=args.abstention_threshold,
        )
        scores.append(score)
        # Empty/degraded answerable retrieval is a zero, not an omitted denominator row.
        answerable_top1.append(hits[0].score if hits else 0.0)
        per_query.append({
            **score.__dict__,
            "status": status,
            "failure_reason": failure_reason,
            "timings_ms": {name: value * 1000 for name, value in latency.items()},
        })

    abstention = None
    unanswerable_file_hash = None
    if args.unanswerable:
        unanswerable_path = Path(args.unanswerable)
        unanswerable_file_hash = _sha256(unanswerable_path)
        un_queries = _load_unanswerable(unanswerable_path)
        un_top1: list[float] = []
        for uq in un_queries:
            hits, _lat = backend.search(uq, args.mode, args.top_k)
            un_top1.append(hits[0].score if hits else 0.0)
        abstention = abstention_correctness(
            answerable_top1, un_top1, threshold=args.abstention_threshold)

    overall = aggregate_answer_scores(scores)
    by_type = breakdown_by(scores, "query_type")
    by_lang = breakdown_by(scores, "language")
    failures = identity_failures(scores)

    # ---- report ----
    print("\n== Answer-quality (deterministic) ==")
    print(f"  identity@1 (all):        {overall['identity_at_1']:.3f}")
    print(f"  identity@1 (known-item): {overall['known_item_identity_at_1']:.3f} "
          f"(n={overall['n_known_item']})   [the legal 'right law at rank 1' metric]")
    print(f"  identity@10 (all):       {overall['identity_at_10']:.3f}")
    print(f"  span-coverage@10:        {overall['span_coverage_at_10']:.3f}  "
          f"(required evidence-group recall; alternatives count once)")
    print(f"  fully-grounded@10:       {overall['fully_grounded_at_10']:.3f}")
    print(f"  confident-wrong (known): {overall['known_item_confident_wrong']:.3f}  "
          f"(score>={args.abstention_threshold:.2f} but WRONG doc — hallucination a score-gate lets through; "
          f"meaningful in rerank mode)")
    print(f"  mean top-1 score:        {overall['mean_top1_score']:.3f}")
    print(f"  margin (top1-top2 doc):  confident-wrong={overall['mean_margin_confident_wrong']:.3f} "
          f"(n={overall['n_confident_wrong']})  vs  confident-correct="
          f"{overall['mean_margin_confident_correct']:.3f} (n={overall['n_confident_correct']})")
    print("\n  identity@1 by query_type:")
    for t, agg in by_type.items():
        print(f"    {t:<18} n={agg['n']:<3} id@1={agg['identity_at_1']:.3f} id@10={agg['identity_at_10']:.3f}")
    if abstention is not None:
        print(f"\n  abstention @ {abstention.threshold:.3f}: "
              f"retention={abstention.retention:.3f} (answerable kept) · "
              f"correct_refusal={abstention.correct_refusal:.3f} (unanswerable refused) · "
              f"false_accept={abstention.false_accept:.3f}")
    if failures:
        print(f"\n  {len(failures)} known-item identity@1 FAILURES (wrong law at rank 1) — first 10:")
        for f in failures[:10]:
            print(f"    {f['id']:<10} gold={f['gold_doc']} got={f['got_top1']}")

    # ---- persist ----
    out_path = Path(args.out) if args.out else (
        cfg.state_dir / "answer_eval"
        / f"answer_eval_{spec.version}_{args.mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    unscorable_queries = {
        query.id: rel[query.id]["failure_reason"]
        for query in gold
        if rel[query.id].get("failure_reason")
    }
    decision_rows = [
        {key: value for key, value in row.items() if key != "timings_ms"}
        for row in per_query
    ]
    result_hash = hashlib.sha256(
        json.dumps(decision_rows, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    result = {
        "schema_version": "answer-accuracy-eval/v2",
        "scoring_revision": "answer-eval-r2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_set_version": spec.version,
        "eval_set_hash": goldset.eval_set_hash(spec.gold),
        "retrieval_fingerprint": retrieval_fingerprint(cfg),
        "mode": args.mode,
        "top_k": args.top_k,
        "expected_queries": len(gold),
        "failed_queries": sum(1 for row in per_query if row["status"] in {"unscorable", "degraded"}),
        "unscorable_queries": unscorable_queries,
        "query_type_filter": sorted(wanted) if wanted else None,
        "evaluated_query_ids_hash": hashlib.sha256(
            "\n".join(query.id for query in gold).encode("utf-8")
        ).hexdigest(),
        "abstention_threshold": args.abstention_threshold,
        "backend": index_info,
        "knobs": {"rerank_candidates": args.rerank_candidates, "citation_route": args.citation_route,
                  "translate_queries": bool(args.translate_queries),
                  "translation_file_sha256": translations_file_hash},
        "unanswerable_file_sha256": unanswerable_file_hash,
        "overall": overall,
        "per_query_type": by_type,
        "per_language": by_lang,
        "identity_failures": failures,
        "abstention": abstention.__dict__ if abstention else None,
        "per_query": per_query,
        "result_hash": result_hash,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
