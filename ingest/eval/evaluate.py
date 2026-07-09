"""Retrieval evaluation harness for the Georgian legal RAG index.

Upgraded, span-anchored harness that gates every retrieval change (mission Phase 1). It:

  * loads the span-anchored golden set, re-grounds every evidence quote against the clean
    snapshot, enforces the document holdout, and lints span→chunk coverage (all fail loud);
  * maps each gold span to the chunks that cover it **under the current chunking config**
    (so the set survives re-chunking/re-embedding);
  * scores five modes — **BM25** (the mandatory baseline) vs dense vs sparse vs hybrid vs
    +rerank — at chunk-level (span-anchored) and doc-level, with Recall@5/@10, nDCG@10,
    MRR@10, per-query-type and per-language breakdowns, and per-stage latency p50/p95;
  * attaches bootstrap 95% CIs, runs paired significance tests for A/B comparisons, and
    appends every run to a persistent experiment log (config hash + eval-set version).

Usage (from the ingest/ project root):

    uv run python -m eval.evaluate --backend fake --mode all --log     # offline demo/self-test
    uv run python -m eval.evaluate --mode all --relevance chunk --log  # real index (Part 3)
    uv run python -m eval.evaluate --backend fake --compare hybrid rerank
"""

import argparse
import json
import random
from pathlib import Path

from ingest.chunking import chunk_document, default_token_counter
from ingest.config import load_config

from . import explog, goldset
from .backend import MODES, ChunkRecord, FakeBackend
from .metrics import (
    METRIC_NAMES,
    aggregate,
    breakdown,
    percentiles,
    query_score,
    values,
)
from .spanmap import graded_relevant_chunks
from .stats import bootstrap_ci, compare


def _token_counter(kind: str, embed_model: str):
    if kind == "word":
        return default_token_counter
    from ingest.embedding import make_token_counter

    return make_token_counter(embed_model)


def build_query_relevance(gold, bodies, chunk_cfg, count_tokens):
    """Per query id → {'chunk': {key: grade}, 'doc': {key: grade}} relevant sets."""
    rel: dict[str, dict[str, dict]] = {}
    for q in gold:
        by_doc: dict[str, list] = {}
        for r in q.relevance:
            by_doc.setdefault(r.document_id, []).append(r)
        chunk_keys: dict[tuple, int] = {}
        doc_keys: dict[tuple, int] = {}
        for document_id, rels in by_doc.items():
            body = bodies.body(q.gold_source, document_id)
            graded = graded_relevant_chunks(body, rels, count_tokens=count_tokens, **chunk_cfg)
            for ci, grade in graded.items():
                chunk_keys[(q.gold_source, document_id, ci)] = grade
            doc_keys[(q.gold_source, document_id)] = max(r.grade for r in rels)
        rel[q.id] = {"chunk": chunk_keys, "doc": doc_keys}
    return rel


def build_fake_corpus(gold, bodies, chunk_cfg, count_tokens, *, n_distractors=200, seed=0, scan_cap=2000):
    """Chunk the gold docs + a bounded sample of distractor docs into a synthetic corpus.

    Distractors are drawn from the first ``scan_cap`` docs of each gold source (the docs
    jsonl are multi-GB, so a full scan is avoided); this only feeds the synthetic ranking
    pool, so a bounded, non-exhaustive sample is fine.
    """
    records: list[ChunkRecord] = []
    gold_set = goldset.gold_docs(gold)

    def add(source, document_id, body):
        for c in chunk_document(body, count_tokens=count_tokens, **chunk_cfg):
            records.append(ChunkRecord(source, document_id, c.chunk_index, c.text))

    for source, document_id in sorted(gold_set):
        add(source, document_id, bodies.body(source, document_id))

    rng = random.Random(seed)
    sources = sorted({s for s, _ in gold_set})
    per_source = max(1, n_distractors // len(sources))
    for source in sources:
        path = goldset.DEFAULT_SNAPSHOT_DOCS / f"{source}.jsonl"
        candidates: list[tuple[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if (source, d["document_id"]) not in gold_set:
                    candidates.append((d["document_id"], d["body_markdown"]))
                if len(candidates) >= scan_cap:
                    break
        for document_id, body in rng.sample(candidates, min(per_source, len(candidates))):
            add(source, document_id, body)
    return records


def run_mode(backend, gold, rel, mode, level, k):
    scores = []
    lat = {"embed": [], "search": [], "rerank": [], "total": []}
    for q in gold:
        hits, stage = backend.search(q.query, mode, k)
        scores.append(
            query_score(q.id, q.query_type, q.query_language, hits, rel[q.id][level], level)
        )
        for s in ("embed", "search", "rerank"):
            lat[s].append(stage[s])
        lat["total"].append(sum(stage.values()))
    return scores, lat


def _fmt_ci(ci) -> str:
    return f"{ci.mean:.3f} [{ci.lo:.3f},{ci.hi:.3f}]"


def print_table(results, level, k):
    print(f"\n== Baseline table (relevance={level}, top_k={k}) — mean [95% CI] ==")
    header = f"{'mode':<8}" + "".join(f"{m:>22}" for m in METRIC_NAMES) + f"{'lat p50/p95 (ms)':>20}"
    print(header)
    print("-" * len(header))
    for mode, (scores, lat) in results.items():
        row = f"{mode:<8}"
        for m in METRIC_NAMES:
            row += f"{_fmt_ci(bootstrap_ci(values(scores, m))):>22}"
        tot = percentiles(lat["total"])
        row += f"{tot['p50'] * 1000:>8.1f}/{tot['p95'] * 1000:<11.1f}"
        print(row)


def print_breakdowns(results):
    for mode, (scores, _lat) in results.items():
        print(f"\n-- {mode}: per-query-type --")
        for grp, agg in breakdown(scores, "query_type").items():
            print(f"   {grp:<18} n={agg['n']:<3} " + " ".join(f"{m}={agg[m]:.3f}" for m in METRIC_NAMES))
        print(f"-- {mode}: per-language --")
        for grp, agg in breakdown(scores, "language").items():
            print(f"   {grp:<18} n={agg['n']:<3} " + " ".join(f"{m}={agg[m]:.3f}" for m in METRIC_NAMES))


def print_stage_latency(results):
    print("\n== Per-stage latency p50/p95 (ms) ==")
    print(f"{'mode':<8}{'embed':>16}{'search':>16}{'rerank':>16}")
    for mode, (_scores, lat) in results.items():
        row = f"{mode:<8}"
        for s in ("embed", "search", "rerank"):
            p = percentiles(lat[s])
            row += f"{p['p50'] * 1000:>7.1f}/{p['p95'] * 1000:<8.1f}"
        print(row)


def qdrant_deps(cfg):
    """Build the shared, expensive Qdrant/BGE deps once (so a paired A/B reuses them)."""
    from ingest.embedding import BGEM3Embedder
    from ingest.qdrant_store import make_client

    client = make_client(cfg)
    embedder = BGEM3Embedder(cfg)
    reranker = None
    if cfg.rerank_enabled:
        import os
        remote = os.environ.get("RERANK_REMOTE_URL")
        if remote:  # offload cross-encoder scoring to a GPU pod (retrieval stays local)
            from ingest.rerank import RemoteBGEReranker
            reranker = RemoteBGEReranker(remote)
        else:
            from ingest.rerank import BGEReranker
            reranker = BGEReranker(cfg)
    info = client.get_collection(cfg.collection_name)
    return client, embedder, reranker, {
        "kind": "qdrant", "collection": cfg.collection_name, "n_points": info.points_count,
    }


def make_backend(kind, cfg, gold, bodies, chunk_cfg, count_tokens, *, knobs=None, deps=None):
    knobs = knobs or {}
    if kind == "fake":
        records = build_fake_corpus(gold, bodies, chunk_cfg, count_tokens)
        return FakeBackend(records), {"kind": "fake", "n_chunks": len(records)}
    from .backend import QdrantBackend

    client, embedder, reranker, info = deps or qdrant_deps(cfg)
    return QdrantBackend(cfg, client, embedder, reranker=reranker, **knobs), info


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate legal RAG retrieval quality.")
    ap.add_argument("--backend", choices=("fake", "qdrant"), default="qdrant")
    ap.add_argument("--mode", choices=(*MODES, "all"), default="all")
    ap.add_argument("--relevance", choices=("chunk", "doc"), default="chunk")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--tokenizer", choices=("word", "bge"), default=None,
                    help="chunking token counter; default: word for fake, bge for qdrant")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="paired A/B on two modes")
    # Retrieval-tuning knobs (all measured through the harness; none re-embed). Each changes
    # the config_hash so its run is a distinct, forever-comparable row in experiments.jsonl.
    ap.add_argument("--rerank-candidates", type=int, default=None, help="rerank pool depth")
    ap.add_argument("--fusion", choices=("rrf", "dbsf"), default="rrf", help="hybrid fusion")
    ap.add_argument("--prefetch-limit", type=int, default=None, help="override recall pool depth")
    ap.add_argument("--hnsw-ef", type=int, default=None, help="HNSW search ef")
    ap.add_argument("--rescore", choices=("on", "off"), default=None, help="int8 rescore")
    ap.add_argument("--sparse-weight", type=float, default=None, help="weighted dense/sparse fusion")
    ap.add_argument("--max-per-doc", type=int, default=None, help="diversity: cap chunks per doc")
    ap.add_argument("--mmr-lambda", type=float, default=None, help="diversity: MMR trade-off 0..1")
    ap.add_argument("--translate-queries", metavar="PATH", default=None,
                    help="I2: authored EN→KA query-translation JSON (embeds the KA text)")
    ap.add_argument("--ab", action="store_true",
                    help="paired A/B: --mode with knobs OFF (A) vs the given knobs ON (B)")
    ap.add_argument("--log", action="store_true", help="append runs to the experiment log")
    ap.add_argument("--log-path", default=str(explog.DEFAULT_LOG), help="experiment-log path")
    args = ap.parse_args()

    cfg = load_config()
    chunk_cfg = {
        "max_tokens": cfg.chunk_tokens,
        "overlap": cfg.chunk_overlap,
        "min_tokens": cfg.chunk_min_tokens,
    }
    tok_kind = args.tokenizer or ("word" if args.backend == "fake" else "bge")
    count_tokens = _token_counter(tok_kind, cfg.embed_model)

    gold = goldset.load_golden_set()
    holdout = goldset.load_holdout()
    bodies = goldset.SnapshotBodies(needed=goldset.gold_docs(gold))

    n_spans = goldset.reground(gold, bodies)
    goldset.enforce_holdout(gold, holdout)
    n_linted = goldset.lint_span_coverage(gold, bodies, count_tokens=count_tokens, **chunk_cfg)
    print(
        f"Loaded {len(gold)} gold queries · {len(holdout)} holdout docs · "
        f"re-grounding {n_spans}/0 drift · span-coverage {n_linted}/0 empty "
        f"(eval_set={goldset.EVAL_SET_VERSION}, tokenizer={tok_kind})"
    )

    rel = build_query_relevance(gold, bodies, chunk_cfg, count_tokens)

    rescore = {"on": True, "off": False}.get(args.rescore)
    translations, translations_hash = (None, None)
    if args.translate_queries:
        from .translations import load_query_translations

        translations, translations_hash = load_query_translations(Path(args.translate_queries), gold)
    knobs = {
        "rerank_candidates": args.rerank_candidates, "fusion": args.fusion,
        "prefetch_limit": args.prefetch_limit, "hnsw_ef": args.hnsw_ef, "rescore": rescore,
        "sparse_weight": args.sparse_weight, "max_per_doc": args.max_per_doc,
        "mmr_lambda": args.mmr_lambda, "translations": translations,
    }
    # The raw translation dict never enters config_hash/logs — its file content hash does.
    active_knobs = {k: v for k, v in knobs.items()
                    if v not in (None, "rrf") and k != "translations"}
    if translations_hash:
        active_knobs["translate_queries"] = translations_hash

    deps = qdrant_deps(cfg) if args.backend == "qdrant" else None
    backend, index_info = make_backend(
        args.backend, cfg, gold, bodies, chunk_cfg, count_tokens, knobs=knobs, deps=deps)
    print(f"Backend: {index_info}  knobs={active_knobs or 'defaults'}")

    if args.ab:
        m = args.mode if args.mode != "all" else "rerank"
        base, _ = make_backend(
            args.backend, cfg, gold, bodies, chunk_cfg, count_tokens, knobs=None, deps=deps)
        results = {
            m: run_mode(base, gold, rel, m, args.relevance, args.top_k),
            f"{m}+knobs": run_mode(backend, gold, rel, m, args.relevance, args.top_k),
        }
        args.compare = [m, f"{m}+knobs"]  # drive the paired-compare + logging paths below
    else:
        modes = list(MODES) if args.mode == "all" else [args.mode]
        if args.compare:
            modes = list(dict.fromkeys(args.compare))
        results = {m: run_mode(backend, gold, rel, m, args.relevance, args.top_k) for m in modes}

    print_table(results, args.relevance, args.top_k)
    print_stage_latency(results)
    print_breakdowns(results)

    if args.compare:
        a, b = args.compare
        print(f"\n== Paired A/B: {a} (A) vs {b} (B), relevance={args.relevance} ==")
        for m in METRIC_NAMES:
            c = compare(m, values(results[a][0], m), values(results[b][0], m))
            print(f"   {m:<9} Δ={c.diff:+.3f} [{c.diff_lo:+.3f},{c.diff_hi:+.3f}] "
                  f"p={c.p_value:.4f} → {c.verdict}")

    if args.log:
        eval_hash = goldset.eval_set_hash()
        for label, (scores, lat) in results.items():
            base_mode = label.split("+")[0]
            # In --ab, the base label logs with knobs OFF; every other run logs the active
            # knobs. Plain runs (no knobs) hash exactly as before → historical rows comparable.
            eff = {} if (args.ab and label == base_mode) else active_knobs
            eff_rc = eff.get("rerank_candidates") or cfg.rerank_candidates
            ch = explog.config_hash({
                "mode": base_mode, "relevance": args.relevance, "top_k": args.top_k,
                "tokenizer": tok_kind, **chunk_cfg,
                "rerank_candidates": eff_rc,
                **{k: v for k, v in eff.items() if k != "rerank_candidates"},
            })
            record = {
                "timestamp": explog.now_iso(),
                "eval_set_version": goldset.EVAL_SET_VERSION,
                "eval_set_hash": eval_hash,
                "config_hash": ch,
                "mode": label,
                "relevance_level": args.relevance,
                "top_k": args.top_k,
                # Descriptive-only mirror of the tuning knobs folded into config_hash, so the
                # experiment log is self-describing for ablation tables (does NOT affect the hash).
                "knobs": {"rerank_candidates": eff_rc,
                          **{k: v for k, v in eff.items() if k != "rerank_candidates"}},
                "backend": index_info,
                "metrics": aggregate(scores),
                "cis": {m: bootstrap_ci(values(scores, m)).__dict__ for m in METRIC_NAMES},
                "latency_ms": {
                    s: {kk: v * 1000 for kk, v in percentiles(lat[s]).items()}
                    for s in ("embed", "search", "rerank", "total")
                },
                "per_query_type": breakdown(scores, "query_type"),
                "per_language": breakdown(scores, "language"),
            }
            explog.append_run(record, Path(args.log_path))
        print(f"\nAppended {len(results)} run(s) to {args.log_path}")


if __name__ == "__main__":
    main()
