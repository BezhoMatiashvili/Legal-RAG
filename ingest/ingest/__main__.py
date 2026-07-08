"""CLI: `python -m ingest ingest ...` and `python -m ingest search ...`."""

import argparse
import dataclasses
import logging

from .config import load_config


def _resolved_cfg(args):
    cfg = load_config()
    if args.collection:
        cfg = dataclasses.replace(cfg, collection_name=args.collection)
    return cfg


def _cmd_ingest(args) -> None:
    from . import pipeline
    from . import qdrant_store as store
    from .embedding import BGEM3Embedder, make_token_counter
    from .progress import IngestProgress

    if args.recreate and args.resume:
        raise SystemExit("--recreate and --resume are mutually exclusive.")

    cfg = _resolved_cfg(args)
    sources = pipeline.resolve_sources(args.source)

    # Recreating the collection invalidates checkpoints — clear them so a later --resume
    # cannot fast-forward past documents that no longer exist.
    if args.recreate:
        for source in sources:
            pipeline.delete_checkpoint(cfg, source)

    client = store.make_client(cfg)
    created = store.ensure_collection(client, cfg, recreate=args.recreate)

    print(f"Loading embedding model {cfg.embed_model} (first run downloads ~2GB)...")
    embedder = BGEM3Embedder(cfg)
    count_tokens = make_token_counter(cfg.embed_model)

    grand_docs = grand_chunks = grand_skipped = 0
    # Per-source result lines are collected and printed *after* the live panel closes —
    # writing to stdout while rich.Live is active would corrupt the display.
    results: list[str] = []
    with IngestProgress(enabled=not args.no_progress) as progress:
        for source in sources:
            progress.add_source(source)
        for source in sources:
            try:
                docs, chunks, skipped = pipeline.ingest_source(
                    cfg, client, embedder, count_tokens, source,
                    run=args.run, limit=args.limit, batch_size=args.batch_size,
                    resume=args.resume, skip_stale_delete=created, progress=progress,
                )
            except Exception:
                progress.finish_source(source, error=True)
                raise
            progress.finish_source(source)
            results.append(f"  {source}: {docs} docs -> {chunks} chunks ({skipped} skipped)")
            grand_docs += docs
            grand_chunks += chunks
            grand_skipped += skipped

    for line in results:
        print(line)
    info = client.get_collection(cfg.collection_name)
    print(f"Done: {grand_docs} docs -> {grand_chunks} chunks ({grand_skipped} skipped). "
          f"Collection '{cfg.collection_name}' now has {info.points_count} points.")


def _cmd_watch(args) -> None:
    from . import pipeline
    from . import qdrant_store as store
    from .embedding import BGEM3Embedder, make_token_counter

    cfg = _resolved_cfg(args)
    sources = pipeline.resolve_sources(args.source)

    # Recreating the collection invalidates the offset state — clear it so the watcher
    # backfills from scratch rather than fast-forwarding past points that no longer exist.
    if args.recreate:
        for source in sources:
            pipeline.delete_watch_state(cfg, source)

    client = store.make_client(cfg)
    created = store.ensure_collection(client, cfg, recreate=args.recreate)

    print(f"Loading embedding model {cfg.embed_model} (first run downloads ~2GB)...")
    embedder = BGEM3Embedder(cfg)
    count_tokens = make_token_counter(cfg.embed_model)

    pipeline.watch_loop(
        cfg, client, embedder, count_tokens, sources,
        batch_size=args.batch_size, poll_interval=args.poll_interval,
        once=args.once, limit=args.limit, skip_stale_delete=created,
    )


def _cmd_snapshot(args) -> None:
    from . import pipeline, snapshot

    cfg = _resolved_cfg(args)
    sources = pipeline.resolve_sources(args.source) if args.source != "all" else list(snapshot.SOURCES_PRESENT)
    print(f"Building clean corpus snapshot from {len(sources)} source(s): {', '.join(sources)}")
    snapshot.build_snapshot(
        cfg, sources=sources, limit=args.limit,
        near_dup=not args.no_near_dup, token_sample=args.token_sample,
    )


def _cmd_embed(args) -> None:
    import json as _json
    import random

    from . import embed_job
    from . import qdrant_store as store
    from .embedding import BGEM3Embedder, make_token_counter

    cfg = _resolved_cfg(args)
    print(f"Loading embedding model {cfg.embed_model} (device={cfg.embed_device or 'auto/cpu'}, "
          f"fp16={cfg.embed_use_fp16}, batch={cfg.embed_batch_size})...")
    embedder = BGEM3Embedder(cfg)

    digest, vec = embed_job.dense_checksum(embedder)
    print(f"VECTOR-SPACE CHECKSUM: sha={digest}  dims[:8]={[round(x, 5) for x in vec[:8]]}")
    print("  (embed the same sentence CPU vs GPU; assert cosine≈1 — guardrail G2)")
    if args.checksum:
        ref = embed_job.SNAPSHOT_DOCS.parent / "checksum_cpu.json"
        embed_job.save_checksum_reference(embedder, ref)
        print(f"  saved CPU reference vector → {ref}")
        return

    count_tokens = make_token_counter(cfg.embed_model)
    client = store.make_client(cfg)
    store.ensure_collection(client, cfg, recreate=args.recreate)

    if args.pilot:
        # gold/holdout docs (so the harness resolves) + a distractor sample per source
        holdout = _json.loads(
            (embed_job.SNAPSHOT_DOCS.parent.parent.parent / "eval" / "holdout_doc_ids.json").read_text()
        )
        gold_by_src: dict[str, set[str]] = {}
        for x in holdout:
            gold_by_src.setdefault(x["source"], set()).add(x["document_id"])
        rng = random.Random(0)
        docs = []
        for source in embed_job.SOURCES:
            docs.extend(embed_job.load_snapshot_docs(source, gold_by_src.get(source, set())))
            gold_ids = gold_by_src.get(source, set())
            pool = [d for d in embed_job.iter_snapshot_docs(source, limit=args.distractors * 3)
                    if d.document_id not in gold_ids]
            docs.extend(rng.sample(pool, min(args.distractors, len(pool))))
        d, c, k = embed_job.embed_docs(cfg, client, embedder, count_tokens, docs,
                                       batch_size=args.batch_size)
        info = client.get_collection(cfg.collection_name)
        print(f"Pilot embedded: {d} docs -> {c} chunks ({k} skipped). "
              f"Collection '{cfg.collection_name}' now has {info.points_count} points.")
        return

    # full-corpus resumable embed (RunPod production profile)
    sources = [args.source] if args.source != "all" else list(embed_job.SOURCES)
    grand_d = grand_c = grand_k = 0
    for source in sources:
        d, c, k = embed_job.embed_source_resumable(
            cfg, client, embedder, count_tokens, source,
            batch_size=args.batch_size, limit=args.limit,
        )
        print(f"  {source}: {d} docs -> {c} chunks ({k} skipped)")
        grand_d += d
        grand_c += c
        grand_k += k
    info = client.get_collection(cfg.collection_name)
    print(f"Done: {grand_d} docs -> {grand_c} chunks ({grand_k} skipped). "
          f"Collection '{cfg.collection_name}' now has {info.points_count} points.")


def _cmd_search(args) -> None:
    from . import qdrant_store as store
    from . import search as search_mod
    from .embedding import BGEM3Embedder

    cfg = _resolved_cfg(args)
    client = store.make_client(cfg)
    embedder = BGEM3Embedder(cfg)
    reranker = None
    if cfg.rerank_enabled and not args.no_rerank:
        from .rerank import BGEReranker
        print(f"Loading reranker {cfg.rerank_model}...")
        reranker = BGEReranker(cfg)
    hits = search_mod.hybrid_search(
        cfg, client, embedder, args.query, top_k=args.top_k,
        reranker=reranker, rerank_candidates=cfg.rerank_candidates,
        rerank_min_score=cfg.rerank_min_score,
        source=args.source, language=args.language, document_type=args.document_type,
        status=args.status, date_from=args.date_from, date_to=args.date_to,
    )
    if not hits:
        print("No results.")
        return
    for rank, hit in enumerate(hits, 1):
        p = hit.payload or {}
        snippet = " ".join((p.get("text") or "").split())[:220]
        status = f"  status={p.get('status')}" if p.get("status") else ""
        print(f"\n#{rank}  score={hit.score:.4f}  [{p.get('source')}/{p.get('document_type')}]{status}  "
              f"{p.get('date_raw') or '—'}  {p.get('title') or ''}")
        print(f"    chunk#{p.get('chunk_index')}  {p.get('source_url') or ''}")
        print(f"    {snippet}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="ingest", description="Ingest/search Georgian legal docs in Qdrant")
    parser.add_argument("--collection", help="override collection name")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="ingest scraped JSONL into Qdrant")
    p_ing.add_argument("--source", required=True, help="spider name or 'all'")
    p_ing.add_argument("--run", default="latest", help="'latest' or a specific run_id")
    p_ing.add_argument("--limit", type=int, default=None, help="cap docs per source (pilot)")
    p_ing.add_argument("--batch-size", type=int, default=256, help="points per upsert batch")
    p_ing.add_argument("--recreate", action="store_true", help="drop & recreate the collection first")
    p_ing.add_argument("--resume", action="store_true", help="resume from the per-source checkpoint")
    p_ing.add_argument("--no-progress", action="store_true",
                       help="disable the live progress panel (use plain tqdm/log output)")
    p_ing.set_defaults(func=_cmd_ingest)

    p_watch = sub.add_parser(
        "watch", help="continuously ingest scraped docs (backfill oldest->newest, then wait for new)")
    p_watch.add_argument("--source", required=True, help="spider name or 'all'")
    p_watch.add_argument("--poll-interval", type=float, default=5.0,
                         help="seconds between polls when idle (default 5)")
    p_watch.add_argument("--batch-size", type=int, default=256, help="points per upsert batch")
    p_watch.add_argument("--once", action="store_true",
                         help="backfill everything not yet ingested, then exit (no waiting)")
    p_watch.add_argument("--recreate", action="store_true",
                         help="drop & recreate the collection and clear watch state, then backfill")
    p_watch.add_argument("--limit", type=int, default=None,
                         help="cap docs per source per drain pass (debug)")
    p_watch.set_defaults(func=_cmd_watch)

    p_snap = sub.add_parser(
        "snapshot", help="build a clean, deduplicated, versioned corpus snapshot (Part 1)")
    p_snap.add_argument("--source", default="all", help="spider name or 'all' (present sources)")
    p_snap.add_argument("--limit", type=int, default=None, help="cap docs per source (dev)")
    p_snap.add_argument("--no-near-dup", action="store_true",
                        help="skip the MinHash/LSH near-duplicate pass (faster)")
    p_snap.add_argument("--token-sample", type=int, default=2000,
                        help="docs/source to sample for the BGE-M3 token-length profile")
    p_snap.set_defaults(func=_cmd_snapshot)

    p_embed = sub.add_parser(
        "embed", help="embed the clean snapshot into Qdrant (CPU pilot / RunPod GPU prod)")
    p_embed.add_argument("--source", default="all", help="source name or 'all'")
    p_embed.add_argument("--pilot", action="store_true",
                         help="embed holdout/gold docs + a distractor sample (interim baseline)")
    p_embed.add_argument("--distractors", type=int, default=400,
                         help="distractor docs/source in --pilot mode")
    p_embed.add_argument("--checksum", action="store_true",
                         help="print the vector-space checksum and exit (CPU-vs-GPU guardrail)")
    p_embed.add_argument("--limit", type=int, default=None, help="cap docs/source (full mode)")
    p_embed.add_argument("--batch-size", type=int, default=256, help="points per upsert batch")
    p_embed.add_argument("--recreate", action="store_true", help="drop & recreate the collection")
    p_embed.set_defaults(func=_cmd_embed)

    p_search = sub.add_parser("search", help="hybrid search the collection")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--source")
    p_search.add_argument("--language")
    p_search.add_argument("--document-type")
    p_search.add_argument("--status", help="in_force / repealed / pending (matsne)")
    p_search.add_argument("--date-from")
    p_search.add_argument("--date-to")
    p_search.add_argument("--no-rerank", action="store_true",
                          help="skip the cross-encoder rerank (raw RRF order)")
    p_search.set_defaults(func=_cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
