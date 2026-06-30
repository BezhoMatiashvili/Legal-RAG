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
