"""CLI: `python -m ingest ingest ...` and `python -m ingest search ...`."""

import argparse
import dataclasses
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from .config import load_config


def _resolved_cfg(args):
    cfg = load_config()
    if args.collection:
        cfg = dataclasses.replace(cfg, collection_name=args.collection)
    return cfg


def _refuse_frozen_candidate_legacy_mutation(cfg, *, command: str) -> None:
    """Keep the frozen target reachable only through the reviewed embed workflow."""

    from .release_inputs import GENERATION_ID, PHYSICAL_COLLECTION

    if cfg.generation_id == GENERATION_ID or cfg.collection_name == PHYSICAL_COLLECTION:
        raise SystemExit(
            f"legacy `{command}` is forbidden for the frozen candidate generation/collection; "
            "use only the reviewed immutable GPU embed workflow"
        )


def _cmd_ingest(args) -> None:
    from . import qdrant_store as store

    if args.recreate and args.resume:
        raise SystemExit("--recreate and --resume are mutually exclusive.")

    cfg = _resolved_cfg(args)
    _refuse_frozen_candidate_legacy_mutation(cfg, command="ingest")
    store.validate_generation_write_target(
        cfg, apply=args.apply, recreate=args.recreate
    )

    from . import pipeline
    from .embedding import BGEM3Embedder, make_token_counter
    from .progress import IngestProgress

    sources = pipeline.resolve_sources(args.source)

    # Recreating the collection invalidates checkpoints — clear them so a later --resume
    # cannot fast-forward past documents that no longer exist.
    if args.recreate:
        for source in sources:
            pipeline.delete_checkpoint(cfg, source)

    client = store.make_client(cfg)
    created = store.ensure_collection(
        client, cfg, recreate=args.recreate, apply=args.apply
    )

    print(f"Loading embedding model {cfg.embed_model} (first run downloads ~2GB)...")
    embedder = BGEM3Embedder(cfg)
    count_tokens = make_token_counter(cfg.tokenizer_model, cfg.tokenizer_revision)

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
    from . import qdrant_store as store

    cfg = _resolved_cfg(args)
    _refuse_frozen_candidate_legacy_mutation(cfg, command="watch")
    store.validate_generation_write_target(
        cfg, apply=args.apply, recreate=args.recreate
    )

    from . import pipeline
    from .embedding import BGEM3Embedder, make_token_counter

    sources = pipeline.resolve_sources(args.source)

    # Recreating the collection invalidates the offset state — clear it so the watcher
    # backfills from scratch rather than fast-forwarding past points that no longer exist.
    if args.recreate:
        for source in sources:
            pipeline.delete_watch_state(cfg, source)

    client = store.make_client(cfg)
    created = store.ensure_collection(
        client, cfg, recreate=args.recreate, apply=args.apply
    )

    print(f"Loading embedding model {cfg.embed_model} (first run downloads ~2GB)...")
    embedder = BGEM3Embedder(cfg)
    count_tokens = make_token_counter(cfg.tokenizer_model, cfg.tokenizer_revision)

    pipeline.watch_loop(
        cfg, client, embedder, count_tokens, sources,
        batch_size=args.batch_size, poll_interval=args.poll_interval,
        once=args.once, limit=args.limit, skip_stale_delete=created,
    )


def _cmd_snapshot(args) -> None:
    from . import snapshot

    cfg = _resolved_cfg(args)
    if args.source != "all":
        raise SystemExit("sealed snapshots require --source all (all seven sources)")
    sources = list(snapshot.SOURCES_PRESENT)
    print(
        f"Building clean corpus snapshot {args.snapshot_id!r} from "
        f"{len(sources)} source(s): {', '.join(sources)}"
    )
    try:
        snapshot.build_snapshot(
            cfg,
            snapshot_id=args.snapshot_id,
            output_root=args.output_root,
            source_state_evidence=args.source_state_evidence,
            preflight=args.preflight,
            sources=sources,
            limit=args.limit,
            near_dup=not args.no_near_dup,
            token_sample=args.token_sample,
        )
    except snapshot.SnapshotSafetyError as exc:
        raise SystemExit(str(exc)) from exc


def _cmd_embed(args) -> None:
    from . import qdrant_store as store

    cfg = _resolved_cfg(args)
    from . import embed_job
    from .embedding import BGEM3Embedder, make_token_counter

    batch_size = getattr(args, "batch_size", 256)
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise SystemExit("--batch-size must be a positive integer")
    # Identity validation is deliberately first: no snapshot, checkpoint, client, or model
    # access is permitted for an ambiguous generation or mutable model revision.
    store.validate_generation_identity(cfg)
    if args.recreate:
        raise SystemExit("immutable generation candidates are create-only; --recreate is forbidden")
    sealed = embed_job.verify_snapshot_docs(args.snapshot_docs)
    embed_job.validate_snapshot_build_config(sealed, cfg)

    initialize_workers = getattr(args, "initialize_workers", None)

    if args.checksum:
        if args.checksum_output is None:
            raise SystemExit("--checksum requires explicit --checksum-output PATH")
        if (
            args.resume
            or args.recreate
            or args.apply
            or args.shard
            or initialize_workers is not None
            or getattr(args, "vector_checksum", None) is not None
        ):
            raise SystemExit(
                "--checksum is incompatible with mutation, resume, sharding, and initialization"
            )
        count_tokens = make_token_counter(
            cfg.tokenizer_model,
            cfg.tokenizer_revision,
        )
        embed_job.recompute_snapshot_chunk_inventory(sealed, cfg, count_tokens)
        print(
            f"Loading embedding model {cfg.embed_model} "
            f"(device={cfg.embed_device or 'auto/cpu'}, fp16={cfg.embed_use_fp16}, "
            f"batch={cfg.embed_batch_size})..."
        )
        embedder = BGEM3Embedder(cfg)
        digest = embed_job.save_checksum_reference(
            embedder,
            args.checksum_output,
            snapshot_root=sealed.root,
        )
        print(f"VECTOR-SPACE CHECKSUM: sha={digest}; saved → {args.checksum_output}")
        return
    if args.checksum_output is not None:
        raise SystemExit("--checksum-output is only valid with --checksum")
    if getattr(args, "vector_checksum", None) is None:
        raise SystemExit("embedding requires --vector-checksum PATH")
    checksum_reference = embed_job.load_checksum_reference(args.vector_checksum)

    shard = None
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
        except (TypeError, ValueError) as exc:
            raise SystemExit("--shard must have form i/n") from exc
        if not (0 <= i < n):
            raise SystemExit(f"--shard i/n must have 0<=i<n (got {args.shard!r})")
        shard = (i, n)
        print(f"  [shard {i}/{n}] embedding every {n}-th document")
    if initialize_workers is not None:
        if (
            isinstance(initialize_workers, bool)
            or not isinstance(initialize_workers, int)
            or initialize_workers < 1
        ):
            raise SystemExit("--initialize-workers must be an integer >= 1")
        if args.resume or shard is not None or args.source != "all":
            raise SystemExit(
                "coordinator initialization requires --source all and no --resume/--shard"
            )
        worker_count = initialize_workers
    else:
        worker_count = shard[1] if shard is not None else 1
        if shard is not None and not args.resume:
            raise SystemExit(
                "sharded workers must use --resume after --initialize-workers N"
            )
    if args.source != "all" and args.source not in embed_job.SOURCES:
        raise SystemExit(f"unsupported snapshot source {args.source!r}")
    if not args.resume and initialize_workers is None and args.source != "all":
        raise SystemExit("a fresh immutable embed must initialize all sources (--source all)")
    sources = (
        [args.source]
        if args.source != "all"
        else list(embed_job.SOURCES)
    )

    from .release_inputs import (
        GENERATION_ID as FROZEN_GENERATION_ID,
        PHYSICAL_COLLECTION as FROZEN_PHYSICAL_COLLECTION,
        SNAPSHOT_ID as FROZEN_SNAPSHOT_ID,
    )

    frozen_candidate = (
        cfg.generation_id == FROZEN_GENERATION_ID
        and cfg.collection_name == FROZEN_PHYSICAL_COLLECTION
        and sealed.snapshot_id == FROZEN_SNAPSHOT_ID
    )
    storage_identity = getattr(args, "storage_identity", None)
    qdrant_storage_root = getattr(args, "qdrant_storage_root", None)
    if frozen_candidate and (storage_identity is None or qdrant_storage_root is None):
        raise SystemExit(
            "the frozen candidate requires --storage-identity and "
            "--qdrant-storage-root for every initialization/resume worker"
        )
    if initialize_workers is not None and initialize_workers > 1 and storage_identity is None:
        raise SystemExit("multi-worker initialization requires --storage-identity PATH")
    storage_sha = embed_job.storage_identity_sha256(
        storage_identity,
        checkpoint_root=(
            embed_job.binding_path(cfg).parent
            if qdrant_storage_root is not None
            else None
        ),
        qdrant_storage_root=qdrant_storage_root,
    )
    if args.resume and not embed_job.binding_path(cfg).exists():
        raise embed_job.EmbedStateError(
            f"embed binding is absent: {embed_job.binding_path(cfg)}"
        )

    # Tokenizer-driven chunk identity is independently recomputed before the first
    # Qdrant client is constructed or any collection/alias endpoint can be contacted.
    count_tokens = make_token_counter(
        cfg.tokenizer_model,
        cfg.tokenizer_revision,
    )
    embed_job.recompute_snapshot_chunk_inventory(sealed, cfg, count_tokens)

    plan_path = getattr(args, "plan", None)
    review_path = getattr(args, "review", None)
    bundle_root = getattr(args, "bundle_root", None)
    reviewed_plan_sha256 = None
    launch_authorization = None
    reviewed_collection_configuration = (
        store.expected_embed_collection_configuration(dense_dim=cfg.dense_dim)
    )
    reviewed_collection_configuration_sha = store.collection_configuration_sha256(
        reviewed_collection_configuration
    )
    if frozen_candidate:
        if plan_path is None or review_path is None or bundle_root is None:
            raise SystemExit(
                "the frozen candidate requires --plan, --review, and --bundle-root "
                "before any Qdrant mutation"
            )
        from . import gpu_workflow

        launch_authorization = gpu_workflow.validate_reviewed_launch(
            plan_path=plan_path,
            review_path=review_path,
            bundle_root=bundle_root,
            cfg=cfg,
            snapshot_docs=sealed.docs,
            vector_checksum=args.vector_checksum,
            storage_identity=storage_identity,
            qdrant_storage_root=qdrant_storage_root,
            initialize_workers=initialize_workers,
            resume=args.resume,
            shard=shard,
            batch_size=batch_size,
            source=args.source,
            apply=args.apply,
        )
        _plan_file, plan_value, reviewed_plan_sha256 = (
            gpu_workflow.load_workflow_plan(plan_path)
        )
        plan_configuration = plan_value.get("collection_configuration")
        if (
            not isinstance(plan_configuration, Mapping)
            or plan_configuration.get("value")
            != reviewed_collection_configuration
            or plan_configuration.get("sha256")
            != reviewed_collection_configuration_sha
        ):
            raise embed_job.EmbedStateError(
                "reviewed plan does not bind the exact complete Qdrant collection "
                "configuration"
            )
        # Use the reviewed plan's exact value downstream, rather than an unbound
        # recomputation, after proving both representations are identical.
        reviewed_collection_configuration = dict(plan_configuration["value"])
        reviewed_collection_configuration_sha = str(plan_configuration["sha256"])
        if (
            launch_authorization.mutation_capability.reviewed_plan_sha256
            != reviewed_plan_sha256
            or launch_authorization.mutation_capability.storage_identity_sha256
            != storage_sha
            or launch_authorization.mutation_capability.launch_evidence_sha256
            != launch_authorization.evidence_sha256
        ):
            raise embed_job.EmbedStateError(
                "reviewed mutation authority differs from the active plan, launch "
                "evidence, or mounted-volume identity"
            )
    elif any(value is not None for value in (plan_path, review_path, bundle_root)):
        raise SystemExit(
            "--plan/--review/--bundle-root are reserved for the frozen candidate workflow"
        )
    mutation_capability = (
        launch_authorization.mutation_capability
        if launch_authorization is not None
        else None
    )
    launch_evidence_sha256 = (
        launch_authorization.evidence_sha256
        if launch_authorization is not None
        else None
    )

    store.validate_generation_write_target(
        cfg,
        apply=args.apply,
        recreate=args.recreate,
    )

    if (
        not args.resume
        and os.path.lexists(embed_job.binding_path(cfg))
        and not os.path.lexists(embed_job.initialization_path(cfg))
    ):
        raise embed_job.EmbedStateError(
            "fresh embed refuses an unowned existing local binding state"
        )

    client = store.make_client(cfg)
    store.refuse_aliased_write_target(client, cfg.collection_name)
    intent_exists = os.path.lexists(embed_job.initialization_path(cfg))
    initialization = None
    if args.resume:
        if frozen_candidate and not intent_exists:
            raise embed_job.EmbedStateError(
                "frozen candidate resume requires its immutable initialization intent"
            )
        if intent_exists:
            initialization = embed_job.prepare_initialization(
                cfg,
                sealed,
                checksum=checksum_reference,
                worker_count=worker_count,
                collection_configuration=reviewed_collection_configuration,
                collection_configuration_sha256=(
                    reviewed_collection_configuration_sha
                ),
                storage_identity_sha256=storage_sha,
                reviewed_plan_sha256=reviewed_plan_sha256,
            )
        # Resume requires the live complete configuration to reproduce the immutable binding.
        store.prepare_embed_collection(
            client,
            cfg,
            resume=True,
            recreate=False,
            apply=args.apply,
            minimum_points=0,
            mutation_capability=mutation_capability,
            reviewed_plan_sha256=reviewed_plan_sha256,
            launch_evidence_sha256=launch_evidence_sha256,
            storage_identity_sha256=storage_sha,
        )
    else:
        if os.path.lexists(embed_job.binding_path(cfg).parent / "coordinator.json"):
            raise embed_job.EmbedStateError(
                "fresh embed refuses a completed coordinator; use --resume"
            )
        if not intent_exists and client.collection_exists(cfg.collection_name):
            raise RuntimeError(
                "fresh immutable embed refuses any pre-existing physical collection "
                f"{cfg.collection_name!r}, including an empty one"
            )
        initialization = embed_job.prepare_initialization(
            cfg,
            sealed,
            checksum=checksum_reference,
            worker_count=worker_count,
            collection_configuration=reviewed_collection_configuration,
            collection_configuration_sha256=reviewed_collection_configuration_sha,
            storage_identity_sha256=storage_sha,
            reviewed_plan_sha256=reviewed_plan_sha256,
        )
        if initialization.recovered:
            store.recover_embed_collection_initialization(
                client,
                cfg,
                expected_configuration=reviewed_collection_configuration,
                apply=args.apply,
                mutation_capability=mutation_capability,
                reviewed_plan_sha256=reviewed_plan_sha256,
                launch_evidence_sha256=launch_evidence_sha256,
                storage_identity_sha256=storage_sha,
            )
        else:
            store.prepare_embed_collection(
                client,
                cfg,
                resume=False,
                recreate=False,
                apply=args.apply,
                minimum_points=0,
                mutation_capability=mutation_capability,
                reviewed_plan_sha256=reviewed_plan_sha256,
                launch_evidence_sha256=launch_evidence_sha256,
                storage_identity_sha256=storage_sha,
            )
    info = client.get_collection(cfg.collection_name)
    collection_configuration = store.collection_configuration(info)
    collection_configuration_sha = store.collection_configuration_sha256(
        collection_configuration
    )
    if (
        initialization is not None
        and (
            collection_configuration != reviewed_collection_configuration
            or collection_configuration_sha
            != reviewed_collection_configuration_sha
        )
    ):
        raise embed_job.EmbedStateError(
            "live Qdrant collection does not reproduce the reviewed complete "
            "configuration bound before collection creation"
        )
    binding = embed_job.prepare_binding(
        cfg,
        sealed,
        checksum=checksum_reference,
        worker_count=worker_count,
        collection_configuration=collection_configuration,
        collection_configuration_sha256=collection_configuration_sha,
        storage_identity_sha256=storage_sha,
        reviewed_plan_sha256=reviewed_plan_sha256,
        resume=args.resume,
        recover_initialization=(
            initialization is not None and initialization.recovered
        ),
    )
    if not args.resume:
        embed_job.initialize_coordinator(
            binding,
            recover=bool(initialization and initialization.recovered),
        )
        if initialize_workers is not None:
            print(
                f"Initialized immutable coordinator for {worker_count} workers; "
                "launch every worker with --resume and its exact --shard i/N."
            )
            return
    embed_job.load_coordinator(binding)
    checkpoints = embed_job.preflight_checkpoints(
        binding,
        sources,
        shard,
        resume=True,
    )
    store.prepare_embed_collection(
        client,
        cfg,
        resume=True,
        recreate=False,
        apply=args.apply,
        minimum_points=sum(
            checkpoint["chunks_completed"] for checkpoint in checkpoints.values()
        )
    )
    embed_job.verify_resume_checkpoint_points(
        cfg,
        client,
        sealed,
        checkpoints,
        count_tokens=count_tokens,
    )

    print(
        f"Loading embedding model {cfg.embed_model} "
        f"(device={cfg.embed_device or 'auto/cpu'}, fp16={cfg.embed_use_fp16}, "
        f"batch={cfg.embed_batch_size})..."
    )
    embedder = BGEM3Embedder(cfg)
    observed_checksum = embed_job.vector_checksum_value(embedder)
    digest = observed_checksum["probe_sha256"]
    if digest != checksum_reference.probe_sha256:
        raise embed_job.EmbedStateError(
            "loaded embedding runtime differs from the bound exact probe-suite checksum"
        )
    vec = observed_checksum["probes"][0]["dense"]
    print(
        f"VECTOR-SPACE CHECKSUM: sha={digest}  "
        f"dims[:8]={[round(x, 5) for x in vec[:8]]}"
    )

    grand_d = grand_c = 0
    for source in sources:
        docs, chunks, skipped = embed_job.embed_source_resumable(
            cfg,
            client,
            embedder,
            count_tokens,
            source,
            binding=binding,
            snapshot_docs=sealed.docs,
            resume=True,
            prepared_checkpoint=checkpoints[source],
            batch_size=batch_size,
            shard=shard,
            mutation_capability=mutation_capability,
            launch_evidence_sha256=launch_evidence_sha256,
        )
        print(f"  {source}: {docs} docs -> {chunks} chunks ({skipped} skipped)")
        grand_d += docs
        grand_c += chunks
    info = client.get_collection(cfg.collection_name)
    print(
        f"Done: {grand_d} docs -> {grand_c} chunks (0 skipped). "
        f"Collection '{cfg.collection_name}' now has {info.points_count} points."
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
    p_ing.add_argument(
        "--apply",
        action="store_true",
        help="permit an approved immutable-generation Qdrant write",
    )
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
    p_watch.add_argument(
        "--apply",
        action="store_true",
        help="permit an approved immutable-generation Qdrant write",
    )
    p_watch.set_defaults(func=_cmd_watch)

    p_snap = sub.add_parser(
        "snapshot", help="build a clean, deduplicated, versioned corpus snapshot (Part 1)")
    p_snap.add_argument(
        "--snapshot-id",
        required=True,
        help="new immutable non-v1 snapshot identifier",
    )
    p_snap.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="parent directory for the create-only snapshot",
    )
    p_snap.add_argument(
        "--source-state-evidence",
        type=Path,
        help="production run-selection ledger with exact items/run.json hashes",
    )
    p_snap.add_argument(
        "--preflight",
        action="store_true",
        help="build a non-production snapshot below ingest/.state/v3",
    )
    p_snap.add_argument("--source", default="all", help="spider name or 'all' (present sources)")
    p_snap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap docs per source (only valid with --preflight)",
    )
    p_snap.add_argument("--no-near-dup", action="store_true",
                        help="skip the MinHash/LSH near-duplicate pass (faster)")
    p_snap.add_argument("--token-sample", type=int, default=2000,
                        help="docs/source to sample for the BGE-M3 token-length profile")
    p_snap.set_defaults(func=_cmd_snapshot)

    p_embed = sub.add_parser(
        "embed", help="embed one sealed snapshot into its immutable physical collection")
    p_embed.add_argument(
        "--snapshot-docs",
        required=True,
        type=Path,
        help="exact SNAPSHOT/docs directory beside a sealed production manifest",
    )
    p_embed.add_argument("--source", default="all", help="source name or 'all'")
    p_embed.add_argument("--checksum", action="store_true",
                         help="create only the CPU/GPU vector-space checksum artifact")
    p_embed.add_argument(
        "--checksum-output",
        type=Path,
        help="create-only checksum JSON path outside the immutable snapshot",
    )
    p_embed.add_argument(
        "--vector-checksum",
        type=Path,
        help="validated exact dense+sparse probe artifact to bind to embedding state",
    )
    p_embed.add_argument(
        "--storage-identity",
        type=Path,
        help="persistent-volume identity file (required for multi-worker initialization)",
    )
    p_embed.add_argument(
        "--qdrant-storage-root",
        type=Path,
        help="explicit mounted Qdrant storage root bound to checkpoints and volume identity",
    )
    p_embed.add_argument(
        "--plan",
        type=Path,
        help="create-only reviewed immutable GPU workflow plan",
    )
    p_embed.add_argument(
        "--review",
        type=Path,
        help="paid-compute approval sidecar for --plan",
    )
    p_embed.add_argument(
        "--bundle-root",
        type=Path,
        help="exact validated frozen release-input bundle bound by --plan",
    )
    p_embed.add_argument(
        "--initialize-workers",
        type=int,
        metavar="N",
        help="create one binding and every zero checkpoint for N resume-only workers",
    )
    p_embed.add_argument("--batch-size", type=int, default=256, help="points per upsert batch")
    p_embed.add_argument(
        "--recreate",
        action="store_true",
        help="forbidden for immutable generation candidates (retained to fail closed)",
    )
    p_embed.add_argument(
        "--resume",
        action="store_true",
        help="require and resume the exact existing binding, checkpoints, and collection",
    )
    p_embed.add_argument(
        "--apply",
        action="store_true",
        help="permit an approved immutable-generation Qdrant write",
    )
    p_embed.add_argument("--shard", default=None, metavar="i/n",
                         help="embed only shard i of n (0-indexed) — one process per GPU into "
                              "the same Qdrant; each shard has its own resumable checkpoint")
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
