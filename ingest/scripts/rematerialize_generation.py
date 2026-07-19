#!/usr/bin/env python3
"""Copy existing Qdrant vectors into rebuilt court-aware payloads without embedding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INGEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INGEST_ROOT))

from ingest.config import load_config  # noqa: E402
from ingest.generation_rematerialize import (  # noqa: E402
    RematerializationError,
    load_source_evidence,
    refuse_frozen_candidate_rematerialization,
    rematerialize_generation,
    rematerialize_scratch,
)
from ingest.qdrant_store import make_client  # noqa: E402
from ingest.snapshot import verify_sealed_snapshot  # noqa: E402


def _source_file(value: str) -> tuple[str, Path]:
    source, separator, path = value.partition("=")
    if not separator or source not in {"ecd", "supremecourt"} or not path:
        raise argparse.ArgumentTypeError(
            "--source-file must be ecd=PATH or supremecourt=PATH"
        )
    return source, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    scratch = subparsers.add_parser(
        "scratch", help="write a run-scoped validation delta from read-only live points"
    )
    scratch.add_argument("--source-collection")
    scratch.add_argument(
        "--source-file",
        action="append",
        type=_source_file,
        required=True,
        help="repeat as ecd=PATH and supremecourt=PATH",
    )
    scratch.add_argument("--run-id", required=True)
    scratch.add_argument("--state-dir", type=Path)
    scratch.add_argument("--batch-size", type=int, default=256)
    scratch.add_argument("--resume", action="store_true")
    scratch.add_argument("--apply", action="store_true")

    production = subparsers.add_parser(
        "production",
        help="restore frozen legacy evidence and create an immutable physical generation",
    )
    production.add_argument("--source-evidence", required=True, type=Path)
    production.add_argument("--snapshot", required=True, type=Path)
    production.add_argument("--generation-id", required=True)
    production.add_argument("--vector-checksum", required=True, type=Path)
    production.add_argument("--actor", required=True)
    production.add_argument("--run-id", required=True)
    production.add_argument("--state-dir", type=Path)
    production.add_argument("--batch-size", type=int, default=256)
    production.add_argument("--resume", action="store_true")
    production.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "production":
        try:
            # Refuse the reserved release before configuration, evidence/model files,
            # or a Qdrant client can be accessed.
            refuse_frozen_candidate_rematerialization(args.generation_id)
        except RematerializationError as exc:
            raise SystemExit(f"rematerialization refused: {exc}") from exc
    cfg = load_config()
    state_dir = (args.state_dir or cfg.state_dir).expanduser().absolute()
    try:
        if args.mode == "production":
            # Run all filesystem/release-input refusals before client creation.  The
            # orchestration function repeats them so direct callers are equally safe.
            load_source_evidence(args.source_evidence)
            verify_sealed_snapshot(
                args.snapshot.expanduser().absolute(),
                allow_preflight=False,
                require_all_sources=True,
            )
        client = make_client(cfg)
        if args.mode == "scratch":
            source_files = dict(args.source_file)
            if len(source_files) != len(args.source_file):
                raise RematerializationError("duplicate --source-file source")
            result = rematerialize_scratch(
                client,
                cfg,
                source_collection=args.source_collection or cfg.collection_name,
                source_files=source_files,
                run_id=args.run_id,
                state_dir=state_dir,
                batch_size=args.batch_size,
                resume=args.resume,
                apply=args.apply,
            )
        else:
            result = rematerialize_generation(
                client,
                cfg,
                source_evidence_path=args.source_evidence,
                snapshot_root=args.snapshot,
                generation_id=args.generation_id,
                vector_checksum_path=args.vector_checksum,
                actor=args.actor,
                run_id=args.run_id,
                state_dir=state_dir,
                batch_size=args.batch_size,
                resume=args.resume,
                apply=args.apply,
            )
    except (RematerializationError, ValueError, OSError) as exc:
        raise SystemExit(f"rematerialization refused: {exc}") from exc
    print(
        json.dumps(
            {
                "target_collection": result.target_collection,
                "report_path": str(result.report_path),
                "document_count": result.document_count,
                "point_count": result.point_count,
                "source_logical_vector_sha256": result.source_logical_vector_sha256,
                "target_logical_vector_sha256": result.target_logical_vector_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
