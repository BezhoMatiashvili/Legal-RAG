#!/usr/bin/env python3
"""Create/review and enforce the immutable v3 GPU Qdrant snapshot workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INGEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INGEST_ROOT))

from ingest.config import load_config  # noqa: E402
from ingest.embed_job import (  # noqa: E402
    load_checksum_reference,
    save_checksum_comparison,
)
from ingest.gpu_workflow import (  # noqa: E402
    GpuWorkflowError,
    create_workflow_plan,
    create_workflow_review,
    restore_local_export,
    seal_remote_export,
)
from ingest.qdrant_store import make_client  # noqa: E402


def _container_paths(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise GpuWorkflowError("--container-path must have KEY=/absolute/path form")
        key, value = raw.split("=", 1)
        if not key or key in result:
            raise GpuWorkflowError("--container-path keys must be unique and non-empty")
        result[key] = value
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan", help="create the validated non-executable plan"
    )
    plan.add_argument("--bundle", type=Path, required=True)
    plan.add_argument("--snapshot-docs", type=Path, required=True)
    plan.add_argument("--cpu-checksum", type=Path, required=True)
    plan.add_argument("--storage-identity", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--workflow-id", required=True)
    plan.add_argument(
        "--container-path",
        action="append",
        default=[],
        metavar="KEY=/ABSOLUTE/PATH",
        help="repeat for every path named by the workflow contract",
    )
    plan.add_argument("--volume-size-gib", type=int, required=True)
    plan.add_argument("--worker-count", type=int, required=True)
    plan.add_argument("--gpu-sku", required=True)
    plan.add_argument("--gpu-count", type=int, required=True)
    plan.add_argument("--total-hourly-usd", type=float, required=True)
    plan.add_argument("--storage-gib-month-usd", type=float, required=True)
    plan.add_argument("--max-runtime-hours", type=float, required=True)
    plan.add_argument("--max-exposure-usd", type=float, required=True)
    plan.add_argument("--auto-teardown", action="store_true", required=True)

    review = subparsers.add_parser(
        "review", help="create the explicit paid-compute review sidecar"
    )
    review.add_argument("--plan", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--reviewed-at", required=True)
    review.add_argument("--paid-approval-id", required=True)

    compare = subparsers.add_parser(
        "compare-checksums", help="seal the fixed >=0.999 CPU/GPU cosine gate"
    )
    compare.add_argument("--cpu", type=Path, required=True)
    compare.add_argument("--runtime", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    export = subparsers.add_parser(
        "seal-export",
        help="seal the complete remote collection and export its snapshot",
    )
    export.add_argument("--plan", type=Path, required=True)
    export.add_argument("--review", type=Path, required=True)
    export.add_argument("--qdrant-storage-root", type=Path, required=True)
    export.add_argument("--snapshot-output", type=Path, required=True)
    export.add_argument("--manifest-output", type=Path, required=True)
    export.add_argument("--apply", action="store_true")

    restore = subparsers.add_parser(
        "restore-local",
        help="restore only the absent generation target and prove its digest",
    )
    restore.add_argument("--plan", type=Path, required=True)
    restore.add_argument("--review", type=Path, required=True)
    restore.add_argument("--export-manifest", type=Path, required=True)
    restore.add_argument("--snapshot", type=Path, required=True)
    restore.add_argument("--proof-output", type=Path, required=True)
    restore.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            cfg = load_config()
            output = create_workflow_plan(
                cfg=cfg,
                bundle_root=args.bundle,
                snapshot_docs=args.snapshot_docs,
                cpu_checksum=args.cpu_checksum,
                storage_identity=args.storage_identity,
                output=args.output,
                workflow_id=args.workflow_id,
                container_paths=_container_paths(args.container_path),
                volume_size_gib=args.volume_size_gib,
                worker_count=args.worker_count,
                gpu_sku=args.gpu_sku,
                gpu_count=args.gpu_count,
                total_hourly_usd=args.total_hourly_usd,
                storage_gib_month_usd=args.storage_gib_month_usd,
                max_runtime_hours=args.max_runtime_hours,
                max_exposure_usd=args.max_exposure_usd,
                auto_teardown=args.auto_teardown,
            )
        elif args.command == "review":
            output = create_workflow_review(
                args.plan,
                args.output,
                reviewer=args.reviewer,
                reviewed_at=args.reviewed_at,
                paid_approval_id=args.paid_approval_id,
            )
        elif args.command == "compare-checksums":
            comparison = save_checksum_comparison(
                load_checksum_reference(args.cpu),
                load_checksum_reference(args.runtime),
                args.output,
            )
            output = comparison.path
        elif args.command == "seal-export":
            if not args.apply:
                raise GpuWorkflowError("seal-export requires --apply")
            cfg = load_config()
            client = make_client(cfg)
            try:
                output = seal_remote_export(
                    client,
                    cfg,
                    plan_path=args.plan,
                    review_path=args.review,
                    qdrant_storage_root=args.qdrant_storage_root,
                    snapshot_output=args.snapshot_output,
                    manifest_output=args.manifest_output,
                )
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        else:
            cfg = load_config()
            client = make_client(cfg)
            try:
                output = restore_local_export(
                    client,
                    cfg,
                    plan_path=args.plan,
                    review_path=args.review,
                    export_manifest_path=args.export_manifest,
                    snapshot_path=args.snapshot,
                    proof_output=args.proof_output,
                    apply=args.apply,
                )
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
    except Exception as exc:  # noqa: BLE001 - all uncertainty is a failed workflow
        print(f"immutable GPU workflow failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "artifact": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
