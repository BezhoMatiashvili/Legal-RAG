#!/usr/bin/env python3
"""Plan conservative raw-artifact pruning; deletion requires two explicit approvals."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

INGEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INGEST_ROOT.parent
sys.path.insert(0, str(INGEST_ROOT))

from ingest.artifacts import (  # noqa: E402
    apply_prune_plan,
    build_prune_plan,
    load_prune_plan,
    parse_utc_datetime,
    prune_plan_document,
    write_prune_plan,
)

APPROVAL_ENV = "ARTIFACT_PRUNE_APPROVED"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="raw crawler artifacts root (default: repository artifacts/)",
    )
    parser.add_argument(
        "--generations-root",
        type=Path,
        default=INGEST_ROOT / "snapshots",
        help="immutable generation root containing verified manifests",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        help="required dry-run output; created owner-only and never overwritten",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        help="previously reviewed plan artifact required by --apply",
    )
    parser.add_argument(
        "--now",
        type=parse_utc_datetime,
        help="timezone-aware planning timestamp (for reproducible audits)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "apply exactly --plan candidates; also requires "
            f"{APPROVAL_ENV}=1"
        ),
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    environment = os.environ if environ is None else environ

    if args.apply:
        if args.plan is None:
            parser.error("--apply requires a separately persisted --plan artifact")
        if args.plan_output is not None or args.now is not None:
            parser.error("--apply cannot create a plan or override its planning timestamp")
        if environment.get(APPROVAL_ENV) != "1":
            parser.error(f"--apply also requires {APPROVAL_ENV}=1")
        plan = load_prune_plan(args.plan)
        deleted = apply_prune_plan(args.plan, approved=True)
        mode = "apply"
    else:
        if args.plan is not None:
            parser.error("--plan is accepted only with --apply")
        if args.plan_output is None:
            parser.error("dry-run requires --plan-output for later independent review")
        plan = build_prune_plan(
            args.artifacts_root,
            args.generations_root,
            now=args.now,
        )
        write_prune_plan(args.plan_output, plan)
        deleted = ()
        mode = "dry-run"

    report = prune_plan_document(plan)
    report["mode"] = mode
    report["deleted"] = list(deleted)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
