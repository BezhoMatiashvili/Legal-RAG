#!/usr/bin/env python3
"""Create a promotion plan or execute it through an explicitly supplied backend."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

INGEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INGEST_ROOT))

from ingest.promotion import (  # noqa: E402
    FORWARD_APPROVAL_ENV,
    PROMOTION_APPROVAL_ENV,
    create_promotion_plan,
    execute_promotion,
    load_promotion_plan,
    write_promotion_plan,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="persist an immutable local promotion plan")
    plan.add_argument("--generation-root", type=Path, required=True)
    plan.add_argument("--snapshot-ref", required=True)
    plan.add_argument("--snapshot-sha256", required=True)
    plan.add_argument("--created-by", required=True)
    plan.add_argument("--promotion-id")
    plan.add_argument("--output", type=Path, required=True)

    apply = commands.add_parser(
        "apply", help="execute a plan through a backend factory"
    )
    apply.add_argument("--plan", type=Path, required=True)
    apply.add_argument("--state", type=Path, required=True)
    apply.add_argument(
        "--backend-factory",
        required=True,
        help="explicit module:callable returning a PromotionBackend",
    )
    apply.add_argument("--optimizer-timeout", type=float, default=900.0)
    apply.add_argument("--apply", action="store_true")
    apply.add_argument("--forward-after-rollback", action="store_true")
    return parser


def _backend_factory(specification: str):
    try:
        module_name, attribute = specification.split(":", 1)
        factory = getattr(importlib.import_module(module_name), attribute)
        backend = factory()
    except (ValueError, ImportError, AttributeError, TypeError) as exc:
        raise SystemExit(
            f"cannot load backend factory {specification!r}: {exc}"
        ) from exc
    return backend


def main(argv: list[str] | None = None, *, environ=None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    environment = os.environ if environ is None else environ

    if args.command == "plan":
        plan = create_promotion_plan(
            args.generation_root,
            snapshot_ref=args.snapshot_ref,
            snapshot_sha256=args.snapshot_sha256,
            created_by=args.created_by,
            promotion_id=args.promotion_id,
        )
        destination = write_promotion_plan(args.output, plan)
        print(destination)
        return 0

    if not args.apply or environment.get(PROMOTION_APPROVAL_ENV) != "1":
        parser.error(
            f"alias mutation requires both --apply and {PROMOTION_APPROVAL_ENV}=1"
        )
    if args.forward_after_rollback and environment.get(FORWARD_APPROVAL_ENV) != "1":
        parser.error(f"final forward switch also requires {FORWARD_APPROVAL_ENV}=1")
    # Parse and reject a reserved frozen-candidate plan before importing/calling a backend
    # factory, which may construct clients or models as a side effect.
    load_promotion_plan(args.plan)
    backend = _backend_factory(args.backend_factory)
    state = execute_promotion(
        args.plan,
        args.state,
        backend,
        forward_after_rollback=args.forward_after_rollback,
        optimizer_timeout_seconds=args.optimizer_timeout,
    )
    print(json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
