#!/usr/bin/env python3
"""Validate immutable candidate release inputs and write one create-only result."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

INGEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INGEST_ROOT.parent
if str(INGEST_ROOT) not in sys.path:
    sys.path.insert(0, str(INGEST_ROOT))

from ingest.release_inputs import (  # noqa: E402
    GENERATION_ID,
    RELEASE_BUNDLE_PATH,
    validate_release_inputs,
)

DEFAULT_BUNDLE = Path(RELEASE_BUNDLE_PATH)


def _create_report(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        payload = (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checked_at = (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    try:
        validated = validate_release_inputs(
            args.bundle,
            repo_root=args.repo_root,
        )
    except Exception as exc:  # noqa: BLE001 - every gate failure must emit a stop report
        reason = str(exc).strip() or type(exc).__name__
        report = {
            "schema_version": 1,
            "generation_id": GENERATION_ID,
            "status": "inconclusive",
            "ceiling": "no_network_or_paid_work",
            "checked_at": checked_at,
            "bundle": os.fspath(args.bundle.expanduser().absolute()),
            "reason": reason,
        }
        _create_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 2
    report = {**validated.report(), "checked_at": checked_at}
    _create_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
