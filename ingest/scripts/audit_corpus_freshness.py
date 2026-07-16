#!/usr/bin/env python
"""Write an immutable weekly corpus freshness/completeness audit report.

This command is offline and read-only with respect to the corpus and serving systems.  Its
only write is a new owner-only JSON report in an already-existing report directory.  Exit
status is 0 for a passing audit, 1 for a recorded SLA breach, and 2 for invalid input or an
operational failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.freshness import (  # noqa: E402
    FreshnessAuditError,
    audit_generation_directory,
    report_filename,
    write_audit_report,
)
from ingest.generation import GenerationFormatError, parse_rfc3339_utc  # noqa: E402


def _audited_at(value: str) -> datetime:
    try:
        return parse_rfc3339_utc(value, field="--audited-at")
    except GenerationFormatError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "generation_dir",
        type=Path,
        help="checksum-verified immutable generation directory",
    )
    parser.add_argument(
        "audit_input",
        type=Path,
        help="immutable official-source observation JSON bound to the generation",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="existing directory in which a new report will be created without replacement",
    )
    parser.add_argument(
        "--audited-at",
        type=_audited_at,
        default=None,
        help="deterministic RFC3339 UTC audit time (defaults to current UTC time)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if not args.report_dir.is_dir() or args.report_dir.is_symlink():
            raise FreshnessAuditError(
                f"report directory must already exist and not be a symlink: {args.report_dir}"
            )
        report = audit_generation_directory(
            args.generation_dir,
            args.audit_input,
            audited_at=args.audited_at or datetime.now(timezone.utc),
        )
        destination = args.report_dir / report_filename(report)
        write_audit_report(destination, report)
    except Exception as exc:  # noqa: BLE001 - the scheduled audit must fail closed
        print(f"corpus freshness audit failed: {exc}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "ok": report.ok,
                "generation_id": report.generation_id,
                "audit_id": report.audit_id,
                "breach_count": len(report.breaches),
                "report": str(destination),
            },
            sort_keys=True,
        )
    )
    if not report.ok:
        for breach in report.breaches:
            location = breach.source or "generation"
            print(f"{location}: {breach.code}: {breach.detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
