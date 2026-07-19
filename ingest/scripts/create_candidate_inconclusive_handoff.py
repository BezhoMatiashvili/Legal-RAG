#!/usr/bin/env python3
"""Create the immutable candidate's fail-closed early-stop handoff report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

INGEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INGEST_ROOT.parent
if str(INGEST_ROOT) not in sys.path:
    sys.path.insert(0, str(INGEST_ROOT))

from ingest.candidate_handoff import (  # noqa: E402
    CandidateHandoffError,
    create_inconclusive_handoff,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--safety-inventory", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--expected-current-symbols-sha256",
        help=(
            "reviewed SHA-256 required when intentional code-map regeneration changed "
            "memory-bank/generated/symbols.md"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        checks = json.loads(args.checks.read_text(encoding="utf-8"))
        if not isinstance(checks, list):
            raise CandidateHandoffError("checks file must contain a list")
        output = create_inconclusive_handoff(
            validation_report=args.validation_report,
            safety_inventory=args.safety_inventory,
            checks=checks,
            output=args.output,
            repo_root=args.repo_root,
            expected_current_symbols_sha256=args.expected_current_symbols_sha256,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        CandidateHandoffError,
    ) as exc:
        raise SystemExit(f"inconclusive handoff refused: {exc}") from exc
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
