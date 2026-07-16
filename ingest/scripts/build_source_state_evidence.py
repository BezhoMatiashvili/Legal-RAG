#!/usr/bin/env python3
"""Create a reviewed-run candidate ledger from explicit exact crawl selections."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

INGEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INGEST_ROOT.parent
sys.path.insert(0, str(INGEST_ROOT))

from ingest.source_state import (  # noqa: E402
    SourceStateError,
    build_source_state_evidence,
    source_state_evidence_companion_paths,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="crawl artifacts root (default: repository artifacts/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new evidence path; existing paths are never replaced",
    )
    parser.add_argument(
        "--select",
        action="append",
        required=True,
        metavar="SOURCE:RUN_ID",
        help="exact run selection; repeat for all seven production sources",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = build_source_state_evidence(
            args.artifacts_root,
            args.output,
            args.select,
        )
    except SourceStateError as exc:
        raise SystemExit(f"source-state evidence refused: {exc}") from exc
    output = args.output.expanduser().absolute()
    candidate, authorization = source_state_evidence_companion_paths(output)
    print(
        "*** AWAITING INDEPENDENT OPERATOR REVIEW ***\n"
        f"created {output} with "
        f"{len(evidence['runs'])} exact run selection(s); do not use it for a "
        "production snapshot until an independent operator has reviewed every row.\n"
        "The evidence is an inseparable same-directory trio:\n"
        f"  output:        {output}\n"
        f"  candidate:     {candidate}\n"
        f"  authorization: {authorization}\n"
        "Preserve all three exact files together; never copy or move the JSON alone."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
