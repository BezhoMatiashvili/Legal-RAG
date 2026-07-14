#!/usr/bin/env python3
"""Atomically publish a prepared immutable generation; never overwrites a destination."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

INGEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INGEST_ROOT))

from ingest.generation import (  # noqa: E402
    iter_document_records,
    iter_sample_checks,
    load_manifest,
)
from ingest.generation_snapshot import publish_generation  # noqa: E402


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_nonstandard_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant: {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation", required=True, help="explicit non-v1 generation ID"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    try:
        source_state = json.loads(
            args.source_state.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(f"invalid source-state JSON: {exc}") from exc
    if not isinstance(source_state, dict):
        raise SystemExit("invalid source-state JSON: top-level value must be an object")
    destination = publish_generation(
        args.output_root,
        args.generation,
        manifest,
        iter_document_records(args.documents, expected_generation_id=args.generation),
        iter_sample_checks(args.samples, expected_generation_id=args.generation),
        source_state,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
