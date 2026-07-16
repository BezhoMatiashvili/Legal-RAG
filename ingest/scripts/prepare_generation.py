#!/usr/bin/env python3
"""Prepare a sealed immutable generation by read-only scanning exact physical Qdrant."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

INGEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INGEST_ROOT))

from ingest import qdrant_store as store  # noqa: E402
from ingest.config import load_config  # noqa: E402
from ingest.generation_prepare import (  # noqa: E402
    GenerationPreparationError,
    prepare_generation,
    validate_preparation_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", required=True)
    parser.add_argument(
        "--snapshot",
        required=True,
        type=Path,
        help="sealed production snapshot directory (preflight snapshots are rejected)",
    )
    parser.add_argument("--physical-collection", required=True)
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument("--runtime-identity", required=True, type=Path)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = dataclasses.replace(
        load_config(),
        generation_id=args.generation_id,
        collection_name=args.physical_collection,
    )
    try:
        # All immutable local identity checks run before client creation/network access.
        validated = validate_preparation_inputs(
            cfg,
            generation_id=args.generation_id,
            snapshot_root=args.snapshot,
            physical_collection=args.physical_collection,
            dependency_lock=args.dependency_lock,
            runtime_identity=args.runtime_identity,
            image_digest=args.image_digest,
            output_dir=args.output_dir,
        )
        client = store.make_client(cfg)
        destination = prepare_generation(
            client,
            cfg,
            generation_id=args.generation_id,
            snapshot_root=args.snapshot,
            physical_collection=args.physical_collection,
            dependency_lock=args.dependency_lock,
            runtime_identity=args.runtime_identity,
            image_digest=args.image_digest,
            actor=args.actor,
            run_id=args.run_id,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            validated_inputs=validated,
        )
    except GenerationPreparationError as exc:
        raise SystemExit(f"generation preparation refused: {exc}") from exc
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
