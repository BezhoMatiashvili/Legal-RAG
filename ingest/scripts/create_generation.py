#!/usr/bin/env python3
"""Publish an exact sealed prepared generation; never accepts loose artifact files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

INGEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INGEST_ROOT))

from ingest.config import load_config  # noqa: E402
from ingest.generation_prepare import (  # noqa: E402
    GenerationPreparationError,
    load_prepared_generation,
    validate_prepared_configuration,
)
from ingest.generation_snapshot import publish_generation  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation",
        required=True,
        help="explicit non-v1 generation ID (must match the prepared manifest)",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        required=True,
        help="checksum-sealed output of scripts/prepare_generation.py",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prepared = load_prepared_generation(args.prepared_dir)
        if prepared.manifest.generation_id != args.generation:
            raise GenerationPreparationError(
                "explicit generation ID does not match prepared manifest"
            )
        validate_prepared_configuration(prepared, load_config())
        publication_provenance = {
            "prepared_checksums_sha256": prepared.checksums_sha256,
            "evidence": dict(prepared.provenance),
        }
        destination = publish_generation(
            args.output_root,
            args.generation,
            prepared.manifest,
            prepared.iter_documents(),
            prepared.iter_samples(),
            prepared.source_state,
            preparation_provenance=publication_provenance,
        )
    except GenerationPreparationError as exc:
        raise SystemExit(f"generation publication refused: {exc}") from exc
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
