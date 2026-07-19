#!/usr/bin/env python3
"""NON-PRODUCTION diagnostic scan of an already-indexed Qdrant collection.

READ-ONLY: this only calls ``scroll`` against Qdrant.  It emits diagnostic document and
sample reconstructions, never source-state or publisher inputs.  Production preparation
is exclusively ``scripts/prepare_generation.py`` against a sealed attested snapshot and
an exact physical generation collection.

Building a full ``GenerationManifest`` additionally needs the model/tokenizer/reranker
revision pins and a dependency lock hash that do not exist in this repo yet — see
``ingest/serverless/runtime-identity.unconfigured.json``. This script does not attempt to
fabricate them; it only produces the parts that are honestly derivable from the live
corpus today, and prints a summary of exactly what it found (including every
:class:`~ingest.generation.ScanIssue`) so a human can decide what to do next.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

INGEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INGEST_ROOT))

from ingest.config import load_config  # noqa: E402
from ingest import qdrant_store as store  # noqa: E402
from ingest.generation_scan import aggregate_documents  # noqa: E402

_PAYLOAD_FIELDS = [
    "source", "document_id", "chunk_index", "text", "content_hash", "document_state_hash",
    "content_kind", "extraction_status", "content_complete", "article_summary",
    "source_binary_url",
]


def _scroll_source(client, collection: str, source: str | None, batch_size: int):
    scroll_filter = None
    if source is not None:
        from qdrant_client import models

        scroll_filter = models.Filter(
            must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))]
        )
    offset = None
    seen = 0
    t0 = time.time()
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            with_payload=_PAYLOAD_FIELDS,
            with_vectors=False,
            limit=batch_size,
            offset=offset,
        )
        for p in points:
            yield p.payload or {}
        seen += len(points)
        if seen and seen % 200_000 < batch_size:
            print(f"  ...scrolled {seen} points ({time.time() - t0:.0f}s)", file=sys.stderr)
        if offset is None:
            break
    print(f"  scrolled {seen} points total in {time.time() - t0:.0f}s", file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-id", required=True,
        help="candidate generation id for the output records (not yet published anywhere)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", default=None, help="scan only this source (bounds memory/time)")
    parser.add_argument("--collection", default=None, help="override Config.collection_name")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="required acknowledgement that output is not publishable production evidence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.diagnostic:
        raise SystemExit(
            "refusing production-like candidate scan: pass --diagnostic to acknowledge "
            "that this output cannot be published"
        )
    cfg = load_config()
    collection = args.collection or cfg.collection_name
    client = store.make_client(cfg)

    print(f"Scanning collection={collection!r} source={args.source!r} (read-only)...", file=sys.stderr)
    result = aggregate_documents(
        _scroll_source(client, collection, args.source, args.batch_size),
        generation_id=args.generation_id,
        cfg=cfg,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = args.output_dir / "documents.jsonl"
    samples_path = args.output_dir / "sample_checks.jsonl"

    with documents_path.open("w", encoding="utf-8") as fh:
        for record in result.documents:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    with samples_path.open("w", encoding="utf-8") as fh:
        for sample in result.samples:
            fh.write(json.dumps(sample, sort_keys=True, ensure_ascii=False) + "\n")

    print(f"documents:  {len(result.documents)} -> {documents_path}", file=sys.stderr)
    print(f"samples:    {len(result.samples)} -> {samples_path}", file=sys.stderr)
    print(f"chunks:     {result.chunk_count}", file=sys.stderr)
    if result.issues:
        print(f"ISSUES: {len(result.issues)} (first 20 shown)", file=sys.stderr)
        for issue in result.issues[:20]:
            print(f"  {issue.source}:{issue.document_id}: {issue.reason}", file=sys.stderr)
    else:
        print("No issues.", file=sys.stderr)
    print(
        "DIAGNOSTIC ONLY: no source_state.json or publishable prepared directory was "
        "emitted. Use scripts/prepare_generation.py for production.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
