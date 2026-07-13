#!/usr/bin/env python
"""Merge a complete delta Qdrant collection into the main collection safely.

The full-corpus GPU embed produced the whole collection and loaded it with a snapshot
*restore* (which replaces the target). An INCREMENTAL embed — the sweep's ~23k newly
scraped docs — instead lands in a small ``georgian_legal_delta`` collection (embedded on
the RunPod GPU, snapshot-transferred back and restored locally under that name). This
script preflights document completeness, upserts the current points, then removes obsolete
high-index tail chunks from shortened documents in the main ``georgian_legal`` collection.

Point ids are deterministic UUIDv5 over ``(source, document_id, chunk_index)``
(``qdrant_store.point_id``), so the upsert is idempotent and additive: re-running never
duplicates, a re-embedded doc's chunks overwrite in place, and only genuinely new chunks
grow the collection. Dense (1024-d) + learned-sparse vectors and the full payload are
copied verbatim. Every current delta point must carry ``document_chunk_count``; legacy or
partial deltas fail closed and must be re-embedded before they can be merged.

Usage (from ingest/):
    .venv/bin/python scripts/merge_delta_collection.py --src georgian_legal_delta
    .venv/bin/python scripts/merge_delta_collection.py --dry-run   # self-test, no main mutation
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ root → `import ingest`

from qdrant_client import models  # noqa: E402

from ingest.config import load_config  # noqa: E402
from ingest.qdrant_store import ensure_collection, make_client  # noqa: E402


def _to_struct(point, *, payload: dict | None = None) -> models.PointStruct:
    """Rebuild an upsertable PointStruct from a scrolled Record (named dense[+sparse] vectors).

    A point may legitimately carry NO sparse vector: BGE-M3 yields empty sparse for some
    inputs, and ``_build_doc_points`` stores the ``sparse`` named vector only when it has
    indices. Guard the lookup so a dense-only point doesn't ``KeyError`` and abort the whole
    (non-resumable) merge. Points that DO carry sparse reconstruct byte-identically.
    """
    vec = point.vector
    out_vec = {"dense": vec["dense"]}
    sparse = vec.get("sparse") if isinstance(vec, dict) else None
    if sparse is not None:
        if not isinstance(sparse, models.SparseVector):
            sparse = models.SparseVector(indices=sparse["indices"], values=sparse["values"])
        out_vec["sparse"] = sparse
    return models.PointStruct(
        id=point.id,
        vector=out_vec,
        payload=point.payload if payload is None else payload,
    )


@dataclass(frozen=True)
class _DocVersion:
    content_hash: str
    chunk_count: int


def _point_metadata(point) -> tuple[tuple[str, str], int, _DocVersion | None]:
    """Return ``((source, document_id), chunk_index, version)`` for one delta point.

    Older stale tails may predate the completeness marker and therefore return ``None`` for
    ``version``. Chunk zero may never be legacy: it identifies the current document version.
    """
    payload = point.payload or {}
    source = payload.get("source")
    document_id = payload.get("document_id")
    chunk_index = payload.get("chunk_index")
    if not isinstance(source, str) or not source:
        raise RuntimeError(f"delta point {point.id!r} has no source")
    if not isinstance(document_id, str) or not document_id:
        raise RuntimeError(f"delta point {point.id!r} has no document_id")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise RuntimeError(f"delta point {point.id!r} has invalid chunk_index={chunk_index!r}")

    content_hash = payload.get("content_hash")
    chunk_count = payload.get("document_chunk_count")
    version = None
    if (isinstance(content_hash, str) and content_hash
            and isinstance(chunk_count, int) and not isinstance(chunk_count, bool)
            and chunk_count > 0):
        version = _DocVersion(content_hash, chunk_count)
    return (source, document_id), chunk_index, version


def _scan_complete_docs(client, src: str, *, batch_size: int) -> dict[tuple[str, str], _DocVersion]:
    """Identify each current doc by chunk zero and prove its full ``0..N-1`` set exists."""
    indices: dict[tuple[str, str], dict[_DocVersion, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    current: dict[tuple[str, str], _DocVersion] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=src,
            limit=batch_size,
            with_vectors=False,
            with_payload=True,
            offset=offset,
        )
        for point in points:
            key, chunk_index, version = _point_metadata(point)
            if chunk_index == 0:
                if version is None:
                    raise RuntimeError(
                        f"delta {src!r} is legacy/incomplete at {key[0]}:{key[1]} chunk 0; "
                        "re-embed it with document_chunk_count before merging"
                    )
                current[key] = version
            if version is not None:
                indices[key][version].add(chunk_index)
        if offset is None:
            break

    errors: list[str] = []
    for key in indices:
        if key not in current:
            errors.append(f"{key[0]}:{key[1]} has no current chunk 0")
    for key, version in current.items():
        actual = indices[key].get(version, set())
        expected = set(range(version.chunk_count))
        if actual != expected:
            missing = sorted(expected - actual)[:8]
            extra = sorted(actual - expected)[:8]
            errors.append(
                f"{key[0]}:{key[1]} expected 0..{version.chunk_count - 1}; "
                f"missing={missing} extra={extra}"
            )
    if errors:
        detail = "; ".join(errors[:10])
        raise RuntimeError(
            f"delta completeness preflight failed for {len(errors)} document(s): {detail}"
        )
    if not current:
        raise RuntimeError(f"delta collection {src!r} contains no complete documents")
    return current


def _delete_stale_tails(
    client,
    dst: str,
    manifests: dict[tuple[str, str], _DocVersion],
    *,
    batch_size: int = 128,
) -> None:
    """Delete obsolete destination chunks in bounded batches, after all upserts succeed."""
    items = list(manifests.items())
    for start in range(0, len(items), batch_size):
        should = []
        for (source, document_id), version in items[start:start + batch_size]:
            should.append(models.Filter(must=[
                models.FieldCondition(key="source", match=models.MatchValue(value=source)),
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                ),
                models.FieldCondition(
                    key="chunk_index", range=models.Range(gte=version.chunk_count)
                ),
            ]))
        client.delete(
            collection_name=dst,
            points_selector=models.FilterSelector(filter=models.Filter(should=should)),
            wait=True,
        )


def merge_collection(client, src: str, dst: str, *, batch_size: int = 256) -> int:
    """Merge complete current docs and safely remove destination tails. Fail closed."""
    manifests = _scan_complete_docs(client, src, batch_size=batch_size)
    offset = None
    total = 0
    merged_indices: dict[tuple[str, str], set[int]] = defaultdict(set)
    while True:
        points, offset = client.scroll(
            collection_name=src,
            limit=batch_size,
            with_vectors=True,
            with_payload=True,
            offset=offset,
        )
        eligible = []
        for point in points:
            key, chunk_index, version = _point_metadata(point)
            if version == manifests.get(key) and chunk_index < version.chunk_count:
                eligible.append(_to_struct(point))
                merged_indices[key].add(chunk_index)
        if eligible:
            client.upsert(collection_name=dst, points=eligible, wait=True)
            total += len(eligible)
        if offset is None:
            break

    incomplete = [
        f"{source}:{document_id}"
        for (source, document_id), version in manifests.items()
        if merged_indices.get((source, document_id), set()) != set(range(version.chunk_count))
    ]
    if incomplete:
        raise RuntimeError(
            "delta changed during merge; refusing stale-tail deletion for incomplete docs: "
            + ", ".join(incomplete[:10])
        )

    # Deletion is deliberately last: a partial/failed upsert can be retried idempotently and
    # can never erase legitimate chunks from the prior complete destination version.
    _delete_stale_tails(client, dst, manifests)
    return total


def _count(client, name: str) -> int:
    return client.count(collection_name=name, exact=True).count


def dry_run(cfg, client) -> None:
    """Prove the merge on real embedded data (the firearms 495 chunks) — main is untouched."""
    src, dst = "_merge_test_src", "_merge_test_dst"
    for name in (src, dst):
        ensure_collection(client, dataclasses.replace(cfg, collection_name=name), recreate=True)

    # Seed src with the firearms points copied out of the main collection (real vectors+payload).
    # The live base predates ``document_chunk_count``, so infer and stamp it on this complete
    # sample before exercising the same fail-closed merge path used for new delta collections.
    sample_points, offset = [], None
    while True:
        pts, offset = client.scroll(
            collection_name=cfg.collection_name,
            limit=256, with_vectors=True, with_payload=True, offset=offset,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="document_id", match=models.MatchAny(any=["18070", "14944"]))
            ]),
        )
        sample_points.extend(pts)
        if offset is None:
            break
    by_doc: dict[tuple[str, str], list] = defaultdict(list)
    for point in sample_points:
        payload = point.payload or {}
        by_doc[(payload.get("source"), payload.get("document_id"))].append(point)
    seeded = 0
    for key, doc_points in by_doc.items():
        indices = {int((point.payload or {})["chunk_index"]) for point in doc_points}
        chunk_count = max(indices, default=-1) + 1
        if indices != set(range(chunk_count)):
            raise RuntimeError(f"dry-run source sample {key} is not contiguous: {sorted(indices)}")
        stamped = [
            _to_struct(
                point,
                payload={**(point.payload or {}), "document_chunk_count": chunk_count},
            )
            for point in doc_points
        ]
        client.upsert(collection_name=src, points=stamped, wait=True)
        seeded += len(stamped)
    print(f"  seeded src with {seeded} firearms points")

    merged = merge_collection(client, src, dst)
    after_first = _count(client, dst)
    merge_collection(client, src, dst)  # idempotency: a second merge must not grow dst
    after_second = _count(client, dst)

    sample, _ = client.scroll(
        collection_name=dst, limit=1, with_vectors=True, with_payload=True,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="document_id", match=models.MatchValue(value="14944"))
        ]),
    )
    dense_ok = bool(sample) and len(sample[0].vector["dense"]) == cfg.dense_dim
    sparse_ok = bool(sample) and len(sample[0].vector["sparse"].indices) > 0
    payload_ok = bool(sample) and sample[0].payload.get("is_consolidated") is True

    print(f"  merged={merged} · dst after 1st={after_first} · after 2nd={after_second} "
          f"· idempotent={after_first == after_second}")
    print(f"  round-trip sample: dense_dim={len(sample[0].vector['dense']) if sample else None} "
          f"· sparse_terms={len(sample[0].vector['sparse'].indices) if sample else None} "
          f"· is_consolidated={sample[0].payload.get('is_consolidated') if sample else None}")

    for name in (src, dst):
        client.delete_collection(name)

    if not (seeded == after_first == after_second and dense_ok and sparse_ok and payload_ok):
        raise SystemExit("❌ DRY-RUN FAILED — see counts above")
    print(f"✅ DRY-RUN PASSED — {seeded} points merged idempotently with dense+sparse+payload "
          f"intact; temp collections dropped, main untouched.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", help="delta collection to merge from (e.g. georgian_legal_delta)")
    ap.add_argument("--dst", help="target collection (default: config collection_name)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dry-run", action="store_true",
                    help="self-test the merge on the firearms sample without touching main")
    args = ap.parse_args()

    cfg = load_config()
    client = make_client(cfg)

    if args.dry_run:
        dry_run(cfg, client)
        return

    if not args.src:
        raise SystemExit("--src is required (or use --dry-run)")
    dst = args.dst or cfg.collection_name
    before = _count(client, dst)
    merged = merge_collection(client, args.src, dst, batch_size=args.batch_size)
    after = _count(client, dst)
    print(f"Merged {merged} points from {args.src!r} into {dst!r}: "
          f"{before} → {after} (+{after - before} net new).")


if __name__ == "__main__":
    main()
