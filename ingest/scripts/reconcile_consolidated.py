#!/usr/bin/env python
"""Reconcile ``is_consolidated`` with matsne's own main-document classifier — no re-embed.

A ``doc_type=main`` crawl (``scrapy crawl matsne -a doc_type=main ...``) enumerates the
full matsne "ძირითადი (კონსოლიდირებული)" universe into
``artifacts/matsne/runs/<run>/main_listed_ids.txt`` — listing membership is the ONLY
authority for "main (consolidated) document" (a never-amended base act has no
publication switcher, so the switcher-derived flag under-covers; see
scripts/backfill_consolidation.py for that older, switcher-based backfill). This script:

1. idempotently ensures the ``is_consolidated`` BOOL / ``consolidated_count`` INTEGER
   payload indexes exist on the live collection (ensure_collection only creates them
   for a fresh collection);
2. sets ``is_consolidated=true`` (payload-only ``set_payload`` — vectors untouched,
   points_count unchanged, eval baseline unaffected) on every listed doc already in the
   index. True-only: it NEVER flips true→false;
3. reports coverage — |listed|, |in index|, |missing| — and writes the missing ids to
   ``missing_consolidated_ids.txt`` next to the ids file. Backfill gaps with
   ``scrapy crawl matsne -a doc_type=main -a seed_ids_file=<missing_consolidated_ids.txt>``
   (doc_type=main stamps the seeded docs consolidated by provenance), embed, then
   re-run this script to confirm coverage.

Completeness guard: a clean doc_type=main crawl writes ``main_listed_ids.txt.complete``
(the id count). Without a matching sentinel the ids file is treated as a PARTIAL
enumeration: the true-only pass still runs (safe), but ``--false-for-unlisted`` is
refused unless ``--allow-partial`` is given — writing false from a partial universe
would mislabel every main doc the enumeration missed.

DURABILITY: flags set here are payload-only and are OVERWRITTEN by any later re-embed
of the same doc (build_payload stamps the item-derived flag; default doc_type=all items
carry only the switcher heuristic, and the snapshot loader drops consolidation
entirely). After any matsne re-embed/merge, re-run this script.

Take ``coordination/locks/qdrant-write.lock`` before running (index build + payload
writes briefly yellow the collection).

Usage (from ingest/):
  .venv/bin/python scripts/reconcile_consolidated.py --run <run_id> [--dry-run]
  .venv/bin/python scripts/reconcile_consolidated.py --ids-file <path> [--batch 300]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ root → `import ingest`

from qdrant_client import models  # noqa: E402

from ingest.config import load_config  # noqa: E402
from ingest.operational import refuse_legacy_operation  # noqa: E402

SIDECAR_NAME = "main_listed_ids.txt"


def resolve_ids_file(cfg, run: str | None, ids_file: str | None) -> Path:
    """Explicit --ids-file wins; else the given run's sidecar; else the newest COMPLETE
    run's sidecar (falling back, with a warning, to the newest partial one)."""
    if ids_file:
        return Path(ids_file)
    runs_dir = cfg.artifacts_root / "matsne" / "runs"
    if run:
        return runs_dir / run / SIDECAR_NAME
    candidates = sorted(runs_dir.glob(f"*/{SIDECAR_NAME}"), key=lambda p: p.parent.name)
    if not candidates:
        raise SystemExit(f"No {SIDECAR_NAME} under {runs_dir} — run a doc_type=main crawl first")
    complete = [p for p in candidates if sentinel_count(p) is not None]
    if complete:
        skipped = [p for p in candidates if p.parent.name > complete[-1].parent.name]
        for p in skipped:
            print(f"WARNING: skipping newer PARTIAL enumeration {p} (no .complete sentinel)")
        return complete[-1]
    print(f"WARNING: no run has a {SIDECAR_NAME}.complete sentinel — using newest (PARTIAL?)")
    return candidates[-1]


def sentinel_count(ids_path: Path) -> int | None:
    """The id count from ``<ids file>.complete``, or None if the sentinel is absent/bad."""
    sentinel = ids_path.with_name(ids_path.name + ".complete")
    if not sentinel.exists():
        return None
    try:
        return int(sentinel.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def load_listed_ids(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"ids file not found: {path}")
    seen: set[str] = set()
    ids: list[str] = []
    for token in path.read_text(encoding="utf-8").split():
        token = token.strip()
        if token and token not in seen:
            seen.add(token)
            ids.append(token)
    return ids


def ensure_payload_indexes(client, collection: str) -> None:
    """Create the consolidation payload indexes if missing (idempotent, additive)."""
    schema = client.get_collection(collection).payload_schema or {}
    for field_name, field_schema in (
        ("is_consolidated", models.PayloadSchemaType.BOOL),
        ("consolidated_count", models.PayloadSchemaType.INTEGER),
    ):
        if field_name in schema:
            print(f"  payload index {field_name}: already present")
            continue
        client.create_payload_index(
            collection, field_name=field_name, field_schema=field_schema, wait=True
        )
        print(f"  created payload index: {field_name}")


def indexed_matsne_doc_ids(client, collection: str) -> dict[str, bool]:
    """One payload-only scroll over matsne points → {document_id: any chunk already true}.

    The "already true" bit guards the optional false pass: a doc whose flag is already
    true (switcher-derived or previously reconciled) is certainly a main document, so it
    must never be flipped false just because a listing enumeration had a gap.
    """
    doc_ids: dict[str, bool] = {}
    flt = models.Filter(
        must=[models.FieldCondition(key="source", match=models.MatchValue(value="matsne"))]
    )
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            with_payload=["document_id", "is_consolidated"],
            with_vectors=False,
            limit=10_000,
            offset=offset,
        )
        for pt in points:
            payload = pt.payload or {}
            doc_id = payload.get("document_id")
            if doc_id:
                doc_ids[doc_id] = doc_ids.get(doc_id, False) or payload.get("is_consolidated") is True
        if offset is None:
            return doc_ids


def count_consolidated(client, collection: str) -> int:
    return client.count(
        collection_name=collection,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(key="source", match=models.MatchValue(value="matsne")),
                models.FieldCondition(key="is_consolidated", match=models.MatchValue(value=True)),
            ]
        ),
        exact=True,
    ).count


def set_consolidated(
    client, collection: str, ids: list[str], value: bool, batch: int, sleep: float
) -> int:
    """Batched payload-only writes; every chunk of every listed doc gets the flag."""
    written = 0
    for i in range(0, len(ids), batch):
        chunk = ids[i : i + batch]
        client.set_payload(
            collection_name=collection,
            payload={"is_consolidated": value},
            points=models.Filter(
                must=[
                    models.FieldCondition(key="source", match=models.MatchValue(value="matsne")),
                    models.FieldCondition(key="document_id", match=models.MatchAny(any=chunk)),
                ]
            ),
            wait=True,
        )
        written += len(chunk)
        if written % 3000 < batch:
            print(f"  set_payload({value}) progress: {written}/{len(ids)} docs")
        time.sleep(sleep)  # throttle so a long run can't overwhelm Qdrant
    return written


def main() -> None:
    refuse_legacy_operation("in-place consolidation payload reconciliation")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", help="matsne run id whose main_listed_ids.txt to use (default: newest)")
    ap.add_argument("--ids-file", help="explicit path to a listed-ids file (overrides --run)")
    ap.add_argument("--batch", type=int, default=300,
                    help="doc ids per set_payload call (smaller = gentler on Qdrant)")
    ap.add_argument("--sleep", type=float, default=0.3,
                    help="seconds between set_payload calls (throttle Qdrant load)")
    ap.add_argument("--collection", help="override collection name")
    ap.add_argument("--false-for-unlisted", action="store_true",
                    help="also set is_consolidated=false on indexed matsne docs NOT in the "
                    "listed universe (amendment/informational acts), making the boolean "
                    "authoritative for every matsne doc — heavier write pass; refused on a "
                    "partial (sentinel-less) enumeration unless --allow-partial")
    ap.add_argument("--allow-partial", action="store_true",
                    help="permit --false-for-unlisted even when the ids file has no "
                    "matching .complete sentinel (DANGEROUS: a partial enumeration "
                    "mislabels every main doc it missed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report coverage + write the missing-ids file, but no Qdrant writes")
    args = ap.parse_args()

    cfg = load_config()
    collection = args.collection or cfg.collection_name
    ids_path = resolve_ids_file(cfg, args.run, args.ids_file)
    listed = load_listed_ids(ids_path)
    expected = sentinel_count(ids_path)
    is_complete = expected is not None and expected == len(listed)
    print(
        f"listed main (consolidated) ids: {len(listed)}  ({ids_path}) — "
        + ("COMPLETE enumeration" if is_complete else "PARTIAL/unverified enumeration")
    )
    if not is_complete:
        print(
            "WARNING: no matching .complete sentinel (crawl aborted, mid-flight, or count "
            "drifted). The true-only pass is safe; coverage numbers are a lower bound."
        )
        if args.false_for_unlisted and not args.allow_partial:
            raise SystemExit(
                "--false-for-unlisted refused on a partial enumeration: it would mislabel "
                "every main doc the crawl missed. Re-run the doc_type=main crawl to "
                "completion, pass --ids-file for a known-complete file, or override with "
                "--allow-partial."
            )

    from ingest.qdrant_store import make_client

    client = make_client(cfg)

    if not args.dry_run:
        ensure_payload_indexes(client, collection)

    print("scrolling indexed matsne document_ids (payload-only, ~minutes) ...")
    indexed = indexed_matsne_doc_ids(client, collection)
    present = [doc_id for doc_id in listed if doc_id in indexed]
    missing = [doc_id for doc_id in listed if doc_id not in indexed]

    missing_path = ids_path.parent / "missing_consolidated_ids.txt"
    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    print(
        f"coverage: {len(present)}/{len(listed)} listed docs in `{collection}`; "
        f"{len(missing)} missing → {missing_path}"
    )

    if args.dry_run:
        print("dry-run: no payload writes.")
        return

    before = count_consolidated(client, collection)
    written = set_consolidated(client, collection, present, True, args.batch, args.sleep)
    n_false = 0
    if args.false_for_unlisted:
        listed_set = set(listed)
        # Never flip true→false: an already-true doc is certainly main, even if the
        # listing enumeration missed it (dropped page, mid-crawl publication).
        unlisted = sorted(
            doc_id
            for doc_id, already_true in indexed.items()
            if doc_id not in listed_set and not already_true
        )
        n_false = set_consolidated(client, collection, unlisted, False, args.batch, args.sleep)
    after = count_consolidated(client, collection)
    print(
        f"reconcile done: is_consolidated=true set on {written} docs"
        + (f", false on {n_false} unlisted docs" if args.false_for_unlisted else "")
        + f"; consolidated matsne points {before} → {after}."
    )


if __name__ == "__main__":
    main()
