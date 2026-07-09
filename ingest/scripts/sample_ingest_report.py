#!/usr/bin/env python
"""Emit a *sample* daily-ingestion report via the real write_ingest_report() code path.

The report schema (per-source docs/added/updated/unchanged/skipped/chunks + schema-drift +
totals) is produced by ingest.pipeline.write_ingest_report during `ingest watch`. The corpus
is currently static (a real watch today would show all-unchanged), so to demonstrate the
format with the delta/drift fields populated we build an illustrative `states` dict: real
per-source chunk totals pulled from the live Qdrant collection, plus plausible daily deltas
(a handful of added/updated docs) and one schema-drift example. Written with kind="sample"
so it is never mistaken for real telemetry. Output: .state/reports/ingest-<date>.json
"""

from __future__ import annotations

from qdrant_client import QdrantClient, models

from ingest.config import load_config
from ingest.pipeline import write_ingest_report

# Per-source document counts of clean snapshot v1 (the union of all runs/*; see memory).
DOC_COUNTS = {
    "matsne": 133033, "napr": 25214, "ecd": 21827,
    "constcourt": 3076, "tas": 1344, "tbappeal": 81,
}
# Plausible deltas for one day's watch (added, updated); the rest are unchanged.
DAILY_DELTA = {
    "matsne": (9, 24), "napr": (2, 1), "ecd": (3, 5),
    "constcourt": (1, 0), "tas": (0, 2), "tbappeal": (0, 0),
}
AVG_CHUNKS_PER_DOC = 13.9  # 2,453,915 chunks / 176,575-ish docs


def main() -> int:
    cfg = load_config()
    client = QdrantClient(url=cfg.qdrant_url, api_key=cfg.qdrant_api_key)
    states: dict = {}
    for src, docs in DOC_COUNTS.items():
        chunks_total = client.count(
            cfg.collection_name,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="source", match=models.MatchValue(value=src))]),
            exact=True,
        ).count
        added, updated = DAILY_DELTA[src]
        st = {
            "docs": docs,
            "added": added,
            "updated": updated,
            "unchanged": docs - added - updated,
            "skipped": 0,
            # chunks re-embedded this run = only those of the added/updated docs
            "chunks": round((added + updated) * AVG_CHUNKS_PER_DOC),
            "updated_at": "2026-07-09T00:00:00+00:00",
            "_note": f"illustrative; {chunks_total} chunks indexed for {src}",
        }
        if src == "tas":  # demonstrate schema-drift detection
            st["schema_drift"] = {"appeal_outcome": updated}
        states[src] = st
    path = write_ingest_report(cfg, states, kind="sample")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
