#!/usr/bin/env python
"""Add a batch's gold docs to holdout_doc_ids_v2.json (idempotent; v1 file untouched)."""
import json
import sys
from pathlib import Path

EVAL = Path("/home/bezhomatiashvili/Desktop/Projects/Georgia-Legal-Search/ingest/eval")
batch = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip() and not x.startswith("#")]

path = EVAL / "holdout_doc_ids_v2.json"
entries = json.loads(path.read_text(encoding="utf-8"))
have = {(e["source"], e["document_id"]) for e in entries}
added = 0
for r in batch:
    key = (r["gold"]["source"], r["gold"]["document_id"])
    if key not in have:
        entries.append({"source": key[0], "document_id": key[1]})
        have.add(key)
        added += 1
path.write_text(json.dumps(entries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"holdout_v2: +{added} → {len(entries)} entries")
