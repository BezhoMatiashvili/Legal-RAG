#!/usr/bin/env python
"""Prepare authoring inputs: per-doc bundle files + the ordered pair plan (I5 Phase 3).

- scratchpad/docs/{source}__{id}.body.txt   — EXACT body_markdown bytes (quote source)
- scratchpad/docs/{source}__{id}.meta.json  — metadata the authors cite (number, dates…)
- scratchpad/pair_plan.jsonl                — one row per planned pair: qid, type, doc

Tops up the 5 pairs lost to dropped docs (keyword/natural/paraphrase/2×temporal) on
suitable spare candidates. qids run q104.. in plan order; batches slice this plan.
"""
import json
import random
from collections import Counter
from pathlib import Path

S = Path("/tmp/claude-1000/-home-bezhomatiashvili-Desktop-Projects-Georgia-Legal-Search/93fba822-bae2-4bae-ad6d-e30cf8b61e38/scratchpad")
INGEST = Path("/home/bezhomatiashvili/Desktop/Projects/Georgia-Legal-Search/ingest")
TARGET = {"legal_citation": 83, "cross_lingual": 78, "natural_question": 68,
          "keyword": 59, "paraphrase": 49, "temporal": 60}

cands = [json.loads(x) for x in open(S / "candidates.jsonl", encoding="utf-8")]
rng = random.Random(20260710)

# --- top up planned pairs to the full 397 -------------------------------------------
have = Counter(t for d in cands for t in d["planned_types"])
for qtype, want in TARGET.items():
    deficit = want - have[qtype]
    if deficit <= 0:
        continue
    def ok(d):
        if qtype in d["planned_types"] or len(d["planned_types"]) >= 2:
            return False
        if qtype == "temporal":
            return bool(d.get("status") or d.get("expiry_date") or d.get("in_force_date"))
        if qtype == "legal_citation":
            return bool(d.get("document_number") or d.get("registration_code"))
        return True
    pool = [d for d in cands if ok(d)]
    rng.shuffle(pool)
    for d in pool[:deficit]:
        d["planned_types"].append(qtype)
        have[qtype] += 1

# --- load bodies for every doc that has planned pairs -------------------------------
used = [d for d in cands if d["planned_types"]]
by_source: dict[str, set] = {}
for d in used:
    by_source.setdefault(d["source"], set()).add(d["document_id"])

records: dict[tuple, dict] = {}
for root in (INGEST / "snapshots" / "v1" / "docs", INGEST / "snapshots" / "v2-delta" / "docs"):
    for source, want in by_source.items():
        p = root / f"{source}.jsonl"
        if not p.exists():
            continue
        remaining = {w for w in want if (source, w) not in records}
        if not remaining:
            continue
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["document_id"] in remaining:
                records[(source, r["document_id"])] = r
                remaining.discard(r["document_id"])
                if not remaining:
                    break

missing = [(d["source"], d["document_id"]) for d in used
           if (d["source"], d["document_id"]) not in records]
assert not missing, f"bodies missing: {missing[:5]}"

docs_dir = S / "docs"
docs_dir.mkdir(exist_ok=True)
for key, r in records.items():
    stem = f"{key[0]}__{key[1].replace('/', '_')}"
    (docs_dir / f"{stem}.body.txt").write_text(r["body_markdown"], encoding="utf-8")
    meta = {k: r.get(k) for k in ("source", "document_id", "title", "date", "date_raw",
                                  "document_type", "document_number", "registration_code",
                                  "status", "status_raw", "in_force_date", "expiry_date",
                                  "court", "parties", "source_url", "body_char_len")}
    (docs_dir / f"{stem}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

# --- ordered pair plan ----------------------------------------------------------------
plan = []
for d in used:
    for t in d["planned_types"]:
        plan.append({"query_type": t, "source": d["source"], "document_id": d["document_id"],
                     "stratum": d.get("stratum"), "title": records[(d["source"], d["document_id"])].get("title"),
                     "body_char_len": records[(d["source"], d["document_id"])].get("body_char_len")})
rng.shuffle(plan)
for i, row in enumerate(plan):
    row["qid"] = f"q{104 + i}"
with open(S / "pair_plan.jsonl", "w", encoding="utf-8") as f:
    for row in plan:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"pairs planned: {len(plan)} over {len(used)} docs; bundles: {len(records)}")
print("per type:", dict(sorted(Counter(r['query_type'] for r in plan).items())))
print("per source:", dict(sorted(Counter(r['source'] for r in plan).items())))
