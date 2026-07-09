# Delta embed runbook — GPU-embed the sweep's new docs and merge into main

The full-corpus embed (Part 3) embedded the whole snapshot on a RunPod GPU, snapshotted the
collection, and **restored** it into local Qdrant (`finish_load.sh` — replaces the target).
The completeness **sweep** adds only ~23k new matsne docs (~300k chunks). CPU here embeds at
~15–33 s/chunk (weeks for the delta), so the delta also goes to GPU — but it must be **merged**
into the existing `georgian_legal`, not replace it.

Two scripts do the delta-specific work (both proven on the firearms sample):
- `scripts/embed_delta.py` — reads RAW scraped `items.jsonl` (keeps `is_consolidated`), normalizes, embeds, upserts into a collection.
- `scripts/merge_delta_collection.py` — upserts every point of a delta collection into main (idempotent; deterministic point ids).

## 0. Prereqs (after the sweep finishes)
- Sweep done: `artifacts/matsne/runs/<new-run-dirs>/items.jsonl` hold the recovered docs.
- Confirm the delta size / that consolidation is present:
  ```
  .venv/bin/python scripts/embed_delta.py --dry-run --runs-since 20260709T080007Z
  # → "delta: N unique docs (... consolidated)"  ; N drives the pod size/time
  ```
  (`--runs-since` = the first of our post-original run ids; it includes the seed, both sweep runs.)

## 1. Provision the pod  (reuse existing orchestrator)
```
.venv/bin/python scripts/runpod_orchestrate_multi.py create      # or the single-GPU runpod_orchestrate.py
```
A single 4090 is plenty for ~300k chunks (~5–6 min). Deliver the `ingest/` tree to the pod as
Part 3 did, **plus** the new run dirs' `items.jsonl` (rsync `artifacts/matsne/runs/<new>/` to
`/workspace/ingest_items/`). The pod already builds the clean BGE-M3 venv (see
`runpod_embed_multi.sh` lines 19–29) — reuse that venv setup.

## 2. Embed the delta on the pod → `georgian_legal_delta`
On the pod (GPU), with the same env as `runpod_embed_multi.sh` (`EMBED_DEVICE=cuda`,
`EMBED_USE_FP16=true`, local Qdrant at `127.0.0.1:6333`):
```
# G2 guardrail: same vector space as CPU queries — must match ingest/snapshots/v1/checksum_cpu.json
python -m ingest embed --checksum        # writes checksum_gpu.json; assert cosine≈1 vs CPU ref
python scripts/embed_delta.py --items /workspace/ingest_items/*/items.jsonl \
       --collection georgian_legal_delta --batch-size 256
```
Then snapshot that collection and copy it out (mirrors `runpod_embed_multi.sh` lines 81–87):
```
SNAP=$(curl -s -X POST http://127.0.0.1:6333/collections/georgian_legal_delta/snapshots | jq -r .result.name)
cp /workspace/qdrant_snapshots/georgian_legal_delta/$SNAP /workspace/out/georgian_legal_delta.snapshot
```

## 3. Transfer back + restore to a LOCAL delta collection
Download the (small, ~1–2 GB) snapshot like `finish_load.sh` does (rsync the file), then
restore it into a **separate** local collection — NOT over `georgian_legal`:
```
curl -X POST 'http://localhost:6333/collections/georgian_legal_delta/snapshots/upload?priority=snapshot' \
     -F snapshot=@<downloaded>/georgian_legal_delta.snapshot
```

## 4. Merge the delta into main  (idempotent upsert)
```
.venv/bin/python scripts/merge_delta_collection.py --src georgian_legal_delta
# → "Merged C points from 'georgian_legal_delta' into 'georgian_legal': before → after (+net new)"
```
Verify main is `points_count = 2,453,915 + delta chunks` and a new doc is retrievable with
`status` + `is_consolidated`. Then drop the temp collection and terminate the pod:
```
curl -X DELETE http://localhost:6333/collections/georgian_legal_delta
.venv/bin/python scripts/runpod_orchestrate_multi.py terminate
```

## Notes
- **Idempotent throughout**: point ids are UUIDv5 over `(source, document_id, chunk_index)`, so
  re-running embed or merge overwrites in place — never duplicates. Safe to retry any step.
- **Consolidation preserved**: `embed_delta.py` normalizes the raw items (not the snapshot), so
  `is_consolidated` / `consolidated_count` land in the payload; the backfill only covered the
  pre-existing docs.
- **Simpler alternative** (no pod Qdrant): extend `embed_delta.py` to dump PointStructs to a JSONL
  file instead of upserting, download that, and upsert locally. The merge is then unnecessary. The
  snapshot path above is used here because it reuses the Part-3 machinery verbatim.
- **Cost**: full corpus was ~$4 on 8×4090/45 min; the delta is ~13% of that on one GPU → well under $1.
