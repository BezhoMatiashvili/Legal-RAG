# I6 post-scrape runbook — complete the corpus, re-baseline, launch the v2 re-embed

Execute in order once the user confirms the scrape is finished. All commands from `ingest/`.
The I6 pipeline itself (v2 headers, payload re-embed, autonomous orchestrator) is built and
smoke-validated — this runbook is only the sequencing around it.

## 1. Preconditions
```bash
docker compose up -d   # if not running
curl -s -H "api-key: $QDRANT_API_KEY" localhost:6333/collections/georgian_legal | grep -E 'points_count|status'
# require status green; record points_count
```

## 2. Embed the newly-scraped docs into the LIVE collection (v1 headers — normal pipeline)
```bash
.venv/bin/python scripts/verify_all_embedded.py       # enumerate missing per source (~6 min)
```
- **matsne runs, large delta (>~2k docs):** GPU delta (proven path)
  `.venv/bin/python scripts/runpod_orchestrate_delta.py --runs-since <run-id>` →
  `.venv/bin/python scripts/merge_delta_collection.py --dry-run` → `--src georgian_legal_delta`
- **small deltas / other sources:** per-source local embed, e.g.
  `.venv/bin/python scripts/embed_delta.py --items <items.jsonl> --collection georgian_legal`
  (CPU; fine for hundreds of docs) or the `watch --once` route used by `scripts/daily_ingest.sh`.
- Re-run `verify_all_embedded.py` until **0 missing** (only no_text/quarantined excluded).

## 3. Re-baseline v1 at the new corpus state (references for the I6 gate)
The corpus changed → all previous rows expire (improvement.md §8). Translations stay ON
(production-like). int8 CPU rerank is fine — measured identical to fp32 (I7).
```bash
RERANK_ENABLED=false .venv/bin/python -m eval.evaluate --backend qdrant --mode hybrid \
  --translate-queries eval/query_translations_v1.json --relevance chunk --log            # ~3 min
RERANK_ENABLED=true RERANK_BACKEND=onnx OMP_NUM_THREADS=8 nohup .venv/bin/python -m eval.evaluate \
  --backend qdrant --mode rerank --rerank-candidates 50 \
  --translate-queries eval/query_translations_v1.json --relevance chunk --log \
  > /tmp/rebaseline_rerank.log 2>&1 &                                                    # ~1 h CPU
```

## 4. Update the gate references in the orchestrator
Edit `scripts/runpod_orchestrate_reembed.py`: set `REF_HYBRID` / `REF_RERANK` (overall +
all slice tuples) from the two rows just logged in `eval/experiments.jsonl`. Note the
reference rerank row is now int8; the pod eval uses fp32 GPU — measured identical (I7
parity row: 0.320 = 0.320), note it in the ledger anyway.

## 5. Fresh export (the old one predates the new docs)
```bash
rm -rf .state/reembed_v2/rows
.venv/bin/python scripts/reembed_export.py --out .state/reembed_v2/rows                  # ~25 min
```

## 6. Launch the autonomous I6 run (~$5, budget-capped at $6.50, always terminates)
```bash
nohup .venv/bin/python scripts/runpod_orchestrate_reembed.py > ~/gpu_embed_work/reembed_v2.log 2>&1 &
tail -f ~/gpu_embed_work/reembed_v2.log
# emergency: .venv/bin/python scripts/runpod_orchestrate_reembed.py terminate
```
It will: provision (4090 primary) → push rows → re-embed all points with v2 headers under
the same ids/payloads → eval hybrid + rerank@50 over the SSH tunnel against the pod's
Qdrant → gate (rerank nDCG ≥ ref +0.02, no slice −0.02) → pull + restore the snapshot as
`georgian_legal_v2` ONLY on PASS → terminate + cost report.

## 7. After the verdict
- **PASS:** smoke `georgian_legal_v2` via CLI (`COLLECTION_NAME=georgian_legal_v2 … search`),
  then flip `COLLECTION_NAME` in `.env` (+ `/mcp` reconnect); keep `georgian_legal` as instant
  rollback until confident. Re-calibrate abstention on v2 (`scripts/calibrate_min_score.py`).
  Ledger row + HANDOFF + commit (`EMBED_HEADER_V2=true` also goes into `.env` so future
  delta embeds write v2 headers into the v2 collection!).
- **FAIL:** nothing was restored; ledger row with the numbers; the $ spent bought the answer.
  Delete `~/gpu_embed_work/out_reembed_v2` leftovers.
- Either way: verify `pods=[]`, record COST line, update memory.

---

## STATUS 2026-07-11: I6 SHELVED (corpus complete; blocked on GPU capacity + budget)

Corpus completed on GPU (delta merged, 2,654,818 pts, 0 missing, green). I6 v2-header
re-embed attempted but shelved:
- **4-GPU / 1×4090 capacity unavailable** (RunPod SUPPLY_CONSTRAINT); only A5000 offered
  (~3-4× slower → ~17h, doesn't fit time/budget).
- **Batch-size perf bug FOUND + FIXED**: `reembed_v2.py` used BATCH=64 (starved the 4090s,
  ~90/s on 2×4090); now `BATCH=int(os.getenv("EMBED_BATCH_SIZE") or 256)` → ~256/s expected.
- Balance $4.71 — too tight for a 4090 (~$4.8 for a ~5.8h single-GPU run).

**Everything is staged for a one-command relaunch when 4090 (or 2-4×4090) capacity +
~$6-8 budget align:**
1. Rows already exported: `.state/reembed_v2/rows` (14 files, 2,654,818). Re-export only if
   the corpus changed again: `.venv/bin/python scripts/reembed_export.py --out .state/reembed_v2/rows`.
2. References already computed: `.state/ref_v2.json` (v1-headers @ current corpus, v2 golden
   set 337, rerank@50 nDCG **0.450** → gate needs **≥0.470**). Re-run `scratchpad/ref_v2_driver.sh`
   only if the corpus changed.
3. In `scripts/runpod_orchestrate_reembed.py`: set `GPU_COUNTS=[4,2,1]` (or `[2,1]`),
   `GPU_PREFERENCE` 4090-first, `BUDGET_CEILING` to your funded amount (e.g. 6.5).
4. Launch: `nohup .venv/bin/python scripts/runpod_orchestrate_reembed.py > ~/gpu_embed_work/reembed_v2.log 2>&1 &`
   It embeds all chunks with v2 headers → evals over tunnel on the frozen v2 golden set →
   gate (rerank nDCG ≥ ref+0.02, no slice regression) → restores `georgian_legal_v2` ONLY on PASS.
   Progress bar: `python scripts/reembed_progress.py --total 2654818 --collection georgian_legal_v2 &`
   + monitor :8770.

Better still (per improvement.md I6): run it **piggybacked on the next re-embed** you do for
any other reason, so it's not a standalone spend.
