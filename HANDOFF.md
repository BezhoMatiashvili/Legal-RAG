# Georgian Legal RAG — Session Handoff (paste this into the next session)

You are continuing a **phased build** of a Georgian legal question-answering / retrieval
system (~184k Georgian legal docs → hybrid search + grounded cited answers via an MCP
server). Most of the retrieval stack already existed; we extend and harden it.

**Read first, in order:** `prompt.md` (the full phased spec — "Project reality", "Global
guardrails", "RunPod runbook" apply throughout; **you are on Part 3, Phase C**), then your
persistent memory (`MEMORY.md` + the `memory/` files load automatically; the
`georgian-legal-rag-state.md` one has the blow-by-blow current state). `ingest/docs/gap-audit.md`
is a Part-0 artifact — **its corpus counts are stale** (see corrections below).

---

## Where we are  (TL;DR)

**Parts 0–2: DONE. Part 3: Phase A (GPU embed) DONE ✅ · Phase B (code) DONE ✅ · Phase C
(measure) DONE ✅ · Phase D (report) DONE ✅ → `ingest/eval/phase_c_report.md`.**

**Phase C/D headline (2026-07-09):** real full-corpus baseline done. Neural stack beats the BM25
floor on all point estimates (large margins; per-mode CIs overlap + no paired rerank-vs-baseline test
run → significance untested, recommended follow-up) — **hybrid+rerank@80 nDCG@10 0.289 vs bm25 0.192
(+50%)**, R@10 0.388 vs 0.243;
reranking is the biggest lever (hybrid nDCG 0.182→0.289) and the ONLY thing that handles
cross-lingual (22 EN→KA pairs: bm25/sparse **0.000**, hybrid 0.136, **rerank@80 R@10 0.273**; deeper
rerank helps XL most). Routing = **TIE** (paired, p≥0.11). Diversity: cap = tie, **MMR strongly
harmful** (0.289→0.074) → off. **Recommended interim prod config: `hybrid+rerank@50, no diversity`,
`retrieval_fingerprint=81c807b279399098`.** Serving caveat: CPU rerank ~25s/q@50 — reranker needs
int8/ONNX or GPU accel for interactive use. Rerank quality measured on a **RunPod 4090 over an SSH
tunnel** (new `RemoteBGEReranker` + `scripts/runpod_rerank*.py`; retrieval local; ~11h→40min, ~$1.20,
pod terminated). BM25 floor = new `eval/bm25_full.FullCorpusBM25` (full-corpus, exact-parity, cached).
**222 tests pass, ruff clean. Do NOT commit unless asked.**

**⚠ DEFERRED (user decision):** the Qdrant recall-tuning sweep (fusion/prefetch/ef×rescore/
sparse-weight) was NOT run — a concurrent **user job `scripts/backfill_consolidation.py`** began
re-indexing `georgian_legal` (status→yellow), timing out hybrid queries. Core results predate it
(safe). Re-run on a green index: **`bash scripts/phase_c_full.sh tuning`**.

---

## 🧾 2026-07-09/10 late session — full-corpus embed coverage, verified

The July crawls grew the scraped universe to **208,218 docs** (matsne 156,676 · napr
25,214 · ecd 21,827 · tas 1,344 · constcourt 3,076 · tbappeal 81). Coverage ground truth
is the new **`ingest/scripts/verify_all_embedded.py`**: derives ids exactly like ingest
(`SourceSpec.id_fields` — ecd = `decision_document_id`, NOT its raw `document_id` field),
one payload-only scroll (~2.5 min), classifies missing into true-missing vs `no_text`
(empty body, zero chunks) vs `quarantined` (`snapshots/v1/quarantine.jsonl`, 257 docs
excluded by hygiene by design), writes `ingest/.state/embed_coverage.json` (+`.partial`
for `--sources` runs — never clobbers the global report). The session monitor (:8770,
restart after .py edits) has a new **"Embedding coverage → Qdrant" panel** rendering it:
live points/status, per-source bars, batch-pipeline stage.

Gap-closing done this session: 2 never-scraped matsne ids (23230, 24178) seed-fetched
(`-s DEDUP_ENABLED=False` — seen.sqlite blocks re-fetch otherwise) + CPU-embedded (550
chunks); 5 near-empty constcourt docs embedded; `embed_delta.py` gained `--source`.
Batch-2 (5,738 matsne docs): first pod embed SUCCEEDED but the snapshot pull died with
the pod (~$0.89 lost — see memory [[delta-embed-pull-hazard]]; pull timeout now 1800 s,
was 4 h); the recovery re-run (orchestrate → merge → verify) **COMPLETED 2026-07-09
20:33Z, $0.26**: 37,815 points merged → **`georgian_legal` = 2,637,645 points**, and the
final verify exited 0 — **0 missing anywhere: 207,940/208,218 scraped docs embedded, 97
empty-body + 181 quarantined excluded by design** (constcourt 3,074/3,076 · ecd
21,827/21,827 · matsne 156,404/156,676 · napr 25,210/25,214 · tas 1,344/1,344 · tbappeal
81/81). Pod terminated, pods=[], balance $10.53.

## ▶ NEXT SESSION — start here (Part 3 is essentially COMPLETE)

**Read the report first:** `ingest/eval/phase_c_report.md` (+ regenerable tables
`ingest/eval/phase_c_report_tables.md`). It has the full baseline, the recommended production
config (`hybrid+rerank@50`, `retrieval_fingerprint=81c807b279399098`), and honest caveats. The
old "IMMEDIATE NEXT — Phase C" / "Phase D" sections below are **historical** — that plan was executed.

**Nothing is required next.** Open options, in rough priority:
1. **(optional) Finish the deferred tuning grid** once your `backfill_consolidation.py` finishes and
   `curl -s localhost:6333/collections/georgian_legal | grep status` shows **green**:
   `bash scripts/phase_c_full.sh tuning` → then regenerate tables with
   `scripts/build_phase_c_report.py --gpu-log eval/experiments_gpu.jsonl --sweep-log <tier1 log>`.
   Secondary knobs; won't change the recommendation.
2. **(optional) Harden the significance claims** — the rerank-vs-hybrid win is by point estimate
   (CIs overlap, n=103); a paired permutation test (`--compare hybrid rerank`, but rerank on CPU is
   slow/OOM — do it via the GPU rerank path or an accelerated reranker) would certify it. And a
   clean per-depth CPU rerank-latency measurement (`scripts/rerank_latency_probe.py`, run with
   `OMP_NUM_THREADS=8`) would replace the derived latencies.
3. **(optional) Make rerank practical for serving** — CPU rerank@50 is ~25 s/query; needs int8/ONNX
   or the on-demand GPU rerank path. See memory **[[serving-latency-and-gpu-rerank]]**.
4. **Commit** — everything is uncommitted (per the standing rule). Nothing has been committed.

**⚠ Critical gotchas for a fresh session (from memory):**
- **Git:** the tree is on branch **`dev` @ `7992ce9`** (our custom offset-aware stack). `origin/dev`
  is the **LangChain fork — do NOT `git pull`/merge it** (clobbers `build_payload`/`is_consolidated`/
  offset chunker). `git reset --hard 7992ce9` recovers. See **[[git-branch-fork-hazard]]**.
- **RAM:** 30 GB box can't co-run Qdrant + a resident reranker + desktop without swap-thrash. The
  harness now loads the reranker **only** for `rerank` modes (`RERANK_ENABLED=false` else) — baked
  into `scripts/phase_c_full.sh`. See **[[eval-ram-reranker-swap]]**.
- **GPU rerank is fragile** — it once orphaned a billing pod; **always** `python scripts/runpod_rerank.py down`
  after use and check `myself{pods}` is empty. See **[[serving-latency-and-gpu-rerank]]**.
- **Concurrent writers:** if `backfill_consolidation.py` (or any ingest) is running, the index is
  yellow and eval queries time out — wait for green.

**New this session (all under `ingest/`, tested — 222 pass, ruff clean):**
`eval/bm25_full.py` (full-corpus BM25, cached `eval/.bm25_full/`) · `ingest/rerank.py`
`RemoteBGEReranker` + CPU thread-pin/length-bucket · `scripts/runpod_rerank.py` +
`runpod_rerank_server.py` (GPU rerank over SSH tunnel) · `scripts/phase_c_full.sh`
(`{tier1|gpu_rerank|diversity|tuning}`) · `scripts/build_phase_c_report.py` ·
`scripts/rerank_latency_probe.py` · `scripts/sample_ingest_report.py` · additive `knobs` field on
eval rows. GPU rows → `eval/experiments_gpu.jsonl` (quality only; not CPU latency).

---

**🎉 THE CORPUS IS EMBEDDED, VERIFIED, AND PERMANENT.** The full corpus is embedded into local
Qdrant collection **`georgian_legal` = 2,453,915 points, status GREEN** (HNSW indexed), persisted
on disk via the docker bind-mount (`ingest/qdrant_storage/`). It never needs re-embedding unless
the model/chunking changes. Retrieval is **proven working end-to-end** (see "Validation" below).
**The RunPod pod is terminated (pods=[]); balance $4.30.** Nothing is billing.

Per-source point counts (sum = 2,453,915): matsne 1,855,148 · ecd 383,452 · napr 141,427 ·
constcourt 71,292 · tas 2,400 · tbappeal 196.

---

## ⚑ Standing decisions (unchanged — still binding)

- **Architecture pivot (2026-07-08):** the product is an **MCP server whose client is Claude**;
  **Claude composes the grounded, cited answers → NO local generation LLM. Part 4 is cut.**
  GPU/RunPod was needed for exactly ONE job — the Part 3 corpus embed — which is now DONE.
  Retrieval quality is the whole product. Privacy caveat the user accepted: retrieved text (incl.
  PII) is sent to Claude at answer time; ingest/embed/eval stay local (no external APIs there).
  See `memory/mcp-first-claude-composes.md`.
- **Git:** stay on **our custom offset-aware stack**. `origin/dev` (`877462c`) is a divergent
  LangChain fork — **do NOT merge / do NOT `git pull`** (its chunker drops the char offsets the
  span-anchored eval needs). Our work is **uncommitted on local `dev`**.
- **If a test fails, fix the code, never the test.** Never edit existing tests. If data/an
  assumption looks wrong, **tell the user** — don't silently work around it.
- **Proof, not vibes:** every retrieval change must show a statistically-meaningful win (bootstrap
  CIs + paired significance), ideally vs a BM25 baseline. Append every run to
  `ingest/eval/experiments.jsonl`.
- **Privacy:** NO external APIs ever (local CPU only; the one-time GPU job is done). **PII stays
  IN the index** — never redact the corpus. `content_hash` is additive metadata only.
- **Do NOT commit** anything unless the user explicitly asks. Everything is uncommitted on `dev`.
- **Run through Part 3 to its STOP gate** (the user chose no mid-part stop, no checkpoint commit).

---

## Critical facts & corrections

1. **True corpus = 184,575 unique docs**; clean snapshot v1 = **184,318 docs** under
   `ingest/snapshots/v1/docs/*.jsonl`. Embedding it produced **2,453,915 chunks** (~13.3 chunks/doc;
   matsne is chunk-dense).
2. **Golden set = 103 span-anchored pairs / 51 holdout docs** (`ingest/eval/golden_set_v1.jsonl`
   — 106 lines incl. header, 103 pairs; `holdout_doc_ids.json`). Types: natural_question 32,
   cross_lingual 22 (EN→KA), keyword 21, legal_citation 17, paraphrase 11; lang ka 81 / en 22.
   The 22 cross_lingual / en pairs are the load-bearing slice for the routing experiment.
3. **This box is CPU-only** (Intel Arc, no NVIDIA); `torch` pinned to pytorch-cpu. Serving/eval
   run on CPU → model inference is slow-ish (BGE-M3 encode ~1–2 s/query; cross-encoder rerank of
   80 candidates is the per-query bottleneck). Plan Phase C runtimes accordingly.
4. **RunPod:** restricted key in `ingest/.env` as `RUNPOD_API_KEY` (git-ignored, never print).
   **Balance $4.30** (started $15; ~$10.70 spent — most of the overage was the flaky 24 GB
   snapshot download + false-alarm retries, not the embed itself). Cloudflare needs a browser
   `User-Agent` header on every GraphQL call. **No pods running.**
5. **G2 "one vector space" PASSED:** cosine(CPU-fp32 ref, GPU-fp16) = **0.999999**. CPU ref =
   `ingest/snapshots/v1/checksum_cpu.json`; GPU ref = `~/gpu_embed_work/out_multi/checksum_gpu.json`.
6. **Snapshot backup on disk:** the 24 GB Qdrant snapshot is at
   `~/gpu_embed_work/out_multi/georgian_legal.snapshot` (md5 `378985d49eac43f681e90a4fbdeb47ea`,
   24,166,869,504 B) + its 58 parts in `.../parts/`. Re-restore anytime with
   `curl -X POST 'http://localhost:6333/collections/georgian_legal/snapshots/upload?priority=snapshot' -F snapshot=@<file>`.

---

## Validation — retrieval works (proven this session)

Direct CLI (bypasses the MCP server): `cd ingest && .venv/bin/python -m ingest search "<query>" --top-k 5`.
A **cross-lingual** test — English query *"annual paid leave duration for employees"* — correctly
returned **საქართველოს შრომის კოდექსი (Labor Code) Art. 31 (24 working days/yr, in_force)** plus
related matsne acts and ecd court decisions, with calibrated cross-encoder scores (0.79–0.83).
Cross-lingual retrieval + hybrid + rerank + multi-source + rich metadata all confirmed working.

**⚠ Stale MCP server gotcha (this session only):** the MCP tool `legal_search` errored with
`Invalid device string: '# blank = follow EMBED_DEVICE...'`. The **config code is CORRECT** —
verified `cfg.rerank_device is None` and `_device_opt()` strips inline `.env` comments. The error
was from the long-running MCP-server *process* holding stale state. A **fresh session's MCP server
should be fine**; if it recurs, reconnect via `/mcp` (no code change needed). The direct `ingest
search` CLI always works (uses current config).

---

## Part 3 Phase B (DONE) — code landed, 204 tests pass, ruff clean

All CPU-validated, unit-tested; no re-embed needed. Key additions:
- **Watch hardening:** `content_hash` in payload (`qdrant_store.build_payload`, additive — PII stays
  plaintext) + `KEYWORD_FIELDS`; O(1) `_unchanged` skip-guard at both embed sites; daily ingestion
  report JSON under `.state/reports/`; JSON schema-drift warnings (`sources.py`).
- **Query logging:** `ingest/ingest/querylog.py` (`build_query_record` — no chunk text — +
  `append_query_log` to `.state/queries.jsonl`, local-only) hooked into `mcp_server.legal_search`.
- **`config.retrieval_fingerprint(cfg)`** (16-hex over collection/model/rerank/chunk knobs).
- **MCP additions** (`mcp_server.py`): `ingest_status`, `legal_get_document_versions`
  (registration_code lineage), `legal_health`, overlap-aware `_stitch_overlap` in
  `legal_get_document`, `char_start/char_end` in `_hit_dict`, fingerprint stamp + query logging.
- **Harness knobs** (`eval/backend.py`, `eval/evaluate.py`): `routed` mode (drops sparse for EN);
  faithful rerank depth (`candidate = max(rerank_candidates, k)`, rc=80 default unchanged);
  `diversify()` (per-doc cap + MMR) in `search.py`; CLI flags `--rerank-candidates --fusion
  {rrf,dbsf} --prefetch-limit --hnsw-ef --rescore --sparse-weight --max-per-doc --mmr-lambda
  --ab --compare`, all folded into `config_hash`.

---

## ~~▶ IMMEDIATE NEXT — Phase C~~ (HISTORICAL — this plan was executed; see "NEXT SESSION" above)

The index is **green** and ready. A runner script is already written:
**`ingest/scripts/phase_c.sh`** (runs dense → sparse → hybrid → rerank, then `--compare hybrid
routed`, each `--log`-ged with CIs). Launch it in the background and report the tables:
```
cd /home/bezhomatiashvili/Desktop/Projects/Georgia-Legal-Search/ingest
nohup bash scripts/phase_c.sh > /home/bezhomatiashvili/gpu_embed_work/phase_c.log 2>&1 &
tail -f /home/bezhomatiashvili/gpu_embed_work/phase_c.log
```
Or run modes individually: `.venv/bin/python -m eval.evaluate --backend qdrant --mode <m>
--relevance chunk --log`  (modes: dense sparse hybrid rerank routed).

Then the remaining Phase C experiments (all via the harness, CIs, no re-embed):
1. **Routing lift:** `--compare hybrid routed` — read the `per_query_type["cross_lingual"]` /
   `per_language["en"]` slice (the 22 EN pairs); overall must not regress KA.
2. **Rerank-depth ablation:** `--mode rerank --rerank-candidates {10,30,50,80}` — pick the
   shallowest depth whose nDCG@10 CI matches the best, vs its latency cost.
3. **Diversity:** `--mode rerank --max-per-doc 3 --ab` and `--mmr-lambda 0.5 --ab`.
4. **CPU tuning:** `--fusion dbsf --ab`, `--prefetch-limit {50,100,200,400}`, `--hnsw-ef
   {64,128,256}` × `--rescore {on,off}` recall/latency knee, `--sparse-weight` on the EN slice.

**⚠ BM25 baseline caveat (must handle):** the harness `QdrantBackend._ensure_bm25()` scrolls the
**entire 2.45M-chunk collection** into an in-memory TF-per-chunk list and **linear-scans all of it
per query** → impractical here (RAM + ~minutes/query). Do **NOT** run `--mode all` or `--mode bm25`
as-is. For the BM25 comparison row, either (a) build BM25 over only the 51 holdout docs' chunks +
distractors (scoped, like the FakeBackend), (b) use Qdrant native full-text (`MatchText` on a
text index) as the lexical baseline, or (c) report neural-vs-neural and note BM25 was out of scope
at full-corpus scale. Pick one and `log()` the choice. (Fix code, not tests, if you extend it.)

## ~~Phase D — STOP-gate report~~ (HISTORICAL — DONE; report at `ingest/eval/phase_c_report.md`)

Write the report: harness comparison tables (modes + routing + rerank depths + diversity + Qdrant
recall settings, each with CIs/latency); the GPU embed completion (2,453,915 chunks, G2 cosine
0.999999, pod-hours, $10.70 spent / $4.30 left); **one recommended interim production config**
(with its config hash); a sample daily-ingestion report. Update this HANDOFF.md + `memory/` with
the real numbers. Confirm `pytest` (204) + ruff clean. **Do NOT commit** unless asked.

---

## Session artifacts & handy commands (from `ingest/`)

New scripts this session (under `ingest/scripts/`): `finish_load.sh` (resume snapshot download +
reassemble + md5 + restore — re-runnable), `phase_c.sh` (baseline eval), `runpod_embed_multi.sh`
(multi-GPU on-pod embed, `SHARDS=N`), `runpod_orchestrate_multi.py` (multi-GPU orchestrator),
`monitor_server.py` (localhost:8765 dashboard). Re-embed recipe (only if model/chunking changes):
8× single pod is the sweet spot — `SHARDS=8` on an 8×4090 pod, ~40–45 min, ~$4.

- Tests: `.venv/bin/python -m pytest tests -q`  (204 pass)
- Direct search: `.venv/bin/python -m ingest search "<q>" --top-k 5`
- Collection check: `curl -s localhost:6333/collections/georgian_legal | python3 -m json.tool | grep -E 'points_count|status'`
- Harness (synthetic smoke): `.venv/bin/python -m eval.evaluate --backend fake --mode all --log`
- Qdrant: `docker compose up -d` (127.0.0.1:6333; data persists in `ingest/qdrant_storage/`)

<!-- UPDATE THIS FILE as Phase C/D progress: record the real baseline numbers, the chosen BM25
approach, the recommended production config hash, and remaining budget. -->
