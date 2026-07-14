# Eval & ops (golden set, harness, RunPod orchestration, monitoring)
> Drill-down memory. Not auto-loaded — opened per the pre-modification ritual in INDEX.md. Anchors linted by ingest/scripts/gen_code_map.py --check.

## Overview
This area is the measurement-and-operations half of the repo. `ingest/eval/` is a span-anchored retrieval eval harness that gates every retrieval change against the frozen v1 golden set (103 query→doc pairs over 51 holdout docs) with Recall@5/@10, graded nDCG@10, MRR@10, bootstrap 95% CIs, and paired sign-flip permutation tests; every `--log` run appends one row to the permanent ledger `ingest/eval/experiments.jsonl` (35 rows) keyed by a stable `config_hash`. `ingest/scripts/` is the ops toolbox: the Phase-C sweep drivers, RunPod GPU embed/rerank orchestration with always-terminate guarantees, the delta-embed→merge pipeline, embed-coverage verification (the coverage ground truth), snapshot publishing to the serverless worker, the daily ingest cron, and two loopback dashboards (:8770 session monitor, :8765 embed-pod monitor). `improvement.md` at repo root is the gated improvement runbook (one knob per attempt, gates G1–G5, keep-or-revert, append-only ledger §8, re-baseline on corpus change). Current state (2026-07-10): re-baseline at 2,637,645 points; I1 citation routing REVERTED→BLOCKED with its code surviving ONLY as `.improvements/i1_citation_route_{full,partial}.patch`; I2 EN→KA query translation in flight and uncommitted (`ingest/eval/translations.py` + `ingest/eval/query_translations_v1.json`, last `--ab` row shows hybrid nDCG@10 0.168→0.211 with the translate_queries knob). The harness deliberately imports production retrieval code from `ingest/ingest/` so eval numbers measure exactly what the legal_rag MCP server serves.

## Entrypoints
- `.venv/bin/python -m eval.evaluate` (from `ingest/`) → the harness CLI (`ingest/eval/evaluate.py:main`): `--backend {fake,qdrant} --mode {bm25,dense,sparse,hybrid,rerank,routed,all} --relevance {chunk,doc} --compare A B --ab --log`, plus tuning knobs `--rerank-candidates --fusion --prefetch-limit --hnsw-ef --rescore --sparse-weight --max-per-doc --mmr-lambda --translate-queries PATH`.
- `.venv/bin/python -m eval.bm25_full build` → one-time disk-backed full-corpus BM25 index into `ingest/eval/.bm25_full/` (`ingest/eval/bm25_full.py:FullCorpusBM25.build`); mandatory before `--mode bm25` at 2.6M-chunk scale.
- `bash ingest/scripts/phase_c_full.sh {tier1|diversity|tuning|tier2|gpu_rerank|docs|all}` → the sweep driver; its `run()` wrapper sets `RERANK_ENABLED=true` only when args contain `--mode rerank`, 60-min timeout per run. `ingest/scripts/phase_c.sh` is the older simple baseline loop.
- `.venv/bin/python scripts/runpod_rerank.py up|down` → GPU rerank pod + SSH tunnel at http://localhost:8900 (`ingest/scripts/runpod_rerank.py:up`; `down` is a CLI arg in the `__main__` block, not a function — emergency terminate reading `~/gpu_embed_work/pod.id`).
- `python scripts/session_monitor.py [port]` → read-only loopback dashboard :8770 (`ingest/scripts/session_monitor.py:main`).
- `.venv/bin/python scripts/monitor_server.py` → :8765 read-only embed-pod dashboard over SSH (`ingest/scripts/monitor_server.py:main`).
- `.venv/bin/python scripts/verify_all_embedded.py [--sources a,b]` → embed-coverage ground truth, exit 0/1 (`ingest/scripts/verify_all_embedded.py:main`).
- `.venv/bin/python scripts/verify_delta_embedded.py --runs-since <id>` → per-id chunk-count check for delta runs (`ingest/scripts/verify_delta_embedded.py:main`).
- `.venv/bin/python scripts/verify_matsne_completeness.py` → live-site completeness audits (`ingest/scripts/verify_matsne_completeness.py:main`).
- `.venv/bin/python scripts/validate_supremecourt_partial.py --run-dir <run>` → offline fail-closed admission of the exact graceful four-hour cumulative Supreme Court artifact before dry-run/coverage/GPU work, including independently derived cursors and recursive parent scope/hash/content preservation for resumed runs (`ingest/scripts/validate_supremecourt_partial.py:main`).
- `.venv/bin/python scripts/embed_delta.py --dry-run --strict ...` → exact source-aware document/chunk manifest without Qdrant writes; actual run-scoped writes require `--apply` and `QDRANT_WRITE_APPROVED=1`.
- `.venv/bin/python scripts/runpod_orchestrate_delta.py --source <source> --items ... --run-id ... --apply` → one Secure RTX 4090 source-aware delta with separate Qdrant write/recreate and spend approvals, a repository-wide spend lock, continuous reserve checks, and unconditional confirmed termination (`ingest/scripts/runpod_orchestrate_delta.py:main`).
- `scripts/publish_snapshot.py --upload|--verify --apply --cold-restore-confirmed` → schema-v2 generation transport only; implicit-main `--create` is disabled and upload still requires a proven conditional activator (`ingest/scripts/publish_snapshot.py:main`).
- `bash ingest/scripts/daily_ingest.sh [--dry-run]` → scrape → `python -m ingest watch --once` → verify_all_embedded gate; flock + coordination-lock aware; systemd units in `ingest/systemd/`.
- `python scripts/build_phase_c_report.py [--log ...] [--sweep-log ...]` → experiments.jsonl → markdown tables (`ingest/scripts/build_phase_c_report.py:load_rows`).
- `.venv/bin/python scripts/rerank_latency_probe.py [out.json]` → isolated CPU rerank p50/p95 per depth (`ingest/scripts/rerank_latency_probe.py:main`).
- `.venv/bin/python scripts/backfill_consolidation.py` → LEGACY switcher-derived payload-only `is_consolidated` backfill (false pass now opt-in `--write-false`); superseded as authority by `ingest/scripts/reconcile_consolidated.py:main` (listing-based, 2026-07-10); user-owned per improvement.md (`ingest/scripts/backfill_consolidation.py:main`).

## Modules

### ingest/eval/evaluate.py
Harness entrypoint: fail-loud golden-set validation, span→chunk relevance under the CURRENT chunk config, mode runs, tables/breakdowns/latency, paired A/B, `--log` append. All active tuning knobs fold into config_hash.
- `ingest/eval/evaluate.py:main` — CLI: parse knobs → `goldset.load_golden_set`/`reground`/`enforce_holdout`/`lint_span_coverage` → build backend(s) → run modes → print + `stats.compare` → `explog.config_hash`/`append_run`. calls: `ingest/eval/goldset.py:load_golden_set`, `ingest/eval/goldset.py:reground`, `ingest/eval/goldset.py:enforce_holdout`, `ingest/eval/goldset.py:lint_span_coverage`, `ingest/eval/goldset.py:eval_set_hash`, `ingest/eval/translations.py:load_query_translations` (only under `--translate-queries`; the file-content hash, never the dict, enters config_hash), `ingest/eval/explog.py:config_hash`, `ingest/eval/explog.py:append_run`, `ingest/eval/stats.py:bootstrap_ci`, `ingest/eval/stats.py:compare`, `ingest/ingest/config.py:load_config`. called_by: `ingest/scripts/phase_c_full.sh`, `ingest/scripts/phase_c.sh`, improvement.md loop.
- `ingest/eval/evaluate.py:build_query_relevance` — per-query {key: grade} at chunk+doc granularity; re-derives chunk judgments so the gold set survives re-chunking. calls: `ingest/eval/spanmap.py:graded_relevant_chunks`, `ingest/eval/goldset.py:SnapshotBodies.body`.
- `ingest/eval/evaluate.py:qdrant_deps` — builds client + `BGEM3Embedder` + (if `cfg.rerank_enabled`) `BGEReranker`, or `RemoteBGEReranker` when `RERANK_REMOTE_URL` is set — built ONCE so paired A/B reuses them; this is where RERANK_ENABLED matters for RAM. calls: `ingest/ingest/qdrant_store.py:make_client`, `ingest/ingest/embedding.py:BGEM3Embedder`, `ingest/ingest/rerank.py:BGEReranker`, `ingest/ingest/rerank.py:RemoteBGEReranker`.
- `ingest/eval/evaluate.py:run_mode` — per-query search → `ingest/eval/metrics.py:query_score` + per-stage latency (embed/search/rerank).
- `ingest/eval/evaluate.py:make_backend` — fake vs qdrant backend factory (knobs → `QdrantBackend`).

### ingest/eval/backend.py
The two backends. `ingest/eval/backend.py:MODES` = (bm25, dense, sparse, hybrid, rerank, routed).
- `ingest/eval/backend.py:QdrantBackend.search` — translations substitution first; bm25 → `_ensure_bm25`; routed drops sparse for EN via `detect_language`; rerank scores the whole fetched pool via `rerank_points`; diversity via `diversify`. calls: `ingest/ingest/search.py:detect_language`, `ingest/ingest/search.py:rerank_points`, `ingest/ingest/search.py:diversify`, `ingest/eval/bm25_full.py:FullCorpusBM25.search`. called_by: `ingest/eval/evaluate.py:run_mode`.
- `ingest/eval/backend.py:QdrantBackend._ensure_bm25` — prefers the prebuilt disk index (`FullCorpusBM25.cache_exists`/`FullCorpusBM25.load` with `expect_collection=` staleness guard); refuses the in-memory scroll above 200k chunks (RAM guard).
- `ingest/eval/backend.py:QdrantBackend._manual_fusion` — min-max-normalised weighted dense/sparse fusion for the `--sparse-weight` sweep (RRF is unweighted).
- `ingest/eval/backend.py:FakeBackend` — deterministic lexical backend (idf dense + tf sparse + RRF + Jaccard rerank), no torch; powers unit tests offline.

### ingest/eval/goldset.py
Loads/validates the span-anchored golden set; judgments are char spans into snapshot body_markdown, never chunk ids. Three fail-loud invariants on load.
- `ingest/eval/goldset.py:load_golden_set` — parse `ingest/eval/golden_set_v1.jsonl` (skips `#` comment lines) → list of GoldQuery. called_by: `ingest/eval/evaluate.py:main`, `ingest/scripts/rerank_latency_probe.py` (reads the jsonl directly for query text).
- `ingest/eval/goldset.py:reground` — asserts `body[char_start:char_end] == evidence_quote` under NFC; drift = snapshot hygiene changed → ValueError with ids.
- `ingest/eval/goldset.py:enforce_holdout` — every gold doc must be in `ingest/eval/holdout_doc_ids.json`.
- `ingest/eval/goldset.py:lint_span_coverage` — every span → ≥1 chunk under current config. calls: `ingest/eval/spanmap.py:map_spans_to_chunks`.
- `ingest/eval/goldset.py:SnapshotBodies` — lazy body_markdown loader over `ingest/snapshots/v1/docs/<source>.jsonl` with module-level `_SOURCE_CACHE` and a `needed` filter (matsne jsonl is multi-GB); `ingest/eval/goldset.py:SnapshotBodies.body` is the accessor.
- `ingest/eval/goldset.py:eval_set_hash` — 16-hex sha256 of the golden file, stamped on every experiments.jsonl row.
- `ingest/eval/goldset.py:EVAL_SET_VERSION` — "v1"; growth is additive-only via a future golden_set_v2.jsonl (improvement.md I5).

### ingest/eval/metrics.py
Scoring: Recall@5/@10 (hit-rate), graded nDCG@10 (gain 2^grade−1), MRR@10; chunk vs doc granularity; breakdowns; latency percentiles.
- `ingest/eval/metrics.py:score_ranking` — (recalls, ndcg, mrr) for one ranked key list vs graded relevant dict. called_by: `ingest/eval/metrics.py:query_score` (used by `ingest/eval/evaluate.py:run_mode`).
- `ingest/eval/metrics.py:METRIC_NAMES` — ("recall5","recall10","ndcg10","mrr10"), the gate-monitored metrics.
- `ingest/eval/metrics.py:breakdown` — per_query_type / per_language aggregates that improvement.md G2 slice checks read.

### ingest/eval/stats.py
numpy-only statistics: percentile bootstrap CIs, paired diff CI, sign-flip permutation p, ADOPT/TIE decision.
- `ingest/eval/stats.py:compare` — paired A/B per metric; significant iff p<0.05 AND CI excludes 0; the decision rule improvement.md G3 leans on. called_by: `ingest/eval/evaluate.py:main` (`--compare`/`--ab`).
- `ingest/eval/stats.py:bootstrap_ci` — seeded (`ingest/eval/stats.py:DEFAULT_SEED` = 12345) 10k-resample CI; deterministic reruns.
- `ingest/eval/stats.py:paired_permutation_p` — sign-flip permutation test.

### ingest/eval/explog.py
Persistent experiment ledger.
- `ingest/eval/explog.py:config_hash` — sha256-16 over `ingest/eval/explog.py:LOGIC_REV` ("eval-r1") + active retrieval knobs; LOGIC_REV bumps when scoring changes make history incomparable. called_by: `ingest/eval/evaluate.py:main`.
- `ingest/eval/explog.py:append_run` — append-only JSONL writer; `ingest/eval/explog.py:DEFAULT_LOG` = `ingest/eval/experiments.jsonl` (35 rows). `ingest/eval/experiments_gpu.jsonl` (8 rows) segregates GPU-tunnel runs whose latency is NOT CPU serving latency.

### ingest/eval/translations.py
I2 (in flight, staged-uncommitted): loader for the static authored EN→KA table `ingest/eval/query_translations_v1.json` (22 EN golden queries; no runtime translation API anywhere).
- `ingest/eval/translations.py:load_query_translations` — fail-loud: entry `query` must byte-match a golden query, source must detect as en, target must be Georgian script; returns ({original: ka}, 16-hex content hash) — the hash, not the dict, enters config_hash. calls: `ingest/ingest/search.py:detect_language`. called_by: `ingest/eval/evaluate.py:main`.
- `ingest/eval/translations.py:DEFAULT_TRANSLATIONS` — path constant to the v1 json.

### ingest/eval/bm25.py + ingest/eval/bm25_full.py
Dependency-free Okapi BM25; `bm25.py` is the single source of truth for tokenization/scoring, `bm25_full.py` is the disk-backed memory-mapped CSR full-corpus variant (scoring parity asserted in `ingest/tests/test_bm25_full.py`).
- `ingest/eval/bm25.py:tokenize` — shared tokenizer (unicode `\w+` casefold). called_by: `ingest/eval/bm25_full.py`, `ingest/eval/backend.py:FakeBackend`.
- `ingest/eval/bm25.py:BM25Index` — in-memory reference implementation (small corpora / fake backend only).
- `ingest/eval/bm25_full.py:FullCorpusBM25` — `build`/`cache_exists`/`load`/`search`; `load(expect_collection=...)` checks collection NAME only, not content. called_by: `ingest/eval/backend.py:QdrantBackend._ensure_bm25`.

### ingest/eval/spanmap.py
Maps gold char spans to covering chunks under the current chunk config: body-region overlap plus heading-governance (spans inside Markdown headings map to the chunks the heading governs, since `ingest/ingest/chunking.py:build_embed_text` carries the heading).
- `ingest/eval/spanmap.py:graded_relevant_chunks` — spans+grades → {chunk_index: max grade}. calls: `ingest/ingest/chunking.py:chunk_document`, `ingest/ingest/chunking.py:heading_spans` (via `ingest/eval/spanmap.py:chunks_covering_span`). called_by: `ingest/eval/evaluate.py:build_query_relevance`.
- `ingest/eval/spanmap.py:map_spans_to_chunks` — coverage variant. called_by: `ingest/eval/goldset.py:lint_span_coverage`.

### ingest/scripts/phase_c_full.sh (+ phase_c.sh)
Phase-C sweep driver. `run()` sets `RERANK_ENABLED=true` ONLY when args contain `--mode rerank`, applies `timeout 3600`, filters model-loading noise. Tiers: `tier1` (core modes + hybrid-vs-routed A/B + rerank depth 10/30/50/80), `diversity` (max-per-doc/MMR, GPU-offloadable via RERANK_REMOTE_URL), `tuning` (dbsf `--ab`, prefetch, ef/rescore, sparse-weight grid), `tier2` = diversity+tuning, `gpu_rerank` (requires the runpod_rerank tunnel; logs to `experiments_gpu.jsonl`, NOT the main ledger), `docs` (doc-level relevance). Shell functions — not linted; calls `python -m eval.evaluate`.

### ingest/scripts/runpod_rerank.py + runpod_rerank_server.py
GPU cross-encoder over an SSH tunnel with hard termination guarantees.
- `ingest/scripts/runpod_rerank.py:up` — keypair → provision → push server → setsid launch → health poll → tunnel (localhost:8900) → hold loop; atexit/SIGINT/SIGTERM `ingest/scripts/runpod_rerank.py:_terminate` + `ingest/scripts/runpod_rerank.py:MAX_UPTIME_S` 120-min billing watchdog + stop-file. calls: `ingest/scripts/runpod_orchestrate.py:step_keypair`, `ingest/scripts/runpod_orchestrate.py:step_provision`, `ingest/scripts/runpod_orchestrate.py:step_wait_ssh`, `ingest/scripts/runpod_orchestrate.py:ssh_capture`, `ingest/scripts/runpod_orchestrate.py:push_content`, `ingest/scripts/runpod_orchestrate.py:terminate` (imported `as O`).
- `ingest/scripts/runpod_rerank_server.py:score` — runs ON the pod; mirrors `ingest/ingest/rerank.py:BGEReranker` scoring exactly (BAAI/bge-reranker-v2-m3, max_length 512, sigmoid logit) so GPU scores are numerically comparable to CPU; POST /score consumed by `ingest/ingest/rerank.py:RemoteBGEReranker` via RERANK_REMOTE_URL.

### ingest/scripts/runpod_orchestrate.py (+ _multi.py, runpod_embed*.sh)
One-time full-corpus GPU embed orchestrator and the shared RunPod library: encrypt-package code+snapshot (passphrase memory-only), browser-UA GraphQL provisioning (`ingest/scripts/runpod_orchestrate.py:gql` — Cloudflare 403 without the UA), tmux-run embed, poll, pull snapshot, cosine checksum gate (`ingest/scripts/runpod_orchestrate.py:step_verify_g2`), ALWAYS terminate, restore locally.
- `ingest/scripts/runpod_orchestrate.py:step_provision` / `ingest/scripts/runpod_orchestrate.py:step_wait_ssh` / `ingest/scripts/runpod_orchestrate.py:ssh_capture` / `ingest/scripts/runpod_orchestrate.py:push_content` / `ingest/scripts/runpod_orchestrate.py:terminate` — pod lifecycle helpers (idempotent terminate). called_by: `ingest/scripts/runpod_rerank.py`, `ingest/scripts/runpod_orchestrate_delta.py`.
- `ingest/scripts/runpod_orchestrate_multi.py:main` — 4x4090 sharded variant.

### Delta pipeline (embed_delta / merge_delta_collection / runpod_orchestrate_delta / verify_delta_embedded)
Ships only explicit source items to a uniquely named GPU collection, validates the returned snapshot, restores the staging collection locally, then leaves merge as a separate guarded operation.
- `ingest/scripts/runpod_orchestrate_delta.py:main` — exact host dry-run and pre-spend local-Qdrant/absent-target proof → owner-only exclusive spend lock → live balance/zero-pod/Secure-4090 stock+price/cleanup-margin/$2-reserve gate → package/key → repeat the live gate immediately before deploy → exactly one provider- and hardware-attested 4090 → monotonic parent billing alarm across all blocking paid work → separate pre-embed checksum → strict embed → snapshot SHA/count/ID/UUID/chunk validation → signal-shielded strict name/ID reconciliation and confirmed termination → target recheck and approved unique local restore. Budget exhaustion bypasses retries; non-strict cleanup never forgets an uncertain deploy name.
- `ingest/scripts/embed_delta.py:main` — source-aware RAW normalization and exact tokenizer/chunker manifest parity; writes only an explicit run-scoped collection after the dual CLI/env approval.
- `ingest/scripts/merge_delta_collection.py:run_merge_workflow` — exact run-manifest preflight, Qdrant write lock, durable hashed rollback, idempotent merge/stale-tail cleanup, exact post-check + existing-Supreme preservation, rollback on failure, and cleanup only of the manifest-bound staging collection. The legacy CLI entry point is intentionally disabled under the generation migration.
- `ingest/scripts/verify_delta_embedded.py:delta_doc_ids` + `ingest/scripts/verify_delta_embedded.py:main` — exact per-id chunk-count verification for `--runs-since` deltas.

### ingest/scripts/verify_all_embedded.py
Embed-coverage ground truth (~3 min; exit 1 if anything missing). 2026-07-09: VERIFIED 0 missing (207,940/208,218 embedded; 278 excluded by design).
- `ingest/scripts/verify_all_embedded.py:scraped_universe` — universe = unique per-source document_ids across `artifacts/<source>/{latest,runs/*}/items.jsonl`, ids derived EXACTLY like ingest (':'.join of SourceSpec.id_fields — ecd uses decision_document_id, constcourt legal_id, tbappeal slug). calls: `ingest/ingest/sources.py:SOURCES`.
- `ingest/scripts/verify_all_embedded.py:embedded_universe` — ONE full collection scroll of source+document_id payloads.
- `ingest/scripts/verify_all_embedded.py:quarantined_ids` — subtracts `ingest/snapshots/v1/quarantine.jsonl` (by-design exclusions) so a clean exit 0 is reachable; `ingest/scripts/verify_all_embedded.py:no_text_ids` classifies empty-body misses.
- Writes `ingest/.state/embed_coverage.json` + `ingest/.state/embed_missing.txt`; called by `ingest/scripts/daily_ingest.sh` stage 3 as its success gate; the json is rendered by session_monitor.

### ingest/scripts/session_monitor.py + monitor_server.py
Read-only stdlib dashboards.
- `ingest/scripts/session_monitor.py:main` — ThreadingHTTPServer on :8770 (arg overrides): live Claude Code sessions from ~/.claude transcripts (`ingest/scripts/session_monitor.py:sessions_state`), RAM/swap, Qdrant RAM, leaked-MCP-server count, GPU pod health (`ingest/scripts/session_monitor.py:gpu_state`), recent legal_search latencies, and `ingest/scripts/session_monitor.py:coverage_state` rendering `.state/embed_coverage.json`.
- `ingest/scripts/monitor_server.py:main` — :8765 embed-pod dashboard over SSH.

### ingest/scripts/publish_snapshot.py
Transports a prebuilt immutable schema-v2 generation to the RunPod network volume; it no longer snapshots an implicit live collection.
- `ingest/scripts/publish_snapshot.py:upload` rehashes the local owner-only artifact, resumes only on remote size+SHA identity, and requires a proven conditional activator before `_publish_manifest_object` activates `publish/manifest.json` LAST.
- `ingest/scripts/publish_snapshot.py:verify` requires an explicitly confirmed cold worker and exact generation ID/manifest SHA/point/identity parity. Upload and verify also require `--apply` plus `PUBLISH_REMOTE_APPROVED=1`; legacy create and destructive cleanup are disabled.

### ingest/scripts/daily_ingest.sh
3-stage daily pipeline: scrape (seen.sqlite delta-only) → `python -m ingest watch --source all --once` → verify_all_embedded gate. flock self-exclusion; exits 0 without acting when a `coordination/locks` lock is held; every stage idempotent; per-stage timeouts (DAILY_INGEST_*_TIMEOUT); systemd units in `ingest/systemd/`.

### Reporting/ops satellites
- `ingest/scripts/build_phase_c_report.py:load_rows` — experiments.jsonl → markdown tables (`ingest/eval/phase_c_report_tables.md`, feeding `ingest/eval/phase_c_report.md`); paired A/B verdicts (Δ/CI/p/ADOPT-TIE) are NOT in the jsonl — `ingest/scripts/build_phase_c_report.py:parse_paired` regex-scrapes them from sweep stdout logs.
- `ingest/scripts/rerank_latency_probe.py:main` — reranker-only CPU latency at depth {10,30,50,80} (no co-loaded BGE-M3/Qdrant → no OOM); writes rerank_xval.json (default `/tmp/rerank_xval.json`, argv[1] overrides; the kept copy is `ingest/eval/rerank_xval.json`) used to cross-validate GPU vs CPU scores.
- `ingest/scripts/backfill_consolidation.py:main` — LEGACY set_payload-only `is_consolidated` flag (no re-embed; false pass opt-in since 2026-07-10; authority = `ingest/scripts/reconcile_consolidated.py:main`).
- `ingest/scripts/verify_matsne_completeness.py:main` — three live-site audits (`ingest/scripts/verify_matsne_completeness.py:audit_advertised`, reference-closure, id-enum) → residual_missing_ids.txt for seed re-fetch.

### improvement.md (repo root) + .improvements/
The gated improvement runbook: hard rules (never pull origin/dev, RERANK_ENABLED=false for non-rerank evals, no external APIs in eval, checkpoint-commit permission is Step 0), the per-attempt loop (ONE knob → tests → cheap hybrid iterate + rerank@50 confirm → gates G1–G5 → keep-or-revert → ledger), queue I1–I7, DO-NOT list (MMR measured harmful), runtime budgets, append-only ledger §8 with the re-baseline rule. `.improvements/` is the patch graveyard: `i1_citation_route_full.patch` (new citations.py + two test files + search.py/config.py/backend.py/evaluate.py edits) and `i1_citation_route_partial.patch` (the measured `ids` variant) are the ONLY surviving copies of I1 — the worktree files `ingest/ingest/citations.py`, `ingest/tests/test_citations.py`, `ingest/tests/test_citation_route.py` are deleted and were never committed. Re-apply with `git apply` only after consolidation backfill + I5.

## Artifact flows
- Consumes: `ingest/eval/golden_set_v1.jsonl` (frozen ground truth) + `ingest/eval/holdout_doc_ids.json`; snapshot bodies from `ingest/snapshots/v1/docs/<source>.jsonl`; `ingest/snapshots/v1/quarantine.jsonl` (verify exclusions); `artifacts/<source>/{latest,runs/*}/items.jsonl` (scrape output, coverage universe + delta staging); `ingest/eval/query_translations_v1.json` (I2); `ingest/.env` (RunPod creds).
- Produces: `ingest/eval/experiments.jsonl` (append-only ledger, 35 rows) and `ingest/eval/experiments_gpu.jsonl` (8 rows, GPU-tunnel latency only); `ingest/eval/.bm25_full/` (git-ignored disk BM25 index); `ingest/eval/phase_c_report_tables.md` / `phase_c_report.md`; `ingest/eval/rerank_xval.json`; `ingest/.state/embed_coverage.json` + `embed_missing.txt`; `.improvements/*.patch`; improvement.md ledger rows.
- Qdrant collections: `georgian_legal` (main, 2,637,645 points post delta-merge), `georgian_legal_delta` (pod-embedded delta, merged then disposable); collection snapshots pulled from pods and restored locally; publish artifacts on the RunPod volume (`publish/manifest.json` last, consumed by `ingest/serverless/qdrant_boot.py:needs_restore`).
- State/locks: `ingest/.state/` (daily_ingest flock, watch offsets), `coordination/locks/` (daily_ingest defers to eval/writer locks), `~/gpu_embed_work/pod.id` + stop-files on the operator box.

## Hazards
- RAM/swap thrash: `RERANK_ENABLED` defaults to TRUE in `ingest/ingest/config.py:load_config`; loading the 2.3GB reranker for non-rerank eval modes evicts Qdrant's ~21GB hot set to swap → ~275 s/query (a sparse run once took 7.9h). phase_c_full.sh's `run()` sets it per-mode, but any hand-run `python -m eval.evaluate` for a non-rerank mode MUST export RERANK_ENABLED=false.
- Never run `--mode bm25` without the prebuilt `ingest/eval/.bm25_full/` index: `ingest/eval/backend.py:QdrantBackend._ensure_bm25` refuses the in-memory scroll above 200k chunks, and `ingest/eval/bm25_full.py:FullCorpusBM25.load` checks collection NAME but not content — a re-embedded/merged corpus silently serves a stale BM25 index until `python -m eval.bm25_full build` is rerun.
- Frozen yardstick: never edit `ingest/eval/golden_set_v1.jsonl` (its `eval_set_hash` anchors all ledger rows); growth is additive-only (golden_set_v2.jsonl per I5). `experiments.jsonl` is append-only, including failed attempts.
- Re-baseline rule: eval rows are comparable ONLY at the same corpus points_count (currently 2,637,645; the 2026-07-09 Phase-C numbers at 2,453,915 are a different world). Cross-corpus comparison is the top silent-wrongness risk here.
- config_hash subtleties: includes `LOGIC_REV` and only ACTIVE knobs (None and default fusion are stripped, so plain runs hash identically to historical rows); the `knobs` log field is descriptive-only; the translations dict never enters the hash — its file-content sha256-16 does.
- Goldset fail-loud tripwires (`reground`, `enforce_holdout`, `lint_span_coverage`): changes to snapshot hygiene, chunk config, or tokenizer make eval ERROR by design — never "fix" by editing the golden set.
- I1 revert left fragile git state: `ingest/ingest/citations.py` and the two test files are staged-then-deleted (' D', never committed); `.improvements/i1_citation_route_full.patch` is the only copy of that work — a careless `git checkout -- .` or index reset destroys recoverability.
- I2 is mid-flight and uncommitted (`ingest/eval/translations.py`, `ingest/eval/query_translations_v1.json`): other sessions must not revert or "clean up" these files (coordination protocol in coordination/README.md).
- RunPod billing leaks: pods must ALWAYS be terminated (`python scripts/runpod_rerank.py down`; an orphaned billing pod has happened); the 120-min watchdog is a backstop, not the protocol. Source-aware delta spend additionally requires `runpod_spend_lock` from its live account gate through confirmed cleanup, with cost measured from the provision attempt and the $2 reserve checked at each long-phase boundary and on every embed poll/completion check.
- Delta snapshot pulls can truncate (2026-07-09: dead pod, truncated tar, $0.89 lost) — pull timeout is now 1800s; `tar -tf` any snapshot before restore; a failed pull loses the pod's vectors entirely.
- `experiments_gpu.jsonl` latency is tunnel RTT, not CPU serving latency — never quote it; use rerank_latency_probe for CPU numbers (full rerank eval OOMs if BGE-M3 + reranker + Qdrant co-load, hence the isolated probe).
- verify_all_embedded id-parity: ids must come from SourceSpec.id_fields join, not a raw document_id field — ecd/constcourt/tbappeal would otherwise report phantom missing docs.
- build_phase_c_report scrapes A/B verdicts from sweep STDOUT logs by regex — deleting /tmp sweep logs loses the verdicts permanently.
- publish_snapshot ordering is load-bearing: manifest.json uploads LAST; uploading it early lets a half-finished ~24GB upload trigger a broken serverless restore.
- MCP staleness: after editing anything under `ingest/`, the running legal_rag MCP server serves stale code until reconnected via /mcp; the CLI eval path is always current.
- No-peeking: gate thresholds are fixed before the run; max two variants per queue item then BLOCKED; seeds are fixed (12345) so identical reruns are deterministic anyway.

## Cross-area
- Retrieval-serving: `ingest/eval/backend.py:QdrantBackend.search` imports the production code path — `ingest/ingest/search.py:detect_language` / `ingest/ingest/search.py:rerank_points` / `ingest/ingest/search.py:diversify` — and `ingest/eval/evaluate.py:qdrant_deps` builds `ingest/ingest/embedding.py:BGEM3Embedder`, `ingest/ingest/qdrant_store.py:make_client`, `ingest/ingest/rerank.py:BGEReranker` / `ingest/ingest/rerank.py:RemoteBGEReranker`; eval numbers therefore measure the same code `ingest/ingest/mcp_server.py` serves. See retrieval-serving.md.
- Chunking/config: `ingest/eval/spanmap.py:graded_relevant_chunks` and `ingest/eval/goldset.py:lint_span_coverage` chunk with `ingest/ingest/chunking.py:chunk_document` / `ingest/ingest/chunking.py:heading_spans` under `ingest/ingest/config.py:load_config` chunk_cfg — a chunk-config change over there silently re-shapes gold relevance here (span-coverage lint fails loud if a span becomes uncoverable). See ingest-pipeline.md.
- Snapshots/hygiene: `ingest/eval/goldset.py:SnapshotBodies` reads `ingest/snapshots/v1/docs/<source>.jsonl` produced by the ingest snapshot/hygiene pipeline; any normalization change there trips `ingest/eval/goldset.py:reground` drift errors here.
- Env knobs: RERANK_ENABLED / RERANK_REMOTE_URL are consumed via `ingest/ingest/config.py:load_config` (rerank_enabled / rerank_remote_url, part of the retrieval fingerprint); phase_c_full.sh and improvement.md rules set them per-run. The runpod_rerank tunnel URL (http://localhost:8900) feeds `ingest/ingest/rerank.py:RemoteBGEReranker` for BOTH the eval path and live serving — one env var flips both.
- Ingest identity: `ingest/scripts/verify_all_embedded.py:scraped_universe` derives ids from `ingest/ingest/sources.py:SOURCES` SourceSpec.id_fields; `ingest/scripts/embed_delta.py:main` normalizes via `ingest/ingest/sources.py:normalize`; delta point ids come from `ingest/ingest/qdrant_store.py:point_id` making `ingest/scripts/merge_delta_collection.py:merge_collection` idempotent.
- Serverless: `ingest/scripts/publish_snapshot.py:_publish_manifest_object` manifest-last protocol is consumed by `ingest/serverless/qdrant_boot.py:needs_restore`; the MCP thin client is `ingest/ingest/remote_search.py` (SEARCH_BACKEND=remote). See serverless-hosting.md.
- Scraper: `ingest/scripts/daily_ingest.sh` spans scraper (legal_scrapers run), ingest embed (`python -m ingest watch`), and this area's verify gate; it respects `coordination/locks/` from the repo-root coordination protocol. See scraper.md.
- I1 patches land OUTSIDE this area when re-applied: `.improvements/i1_citation_route_full.patch` touches `ingest/ingest/search.py:hybrid_search` and `ingest/ingest/config.py` (plus eval/backend.py + eval/evaluate.py plumbing).
