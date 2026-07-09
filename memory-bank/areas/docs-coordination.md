# Docs, config & coordination (knowledge layout, standing rules)
> Drill-down memory. Not auto-loaded — opened per the pre-modification ritual in INDEX.md. Anchors linted by ingest/scripts/gen_code_map.py --check.

## Overview
This area is the repo's knowledge and control layer: which document is authoritative for what, the config surfaces, and the multi-session coordination protocol. Knowledge is deliberately layered: `prompt.md` is the frozen phased build spec (Parts 0–5 + appendix, global guardrails, RunPod runbook); `HANDOFF.md` is the living session-state doc (Parts 0–3 DONE, Part 4 CUT — Claude composes answers, Part 5 = current frontier; corpus verified at 2,637,645 points / 207,940 of 208,218 docs); `improvement.md` is the gated retrieval-accuracy runbook with a keep-only-if-gate-passes protocol and an append-only ledger (I1 citation routing REVERTED → BLOCKED; I2 EN→KA translation in flight); `ingest/docs/*.md` are operational runbooks; `ingest/eval/phase_c_report.md` is the measured evidence base. `coordination/` (gitignored, never in diffs) is a live multi-session protocol: per-session claim files, an append-only message board, and machine-resource lock files for the 30GB CPU-only box. The product decision (2026-07-08 pivot) is MCP-first: Claude is the client that composes cited answers, there is no local generation LLM — retrieval quality is the whole product. Standing rules repeated across `CLAUDE.md`/`HANDOFF.md`/`improvement.md`: never commit unasked; never pull/merge `origin/dev` (divergent LangChain fork); never edit existing tests to make them pass; no external APIs in ingest/embed/eval; PII stays in the index. Root `README.md` exists and is current (session-a-delivery rewrote it as a Part 5 deliverable after moving the old scraper README to `scraper/README.md`).

## Entrypoints
- `/mcp` in Claude Code → `.mcp.json` spawns `uv run --directory ingest python -m ingest.mcp_server` → `ingest/ingest/mcp_server.py:main` (pidfile singleton; the product surface, 8 tools; OMP/MKL_NUM_THREADS=14).
- `bash ingest/scripts/daily_ingest.sh` → scrape (seen.sqlite delta) → `python -m ingest watch --source all --once` (`ingest/ingest/pipeline.py:watch_loop`) → `ingest/scripts/verify_all_embedded.py:main` (exit 0 required). Wired to `ingest/systemd/legal-ingest.timer` — deliberately NOT enabled.
- `ingest/systemd/legal-monitor.service` → `ingest/scripts/session_monitor.py:main` — ops dashboard on `http://localhost:8770`.
- `python run_all.py` → `run_all.py:main` — scrape + ingest-watch together.
- `cd ingest && uv run python -m ingest {ingest|watch|search}` — CLI documented in `ingest/README.md`.
- `docker compose up -d` (in `ingest/`) → local Qdrant, image pinned `qdrant/qdrant:v1.18.2`, loopback ports 6333/6334.
- `uv run --group publish python scripts/publish_snapshot.py …` → `ingest/scripts/publish_snapshot.py:main` — serverless snapshot publish (boto3 kept out of the MCP runtime env).
- `bash ingest/scripts/phase_c_full.sh {tier1|tuning}` — eval baselines/grids, per improvement.md preflight.
- `python3 ingest/scripts/gen_code_map.py --check` → `ingest/scripts/gen_code_map.py:lint` — lints every memory-bank anchor and diffs `memory-bank/generated/symbols.md` against a fresh regeneration.

## Modules

### CLAUDE.md — repo entry instructions
Checked-in instructions every Claude Code session loads first.
- Multi-session pointer — before editing any tracked file: read `coordination/README.md`, register in `coordination/sessions/`, check other sessions' claims; uncommitted diffs you didn't make are other sessions' in-flight work, never revert/fix/commit them.
- Read-first pointers — `HANDOFF.md` = project state, `improvement.md` = gated improvement queue.
- Standing rules — never commit unasked; never pull/merge `origin/dev`; never edit existing tests to make them pass.

### HANDOFF.md — living session state
Updated at end of every session; the freshest source when docs conflict.
- Part status: 0–3 DONE, 4 CUT, 5 = frontier; appendix budget-gated (quotes $10.53 RunPod credits — improvement.md says ~$4.30; verify live before spending).
- Phase C headline quotes `ingest/eval/phase_c_report.md`: fingerprint `81c807b279399098`, hybrid+rerank@80 nDCG@10 0.289 vs BM25 0.192 (+50%); recommended interim prod config hybrid+rerank@50 no diversity.
- Corpus-verified section pins coverage ground truth to `ingest/scripts/verify_all_embedded.py:main` (id-parity with `ingest/ingest/sources.py:SourceSpec` id_fields) writing `ingest/.state/embed_coverage.json`, rendered by `ingest/scripts/session_monitor.py:coverage_state`.
- Eval-comparability caveat: all Phase C numbers were measured on the older 2,453,915-point index; re-baseline (`ingest/scripts/phase_c_full.sh`) and rebuild `ingest/eval/.bm25_full/` before any A/B.
- NEXT SESSION menu: Part 5 delivery, serverless provisioning+seed, Part 3 leftovers, golden_set_v2 promotion from `ingest/.state/queries.jsonl`, budget-gated ablations.
- Critical-gotchas and standing-decisions sections mirror the Hazards list below.

### improvement.md — gated accuracy runbook + ledger
Hard rules, preflight, measure-gate-keep/revert loop, ordered queue I1–I7, DO-NOT list, append-only ledger.
- Gate: keep only if target-metric threshold met, no monitored slice regresses >0.02 abs, rerank p50 latency growth ≤10%, tests+ruff clean, every knob folded into the config hash; max two variants then BLOCKED.
- I1 citation routing: ATTEMPTED 2026-07-10 → REVERTED → BLOCKED (golden citation queries' gold docs are decisions/amendments *citing* the number, not the numbered act; bare numbers ambiguous in matsne). Patch preserved at `.improvements/i1_citation_route_full.patch` (+ `_partial`); the revert deleted `ingest/ingest/citations.py` and its tests. Touched `ingest/ingest/search.py:hybrid_search` and `ingest/ingest/mcp_server.py:legal_lookup`. Re-attempt only after consolidation backfill + I5.
- I2 EN→KA query translation (next): static checked-in `ingest/eval/query_translations_v1.json` loaded by `ingest/eval/translations.py:load_query_translations` (no runtime APIs) + a backend knob + `ingest/ingest/mcp_server.py:legal_search` docstring guidance; gate = cross_lingual R@10 +≥0.09. Files are in the tree per git status but no ledger row yet — in flight, claimed by the improvement session.
- I3–I7: fusion/sparse-weight tuning, agentic retry + abstention calibration, golden set growth to 150–250 pairs (additive v2, v1 frozen), embed-header v2 (PARKED), ONNX int8 reranker.
- DO-NOT list (measured): MMR (0.289→0.074), blanket multi-query/RAG-fusion, default HyDE, ColBERT, rerank depth >80, `--mode all`, evals on a yellow index.
- Ledger + re-baseline rule keyed on `points_count`; current baseline row 2026-07-10 @ 2,637,645 pts: hybrid nDCG 0.171 / R@10 0.311 / XL 0.136 / citation 0.529 / paraphrase 0.000. Gate reads `ingest/eval/experiments.jsonl` rows written by `ingest/eval/evaluate.py:main` (`--log`); eval-set identity via `ingest/eval/goldset.py:eval_set_hash`.

### prompt.md — frozen phased spec
Paste-one-Part-at-a-time build spec; HANDOFF tracks status against it. Do not edit.
- Module map names the pre-existing core: `ingest/ingest/sources.py:CanonicalDoc` + `ingest/ingest/sources.py:SourceSpec`, `ingest/ingest/search.py:hybrid_search` + `ingest/ingest/search.py:build_filter`, `ingest/ingest/pipeline.py:watch_loop`, MCP tools in `ingest/ingest/mcp_server.py:legal_search`.
- Global guardrails: privacy absolute (no external APIs, RunPod Secure Cloud only, wipe pods), one BGE-M3 vector space with checksum verification, CPU serving ~60 q/day p95≤10s, nothing ships without CIs + paired significance + BM25 floor.
- Corpus contract: true corpus = union across `artifacts/<source>/runs/*` (constcourt/tas bulk not in `latest/`); body text = `body_markdown`.
- Eval-set contract: `ingest/eval/golden_set_v1.jsonl` (span-anchored relevance, NFC-exact, loaded by `ingest/eval/goldset.py:load_golden_set`) + `ingest/eval/holdout_doc_ids.json` (51 docs excluded from any future synthetic training).
- RunPod runbook (6-step) and Part 5 spec (deployment docs, README, daily-ingestion schedule) — the current frontier.

### coordination/ — live multi-session protocol (gitignored)
Never appears in git diffs or checkpoints; always on disk.
- `coordination/README.md`: session registration (one owner-only file `coordination/sessions/<name>.md`, final line SESSION ENDED); claims check (`grep -rl <path> coordination/sessions/` excluding your own); messages.md write rule — append ONLY via shell `cat >> … <<'EOF'` (Edit/Write tools clobber concurrent appends); locks table `coordination/locks/{qdrant-write,reranker-ram,eval-run}.lock` protecting the 30GB box; conflict resolution (gated work beats ungated); shared-hot-file etiquette (HANDOFF.md: announce, re-read immediately before editing, touch own sections only; `.gitignore`/`CLAUDE.md`/pyproject: re-read, keep diffs additive).
- `coordination/sessions/session-a-delivery.md`: Part 5 delivery track — claims root README, `ingest/docs/deployment.md`, `ingest/scripts/daily_ingest.sh`, `ingest/systemd/*`, CLAUDE.md; explicitly NOT touching the I1/I2 files (`ingest/ingest/search.py`, eval) or the serverless track files.
- `coordination/sessions/session-memory-bank.md`: this memory-bank track — claims `memory-bank/**`, `ingest/scripts/gen_code_map.py`, additive-only CLAUDE.md/`ingest/pyproject.toml` edits.
- `coordination/messages.md`: append-only board; poll it + `git status --short` before every shared-file write.

### READMEs
- `README.md` (root, current): product overview with pipeline diagram, corpus/index table (208,218 scraped / 207,940 embedded / 2,637,645 points), recommended serving config + golden-set scores.
- `scraper/README.md`: 7 spiders sharing BaseLegalSpider scaffolding; artifacts layout `artifacts/<spider>/{runs/<run_id>/,latest/,seen.sqlite}`; dedup = skip-forever by identity (same identity ingest uses; force with `-s DEDUP_ENABLED=False`); per-spider quirks (tas=Playwright/DWR, napr=PDF, constcourt=DOCX, supremecourt excluded from corpus); `run_all.py:main` = scrape + ingest-watch.
- `ingest/README.md`: CLI usage, watch-mode guarantees (backfills across ALL `runs/*` oldest→newest then tails; per-source byte-offset checkpoint `ingest/.state/<source>.watch.json`, offsets advance only over acknowledged upserts — `ingest/ingest/pipeline.py:watch_drain_source`), security posture (Qdrant loopback-only, API key never over non-HTTPS), idempotency (deterministic point IDs, stale-chunk removal). STALE: predates Part 5 — lists only 3 MCP tools; the server has 8.

### Config surfaces
- `.mcp.json`: registers the `legal_rag` stdio server → `ingest/ingest/mcp_server.py:main`. Server caches code and `.env` at spawn — `/mcp` reconnect after editing `ingest/`.
- `ingest/.env.example` → consumed by `ingest/ingest/config.py:load_config`; every retrieval knob folds into `ingest/ingest/config.py:retrieval_fingerprint` (16-hex, stamped into every response and query-log row). Keys: QDRANT_URL/API_KEY, COLLECTION_NAME=georgian_legal, EMBED_MODEL=BAAI/bge-m3, DENSE_DIM=1024, RERANK_* (CANDIDATES=80, MIN_SCORE=0.3), CHUNK_* (512/80/64), ARTIFACTS_ROOT. Live `.env` (gitignored) adds RUNPOD_API_KEY/VOLUME_ID/ENDPOINT_ID; `ingest/docs/deployment.md` documents SEARCH_BACKEND (local|remote), RERANK_REMOTE_URL, QUERY_LOG_ENABLED/QUERY_LOG_PATH, RUNPOD_API_TIMEOUT.
- `ingest/pyproject.toml`: standalone uv application, Py 3.11–3.13 (torch lacks 3.14 wheels), torch pinned to the pytorch-cpu index; dependency-groups dev=pytest+ruff, publish=boto3 (only `ingest/scripts/publish_snapshot.py:main`).
- Root `pyproject.toml`: scraper project, Py>=3.14 — scrapy, scrapy-playwright (tas), pdfplumber (napr), mammoth (constcourt). Two venvs on purpose.
- `ingest/docker-compose.yml`: qdrant image PINNED to v1.18.2 (snapshots only compatible within the same minor version as the serverless worker bundles); loopback 6333/6334; `${QDRANT_API_KEY:?}` enforced; data at `ingest/qdrant_storage/`.
- `ingest/systemd/`: `legal-ingest.service` + `legal-ingest.timer` (daily 05:00 with catch-up — created, deliberately NOT enabled) and `legal-monitor.service` (:8770; restart after editing `ingest/scripts/session_monitor.py`).

### ingest/docs/ — operational runbooks
- `ingest/docs/deployment.md`: components/ports, .env reference, bring-up order, three serving modes — local in-process (`ingest/ingest/search.py:hybrid_search`), GPU rerank pod on :8900 via SSH tunnel (`ingest/scripts/runpod_rerank_server.py:score`, RERANK_REMOTE_URL, RRF fallback on failure), fully remote (`SEARCH_BACKEND=remote` → `ingest/ingest/remote_search.py:RunPodQueueClient`). Daily chain: `ingest/scripts/daily_ingest.sh` flock-self-excludes, SKIPS if any `coordination/locks/*.lock` is held, takes qdrant-write + reranker-ram locks, DAILY_INGEST_LOOKBACK_DAYS=14 default. Recovery: snapshot restore (always `tar -tf` first) then wide-lookback ingest; delta chain `ingest/scripts/runpod_orchestrate_delta.py` → `ingest/scripts/merge_delta_collection.py:merge_collection` → `ingest/scripts/verify_all_embedded.py:main`.
- `ingest/docs/runpod-serverless.md`: whole read path (Qdrant+BGE-M3+reranker) in ONE scale-to-zero endpoint; local Qdrant stays write master. Publish protocol: `ingest/scripts/publish_snapshot.py:create` / `ingest/scripts/publish_snapshot.py:upload` (resumable 256MB parts) / `ingest/scripts/publish_snapshot.py:verify` / `ingest/scripts/publish_snapshot.py:cleanup`; manifest.json uploaded LAST = atomic publish signal; worker = `ingest/serverless/handler.py:main` + `ingest/serverless/qdrant_boot.py:maybe_restore` (failed restore = keep serving old data with STALE warning). Max workers must stay 1 (Qdrant exclusive storage lock). Status: code landed; endpoint provisioning + snapshot seed PENDING.
- `ingest/docs/delta_embed_runbook.md`: `ingest/scripts/embed_delta.py:main` reads RAW `items.jsonl` (preserves `is_consolidated`; `--dry-run --runs-since` for sizing; NO hygiene gate) → pod embed with checksum guard → snapshot → restore to `georgian_legal_delta` → `ingest/scripts/merge_delta_collection.py:merge_collection`; idempotent throughout via UUIDv5(source, document_id, chunk_index).
- `ingest/docs/gap-audit.md`: Part 0 requirement→exists/extend/build matrix. Its corpus counts (174,942) are explicitly STALE — do not cite; current truth lives in HANDOFF.md.

### ingest/eval/phase_c_report.md — evidence base
Part 3 STOP-gate report on the 2,453,915-chunk index; quoted verbatim by HANDOFF.md and improvement.md.
- Headlines: hybrid+rerank@80 nDCG@10 0.289 vs BM25 0.192; cross-lingual recovered only by rerank (BM25/sparse structurally 0.000 on EN→KA); MMR significantly harmful; measured CPU rerank p50 16.2s@10 / 66.9s@50 / 114.4s@80.
- Companion tables regenerable via `ingest/scripts/build_phase_c_report.py:load_rows` etc. from the permanent run log `ingest/eval/experiments.jsonl` (each row carries config_hash, eval-set hash, knobs).

### memory-bank/ and user memory
- `memory-bank/generated/symbols.md` is auto-generated by `ingest/scripts/gen_code_map.py:generate`; hand-written files (this one) are linted by `ingest/scripts/gen_code_map.py:lint` — repo-relative path + `:symbol` anchors only, never line numbers.
- `~/.claude/projects/...-Georgia-Legal-Search/memory/` (MEMORY.md + topic files) overlaps HANDOFF gotchas heavily; HANDOFF.md is the fresher source where they conflict (e.g. corpus-missing-base-laws is RESOLVED).

## Artifact flows
- `coordination/` (gitignored): `sessions/*.md` claim files, `messages.md` (shell-append only), `locks/*.lock` (qdrant-write, reranker-ram, eval-run; empty dir when unheld).
- `ingest/.state/` (gitignored): `embed_coverage.json` (written by `ingest/scripts/verify_all_embedded.py:main`, rendered on :8770), `mcp_server.pid` (singleton guard), `queries.jsonl` (query log, QUERY_LOG_PATH — golden_set_v2 candidate source), `<source>.watch.json` byte-offset checkpoints, `daily_ingest.flock`.
- `ingest/eval/experiments.jsonl` (tracked, append-only run log; + `experiments_gpu.jsonl`); `ingest/eval/.bm25_full/` cache (gitignored, stale after corpus growth).
- Snapshots: `ingest/snapshots/v1/` (gitignored; 184,318 hygiene-clean docs + quarantine + checksum), 24GB full-index + 1.24GB delta Qdrant snapshots in `~/gpu_embed_work/`; serverless publishes snapshot + manifest.json (last) to the RunPod S3 volume.
- Qdrant collections: `georgian_legal` (2,637,645 points, prod), `georgian_legal_delta` (merge staging for delta embeds).
- `.improvements/i1_citation_route_full.patch` + `_partial.patch`: reverted I1 diff, preserved for post-consolidation re-attempt.
- `memory-bank/generated/symbols.md`: regenerated by `python3 ingest/scripts/gen_code_map.py` (no flag); `--check` diffs it.

## Hazards
- `origin/dev` is a divergent LangChain fork — `git pull`/merge clobbers build_payload/is_consolidated/offset-chunker work; improvement.md also forbids `git reset` to commits you didn't create.
- Multiple concurrent sessions, nearly everything uncommitted on `dev`: uncommitted diffs you didn't make are other sessions' in-flight work — check `coordination/sessions/` claims before editing any tracked file.
- `coordination/messages.md` must be written ONLY via shell append (`cat >> … <<'EOF'`); Edit/Write tools clobber concurrent appends. `coordination/` is gitignored — it never shows in diffs, so don't rely on git to see it.
- The legal_rag MCP server caches code and `.env` at spawn (silently serves stale code until `/mcp` reconnect) and its pidfile singleton guard SIGTERMs the previous instance (`ingest/ingest/mcp_server.py:main`).
- 30GB CPU-only box: Qdrant + one model stack is the limit; a second resident model load swap-thrashes to ~275 s/query — RERANK_ENABLED=false for non-rerank eval modes; coordinate via `coordination/locks/`.
- Any Qdrant write turns the index yellow and hybrid queries time out for everyone; do NOT enable `legal-ingest.timer` while gated evals are in flight (its lock check only protects sessions that actually hold `eval-run.lock`).
- Eval numbers are comparable only at the same corpus `points_count` (improvement.md §8); all Phase C numbers and `ingest/eval/.bm25_full/` predate the +183k-point growth — re-baseline before any gate decision.
- `ingest/eval/golden_set_v1.jsonl` is a frozen yardstick (`ingest/eval/goldset.py:eval_set_hash` anchors history) — grow only additively as v2; never edit existing tests.
- RunPod: API needs a browser User-Agent (Cloudflare 403 error 1010); pods bill until terminated — always verify pods=[] after any job; `tar -tf` every snapshot before trusting it. Balance figures disagree: HANDOFF.md says $10.53, improvement.md says ~$4.30 — verify live before spending.
- Serverless config drift: RERANK_CANDIDATES/RERANK_MIN_SCORE on the endpoint must stay in lockstep with `ingest/.env` or scores silently change; Max workers stays 1; manifest.json always uploaded last.
- `ingest/docker-compose.yml` Qdrant pin (v1.18.2) is coupled to the version bundled in `ingest/serverless/Dockerfile` — snapshot compatibility is per minor version; do not bump casually.
- `seen.sqlite` is skip-forever by identity: source-side content changes are never re-fetched (force with `-s DEDUP_ENABLED=False`); `ingest/scripts/embed_delta.py:main` has no hygiene gate — never feed it quarantined ids.
- Chunking (CHUNK_TOKENS/CHUNK_OVERLAP/CHUNK_MIN_TOKENS) or embedding (EMBED_MODEL/DENSE_DIM) changes invalidate the whole index — one-vector-space rule with checksum verification.
- Doc staleness map: `ingest/README.md` lists 3 of the 8 MCP tools; `ingest/docs/gap-audit.md` corpus counts are historical; HANDOFF.md is the freshest state doc and the tie-breaker.

## Cross-area
- see ingest-core.md — prompt.md's module map and `ingest/README.md` watch semantics anchor to `ingest/ingest/pipeline.py:watch_loop` / `ingest/ingest/pipeline.py:watch_drain_source` and `ingest/ingest/sources.py:SourceSpec`; `ingest/scripts/daily_ingest.sh` drives `python -m ingest watch --once`; coverage ground truth `ingest/scripts/verify_all_embedded.py:main` checks id-parity against SourceSpec id_fields.
- see retrieval-serving.md — `.mcp.json` and `ingest/docs/deployment.md` boot/govern `ingest/ingest/mcp_server.py:main`; improvement.md I1/I2 target `ingest/ingest/search.py:hybrid_search` and the `ingest/ingest/mcp_server.py:legal_search` docstring; all knobs flow `ingest/.env` → `ingest/ingest/config.py:load_config` → `ingest/ingest/config.py:retrieval_fingerprint`; serving modes switch to `ingest/ingest/remote_search.py:RunPodQueueClient` (serverless) or `ingest/scripts/runpod_rerank_server.py:score` (GPU pod).
- see eval-scripts-ops.md — improvement.md's gate consumes `ingest/eval/experiments.jsonl` rows written by `ingest/eval/evaluate.py:main`; the eval contract is `ingest/eval/goldset.py:load_golden_set` + `ingest/eval/goldset.py:eval_set_hash` over `ingest/eval/golden_set_v1.jsonl`; I2 wiring lives in `ingest/eval/translations.py:load_query_translations`; runbooks here drive `ingest/scripts/publish_snapshot.py:main`, `ingest/scripts/embed_delta.py:main`, `ingest/scripts/merge_delta_collection.py:merge_collection`, and `ingest/scripts/session_monitor.py:main`.
