# Deployment runbook — Georgian Legal RAG stack

How to bring up, operate, and recover the full stack on a single box. Architecture and
project state live in the repo-root `README.md` and `HANDOFF.md`; this file is only
*how to run it*.

Components:

| Component | What | How it runs |
|---|---|---|
| Qdrant | vector DB, collection `georgian_legal` | Docker (`ingest/docker-compose.yml`), data bind-mounted at `ingest/qdrant_storage/` |
| MCP server | `legal_rag` — the product surface (Claude is the client) | spawned per-connection by the MCP client from `.mcp.json` (stdio) |
| Session monitor | ops dashboard, `http://localhost:8770` | manual or `legal-monitor.service` |
| Daily ingest | seven-source scrape → embed delta → verify | `scripts/daily_ingest.sh` via `legal-ingest.timer` |
| Weekly audit | immutable source/completeness/freshness audit | `scripts/audit_corpus_freshness.py` via `legal-corpus-freshness-audit.timer` |
| GPU rerank (optional) | remote reranker for interactive latency | RunPod pod (`scripts/runpod_rerank.py`) or serverless (`docs/runpod-serverless.md`) |

Two virtualenvs, on purpose: repo-root `.venv` (Python 3.14, scraper), `ingest/.venv`
(Python 3.12, torch/BGE-M3). Don't mix them.

## 0. Bootstrap (fresh clone)

```bash
uv sync                                  # repo root → .venv (scraper, Py 3.14)
uv run playwright install chromium       # only the tas spider needs a browser
cd ingest && uv sync                     # ingest/.venv (torch + FlagEmbedding, Py 3.12)
cp .env.example .env                     # then SET QDRANT_API_KEY in .env (required)
```

## Ports (all loopback-only)

| Port | Service |
|---|---|
| 6333 | Qdrant REST + dashboard (`/dashboard`) — needs `api-key` header |
| 6334 | Qdrant gRPC |
| 8770 | session monitor |
| 8900 | GPU rerank pod HTTP (`POST /score`, `GET /health`), reached over an SSH tunnel |

## 1. Configuration — `ingest/.env`

`cp ingest/.env.example ingest/.env` and adjust. Everything is read by
`ingest.config.load_config()` (python-dotenv). Key variables (defaults in parentheses):

| Variable | Meaning |
|---|---|
| `QDRANT_URL` (`http://localhost:6333`) | Qdrant endpoint. Non-localhost **requires** `QDRANT_API_KEY` and HTTPS |
| `QDRANT_API_KEY` | required by docker-compose (`${QDRANT_API_KEY:?…}`) — set it always |
| `COLLECTION_NAME` (`georgian_legal`) | serving collection |
| `SEARCH_BACKEND` (`local`) | `local` = embed/search/rerank in-process; `remote` = MCP tools RPC to the RunPod serverless worker, no local models/Qdrant needed |
| `RUNPOD_ENDPOINT_ID` / `RUNPOD_API_KEY` / `RUNPOD_API_TIMEOUT` (240) | serving path reads them only when `SEARCH_BACKEND=remote`; `RUNPOD_API_KEY` is also required by the GPU-pod scripts (`runpod_rerank.py`, `runpod_orchestrate*.py`, `publish_snapshot.py`) |
| `EMBED_MODEL` (`BAAI/bge-m3`) / `TOKENIZER_MODEL` (`BAAI/bge-m3`) / `RERANK_MODEL` (`BAAI/bge-reranker-v2-m3`) | model identities recorded in every production generation |
| `EMBED_REVISION` / `TOKENIZER_REVISION` / `RERANK_REVISION` | immutable lowercase commit revisions; all are mandatory in production mode |
| `DENSE_DIM` (1024) | one vector space for corpus & queries — never change one side only |
| `PRODUCTION_MODE` (`false`) / `GENERATION_ID` / `GENERATION_DIR` | production startup is fail-closed unless the configured immutable generation and collection identity agree exactly |
| `QDRANT_WRITE_APPROVED` / `QDRANT_RECREATE_APPROVED` | one-shot mutation approvals; the primary writer also requires `--apply`, an exact `georgian_legal__gen_<generation>` target, and separate recreate approval for deletion |
| `EMBED_DEVICE` (blank=cpu) / `EMBED_USE_FP16` (false) / `EMBED_BATCH_SIZE` (8) | embed runtime |
| `RERANK_ENABLED` (true) | false → results keep raw RRF fusion order (a *local* reranker failure surfaces as a tool error; automatic RRF fallback exists only for the remote reranker — see `RERANK_REMOTE_URL`) |
| `RERANK_MODEL` (`BAAI/bge-reranker-v2-m3`) / `RERANK_CANDIDATES` (80) / `RERANK_MIN_SCORE` (0.3, `none` disables) | rerank stage. Default 80 = fingerprint `06a64f548fcb4d59` (best quality); 50 = the recommended interim serving config `81c807b279399098` (~40% faster on CPU) |
| `RERANK_DEVICE` (follows `EMBED_DEVICE`) / `RERANK_USE_FP16` (false) | rerank runtime |
| `RERANK_BACKEND` (`torch`) / `ONNX_RERANK_PATH` (`ingest/.state/onnx/bge-reranker-v2-m3-int8.onnx`) | `onnx` = int8 quantized reranker: **2.6× faster CPU rerank (~28 s vs ~72 s @ rc=50) at measured-identical nDCG** (improvement I7). Requires a one-time `uv run --group onnx python scripts/export_onnx_reranker.py`. Changes the fingerprint when on; int8 scores drift ≤0.1 from fp32 — re-run `scripts/calibrate_min_score.py` before relying on the 0.92 abstention threshold under onnx |
| `RERANK_REMOTE_URL` | e.g. `http://localhost:8900` → rerank scoring is delegated to a GPU pod; on any failure serving falls back to RRF order |
| `CHUNK_TOKENS`/`CHUNK_OVERLAP`/`CHUNK_MIN_TOKENS` (512/80/64) | chunker — changing these invalidates the index |
| `ARTIFACTS_ROOT` (`<repo>/artifacts`) | scraper output the ingester reads |
| `QUERY_LOG_ENABLED` (true) / `QUERY_LOG_PATH` (`ingest/.state/queries.jsonl`) | per-query serving log (inputs for golden-set v2 candidates) |

The serving config is stamped as `retrieval_fingerprint` (16-hex, `config.py`) into
every `legal_search` response (and `ingest_status`/`legal_health`) and into every
query-log row — any answer is traceable to the exact config that produced it.

## 2. Bring-up order

```bash
# 1) Qdrant. Starting the process proves only platform liveness.
cd ingest && docker compose up -d

# Exact corpus readiness requires a verified immutable generation. This streams the
# configured collection/alias and writes an owner-only integrity report beside it.
.venv/bin/python scripts/verify_generation.py "$GENERATION_DIR" \
  --collection "$COLLECTION_NAME"
# Exit 0 is required. A green or merely populated collection is not corpus readiness.

# 2) index present? If points_count is 0 or the collection is missing → §5 Recovery.

# 3) MCP server — nothing to start manually. The client (Claude Code / Claude Desktop)
#    spawns it from .mcp.json:  uv run --directory <repo>/ingest python -m ingest.mcp_server
#    (stdio transport, OMP/MKL_NUM_THREADS=14).

# 4) monitor (optional)
.venv/bin/python scripts/session_monitor.py 8770   # → http://localhost:8770
```

MCP server behavior worth knowing:
- **Singleton pidfile guard** (`ingest/.state/mcp_server.pid`): on start it SIGTERMs a
  previous live server whose cmdline matches `ingest.mcp_server`. This prevents stacked
  ~4.6 GB model processes on every `/mcp` reconnect (which swap-killed the box before).
- **Code/.env are cached at spawn.** After editing anything under `ingest/`, reconnect
  the client (`/mcp` in Claude Code) or you silently run stale code.
- Models load lazily on the first `legal_search` call (~2 GB embedder + ~2.3 GB reranker
  in local mode). Tools that don't search (`legal_lookup`, `legal_browse`,
  `legal_get_document`, …) never load torch.
- Smoke test without MCP: `cd ingest && .venv/bin/python -m ingest search "შრომის კოდექსი" --top-k 5`

## 3. Serving modes

1. **Local (default).** Everything in-process. Measured CPU rerank p50 on this box:
   ~114 s/query at the default depth 80, ~67 s @50, ~16 s @10 — fine for agentic/batch
   use, not interactive. `RERANK_ENABLED=false` gives sub-second RRF-only answers at
   lower precision.
2. **Local + GPU rerank pod.** `scripts/runpod_rerank.py up` provisions a pod serving
   `runpod_rerank_server.py` on :8900 over an SSH tunnel; set `RERANK_REMOTE_URL=http://localhost:8900`.
   ⚠ Fragile and billable: **always** `scripts/runpod_rerank.py down` after, and verify
   RunPod shows `pods=[]`. Serving auto-falls back to RRF order if the pod dies.
3. **Fully remote (serverless).** `SEARCH_BACKEND=remote` turns the MCP server into a thin
   client of a scale-to-zero RunPod serverless endpoint (GPU embed+rerank+Qdrant in one
   worker) — the retrieval tools then need no local models or Qdrant (exception:
   `ingest_status` stays local and still reads the local Qdrant + watcher state).
   Provisioning, snapshot seeding, and cost model: `docs/runpod-serverless.md`.

## 4. Daily ingestion schedule

`scripts/daily_ingest.sh` chains: scrape (seven corpus sources, including Supreme Court;
`seen.sqlite` keeps it
delta-only) → `python -m ingest watch --source all --once` (CPU-embeds the delta, drains,
exits) → `scripts/verify_all_embedded.py` (document-ID coverage; exit 0 required).
Logs to `ingest/.state/daily_ingest.log`. `--dry-run` prints the plan + preflight only.

**Release hold:** do not enable the timer. Every non-dry invocation, including a manual
run, is intentionally disabled because this legacy path writes a serving corpus in place.
`DAILY_INGEST_APPROVED=1` remains a necessary authorization marker but is not sufficient to
re-enable it. Replace the writer with a new immutable-generation workflow, then pass exact
integrity, same-corpus retrieval, hosted CI, and release-identity gates before changing this
guard. An old ID-only coverage report is not approval.

Safety: `flock` self-exclusion; **skips cleanly if any `coordination/locks/*.lock` is
held** (an eval or another writer is running — ingest writes turn the index yellow); takes
`qdrant-write` + `reranker-ram` locks while running; per-stage `timeout`s; every stage
idempotent, so a killed run just resumes next day.

Stage the timer files (05:00 local, catch-up after downtime), but do not enable them until the
release hold above is cleared:

```bash
mkdir -p ~/.config/systemd/user
cp ingest/systemd/legal-ingest.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
# Do not uncomment DAILY_INGEST_APPROVED or enable the timer. A future reviewed
# immutable-generation service must replace this staged legacy unit first.
loginctl enable-linger $USER        # fire without an active login session
systemctl --user list-timers legal-ingest.timer   # check next run
journalctl --user -u legal-ingest.service -e      # logs (also .state/daily_ingest.log)
```

Tuning via env (in the service file): `DAILY_INGEST_LOOKBACK_DAYS` (default 14) — run a
wide sweep monthly (`DAILY_INGEST_LOOKBACK_DAYS=90 scripts/daily_ingest.sh`) to catch
late-published documents; dedup makes wide windows safe, just slower. Summary-only or
incomplete TAS/Tbilisi Appeal records remain quarantined from authoritative evidence.

The weekly audit is read-only with respect to the corpus and may be staged separately from
the disabled legacy writer. It checksum-loads an immutable generation and an
official-source observation file, compares authoritative version records by
`(source, document_id, version_id)`, and creates a new owner-only report without replacing
an earlier report:

```bash
install -d -m 700 ~/.local/state/georgia-legal-search/freshness-audits
cp ingest/systemd/legal-corpus-freshness-audit.{service,timer} ~/.config/systemd/user/
# Set LEGAL_SEARCH_REPO, LEGAL_CORPUS_GENERATION, and LEGAL_CORPUS_FRESHNESS_INPUT in:
$EDITOR ~/.config/georgia-legal-search.env
systemctl --user daemon-reload
systemctl --user enable --now legal-corpus-freshness-audit.timer
```

The observation file must itself come from the private official-source audit workflow;
the timer does not crawl the network. A current-law answer is eligible only when serving
has installed a passing, unexpired report whose generation-manifest checksum matches the
active immutable generation.

**After approval, do not run the timer while gated retrieval evals are in flight** (see
`improvement.md`): the embed stage yellows the index. The lock check protects eval runs
only if the eval session holds `coordination/locks/eval-run.lock`.

## 5. Immutable recovery and reversible promotion

Never restore a snapshot into `georgian_legal` or `georgian_legal_delta`. Those names are
read-only during hardening, and the stable serving name must ultimately be an alias. A
candidate is restored only as `georgian_legal__gen_<generation>` from an immutable,
checksummed generation whose integrity report is all-green.

Create the immutable generation locally first. The command rejects `v1`, mismatched IDs,
existing destinations, invalid checksums, incomplete accounting, and non-private output:

```bash
cd ingest
.venv/bin/python scripts/create_generation.py \
  --generation "$GENERATION_ID" \
  --manifest "$PREPARED/manifest.json" \
  --documents "$PREPARED/documents.jsonl" \
  --samples "$PREPARED/sample_checks.jsonl" \
  --source-state "$PREPARED/source_state.json" \
  --output-root "$GENERATION_OUTPUT_ROOT"
```

Persist a promotion plan without touching Qdrant:

```bash
.venv/bin/python scripts/promote_generation.py plan \
  --generation-root "$GENERATION_OUTPUT_ROOT/$GENERATION_ID" \
  --snapshot-ref "file://$CANDIDATE_SNAPSHOT" \
  --snapshot-sha256 "$CANDIDATE_SNAPSHOT_SHA256" \
  --created-by "$OPERATOR" \
  --output "$PROMOTION_PLAN"
```

Applying the plan is a separately approved maintenance action. It requires a deployment-
specific `PROMOTION_CHECKS_FACTORY=module:callable` that supplies real smoke and readiness
checks, `--apply`, and `PROMOTION_APPROVED=1`. The concrete backend restores and exactly
verifies the physical candidate, waits for optimizer green, runs both semantic checks,
atomically switches the existing alias, switches back and proves rollback. It leaves the
previous generation serving unless a second `PROMOTION_FORWARD_APPROVED=1` approval is
given with `--forward-after-rollback`:

```bash
PROMOTION_APPROVED=1 \
PROMOTION_CHECKS_FACTORY=deployment_checks:make_checks \
.venv/bin/python scripts/promote_generation.py apply \
  --plan "$PROMOTION_PLAN" --state "$PROMOTION_STATE" \
  --backend-factory ingest.qdrant_promotion:make_qdrant_promotion_backend \
  --apply
```

The current legacy deployment uses the physical serving name and therefore has no alias to
switch. The backend deliberately refuses to create that alias. First create and verify
independent legacy and candidate copies; freeing the legacy name and creating the stable
alias requires explicit maintenance approval and is not automated here. Promotion never
deletes either generation, and no artifact cleanup is part of recovery.

`scripts/verify_all_embedded.py` remains a diagnostic ID-coverage check only. It cannot
substitute for `scripts/verify_generation.py`, which checks exact document/chunk accounting,
payload identity, dense and sparse vectors, deterministic samples, freshness, and quality.

## 6. Ops gotchas (learned the hard way)

- **30 GB RAM, CPU-only box:** Qdrant + one resident model stack + desktop is the limit.
  Never co-run two model loads (second session, eval with reranker, watch embed) —
  swap-thrash takes queries from seconds to ~275 s. Coordinate via `coordination/locks/`.
- **Yellow index** = a writer is active or HNSW is indexing; hybrid queries may time out.
  Check before evals: `status "green"` (§2 curl).
- **RunPod:** API calls need a browser `User-Agent` (Cloudflare 403 error 1010 otherwise);
  after ANY pod job verify `pods=[]` — an orphaned pod bills until stopped.
- **Scraper refresh:** `seen.sqlite` stores last successful content state and a refresh
  deadline. Each run directly schedules at most 2,000 oldest-due identities (Matsne 30
  days, TAS 7 days, pending/draft 1 day), independently of its discovery window. Never
  delete this database to force a crawl; it is recovery and retry evidence. Use an
  explicitly scoped seed/backfill workflow when a particular identity needs repair.
- **`origin/dev` is a divergent fork** — never pull/merge it (`HANDOFF.md` gotchas).
- **Monitor is long-running** — restart it after editing `session_monitor.py`; only the
  HTML reloads per request.
