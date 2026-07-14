# Qdrant ingestion (Georgian legal RAG)

A standalone batch tool that loads the scraper's JSONL output into **Qdrant** for
**hybrid (dense + sparse) retrieval** of Georgian legal documents.

It is intentionally a separate project from the scraper: it runs on **Python 3.12**
(BGE-M3 needs `torch`, which has no 3.14 wheels) and only reads the scraper's
`artifacts/<spider>/latest/items.jsonl` files — so embeddings can be (re)built without
re-crawling.

## Stack
- **Embeddings:** `BAAI/bge-m3` (1024-d dense + multilingual learned-sparse) via FlagEmbedding.
- **Vector DB:** Qdrant (local Docker), one collection with named `dense` + `sparse` vectors.
- **Retrieval:** dense + sparse fused server-side with RRF (Qdrant Query API), with
  payload filters (source / language / document_type / date range).

## Setup
```bash
cd ingest
cp .env.example .env            # adjust if needed
docker compose up -d            # start Qdrant -> http://localhost:6333/dashboard
uv sync                         # installs torch + FlagEmbedding + qdrant-client on Py 3.12
```

## Ingest (immutable candidate only)

Mutating commands never target the stable `georgian_legal` serving alias. They require
an explicit generation, its exact physical collection name, `--apply`, and an independent
approval. Recreating even a candidate collection additionally requires
`QDRANT_RECREATE_APPROVED=1`.

```bash
export GENERATION_ID=gen_20260713_candidate
export COLLECTION_NAME="georgian_legal__gen_${GENERATION_ID}"
export QDRANT_WRITE_APPROVED=1

# small candidate slice to validate end-to-end
uv run python -m ingest ingest --source ecd --limit 30 --apply

# destructive candidate recreation is a separate approval
QDRANT_RECREATE_APPROVED=1 uv run python -m ingest ingest \
  --source all --recreate --apply
```
Flags: `--source <spider|all>`, `--limit N` (cap docs), `--run latest|<run_id>`,
`--recreate` (drop+recreate an approved candidate), `--resume` (continue from checkpoint),
`--batch-size`, `--no-progress`.

A live progress panel (one row per source: `docs/total`, chunks, skipped, rate, phase)
is shown while ingesting, mirroring the scraper's panel. It quiets FlagEmbedding's tqdm
bars and the per-request `httpx` logs so the terminal stays readable. It auto-disables
when stdout is not a TTY (pipes/CI/cron) — there the original tqdm/log output is used —
and can be forced off with `--no-progress`.

Re-running is idempotent (deterministic point IDs); stale chunks of shrunken documents
are removed automatically. The per-source checkpoint only advances over **acknowledged**
writes, so a crash + `--resume` never silently skips un-written docs. `--recreate` clears
the checkpoint, and `--recreate`/`--resume` are mutually exclusive. Malformed JSONL lines
are skipped and counted, not fatal.

## Continuous watch mode
Instead of a one-shot ingest, `watch` keeps running: it first **backfills every
already-scraped document oldest→newest** (reading across *all* `artifacts/<source>/runs/*`,
not just `latest`), then **waits and ingests new documents as the scraper produces them**.

`watch` has the same immutable-generation and approval requirements. Scheduled/live
invocation remains disabled; for an explicitly approved candidate, use
`python -m ingest watch ... --apply` with the environment above.
Flags: `--source <spider|all>`, `--poll-interval N` (idle seconds between polls, default 5),
`--once` (backfill-then-exit, no waiting), `--recreate` (drop+recreate the collection and
clear watch state), `--batch-size`, `--limit` (debug cap). The global `--collection` applies.

How it stays correct and cheap:
- Progress is tracked in a **per-source byte-offset checkpoint** `ingest/.state/<source>.watch.json`
  (one offset per run file). This is **separate** from the plain `ingest` checkpoint
  `<source>.json` and never touches it — so the two commands don't corrupt each other.
- An offset only advances over an **acknowledged** (`wait=True`) upsert, so a crash or a
  Qdrant outage never skips a document; Qdrant errors are retried with backoff.
- Only newline-terminated lines are consumed, so a half-written line from an in-progress
  crawl is left for the next poll, never lost or skipped.
- `SIGINT`/`SIGTERM` trigger a graceful stop: finish the in-flight batch, do one final drain
  (so docs written just before shutdown still land), then exit.

Run **one watcher per collection** — two would race the shared state file. The first watch
run after using plain `ingest` re-embeds whatever `ingest` already wrote (idempotent, a
one-time cost); pick one path going forward (`watch` is recommended).

## Security / deployment
- Docker binds Qdrant to **127.0.0.1** only (never world-exposed by default).
- `QDRANT_API_KEY` is **required** when `QDRANT_URL` is a non-local host; the client also
  refuses to send the key over non-HTTPS. Set it in `.env` (it is also enforced on the
  local container via `QDRANT__SERVICE__API_KEY`).
- Untrusted PDF/DOCX are size/zip-ratio guarded before parsing.

## Search
```bash
uv run python -m ingest search "ადმინისტრაციული საჩივრის დაკმაყოფილება" --source ecd --top-k 5
uv run python -m ingest search "ბს-543" --document-type court_decision      # exact case no -> sparse half
uv run python -m ingest search "..." --language ka --date-from 2020-01-01 --date-to 2020-12-31
```

## MCP server (use the corpus as RAG inside Claude)
`ingest/ingest/mcp_server.py` is a stdio MCP server that wraps the same hybrid retrieval
as `ingest search`, exposing three read-only tools to an MCP client (e.g. Claude Code):
`legal_search`, `legal_get_document` (reassemble a full doc by `source` + `document_id`),
and `legal_collection_info`.

It is registered for this repo via the checked-in `.mcp.json` at the repo root, which runs
`uv run --directory ingest python -m ingest.mcp_server` — so it uses this `ingest` env and
reads `ingest/.env` for `QDRANT_URL` / `COLLECTION_NAME`. Qdrant must be running
(`docker compose up -d`); the first `legal_search` call lazily loads the ~2GB BGE-M3 model
(one-time), while `legal_collection_info` / `legal_get_document` never load it.

```bash
# boot-check / explore the tools without Claude (lists tools, lets you call them):
npx @modelcontextprotocol/inspector uv run --directory ingest python -m ingest.mcp_server
```
In Claude Code: `/mcp` should show `legal_rag` connected, then just ask a question about
the corpus and Claude will call `legal_search`.

## Tests
```bash
uv run python -m pytest          # offline: chunking, source mapping, point IDs
```

## Scaling beyond the pilot
Full corpus (tas ~500k, ecd ~44k) → millions of chunks. For the full run: keep
quantization + `on_disk`, raise `EMBED_BATCH_SIZE` (and use a GPU if available), and
consider lowering then re-raising HNSW `m` around bulk load. CPU/MPS embedding is slow
at that scale.
