# Codex task: full Supreme Court (supremecourt.ge) scrape → embed on RunPod → load into Qdrant

## Goal
Ingest the **entire** Supreme Court of Georgia corpus (supremecourt.ge) into the
`georgian_legal` Qdrant collection: scrape as completely as the site allows, embed with
BGE-M3 on a RunPod GPU, merge into the main collection, and make it queryable through the
`legal_rag` MCP server. Today the `supremecourt` source has **0 points** in the index even
though the spider and 125 scraped items already exist — this task closes that gap and makes
it complete.

## Read first (do NOT skip — this repo has hard rules)
- `HANDOFF.md` (state), `memory-bank/INDEX.md` + `memory-bank/contracts.md` (architecture +
  blast radius), `coordination/README.md` (multi-session protocol — register a session,
  claim files, other sessions may be running).
- `ingest/docs/delta_embed_runbook.md` — the exact scrape→pod-embed→merge machinery you will reuse.
- Do the "pre-modification ritual" in the INDEX before editing anything under `ingest/` or `scraper/`.

### Standing guardrails (violating these breaks the repo)
- **Never `git pull`/merge `origin/dev`** — it's a divergent LangChain fork that clobbers this
  stack. Never commit unless the user explicitly asks. Never edit existing tests to make them pass.
- **RAM discipline (30 GB box, Qdrant ~21 GB resident):** run every non-rerank process with
  `RERANK_ENABLED=false`. Never co-load the cross-encoder with other work locally.
- **MCP env-at-spawn:** the `legal_rag` server caches code + `.env` at spawn; after any edit
  under `ingest/` or to `ingest/.env`, reconnect with `/mcp` (or results are stale).
- **RunPod:** the API needs a browser `User-Agent` (Cloudflare returns 403 otherwise). Check
  `clientBalance` before any paid job (was ~$10.53). **Always terminate the pod when done** —
  a leaked pod bills silently. Idempotency everywhere: point ids are `uuid5(source, document_id,
  chunk_index)`, so re-running embed/merge overwrites in place, never duplicates.

## Key facts about this source (verified in-repo)
- Spider: `scraper/legal_scrapers/spiders/supremecourt_spider.py`, name `supremecourt`.
  - Walks 3 chambers (`palata` 0/1/2 = administrative/civil/criminal) via the `/ka/getCases`
    AJAX endpoint, server-side date filter `tarigiDan`/`tarigiMde` (YYYY/MM/DD), increments
    `page` until a page returns no cases. Safety cap `MAX_PAGES=5000`.
  - `ROBOTSTXT_OBEY=False` (user-approved exception, scoped to this spider), `DOWNLOAD_DELAY=8`,
    1 concurrent request — **slow and serial by design** (site rate-limits hard).
  - Per case it fetches the full HTML page and extracts `div.case-single#modalBody` →
    `body_markdown`. It records `docx_url` (`/ka/download/{id}/{palata}`) **but never downloads
    the DOCX.**
  - Dedup key = `(case_id, chamber)`, persisted in `seen.sqlite` (resumable across runs).
- Identity (`ingest/ingest/sources.py` `SOURCES["supremecourt"]`): `id_fields=("case_id","chamber")`
  → `document_id = "{case_id}:{chamber}"`; `document_type=court_decision`; `number_fields=("case_number",)`.
  This must stay in lockstep with the spider `DEDUP_KEY` — **do not change identity fields.**
- `case_number` format is `ბს-149(კ-26)`, `ას-477-2026`, etc. (Georgian letters). The 18-digit
  numeric form (e.g. `330100122006207137`) is a **first-instance** case number, NOT a Supreme
  Court `case_number` — see success criteria.
- Existing scraped data: `artifacts/supremecourt/latest/items.jsonl` (125 items) and
  `artifacts/supremecourt/runs/*/items.jsonl` (max 418), all dated May–June 2026 only.

---

## Phase 0 — FEASIBILITY DIAGNOSIS (do this before any long crawl; report back)
Prior full-range runs (`start-1900-01-01`) still returned only 2/12/418/125 items, all from
mid-2026. Figure out **why** before committing to a multi-hour crawl:
1. Hit `/ka/getCases` directly (curl, browser UA, `X-Requested-With: XMLHttpRequest`) with an
   **old** date window (e.g. `tarigiDan=2015/01/01&tarigiMde=2016/12/31`, each `palata`) and
   confirm it actually returns older cases — i.e. the date filter and pager really walk back in
   time, rather than the endpoint only exposing recent cases.
2. Confirm pagination terminates on empty pages and isn't silently truncated by rate-limit
   429s / interrupted runs (check the earlier run logs / stats for `dedup/skipped`, HTTP
   non-200, MAX_PAGES warnings).
3. Enumerate whether there are **other document-bearing sections** of supremecourt.ge beyond
   `/getCases` (plenum/generalization decisions, older archives) worth including.

**Report:** is a complete historical scrape actually reachable through this endpoint? If the
API caps at recent cases, say so and stop — the rest of the plan can't fix a source-side limit.
Only proceed to Phase 1 once feasibility is confirmed.

## Phase 1 — Full scrape (all chambers, full history)
- Run the spider directly (NOT `run_all.py` — it intentionally excludes supremecourt from the
  production watcher). From `scraper/`:
  ```
  uv run python -m legal_scrapers.run --only supremecourt \
      --start-date 1900-01-01 --end-date <TODAY>
  ```
  (Equivalent: `scrapy crawl supremecourt -a start_date=1900-01-01 -a end_date=<TODAY>`.)
- For a guaranteed clean full pass, wipe this spider's `seen.sqlite` once at the start (or pass
  `--no-dedup`); otherwise leave dedup ON so an interrupted crawl **resumes** instead of
  restarting. Given DELAY=8 serial, expect this to take a long time — run it detached/resumable
  and monitor.
- Completeness levers to actively verify (the user asked for "everything possible"):
  - all 3 chambers covered (spider does this — confirm all three produced pages);
  - **watch for the `MAX_PAGES=5000` cap warning** in the log — if hit, coverage was truncated;
    raise the cap or split into narrower date windows and iterate until every window drains;
  - **DOCX gap:** compare a few `body_markdown` values against the actual DOCX at `docx_url`. If
    the HTML modal body is truncated/partial vs the DOCX, extend the spider to fetch + convert
    the DOCX (this touches `SupremecourtItem`, the spider, and possibly the ingest normalize
    path — do the pre-modification ritual and update `memory-bank/` in the same change). If the
    modal body is complete, note that and skip.
- **Checkpoint — report:** total unique docs scraped, the **actual earliest↔latest date span**
  captured (proves history depth), per-chamber counts, and any truncation warnings. Do not
  proceed until the scrape is confirmed complete.

## Phase 2 — Dry-run the delta (sizing)
From `ingest/` (CPU, `RERANK_ENABLED=false`):
```
.venv/bin/python scripts/embed_delta.py --source supremecourt \
    --items ../artifacts/supremecourt/latest/items.jsonl --dry-run
```
Report the unique-doc / chunk count (drives pod time/cost). Also run
`scripts/verify_all_embedded.py --sources supremecourt` to see the current 0-embedded baseline
and to confirm id-parity derivation works for this source.

## Phase 3 — Embed on a RunPod GPU → `georgian_legal_delta`
Follow `ingest/docs/delta_embed_runbook.md` (reuse the machinery verbatim):
1. Check RunPod balance first. Provision one GPU (a single 4090 is plenty for this small source):
   `.venv/bin/python scripts/runpod_orchestrate.py create`.
2. Deliver the `ingest/` tree + `artifacts/supremecourt/latest/items.jsonl` to the pod
   (`/workspace/ingest_items/supremecourt/items.jsonl`). Build the pod's BGE-M3 venv as the
   runbook / `runpod_embed_multi.sh` does (`EMBED_DEVICE=cuda`, `EMBED_USE_FP16=true`, local
   Qdrant on the pod at `127.0.0.1:6333`).
3. **G2 vector-space guardrail:** `python -m ingest embed --checksum` on the pod and assert
   cosine ≈ 1 vs `ingest/snapshots/v1/checksum_cpu.json` — the GPU vectors must live in the same
   space as the CPU-query vectors, or retrieval silently degrades.
4. Embed into a **separate** collection (never over `georgian_legal`):
   ```
   python scripts/embed_delta.py --source supremecourt \
       --items /workspace/ingest_items/supremecourt/items.jsonl \
       --collection georgian_legal_delta --batch-size 256
   ```
5. Snapshot `georgian_legal_delta`, copy it to `/workspace/out/`, download it, then
   **terminate the pod** (`scripts/runpod_orchestrate.py terminate`).

## Phase 4 — Restore locally + merge into main
```
# restore the downloaded snapshot into a SEPARATE local collection
curl -X POST 'http://localhost:6333/collections/georgian_legal_delta/snapshots/upload?priority=snapshot' \
     -F snapshot=@<downloaded>/georgian_legal_delta.snapshot
# idempotent UUIDv5 upsert into georgian_legal
.venv/bin/python scripts/merge_delta_collection.py --src georgian_legal_delta
# then drop the temp collection
curl -X DELETE http://localhost:6333/collections/georgian_legal_delta
```

## Phase 5 — Verify coverage
- `scripts/verify_all_embedded.py --sources supremecourt` → expect ~0 missing for supremecourt.
- `mcp__legal_rag__legal_collection_info` should now list a non-zero `supremecourt` point count
  (it's currently absent — total was 2,629,965 with supremecourt = 0).
- Spot-check: `legal_lookup` a known `case_number` (e.g. `ას-477-2026`) and `legal_get_document`
  its full body.

## Phase 6 — Make it queryable through the MCP server (do NOT omit)
The live MCP is running `SEARCH_BACKEND=remote` — it serves from a **published snapshot on the
RunPod volume, not local Qdrant.** Merging into local `georgian_legal` is **invisible** to it
until one of:
- **Publish** a fresh snapshot: `scripts/publish_snapshot.py` (manifest-LAST protocol — see the
  `snapshot-publish-protocol` contract; verify with the `refresh` op afterward), **or**
- set `SEARCH_BACKEND=local` in `ingest/.env` and reconnect `/mcp` to serve from local Qdrant.
Pick the one the user wants; confirm via `legal_health` fingerprint + a live `legal_search`.

## Success criteria & honesty caveat
- `supremecourt` shows a large non-zero point count in `legal_collection_info`, spanning the
  full historical date range Phase 0/1 proved reachable (not just 2026), queryable via MCP.
- **Be honest about the original motivating question:** the number `330100122006207137` is a
  first-instance case number, not a Supreme Court `case_number`. A complete supremecourt ingest
  makes it reachable **only** via full-text `contains` search, and **only if that string
  literally appears inside a decision body.** State clearly that this whole effort may still not
  surface that specific case — do not imply otherwise.

## Checkpoints (pause and report at each; don't burn GPU/time on unverified assumptions)
1. After Phase 0 — is a full historical scrape actually feasible?
2. After Phase 1 — scraped count + real date span + truncation warnings.
3. After Phase 2 dry-run — delta size + confirm RunPod balance before provisioning.
4. After Phase 4 — merged point delta (before→after).
5. After Phase 6 — live MCP query working, with the honesty caveat restated.
