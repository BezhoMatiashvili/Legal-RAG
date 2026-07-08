# Gap Audit — Georgian Legal RAG

**Date:** 2026-07-07 · **Baseline for:** the phased build in `prompt.md` (Part 0 → Part 5 + Appendix).
**Scope:** maps every mission requirement to its current status in the repo so later phases are measured against a known starting point. This is the Part 0 §7 deliverable.

**Legend:** ✓ **exists** (implemented, verified in code) · ~ **extend** (partial — code exists but must be hardened/expanded) · ✗ **build** (not present yet).

> Corpus this audit is grounded on (measured 2026-07-07 from `artifacts/`, `supremecourt` excluded):
> matsne 128,196 · napr 23,321 · ecd 18,924 · constcourt 3,076 · tas 1,344 · tbappeal 81 — **total 174,942 docs**.
> (`latest/items.jsonl` = full corpus for matsne/ecd/napr/tbappeal; constcourt & tas require the `runs/*` union — their `latest/` holds only 3 and 1,044 docs respectively.)

---

## Global guardrails (`prompt.md:42–52`)

| # | Requirement | Status | Evidence / gap |
|---|---|---|---|
| G1 | No external LLM/embedding/judge APIs — local CPU or RunPod only | ✓ | BGE-M3 (`embedding.py`), reranker (`rerank.py`), Qdrant all local; no outbound API calls in code. Wipe-pod-before-terminate is a RunPod runbook item (Part 3/4). |
| G2 | One vector space — same BGE-M3 version for corpus & query, verified by a checksum sentence | ~ | Same `BGEM3Embedder` used for both ingest and query (`__main__.py`, `search.py`, `mcp_server.py`) ✓; **no checksum-sentence guard** to catch a model/version drift → build. |
| G3 | CPU at serving, GPU only for heavy jobs (embed, LLM-judge) | ~ | Device is configurable (`EMBED_DEVICE`/`RERANK_DEVICE`, default auto→CPU); no codified policy/enforcement that serving stays CPU. |
| G4 | GPU budget = $15 (RunPod) | ✗ | No cost tracking. |
| G5 | Quality first: ~60 queries/day, p95 ≤ 10 s | ✗ | No latency SLO instrumentation. |
| G6 | Nothing ships without numbers — bootstrap CIs + paired significance; BM25 baseline in every report | ✗ | Eval prints point metrics only; no CIs, no significance test, no BM25 baseline. |
| G7 | Out of scope: ColBERT/multi-vector, HyDE/query expansion, web UI, multi-tenant auth | ✓ | Correctly absent. |

---

## Part 0 — Orientation & baseline (`prompt.md:69–81`) — **this task**

| Step | Status | Notes |
|---|---|---|
| Extract corpus + per-source counts | ✓ | 174,942 docs across 6 sources (table above). |
| `docker compose up` (Qdrant) | ✓ | `legal-qdrant` healthy on 127.0.0.1:6333/6334. |
| Fix stale `.mcp.json` path | ✓ | `/Users/bezhomatiashvili/Desktop/matsne_scraping/ingest` → this repo's `ingest/`. |
| Run tests | ~ | Ingest suite run; scraper unit suite skipped per user instruction (offline, runnable on request). |
| Pilot ingest `--limit 50` | ✓ | 6 present sources (looped — `--source all` would crash on the absent `supremecourt`). |
| Smoke-test retrieval (KA + EN) + MCP tools | ✓ | `ingest search` + 5 MCP tools. |
| Baseline eval (recall@k/MRR/nDCG, rerank off/on) | ~ | Runs end-to-end; on the 50-doc pilot it is a **plumbing** baseline only — trustworthy numbers require the full-corpus embed (Part 3). |
| Gap-audit table | ✓ | This document. |

---

## Already-built inventory (`prompt.md:19–29`) — the "exists ✓" core

| Capability | Status | Module |
|---|---|---|
| 6-source scraper + `artifacts/<src>/{runs,latest}` layout + `seen.sqlite` dedup | ✓ | `scraper/legal_scrapers/*` |
| Canonical doc + per-source normalize/field-mapping | ✓ | `ingest/ingest/sources.py` (`SOURCES`, `normalize`) |
| Heading-aware, token-bounded chunking w/ overlap + heading-path prefix | ✓ | `ingest/ingest/chunking.py` |
| BGE-M3 dense (1024-d) + learned-sparse embedding | ✓ | `ingest/ingest/embedding.py` |
| Single Qdrant collection: named dense(cosine)+sparse, int8 quant, on-disk payload, UUIDv5 idempotent upsert, payload indexes, stale-chunk cleanup | ✓ | `ingest/ingest/qdrant_store.py` |
| Hybrid search: dense+sparse server-side RRF + `build_filter` + optional rerank | ✓ | `ingest/ingest/search.py` |
| Cross-encoder rerank (`bge-reranker-v2-m3`) + min-score gate | ✓ | `ingest/ingest/rerank.py` |
| Batch ingest (idempotent, resumable checkpoint) + continuous `watch` (backfill oldest→newest across `runs/*`) | ✓ | `ingest/ingest/pipeline.py` |
| MCP server `FastMCP("legal_rag")` | ✓ | `ingest/ingest/mcp_server.py` — **5 tools**: `legal_search`, `legal_get_document`, `legal_lookup`, `legal_browse`, `legal_collection_info` |
| Retrieval eval (recall@k/MRR/nDCG, rerank on/off) + seed queries | ✓/~ | `ingest/eval/evaluate.py` + `queries.jsonl` (16 seed) — upgrade target of Part 2 |
| Per-source & field filters (source/language/document_type/court/status/date/number/parties/contains) | ✓ | `search.build_filter` |
| Local-only security: Qdrant bound to 127.0.0.1; API key required for remote; PDF/DOCX guards | ✓ | `qdrant_store.make_client`, `docker-compose.yml` |

---

## Part 1 — Corpus hygiene (`prompt.md:85–99`)

| Requirement | Status | Gap |
|---|---|---|
| `normalize()` at full scale | ~ | Per-source normalize exists; full-corpus hygiene pass not run/validated. |
| Junk / mojibake / **NUL-byte** stripping | ✗ | Not implemented. |
| Unicode **NFC** normalization | ✗ | Not applied in normalize/chunking. |
| Legal-structure extraction (`მუხლი`/headings/clauses) + coverage metric | ~ | Chunking splits on Markdown ATX headings; no legal-clause/article extraction or coverage measurement. |
| Dedup: content hash + MinHash/LSH, cluster report (no auto-delete); link amended matsne via `registration_code` | ✗ | Only scraper-level identity dedup (`seen.sqlite`); no near-dup clustering or amendment linking. |
| Corpus profile report incl. **PII presence counts** (tas/napr) | ✗ | Not implemented. |
| Versioned clean-corpus **snapshot v1** + manifest | ✗ | Not implemented. |

## Part 2 — Eval harness upgrade (`prompt.md:103–119`)

| Requirement | Status | Gap |
|---|---|---|
| Document-level **holdout** (anti-contamination) | ✗ | No holdout split. |
| **Span-anchored** golden set `(doc_id, char_start, char_end, grade∈{0,1,2})`, span→chunk at eval time | ✗ | Current gold is an identifier-filter spec resolved against the live index (`evaluate._scroll_doc_ids`); binary, doc-level. |
| Query-type tags (keyword, natural-question, paraphrase, cross-lingual EN→KA, citation) | ✗ | Not present. |
| 50–100 GE query→span pairs | ~ | 16 seed queries only. |
| Metrics: Recall@5/@10, nDCG@10, MRR@10, per-query-type, **latency p50/p95 per stage** | ~ | recall@k/MRR/nDCG@k exist at a single `k`; no @5/@10 split, no per-type breakdown, no latency. |
| **Bootstrap CIs + paired significance** | ✗ | Point metrics only. |
| **Mandatory BM25 baseline** | ✗ | No BM25 path. |
| Eval-set versioning + regression runner + persistent **experiment log** | ✗ | Not present. |
| **English (EN→KA) queries** in the set | ✗ | None in `queries.jsonl` yet. |

## Part 3 — Retrieval hardening / cross-lingual / daily ingest (`prompt.md:123–140`)

| Requirement | Status | Gap |
|---|---|---|
| Context-enriched, legal-structure-aware chunking (~450 tok/50 overlap; context prepended to embedded text only) | ~ | Chunking is 512/80 and already prepends heading-path context; needs 450/50 + legal-structure awareness + context-only-in-embedding. |
| Four runnable modes: dense-only, sparse-only, hybrid, **BM25** | ~ | Hybrid RRF exists; no explicit dense-only/sparse-only mode switch; **BM25 absent**. |
| Language-aware routing (KA→hybrid, EN→dense-heavy / drop sparse) | ✗ | No routing. |
| Reranking over top-30–50 + depth ablation (10/30/50) | ~ | Rerank exists (candidates=80); no ablation harness. |
| Diversity: max-chunks-per-doc cap + optional MMR | ✗ | Not implemented. |
| Full-document access (`legal_get_document`) | ✓ | Implemented. |
| Daily incremental `watch` + ingestion report + schema-drift alerts | ~ | `watch` exists; no daily report or drift alerting. |
| Query logging from day one | ✗ | Not implemented. |
| CPU tuning (RRF vs score fusion, prefetch depth, sparse weight, `ef`/`exact`) | ~ | Config knobs exist; no systematic tuning study. |
| **RunPod full-corpus embedding job** | ✗ | Not built (Part 3 GPU job; produces the real baseline). |

## Part 4 — Answer generation & E2E eval (`prompt.md:144–153`)

| Requirement | Status | Gap |
|---|---|---|
| Local open generation LLM (7–9B, int8/int4, model-agnostic, no external APIs) | ✗ | Not present. |
| Grounded composer — Georgian answers, cite doc_id + article/section + span per claim, full-doc retrieval, graceful refusal below confidence | ✗ | Not present. |
| LLM-as-judge (faithfulness/relevance/context precision, larger open model on RunPod) | ✗ | Not present. |
| Failure triage (retrieval vs rerank vs generation attribution) | ✗ | Not present. |

## Part 5 — MCP completion & deployment (`prompt.md:157–166`)

| Requirement | Status | Gap |
|---|---|---|
| Tool set: `search`, `ask`, `get_document`, `get_document_versions`, `ingest_status` | ~ | `search`(=`legal_search`) ✓, `get_document` ✓; **`ask`, `get_document_versions`, `ingest_status` ✗**. Bonus tools present: `legal_lookup`, `legal_browse`, `legal_collection_info`. |
| Config-hash stamping on every response | ✗ | Not implemented. |
| Request logging | ✗ | Not implemented. |
| Graceful degradation | ~ | MCP lazily loads models; non-semantic tools work without the model; no explicit degraded-mode contract. |
| Health check | ~ | Qdrant `/healthz` exists; no MCP-level health tool. |
| Deployment docs (Qdrant + models + MCP via compose/systemd, cron daily ingest) | ~ | Qdrant `docker-compose.yml` ✓; no model-serving/systemd/cron deploy docs. |
| Fix `.mcp.json` | ✓ | Done (Part 0). |
| README w/ current production config + eval scores | ~ | READMEs exist but stale: list **3** MCP tools vs **5** implemented; no eval scores. |

## Appendix — Deferred RunPod fine-tuning track (`prompt.md:170–179`)

| Requirement | Status |
|---|---|
| Re-embedding ablations (chunk 256/450/800, context prefix) | ✗ (deferred) |
| Synthetic training data (excluding holdout) | ✗ (deferred) |
| BGE-M3 embedder fine-tune (`unified_finetuning=True`) | ✗ (deferred) |
| Reranker fine-tune | ✗ (deferred) |
| Mechanical ADOPT/REJECT `compare_models` gate | ✗ (deferred) |
| Alias-swap cutover | ✗ (deferred) |

---

## Cross-cutting notes discovered during Part 0

- **`--source all` is unsafe with this corpus.** `resolve_sources("all")` returns all 7 registered sources incl. `supremecourt`, and `_cmd_ingest` re-raises on the first missing source — so `--source all` dies at `supremecourt` before reaching `tas`. Ingest the 6 present sources explicitly (loop; only the first with `--recreate`).
- **constcourt/tas full ingest needs `runs/*`, not `latest/`.** `ingest ingest` reads only `latest/items.jsonl`; for these two that is a partial slice (constcourt `latest`=3 docs, tas `latest`=1,044 of 1,344). Full ingest must use `watch` (reads all `runs/*`) or per-run `ingest --run runs/<id>`. → surface as an explicit Part 3 ingestion task.
- **`tas` ≠ Supreme Court.** `tas` = Tbilisi Architecture Service (45-field `TasItem`, PII-bearing). `supremecourt` is the excluded source.
- **Doc/code drift:** `ingest/README.md` documents 3 MCP tools; the server exposes 5. Fold into the Part 5 README refresh.
