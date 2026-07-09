# Georgian Legal RAG — Phased Build Prompts

**How to use this file:** paste **one Part at a time** as a chat message. Do that Part fully,
report the metrics/artifacts named in its **STOP** gate, and wait for my approval before I paste
the next Part. The "Project reality", "The corpus", "Global guardrails", and "RunPod Secure Cloud
runbook" sections below apply to **every** Part — re-read them at the start of each Part.

---

## Project reality (applies to every Part)

This is **not a greenfield build**. The repo already implements most of the retrieval stack.
**Extend and harden the existing code — do not rebuild it.** Match its conventions: frozen
dataclasses for data models, modern type hints (`str | None`, `list[...]`), **lazy** ML imports
(so offline tests skip torch), env-driven `Config` in `ingest/ingest/config.py`, `ruff`, and
`pytest` for `ingest` / `unittest` for `scraper`.

- **Monorepo:** `scraper/` (Scrapy, Python ≥3.14) + `ingest/` (RAG stack, Python 3.11–3.13, `uv`) + `run_all.py`.
- **Already built (`ingest/ingest/`) — reuse, don't recreate:**
  - `sources.py` — `CanonicalDoc` + per-source `SourceSpec` (`normalize()` unifies the 6 sources' differing field names; Georgian month/status parsing already here). Canonical `doc_id` per source: matsne=`document_id`, ecd=`decision_document_id`, constcourt=`legal_id`, napr=`document_id`, tbappeal=`slug`, tas=`document_id`.
  - `chunking.py` — structure-aware markdown chunking, BGE-M3 tokenizer, overlap.
  - `embedding.py` — `BGEM3Embedder` (dense 1024-d + learned-sparse in one encode pass).
  - `qdrant_store.py` — one collection, named `dense`(cosine)+`sparse` vectors, int8 quant + on-disk, deterministic UUIDv5 point ids (idempotent upsert), payload indexes, stale-chunk cleanup.
  - `search.py` — `hybrid_search()` (server-side RRF prefetch fusion) + `build_filter()` + optional rerank.
  - `rerank.py` — `bge-reranker-v2-m3`, device auto-select.
  - `pipeline.py` — batch `ingest` + resumable `watch` daemon (byte-offset checkpoints, graceful drain).
  - `mcp_server.py` — `FastMCP("legal_rag")`: `legal_search`, `legal_get_document`, `legal_lookup`, `legal_browse`, `legal_collection_info`.
  - CLI: `uv run python -m ingest {ingest|watch|search}` (run from `ingest/`).
- **Eval already present and already upgraded:** `ingest/eval/evaluate.py` (recall@k / MRR / nDCG, rerank on/off) now defaults to the span-anchored golden set `ingest/eval/golden_set_v1.jsonl` (see "The eval set" below). We extend this harness, never replace it.
- **Infra:** Qdrant via `ingest/docker-compose.yml` (binds 127.0.0.1; collection `georgian_legal`). Config via `ingest/.env` (template `ingest/.env.example`: `QDRANT_URL`, `COLLECTION_NAME`, `EMBED_MODEL=BAAI/bge-m3`, `DENSE_DIM=1024`, chunk/rerank knobs).

## The corpus (applies to every Part)

- File: `artifacts_without_supremecourt.gz` at repo root — a **tar.gz** (~1.8 GB → ~13 GB). Extract to `./artifacts/` (git-ignored). Layout: `artifacts/<source>/{runs/<run_id>,latest}/items.jsonl`, one JSON doc per line. `run_id` sorts chronologically.
- **~184,575 ingestable documents** (the true corpus is the **union across all `runs/*`**, not `latest` alone). Approx per source: `matsne` ≈128k; `napr` ≈23k; `ecd` ≈19k; `constcourt` ≈5–6k; `tas` ≈1,044 in this dump (its source has ~500k); `tbappeal` ≈81.
- **Caveat — `latest` ≠ full corpus for every source:** `latest/items.jsonl` is the complete corpus for `matsne`/`ecd`/`napr`/`tbappeal`, but for **`constcourt` and `tas` the bulk lives in earlier `runs/*/items.jsonl` dirs** (constcourt `latest` holds only ~3 new docs). Use the `watch`/backfill path (reads across **all** runs) — not just `latest` — to ingest the full set.
- **Six sources (no `supremecourt` in this dump):** `matsne` (legislation — richly structured, dominant volume), `ecd` (court decisions — long bodies), `napr` (registry decisions — PDF-derived, contains NUL bytes), `constcourt` (constitutional acts — DOCX-derived), `tas` (Tbilisi architecture permits — 43 fields, PII-rich, two body fields `body_markdown`+`response_markdown`), `tbappeal` (appeals — tiny).
- **~99% Georgian (Mkhedruli).** English is a **query-side** requirement (EN query → KA doc), not a corpus split.
- **Body text = `body_markdown`** in every source; `sources.py` already maps per-source ids/dates/numbers/titles onto `CanonicalDoc`. Legal structure (მუხლი / articles, `#` headings, numbered clauses, `დადგენილება №N`) is strongest in `matsne`, weaker in court decisions.
- **Sensitivity:** `tas`/`napr` contain personal data (national IDs, DOB, addresses, phones, emails, owner names). **No redaction workstream is being built and no PII scan runs anywhere** — PII stays in the index; the local-only + no-external-API + secure-pod posture below is the safeguard.

## The eval set (already generated — applies to every Part)

Part 2's golden set already exists and is the gate for every downstream change. **Do not regenerate or wait for me to author pairs — consume these files:**

- `ingest/eval/golden_set_v1.jsonl` — **103 span-anchored Q&A pairs**, every one provably grounded: each `evidence_quote` is an exact NFC substring of the cited document's cleaned `body_markdown`, with `char_start`/`char_end` computed by string search. Fields per record: `{id, query, query_type, query_language, source, document_id, gold:{source,document_id}, relevance:[{document_id, evidence_quote, char_start, char_end, grade}], answer, doc_title, note}`. `evaluate.py` reads `query` + `gold` (resolved to live doc ids by filter); the extra fields power the span-anchored harness.
  - By source: matsne 22, tbappeal 20, napr 19, constcourt 16, ecd 14, tas 12.
  - By query type: natural_question 32, cross_lingual 22 (EN→KA), keyword 21, legal_citation 17, paraphrase 11.
- `ingest/eval/holdout_doc_ids.json` — the **51 documents** these pairs cite, reserved as the anti-contamination holdout: eval queries come only from these docs, and they are **excluded from any future synthetic training data**.

## Global guardrails (applies to every Part)

- **Privacy is absolute:** NO external LLM/embedding/judge APIs anywhere — not for generation, synthetic data, or evaluation. Only local CPU or rented **RunPod Secure Cloud** pods. For any pod touching corpus data: compressed+encrypted single-file transfer in/out, and **wipe pod volumes before terminate**. Treat any fine-tuned weights and the index as confidential.
- **GPU is required for the heavy jobs, not optional:** I have purchased **RunPod credits** specifically to accelerate this. The full-corpus embedding (Part 3), the LLM-as-judge eval (Part 4), and both fine-tunes (appendix) **must run on a RunPod Secure Cloud GPU (CUDA FP16)** — do not fall back to CPU for these except as a tiny correctness pilot. CPU remains the target only for **query-time serving**.
- **One vector space:** the **same BGE-M3 version** must produce corpus vectors (GPU job) and query vectors (CPU serving). Verify with a checksum: embed one fixed sentence in both environments and assert the vectors match before trusting any GPU-embedded index.
- **CPU at serving time:** every query-time component runs well on CPU (int8/ONNX where it helps). RunPod is for one-shot heavy jobs only; daily operations never touch it.
- **Quality first:** ~60 queries/day, p95 ≤ 10 s end-to-end is fine. Prefer the higher-quality config when quality and speed conflict.
- **Nothing ships without numbers:** every change is A/B'd through the eval harness with **bootstrap CIs + a paired significance test**; adopt only a **significant win** (or a statistical tie with a real speed/simplicity gain). A **BM25 baseline** appears in every report.
- **Incremental:** at each Part's **STOP** gate, report the named metrics/artifacts and wait for approval before continuing.
- **Engineering standards:** Python 3.12 for `ingest`, full type hints, official `qdrant-client`, env-driven config, no hardcoded credentials, every long job resumable. Tests required for: chunking edge cases (Georgian punctuation, mixed KA/EN, clause boundaries), JSON schema drift, and span→chunk mapping.
- **Out of scope (do not build):** ColBERT/multi-vector; HyDE / query-time expansion (revisit only if failure triage shows vocabulary-mismatch misses); PII redaction/scanning; web UI; multi-tenant auth.

## RunPod Secure Cloud runbook (applies to every GPU Part)

Every GPU job (Part 3 full-corpus embed, Part 4 judge, appendix fine-tunes) follows the same self-contained pattern:

1. **Provision** a Secure Cloud GPU pod (single GPU sufficient for BGE-M3 FP16; pick the cheapest that fits — budget is **~$15 of credits total**, so measure the CPU pilot's per-doc time and estimate pod-hours before launching). Credentials: a **restricted RunPod API key lives in `ingest/.env`** — never hardcode it. The RunPod REST API sits behind Cloudflare and **returns 403 error 1010 without a browser `User-Agent` header** — always send one.
2. **Transfer in:** a single compressed+encrypted file (snapshot chunks / training pairs). Decrypt on-pod only.
3. **Run** inside `tmux`: a setup script pins the exact model version, runs the job checkpointed/resumable, and writes a `DONE` marker on success.
4. **Verify** the vector-space checksum sentence (GPU vs CPU) before trusting output.
5. **Transfer out:** single compressed+encrypted result file (Qdrant-ready vectors / weights / verdict artifact).
6. **Wipe** the pod volume, then **terminate**. Log pod-hours spent against the budget.

---

## Part 0 — Orientation, data landing & honest baseline

**Goal:** get the real corpus into the existing pipeline and establish the baseline everything is measured against, plus a written gap audit.

1. Extract `artifacts_without_supremecourt.gz` → `./artifacts/`. Verify integrity; report **per-source doc counts** and total. Note: `latest/items.jsonl` line counts are the full corpus for `matsne`/`ecd`/`napr`/`tbappeal`, but **`constcourt`/`tas` need the earlier `runs/*` dirs** for their full set — count those too (expect the runs-union total near ~184,575).
2. `cd ingest && docker compose up -d` (Qdrant). Fix the **stale `.mcp.json` path** (it points at an old macOS path) to this repo.
3. Run existing tests: `scraper` (`uv run python -m unittest discover tests`) and `ingest` (`uv run python -m pytest`). Report pass/fail.
4. Pilot-ingest a small slice across all sources: `uv run python -m ingest ingest --source all --limit 50 --recreate`. Confirm it survives the real data quirks (tas 43-field records, constcourt DOCX-derived bodies, napr PDF-derived text with NUL bytes). Report chunk counts + any skips.
5. Smoke-test retrieval: one Georgian query and one **English** query via `ingest search`; then boot the MCP server (`npx @modelcontextprotocol/inspector uv run --directory ingest python -m ingest.mcp_server`) and call each tool once.
6. Run `ingest/eval/evaluate.py` against `golden_set_v1.jsonl` → record **recall@k / MRR / nDCG, rerank off vs on**. This is the **baseline** (only the holdout docs present in the pilot index will resolve; note how many of the 103 are evaluable at this stage).
7. Write a **gap-audit table** (mission requirement → exists ✓ / extend / build) to `ingest/docs/`.

**STOP — report:** per-source doc counts, test results, pilot ingest stats, baseline eval numbers (evaluable-query count noted), gap-audit table.

---

## Part 1 — Corpus hygiene, dedup, profiling & versioned snapshots (mission Phase 0)

**Goal:** turn raw scraped JSONL into a **versioned, clean, deduplicated corpus snapshot** that every downstream Part consumes.

Reuse `sources.py` (`normalize()`), the scraper's `seen.sqlite` dedup, and the content-hash / UUIDv5 id scheme in `qdrant_store.py`.

1. **Canonical validation at full scale:** run `normalize()` over every doc; skip+count malformed (missing id, empty `body_markdown`); report coverage per source.
2. **Junk/damage filters:** empty/near-empty bodies, truncation heuristics, **NUL-byte / control-char stripping** (napr PDF→text bodies contain embedded `\x00`), and **Georgian mojibake detection** (replacement chars, wrong-encoding signatures, non-Mkhedruli garbage). Quarantine list with counts — never silently drop.
3. **Unicode NFC normalization** for Mkhedruli (idempotent; round-trip test). Applied to embedded/index text; raw preserved. (This is the same NFC form the golden set's spans were grounded against.)
4. **Legal-structure extraction & coverage:** detect articles (`მუხლი`), `#` headings, numbered clauses per source; report **structure-detection coverage** (drives Part 3 chunking + citations).
5. **Deduplication:** exact via content hash; near-dup at **doc level** via MinHash/LSH. `matsne` is full of amended versions (`ცვლილების შეტანის შესახებ`) and re-publications — **report clusters for review, do NOT auto-delete**; amended versions may all remain, flagged as versions of one act (link via `registration_code`).
6. **Corpus profile report:** language split (per-doc KA/EN/mixed detection), length distribution (BGE-M3 tokens), doc-type & date distributions, per-source counts, structure coverage, dedup-cluster stats, and **PII-field presence counts** per source (counts only — no scan of values, nothing exported; PII stays in the index).
7. **Versioned clean-corpus snapshot v1:** stable `doc_id` + `content_hash` per doc; a manifest (version, timestamp, counts, config hash). Downstream Parts read snapshots only. **Verify the 51 holdout docs survive hygiene intact** (their spans must still slice back exactly) — quarantine that would drop a holdout doc is a bug to fix, not to accept.

**STOP — report:** corpus profile report, dedup cluster report, junk/quarantine counts, snapshot v1 manifest, holdout-integrity check.

---

## Part 2 — Evaluation harness upgrade (mission Phase 1)

**Goal:** an eval harness rigorous enough to gate every future change and survive any re-chunking / re-embedding / model swap. **Upgrade `ingest/eval/evaluate.py`, don't discard it. The golden set already exists (`golden_set_v1.jsonl`, 103 pairs) — build the harness around it; do NOT wait for hand-authored pairs and do NOT regenerate the set.**

1. **Document-level holdout (anti-contamination):** `ingest/eval/holdout_doc_ids.json` (51 docs) is the reserved holdout. Enforce it: eval queries come only from these docs, and they must be **excluded from any future synthetic training data**. Split by document, never by chunk.
2. **Span-anchored golden set (already in this form):** judgments are stored as `(doc_id, char_start, char_end, grade)` in `golden_set_v1.jsonl` — **never chunk_ids**. Make the harness map each span → whichever chunks overlap it under the **current** chunking config, so the set survives re-chunking. Support multiple spans per query. The query-type tags are already present: `keyword`, `natural_question`, `paraphrase`, `cross_lingual` (EN→KA), `legal_citation`.
3. **Re-grounding self-check (replaces bootstrap authoring):** on load, assert every `evidence_quote` still slices back exactly from the current cleaned body (`body[char_start:char_end] == evidence_quote` under NFC). Any drift means the hygiene/normalization changed — fail loudly with the offending ids rather than silently mis-scoring. (Authoring is done; this guards it.)
4. **Metrics:** Recall@5/@10, nDCG@10, MRR@10, **per-query-type breakdowns** (natural_question / cross_lingual / keyword / legal_citation / paraphrase), latency p50/p95 **per stage** (embed / search / rerank).
5. **Statistical rigor:** bootstrap CIs + paired significance test on per-query scores for every A/B. Decision rule: adopt only on a significant win, or a tie with a real speed/simplicity gain.
6. **BM25 baseline (mandatory):** plain BM25 over the same chunks, in every report — the neural-stack sanity floor.
7. **Synthetic eval expansion (optional, later, local/RunPod, open models only):** Georgian queries generated from held-out chunks; **always reported separately** from the human-verified `golden_set_v1`. Not needed to pass this Part.
8. **Eval-set versioning + promotion path:** `golden_set_v1` is version 1; tag every score with its eval-set version. Path forward: logged real queries (Part 3) → candidate pool → hand-verified → `golden_set_v2`.
9. **Regression runner:** one command evaluates any retrieval config (config hash) and appends to a persistent **experiment log** (config hash, metrics, CIs, eval-set version, timestamp) — every change A/B-comparable forever.

Add tests for the **span→chunk mapping** (Georgian punctuation, clause boundaries, mixed KA/EN).

**STOP — report:** the harness running against `golden_set_v1` (103 pairs, all re-grounding-checks passing), a baseline table (BM25 vs dense vs sparse vs hybrid vs +rerank) with CIs + latency + per-type breakdown, and a regression-log entry.

---

## Part 3 — Retrieval pipeline hardening, cross-lingual routing, daily ingest & query logging (mission Phase 2)

**Goal:** raise retrieval quality through **measured toggles**, make every retrieval mode runnable, harden the daily incremental path, log queries, and — on GPU — embed the full corpus. All comparisons go through the Part 2 harness with CIs.

Reuse `chunking.py`, `search.py`, `pipeline.py` (`watch`), `embedding.py`, `qdrant_store.py`.

1. **Legal-structure-aware, context-enriched chunking:** chunk along detected legal boundaries (articles/sections/clauses) with a token-window fallback (~450 tokens, 50 overlap) via the BGE-M3 tokenizer; avoid splitting a numbered clause mid-way. **Prepend context to the embedded text only** (document title + doc type + section/article path); the stored display text stays clean.
2. **Four runnable modes:** dense-only, sparse-only, hybrid, and BM25 — each selectable so the harness compares all four.
3. **Language-aware routing:** detect query language. Georgian → full hybrid (dense+sparse, server RRF). English/cross-lingual → **dense-heavy** (drop or strongly downweight the sparse branch; cross-language lexical matching is noise). Measure both against the 22 `cross_lingual` pairs.
4. **Reranking:** `bge-reranker-v2-m3` over top-30–50 fused candidates, toggleable, int8/ONNX on CPU, truncate passages to ~256 tokens if needed for latency. Ablate depth 10/30/50 (quality-first).
5. **Diversity:** max-chunks-per-document cap + optional MMR, as measured toggles (legal queries often legitimately concentrate in one act — measure, don't assume).
6. **Full-document access:** verify `legal_get_document` reassembles the complete original; any chunk can expand to its full source doc on demand.
7. **Daily incremental pipeline (core feature):** harden `watch` mode — same clean→structure→chunk→embed (CPU)→upsert flow; detect changed docs by content hash (delete stale chunks, insert new, handle amended versions); emit a **daily ingestion report** (throughput, chunk counts, queue depth so growth is visible weeks before it strains CPU); alert on **source JSON schema drift** and extraction failures. Never requires RunPod.
8. **Query logging from day one:** query text, language, retrieved ids/scores, config hash, timestamp, optional feedback → feeds the eval promotion path and future training data.
9. **CPU tuning (no re-embed needed), via the harness:** RRF vs Qdrant score fusion; prefetch depths (dense/sparse 50/100/200); sparse-branch weight for cross-lingual; Qdrant `ef` + `exact=true` vs int8 recall check (rescore) to confirm quantization costs no recall.
10. **RunPod Secure-Cloud full-corpus embedding job (REQUIRED — GPU):** one code path, two profiles — a tiny CPU pilot (FP32, small batch) only to prove correctness, then the **RunPod production run (CUDA FP16, batch 256, checkpointed/resumable)** that embeds the whole ~184,575-doc corpus, dense+sparse together, Qdrant-ready output. Self-contained per the RunPod runbook (setup script, tmux launch, `DONE` marker, encrypted single-file transfer, wipe-and-terminate). **Verify CPU-vs-GPU vector identity via the checksum sentence** before loading the GPU-embedded vectors into the serving index. *(Chunk-size / context-prefix ablations that require re-embedding are deferred to the appendix.)*

**STOP — report:** harness comparison across modes + routing + rerank depths + diversity + Qdrant recall settings (each with CIs/latency), the full-corpus GPU embed completed with vector-identity check + pod-hours spent, one recommended interim production config (config hash), and a sample daily-ingestion report.

---

## Part 4 — Answer generation, end-to-end eval & failure triage (mission Phase 5)

**Goal:** the real greenfield — grounded Georgian answers with citations, evaluated end-to-end. Reuse `search.py` retrieval; add an `ask` path.

1. **Generation model:** a local open LLM served on the CPU server (7–9B class, int8/int4 via llama.cpp or similar), the strongest Georgian-capable open model that fits the latency budget. Make the composer **model-agnostic** so it can be swapped later. NO external APIs.
2. **Grounded composer:** answers in Georgian (or the query's language); **cite doc_id + article/section + span for every claim**; offer full-document retrieval; **refuse gracefully below a retrieval-confidence threshold** (in legal contexts, a wrong answer is worse than none). The golden set's `answer` field gives a reference answer per query for spot-checking faithfulness.
3. **End-to-end eval (LLM-as-judge on GPU — REQUIRED):** judge faithfulness, answer relevance, and context precision on a fixed question set drawn from `golden_set_v1`. The judge is a **larger open model on a RunPod Secure Cloud GPU during eval sessions** (never the generation model, never an external API) — run it per the RunPod runbook. Calibrate judge verdicts against my manual spot-checks. Results go into the same experiment log.
4. **Failure triage:** automatic per-stage attribution of eval failures (retrieval miss vs rerank miss vs generation miss) into a report that says where quality is lost — this report decides where optimization effort goes next.

**STOP — report:** sample grounded answers (a Georgian and an English query), a refusal-behavior demo, end-to-end judge scores (with pod-hours + calibration notes), and the failure-triage report.

---

## Part 5 — MCP server completion & deployment (mission Phase 6)

**Goal:** finish the delivery layer and make the whole stack runnable and operable. Reuse/extend `mcp_server.py`.

1. **Tool set:** expose (rename/add as needed) — `search` (ranked chunks + citations + scores), `ask` (grounded Georgian answer + citations), `get_document` (full original by doc_id), `get_document_versions` (amendment chain from Part 1 dedup / `registration_code`, if detected), `ingest_status` (last daily-ingestion report).
2. **Production concerns:** request logging (feeds the query log), **config-hash stamping on every response** (audit trail — any answer traceable to the exact pipeline version), graceful degradation (reranker or generator down → retrieval-only), and a health check.
3. **Deployment docs:** how to run the whole stack (Qdrant + models + MCP server) on a Linux server via docker-compose/systemd; resource requirements; the daily-ingestion schedule (cron / systemd timer). Fix `.mcp.json`; add an app Dockerfile if useful.
4. **README:** architecture, local-vs-RunPod split with data-transfer + wipe procedures, Qdrant schema, holdout + eval-versioning policy, daily-ingestion runbook, MCP tool reference, and the **current production config with its eval scores**.

**STOP — report:** MCP tools demoed via the inspector, health-check + degradation demo, deployment docs, and README showing the current production config + eval numbers.

---

## Appendix — Deferred track (RunPod Secure Cloud GPU): optimization ablations + fine-tuning

Run these only once the retrieval + generation baseline and the eval harness are solid and RunPod budget remains. Everything here is gated by the Part 2 harness and runs on **GPU per the RunPod runbook** — CPU is not a fallback for training.

- **Re-embedding optimization ablations** on ~50K-chunk samples (GPU): chunk size 256 / 450 / 800; context prefix none / title+section-path / full 1-sentence LLM enrichment (bulk-generated on-pod with an open model). Adopt only significant winners.
- **Synthetic training data (on-pod, open models only):** strong open LLM on the pod generates Georgian (query, positive-span) pairs from the corpus **excluding all 51 holdout docs**; diverse query types incl. citation-style and ~10–15% English cross-lingual; quality-filtered; hard negatives mined from the current index.
- **Embedder fine-tune (RunPod Secure Cloud GPU, CUDA FP16):** FlagEmbedding BGE-M3, contrastive loss, `unified_finetuning=True` (dense **and** sparse heads together), iterative hard-negative re-mining rounds until eval gains lose significance.
- **Reranker fine-tune (RunPod Secure Cloud GPU, CUDA FP16):** same pairs/negatives, cross-encoder training of `bge-reranker-v2-m3`.
- **Mechanically-enforced adoption gate:** a `compare_models` command runs stock vs fine-tuned through the identical eval pipeline on the human-verified `golden_set_v1` holdout and emits a verdict artifact (per-query-type metrics, CIs, paired p-values, explicit **ADOPT/REJECT**; refuses ADOPT if any query type regresses significantly). The re-embed and cutover scripts **require this ADOPT artifact as an input** and refuse to run without it.
- **Cutover:** full re-embed as a planned RunPod GPU job → new Qdrant collection → **alias swap** (zero-downtime) with documented rollback. Fine-tuned weights are confidential and must stay CPU-runnable for serving. Verify the vector-space checksum on the new weights before the swap.
