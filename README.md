# Georgia Legal Search

Accuracy-first retrieval and evidence-validated answering over Georgian legislation and
court practice. The legacy MCP search surface remains available for research, while the
new strict `legal_ask` surface is server-owned and returns an answer only after exact
identity/version/evidence validation and selective-risk calibration. Missing private
models, stale or ambiguous evidence, and degraded stages produce clarification or
abstention—not an unverified legal answer.

```
matsne.gov.ge ──┐  scrape        normalize      chunk (512 tok,      embed (BGE-M3:
ecd.court.ge    │  (Scrapy,      (per-source    80 overlap,          1024-d dense +
constcourt.ge   ├─ seen.sqlite ─ SourceSpec, ── heading-aware ────── learned sparse) ──┐
napr.gov.ge     │  dedup)        hygiene +      offsets into                           │
tas.ge          │                quarantine)    body_markdown)                         ▼
tbappeal.court.ge│                                             Qdrant `georgian_legal`
supremecourt.ge ┘
                                                                (2.64M points, hybrid
        MCP client ◄──────── legal_rag MCP server ◄──────────── RRF + cross-encoder
                            (ask/context/search/lookup/…)        rerank + validators)
```

| | |
|---|---|
| Corpus | 208,218 scraped docs; **207,940 embedded** (97 empty-body + 181 hygiene-quarantined excluded by design) |
| Index | Qdrant collection `georgian_legal`, **2,637,645 points** (named `dense` + `sparse` vectors, int8-quantized, on-disk payload) |
| Sources | Legacy live generation: matsne, NAPR, ECD, Constitutional Court, TAS and Tbilisi Appeal. New full-generation builders also include Supreme Court; incomplete/summary-only TAS and Tbilisi Appeal records are quarantined. |
| Embeddings | `BAAI/bge-m3` — one vector space for corpus (GPU-embedded) and queries (CPU) |
| Serving config | shipped default **hybrid + rerank@80, no diversity** (`retrieval_fingerprint=06a64f548fcb4d59`); recommended interim config is rerank@50 (`81c807b279399098` — set `RERANK_CANDIDATES=50`, ~40% faster on CPU at a small quality cost) |
| Audited v2 retrieval baseline | Success@10 **58.2%**, nDCG@10 **0.483**, evidence-span coverage about **56.7%**. These are retrieval measurements, not end-to-end legal-answer accuracy. |

## Repo layout

| Path | What |
|---|---|
| `scraper/` | Scrapy spiders for the six sources (+supremecourt), per-run JSONL artifacts, cross-run dedup — see `scraper/README.md` |
| `ingest/` | everything else: normalization, hygiene, chunking, embedding, Qdrant, MCP server, eval harness — see `ingest/README.md` |
| `ingest/docs/deployment.md` | **how to run the stack** (bring-up, serving modes, daily-ingest timer, recovery) |
| `ingest/docs/accuracy-first.md` | strict answering/evidence contract, failure modes, and release gates |
| `ingest/docs/runpod-serverless.md` | scale-to-zero remote serving (GPU + Qdrant in one serverless worker) |
| `ingest/docs/delta_embed_runbook.md` | GPU path for embedding large scrape deltas |
| `ingest/eval/phase_c_report.md` | the measured Phase C report behind the serving config |
| `coordination/` | multi-session coordination protocol (gitignored; see `CLAUDE.md`) |

## Quick start

```bash
# 1) Qdrant + index (see ingest/docs/deployment.md if the collection is empty)
cd ingest && cp .env.example .env
chmod 600 .env
echo "QDRANT_API_KEY=$(openssl rand -hex 24)" >> .env   # compose refuses to start without it
docker compose up -d

# 2) search from the CLI (no MCP needed; first run downloads ~6.5 GB of models —
#    ~4.3 GB BGE-M3 full snapshot + ~2.2 GB reranker; --no-rerank skips the reranker)
uv sync && uv run python -m ingest search "შრომის კოდექსი შვებულება" --top-k 5

# 3) as an MCP server: the repo-root .mcp.json registers `legal_rag` for Claude Code.
```

MCP tools: `legal_ask` (strict answer/clarify/abstain), `legal_get_context` (exact
content-bound evidence and neighbors), `legal_search` (hybrid+rerank research results;
scores are relevance, not confidence),
`legal_get_document` (full text by id), `legal_get_document_versions` (consolidation
lineage), `legal_lookup` (exact document_number / registration_code), `legal_browse`
(faceted listing), `legal_collection_info`, `ingest_status`, `legal_health`. Search
responses (plus `ingest_status`/`legal_health`) are stamped with the
`retrieval_fingerprint` of the config that produced them, and every row of the query
log (`ingest/.state/queries.jsonl` — promotion source for golden set v2) carries it too.

## Qdrant schema (per chunk)

Legacy point IDs are UUIDv5 of `source:document_id:chunk_index`. Clean immutable
generations use `source:document_id:version_id:chunk_index`, preventing two legal versions
from overwriting one another while preserving idempotent rebuilds. Vectors:
`dense` (1024, cosine, on-disk) + `sparse`. Payload highlights — full list in
`ingest/ingest/qdrant_store.py`:

- **identity:** `generation_id`, `source`, `document_id`, `version_id`, `chunk_index`,
  `content_hash`, `passage_hash`, source/normalizer/chunker/model revisions
- **document:** `title`, `document_type`, `document_number`, `registration_code`,
  `status`, `is_consolidated`, `consolidated_count`, `date`, `in_force_date`,
  `expiry_date`, `court`, `parties`, `language`, `source_url`
- **structure/version:** `article_id`, `clause_id`, `chapter`, `heading_path`, `parent_id`,
  `supersedes`, `effective_from`/`effective_to`, consolidation/repeal status
- **chunk:** exact canonical `text`, `token_count`, character/page offsets and passage hash

Indexes: keyword on the identity/filter fields, full-text (multilingual tokenizer) on
`text`/`title`/`parties`, datetime ranges on the date fields.

## Evaluation & versioning policy

- **Golden set v1 is frozen**: `ingest/eval/golden_set_v1.jsonl` — 103 span-anchored
  query→evidence pairs over 51 holdout docs (`holdout_doc_ids.json`; those docs are
  excluded from tuning). Types: natural_question 32, cross_lingual 22, keyword 21,
  legal_citation 17, paraphrase 11. Growing the set = **additive `golden_set_v2`**,
  v1 stays the yardstick until explicitly switched.
- **Every retrieval change goes through the harness** (`ingest/eval/`): clustered CIs,
  paired permutation tests (`--compare`), BM25 full-corpus floor in every report; every
  run appends to `ingest/eval/experiments.jsonl` (the permanent record).
- **Two hashes, two meanings:** `config_hash` identifies an *eval* configuration
  (knobs + eval-set hash + scoring-logic rev); `retrieval_fingerprint` identifies the
  *serving* configuration. Don't conflate them.
- **Re-baseline rule:** eval numbers are comparable only at the same corpus state —
  record the immutable generation and `points_count` with every run; after any new
  generation, rerun both production paths twice before gating anything.
- End-to-end promotion additionally requires the severe/material error, coverage,
  mechanical-citation, determinism, slice-regression and latency gates documented in
  `ingest/docs/accuracy-first.md`.
- CI runs the hermetic ingest suite with `pytest -m "not snapshot"`; releases on the data
  host must additionally run `cd ingest && .venv/bin/python -m pytest -m snapshot -q` against
  the immutable local snapshot before any retrieval-affecting promotion.

## Operations

Deployment, serving modes (local CPU / GPU rerank pod / fully-remote serverless), the
**daily ingestion timer** (`ingest/systemd/`, scrape→embed→verify with lock-guards), and
recovery runbooks all live in **`ingest/docs/deployment.md`**. Ops dashboard:
`ingest/scripts/session_monitor.py` → http://localhost:8770 (Qdrant status, embed
coverage, pipeline tracker, query latencies).

Known constraints: 30 GB CPU-only box (one resident model stack at a time; measured CPU
rerank p50 ≈ 67 s/query @50, 114 s @80 — use the GPU/serverless path for interactive
latency);
writes turn the index yellow (hybrid queries time out until green); `origin/dev` is a
divergent fork — never pull/merge it.

## Privacy stance

Ingest, embedding, evaluation, translation, generation and verification are private-
infrastructure only—**no external LLM/embedding/judge APIs**. Model/data licenses require
an explicit commercial-use attestation before a configuration is production-eligible.
