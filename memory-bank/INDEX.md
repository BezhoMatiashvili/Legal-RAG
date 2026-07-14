# Georgia Legal Search — Project Memory (INDEX)

> Auto-loaded every session (with [contracts.md](contracts.md)) via CLAUDE.md `@import`.
> Drill-down files in [areas/](areas/) are opened on demand — see the ritual below.
> Anchors are linted: `python3 ingest/scripts/gen_code_map.py --check`.

## What this project is

Georgian legal search/question-answering: ~208k Georgian legal documents (matsne, napr,
ecd, constcourt, tas, tbappeal) scraped → normalized/chunked → embedded with BGE-M3
(dense + learned-sparse) → **2.64M-point Qdrant hybrid index** → served through the
**`legal_rag` MCP server**. **MCP-first architecture (2026-07-08 pivot): Claude is the
client and composes the grounded, cited answers — there is NO local generation LLM
(Part 4 cut). Retrieval quality is the whole product.** Ingest/embed/eval are 100%
local (no external APIs); PII stays in the index; GPU only as explicit RunPod batch
jobs. Two Python worlds: `scraper/` (Scrapy, py≥3.14) and `ingest/` (uv project,
py 3.11–3.13, CPU torch), glued by `run_all.py`.

## Where truth lives (document authority)

| Doc | Owns | On conflict |
|---|---|---|
| `HANDOFF.md` | session STATE: status per part, corpus numbers, next-session menu | wins on state |
| `memory-bank/` (this) | ARCHITECTURE + blast radius | wins on architecture |
| `improvement.md` | eval protocol, gated improvement queue, attempt ledger | wins on eval process |
| `prompt.md` | frozen phased spec (Parts 0–5, guardrails, RunPod runbook) | historical intent |
| `coordination/` | live multi-session protocol (gitignored) | wins on right-now |
| auto-memory (`~/.claude/...`) | machine-local incident log | promote durable facts here |

## Area map — open the matching file BEFORE editing that area

| Area | File | Covers |
|---|---|---|
| Ingest core | [areas/ingest-core.md](areas/ingest-core.md) | spiders, SourceSpec normalize, hygiene/dedup, offset chunker, embed, Qdrant upsert, delta path |
| Retrieval & serving | [areas/retrieval-serving.md](areas/retrieval-serving.md) | hybrid_search/fusion/routing, reranker (CPU/remote/fallback), MCP server tools, serverless endpoint |
| Eval & ops | [areas/eval-ops.md](areas/eval-ops.md) | golden set, eval harness/metrics/ledger, RunPod orchestration, coverage verification, monitor |
| Docs & coordination | [areas/docs-coordination.md](areas/docs-coordination.md) | knowledge layout, standing rules, configs, multi-session protocol |
| Symbol inventory | [generated/symbols.md](generated/symbols.md) | AUTO-GENERATED: every module → symbols with `file:line` + import edges |

## Top invariants (one line each — full blast radius in contracts.md)

| # | Invariant | Section |
|---|---|---|
| 1 | Point ids are deterministic UUIDv5 of (doc, chunk) — re-upserts overwrite, merges dedupe | [point-identity](contracts.md#point-identity) |
| 2 | `SourceSpec.id_fields`, spider dedup key, and coverage verifier derive the SAME id | [id-parity-triple](contracts.md#id-parity-triple) |
| 3 | Payload field names are a cross-file string contract (writer ↔ indexes ↔ filters ↔ MCP hit dict) | [payload-contract](contracts.md#payload-contract) |
| 4 | Raw-text and cleaned-text hashes are different spaces — never compare across | [content-hash-semantics](contracts.md#content-hash-semantics) |
| 5 | Chunk char offsets anchor the golden set — chunker/snapshot text changes invalidate eval | [offsets-golden-set](contracts.md#offsets-golden-set) |
| 6 | Local and RunPod rerankers must return identical scores (same model + normalization) | [reranker-score-parity](contracts.md#reranker-score-parity) |
| 7 | Every remote path has a defined fallback marker (rerank→RRF; SEARCH_BACKEND=remote contract) | [remote-fallback](contracts.md#remote-fallback) |
| 8 | Snapshot publish is manifest-LAST; remote boot must fail soft | [snapshot-publish-protocol](contracts.md#snapshot-publish-protocol) |
| 9 | `config_hash` (eval identity, knobs fold in) ≠ `retrieval_fingerprint` (serving identity) | [config-hash-vs-fingerprint](contracts.md#config-hash-vs-fingerprint) |
| 10 | MCP server caches code+.env at spawn — `/mcp` reconnect after editing `ingest/` | [env-at-spawn](contracts.md#env-at-spawn) |
| 11 | Never co-load reranker with non-rerank work on this 30 GB box (`RERANK_ENABLED=false`) | [ram-discipline](contracts.md#ram-discipline) |

## Pre-modification ritual (blast radius — do this BEFORE editing `ingest/` or `scraper/`)

1. Open the matching **areas/** file section for the module you're touching.
2. Re-read every **contracts.md** section that names any symbol you'll change
   (the invariants table above is the index into them).
3. **Grep the symbol repo-wide** for callers the docs missed
   (`generated/symbols.md` gives the module/import view).
4. Then follow `coordination/README.md`: claim the files, check others' claims.

## Memory upkeep

- **Same-session rule:** a change that touches any symbol named in memory-bank/ updates
  the touched area section + contracts.md in the SAME session.
- **Linter:** `python3 ingest/scripts/gen_code_map.py --check` — runs in the
  SessionStart hook and belongs next to pytest/ruff (improvement.md gate G5).
  Regenerate the symbol map after structural changes: `python3 ingest/scripts/gen_code_map.py`.
- **Git (user-authorized standing exception, 2026-07-09):** memory-bank/ is tracked;
  sessions MAY commit **memory-bank/-only** changes without asking — one commit per
  session, message prefix `memory:`. Code commits still require asking, always.
- **Shared-hot:** INDEX.md and contracts.md follow HANDOFF.md etiquette — announce on
  the coordination board, re-read immediately before editing, keep diffs additive.
- **Promotion:** when auto-memory MEMORY.md nears its cap, move durable hazards into
  contracts.md and leave a pointer.

## Freshness stamp (update when corpus/serving state changes — rides the re-baseline ritual)

- **Corpus:** `georgian_legal` = **2,637,645 points**; scraped universe 208,218;
  coverage verified 0 missing — 207,940 embedded, 278 excluded by design (97 empty-body,
  181 hygiene-quarantined) (2026-07-09, `ingest/scripts/verify_all_embedded.py`).
- **Serving config:** `hybrid+rerank@50`, `retrieval_fingerprint=81c807b279399098`
  (measured pre-consolidation — re-baseline before any A/B, improvement.md §8).
- **RunPod balance:** ~$10.53 (topped up; HANDOFF 2026-07-10). **Stamp updated:** 2026-07-10.
