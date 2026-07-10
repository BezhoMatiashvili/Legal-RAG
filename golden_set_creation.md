# golden_set_creation.md — Grow the golden set to 500 pairs (I5, production-quality)

ultrathink. Grow the retrieval golden set from 103 to 500 pairs (`golden_set_v2`), production-quality.

**Read first, in order:** `improvement.md` §I5 (the spec — this task IS I5), `HANDOFF.md`, memory,
`ingest/eval/goldset.py` (the loader = the contract), `ingest/eval/golden_set_v1.jsonl` (the recipe by example).

---

## HARD RULES (violating any of these poisons every future eval — no exceptions)

1. **ADDITIVE ONLY.** `golden_set_v1.jsonl` is FROZEN — never edit, reorder, or re-anchor it. v2 is a new
   file (`golden_set_v2.jsonl` = v1 pairs + new pairs, or a superset-loader — investigate which fits
   `goldset.py`). v1 stays the default gating yardstick; add a `--golden-set` flag to `eval/evaluate.py`
   (it does not exist yet) and thread it through `load_golden_set`, `load_holdout`, `eval_set_hash`, and
   `SnapshotBodies`.

2. **EXACT v1 RECIPE per pair:** span-anchored `relevance[]` with `char_start`/`char_end` into the snapshot
   `body_markdown`; `evidence_quote` byte-exact (the loader's `reground()` fails loud otherwise); graded
   relevance; gold docs added to `holdout_doc_ids.json`. Every loader invariant (reground, holdout,
   span-coverage lint) must pass on the FULL v2 file before you call anything done.

3. **GROUNDING LANDMINE — investigate before authoring:** `SnapshotBodies` reads bodies from
   `ingest/snapshots/v1/docs/*.jsonl` (184,318 docs). Documents scraped AFTER snapshot v1 are not in it.
   Extend the snapshot mechanism ADDITIVELY for new gold docs (e.g. a `snapshots/v2-delta` dir the loader
   also reads) — never regenerate or touch v1 snapshot files. If a candidate doc's `body_markdown` in the
   snapshot differs from the live payload text, drop the doc rather than patch either.

4. **NO CONTAMINATION:**
   - Sample candidate docs from corpus strata (source × document_type × date buckets, proportional with
     floors), NEVER from documents used to tune past improvements, and never by picking docs that some
     retrieval mode already ranks well (that optimizes the eval toward the current system).
   - Author each query FROM the document text (read doc → write the question a lawyer/citizen would ask),
     never from retrieval results.
   - Grep-verify that no new query string appears in `eval/query_translations_v1.json` or any alias/tuning
     artifact, and vice versa.

5. **PII:** gold docs and quotes stay local; no external APIs anywhere in the pipeline.

---

## TARGET DISTRIBUTION (500 total = 103 v1 + ~397 new; per-slice n≥80 for slice-level gating)

| Slice | Target | Notes |
|---|---|---|
| `legal_citation` / known-item | ~100 | **CRITICAL FIX vs v1: the gold doc must BE the cited act** (verify payload `document_number`/`registration_code` matches the citation in the query), not a decision merely citing it. Cover: matsne №+act-word, registration codes, ecd 18-digit case ids, constcourt N-forms, tas AR-ids, article references into consolidated base laws (`is_consolidated=True` docs now exist). |
| `cross_lingual` (EN→KA) | ~100 | Natural English legal questions; also author the KA translation entries — `eval/query_translations_v2.json`, keyed by id, loader-validated, byte-matching queries. |
| `natural_question` (KA) | ~100 | |
| `keyword` (KA) | ~80 | |
| `paraphrase` (KA) | ~60 | Currently the worst slice (0.000): paraphrases must share ZERO content-word overlap with the evidence span, verified by a token check. |
| temporal/status | ~60 | New `query_type` or fold into keyword — check `breakdown()` handles new types. Queries like „2019 წლის №71 ბრძანება ძალაშია?" that exercise date/status metadata. |

Seed from real usage where possible: `ingest/.state/queries.jsonl` (local only).

---

## PROCESS (batched, verified, resumable)

Work in batches of ~50 pairs: author → machine-validate → adversarially verify → append → run loader
invariants + `pytest tests/test_goldset.py` → commit checkpoint (ask permission once at session start).

**ADVERSARIAL VERIFICATION** (separate pass, fresh agent/subagent per batch, NOT the author):
for each pair independently check:

- (a) the evidence span actually answers the query;
- (b) the query is answerable from the doc without hindsight knowledge;
- (c) `query_type` and `query_language` labels are correct;
- (d) the quote is byte-exact at `[char_start:char_end]`;
- (e) the query is not a near-duplicate (>0.8 token overlap) of any existing v1/v2 query.

Reject-and-replace failures; log the rejection rate.

---

## ACCEPTANCE (all required before declaring done)

- Loader invariants green on v2; full `pytest` + `ruff` clean; NEW tests for the v2 loader path and the
  `--golden-set` flag (never modify existing tests).
- Distribution table (per `query_type` × `language` × `source`) reported to me.
- One hybrid-mode eval run on v2 (`RERANK_ENABLED=false`, ~5 min at n=500) logged with `--log` to prove the
  file evaluates end-to-end; compare v1-subset metrics on the v2 run vs historical v1 rows as a consistency
  check.
- **Gating STAYS on v1 until I explicitly sign off switching to v2** — state this in `HANDOFF.md`.
- `HANDOFF.md` + `improvement.md` ledger updated; everything committed (one commit per batch is fine).

**Corpus state note:** record `points_count` before starting; if my consolidation/scrape backfill lands
mid-task, finish authoring anyway (pairs are corpus-independent) but flag that baselines need re-running.

---

## Usage notes

1. Run this in a **fresh session** (fresh context = the adversarial verifier isn't the author who wrote the
   pairs — genuinely independent checking), ideally after the scrape lands so the newest documents are
   eligible as gold docs.
2. Expect **multiple sessions**. ~400 verified pairs at real quality is the largest single work item in the
   queue — the batch-of-50 + commit-per-batch structure is what makes it resumable. If a session dies at
   batch 4, nothing is lost.
3. The single most valuable requirement is the citation-slice fix (gold doc must *be* the cited act) —
   that's what unlocks re-attempting I1 citation routing and makes the I6 re-embed measurable where it
   actually helps.
