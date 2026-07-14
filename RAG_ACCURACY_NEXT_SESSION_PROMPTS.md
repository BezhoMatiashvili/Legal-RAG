# Georgian Legal RAG — Accuracy & Quality Prompts for Next Sessions

These prompts turn the 2026-07-12/13 end-to-end audit into independently executable
next-session tasks. Run them in order unless a prompt says it is blocked. Do not combine
multiple retrieval-changing prompts into one experiment: one measured change at a time makes
the result attributable and safely reversible.

Before every prompt, the future session must read `HANDOFF.md`, `improvement.md`,
`quality-improvement-runbook.md`, `memory-bank/INDEX.md`, `memory-bank/contracts.md`, and the
relevant `memory-bank/areas/*.md` file. The repository has a heavily modified, multi-session
working tree. Preserve unrelated work, inspect coordination/claims, never merge `origin/dev`,
never make a destructive Qdrant change without explicit approval, and never commit unless the
user authorizes it.

The mandatory rule is repeated inside every prompt: **if the attempted change does not produce
a measured accuracy or quality improvement, revert the behavior/configuration to the older
version and record the failed experiment.** A successful implementation with no quality gain is
not a reason to keep a retrieval-affecting change.

---

## Prompt 1 — Make Evaluation Reproducible and Production-Faithful

```text
You are working in the Georgia-Legal-Search repository. Your single objective for this session is
to make retrieval evaluation reproducible and faithful to the production search path before any
more model or ranking optimization is attempted.

Read HANDOFF.md, improvement.md, quality-improvement-runbook.md, memory-bank/INDEX.md,
memory-bank/contracts.md, and memory-bank/areas/eval-ops.md first. Inspect git status and active
coordination claims. Preserve all unrelated changes. Do not commit, spend GPU credits, start a
remote resource, publish a snapshot, or mutate the live collection without explicit user approval.

Evidence motivating the work:
- experiments_gpu.jsonl contains runs with the same config hash, eval hash, collection name, and
  point count but nDCG@10 varying roughly 0.134–0.450.
- eval/backend.py reimplements production behavior and can diverge from ingest/search.py.
- scripts/build_phase_c_report.py reads points/points_count while eval records n_points.
- a mode can be labeled rerank even when no reranker is loaded.

Implement the smallest safe evaluation-only changes that achieve all of the following:
1. Add a complete run manifest: git commit plus dirty-diff hash, corpus snapshot/content revision,
   collection identity, point count, Qdrant version/index settings, embedding/reranker/tokenizer
   model revisions or artifact hashes, preprocessing/chunk/header versions, translation artifact
   hash, backend/hardware, and relevant seeds.
2. Save per-query ranked IDs, scores, stage timings, retrieval mode, and a deterministic result
   hash alongside aggregate metrics.
3. Add a production-parity evaluation mode that calls the shared production retrieval code rather
   than maintaining a second ranking implementation. Add parity tests on a small deterministic
   collection or fixtures.
4. Fail loudly if rerank mode is requested without an actual reranker.
5. Fix the report index key to use n_points and reject reports that mix corpus manifests or eval
   hashes as though they were comparable.
6. Pin golden_set_v2's expected 337 records, frozen hash, v1 prefix identity, holdout superset, and
   distribution in CI.
7. Replay one baseline twice. Identical manifests must yield identical ranked IDs/order, or the
   run must fail and report the first divergence.

Do not tune retrieval in this session. Run focused tests, the full offline test suite, ruff, the
memory/symbol-map check, and the duplicate baseline replay if local resources permit. Report exact
commands, corpus revision, manifests, parity result, and any remaining nondeterminism.

HARD KEEP/REVERT RULE: if an attempted evaluator change alters production retrieval or makes
accuracy/quality results less trustworthy, less reproducible, or worse, revert to the older
implementation. Keep only evaluation changes that pass parity and reproducibility checks. Record
every failed attempt and why it was reverted. Never weaken a test or gate to obtain a pass.
```

---

## Prompt 2 — Correct Metrics and Evidence Judgments

```text
Your single objective is to correct the retrieval and context-sufficiency metrics so they measure
legal evidence rather than redundant overlapping chunks.

Read HANDOFF.md, improvement.md, quality-improvement-runbook.md, memory-bank/areas/eval-ops.md,
ingest/eval/metrics.py, spanmap.py, answer_eval.py, goldset.py, and their tests. Inspect git status
and coordination claims. Preserve unrelated work and the frozen v1/v2 data files.

Problems to address:
- current Recall@k is actually binary any-hit Success@k;
- every chunk overlapping a gold span is treated as a separate relevant item, so nDCG rewards
  redundant overlap;
- fully_grounded_at_k requires all overlapping alternative chunks instead of at least one chunk
  per evidence span;
- all 337 v2 queries currently contain one span, one document, and one grade, so multi-evidence
  completeness is not measured.

Implement evaluation semantics with backward-compatible reporting:
1. Rename the current metric to Success@k while retaining a clearly labeled legacy field only if
   historical report compatibility requires it.
2. Represent each evidence span as an equivalence group of chunks. Retrieving any member satisfies
   that span exactly once.
3. Add evidence-span Recall@k/coverage, document Precision@k, candidate Recall@50, duplicate-chunk
   rate, context precision/noise, known-item identity@1, filter compliance, and status/version
   correctness hooks.
4. Make fully-grounded mean every required evidence span is covered, not every overlapping chunk.
5. Add multi-span and multi-document synthetic unit fixtures without editing frozen golden data.
6. Preserve old experiment rows; introduce an explicit scoring-logic revision so incomparable
   metrics never share a config identity.
7. Add tests for overlap, headings, adjacent chunks, multiple acceptable documents, multiple spans,
   and zero-hit queries.

Run unit tests, full eval tests, ruff, and a small fake-backend report demonstrating old versus new
semantics. Explain which historical numbers remain comparable and which must be re-baselined.

HARD KEEP/REVERT RULE: keep the new metric implementation only if it is demonstrably more correct,
passes all fixtures, and does not silently reinterpret historical results. If accuracy/quality
measurement does not improve, or the new logic produces less valid judgments, revert to the older
metric implementation and record the failed attempt. Never change gold labels or tests merely to
make the new scoring pass.
```

---

## Prompt 3 — Deliver Complete, Structured Evidence to the Answer Composer

```text
Your single objective is to close the gap between the chunk that wins retrieval/reranking and the
evidence the MCP client actually receives.

Read the repository state/runbooks, memory-bank/areas/retrieval-serving.md,
ingest/ingest/mcp_server.py, search.py, qdrant_store.py, chunking.py, and MCP tests. Inspect git
status and active coordination claims. Preserve unrelated changes. Reconnect/restart MCP only when
the user authorizes the required serving validation.

Current failure mode: default Markdown returns only the first 700 characters of a hit, while JSON
and offline evaluation use full chunk text. legal_get_document reconstructs from chunks and cannot
faithfully restore consumed Markdown headings. There is no bounded middle-sized article/neighbor
context operation.

Implement behind a reversible knob or versioned response contract:
1. Define a structured evidence object with a stable evidence ID, source/document/chunk identity,
   title, official identifiers, status and validity dates, heading/article/page anchor, exact
   char offsets, exact evidence text, match type, retrieval mode, reranker-applied flag, and score.
2. Replace fixed 700-character prefixes with full selected evidence under an explicit total token
   budget. Do not silently cut the evidence-bearing tail.
3. Add legal_get_context (or equivalent) for a chunk plus bounded neighbors, governing article,
   or exact char range. Deduplicate overlap and enforce max_tokens.
4. Make full-document retrieval use an exact canonical body store or clearly label reconstruction
   as non-exact; preserve headings and provenance.
5. Treat retrieved document text as untrusted data, not instructions.
6. Update the production-faithful judge/context batch to use exactly the same response packing as
   MCP production.
7. Add tests for token budgets, tail evidence, headings, offsets, overlap, degraded reranker mode,
   and response compatibility.

Measure context evidence coverage and final-answer/citation quality on a fixed sample before and
after. Also measure response tokens and latency. Retrieval ranking itself should remain unchanged
unless separately gated.

HARD KEEP/REVERT RULE: if the new evidence contract does not measurably improve evidence coverage,
answer accuracy, citation quality, or context sufficiency—or if it causes an unacceptable quality
regression—restore the older response behavior/configuration and record the failed experiment.
Do not keep a larger context merely because it returns more text; it must improve measured quality.
```

---

## Prompt 4 — Make Scraping and Incremental Ingestion Freshness-Safe

```text
Your single objective is to stop stale same-ID legal documents and failed extractions from being
treated as permanently complete.

Read the runbooks plus memory-bank/areas/ingest-core.md, scraper/legal_scrapers/spiders/base.py,
pipelines.py, every affected spider, ingest/ingest/pipeline.py, hygiene.py, sources.py, snapshot.py,
and current tests. Inspect the dirty tree and coordination claims. Do not run a real scrape, ingest,
delete seen state, or mutate Qdrant without explicit user approval.

Implement safely and incrementally:
1. Replace skip-forever semantics with per-source revisit policy metadata. Prioritize mutable
   Matsne consolidated acts and other sources whose body/status/attachments change under one ID.
2. Create a retrieval-affecting document fingerprint over the cleaned body plus title, identifiers,
   dates/status, consolidation/version metadata, source URL, and attachment checksum.
3. Skip reindex only when that full fingerprint matches. Metadata-only changes must update payloads
   even if the body is unchanged.
4. Mark an identity successfully seen only after extraction and validation. Empty/failed PDF/DOCX
   or detail-page fallbacks must enter a retry/dead-letter workflow rather than becoming permanent.
5. Add last_checked_at, extraction status, retry count, source revision, and failure reason without
   losing raw artifacts.
6. Make coverage verification check fingerprint parity, contiguous chunks, vector presence, metadata
   parity, and stale-tail absence—not just existence of one point.
7. Add offline tests for a changed consolidated body, status-only change, failed-then-successful
   attachment, unchanged revisit, and retry exhaustion.

Do not perform the corpus-wide refresh in this session unless separately authorized. Deliver the
safe code path, migration/backfill plan, tests, and a dry-run report showing what would be revisited.

HARD KEEP/REVERT RULE: if the new freshness logic does not improve content/metadata accuracy or
causes missed documents, duplicate ingestion, or lower measured RAG quality, revert to the older
implementation/configuration and record the failed attempt. Keep a new revisit policy only after
parity tests and a controlled sample prove it improves freshness without accuracy regressions.
```

---

## Prompt 5 — Unify Cleaning, Extraction, and Canonical Document Preparation

```text
Your single objective is to create one canonical prepare_document path used by snapshot, batch,
watch, delta, and GPU ingestion so all writers embed identical, validated text and metadata.

Read the runbooks, memory-bank/areas/ingest-core.md, hygiene.py, snapshot.py, pipeline.py,
embed_job.py, scripts/embed_delta.py, qdrant_store.py, scraper document/Markdown utilities, and
tests. Inspect git status/coordination and preserve unrelated work. Do not rebuild or mutate the
live collection without explicit user approval.

Implement a pure, heavily tested preparation stage:
validate raw record -> normalize metadata -> assess/quarantine -> clean controls/NFC -> remove or
replace non-content -> detect legal structure -> produce canonical body/provenance -> chunk.

Requirements:
1. Every ingestion writer must call the same preparation function and produce the same content hash,
   offsets, payload metadata, chunks, and quarantine decision for the same source record.
2. Strip data-URI/base64 image payloads from indexed text while retaining bounded alt text or an
   image/formula marker. Collapse pathological underscore/dash form blanks. Preserve raw material.
3. Fetch/index authoritative TB Appeals PDF and Supreme Court DOCX bodies instead of only summaries
   when available. Preserve attachment checksum, MIME type, page count, extraction method/version,
   and confidence.
4. Add selective OCR/layout fallback for low-density or scanned PDF pages; remove repeated headers,
   footers, and file:///tmp artifacts; retain stable page anchors.
5. Validate Georgian dates using real calendar validation. The month morphology fix may already be
   present in the dirty tree; do not overwrite it. Plan a payload repair for data indexed before it.
6. Validate embedding output length equals chunk count and vectors are finite.
7. Add parity tests proving snapshot/watch/delta writers produce identical results.

Run offline tests and build a small report comparing old/new extracted text and chunks for difficult
Matsne, NAPR, Constitutional Court, TB Appeals, and Supreme Court examples. Do not re-embed the full
corpus yet.

HARD KEEP/REVERT RULE: adopt the canonical preparation path only if sampled extraction quality,
chunk validity, and downstream retrieval accuracy measurably improve with no unacceptable slice
regression. If quality/accuracy does not improve, revert the affected behavior to the older path
and record the failed experiment. Never discard raw source data, and never keep aggressive cleaning
that removes legally meaningful text.
```

---

## Prompt 6 — Repair Temporal Semantics, Version Lineage, and Citation Certainty

```text
Your single objective is to make current-law, historical as-of, version-lineage, and exact-citation
answers legally safer and unambiguous.

Read the repository runbooks, memory-bank/areas/retrieval-serving.md and ingest-core.md,
ingest/ingest/citations.py, sources.py, qdrant_store.py, search.py, mcp_server.py, relevant spiders,
reconcile_consolidated.py, and tests. Inspect the dirty tree and coordinate ownership before edits.

Implement behind reversible feature flags:
1. Normalize official identifiers. Treat all-zero, incomplete, and known placeholder registration
   codes as null/non-identifiers. Do not group unrelated acts into a version family.
2. Model explicit version/consolidation relationships rather than assuming any shared registry code
   is a lineage. Preserve provenance and ambiguity.
3. Add an as_of query/filter contract using effective and expiry intervals, with clear behavior for
   unknown dates. Distinguish publication date, adoption date, effective date, expiry date, and the
   date of the consolidated text.
4. Return currentness/version warnings in evidence and full-document responses. status=in_force must
   not by itself claim that an amendment is the current consolidated text.
5. Keep exact citation routing but separate exact identity match from semantic relevance. Return
   match_type, ambiguity count, candidate documents, and original rerank score. Do not assign a fake
   calibrated 1.0 relevance score or bypass caller filters.
6. Support multilingual identifier forms, English No./Order/Law patterns, case numbers, aliases, and
   multiple references in one query, with safe ambiguity handling.
7. Add tests for the 3,449-document all-zero-code failure class, duplicate document numbers, wrong
   year/source, current versus repealed, as-of boundaries, conflicting versions, and nonexistent IDs.

Evaluate known-item identity@1, citation precision/recall, current/as-of correctness, false pin rate,
and overall/slice retrieval metrics. Recalibrate abstention separately; a sigmoid score is not a
probability.

HARD KEEP/REVERT RULE: keep a temporal/citation change only if citation identity and current/as-of
accuracy measurably improve with no unacceptable overall or critical-slice regression. If quality
or accuracy does not improve, switch back to the older resolver/routing behavior and record the
failed experiment. Never keep a rule merely because it resolves more IDs if it resolves them to the
wrong document or version.
```

---

## Prompt 7 — Enforce Chunk Bounds and Align Reranker Context

```text
Your single objective is to fix chunk-budget violations and make the reranker judge the same legal
context that recall used.

Read all runbooks, memory-bank/areas/retrieval-serving.md and ingest-core.md, chunking.py,
structure.py, pipeline.py, qdrant_store.py, rerank.py, search.py, eval backend/metrics, and tests.
Inspect git status and claims. Do not launch a full re-embed or GPU resource without explicit user
approval.

Audit evidence: sampled production-tokenizer chunks exceeded the configured 512-token bound, with
extremes caused by base64, long form blanks/atomic strings, overlap seeding, and tail folding.
Reranking currently receives body text only, although recall embeds title/type/heading context, and
query+passage is hard-truncated at 512.

Implement as separately measurable knobs:
1. Guarantee every normal chunk obeys the configured token limit, including overlap and tail logic.
   Split token-heavy single strings safely; never duplicate atoms while folding tails.
2. Report zero-overlap cases and define bounded token-level fallback overlap where whole atoms cannot
   satisfy it.
3. Add source-aware legal boundaries: Matsne articles/subparagraphs; court/registry numbered clauses;
   issue/reasoning/holding when extractable; table rows with repeated headers; stable page/article
   anchors.
4. Build reranker text from title, official number, status/date/consolidation, heading, and clean body,
   matching the recall representation without injecting misleading metadata.
5. A/B reranker max lengths such as 512, 768, and 1024 on the same candidate pools. Measure truncation,
   latency, candidate Recall@50, post-rerank Success/nDCG, and evidence coverage.
6. Add invariant and regression tests for every failure class.

Do not combine the chunking and reranker knobs in the first experiment. Measure each independently,
then test the combination only if both independently pass. Plan any required corpus rebuild into a
new collection with alias-based rollback.

HARD KEEP/REVERT RULE: if a chunk or reranker variant does not measurably improve accuracy/quality,
revert that behavior/configuration to the older variant and record its metrics as a failed attempt.
No quality gain means no promotion, even when the code is cleaner or the context window is larger.
```

---

## Prompt 8 — Add Enforced Bilingual Rewriting and Hierarchical Retrieval

```text
Your single objective is to improve difficult paraphrase, cross-lingual, transliterated, and
known-document retrieval without replacing BGE-M3.

Read the runbooks, memory-bank/areas/retrieval-serving.md, search.py, embedding.py, rerank.py,
mcp_server.py, eval translations/backend, query logs, and tests. Inspect the dirty tree and active
claims. Use one reversible knob per experiment.

Implement and evaluate in this order:
1. Preserve the original query, normalize NFC, and classify Georgian, English, Russian, Latin
   transliteration, mixed script, identifier-only, and unknown queries using script ratios—not the
   current any-Georgian-letter shortcut.
2. Add an explicit, logged Georgian legal rewrite input/step. For non-Georgian queries retrieve the
   original and rewrite, fuse candidates, and rerank once. Do not rely only on an advisory docstring
   or an oracle lookup table keyed to golden queries.
3. Attribute misses: record whether gold was absent at candidate depth, present then demoted by the
   reranker, lost during context packing, or mishandled during answer composition.
4. Test dynamic candidate depth only for hard intents where candidate Recall@50 is insufficient.
5. Add a hierarchical path: ranked title/identifier/document metadata retrieval followed by child
   passage search inside candidate documents. Consider title and body as distinct signals; do not
   introduce an unmeasured full vector-architecture replacement.
6. Test pre-rerank per-document/source balancing separately from the already-harmful post-rerank MMR.
7. Log original query, rewrite, routing decision, candidate sources/docs, and final mode.

Evaluate on production-faithful cross-lingual, paraphrase, keyword, legal-citation, transliteration,
and mixed-script sets. Report candidate Recall@50, final Success/nDCG, identity@1, evidence coverage,
latency, and every slice. Georgian queries must remain unchanged when the rewrite route does not fire.

HARD KEEP/REVERT RULE: keep each rewrite, hierarchical, depth, or balancing variant only if measured
accuracy/quality improves under the pre-registered gate with no unacceptable slice regression. If
it does not improve, revert to the older retrieval behavior/configuration and record the failed
attempt. Do not stack failed variants or keep blanket multi-query expansion by intuition.
```

---

## Prompt 9 — Build a Blind Legal Benchmark and Actual End-to-End Answer Evaluation

```text
Your single objective is to build a legally meaningful blind benchmark and evaluate the actual MCP
tool trace plus final answer—not a hypothetical answer inferred from retrieved context.

Read the runbooks, memory-bank/areas/eval-ops.md, golden-set loaders/validators, answer_eval.py,
dump_judge_batch.py, judge_eval.py, query logs, the MCP research skill, and tests. Preserve frozen v1
and v2 byte-for-byte. Inspect git status/claims and do not send corpus data to a new external service
or API without explicit user authorization and the repository's privacy policy being satisfied.

Design separate train/dev/blind-test assets. Do not treat already tuned-on v2 as a new blind test.
Build the blind set from real legal information needs rather than selecting a document first:
1. Split by document and legal lineage so related versions cannot cross train/dev/test.
2. Cover every intended source, especially Supreme Court; Georgian, English, Russian, Latin
   transliteration, mixed script, typos, inflections, and OCR-noisy forms.
3. Include current-law, historical as-of, amendment lineage, conflicting authority, multiple
   provisions/documents, known-item, ambiguous citation, and comparison tasks.
4. Add a separate negative set: nonexistent/absent acts, ambiguous numbers, out-of-scope questions,
   future dates, and insufficient/conflicting evidence.
5. Pool candidates from BM25, dense, hybrid, citation routing, current reranker, and experimental
   rerankers. Have two Georgian legal reviewers independently grade candidates 0–3, adjudicate
   disagreements, and report agreement.
6. Store all acceptable documents, evidence spans, authority/status, applicable date, expected
   citations, answer, and abstention expectation.
7. Execute the actual MCP-first workflow with a fixed client prompt/model version. Capture tool
   calls, rewrites, retrieved/packed evidence, and the final candidate answer.
8. Grade claim-level entailment, legal correctness, completeness, citation precision/recall,
   identifier resolution, quote exactness, version/currentness, and refusal behavior. Validate that
   every batch ID receives exactly one verdict. Human-calibrate any LLM judge.
9. Use cluster bootstrap/tests by underlying legal task/document and report risk-versus-coverage for
   abstention, not a single saturated sigmoid threshold.

Deliver the benchmark specification, immutable manifests/hashes, validation tests, a small reviewed
pilot, and the end-to-end trace/eval runner. Do not silently promote the new set as the release gate
until the user approves its labels and distribution.

HARD KEEP/REVERT RULE: if an evaluation or answer-composition change does not measurably improve
legal accuracy, faithfulness, citation quality, completeness, or calibrated refusal, revert the
behavior/configuration to the older one and record the failed attempt. Never alter blind labels,
exclude failures, or weaken judge criteria to make a candidate look better.
```

---

## Prompt 10 — Close the Feedback Loop, Then Tune or Fine-Tune Models

```text
Your single objective is to create trustworthy production feedback/hard negatives and use them to
test stronger or in-domain rerankers only after Prompts 1–9 have established valid measurement and
data paths.

Read all runbooks, memory-bank/areas/eval-ops.md and retrieval-serving.md, querylog.py,
mcp_server.py, remote_search/serverless code, result caching, current eval manifests, and privacy
rules. Inspect the dirty tree and coordination claims. Do not provision GPU resources, deploy,
publish, or commit without explicit user approval.

First close observability gaps:
1. Add a local trace ID across search, lookup, context/full-document reads, version resolution, and
   the final answer.
2. Record backend, worker corpus/model revision, cache hit, fallback/degraded mode, cold start, stage
   latency, hit IDs/scores/match types, rewrite/routing, selected evidence, emitted citations,
   answerability/refusal, and outcome. Do not add unnecessary body text to logs.
3. Remote searches must return useful hit metadata to local analytics; they must not all appear as
   zero-hit rows. Log cache hits and errors explicitly.
4. Add local structured feedback categories: correct authority, wrong act/case, wrong version/status,
   wrong passage, incomplete, invalid citation, should-have-abstained, and excess personal data.
5. Define PII-aware retention/rotation and access policy.

Then build training data without contaminating the blind test:
6. Mine negatively rated/reformulated real queries and candidate near-misses into reviewed training
   triples. Exclude every blind test document/lineage.
7. Establish the unchanged stock reranker baseline on the exact same corpus manifest and candidate
   pools.
8. A/B one stronger multilingual reranker at a time. Only after a model wins should you consider
   in-domain fine-tuning with reviewed positives and hard negatives.
9. Recalibrate answerability on genuine answerable and unanswerable sets for every model. Record
   model/tokenizer artifact hashes and serving parity.
10. Promote via a reversible model/config switch, never by overwriting the known-good artifact.

Gate on overall legal-answer quality plus paraphrase, cross-lingual, keyword, citation, temporal,
source, and status/version slices; also enforce latency and false-answer/false-refusal limits.

HARD KEEP/REVERT RULE: if a model, fine-tune, feedback-derived change, or serving configuration does
not measurably improve accuracy/quality under the fixed blind gate—or regresses a critical slice—
restore the older model/configuration and record the attempt as failed. Do not keep a newer model
because it is larger, newer, faster, or more expensive; measured legal quality decides.
```

---

## Promotion Checklist Shared by All Sessions

A retrieval-affecting change is eligible for promotion only when all applicable items are true:

- The baseline and candidate use the same corpus manifest, or two explicitly matched A/B
  collections when preprocessing must differ.
- Tests, ruff, and memory/symbol-map checks pass without weakening existing tests.
- The intended primary accuracy/quality metric improves under the pre-registered gate.
- No critical source, language, query-type, status, or temporal slice regresses beyond its declared
  non-inferiority margin.
- Citation identity, current/as-of correctness, evidence coverage, false-answer rate, and
  false-refusal rate are reported where relevant.
- Latency/resource use stays within the declared budget.
- Serving fingerprint plus corpus/model revision changes when behavior changes.
- The older implementation/model/collection remains immediately recoverable.
- Failed experiments remain in the ledger with their metrics and reason for rollback.

If these conditions are not met, **do not promote the candidate: revert to the older version.**
