# Codex execution prompt — production hardening and accuracy-preserving release

Use this file as the authoritative execution prompt for the Georgia Legal Search repository.

Suggested invocation:

> Follow `PROD_READINESS_REPORT.md` from start to finish. Implement every locally actionable
> production-readiness item, use the required test-and-revert loop, preserve retrieval quality,
> and stop only at a documented authorization or external-environment boundary.

## Role and objective

You are the senior engineer responsible for taking this repository from its current hardened state
to a genuinely production-ready release without reducing legal retrieval accuracy, recall,
precision, citation correctness, temporal filtering, or corpus completeness.

Work autonomously on safe local changes. Verify every claim against the current files and runtime;
do not assume that the historical test counts or findings below are still current. Continue until
all locally actionable blockers are complete and verified. If a required step needs credentials,
paid infrastructure, a live data mutation, or another authorization listed below, stop at that
boundary with an exact handoff instead of guessing or weakening the gate.

The acceptable terminal states are:

1. **Production ready:** every local and external completion criterion in this prompt passed for
   the exact release SHA; or
2. **Locally hardened, externally blocked:** all safe local work is complete, production remains
   unchanged, and the final response identifies the exact approval, command, artifact, credential,
   or result required next.

Never call the system production ready merely because unit tests pass.

## Current context to verify, not blindly trust

The previous review was performed on 2026-07-12 from base commit `00fc6a9` on `dev`. At that point:

- root scraper/launcher tests: 130 passed;
- ingest hermetic tests: 417 passed, 3 snapshot tests deselected;
- snapshot-backed tests: 3 passed;
- root and ingest Ruff, lock checks, symbol references, shell syntax, Compose rendering, and
  `git diff --check` passed;
- the observed retrieval fingerprint was `5bdd6800701dbeb6` under the live local environment;
- the tree already contained substantial modified and untracked user work;
- scheduled ingest failed closed unless `DAILY_INGEST_APPROVED=1`;
- serverless warm snapshot replacement returned `unsafe_warm_restore` instead of deleting the
  serving collection.

Previously completed hardening included cleaned/state-aware writer parity, generation-safe delta
merge, strict chunk budgets, Georgian date fixes, consolidation snapshot fields, metadata-only
change detection, retry/dead-letter failure visibility, bounded browse memory, conservative result
caching, serverless disk guards, crawl-quality signaling, owner-only artifact permissions, and
clean-CI snapshot markers. Inspect and test these invariants; do not redo or remove them casually.

## Non-negotiable safety rules

### Preserve the user's work

- Begin with `git status --short`, the current branch/SHA, and a scoped diff inventory.
- Treat every pre-existing modification and untracked file as user-owned.
- Never use `git reset --hard`, broad `git checkout --`, `git clean`, destructive restore commands,
  or any operation that can erase unrelated work.
- Before touching a dirty file, inspect its complete current diff and preserve unrelated edits.
- Make changes with small, reviewable patches. Reverse only lines introduced by your own failed
  change; never roll back the whole dirty file.
- Do not commit, push, open a PR, deploy, or alter remote state without explicit approval.

### Mandatory test-after-every-change loop

For every logical change, without exception:

1. State the defect, the invariant being protected, and the focused test that will prove it.
2. Add or strengthen a regression test first when practical. Never delete, skip, mute, loosen, or
   rewrite a valid test merely to obtain green output.
3. Make one small reversible change.
4. Immediately run the narrowest relevant test plus lint/static checks for the touched files.
5. If it fails, diagnose and fix only that logical change, then rerun it.
6. If it cannot be made green safely, reverse only that change and record the attempted fix and
   reason. Do not continue on a red baseline.
7. After completing one blocker, run its affected integration suite.
8. After all local changes, run every final gate listed below.

No untested change may remain. Do not stack multiple retrieval-affecting experiments before
measuring the first one.

### Protect retrieval quality

- Do not mutate the live serving index, `snapshots/v1`, or the frozen v1/v2 golden sets during local
  implementation.
- Do not change embedding models, model revisions, vector dimensions, tokenization, chunking,
  fusion, routing, reranking, score thresholds, filters, or corpus membership without treating the
  change as retrieval-affecting.
- Put experimental behavior behind an explicit default-off setting where feasible. Add retrieval
  fingerprint sensitivity and default-stability tests.
- Create candidate corpora as new immutable generations. Never rebuild a frozen generation in
  place.
- Compare baseline and candidate with identical corpus identity, point count, evaluation-set hash,
  model revisions, query translations, serving path, and evaluation configuration.
- Report every overall, language, query-type, latency, and abstention result. Never cherry-pick
  slices, reroll evaluations, or change a threshold after seeing results.
- Reject and reverse any retrieval-affecting candidate that fails a gate. A plausible explanation
  is not a substitute for measured quality.

Apply the G1–G5 gates in `improvement.md` exactly. At minimum they require:

- **G1:** the declared target metric reaches its predeclared improvement threshold;
- **G2:** no monitored overall nDCG@10/Recall@10, Georgian/English, or query-type metric regresses
  by more than 0.02 absolute;
- **G3:** gains below 0.02 are indeterminate and rejected unless paired comparison has `p < 0.05`;
- **G4:** rerank p50 grows no more than 10%, and non-rerank p50 remains below one second;
- **G5:** tests and Ruff pass, and the retrieval fingerprint changes only when the serving
  configuration intentionally changes.

## Authorization boundary

Local read-only inspection, editing workspace files, and isolated tests/fixtures are authorized.
Obtain explicit user approval immediately before any of the following:

- downloading dependencies or models, or using restricted network access;
- reading, changing, printing, or transmitting secrets or credentials;
- writing to live Qdrant, deleting points, regenerating/re-embedding the real corpus, or replacing
  a real snapshot;
- starting or changing RunPod/cloud/GPU resources or incurring cost;
- building or pushing a release image when it accesses external registries;
- committing, pushing, opening a PR, merging, or changing GitHub state;
- deploying, swapping a live alias, enabling scheduled ingest, changing serving configuration, or
  deleting/retiring a collection, snapshot, endpoint, or volume.

Do not evade an approval boundary. Prepare and test the local implementation, then provide the
exact safe command and expected evidence needed to continue.

## Required execution plan

Maintain a live plan and work in the following order. Mark an item complete only after its tests and
acceptance criteria pass.

### Phase 0 — establish a trustworthy baseline

1. Inventory architecture, dirty files, untracked files, current locks, ignored runtime data, and
   available local snapshots without exposing secrets.
2. Read `README.md`, `HANDOFF.md`, `improvement.md`, `memory-bank/contracts.md`, deployment
   runbooks, CI, ingestion/writer code, evaluation code, scraper pipelines, and serverless restore
   code relevant to these blockers.
3. Run the baseline gates. If an existing failure is unrelated to the planned work, diagnose and
   report it before editing; do not normalize a red baseline.
4. Confirm that scheduled ingest remains disabled/fail-closed and that no writer/eval lock is held
   before any integration test touching Qdrant.
5. Record current test counts, retrieval fingerprint, collection identity/count/status if a
   read-only local Qdrant is available, and immutable evaluation/corpus hashes.

### Phase 1 — implement a real index-integrity promotion gate

Upgrade or replace `ingest/scripts/verify_all_embedded.py`. Keep document-ID coverage visible as a
separate metric, but do not describe it as integrity verification.

The new verifier must fail closed and validate, per document/generation:

- the expected latest cleaned `document_state_hash`;
- exactly one complete contiguous chunk range `0..document_chunk_count-1`;
- no stale tail or duplicate logical chunk;
- required payload keys, types, schema/generation version, source, and document identity;
- dense vector presence and exact configured dimension;
- sparse index/value presence, equal cardinality, valid finite values, and valid indices;
- embedding model/revision, vector-space identity, chunk/header configuration, corpus generation,
  and retrieval fingerprint against an immutable manifest;
- quarantined/no-text documents as explicit reasoned exclusions, not permanent blanket exemptions;
- deterministic sampled logical checks against cleaned source text and expected chunk hashes.

Requirements:

- stream/batch the corpus; never materialize millions of points in memory;
- emit a machine-readable owner-only report, counts by failure class, and bounded samples;
- distinguish coverage, integrity, freshness, and quality; never collapse them into one green flag;
- add unit tests for missing chunks, stale tails, wrong hashes, wrong dimensions, malformed sparse
  vectors, payload drift, legacy generations, quarantines, and a valid generation;
- add a miniature disposable-Qdrant integration test if the local environment supports it without
  external mutation.

Do not set `DAILY_INGEST_APPROVED=1`; this phase only creates the prerequisite gate.

### Phase 2 — add production-parity retrieval evaluation

Add an evaluation path that exercises the same production behavior rather than duplicating an
approximation. It must cover:

- language detection/routing, including the actual English behavior;
- translated-query behavior only where production uses it;
- dense/sparse candidate generation and fusion;
- configured candidate depth;
- local or remote reranker behavior;
- the production score threshold/gate and abstention behavior;
- final formatting/deduplication only where it can affect evaluated results;
- degraded/fallback behavior as a separately reported failure mode, never silently mixed into the
  candidate score.

Record complete provenance: immutable corpus/snapshot hash, collection generation and exact point
count, exact embedding/reranker model revisions, dense dimension, tokenizer/chunk settings,
retrieval fingerprint, golden-set hash, translation hash, dependency/image identity, and Git SHA.

Add tests proving production and evaluation request construction remain equivalent. Preserve the
existing modes as explicit ablations. Do not change reranker maximum length or score thresholds in
this phase; first make their current behavior measurable.

### Phase 3 — close source-completeness and freshness gaps

Address each item as a separate tested change:

1. **Mutable Matsne/TAS refresh:** replace skip-forever identity dedup for mutable records with
   bounded refresh/state semantics. A consolidated act, draft, or decision that changes under the
   same ID must be fetched and emitted again. Do not mark an item seen until the durable feed result
   is known; do not create an unbounded full-site daily crawl.
2. **TB Appeals full decisions:** fetch and safely extract linked ruling PDFs when present. Preserve
   the article summary as metadata/fallback evidence, but never label a summary as the full ruling.
   Extraction failure must be visible and retryable rather than silently indexed as complete.
3. **Pagination reconciliation:** for ECD, NAPR, TAS, Matsne, Supreme Court, and TB Appeals where
   applicable, compare advertised totals/pages with unique IDs observed. Detect page shifts,
   duplicates, truncation, WAF bodies, callback failures, and exhausted retries. A materially
   incomplete run must exit nonzero and keep a bounded repair manifest.
4. **Binary safety:** add MIME/signature validation, bounded bytes/pages/text, time/resource
   isolation, scanned/no-text classification, truncation flags, and malformed PDF/DOCX tests. Do
   not introduce OCR or external services without approval.
5. **Artifact lifecycle:** add owner-only rotation/retention for logs, caches, failure manifests,
   and sensitive TAS artifacts without deleting evidence required for recovery.

Use recorded/synthetic fixtures with no real personal data. Do not weaken crawl politeness or
silently randomize identities to bypass site controls.

### Phase 4 — make corpus snapshots immutable and generation-safe

Implement a new snapshot generation format without changing `snapshots/v1`:

- build in a temporary sibling directory;
- write data, quarantine, dedup, provenance, and per-file checksums;
- include model revisions, vector identity, chunking/header configuration, source artifact hashes,
  document/chunk counts, consolidation fields, retrieval fingerprint, and Git SHA;
- validate every file and invariant before publishing;
- write the manifest last and atomically rename/promote the complete generation;
- refuse an existing generation name and refuse incomplete/legacy inputs unless an explicit,
  tested migration path is selected;
- retain prior immutable generations and define rollback/retention rules.

Add interrupted-build, checksum-mismatch, existing-generation, round-trip, and atomic-publication
tests. Snapshot-backed tests may read the existing local fixture, but must never rewrite v1.

### Phase 5 — implement safe serverless blue/green promotion

Replace the current cold-only operational workaround with a verified promotion design:

1. Restore a published snapshot into a uniquely named candidate physical collection.
2. Run schema, count, integrity, model/vector identity, golden-query smoke, and readiness checks on
   that candidate.
3. Keep the serving name as a Qdrant alias and atomically switch it only after verification.
4. Retain the prior physical collection and manifest as last-known-good rollback.
5. Prove rollback by switching the alias back.
6. Clean up only generations outside the retention policy and only after explicit authorization.
7. Serialize concurrent publishers and make every retry/idempotency state explicit.

Handle migration from a legacy physical collection that currently occupies the desired alias name
without deleting it before a verified candidate exists. If Qdrant API/version behavior is unclear,
inspect the installed client and official version-matched documentation; do not invent endpoints.

Add fake-client unit tests plus a disposable local-Qdrant snapshot/alias round-trip. Keep warm
replacement fail-closed until that integration test passes. Do not operate the real endpoint in
this phase without approval.

### Phase 6 — freeze deployment numerics and supply-chain inputs

Make deployment reproducible, but pin only observed/proven values:

- capture the exact CUDA torch wheel/build from the known-good deployed GPU worker;
- pin every serverless Python runtime dependency with hashes where the toolchain supports it;
- pin the Python base image by digest;
- verify the Qdrant archive with a version-specific checksum before extraction;
- add immutable Hugging Face revision settings for embedding, tokenizer, and reranker models;
- stamp these identities into health, collection/snapshot manifests, evaluation logs, and serving
  compatibility checks;
- fail startup when the query model/vector identity does not match the collection.

Prepare local code/tests without guessing the deployed CUDA version or model commits. If the
known-good worker must be queried, stop and request approval with the exact read-only command.
Build and numeric comparison against a real GPU/image are external gates.

### Phase 7 — create and evaluate a candidate corpus generation

This phase mutates substantial data and requires explicit approval before execution.

After approval:

1. Capture a baseline on the unchanged last-known-good corpus and production-parity path.
2. Build a new immutable snapshot/corpus generation; never overwrite v1 or the live collection.
3. Repair/re-embed documents affected by cleaning, dates, metadata-state hashing, consolidation,
   chunk packing, and generation semantics into a candidate collection.
4. Run the full integrity verifier and wait for green optimizer/index status.
5. Confirm baseline and candidate use the required identical evaluation identity. If corpus content
   intentionally differs, create the correct common evaluation anchor and document why historical
   scores are not directly comparable; never pretend differing corpora are a controlled A/B.
6. Run v1 historical sanity checks, the frozen v2 gate, production-parity evaluation, every slice,
   latency, abstention, and citation-offset checks.
7. Apply G1–G5 without adjustment. Reject the candidate on any gate failure and keep production
   unchanged.

### Phase 8 — build the exact release artifact and validate deployment

This phase requires explicit approval for Git, registry, cloud, and deployment actions.

1. Separate unrelated user work from the intended release without discarding it.
2. Produce reviewed, scoped commits and identify the exact candidate SHA.
3. Run clean-checkout CI on that SHA, including the hermetic suite; run snapshot and integration
   gates on the data host.
4. Build the fully pinned image and record its digest/SBOM/security scan results.
5. On the actual platform, test cold restore, candidate integrity, blue/green promotion, serving
   smoke tests, rollback, cache invalidation, disk/RAM headroom, timeouts, and failure behavior.
6. Promote only after all evidence is green.
7. Only then request explicit approval to set `DAILY_INGEST_APPROVED=1`, enable the timer, and switch
   production serving.

## Final verification gates

Run from the repository root unless a subshell path is shown. Adapt only when the project tooling
has legitimately changed, and explain any adaptation.

```bash
# Root/scraper world
uv lock --check
uv sync --frozen --dev
uv run ruff check scraper tests run_all.py
uv run pytest tests/ -q

# Ingest/RAG world
cd ingest
uv lock --check
uv sync --frozen --dev
RERANK_ENABLED=false uv run ruff check .
RERANK_ENABLED=false uv run python scripts/gen_code_map.py --check
RERANK_ENABLED=false uv run pytest -q -m "not snapshot"
uv run pytest -q -m snapshot

# Repository/deployment static gates
cd ..
git diff --check
find ingest/scripts plugins .claude -type f -name '*.sh' -print0 \
  | xargs -0 -r -n1 bash -n
docker compose --env-file ingest/.env -f ingest/docker-compose.yml config --quiet
```

Also run all newly added integration tests, the integrity verifier against the candidate, the
production-parity evaluation, image build/scan, cold restore, alias promotion, and rollback gates.
Do not claim those external gates passed when they were not executed.

## Definition of done

You may report **production ready** only when all of the following are true:

- the exact candidate Git SHA passes a clean reproducible CI run;
- root, ingest, snapshot, lint, lock, symbol, shell, Compose, whitespace, and all new integration
  gates pass;
- integrity verification proves current hashes, complete generations, payload/vector identity,
  and sampled logical correctness—not merely ID coverage;
- production-parity v2 evaluation passes every G1–G5 quality and latency condition;
- the immutable candidate generation can be cold-restored, promoted blue/green, and rolled back;
- deployment dependencies, images, binaries, and model revisions are immutable and verified;
- source completeness failures are visible, retryable, and release-gating;
- production smoke tests pass without degraded fallback or stale warnings;
- scheduled ingest remains disabled until the final explicit approval;
- the release blocker count is zero.

If any condition depends on unavailable external evidence, finish all safe local work and report
**locally hardened, externally blocked**. List each remaining blocker with:

1. why it cannot be completed locally;
2. the exact approval or artifact needed;
3. the exact next command/action;
4. expected success evidence;
5. rollback or no-mutation guarantee.

## Required final response format

Lead with the honest verdict. Then provide:

- changes made, grouped by blocker;
- focused test result after every change;
- final regression and static-gate results;
- retrieval baseline/candidate metrics and every G1–G5 decision, if executed;
- live/external actions performed, with approvals and rollback evidence;
- remaining blockers and exact next steps;
- confirmation that pre-existing user work was preserved;
- links to the main changed files and generated machine-readable reports.

Do not omit failed experiments. Do not claim accuracy improved without measured production-parity
evidence. Do not claim production readiness while any required gate remains unexecuted or red.
