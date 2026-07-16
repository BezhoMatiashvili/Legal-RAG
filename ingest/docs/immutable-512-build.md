# Immutable 512-token candidate build

This is the operator contract for the later production build:

`attested unique snapshot -> physical candidate embed -> sealed generation preparation -> immutable publication -> physical verification -> repeated production evaluation`

Every command in this document is marked **NOT EXECUTED**. None was run while this
contract was written. The generation artifact schema remains **2**. Retrieval behavior is
bound by retrieval fingerprint revision **2**, which deliberately excludes generation,
collection, and alias names; those storage and access identities are recorded separately in
the point payloads, sealed preparation and publication provenance, verification report,
evaluation provenance, and cache key.

## Release blockers

Do not start the build until every item below exists and has been independently reviewed:

- A strict source-state evidence JSON file selecting exact successful run IDs for all seven
  sources: `matsne`, `napr`, `ecd`, `constcourt`, `supremecourt`, `tas`, and `tbappeal`.
  Each entry must bind the exact `items.jsonl` and completion-record SHA-256. Each completion
  record must attest a successful outcome, quality pass, durable feed, zero failures, and a
  completion timestamp. Current historical `run.json` files generally do not provide all of
  this evidence, so the full build is blocked until new qualifying crawls produce the immutable
  seven-source ledger. Historical startup-only records must never be edited, completed, or
  retrofitted into attestations.
  The JSON is accepted only with its hidden permanent candidate and authorization companion
  in the same directory. These three files are one evidence unit: build them at their final
  reviewed path and never copy, move, archive, or approve the JSON by itself.
- Immutable lowercase hexadecimal revisions for the embedding model, tokenizer, and
  reranker. Names or revisions must never be inferred from a cache, branch, tag, or default.
- A production dependency lock whose requirements are exact and hash-locked, plus the
  validated, configured runtime-identity JSON that matches it. The checked-in
  `serverless/runtime-identity.unconfigured.json` is intentionally invalid for this build.
- The final OCI image digest in `sha256:<64 lowercase hex>` form. A tag, intermediate image,
  or locally guessed digest is not acceptable.
- An explicit actor and run ID, Qdrant credentials for the intended deployment, enough local
  model cache to run offline, and a reviewed release configuration containing every retrieval
  knob. The same configuration must be used for embedding, preparation, verification, and
  evaluation.

The source evidence has this exact top-level shape (one or more exact runs may be selected
per source):

```json
{
  "schema_version": 1,
  "runs": [
    {
      "source": "matsne",
      "run_id": "EXACT_RUN_ID",
      "items_sha256": "64_lowercase_hex",
      "completion_record_sha256": "64_lowercase_hex"
    }
  ]
}
```

## 0. Produce and review future crawl evidence

This section describes future crawler operations only. It was not executed while this contract
was written. Attested feeds are local filesystem outputs: a remote feed URI cannot provide the
required regular-file, fsync, byte-size, and SHA-256 proof. Use a fresh date window agreed by the
release operator, preserve each exact run directory, and never use `latest` as a selection. Use
the scraper-only commands below, not the repository-root `run_all.py` launcher, which also starts
the ingest watcher and is outside this evidence-building step.

From the repository root, set the reviewed crawl window once in the operator shell.

**NOT EXECUTED — reviewed crawl window**

```bash
export CRAWL_START_DATE=REPLACE_WITH_YYYY_MM_DD
export CRAWL_END_DATE=REPLACE_WITH_YYYY_MM_DD
```

The six ordinary sources require a natural `finished` close. Run them without the Supreme Court
time budget.

**NOT EXECUTED — six ordinary sources with the combined scraper runner**

```bash
cd scraper

uv run python -m legal_scrapers.run \
  --only matsne ecd constcourt napr tas tbappeal \
  --start-date "$CRAWL_START_DATE" \
  --end-date "$CRAWL_END_DATE"

cd ..
```

Supreme Court evidence is accepted only from the strict partial workflow with an exact
14,400-second limit. A timeout reason and `partial_by_design=true` do not establish success by
themselves: the completion attester must also record the unchanged strict validator proof,
matching run/items/manifest/journal hashes, and zero unresolved failures.

**NOT EXECUTED — strict 14,400-second Supreme Court crawl**

```bash
cd scraper

uv run python -m legal_scrapers.run \
  --only supremecourt \
  --start-date "$CRAWL_START_DATE" \
  --end-date "$CRAWL_END_DATE" \
  --max-runtime-seconds 14400

cd ..
```

The supported single-spider equivalents use the same settings and completion extension. Stop
immediately if any command exits nonzero; do not select that run or continue as if the set were
complete. A zero Scrapy process exit is not acceptance evidence by itself: select the run only
when its exact run-scoped `run.json` has `outcome="success"` and the evidence builder accepts it.

**NOT EXECUTED — ordinary single-spider equivalents**

```bash
cd scraper
set -e

uv run scrapy crawl matsne -a start_date="$CRAWL_START_DATE" -a end_date="$CRAWL_END_DATE"
uv run scrapy crawl ecd -a start_date="$CRAWL_START_DATE" -a end_date="$CRAWL_END_DATE"
uv run scrapy crawl constcourt -a start_date="$CRAWL_START_DATE" -a end_date="$CRAWL_END_DATE"
uv run scrapy crawl napr -a start_date="$CRAWL_START_DATE" -a end_date="$CRAWL_END_DATE"
uv run scrapy crawl tas -a start_date="$CRAWL_START_DATE" -a end_date="$CRAWL_END_DATE"
uv run scrapy crawl tbappeal -a start_date="$CRAWL_START_DATE" -a end_date="$CRAWL_END_DATE"

cd ..
```

**NOT EXECUTED — strict Supreme Court single-spider equivalent**

```bash
cd scraper

uv run scrapy crawl supremecourt \
  -a start_date="$CRAWL_START_DATE" \
  -a end_date="$CRAWL_END_DATE" \
  -s CLOSESPIDER_TIMEOUT=14400

cd ..
```

After all seven exact run-scoped completion records exist, build the schema-v1 ledger once at a
new path. The builder is create-only and rejects omitted sources, unsafe identities, `latest`,
startup-only or failed records, changed files, and any destination that already exists. Multiple
exact `--select SOURCE:RUN_ID` arguments for one source are allowed when the reviewed release
intentionally needs more than one run.

**NOT EXECUTED — create-only seven-source evidence selection**

```bash
cd ingest

export SOURCE_STATE_EVIDENCE=/secure/release-inputs/source-state-evidence.json

.venv/bin/python scripts/build_source_state_evidence.py \
  --artifacts-root ../artifacts \
  --output "$SOURCE_STATE_EVIDENCE" \
  --select matsne:EXACT_MATSNE_RUN_ID \
  --select napr:EXACT_NAPR_RUN_ID \
  --select ecd:EXACT_ECD_RUN_ID \
  --select constcourt:EXACT_CONSTCOURT_RUN_ID \
  --select supremecourt:EXACT_SUPREMECOURT_RUN_ID \
  --select tas:EXACT_TAS_RUN_ID \
  --select tbappeal:EXACT_TBAPPEAL_RUN_ID

cd ..
```

The builder's `AWAITING INDEPENDENT OPERATOR REVIEW` warning is a required handoff, not a
success approval. A different operator must inspect the seven explicit identities, compare both
hashes for every selected immutable run, confirm all three evidence files are mode `0600`, verify
that the JSON and permanent candidate are the same inode, inspect the authorization binding, and
record all three exact hashes in the release ticket. Do not review or approve a moving copy, and
never copy or move the JSON without its same-directory companions.

**NOT EXECUTED — independent operator review of the exact ledger**

```bash
cd ingest

SOURCE_STATE_LEAF_SHA256="$(printf '%s' "$(basename -- "$SOURCE_STATE_EVIDENCE")" | sha256sum | cut -d' ' -f1)"
SOURCE_STATE_CANDIDATE="$(dirname -- "$SOURCE_STATE_EVIDENCE")/.source-state-${SOURCE_STATE_LEAF_SHA256}.candidate"
SOURCE_STATE_AUTHORIZATION="$(dirname -- "$SOURCE_STATE_EVIDENCE")/.source-state-${SOURCE_STATE_LEAF_SHA256}.authorization.json"

stat --format='%a %d:%i %n' \
  "$SOURCE_STATE_EVIDENCE" \
  "$SOURCE_STATE_CANDIDATE" \
  "$SOURCE_STATE_AUTHORIZATION"
test "$(stat --format='%d:%i' "$SOURCE_STATE_EVIDENCE")" = \
  "$(stat --format='%d:%i' "$SOURCE_STATE_CANDIDATE")"
.venv/bin/python -m json.tool "$SOURCE_STATE_EVIDENCE"
.venv/bin/python -m json.tool "$SOURCE_STATE_AUTHORIZATION"
sha256sum \
  "$SOURCE_STATE_EVIDENCE" \
  "$SOURCE_STATE_CANDIDATE" \
  "$SOURCE_STATE_AUTHORIZATION"

export REVIEWED_SOURCE_STATE_EVIDENCE_SHA256=REPLACE_WITH_REVIEWED_64_LOWERCASE_HEX
test "$(sha256sum "$SOURCE_STATE_EVIDENCE" | cut -d' ' -f1)" = \
  "$REVIEWED_SOURCE_STATE_EVIDENCE_SHA256"

cd ..
```

## Pin one release identity

Set these values once in a clean operator shell. The example candidate IDs are explicit and
may be replaced only with another reviewed, non-legacy ID; never use `latest`. Paths under
`/secure/release-inputs` represent external attested inputs that do not exist in this
repository today.

**NOT EXECUTED — release identity and reviewed configuration**

```bash
cd ingest

export SNAPSHOT_ID=v3_512_attested_20260715_01
export GENERATION_ID=v3_512_candidate_20260715_01
export PHYSICAL_COLLECTION="georgian_legal__gen_${GENERATION_ID}"

export SNAPSHOT_OUTPUT_ROOT="$PWD/snapshots/v3"
export SNAPSHOT_DIR="$SNAPSHOT_OUTPUT_ROOT/$SNAPSHOT_ID"
export SOURCE_STATE_EVIDENCE=/secure/release-inputs/source-state-evidence.json
export REVIEWED_SOURCE_STATE_EVIDENCE_SHA256=REPLACE_WITH_REVIEWED_64_LOWERCASE_HEX

export PREPARED_DIR="$PWD/.state/v3/prepared/$GENERATION_ID"
export GENERATION_OUTPUT_ROOT="$PWD/../artifacts/generations"
export GENERATION_DIR="$GENERATION_OUTPUT_ROOT/$GENERATION_ID"
export CHECKSUM_OUTPUT="$PWD/.state/embed/$GENERATION_ID/vector-space-checksum.json"
export EVAL_LOG="$PWD/.state/v3/evaluation/${GENERATION_ID}.jsonl"

export DEPENDENCY_LOCK=/secure/release-inputs/requirements-production.lock
export RUNTIME_IDENTITY=/secure/release-inputs/runtime-identity.configured.json
export FINAL_IMAGE_DIGEST=sha256:REPLACE_WITH_64_LOWERCASE_HEX
export PREPARATION_ACTOR=REPLACE_WITH_ATTESTED_ACTOR
export PREPARATION_RUN_ID=prepare-v3-512-20260715-01

export EMBED_MODEL=BAAI/bge-m3
export TOKENIZER_MODEL=BAAI/bge-m3
export RERANK_MODEL=BAAI/bge-reranker-v2-m3
export EMBED_REVISION=REPLACE_WITH_IMMUTABLE_LOWERCASE_HEX
export TOKENIZER_REVISION=REPLACE_WITH_IMMUTABLE_LOWERCASE_HEX
export RERANK_REVISION=REPLACE_WITH_IMMUTABLE_LOWERCASE_HEX

export DENSE_DIM=1024
export CHUNK_TOKENS=512
export CHUNK_OVERLAP=80
export CHUNK_MIN_TOKENS=64

# These must be the reviewed release values; preparation rejects drift from the points.
export RERANK_ENABLED=REPLACE_WITH_TRUE_OR_FALSE
export RERANK_CANDIDATES=REPLACE_WITH_REVIEWED_INTEGER
export RERANK_MIN_SCORE=REPLACE_WITH_REVIEWED_NUMBER
export RERANK_BACKEND=REPLACE_WITH_REVIEWED_BACKEND
export RERANK_CONTEXT_ENRICHED=REPLACE_WITH_TRUE_OR_FALSE
export RERANK_MAX_LENGTH=REPLACE_WITH_REVIEWED_INTEGER
export CITATION_ROUTE=REPLACE_WITH_REVIEWED_ROUTE
export EMBED_HEADER_V2=REPLACE_WITH_TRUE_OR_FALSE

export COLLECTION_NAME="$PHYSICAL_COLLECTION"
export GENERATION_DIR
export PRODUCTION_MODE=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for required_name in \
  FINAL_IMAGE_DIGEST PREPARATION_ACTOR \
  EMBED_REVISION TOKENIZER_REVISION RERANK_REVISION \
  RERANK_ENABLED RERANK_CANDIDATES RERANK_MIN_SCORE RERANK_BACKEND \
  RERANK_CONTEXT_ENRICHED RERANK_MAX_LENGTH CITATION_ROUTE EMBED_HEADER_V2
do
  required_value="${!required_name:-}"
  if [[ -z "$required_value" || "$required_value" == *REPLACE_WITH_* ]]; then
    printf 'unconfigured release value: %s\n' "$required_name" >&2
    exit 2
  fi
done
```

Replace every `REPLACE_WITH_...` value before running anything. Do not allow `.env` to
silently supply a different value. Record the final environment through the validated runtime
identity and preparation provenance.

## 1. Build the attested snapshot

Production snapshotting requires all seven sources and the evidence ledger. It has no
`--preflight` or `--limit`, retains the near-duplicate pass, and uses an immutable tokenizer
revision for positive token sampling. The destination is create-only and intentionally outside
the preflight-only `.state/v3/snapshots` tree and outside frozen `snapshots/v1`.

**NOT EXECUTED — attested full snapshot from the exact reviewed evidence file**

```bash
test "$(sha256sum "$SOURCE_STATE_EVIDENCE" | cut -d' ' -f1)" = \
  "$REVIEWED_SOURCE_STATE_EVIDENCE_SHA256"

.venv/bin/python -m ingest snapshot \
  --snapshot-id "$SNAPSHOT_ID" \
  --output-root "$SNAPSHOT_OUTPUT_ROOT" \
  --source-state-evidence "$SOURCE_STATE_EVIDENCE" \
  --source all \
  --token-sample 2000
```

The sealed manifest must list all seven sources, including Supreme Court, and bind the exact
run inventory, build knobs, every source file's size and SHA-256, `corpus_sha256`, and
`snapshot_sha256`. Incomplete TAS and Tbilisi Appeal records must remain in snapshot quarantine;
they are not admissible generation documents. Do not proceed if `snapshots/v1` changed.

## 2. Embed only the physical candidate

The checksum file is explicit, create-only, and outside the snapshot. It is useful for
comparing the exact vector space across approved compute environments and cannot authorize a
write by itself.

**NOT EXECUTED — vector-space checksum**

```bash
.venv/bin/python -m ingest --collection "$PHYSICAL_COLLECTION" embed \
  --snapshot-docs "$SNAPSHOT_DIR/docs" \
  --checksum \
  --checksum-output "$CHECKSUM_OUTPUT"
```

The fresh embed command creates the exact physical collection if it is absent. It refuses a
non-empty existing collection. Do not add `--recreate`; destructive recreation has its own
approval and is not part of this contract. `--apply` plus `QDRANT_WRITE_APPROVED=1` is the
explicit authorization for this future candidate write.

**NOT EXECUTED — fresh physical embed**

```bash
QDRANT_WRITE_APPROVED=1 \
.venv/bin/python -m ingest --collection "$PHYSICAL_COLLECTION" embed \
  --snapshot-docs "$SNAPSHOT_DIR/docs" \
  --source all \
  --batch-size 256 \
  --apply
```

If and only if that run was interrupted, resume against the same sealed snapshot, binding,
variant checkpoints, configuration, and existing physical collection. A missing, corrupt, or
mismatched binding/checkpoint/collection is fatal. Never combine `--resume` and `--recreate`.

**NOT EXECUTED — exact resume**

```bash
QDRANT_WRITE_APPROVED=1 \
.venv/bin/python -m ingest --collection "$PHYSICAL_COLLECTION" embed \
  --snapshot-docs "$SNAPSHOT_DIR/docs" \
  --source all \
  --batch-size 256 \
  --resume \
  --apply
```

The write target is always `georgian_legal__gen_<generation>`. Never embed, restore, upsert,
or recreate through the serving alias `georgian_legal`.

## 3. Seal complete generation preparation

Preparation is read-only against Qdrant and create-only on disk. It verifies the dependency
lock and runtime identity, scans the exact physical collection, joins every admitted point to
the clean snapshot ledger one-to-one, and seals the deterministic documents, chunk-zero samples,
strict source state, manifest, provenance, and checksum inventory. It rejects the unconfigured
runtime identity, a non-digest image, mutable model revisions, preflight snapshots, quarantine
leakage, missing or extra documents, non-contiguous chunks, point-ID or payload drift, and any
configuration mismatch.

**NOT EXECUTED — sealed generation preparation**

```bash
.venv/bin/python scripts/prepare_generation.py \
  --generation-id "$GENERATION_ID" \
  --snapshot "$SNAPSHOT_DIR" \
  --physical-collection "$PHYSICAL_COLLECTION" \
  --dependency-lock "$DEPENDENCY_LOCK" \
  --runtime-identity "$RUNTIME_IDENTITY" \
  --image-digest "$FINAL_IMAGE_DIGEST" \
  --actor "$PREPARATION_ACTOR" \
  --run-id "$PREPARATION_RUN_ID" \
  --output-dir "$PREPARED_DIR" \
  --batch-size 256
```

## 4. Publish the immutable generation

Publication consumes only the sealed prepared directory, revalidates its checksum inventory and
current configuration, and atomically creates `$GENERATION_OUTPUT_ROOT/$GENERATION_ID`. It never
overwrites an existing generation and performs no Qdrant mutation.

**NOT EXECUTED — immutable local publication**

```bash
.venv/bin/python scripts/create_generation.py \
  --generation "$GENERATION_ID" \
  --prepared-dir "$PREPARED_DIR" \
  --output-root "$GENERATION_OUTPUT_ROOT"
```

## 5. Verify the exact physical collection

Verification streams the exact physical collection, checks collection compatibility plus every
point against the immutable generation, and creates the sibling
`$GENERATION_OUTPUT_ROOT/$GENERATION_ID.verification.json`. That sidecar binds the generation
manifest and the exact physical collection; evaluation refuses a missing, stale, non-green, or
wrong-collection report.

**NOT EXECUTED — exact-physical verification**

```bash
.venv/bin/python scripts/verify_generation.py \
  "$GENERATION_DIR" \
  --collection "$PHYSICAL_COLLECTION" \
  --page-size 256
```

Do not substitute the serving alias. Verification is intentionally direct-physical and performs
no alias lookup.

## 6. Run both production gates repeatedly

Both commands query the exact physical collection directly and use the normal production-parity
evaluator. `--repeat 2` makes repeat determinism explicit. The evaluator records the serving
alias, physical collection, queried collection, direct-physical access kind, generation, model
identity, and retrieval fingerprint revision separately.

Create the private evaluation-log parent before the first append:

**NOT EXECUTED — evaluation output directory**

```bash
install -d -m 700 "$PWD/.state/v3/evaluation"
```

**NOT EXECUTED — repeated production evaluation**

```bash
.venv/bin/python -m eval.evaluate \
  --backend qdrant \
  --mode production \
  --relevance chunk \
  --top-k 10 \
  --tokenizer bge \
  --citation-route "$CITATION_ROUTE" \
  --golden-set v2 \
  --repeat 2 \
  --log \
  --log-path "$EVAL_LOG"
```

**NOT EXECUTED — repeated accuracy-strict evaluation**

```bash
.venv/bin/python -m eval.evaluate \
  --backend qdrant \
  --mode accuracy_strict \
  --relevance chunk \
  --top-k 10 \
  --tokenizer bge \
  --translate-queries eval/query_translations_v2.json \
  --citation-route "$CITATION_ROUTE" \
  --golden-set v2 \
  --repeat 2 \
  --log \
  --log-path "$EVAL_LOG"
```

Do not use `scripts/eval_local_bypass_run.py`; it intentionally omits the production manifest,
provenance, and verification chain and cannot support a release claim.

## 7. Separately authorize any future promotion

Passing preparation, publication, verification, and evaluation does not authorize promotion.
First create a generation-specific, immutable plan. Set `SNAPSHOT_SHA256` to the exact sealed
manifest value, never to a moving reference.

**NOT EXECUTED — local promotion plan only**

```bash
export SNAPSHOT_SHA256=REPLACE_WITH_SEALED_SNAPSHOT_SHA256
export PROMOTION_PLAN="$PWD/.state/v3/promotion/${GENERATION_ID}.plan.json"
export PROMOTION_STATE="$PWD/.state/v3/promotion/${GENERATION_ID}.state.json"
install -d -m 700 "$PWD/.state/v3/promotion"

.venv/bin/python scripts/promote_generation.py plan \
  --generation-root "$GENERATION_DIR" \
  --snapshot-ref "file://$SNAPSHOT_DIR" \
  --snapshot-sha256 "$SNAPSHOT_SHA256" \
  --created-by "$PREPARATION_ACTOR" \
  --promotion-id "promote-${GENERATION_ID}-01" \
  --output "$PROMOTION_PLAN"
```

Applying the plan is a distinct maintenance change. It requires an approved deployment-specific
smoke/readiness implementation, an existing serving alias, `--apply`, and the explicit approval
environment variable. The backend refuses to create a missing alias and proves rollback. The
following boundary is documentary only and must not be crossed without new authorization:

**NOT EXECUTED — separately authorized alias promotion**

```bash
PROMOTION_APPROVED=1 \
PROMOTION_CHECKS_FACTORY=deployment_checks:make_checks \
.venv/bin/python scripts/promote_generation.py apply \
  --plan "$PROMOTION_PLAN" \
  --state "$PROMOTION_STATE" \
  --backend-factory ingest.qdrant_promotion:make_qdrant_promotion_backend \
  --apply
```

Do not add `--forward-after-rollback` without a second, separate
`PROMOTION_FORWARD_APPROVED=1` authorization. Promotion never grants permission to delete either
generation.

## Prohibited shortcuts

- No `latest`, implicit run selection, mutable model tag, guessed revision, or placeholder
  collection-derived source state.
- No full embed, Qdrant mutation, RunPod launch, external download, publication, or promotion
  while merely validating this contract.
- No legacy `runpod_orchestrate.py`, `runpod_orchestrate_multi.py`, re-embed, or delta
  orchestration path for this generation.
- No write through `georgian_legal`; that name is the stable serving alias and is read-only
  outside the separately approved atomic promotion step.
- No preparation or publication from a preflight snapshot, no checksum inside a snapshot, and
  no generation artifact assembled from the non-production diagnostic
  `scripts/scan_generation_candidate.py`.
- No evaluation through the local bypass helper, no serving-alias evaluation for this physical
  candidate gate, and no evaluation without the exact all-green verification sidecar.
