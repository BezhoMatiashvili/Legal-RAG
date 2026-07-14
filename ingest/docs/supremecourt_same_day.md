# Supreme Court same-day partial release

This runbook publishes a bounded, newest-first Supreme Court release. It is deliberately
not a claim that the 85,851-case archive is complete. A run revisits recent date windows,
skips previously exported identities before detail fetches, and spends its fixed time budget
only on complete official HTML bodies.

## Crawl contract

Run from `scraper/`:

```bash
../.venv/bin/python -m legal_scrapers.run \
  --only supremecourt \
  --start-date 1900-01-01 \
  --end-date 2026-07-13 \
  --max-runtime-seconds 14400 \
  --no-progress
```

The spider coordinates administrative, civil, and criminal chambers through one global
newest-first queue. It probes all windows at the newest pending end date, splits multi-day
windows whose authoritative total exceeds 30, expands sparse older windows, and paginates
only an irreducible one-day window. Requests remain single-concurrency with a fixed eight
second delay. New identities use only `/ka/fullcase/{id}/{palata}` and must contain a
nonempty `div.case-single#modalBody`; DOCX, PDF, proxy, and metadata-only fallbacks are not
part of this run.

Each accepted item is appended and fsynced to `items.journal.jsonl` before its identity is
committed to `seen.sqlite`. On every completed window the spider writes a resumable partial
manifest; on graceful close it atomically materializes cumulative `items.jsonl`, its SHA-256,
new/known and per-chamber counts, date spans, the fully completed global frontier, chamber
resume cursors, retries, unresolved failures, and `partial_by_design=true`.

Do not edit crawler Python files while a production crawl is running. Scrapy may inspect a
callback's source after the process has started, and a changed on-disk line map invalidates that
introspection even though the old code object is still resident. Freeze and hash the crawler
sources before starting the four-hour timer.

For a later crawl with the same start/end bounds, the spider validates the newest finalized run
with the offline artifact gate and resumes from its independently derived chamber cursors. The
child manifest hash-links the parent manifest and items, and validation requires the parent
scope plus every parent identity/body fingerprint to remain intact. A different end date starts
fresh so later releases cannot be skipped.

The first-instance literal `330100122006207137` is not a Supreme Court `case_number`. It is
indexed only when it occurs verbatim in a downloaded decision body.

## Validation and delta embedding

Validate the final cumulative `items.jsonl` as the release corpus. Derive the paid delta from the
filtered Supreme Court coverage report instead of embedding the cumulative file or trusting the
current-run journal alone: the staged missing set preserves the 557 documents already embedded in
the main collection and also catches any earlier successfully scraped identity that is still
missing. Do not use the historical `latest/items.jsonl`, which was only the final 125-record
July 10 export before this workflow.

The validator derives every resume cursor and the global completed frontier from the completed
window ledger; copied cursor strings alone are not accepted.

```bash
.venv/bin/python scripts/validate_supremecourt_partial.py \
  --run-dir ../artifacts/supremecourt/runs/<crawl-run>

RERANK_ENABLED=false .venv/bin/python scripts/embed_delta.py \
  --source supremecourt \
  --items ../artifacts/supremecourt/runs/<crawl-run>/items.jsonl \
  --collection georgian_legal_delta_supremecourt_<run-id>_fullcheck \
  --dry-run --strict \
  --manifest-out .state/supremecourt-<run-id>-dry-run.json

RERANK_ENABLED=false .venv/bin/python scripts/verify_all_embedded.py \
  --coverage-only --sources supremecourt

.venv/bin/python scripts/stage_missing_items.py \
  --missing .state/embed_missing.partial.txt \
  --out .state/supremecourt-<run-id>-missing

RERANK_ENABLED=false .venv/bin/python scripts/embed_delta.py \
  --source supremecourt \
  --items .state/supremecourt-<run-id>-missing/supremecourt.jsonl \
  --collection georgian_legal_delta_supremecourt_<run-id> \
  --dry-run --strict \
  --manifest-out .state/supremecourt-<run-id>-delta.json
```

The paid workflow requires both explicit operator approval channels and uses exactly one
Secure Cloud RTX 4090:

```bash
RUN_ID=<unique-run-id>
GPU_WORKDIR=.state/gpu-work-$RUN_ID \
QDRANT_WRITE_APPROVED=1 QDRANT_RECREATE_APPROVED=1 \
RUNPOD_SPEND_APPROVED=1 RERANK_ENABLED=false \
  .venv/bin/python scripts/runpod_orchestrate_delta.py \
  --source supremecourt \
  --items .state/supremecourt-<run-id>-missing/supremecourt.jsonl \
  --run-id "$RUN_ID" --apply
```

Before provisioning, the orchestrator requires zero active pods and checks the live balance,
secure 4090 stock, price, a conservative runtime estimate, a separately priced 35-minute
cleanup allowance, and an untouched $2 reserve. It repeats that complete live gate after
packaging/key generation immediately before deploy. A monotonic parent alarm bounds every
blocking paid call and budget exhaustion cannot be retried. The pod attests exactly one RTX
4090, runs the CPU/GPU checksum to its own output before embedding, requires cosine similarity
at least 0.999, embeds with CUDA FP16 and batch size 256, and emits an exact identity/count/SHA
snapshot. Signals delegate to a shielded strict `finally`; an uncertain lost deploy response
keeps its unique name until sustained successful account checks prove absence. Local restore
begins only after termination is confirmed.

## Merge and publication boundary

`scripts/merge_delta_collection.py` contains the run-manifest, UUIDv5, source/document/chunk,
write-lock, rollback-snapshot, existing-Supreme-identity, and exact post-merge checks needed by
the same-day delta. Its library workflow deletes only the manifest-bound
`georgian_legal_delta_supremecourt_<run-id>` collection after a verified merge; the mixed
`georgian_legal_delta` collection is never a cleanup target.

The current repository's serving track is generation-only and intentionally refuses the old
implicit-main publication path. `scripts/publish_snapshot.py` accepts only a verified immutable
schema-v2 physical generation, requires a proven conditional manifest activator, writes the
manifest last, and requires explicit cold-restore confirmation. Do not bypass those gates with
a size-only S3 resume or an unconditional manifest overwrite. If the deployed endpoint still
uses the legacy worker image, publishing today's merged main collection is blocked until either:

1. the verified generation worker, generation artifacts, physical collection, immutable image,
   and conditional activator are deployed; or
2. the user explicitly authorizes a separately reviewed legacy compatibility release.

After an approved cold publication, keep FlashBoot disabled until the new manifest has restored
and exact point/identity parity is healthy, then re-enable it, reconnect `/mcp`, and verify
`legal_health`, `legal_collection_info`, `legal_search`, `legal_lookup`, and
`legal_get_document` against all three chambers and the newest decisions.
