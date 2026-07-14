# Cross-cutting contracts (blast-radius file)

> Auto-loaded every session together with [INDEX.md](INDEX.md). Each section is one
> invariant that spans files/processes — exactly the coupling a call-graph tool cannot
> see. The "if you change one side" list is the minimum blast-radius check before
> editing any named symbol. Anchors are `<repo-path>.py:<symbol>`, linted by
> `python3 ingest/scripts/gen_code_map.py --check`. All sections were verified against
> the code on 2026-07-09.


### point-identity
**Invariant:** Every Qdrant point id is `uuid5(NAMESPACE, f"{source}:{document_id}:{chunk_index}")`, so the same (source, doc, chunk) always maps to the same UUID and every upsert path overwrites in place instead of duplicating.
**Participants:**
- `ingest/ingest/qdrant_store.py:point_id` — the only id constructor (deterministic UUIDv5)
- `ingest/ingest/qdrant_store.py:NAMESPACE` — fixed namespace UUID keeping ids stable across runs/machines
- `ingest/ingest/sources.py:SourceSpec.id_fields` — upstream half of the key: `document_id = ":".join(id_fields values)` in `ingest/ingest/sources.py:SourceSpec.normalize`
- `ingest/ingest/pipeline.py:ingest_source` — batch-ingest upsert path, ids from `point_id`
- `ingest/ingest/pipeline.py:_build_doc_points` — shared point builder for watch (`ingest/ingest/pipeline.py:watch_drain_source`) and GPU embed jobs (`ingest/ingest/embed_job.py:embed_docs`, driven by `ingest/scripts/embed_delta.py:main`)
- `ingest/ingest/pipeline.py:_indexed_document_state` — O(1) retrieve of a doc's chunk-0 by its deterministic id; watch's skip-unchanged check depends on it
- `ingest/scripts/merge_delta_collection.py:_audit_delta` / `run_merge_workflow` — require every manifest-bound point id to equal `point_id`, then copy current points, delete stale destination tails, and prove exact post-merge coverage; idempotency still depends on deterministic ids
- `ingest/tests/test_ids.py:test_point_id_is_deterministic` — locks determinism and per-component variation

**If you change one side, also check:**
- Changing `ingest/ingest/qdrant_store.py:NAMESPACE`, the key string format in `point_id`, or any source's `id_fields` re-keys every point: re-upserts then DUPLICATE against the live collection (~2.64M points) — requires collection recreate + full re-embed and an eval re-baseline (see [improvement.md](../improvement.md)).
- Changing chunk boundaries (`ingest/ingest/chunking.py:chunk_document`) shifts `chunk_index`/chunk count: every writer must carry `document_chunk_count`; batch/watch and the fail-closed delta merge clean stale tails only after acknowledged replacement upserts.
- `ingest/ingest/pipeline.py:_indexed_document_state` hard-codes chunk index 0; any scheme where a doc's first chunk isn't index 0 silently disables watch's unchanged-skip.
- `ingest/scripts/verify_all_embedded.py` checks coverage by `document_id` parity from `SourceSpec.id_fields`, not by point id — keep it in sync with any id-derivation change.

**Breaks silently when:** a writer path builds ids outside `point_id` or the key inputs drift in meaning: upserts stop overwriting, docs double-count in the index, and search still "works" while dedup, coverage counts, and merge idempotence quietly rot.

### id-parity-triple
**Invariant:** For each source, three independently-coded identity derivations must select the same item fields and join them with `":"` — `SourceSpec.id_fields` (canonical `document_id` at ingest), the spider's `DEDUP_KEY` (cross-run seen.sqlite skip key), and `verify_all_embedded.py`'s scraped-universe id — or embed-coverage verification compares apples to oranges.
**Participants:**
- `ingest/ingest/sources.py:SourceSpec.id_fields` — per-source identity field tuple; `ingest/ingest/sources.py:SourceSpec.build` joins the non-empty values into `document_id` (the Qdrant payload id). Single source of truth on the ingest side via the `SOURCES` registry.
- `scraper/legal_scrapers/spiders/base.py:BaseLegalSpider.DEDUP_KEY` — hand-copied per-spider mirror of `id_fields` (e.g. `scraper/legal_scrapers/spiders/ecd_spider.py:DEDUP_KEY` = `decision_document_id`, `scraper/legal_scrapers/spiders/supremecourt_spider.py:DEDUP_KEY` = `case_id`+`chamber`); `scraper/legal_scrapers/spiders/base.py:BaseLegalSpider.dedup_key` builds the `":"`-joined seen.sqlite key from it (fail-open: `None` if any field missing).
- `scraper/legal_scrapers/pipelines.py:DedupPipeline.process_item` — generic sources enforce and persist the key per item. Supreme Court instead uses `scraper/legal_scrapers/pipelines.py:SupremecourtDurablePipeline`, which fsyncs the complete item journal before `mark_seen`; `SupremecourtSpider._reconcile_seen_store` removes only keys with no durable full-body item.
- `ingest/scripts/verify_all_embedded.py:scraped_universe` — re-derives ids from raw `items.jsonl` via `SOURCES[dir].id_fields` (replicates `build`'s join); `ingest/scripts/verify_all_embedded.py:embedded_universe` scrolls Qdrant `source`+`document_id` payloads for the diff.
- `ingest/ingest/qdrant_store.py:point_id` — uuid5 over `{source}:{document_id}:{chunk_index}`; idempotent upserts/deltas inherit the same identity.
**If you change one side, also check:**
- Renaming/adding an id field in `SOURCES` (`ingest/ingest/sources.py`) → update that spider's `DEDUP_KEY` and wipe or migrate its `seen.sqlite` (under the spider's artifacts dir, opened by `scraper/legal_scrapers/spiders/base.py:BaseLegalSpider.open_dedup_store`) — old keys stop matching and every doc re-scrapes (or, if fields shrink, distinct docs collide).
- Same change shifts `document_id` → `ingest/ingest/qdrant_store.py:point_id` mints new uuids, orphaning existing points; re-embed or migrate before trusting counts.
- Spider `name` must stay equal to its `SOURCES` key — `ingest/scripts/verify_all_embedded.py:scraped_universe` matches artifact dir names against `SOURCES` and skips unknown dirs.
- `verify_all_embedded.py` itself needs no edit (it imports `SOURCES`), but re-run it after any identity change; quarantine entries (`ingest/scripts/verify_all_embedded.py:quarantined_ids`) are keyed on the old `document_id` too.
**Breaks silently when:** nothing crashes — spiders wrongly skip or endlessly re-fetch documents and `verify_all_embedded.py` reports phantom missing (or a false 0-missing), so the "VERIFIED 0 missing" coverage claim stops meaning anything.

### supremecourt-partial-release
**Invariant:** A timed Supreme Court artifact is a cumulative, explicitly partial, newest-first set of complete official HTML bodies. Across all three chambers a detail may run only when its date is strictly newer than every real pending probe end and every virtual adjacent-successor end; an identity becomes seen only after its full item is fsynced. A same-scope continuation may resume only from independently derived chamber cursors in a finalized, recursively validated parent; the child hash-links the exact parent manifest/items and preserves every parent identity/body fingerprint. The final four-hour artifact is admissible downstream only after the strict offline validator proves the cumulative file, journal, manifest, runtime, resume chain, and full-body provenance agree exactly.
**Participants:**
- `scraper/legal_scrapers/spiders/supremecourt_spider.py:NewestFirstPlanner` — global window heap, split-above-30 and expand-below-20 policy
- `scraper/legal_scrapers/spiders/supremecourt_spider.py:SupremecourtSpider._advance_frontier` — safe cutoff is the maximum of the next real probe end and each ready continuing window's virtual successor (`window.start - 1 day`); cards at or below it wait, and known IDs are skipped before detail requests
- `scraper/legal_scrapers/spiders/supremecourt_spider.py:SupremecourtSpider._discover_resume_state` — selects only the newest finalized validator-admitted run with identical lower/start bounds, seeds each chamber from its derived cursor, and records a manifest/items hash-linked parent; a changed end bound starts fresh so new releases remain visible
- `scraper/legal_scrapers/spiders/supremecourt_spider.py:SupremecourtSpider.parse_detail` — requires the official chamber/identity and a nonempty `div.case-single#modalBody`
- `scraper/legal_scrapers/pipelines.py:SupremecourtDurablePipeline` — persist+fsync, then mark seen, then settle the window
- `scraper/legal_scrapers/spiders/supremecourt_spider.py:SupremecourtSpider._manifest` / `SupremecourtSpider.closed` — atomic cumulative materialization, SHA, counts/spans/frontiers/cursors/failures, `partial_by_design=true`; the final manifest falls back to the spider's independent monotonic clock because `spider_closed` runs before CoreStats supplies elapsed time
- `scraper/legal_scrapers/run.py:crawl_quality_issues` — accepts `closespider_timeout` only for a spider declaring the intentional partial contract
- `ingest/scripts/validate_supremecourt_partial.py:validate_run` — fail-closed, offline admission gate for the exact final `closespider_timeout`/14,400-second manifest, private regular artifact paths, SHA/count/order/identity/body/journal/chamber/frontier/cursor/retry/failure parity, recursively hash-bound same-scope parent preservation, and first-instance-number provenance
**If you change one side, also check:**
- Window/card ordering changes need the strict cross-chamber, unequal-window virtual-successor, and adaptive-boundary cases in `tests/test_supremecourt_newest_first.py`; Scrapy request priority alone is not a correctness proof because pending list probes can reveal newer cards.
- Dedup changes must preserve journal-before-seen ordering and exact ghost reconciliation; a network/parse failure stays unseen and retryable.
- Resume changes must derive cursors from the completed-window ledger, recursively validate parent scope/hash/content, and retain every parent fingerprint; never trust manifest cursor strings alone.
- Final items and manifest must be written only after in-flight item processing drains; keep the independent monotonic duration and require `ingest/scripts/validate_supremecourt_partial.py:validate_run` before tokenizer/model/Qdrant/remote work. Downstream GPU input is the validated cumulative run `items.jsonl`, never a historical non-cumulative `latest/items.jsonl`.
- Freeze and hash every `scraper/legal_scrapers/**/*.py` source before the production timer starts, and do not edit those files until the Scrapy process exits; Python/Scrapy callback introspection may read live source and reject an otherwise valid resident code object after line mappings change.
**Breaks silently when:** known decisions make a broad window look complete, a ready window's not-yet-queued successor is omitted from the cutoff, copied resume cursors or unbound parent bytes are trusted, elapsed time is trusted from CoreStats during the earlier spider-close signal, or `seen.sqlite` is committed before the body. Editing a live spider can instead fail loudly late in the run during callback-source introspection. The result may contain plausible cases but be out of order, short of four hours, missing parent bodies, or unrecoverable.

### runpod-delta-spend-safety
**Invariant:** A paid source-aware delta run may create exactly one Secure Cloud RTX 4090 only after the local restore target is reachable and absent and a live account/stock/price/reserve gate passes. One owner-only nonblocking flock spans that live gate through strict termination confirmation; the gate is repeated immediately before deploy, a monotonic process alarm bounds every blocking paid call, a separately priced cleanup margin protects strict reconciliation, and the $2 reserve remains outside both compute and cleanup budgets. Local restore cannot begin while a created pod may still be billing.
**Participants:**
- `ingest/scripts/runpod_orchestrate_delta.py:preflight_local_delta_restore` — before spend, require loopback Qdrant to be reachable and the unique run-scoped target collection to be absent; restore repeats the absence check to close the race
- `ingest/scripts/runpod_orchestrate_delta.py:runpod_spend_lock` / `run_locked_paid_workflow` — owner-only regular `ingest/.state/runpod-spend.lock`, held from the live zero-pod/price gate through `run_paid_workflow`'s confirmed cleanup
- `ingest/scripts/runpod_orchestrate_delta.py:runpod_spend_gate` — browser-UA live balance, zero active pods, Secure RTX 4090 stock/price, conservative runtime cost, priced 35-minute cleanup allowance, and untouched $2 reserve; run again after packaging/key generation immediately before provisioning
- `ingest/scripts/runpod_orchestrate_delta.py:attest_provider_pod` / `attest_single_4090` — after provisioning require the sole active pod to have the exact ID/name, a positive price no greater than the gated price, non-false provider security state (or retain the explicit SECURE request when the API field is unsupported), and exactly one hardware-reported RTX 4090
- `ingest/scripts/runpod_orchestrate_delta.py:paid_budget_watchdog` / `enforce_reserve_budget` / `run_paid_workflow` — monotonic cost clock and POSIX alarm start at the provision attempt, so even blocked SSH/transfers cannot outlive the compute allowance; budget exhaustion is non-retryable. Signals delegate to a signal-shielded strict `finally`; uncertain lost deploy responses retain their unique name until sustained successful account queries prove absence.
**If you change one side, also check:**
- Provider queries, pod naming, or provisioning fields require the live gate and post-provision attestation tests in `ingest/tests/test_supremecourt_delta_workflow.py`; never add a fallback GPU or accept an unavailable/higher live price.
- Moving the spend lock, live re-gate, watchdog, or budget clock must preserve the boundary from the serialized account gate through confirmed cleanup. No paid blocking call may escape the parent watchdog, and cleanup margin must remain separate from the $2 reserve.
- Restore ordering must remain: validate downloaded outputs, terminate and confirm the created pod absent, then recheck the local target and restore the run-scoped collection.
**Breaks silently when:** the account gate and provision are not serialized, the billed pod is not re-attested, a retry swallows `ReserveBudgetError`, non-strict cleanup forgets an uncertain deploy name, a second signal interrupts strict cleanup, or restore begins before confirmed absence. Each can produce a technically valid snapshot while overspending, using the wrong provider/GPU, colliding with another run, or leaving a billing pod alive.

### payload-contract
**Invariant:** Every Qdrant point payload key is a bare string written once by `ingest/ingest/qdrant_store.py:build_payload` and read back by name (`payload.get("...")` / `FieldCondition(key="...")`) across ingestion, indexing, filtering, serving, and eval — there is no shared schema object, so writer and all readers must agree on the exact strings.
**Participants:**
- `ingest/ingest/qdrant_store.py:build_payload` — sole writer; canonical field set (source, document_id, chunk_index, title, date, date_raw, language, document_type, court, source_url, document_number, registration_code, parties, status, is_consolidated, consolidated_count, in_force_date, expiry_date, heading, token_count, char_start, char_end, text, content_hash) plus `doc.promoted` via `setdefault`
- `ingest/ingest/qdrant_store.py:KEYWORD_FIELDS` (+ `BOOL_FIELDS`, `INTEGER_FIELDS`, `DATETIME_FIELDS`, `TEXT_FIELDS`) — index-field tuples consumed by `ingest/ingest/qdrant_store.py:ensure_collection` to create payload indexes
- `ingest/ingest/sources.py:PROMOTED_KEYWORD_FIELDS` / `ingest/ingest/sources.py:PROMOTED_TEXT_FIELDS` — per-source promoted keys (e.g. TAS applicant fields) that `ensure_collection` also indexes; values come from `ingest/ingest/sources.py:CanonicalDoc`.promoted
- `ingest/ingest/qdrant_store.py:delete_doc_chunks_from` — filters on `source`/`document_id`/`chunk_index` to drop stale chunks
- `ingest/ingest/search.py:build_filter` — maps every filter kwarg to a hard-coded payload key; keyword fields need the KEYWORD index, `parties`/`contains→text` need the TEXT index, `date_from/to` range over RFC-3339 `date`
- `ingest/ingest/search.py:rerank_points` — reads `text` from candidate payloads for cross-encoder scoring
- `ingest/ingest/mcp_server.py:_hit_dict` / `ingest/ingest/mcp_server.py:_format_hit_md` — shape search hits for JSON/markdown output (note: emits payload `date_raw` under output key `date`)
- `ingest/ingest/mcp_server.py:legal_get_document` + `ingest/ingest/mcp_server.py:_stitch_overlap` — reassemble a doc ordered by `chunk_index`, overlap-trim via `char_start`/`char_end`; `ingest/ingest/mcp_server.py:_dedup_documents` collapses browse results per (source, document_id)
- `ingest/ingest/pipeline.py:_indexed_document_state` — watch-mode change detection via chunk-0 `document_state_hash` (legacy payloads are reconstructed from cleaned body hash + metadata)
- `ingest/ingest/__main__.py:_cmd_search` — CLI result rendering from raw payload keys
- `ingest/eval/backend.py:QdrantBackend._points_to_hits` (and `QdrantBackend._ensure_bm25`) — eval Hit keys from `source`/`document_id`/`chunk_index`
- `ingest/eval/bm25_full.py:FullCorpusBM25` — builds the disk BM25 index by scrolling `text` + the same identity triple
- `ingest/ingest/querylog.py:build_query_record` — downstream of `_hit_dict` output keys (`document_id`/`source`/`chunk_index`/`score`)
**If you change one side, also check:**
- Renaming/adding a canonical field: `build_payload` AND the field tuples in `qdrant_store.py` (`KEYWORD_FIELDS`/`TEXT_FIELDS`/`BOOL_FIELDS`/`INTEGER_FIELDS`/`DATETIME_FIELDS`) so `ensure_collection` indexes it — existing collections need a manual `create_payload_index` (ensure_collection only runs at creation)
- Filterable fields: `ingest/ingest/search.py:build_filter` kwargs and the MCP tool input models in `ingest/ingest/mcp_server.py` (`SearchInput` etc.) that forward them
- Display/output fields: `ingest/ingest/mcp_server.py:_hit_dict`, `_format_hit_md`, `legal_get_document`, `_dedup_documents`, `ingest/ingest/__main__.py:_cmd_search`
- Identity triple (`source`, `document_id`, `chunk_index`): `ingest/ingest/qdrant_store.py:point_id`, `delete_doc_chunks_from`, `ingest/eval/backend.py:QdrantBackend._points_to_hits`, `ingest/eval/bm25_full.py:FullCorpusBM25`, `ingest/ingest/querylog.py:build_query_record`
- `text` / `content_hash` / `document_state_hash` / `char_start`/`char_end`: `ingest/ingest/search.py:rerank_points`, `ingest/ingest/pipeline.py:_indexed_document_state`, `ingest/ingest/mcp_server.py:_stitch_overlap`
- Promoted per-source fields: `ingest/ingest/sources.py:PROMOTED_KEYWORD_FIELDS`/`PROMOTED_TEXT_FIELDS` (setdefault means a promoted key colliding with a canonical one is silently dropped)
- `ingest/serverless/handler.py` delegates to the `mcp_server` tools, so it inherits — no separate field list
- `is_consolidated` SEMANTICS (since 2026-07-10): "matsne main (consolidated) document" — matsne's `type=main` class (~52k base acts carrying current consolidated text), NOT merely "has ≥1 publication-switcher version". New snapshot records/loaders preserve the flag, but the immutable v1 artifact predates those fields; use a new snapshot generation or sidecar join before a full re-embed. Payload-only reconciliation after a v2-header embed creates payload/vector disagreement and therefore requires a coordinated re-embed.
**Breaks silently when:** a reader's `payload.get("field")` string no longer matches what `build_payload` wrote (or the field was never added to the index tuples / an existing collection's indexes), so filters match nothing, hit fields render as None, watch re-embeds unchanged docs, and rerank scores empty strings — all without any exception.

### content-hash-semantics
**Invariant:** Snapshot, batch, watch, and delta writers all hash the same `hygiene.clean_text` output. `content_hash` identifies cleaned body text; `document_state_hash` additionally identifies payload/embedding-relevant metadata plus chunk/model/header configuration so same-body legal status/title/date/consolidation changes are not skipped.
**Participants:**
- `ingest/ingest/dedup.py:content_hash` — the sole hash function (SHA-256 of whatever text it is given)
- `ingest/ingest/qdrant_store.py:build_payload` — stamps cleaned `content_hash`, optional `document_state_hash`, and `document_chunk_count` on every point
- `ingest/ingest/pipeline.py:_prepare_doc_for_index` — applies the shared hygiene/quarantine contract before every incremental writer hashes/chunks/embeds
- `ingest/ingest/pipeline.py:_document_state_hash` / `_indexed_document_state` — produce and read the full writer identity used by watch skip-unchanged
- `ingest/ingest/snapshot.py:build_snapshot` / `_snapshot_record` — hash and persist the same cleaned body used by incremental writers
- `ingest/ingest/qdrant_store.py:KEYWORD_FIELDS` — `content_hash` remains keyword indexed; state/chunk-count markers are merge/watch metadata and are not search indexes
**If you change one side, also check:**
- Changing `hygiene.clean_text`, state fields, chunk configuration, model identity, or v2 header mode changes writer identity and requires a coordinated new immutable snapshot/re-embed plus retrieval re-baseline.
- Legacy payloads without `document_state_hash` are reconstructed from stored metadata under the current config to avoid a needless one-time full-corpus rewrite; once rewritten, the explicit state hash is authoritative.
- `ingest/ingest/mcp_server.py` exposes payload hashes in full-document JSON; external consumers inherit cleaned-body semantics.
**Breaks silently when:** a writer bypasses `_prepare_doc_for_index` or adds payload/embedding-relevant metadata without folding it into `_document_state_hash`; watch may then skip a legally meaningful change.

### offsets-golden-set
**Invariant:** Golden-set judgments are half-open char spans `[char_start, char_end)` into the cleaned snapshot `body_markdown` (never chunk ids), and every producer/consumer of offsets works in that single coordinate space — `reground` must slice each `evidence_quote` back exactly (NFC-compared), and chunk offsets from `chunk_document` let spans be re-mapped to `chunk_index` qrels under whatever chunk config is currently evaluated.
**Participants:**
- `ingest/ingest/chunking.py:chunk_document` — sets `Chunk.char_start`/`char_end` in the input body's own coordinate space (tracked through split/strip, never re-located); `ingest/ingest/chunking.py:Chunk` carries them (`-1` sentinel = unset)
- `ingest/ingest/chunking.py:heading_spans` — char ranges of consumed `#` heading lines, so a span inside a heading maps to the chunks that heading governs
- `ingest/ingest/snapshot.py:_snapshot_record` — writes the cleaned `body_markdown` (output of `ingest/ingest/hygiene.py:clean_text`) to `ingest/snapshots/v1/docs/*.jsonl` — this text IS the coordinate space
- `ingest/eval/goldset.py:SnapshotBodies` — loads that `body_markdown` for the harness
- `ingest/eval/goldset.py:reground` — loud `ValueError` if any `golden_set_v1.jsonl` span no longer slices back to its `evidence_quote` (catches snapshot-text/hygiene drift)
- `ingest/eval/goldset.py:lint_span_coverage` — loud `ValueError` if any span maps to zero chunks under the current config
- `ingest/eval/spanmap.py:map_spans_to_chunks` / `ingest/eval/spanmap.py:graded_relevant_chunks` — span → covering `chunk_index` set via `ingest/eval/spanmap.py:chunks_covering_span` (body overlap ∪ heading governance)
- `ingest/eval/evaluate.py:build_query_relevance` — builds qrels keyed `(source, document_id, chunk_index)`; `ingest/eval/evaluate.py:main` runs the reground + holdout + lint preflight and stamps `ingest/eval/goldset.py:eval_set_hash`
- `ingest/ingest/qdrant_store.py:build_payload` — persists `chunk.char_start`/`char_end` into every Qdrant point payload
- `ingest/ingest/mcp_server.py:_hit_dict` / `ingest/ingest/mcp_server.py:_stitch_overlap` — serving-side consumers of the payload offsets (citation span; overlap-stitch signal)
- Data: `ingest/eval/golden_set_v1.jsonl`, `ingest/eval/holdout_doc_ids.json`

**If you change one side, also check:**
- Chunker offset semantics (`ingest/ingest/chunking.py:_split_sections`, `ingest/ingest/chunking.py:_atoms`, `ingest/ingest/chunking.py:_pack`, `_ARTICLE_LINE_RE`) → `ingest/tests/test_chunk_offsets.py` and `ingest/tests/test_spanmap.py`; the lint only catches *empty* mappings, not wrong ones
- Snapshot text (`ingest/ingest/hygiene.py:clean_text`, `ingest/ingest/snapshot.py:_snapshot_record`, or re-scraping the sources) → `reground` will fail on load; every span in `golden_set_v1.jsonl` must be re-anchored to the new bodies
- Chunk config/tokenizer (`ingest/ingest/config.py:Config` `chunk_tokens`/`chunk_overlap`/`chunk_min_tokens`, `ingest/ingest/embedding.py` tokenizer) → the live index's `chunk_index` values must have been produced under the same config `evaluate.py` passes as `chunk_cfg`, or qrels and retrieved points disagree; re-embed or re-baseline
- Payload shape in `ingest/ingest/qdrant_store.py:build_payload` → `ingest/ingest/mcp_server.py:_hit_dict` citation spans and `_stitch_overlap`

**Breaks silently when:** the live index was chunked under a different config/tokenizer than `ingest/eval/evaluate.py:main`'s `chunk_cfg` (qrels `chunk_index` no longer names the same text as the indexed points), or a chunker offset bug maps spans to a wrong-but-nonempty chunk set — neither trips `reground` nor `lint_span_coverage`, so scores are quietly wrong.

### reranker-score-parity
**Invariant:** The local cross-encoder and the GPU-pod HTTP reranker must produce identical scores — same model (`BAAI/bge-reranker-v2-m3`), fast `AutoTokenizer`, `truncation, max_length=512`, `sigmoid(logit)` → 0..1 — because downstream gates and eval numbers assume one calibrated scale regardless of which path scored.
**Participants:**
- `ingest/ingest/rerank.py:BGEReranker.score` — local (CPU/GPU) reference scorer; hardcodes `_MAX_LENGTH=512`, sigmoid; fp16 only if `RERANK_USE_FP16` and device != cpu (default fp32)
- `ingest/ingest/rerank.py:RemoteBGEReranker.score` — drop-in HTTP client (POST `/score`), enabled by `RERANK_REMOTE_URL`; trusts the server's scale blindly
- `ingest/scripts/runpod_rerank_server.py:score` — pod-side mirror of `BGEReranker.score`; model/max_length come from pod env `RERANK_MODEL`/`RERANK_MAX_LENGTH` (defaults match local), fp32; batch size differs (64 vs 16) but padding is masked so scores match
- `ingest/ingest/search.py:rerank_points` — overwrites `pt.score` with the rerank score and gates on `min_score` — the consumer that makes the 0..1 scale load-bearing
- `ingest/ingest/config.py:Config` — `rerank_min_score` (env `RERANK_MIN_SCORE`, default 0.3) and `rerank_model` (env `RERANK_MODEL`); local model name and pod model name are set from the *same env var in two different processes* — nothing checks they agree
- `ingest/ingest/mcp_server.py:_get_reranker` — serving-time selector: `RERANK_REMOTE_URL` set → `RemoteBGEReranker`, else local `BGEReranker` (also the path the serverless `ingest/serverless/handler.py` reuses, so serverless is parity-by-construction)
- `ingest/eval/evaluate.py:qdrant_deps` — eval-time selector using the same `RERANK_REMOTE_URL` switch; baseline/experiment numbers in `ingest/eval/experiments.jsonl` mix runs from both paths
**If you change one side, also check:**
- Any change to `BGEReranker.score` (normalization, `_MAX_LENGTH`, tokenizer) → mirror in `ingest/scripts/runpod_rerank_server.py:score` (deployed to the pod by `ingest/scripts/runpod_rerank.py:setup_pod`)
- Changing `rerank_model` default in `ingest/ingest/config.py:Config` → update the server's `RERANK_MODEL` default in `ingest/scripts/runpod_rerank_server.py` and re-baseline eval
- Changing the score scale at all → revisit `RERANK_MIN_SCORE` default (0.3) in `ingest/ingest/config.py:Config` and the gate in `ingest/ingest/search.py:rerank_points`
- Response shape `{"scores": [...]}` → both `RemoteBGEReranker.score` and `ingest/scripts/runpod_rerank_server.py:Handler.do_POST`
**Breaks silently when:** the pod runs a different model/max_length/normalization than local (env drift or a one-sided edit) — the remote still returns valid-looking floats, but the `rerank_min_score` gate over- or under-filters and GPU-path eval numbers stop being comparable to CPU-path serving, with no error anywhere.

### remote-fallback
**Invariant:** Remote dependencies fail asymmetrically by design — a vanished remote GPU reranker makes `legal_search` silently retry with `reranker=None` (RRF-fused order, tool keeps serving), while the `SEARCH_BACKEND=remote` thin client NEVER falls back to local (it returns an actionable error string) and its `{"op","params"}` → `{"result"|"error"}` wire shape must stay in lockstep with the serverless worker's op dispatcher.
**Participants:**
- `ingest/ingest/mcp_server.py:legal_search` — owns the reranker fallback: on any `hybrid_search` exception with a remote reranker, retries once with `reranker=None` (RRF order); local rerank errors still raise
- `ingest/ingest/mcp_server.py:_is_remote_reranker` — the marker: `getattr(reranker, "device", None) == "remote"`; there is no `rerank_used` flag
- `ingest/ingest/rerank.py:RemoteBGEReranker` — HTTP drop-in for `BGEReranker`; sets the `device = "remote"` sentinel in `__init__`; selected by `RERANK_REMOTE_URL` in `ingest/ingest/mcp_server.py:_get_reranker`
- `ingest/ingest/search.py:hybrid_search` — fallback-agnostic: `reranker=None` ⇒ returns the RRF-fused order directly
- `ingest/ingest/mcp_server.py:_use_remote` / `ingest/ingest/mcp_server.py:_remote_op` — backend switch (`cfg.search_backend == "remote"`) and the one-RPC wrapper; deliberately NO silent local fallback — errors become agent-facing text ("set SEARCH_BACKEND=local … and reconnect /mcp")
- `ingest/ingest/remote_search.py:RunPodQueueClient.call` — stdlib-only queue client: POST `/run` with `{"input": {"op", "params"}}`, poll `/status`; budget expiry raises `ingest/ingest/remote_search.py:EndpointWarmingUp` without cancelling the job (retry lands warm)
- `ingest/serverless/handler.py:handler` — worker side of the wire contract: dispatches `op` through `ingest/serverless/handler.py:OPS` to the same MCP tool coroutines, returns `{"result": "<tool output string>"}` or `{"error": ...}`; hard-sets `SEARCH_BACKEND=local` at import so the worker never RPCs itself
- `ingest/eval/evaluate.py:qdrant_deps` — second `RemoteBGEReranker` consumer (eval offloads scoring via `RERANK_REMOTE_URL`, no fallback here — eval must not silently change mode)
- `ingest/scripts/publish_snapshot.py:verify` — second `RunPodQueueClient` consumer (`refresh` op, own 1800s budget instead of `cfg.runpod_api_timeout`)
**If you change one side, also check:**
- Renaming/removing the `device = "remote"` sentinel on `ingest/ingest/rerank.py:RemoteBGEReranker` breaks `ingest/ingest/mcp_server.py:_is_remote_reranker` → remote-reranker outages start erroring `legal_search` instead of degrading; conversely, giving `BGEReranker` a `device == "remote"` state would silently swallow local rerank crashes.
- Adding/renaming an op: `ingest/serverless/handler.py:OPS` keys, the `_remote_op("<op>", …)` call sites in `ingest/ingest/mcp_server.py`, and the tool's Pydantic input model must all move together; `refresh` is dispatched before the OPS lookup in `ingest/serverless/handler.py:handler`, and `ingest_status` is local-only by design.
- Changing the job output shape (`result`/`error` keys) touches both `ingest/ingest/remote_search.py:RunPodQueueClient.call` (raises `RemoteOpError` on `out["error"]`) and `ingest/ingest/mcp_server.py:_remote_op` (`out.get("result")`), plus `ingest/scripts/publish_snapshot.py:verify` which parses `result` as JSON.
- New config knobs: `ingest/ingest/config.py:search_backend` / `rerank_remote_url` / `runpod_api_timeout` feed `_use_remote`, `_get_reranker`, `_get_remote_client`; `ingest/tests/test_remote_backend.py` pins the client semantics (do not edit to pass).
- Any change to `ingest/ingest/search.py:hybrid_search`'s exception behavior changes what the fallback catches — the retry fires on ANY exception when the reranker is remote, so a Qdrant error also gets one no-rerank retry.
**Breaks silently when:** the remote reranker stops being identifiable as remote (sentinel renamed, or a wrapper hides `.device`), because `legal_search` then serves RRF-only order — or errors outright — with nothing but a `logger.warning` to distinguish degraded results from reranked ones (the hit `score` semantics silently flip from calibrated 0..1 relevance to RRF rank score).

### snapshot-publish-protocol
**Invariant:** Only a verified immutable schema-v2 physical generation may be published. The SHA-identified snapshot object lands first; a proven conditional compare-and-swap activates `publish/manifest.json` LAST. A cold worker restores into an absent `georgian_legal__gen_<generation_id>` collection and writes ACTIVE only after exact count, vector schema, and every-point generation identity match. Warm replacement and legacy implicit-main manifests fail closed.
**Participants:**
- `ingest/scripts/publish_snapshot.py:upload` — rehashes the owner-only local snapshot and resumes only when remote size + SHA metadata match; requires explicit remote approval and cold-worker confirmation
- `ingest/scripts/publish_snapshot.py:_publish_manifest_object` — revalidates the snapshot and delegates manifest-LAST activation only to a capability-attested conditional activator
- `ingest/scripts/publish_snapshot.py:verify` — requires exact snapshot/generation/manifest-SHA/point/identity parity from `refresh` and `health`
- `ingest/serverless/qdrant_boot.py:validate_publish_manifest` — strict schema-v2 safe-name, physical-collection, immutable model/vector/chunk/retrieval identity gate
- `ingest/serverless/qdrant_boot.py:_collection_compatibility` — exact green collection/vector/count and all-points identity count
- `ingest/serverless/qdrant_boot.py:maybe_restore` — SHA verify, cold-only recover, compatibility proof, durable ACTIVE write; never deletes/replaces a warm serving collection
- `ingest/serverless/qdrant_boot.py:verified_runtime_manifest` / `runtime_readiness` — bind publish, ACTIVE, live Qdrant, and the already-imported search runtime before every operation
- `ingest/serverless/handler.py:_configure_runtime_environment` — derives immutable collection/model revisions from the accepted manifest before MCP import
- `ingest/serverless/Dockerfile` — fail-closed supply-chain inputs; no floating production defaults
- `ingest/docker-compose.yml` — local `qdrant/qdrant:v1.18.2`; snapshots are only compatible within the same minor version
**If you change one side, also check:**
- Manifest schema ↔ `validate_publish_manifest` / `_restore_identity` ↔ handler environment binding ↔ `publish_snapshot.verify`; adding or renaming a field is a protocol migration, not a local refactor.
- Status strings returned by `maybe_restore` (`up_to_date`, `restored`, `error`, `no_manifest`) ↔ `restore_allows_serving` ↔ `publish_snapshot.verify`.
- `REMOTE_PREFIX`/`manifest.json`/`ACTIVE` object names in `ingest/scripts/publish_snapshot.py:upload` ↔ `ingest/serverless/qdrant_boot.py:publish_dir` readers
- Model revisions/fingerprints in the manifest must equal `ingest/ingest/qdrant_store.py:generation_point_identity` under the worker's forced config and every stored point.
- Qdrant version/supply-chain identity: rebuild the immutable image with verified inputs and retain snapshot compatibility before publishing.
- Restore duration assumptions: `PUBLISH_VERIFY_TIMEOUT` default (1800s) in `ingest/scripts/publish_snapshot.py:verify` ↔ the endpoint execution timeout noted in `ingest/serverless/qdrant_boot.py:maybe_restore`
**Breaks silently when:** an uploader trusts size alone, overwrites the manifest unconditionally, a worker accepts a warm in-place restore, or readiness stops recounting the point identity. Those shortcuts can make a mixed or stale corpus appear healthy; all are intentionally refused.

### config-hash-vs-fingerprint
**Invariant:** The repo has three deliberately separate identity digests — `ingest/eval/explog.py:config_hash` (eval-run identity in `ingest/eval/experiments.jsonl`), `ingest/ingest/config.py:retrieval_fingerprint` (serving-config identity stamped on MCP responses and the query log), and the same-named `ingest/ingest/snapshot.py:config_hash` (snapshot-manifest identity) — each retrieval-relevant knob must fold into the digest(s) owning its scope, and one digest must never stand in for another.
**Participants:**
- `ingest/eval/explog.py:config_hash` — 16-hex digest of `LOGIC_REV` + knob material; the eval-run identity key
- `ingest/eval/evaluate.py:main` — assembles the hash material (mode, relevance, top_k, tokenizer, chunk cfg, effective rerank_candidates, active tuning knobs; `--translate-queries` enters as file-content hash, never the raw dict); every new CLI knob MUST fold in
- `ingest/eval/explog.py:append_run` — appends the ledger row (`config_hash` field + descriptive-only `knobs` mirror that does NOT affect the hash; index identity is the `backend` dict from `ingest/eval/evaluate.py:qdrant_deps`, not `retrieval_fingerprint`)
- `ingest/scripts/build_phase_c_report.py:dedup_latest` — keeps newest row per (mode, config_hash, relevance_level); collided hashes silently drop runs here
- `ingest/ingest/config.py:retrieval_fingerprint` — digest of serving knobs (collection, embed model/dim, rerank, chunking); lives in `ingest` so the ingest package never imports `eval` (layering is eval → ingest only)
- `ingest/ingest/mcp_server.py:legal_search`, `ingest/ingest/mcp_server.py:ingest_status`, `ingest/ingest/mcp_server.py:legal_health` — stamp `retrieval_fingerprint` into responses; `ingest/ingest/mcp_server.py:_log_query` routes it into `ingest/ingest/querylog.py:build_query_record`
- `ingest/ingest/snapshot.py:config_hash` — third digest (pipeline_version, logic_rev, hygiene/dedup thresholds, chunk cfg, embed model, sources) written into the snapshot manifest for artifact traceability
**If you change one side, also check:**
- Adding an eval tuning knob (`ingest/eval/evaluate.py:main` argparse): fold it into the `active_knobs` → `explog.config_hash` material AND the descriptive `knobs` mirror; check `ingest/scripts/build_phase_c_report.py:rc_of`/`dedup_latest` still label rows correctly
- Changing eval scoring semantics: bump `ingest/eval/explog.py:LOGIC_REV` (old rows become incomparable by design, not by accident)
- Adding/changing a serving retrieval knob in `ingest/ingest/config.py:load_config`: add it to `ingest/ingest/config.py:retrieval_fingerprint` material; `ingest/tests/test_part3_units.py:test_retrieval_fingerprint_stable_and_sensitive` guards stability+sensitivity
- Changing snapshot hygiene/dedup/chunk logic: bump the `logic_rev` inside `ingest/ingest/snapshot.py:config_hash`
**Breaks silently when:** a new knob (or changed default) alters retrieval behavior without entering the owning digest's material, so two different configurations share a hash — `experiments.jsonl` rows collide, `dedup_latest` discards one as a duplicate re-run, and A/B history is corrupted with no error anywhere.

### env-at-spawn
**Invariant:** The `legal_rag` MCP stdio server snapshots its environment once per process — `load_dotenv()` runs when `ingest/ingest/config.py` is first imported, the first tool call freezes a `Config` into a module global, and every heavy singleton is built from that frozen copy — so edits to `ingest/.env` or any code under `ingest/ingest/` are invisible to MCP tools until the server is respawned (`/mcp` reconnect), while the direct CLI always sees current code and env because it is a fresh process.

**Participants:**
- `.mcp.json` — spawn point: registers `legal_rag` as `uv run --directory .../ingest python -m ingest.mcp_server` (also injects OMP/MKL thread vars, which `.env` cannot override)
- the module-level `load_dotenv()` call in `ingest/ingest/config.py` — module-import-time copy of `ingest/.env` into `os.environ`; `override=False`, so vars already set in the process win
- `ingest/ingest/config.py:load_config` — builds a frozen `Config` from `os.environ` at call time; itself uncached — callers decide the lifetime
- `ingest/ingest/mcp_server.py:_get_cfg` — the process-lifetime `Config` cache (`_cfg` global, assigned once, no reset path); every env read in the server goes through it
- `ingest/ingest/mcp_server.py:_get_client` / `ingest/ingest/mcp_server.py:_get_embedder` / `ingest/ingest/mcp_server.py:_get_reranker` / `ingest/ingest/mcp_server.py:_get_remote_client` — lazy singletons built once from the cached cfg (Qdrant client, BGE-M3, reranker, RunPod client)
- `ingest/ingest/mcp_server.py:_use_remote` — local-vs-serverless routing off the cached `search_backend`; flipping `SEARCH_BACKEND` in `.env` does nothing until reconnect (`ingest/ingest/mcp_server.py:_remote_op` error text says exactly this)
- `ingest/ingest/config.py:retrieval_fingerprint` — stamped into `legal_search` responses, the query log, and `legal_health`; a stale server stamps the stale config's fingerprint
- `ingest/ingest/__main__.py:_resolved_cfg` — the contrast case: `python -m ingest search/embed/...` calls `load_config()` fresh per process (as do `ingest/eval/` and `ingest/scripts/`)
- `ingest/serverless/handler.py` — hard-assigns `SEARCH_BACKEND=local` / `QUERY_LOG_ENABLED=false` into `os.environ` *before* `from ingest import mcp_server`, exploiting `load_dotenv`'s no-override; the worker process caches its cfg the same way

**If you change one side, also check:**
- Adding/renaming an env var or `Config` field: `ingest/ingest/config.py:load_config`, whether it belongs in `ingest/ingest/config.py:retrieval_fingerprint` (retrieval-affecting knobs must change the fingerprint), and the `.env`-edit-requires-reconnect note wherever you document the var.
- Any edit under `ingest/ingest/` or to `ingest/.env`: reconnect `/mcp` before trusting MCP tool output; confirm via the fingerprint in `mcp__legal_rag__legal_health` — but only for config changes, code changes don't move the fingerprint.
- `ingest/serverless/handler.py` import order: the `os.environ` hard-assigns must stay above the `ingest.mcp_server` import, or the worker can route RPCs to itself and write query logs off-box.
- New long-lived processes (daemons, watchers): if they cache `Config`, they inherit this contract — prefer the per-process pattern of `ingest/ingest/__main__.py:_resolved_cfg` unless staleness is acceptable.

**Breaks silently when:** you edit `ingest/.env` or ingest code, call an MCP tool, and get plausible-looking results computed by the pre-edit code/config — no error is raised, the only tell is a stale fingerprint in `legal_health`/query-log entries, and code-only edits don't even move that.

### ram-discipline
**Invariant:** On this 30 GB box (Qdrant resident ~21 GB) the ~2.3 GB BGE cross-encoder may be loaded only by a process that will actually rerank — every other process must run with `RERANK_ENABLED=false` (or point at a GPU pod via `RERANK_REMOTE_URL`), because `rerank_enabled` defaults to **True** and an idle resident reranker evicts Qdrant's hot sparse index to swap.

**Participants:**
- `ingest/ingest/config.py:load_config` — parses `RERANK_ENABLED` (default True), `RERANK_REMOTE_URL`, `RERANK_DEVICE` into `Config.rerank_enabled` / `rerank_remote_url`; the flag is per-process env, nothing global enforces it
- `ingest/ingest/rerank.py:BGEReranker` — local cross-encoder; heavy imports (torch/transformers) deferred into `__init__`, so importing the module is free but instantiating always loads weights
- `ingest/ingest/rerank.py:RemoteBGEReranker` — drop-in HTTP scorer for a GPU pod; loads no local ML deps (the zero-RAM path)
- `ingest/ingest/mcp_server.py:_get_reranker` — lazy async singleton (module global `_reranker` + lock); returns `None` when `rerank_enabled` is false, `RemoteBGEReranker` when `rerank_remote_url` is set, else loads `BGEReranker` off the event loop
- `ingest/ingest/mcp_server.py:_get_embedder` — same lazy-singleton pattern for `ingest/ingest/embedding.py:BGEM3Embedder` (the other resident model)
- `ingest/eval/evaluate.py:qdrant_deps` — **eager** loader: instantiates the reranker whenever `cfg.rerank_enabled`, regardless of which eval modes will run — this is the thrash site the discipline exists for
- `ingest/scripts/phase_c_full.sh:run` — wrapper that exports `RERANK_ENABLED=false` per eval invocation unless args contain `--mode rerank` (76 s/q vs ~275 s/q measured)
- `ingest/ingest/__main__.py` — CLI search loads `BGEReranker` unless `--no-rerank` / `RERANK_ENABLED=false`
- [coordination/README.md](../coordination/README.md) — `reranker-ram.lock` convention in `coordination/locks/`: claim before loading reranker/embedder weights; convention only, no code enforces it

**If you change one side, also check:**
- Adding a reranker consumer or changing when `ingest/eval/evaluate.py:qdrant_deps` loads it → re-check `ingest/scripts/phase_c_full.sh:run`'s `--mode rerank` detection still gates correctly
- Changing the `RERANK_ENABLED` default or name in `ingest/ingest/config.py:load_config` → `phase_c_full.sh`, MCP launch env, and `ingest/ingest/mcp_server.py:_get_reranker` all read it
- Changing `BGEReranker.__init__` (device/thread pinning via `ingest/ingest/rerank.py:_configure_cpu_threads`) → the swap-thrash math changes; re-verify with `ingest/scripts/rerank_latency_probe.py`
- Adding any new model-loading process → claim `coordination/locks/reranker-ram.lock` per [coordination/README.md](../coordination/README.md)

**Breaks silently when:** a non-rerank eval or a second session loads the cross-encoder anyway (default-True flag, eager `qdrant_deps`, or a forgotten lock) — nothing errors; queries just degrade to ~275 s each as Qdrant pages through swap, and long eval runs quietly stretch from hours to days.
