# Ingest core (scrape → normalize → chunk → embed → upsert)
> Drill-down memory. Not auto-loaded — opened per the pre-modification ritual in INDEX.md. Anchors linted by ingest/scripts/gen_code_map.py --check.

## Overview
Turns scraped Georgian legal documents into the hybrid dense+sparse Qdrant index (`georgian_legal`). A Scrapy project (scraper/legal_scrapers, 7 spiders) writes per-run JSONL to `artifacts/<source>/runs/<run_id>/items.jsonl` with cross-run sqlite dedup keyed to the same identity ingest uses. The ingest/ package (standalone uv project) normalizes raw items via a per-source `SourceSpec` registry, cleans/quarantines text, detects legal structure, chunks with an offset-preserving structure-aware chunker, embeds with BGE-M3 (dense + learned-sparse), and upserts deterministic-UUIDv5 points with a rich payload. Two embed paths exist: the batch/watch pipeline reading RAW artifacts (`ingest/ingest/pipeline.py`), and the snapshot path (`ingest/ingest/embed_job.py`) reading the versioned clean snapshot `snapshots/v1` built by `ingest/ingest/snapshot.py`. GPU-scale work (full-corpus and delta embeds) runs on ephemeral RunPod pods via `ingest/scripts/runpod_orchestrate.py` and siblings, gated by a CPU-vs-GPU vector-space checksum (G2, cosine ≥ 0.999), with snapshot pull-back and explicit merge. Coverage ground truth is `ingest/scripts/verify_all_embedded.py` (id-parity with `SourceSpec.id_fields`). `ingest/scripts/publish_snapshot.py` + `ingest/serverless/qdrant_boot.py` implement a manifest-last publish protocol to the RunPod serverless read replica. Daily incremental operation is `ingest/scripts/daily_ingest.sh` (scrape → watch --once → verify) under a systemd timer (ingest/systemd/).

## Entrypoints
- `python -m legal_scrapers.run [--only spiders] [--start-date/--end-date]` — multi-spider scrape run; `scraper/legal_scrapers/run.py:main` + `scraper/legal_scrapers/run.py:select_spiders`. Also `scrapy crawl <spider>` from scraper/.
- `python -m ingest {ingest|watch|snapshot|embed|search}` — CLI dispatcher `ingest/ingest/__main__.py:main`; subcommands `ingest/ingest/__main__.py:_cmd_ingest`, `_cmd_watch`, `_cmd_snapshot`, `_cmd_embed`, `_cmd_search` (lazy imports; `--recreate` clears checkpoints/watch state first).
- `ingest/scripts/daily_ingest.sh` — stage 1 scrape (`legal_scrapers.run`, seen.sqlite keeps it delta-only) → stage 2 `python -m ingest watch --source all --once` → stage 3 `verify_all_embedded.py` (exit code gates success); flock self-exclusion + coordination/locks skip; timer units in ingest/systemd/.
- `ingest/scripts/runpod_orchestrate.py:main` — one-time full-corpus GPU embed: encrypt payload → provision pod → runpod_embed.sh in tmux → G2 gate (`ingest/scripts/runpod_orchestrate.py:step_verify_g2`) → `ingest/scripts/runpod_orchestrate.py:step_restore`; `ingest/scripts/runpod_orchestrate.py:terminate` for emergency cleanup.
- `ingest/scripts/runpod_orchestrate_delta.py:main` — incremental GPU embed of new matsne run items into `georgian_legal_delta` (drives runpod_embed_delta.sh; `--skip-pod` restores an already-pulled snapshot).
- `ingest/scripts/runpod_orchestrate_multi.py:main` — 4x-GPU sharded variant (standalone: own gql/provision/terminate, does NOT import runpod_orchestrate; pod-to-pod corpus pull via `ingest/scripts/runpod_orchestrate_multi.py:transfer_corpus`, N sharded `python -m ingest embed --shard i/N` processes).
- `ingest/scripts/merge_delta_collection.py:merge_collection` — upsert delta-collection points verbatim into `georgian_legal`; idempotent via deterministic point ids; `--dry-run` self-test.
- `ingest/scripts/verify_all_embedded.py:main` — coverage ground truth; writes `ingest/.state/embed_coverage.json` + `embed_missing.txt`.
- `ingest/scripts/verify_delta_embedded.py:delta_doc_ids` / `ingest/scripts/verify_matsne_completeness.py:audit_advertised` — delta coverage and live-site completeness audits.
- `ingest/scripts/backfill_consolidation.py:main` — payload-only set_payload of `is_consolidated` onto already-embedded matsne chunks (no re-embed).
- `ingest/scripts/publish_snapshot.py:main` — `--create/--upload/--verify/--cleanup` publish local collection to the RunPod network volume (boto3 lives in the `publish` dep group, kept out of the MCP env).
- `ingest/scripts/embed_delta.py:main` — delta embed from RAW run items (see Modules).

## Modules

### scraper/legal_scrapers/spiders/base.py
Shared Scrapy base: per-run artifact scaffolding and cross-run dedup.
- `scraper/legal_scrapers/spiders/base.py:BaseLegalSpider` — base for all 7 spiders; `BaseLegalSpider.configure_run_outputs` / `BaseLegalSpider.write_run_metadata` create `artifacts/<name>/runs/<run_id>/` + `latest/` that the whole downstream pipeline reads. called_by: every spider in scraper/legal_scrapers/spiders/, `scraper/legal_scrapers/run.py:main`.
- `scraper/legal_scrapers/spiders/base.py:BaseLegalSpider.open_dedup_store` / `BaseLegalSpider.dedup_key` / `BaseLegalSpider.is_seen` / `BaseLegalSpider.mark_seen` — cross-run dedup in `artifacts/<name>/seen.sqlite`, keyed on each spider's `DEDUP_KEY` class attr (matsne/napr/tas: document_id; ecd: decision_document_id; constcourt: legal_id; tbappeal: slug; supremecourt: case_id+chamber) — deliberately mirrors `ingest/ingest/sources.py:SourceSpec.id_fields` so "already scraped" == "already in the vector DB". called_by: spider parse callbacks; seen.sqlite read directly by `ingest/scripts/verify_matsne_completeness.py:load_seen_ids` and `ingest/scripts/session_monitor.py:_seen_total`.

### scraper/legal_scrapers/spiders/matsne_spider.py
Matsne (legislation) spider: phased search sweep + document detail parse.
- `scraper/legal_scrapers/spiders/matsne_spider.py:MatsneSpider.parse_document` — emits MatsneItem incl. `consolidated_dates`/`consolidated_count` and `is_consolidated = len(switcher options) >= 1` from `#publication-switcher` — the origin of the is_consolidated payload flag. called_by: scrapy engine.
- `scraper/legal_scrapers/spiders/matsne_spider.py:MatsneSpider._load_seed_urls` / `MatsneSpider._split_requests` — seed-id file re-fetch (residual_missing_ids) and recursive result-page splitting for the 1900-start sweep.

### scraper/legal_scrapers/spiders/ (ecd, constcourt, napr, tbappeal, supremecourt, tas)
Remaining source spiders. tas_spider.py is the largest: parses permit records into markdown bodies plus structured applicant PII fields later promoted into the Qdrant payload; uses Playwright with HTTP cache disabled.
- `scraper/legal_scrapers/spiders/tas_spider.py:TasSpider.build_item` / `TasSpider._enrich` — build TasItem with applicant_* fields consumed by `SourceSpec` promote_fields. called_by: scrapy engine.

### ingest/ingest/sources.py
Canonical normalization layer: per-source declarative field maps → CanonicalDoc, plus date/status normalization and schema-drift detection.
- `ingest/ingest/sources.py:SourceSpec` — declarative spec; `id_fields` defines document identity (document_id = joined id_fields values); `promote_fields` copies raw keys (tas PII) verbatim into the payload; consolidation fields wire matsne. called_by: `normalize`, `ingest/scripts/verify_all_embedded.py:scraped_universe` (id-parity).
- `ingest/ingest/sources.py:SOURCES` — registry of 7 specs (matsne, ecd, constcourt, napr, tbappeal, supremecourt, tas). called_by: `ingest/ingest/pipeline.py:resolve_sources`, `ingest/ingest/snapshot.py:build_snapshot`, `ingest/scripts/verify_all_embedded.py:scraped_universe`.
- `ingest/ingest/sources.py:CanonicalDoc` — frozen normalized doc (title/date/status/parties/promoted/is_consolidated/consolidated_count/body_markdown). called_by: `ingest/ingest/qdrant_store.py:build_payload`, `ingest/ingest/snapshot.py:_snapshot_record`, `ingest/ingest/embed_job.py:snapshot_doc_to_canonical`.
- `ingest/ingest/sources.py:normalize` — source + raw item → CanonicalDoc via `SourceSpec.build`. called_by: `ingest/ingest/pipeline.py:ingest_source`, `ingest/ingest/snapshot.py:build_snapshot`, `ingest/scripts/embed_delta.py:main`.
- `ingest/ingest/sources.py:schema_drift` — detect newly-appearing raw keys vs `SourceSpec.declared_keys`. called_by: `ingest/ingest/pipeline.py:_record_schema_drift` (watch path).
- `ingest/ingest/sources.py:PROMOTED_KEYWORD_FIELDS` / `PROMOTED_TEXT_FIELDS` — which promoted PII keys get Qdrant keyword/full-text indexes. called_by: `ingest/ingest/qdrant_store.py:ensure_collection`.

### ingest/ingest/hygiene.py
Text cleaning + damage triage.
- `ingest/ingest/hygiene.py:clean_text` — `strip_control` + `to_nfc`; produces the canonical clean body every downstream hash/offset is computed over. called_by: `ingest/ingest/snapshot.py:build_snapshot`.
- `ingest/ingest/hygiene.py:assess` — DamageReport; quarantine_reason ∈ empty_body/near_empty/mojibake (`NEAR_EMPTY_CHARS`=30, `MOJIBAKE_RATIO`=1%). called_by: `ingest/ingest/snapshot.py:build_snapshot`.

### ingest/ingest/dedup.py
Content hashing and duplicate clustering (report-only, never merged).
- `ingest/ingest/dedup.py:content_hash` — stable doc-content hash; stamped on every chunk payload; watch's skip signal. called_by: `ingest/ingest/qdrant_store.py:build_payload`, `ingest/ingest/snapshot.py:build_snapshot`, `ingest/ingest/pipeline.py:_indexed_content_hash` (comparison).
- `ingest/ingest/dedup.py:near_dup_clusters` / `cluster_by_key` / `cluster_stats` — MinHash/LSH near-dups (Jaccard ≥ 0.85) and matsne amendment groups for snapshot reports. called_by: `ingest/ingest/snapshot.py:_near_dup_for_source`, `build_snapshot`.

### ingest/ingest/structure.py
Legal-structure detection on Georgian text (article markers "მუხლი N", headings, clauses).
- `ingest/ingest/structure.py:detect` — StructureInfo (primary_kind, counts) recorded per snapshot doc. called_by: `ingest/ingest/snapshot.py:build_snapshot`.
- `ingest/ingest/structure.py:article_spans` — article offset spans (eval-side consumer).

### ingest/ingest/chunking.py
THE offset chunker: structure-aware Markdown chunking, token-bounded with overlap; each Chunk carries half-open [char_start, char_end) offsets into the ORIGINAL input text's own coordinate space, so gold evidence spans survive re-chunking.
- `ingest/ingest/chunking.py:chunk_document` — sections split on ATX headings AND article lines (`_ARTICLE_LINE_RE`, `[ \t]`-only whitespace so a line-final მუხლი can't bind to a next-line digit); `_split_sections` → `_atoms` → `_pack` to Config-driven max_tokens=512/overlap=80/min_tokens=64; injectable count_tokens. called_by: `ingest/ingest/pipeline.py:ingest_source`, `ingest/ingest/pipeline.py:_build_doc_points`.
- `ingest/ingest/chunking.py:build_embed_text` — prepends "title > document_type > heading_path" context to the EMBEDDED text only; stored `Chunk.text` stays clean. called_by: `ingest/ingest/pipeline.py:ingest_source`, `_build_doc_points`.
- `ingest/ingest/chunking.py:Chunk` — text/chunk_index/heading_path/token_count/char_start/char_end. called_by: `ingest/ingest/qdrant_store.py:build_payload`, `ingest/eval/spanmap.py:chunks_covering_span` (offset consumer).

### ingest/ingest/qdrant_store.py
All Qdrant I/O primitives: client, collection schema (named dense 1024-d + sparse vectors, payload indexes), deterministic point ids, payload construction.
- `ingest/ingest/qdrant_store.py:point_id` — UUIDv5(`NAMESPACE`, "source:document_id:chunk_index") — deterministic; makes every upsert path idempotent. called_by: `ingest/ingest/pipeline.py:_build_doc_points`, `ingest/ingest/pipeline.py:_indexed_content_hash`, `ingest/ingest/pipeline.py:ingest_source` (ids copied verbatim by merge_delta_collection).
- `ingest/ingest/qdrant_store.py:build_payload` — full chunk payload: canonical fields incl. is_consolidated/consolidated_count, RFC3339 dates (`_rfc3339`), heading, char_start/char_end, text, doc-level `content_hash(doc.body_markdown)` on every chunk; promoted PII merged via setdefault (can never overwrite canonical keys). calls: `ingest/ingest/dedup.py:content_hash`. called_by: `ingest/ingest/pipeline.py:ingest_source`, `_build_doc_points`.
- `ingest/ingest/qdrant_store.py:ensure_collection` — create/verify collection: BOOL(is_consolidated)/INTEGER(consolidated_count)/DATETIME/KEYWORD/TEXT payload indexes + promoted-PII indexes; `_assert_dense_dim` guards dim mismatch. called_by: `ingest/ingest/__main__.py` subcommands, `ingest/scripts/merge_delta_collection.py:merge_collection`, `ingest/scripts/embed_delta.py:main`.
- `ingest/ingest/qdrant_store.py:delete_doc_chunks_from` — drop stale chunks (chunk_index ≥ new count) on re-ingest of a shorter doc. called_by: `ingest/ingest/pipeline.py:ingest_source`, `watch_drain_source`.
- `ingest/ingest/qdrant_store.py:upsert_points` / `make_client` / `sparse_vector` / `point_struct` — thin wrappers; upsert wait=True on checkpoint boundaries. called_by: pipeline, embed_job, scripts.

### ingest/ingest/embedding.py
BGE-M3 wrapper producing dense + learned-sparse vectors; env-profiled (EMBED_DEVICE/FP16/BATCH).
- `ingest/ingest/embedding.py:BGEM3Embedder` — `BGEM3Embedder.encode_passages` / `BGEM3Embedder.encode_query` → Embedded(dense, Sparse). called_by: `ingest/ingest/__main__.py` commands, `ingest/scripts/embed_delta.py:main`, mcp_server (query side, see retrieval-serving.md).
- `ingest/ingest/embedding.py:make_token_counter` — real BGE-M3 tokenizer counter for chunk budgets (tests use `default_token_counter` word proxy). called_by: `ingest/ingest/__main__.py`, `ingest/ingest/snapshot.py:_maybe_token_counter`, `ingest/scripts/embed_delta.py:main`.

### ingest/ingest/config.py
Frozen env-driven Config (Qdrant URL/collection, search_backend local|remote, RunPod endpoint, embed/rerank/chunk knobs, artifacts_root, state_dir).
- `ingest/ingest/config.py:load_config` — reads env once at call time; defaults COLLECTION_NAME=georgian_legal, CHUNK_TOKENS=512/CHUNK_OVERLAP=80/CHUNK_MIN_TOKENS=64. called_by: `ingest/ingest/__main__.py`, every scripts/*.py, mcp_server, `ingest/serverless/handler.py`.
- `ingest/ingest/config.py:retrieval_fingerprint` — 16-hex digest of retrieval-determining knobs (incl. chunk config); stamped into MCP responses + query log. called_by: `ingest/ingest/mcp_server.py` (layering rule: ingest never imports eval).

### ingest/ingest/pipeline.py
Batch + watch ingestion from RAW artifacts: read JSONL → normalize → chunk → embed → upsert; idempotent, resumable; checkpoints advance only over acknowledged (wait=True) upserts.
- `ingest/ingest/pipeline.py:ingest_source` — main batch loop; `--resume` fast-forward with loud failure if checkpoint id never found; per-doc stale-chunk delete; returns (docs, chunks, skipped). calls: `ingest/ingest/sources.py:normalize`, `ingest/ingest/chunking.py:chunk_document`, `build_embed_text`, `ingest/ingest/embedding.py:BGEM3Embedder.encode_passages`, `ingest/ingest/qdrant_store.py:point_id` / `build_payload` / `upsert_points` / `delete_doc_chunks_from`. called_by: `ingest/ingest/__main__.py:_cmd_ingest`.
- `ingest/ingest/pipeline.py:_build_doc_points` — per-doc core (chunk + embed + points), the single place chunk/embed/payload logic is defined for the watch and snapshot paths. calls: `chunk_document`, `build_embed_text`, `ingest/ingest/qdrant_store.py:build_payload`. called_by: `ingest/ingest/embed_job.py:embed_docs`, `embed_source_resumable`, `ingest/ingest/pipeline.py:watch_drain_source`.
- `ingest/ingest/pipeline.py:watch_loop` / `watch_drain_source` — tail artifacts by byte offset (`_read_complete_lines`), skip unchanged docs via `_indexed_content_hash` (O(1) chunk-0 content_hash lookup), schema-drift recording (`_record_schema_drift`), per-source watch state in `.state/`. called_by: `ingest/ingest/__main__.py:_cmd_watch`; daily_ingest.sh stage 2 (`watch --source all --once`).
- `ingest/ingest/pipeline.py:items_path` / `discover_runs` — `artifacts/<source>/{latest,runs}/items.jsonl` resolution. called_by: `ingest_source`, `ingest/scripts/backfill_consolidation.py:_flush`.
- `ingest/ingest/pipeline.py:delete_checkpoint` / `delete_watch_state` — invalidated-state cleanup on `--recreate`. called_by: `ingest/ingest/__main__.py`.

### ingest/ingest/snapshot.py
Builds the versioned clean corpus snapshot (snapshots/v1): unions ALL runs per source newest-first keep-first, normalizes, cleans, quarantines, hashes, detects structure, clusters dups; writes `docs/<source>.jsonl` + `quarantine.jsonl` + reports + `manifest.json`. Reports carry counts only, never body text/PII.
- `ingest/ingest/snapshot.py:build_snapshot` — main driver over `SOURCES_PRESENT` (6 sources — supremecourt excluded by design). calls: `ingest/ingest/sources.py:normalize`, `ingest/ingest/hygiene.py:assess` / `clean_text`, `ingest/ingest/dedup.py:content_hash` / `cluster_by_key` / `near_dup_clusters`, `ingest/ingest/structure.py:detect`, `_write_reports_and_manifest`. called_by: `ingest/ingest/__main__.py:_cmd_snapshot`.
- `ingest/ingest/snapshot.py:_snapshot_record` — snapshot doc schema; persists promoted + structure + CLEANED body but NOT is_consolidated/consolidated_count (see Hazards). called_by: `build_snapshot`.
- `ingest/ingest/snapshot.py:config_hash` — pipeline-version stamp in the manifest for traceability.

### ingest/ingest/embed_job.py
Full-corpus embedding from the CLEAN SNAPSHOT (not raw artifacts) — one code path, CPU pilot vs RunPod GPU profiles; shard-aware resumable; G2 vector-space checksum.
- `ingest/ingest/embed_job.py:embed_source_resumable` — snapshot-source embed with per-source (or per-shard `<source>.shardIofN`) checkpoint; shard=(i,n) → every n-th doc, disjoint by deterministic point ids so N GPUs share one Qdrant. calls: `iter_snapshot_docs`, `ingest/ingest/pipeline.py:_build_doc_points`, `ingest/ingest/qdrant_store.py:upsert_points`. called_by: `ingest/ingest/__main__.py:_cmd_embed` (which runpod_embed.sh/runpod_embed_multi.sh invoke as `python -m ingest embed` on-pod).
- `ingest/ingest/embed_job.py:embed_docs` — non-resumable list-of-docs variant. calls: `_build_doc_points`. called_by: `ingest/ingest/__main__.py:_cmd_embed`, `ingest/scripts/embed_delta.py:main`.
- `ingest/ingest/embed_job.py:snapshot_doc_to_canonical` — snapshot record → CanonicalDoc; extra={} and consolidation fields default to None (snapshot doesn't carry them). called_by: `iter_snapshot_docs`.
- `ingest/ingest/embed_job.py:dense_checksum` / `checksum_cosine` / `save_checksum_reference` — G2 guardrail: embed `CHECKSUM_SENTENCE` in both envs; orchestrator gates restore on cosine ≥ 0.999. called_by: `ingest/ingest/__main__.py:_cmd_embed` (flags), `ingest/scripts/runpod_orchestrate.py:step_verify_g2`, `ingest/scripts/runpod_orchestrate_delta.py:step_verify_g2_delta`.

### ingest/ingest/__main__.py + progress.py
CLI dispatcher with lazy imports (search doesn't load ingest deps and vice versa) and a rich.Live multi-source progress panel.
- `ingest/ingest/__main__.py:main` — argparse over ingest/watch/snapshot/embed/search; `--recreate` clears checkpoints/watch state first.
- `ingest/ingest/progress.py:IngestProgress` — optional TTY-gated live panel; results printed after Live closes. called_by: `ingest/ingest/__main__.py:_cmd_ingest`, `ingest/ingest/pipeline.py:ingest_source`.

### ingest/scripts/embed_delta.py
Delta embed from RAW run items — deliberately NOT the snapshot loader, so is_consolidated/consolidated_count/status survive — into `--collection` (georgian_legal_delta on the pod).
- `ingest/scripts/embed_delta.py:main` — `--runs-since <run-id>` filter (`_resolve_paths`) or explicit `--items`; dedup by document_id (last wins); calls: `ingest/ingest/sources.py:normalize`, `ingest/ingest/embed_job.py:embed_docs`, `ingest/ingest/qdrant_store.py:ensure_collection`. called_by: runpod_embed_delta.sh (on-pod).

### ingest/scripts/runpod_orchestrate.py (+ _delta, _multi)
Full-corpus GPU embed orchestrator: encrypt payload (tar+openssl, passphrase memory-only) → provision Secure-Cloud GPU via browser-UA GraphQL (`gql`) → ssh/rsync in → tmux runpod_embed.sh → poll `out/DONE` → pull snapshot + GPU checksum → G2 gate → guaranteed podTerminate (try/finally + atexit + signals) → restore into local Qdrant.
- `ingest/scripts/runpod_orchestrate.py:step_provision` / `step_poll` / `step_transfer_out` / `step_verify_g2` / `step_restore` / `terminate` — pipeline steps; module constants pin `COLLECTION` = georgian_legal, `QDRANT_VER` = v1.18.2 (matched to LOCAL Qdrant), `COS_GATE` = 0.999, `BUDGET` = $15.
- `ingest/scripts/runpod_orchestrate_delta.py:main` — incremental sibling; `import runpod_orchestrate as O` reuses gql/ssh/provision/terminate helpers; ships only `delta_items/*.jsonl` (`stage_delta_items`), drives runpod_embed_delta.sh, pulls the small delta snapshot (`step_restore_delta`); merge stays a separate explicit step. `_retry` wraps rsync `--partial --append-verify` for flaky uplinks.
- `ingest/scripts/runpod_orchestrate_multi.py:main` — standalone 4x-GPU variant (own `provision`/`launch`/`poll`/`restore`; does NOT import runpod_orchestrate); pod-to-pod corpus pull, N sharded `-m ingest embed --shard i/N` processes via runpod_embed_multi.sh.

### ingest/scripts/merge_delta_collection.py
Scrolls every point of the delta collection and upserts verbatim (dense+sparse+payload) into `georgian_legal`; idempotent because point ids are UUIDv5-deterministic.
- `ingest/scripts/merge_delta_collection.py:merge_collection` — scroll → `_to_struct` → upsert loop; `dry_run` self-test. calls: `ingest/ingest/qdrant_store.py:make_client`, `ensure_collection`. called_by: operator, after runpod_orchestrate_delta.

### ingest/scripts/verify_all_embedded.py
Coverage ground truth: scraped universe minus quarantined/no-text ids vs one full collection scroll of (source, document_id).
- `ingest/scripts/verify_all_embedded.py:scraped_universe` — per-source ids derived EXACTLY like ingest via `SourceSpec.id_fields` join. calls: `ingest/ingest/sources.py:SOURCES`. called_by: daily_ingest.sh stage 3, operator.
- `ingest/scripts/verify_all_embedded.py:quarantined_ids` / `no_text_ids` — exclude snapshots/v1/quarantine.jsonl + no-text docs so exit 0 is reachable (278 excluded by design; 2026-07-09: VERIFIED 0 missing).
- `ingest/scripts/verify_all_embedded.py:embedded_universe` — one full scroll (~3 min); output `.state/embed_coverage.json` is rendered by `ingest/scripts/session_monitor.py:coverage_state` (:8770).

### ingest/scripts/publish_snapshot.py + ingest/serverless/
Two-sided manifest-last publish protocol to the RunPod serverless read replica.
- `ingest/scripts/publish_snapshot.py:create` / `upload` / `verify` / `cleanup` — snapshot the live collection, resumable S3 multipart upload (`PART_SIZE` = 95MB — the gateway 413s larger bodies), snapshot object FIRST and `publish/manifest.json` LAST (`_publish_manifest_object`, atomic completion signal); `--verify` triggers/waits the worker restore.
- `ingest/serverless/qdrant_boot.py:ensure_running` / `maybe_restore` / `needs_restore` — boot a worker-local Qdrant from the network volume; compare publish/manifest.json to publish/ACTIVE (sha256-verify, file:// snapshot-recover, point-count check); spawn-retry because Qdrant holds an exclusive storage lock during worker rollover (endpoint must run max workers = 1). called_by: `ingest/serverless/handler.py:_boot`.
- `ingest/serverless/handler.py:handler` — dispatches ops (`OPS`: search/get_document/lookup/browse/versions/collection_info/health, plus refresh) directly to `ingest.mcp_server` tool coroutines; boot is fail-soft (no billed crash-loops).

### ingest/scripts/runpod_embed.sh / runpod_embed_delta.sh / runpod_embed_multi.sh
On-pod bootstrap shell: decrypt payload, venv, local Qdrant (pinned QDRANT_VER), run `python -m ingest embed` (full/multi) or `scripts/embed_delta.py --items` (delta) under tmux; emit `out/<collection>.snapshot` + `checksum_gpu.json` + `DONE`. Idempotent (skips untar/venv/qdrant if present). called_by: runpod_orchestrate*.py over ssh.

## Artifact flows
- **Produces:** `artifacts/<source>/runs/<run_id>/items.jsonl` + `latest/` mirror + `run_metadata` (spiders); `artifacts/<source>/seen.sqlite` (cross-run dedup); `ingest/snapshots/v1/` (docs/<source>.jsonl, quarantine.jsonl, reports/, manifest.json, checksum_cpu.json); Qdrant collections `georgian_legal` (2.64M points) and transient `georgian_legal_delta`; `ingest/.state/` (`<source>.ckpt`/embed checkpoints, watch state, `embed_coverage.json`, `embed_missing.txt`, `publish/` upload state); RunPod network volume `publish/` (snapshot + manifest.json + ACTIVE).
- **Consumes:** raw artifacts (pipeline batch/watch, snapshot builder, embed_delta, verify_all_embedded); snapshots/v1 (embed_job, verify_all_embedded quarantine exclusion); seen.sqlite (verify_matsne_completeness, session_monitor); pod-side `out/{collection}.snapshot`, `checksum_gpu.json`, `DONE` (orchestrators).
- **Environment:** `ingest/.env` via `ingest/ingest/config.py:load_config`; systemd units in ingest/systemd/ run daily_ingest.sh and the monitor.

## Hazards
- **Snapshot drops consolidation:** `ingest/ingest/snapshot.py:_snapshot_record` does NOT persist is_consolidated/consolidated_count, and `ingest/ingest/embed_job.py:snapshot_doc_to_canonical` rebuilds CanonicalDoc with them defaulting to None — any re-embed from snapshots/v1 silently NULLs the flag on matsne chunks. This is exactly why `ingest/scripts/embed_delta.py` reads RAW items and why `ingest/scripts/backfill_consolidation.py` exists. A fix requires snapshot v2 schema + loader change together.
- **Three-way id parity is maintained by hand:** `ingest/ingest/sources.py:SourceSpec.id_fields` == spider `DEDUP_KEY` (scraper/legal_scrapers/spiders/*) == `ingest/scripts/verify_all_embedded.py:scraped_universe` derivation. Drift makes dedup or coverage verification silently wrong (ecd items carry BOTH document_id and decision_document_id — the wrong pick "works" but mismatches payloads).
- **Point identity is frozen:** `ingest/ingest/qdrant_store.py:NAMESPACE` and the `point_id` format ("source:document_id:chunk_index") define identity; changing either orphans all 2.64M points and breaks merge/watch idempotency invisibly (upserts just start duplicating).
- **content_hash is the watch skip signal** (`ingest/ingest/pipeline.py:_indexed_content_hash` vs chunk-0 payload): any change to `ingest/ingest/hygiene.py:clean_text`, NFC handling, or hashing changes every hash → watch re-embeds the whole corpus. Also: pipeline hashes RAW body_markdown (via `ingest/ingest/qdrant_store.py:build_payload`) while snapshot hashes the CLEANED body — two hash spaces by design; don't "unify" them casually.
- **Offsets are load-bearing for eval:** `Chunk.char_start`/`char_end` are offsets into the input body in its own coordinate space; the golden set's evidence spans depend on them (`ingest/eval/spanmap.py:chunks_covering_span`). Changing hygiene cleaning or chunker section/strip logic shifts spans and silently invalidates golden-set relevance — re-baseline per improvement.md protocol.
- **Checkpoint semantics:** per-source/per-shard checkpoints only advance over wait=True upserts; `--recreate` must clear checkpoints/watch state (done in `ingest/ingest/__main__.py:main`) — invoking pipeline functions directly after recreating a collection without deleting `.state` checkpoints fast-forwards past docs that no longer exist. Resume-id-never-found raises loudly by design; deleting the checkpoint is the recovery.
- **G2 vector-space gate:** snapshots/v1/checksum_cpu.json must exist and GPU cosine ≥ `ingest/scripts/runpod_orchestrate.py:COS_GATE` (0.999) before restoring a GPU-embedded snapshot; skipping it risks a mixed vector space that degrades retrieval with no error anywhere.
- **RunPod:** the API needs a browser User-Agent (Cloudflare 403 error 1010); pods MUST die via the orchestrators' terminate guarantees or `runpod_rerank.py down` (an orphaned billing pod has already happened); snapshot pulls have a 1800s timeout after a truncated-tar incident — `tar -tf` any snapshot before trusting it; `QDRANT_VER` is pinned to the LOCAL Qdrant version so restores work.
- **Publish ordering is load-bearing:** snapshot object uploads FIRST, publish/manifest.json LAST; `ingest/serverless/qdrant_boot.py:maybe_restore` treats the manifest as the atomic completion signal, and the endpoint must run max workers = 1 (Qdrant storage lock; rollover overlap handled by spawn-retry). `PART_SIZE` capped at 95MB (gateway 413s larger despite documented limits).
- **Config/env caching:** Config is frozen and read at `load_config()` call; the legal_rag MCP stdio server caches code+.env at spawn — after editing ingest/, reconnect via /mcp or it silently serves stale code.
- **supremecourt is scraped but excluded from the corpus by design:** `ingest/ingest/snapshot.py:SOURCES_PRESENT` and `ingest/ingest/embed_job.py:SOURCES` list 6 sources, and daily_ingest.sh's default source list matches; adding it to one list but not the others creates partial-corpus confusion.
- **daily_ingest.sh discipline:** flock self-exclusion + coordination/locks check; its lock-skip (exit 0) is intentional, not a failure — index writes during an eval turn the collection yellow and break hybrid queries.
- **RAM:** this 30GB box cannot co-run Qdrant + reranker + desktop; loading the reranker in non-rerank eval modes evicts Qdrant's sparse index into swap (~275s/query) — RERANK_ENABLED=false for non-rerank runs is baked into scripts/phase_c_full.sh and must be preserved.
- **Never edit existing tests to make them pass** (ingest/tests/ pins chunker offset semantics — test_chunk_offsets.py, test_chunking.py — and consolidation/delta behavior).

## Cross-area
- **retrieval-serving (see retrieval-serving.md):** `ingest/ingest/mcp_server.py` is the read path over the index this area builds — imports `ingest/ingest/config.py:load_config` / `retrieval_fingerprint`, `ingest/ingest/qdrant_store.py:make_client`, `ingest/ingest/embedding.py:BGEM3Embedder` (query side), `ingest/ingest/search.py:hybrid_search`. `ingest/serverless/handler.py:OPS` dispatches to the same mcp_server tool coroutines; the MCP thin-client reaches the serverless worker via SEARCH_BACKEND=remote (`ingest/ingest/remote_search.py:RunPodQueueClient`), whose publish side is this area's `ingest/scripts/publish_snapshot.py`.
- **eval (see eval-harness.md):** `ingest/eval/spanmap.py:chunks_covering_span` / `map_spans_to_chunks` consume `ingest/ingest/chunking.py:Chunk` offsets to map golden evidence spans → chunks under any chunking config; eval/backend.py drives `ingest/ingest/search.py:hybrid_search` over the same collection. Layering rule: eval → ingest, never the reverse.
- **scraper → ingest:** `artifacts/<source>/runs/*/items.jsonl` produced by `scraper/legal_scrapers/spiders/base.py:BaseLegalSpider.configure_run_outputs` is consumed by `ingest/ingest/pipeline.py:items_path` / `watch_loop`, `ingest/ingest/snapshot.py:_run_files_desc`, `ingest/scripts/verify_all_embedded.py:scraped_universe`, `ingest/scripts/embed_delta.py:_resolve_paths`; `seen.sqlite` is read by `ingest/scripts/verify_matsne_completeness.py:load_seen_ids`.
- **ops/monitoring (see ops-monitoring.md):** `ingest/scripts/verify_all_embedded.py:main` writes `ingest/.state/embed_coverage.json`, rendered by `ingest/scripts/session_monitor.py:coverage_state` (:8770); `ingest/scripts/session_monitor.py:_seen_total` reads seen.sqlite read-only.
