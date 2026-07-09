# Symbol map (AUTO-GENERATED — do not edit)

Regenerate: `python3 ingest/scripts/gen_code_map.py` · Lint: `--check`.
One section per module: internal-import edges, then top-level classes (with methods)
and functions, each anchored `file.py:LINE`. Line numbers here are kept fresh by the
generator; hand-written memory-bank files must use `path.py:symbol` anchors instead.

## ingest/ingest/__init__.py

## ingest/ingest/__main__.py
imports: ingest/ingest/__init__.py, ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/progress.py, ingest/ingest/rerank.py
- def **_resolved_cfg** `ingest/ingest/__main__.py:10`
- def **_cmd_ingest** `ingest/ingest/__main__.py:17`
- def **_cmd_watch** `ingest/ingest/__main__.py:72`
- def **_cmd_snapshot** `ingest/ingest/__main__.py:100`
- def **_cmd_embed** `ingest/ingest/__main__.py:112`
- def **_cmd_search** `ingest/ingest/__main__.py:185`
- def **main** `ingest/ingest/__main__.py:218`

## ingest/ingest/chunking.py
- def **default_token_counter** `ingest/ingest/chunking.py:35`
- class **Chunk**() `ingest/ingest/chunking.py:41`
- def **_split_keep_pos** `ingest/ingest/chunking.py:53`
- def **_strip_span** `ingest/ingest/chunking.py:69`
- def **_split_sections** `ingest/ingest/chunking.py:76`
- def **heading_spans** `ingest/ingest/chunking.py:130`
- def **_atoms** `ingest/ingest/chunking.py:152`
- def **_pack** `ingest/ingest/chunking.py:201`
- def **chunk_document** `ingest/ingest/chunking.py:244`
- def **build_embed_text** `ingest/ingest/chunking.py:279`

## ingest/ingest/config.py
- def **_bool** `ingest/ingest/config.py:17`
- def **_int** `ingest/ingest/config.py:24`
- def **_float_opt** `ingest/ingest/config.py:29`
- def **_device_opt** `ingest/ingest/config.py:40`
- class **Config**() `ingest/ingest/config.py:54`
- def **load_config** `ingest/ingest/config.py:85`
- def **retrieval_fingerprint** `ingest/ingest/config.py:120`

## ingest/ingest/dedup.py
- def **content_hash** `ingest/ingest/dedup.py:30`
- class **Cluster**() `ingest/ingest/dedup.py:36`
  - def size `ingest/ingest/dedup.py:43`
- def **cluster_by_key** `ingest/ingest/dedup.py:47`
- def **_shingles** `ingest/ingest/dedup.py:61`
- class **NearDupResult**() `ingest/ingest/dedup.py:69`
- def **near_dup_clusters** `ingest/ingest/dedup.py:75`
- def **cluster_stats** `ingest/ingest/dedup.py:129`

## ingest/ingest/embed_job.py
imports: ingest/ingest/__init__.py, ingest/ingest/config.py, ingest/ingest/pipeline.py, ingest/ingest/sources.py
- def **snapshot_doc_to_canonical** `ingest/ingest/embed_job.py:36`
- def **iter_snapshot_docs** `ingest/ingest/embed_job.py:61`
- def **load_snapshot_docs** `ingest/ingest/embed_job.py:76`
- def **dense_checksum** `ingest/ingest/embed_job.py:91`
- def **checksum_cosine** `ingest/ingest/embed_job.py:106`
- def **save_checksum_reference** `ingest/ingest/embed_job.py:116`
- def **_checkpoint_path** `ingest/ingest/embed_job.py:124`
- def **_load_ckpt** `ingest/ingest/embed_job.py:128`
- def **_save_ckpt** `ingest/ingest/embed_job.py:133`
- def **embed_docs** `ingest/ingest/embed_job.py:140`
- def **embed_source_resumable** `ingest/ingest/embed_job.py:166`

## ingest/ingest/embedding.py
imports: ingest/ingest/config.py
- class **Sparse**() `ingest/ingest/embedding.py:14`
- class **Embedded**() `ingest/ingest/embedding.py:20`
- class **BGEM3Embedder**() `ingest/ingest/embedding.py:25`
  - def __init__ `ingest/ingest/embedding.py:26`
  - def _encode `ingest/ingest/embedding.py:42`
  - def encode_passages `ingest/ingest/embedding.py:62`
  - def encode_query `ingest/ingest/embedding.py:65`
- def **make_token_counter** `ingest/ingest/embedding.py:69`

## ingest/ingest/hygiene.py
- def **strip_control** `ingest/ingest/hygiene.py:35`
- def **to_nfc** `ingest/ingest/hygiene.py:43`
- def **clean_text** `ingest/ingest/hygiene.py:48`
- class **DamageReport**() `ingest/ingest/hygiene.py:58`
  - def is_usable `ingest/ingest/hygiene.py:71`
- def **assess** `ingest/ingest/hygiene.py:75`

## ingest/ingest/mcp_server.py
imports: ingest/ingest/__init__.py, ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/querylog.py, ingest/ingest/remote_search.py, ingest/ingest/rerank.py, ingest/ingest/search.py, ingest/ingest/sources.py
- def **_get_cfg** `ingest/ingest/mcp_server.py:68`
- def **_get_client** `ingest/ingest/mcp_server.py:75`
- def **_get_embedder** `ingest/ingest/mcp_server.py:82`
- def **_get_reranker** `ingest/ingest/mcp_server.py:94`
- def **_is_remote_reranker** `ingest/ingest/mcp_server.py:117`
- def **_use_remote** `ingest/ingest/mcp_server.py:127`
- def **_get_remote_client** `ingest/ingest/mcp_server.py:131`
- def **_remote_op** `ingest/ingest/mcp_server.py:143`
- def **_publish_manifest** `ingest/ingest/mcp_server.py:160`
- def **_handle_error** `ingest/ingest/mcp_server.py:169`
- class **ResponseFormat**(str, Enum) `ingest/ingest/mcp_server.py:181`
- def **_hit_dict** `ingest/ingest/mcp_server.py:188`
- def **_format_hit_md** `ingest/ingest/mcp_server.py:217`
- class **SearchInput**(BaseModel) `ingest/ingest/mcp_server.py:243`
- def **legal_search** `ingest/ingest/mcp_server.py:324`
- def **_log_query_remote** `ingest/ingest/mcp_server.py:428`
- def **_log_query** `ingest/ingest/mcp_server.py:452`
- class **GetDocumentInput**(BaseModel) `ingest/ingest/mcp_server.py:473`
- def **_scroll_all** `ingest/ingest/mcp_server.py:490`
- def **_stitch_overlap** `ingest/ingest/mcp_server.py:509`
- def **legal_get_document** `ingest/ingest/mcp_server.py:547`
- def **_dedup_documents** `ingest/ingest/mcp_server.py:622`
- def **_format_doc_line** `ingest/ingest/mcp_server.py:666`
- class **LookupInput**(BaseModel) `ingest/ingest/mcp_server.py:682`
- def **legal_lookup** `ingest/ingest/mcp_server.py:717`
- class **BrowseInput**(BaseModel) `ingest/ingest/mcp_server.py:764`
- def **legal_browse** `ingest/ingest/mcp_server.py:799`
- def **_source_counts** `ingest/ingest/mcp_server.py:854`
- def **legal_collection_info** `ingest/ingest/mcp_server.py:873`
- def **_latest_report** `ingest/ingest/mcp_server.py:936`
- class **StatusInput**(BaseModel) `ingest/ingest/mcp_server.py:946`
- def **ingest_status** `ingest/ingest/mcp_server.py:966`
- class **GetVersionsInput**(BaseModel) `ingest/ingest/mcp_server.py:1030`
- def **legal_get_document_versions** `ingest/ingest/mcp_server.py:1052`
- def **legal_health** `ingest/ingest/mcp_server.py:1128`
- def **_is_our_server** `ingest/ingest/mcp_server.py:1168`
- def **_enforce_singleton** `ingest/ingest/mcp_server.py:1178`
- def **main** `ingest/ingest/mcp_server.py:1202`

## ingest/ingest/pipeline.py
imports: ingest/ingest/__init__.py, ingest/ingest/chunking.py, ingest/ingest/config.py, ingest/ingest/dedup.py, ingest/ingest/sources.py
- def **_indexed_content_hash** `ingest/ingest/pipeline.py:31`
- def **_record_schema_drift** `ingest/ingest/pipeline.py:47`
- def **write_ingest_report** `ingest/ingest/pipeline.py:63`
- def **items_path** `ingest/ingest/pipeline.py:91`
- def **_iter_lines** `ingest/ingest/pipeline.py:95`
- def **_checkpoint_path** `ingest/ingest/pipeline.py:103`
- def **_load_checkpoint** `ingest/ingest/pipeline.py:107`
- def **_save_checkpoint** `ingest/ingest/pipeline.py:114`
- def **delete_checkpoint** `ingest/ingest/pipeline.py:121`
- def **ingest_source** `ingest/ingest/pipeline.py:125`
- def **resolve_sources** `ingest/ingest/pipeline.py:262`
- def **discover_runs** `ingest/ingest/pipeline.py:281`
- def **_read_complete_lines** `ingest/ingest/pipeline.py:299`
- def **_watch_state_path** `ingest/ingest/pipeline.py:329`
- def **_load_watch_state** `ingest/ingest/pipeline.py:333`
- def **_save_watch_state** `ingest/ingest/pipeline.py:342`
- def **delete_watch_state** `ingest/ingest/pipeline.py:351`
- def **_build_doc_points** `ingest/ingest/pipeline.py:355`
- def **watch_drain_source** `ingest/ingest/pipeline.py:384`
- def **watch_loop** `ingest/ingest/pipeline.py:502`

## ingest/ingest/progress.py
- def **_fmt_elapsed** `ingest/ingest/progress.py:44`
- def **_rate_per_min** `ingest/ingest/progress.py:51`
- def **_quiet_background_noise** `ingest/ingest/progress.py:57`
- class **_Row**() `ingest/ingest/progress.py:79`
  - def elapsed `ingest/ingest/progress.py:89`
- class **IngestProgress**() `ingest/ingest/progress.py:97`
  - def __init__ `ingest/ingest/progress.py:106`
  - def __enter__ `ingest/ingest/progress.py:116`
  - def __exit__ `ingest/ingest/progress.py:125`
  - def add_source `ingest/ingest/progress.py:136`
  - def start_source `ingest/ingest/progress.py:144`
  - def set_phase `ingest/ingest/progress.py:155`
  - def update `ingest/ingest/progress.py:163`
  - def finish_source `ingest/ingest/progress.py:171`
  - def __rich__ `ingest/ingest/progress.py:182`

## ingest/ingest/qdrant_store.py
imports: ingest/ingest/chunking.py, ingest/ingest/config.py, ingest/ingest/dedup.py, ingest/ingest/embedding.py, ingest/ingest/sources.py
- def **make_client** `ingest/ingest/qdrant_store.py:47`
- def **point_id** `ingest/ingest/qdrant_store.py:62`
- def **_assert_dense_dim** `ingest/ingest/qdrant_store.py:67`
- def **ensure_collection** `ingest/ingest/qdrant_store.py:81`
- def **_rfc3339** `ingest/ingest/qdrant_store.py:138`
- def **build_payload** `ingest/ingest/qdrant_store.py:153`
- def **sparse_vector** `ingest/ingest/qdrant_store.py:189`
- def **point_struct** `ingest/ingest/qdrant_store.py:193`
- def **upsert_points** `ingest/ingest/qdrant_store.py:197`
- def **delete_doc_chunks_from** `ingest/ingest/qdrant_store.py:202`

## ingest/ingest/querylog.py
- def **build_query_record** `ingest/ingest/querylog.py:15`
- def **append_query_log** `ingest/ingest/querylog.py:43`

## ingest/ingest/remote_search.py
- class **RemoteSearchError**(RuntimeError) `ingest/ingest/remote_search.py:34`
- class **EndpointWarmingUp**(RemoteSearchError) `ingest/ingest/remote_search.py:38`
- class **RemoteOpError**(RemoteSearchError) `ingest/ingest/remote_search.py:42`
- class **RunPodQueueClient**() `ingest/ingest/remote_search.py:46`
  - def __init__ `ingest/ingest/remote_search.py:49`
  - def _request `ingest/ingest/remote_search.py:60`
  - def health `ingest/ingest/remote_search.py:84`
  - def call `ingest/ingest/remote_search.py:88`

## ingest/ingest/rerank.py
imports: ingest/ingest/config.py
- def **_auto_device** `ingest/ingest/rerank.py:25`
- def **_configure_cpu_threads** `ingest/ingest/rerank.py:33`
- class **BGEReranker**() `ingest/ingest/rerank.py:52`
  - def __init__ `ingest/ingest/rerank.py:55`
  - def score `ingest/ingest/rerank.py:71`
- class **RemoteBGEReranker**() `ingest/ingest/rerank.py:98`
  - def __init__ `ingest/ingest/rerank.py:108`
  - def score `ingest/ingest/rerank.py:113`

## ingest/ingest/search.py
imports: ingest/ingest/config.py
- def **detect_language** `ingest/ingest/search.py:16`
- def **_date_bound** `ingest/ingest/search.py:25`
- def **build_filter** `ingest/ingest/search.py:31`
- def **hybrid_search** `ingest/ingest/search.py:102`
- def **_point_dense** `ingest/ingest/search.py:189`
- def **_cosine** `ingest/ingest/search.py:197`
- def **diversify** `ingest/ingest/search.py:206`
- def **rerank_points** `ingest/ingest/search.py:276`

## ingest/ingest/snapshot.py
imports: ingest/ingest/__init__.py, ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/sources.py
- def **_run_files_desc** `ingest/ingest/snapshot.py:42`
- def **config_hash** `ingest/ingest/snapshot.py:55`
- class **SourceStats**() `ingest/ingest/snapshot.py:86`
  - def __post_init__ `ingest/ingest/snapshot.py:106`
- def **_snapshot_record** `ingest/ingest/snapshot.py:115`
- def **build_snapshot** `ingest/ingest/snapshot.py:150`
- def **_maybe_token_counter** `ingest/ingest/snapshot.py:253`
- def **_token_pctls** `ingest/ingest/snapshot.py:261`
- def **_near_dup_for_source** `ingest/ingest/snapshot.py:276`
- def **_pctl** `ingest/ingest/snapshot.py:287`
- def **_write_reports_and_manifest** `ingest/ingest/snapshot.py:294`

## ingest/ingest/sources.py
- def **normalize_status** `ingest/ingest/sources.py:47`
- def **_georgian_month** `ingest/ingest/sources.py:62`
- def **_valid_iso** `ingest/ingest/sources.py:71`
- def **_parse_date** `ingest/ingest/sources.py:82`
- class **CanonicalDoc**() `ingest/ingest/sources.py:114`
- class **SourceSpec**() `ingest/ingest/sources.py:146`
  - def declared_keys `ingest/ingest/sources.py:168`
  - def _first `ingest/ingest/sources.py:182`
  - def _parties `ingest/ingest/sources.py:189`
  - def build `ingest/ingest/sources.py:197`
- def **normalize** `ingest/ingest/sources.py:363`
- def **schema_drift** `ingest/ingest/sources.py:371`

## ingest/ingest/structure.py
- class **StructureInfo**() `ingest/ingest/structure.py:34`
- def **detect** `ingest/ingest/structure.py:44`
- def **article_spans** `ingest/ingest/structure.py:74`

## ingest/eval/__init__.py

## ingest/eval/backend.py
imports: ingest/eval/bm25.py, ingest/eval/bm25_full.py, ingest/eval/metrics.py, ingest/ingest/search.py
- class **ChunkRecord**() `ingest/eval/backend.py:26`
- def **_rrf_fuse** `ingest/eval/backend.py:33`
- class **FakeBackend**() `ingest/eval/backend.py:42`
  - def __init__ `ingest/eval/backend.py:50`
  - def _to_dense `ingest/eval/backend.py:66`
  - def _hit `ingest/eval/backend.py:71`
  - def _dense_rank `ingest/eval/backend.py:75`
  - def _sparse_rank `ingest/eval/backend.py:84`
  - def search `ingest/eval/backend.py:93`
- class **QdrantBackend**() `ingest/eval/backend.py:145`
  - def __init__ `ingest/eval/backend.py:152`
  - def _diversity_on `ingest/eval/backend.py:178`
  - def _candidate `ingest/eval/backend.py:181`
  - def _search_params `ingest/eval/backend.py:193`
  - def _fusion_query `ingest/eval/backend.py:201`
  - def _manual_fusion `ingest/eval/backend.py:205`
  - def _points_to_hits `ingest/eval/backend.py:239`
  - def _ensure_bm25 `ingest/eval/backend.py:248`
  - def search `ingest/eval/backend.py:284`

## ingest/eval/bm25.py
- def **tokenize** `ingest/eval/bm25.py:17`
- class **BM25Index**() `ingest/eval/bm25.py:21`
  - def __init__ `ingest/eval/bm25.py:24`
  - def from_pairs `ingest/eval/bm25.py:50`
  - def _score `ingest/eval/bm25.py:60`
  - def search `ingest/eval/bm25.py:73`

## ingest/eval/bm25_full.py
imports: ingest/eval/bm25.py, ingest/ingest/config.py
- class **FullCorpusBM25**() `ingest/eval/bm25_full.py:38`
  - def __init__ `ingest/eval/bm25_full.py:50`
  - def build `ingest/eval/bm25_full.py:61`
  - def cache_exists `ingest/eval/bm25_full.py:180`
  - def load `ingest/eval/bm25_full.py:184`
  - def build_or_load `ingest/eval/bm25_full.py:209`
  - def search `ingest/eval/bm25_full.py:215`
- def **_main** `ingest/eval/bm25_full.py:247`

## ingest/eval/evaluate.py
imports: ingest/eval/__init__.py, ingest/eval/backend.py, ingest/eval/metrics.py, ingest/eval/spanmap.py, ingest/eval/stats.py, ingest/eval/translations.py, ingest/ingest/chunking.py, ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/qdrant_store.py, ingest/ingest/rerank.py
- def **_token_counter** `ingest/eval/evaluate.py:44`
- def **build_query_relevance** `ingest/eval/evaluate.py:52`
- def **build_fake_corpus** `ingest/eval/evaluate.py:71`
- def **run_mode** `ingest/eval/evaluate.py:108`
- def **_fmt_ci** `ingest/eval/evaluate.py:122`
- def **print_table** `ingest/eval/evaluate.py:126`
- def **print_breakdowns** `ingest/eval/evaluate.py:140`
- def **print_stage_latency** `ingest/eval/evaluate.py:150`
- def **qdrant_deps** `ingest/eval/evaluate.py:161`
- def **make_backend** `ingest/eval/evaluate.py:184`
- def **main** `ingest/eval/evaluate.py:195`

## ingest/eval/explog.py
- def **config_hash** `ingest/eval/explog.py:21`
- def **now_iso** `ingest/eval/explog.py:28`
- def **append_run** `ingest/eval/explog.py:32`
- def **read_log** `ingest/eval/explog.py:40`

## ingest/eval/goldset.py
imports: ingest/eval/spanmap.py
- def **_nfc** `ingest/eval/goldset.py:33`
- class **Relevance**() `ingest/eval/goldset.py:38`
- class **GoldQuery**() `ingest/eval/goldset.py:47`
- def **load_golden_set** `ingest/eval/goldset.py:61`
- def **load_holdout** `ingest/eval/goldset.py:98`
- def **eval_set_hash** `ingest/eval/goldset.py:103`
- def **gold_docs** `ingest/eval/goldset.py:113`
- class **SnapshotBodies**() `ingest/eval/goldset.py:118`
  - def __init__ `ingest/eval/goldset.py:126`
  - def _load_source `ingest/eval/goldset.py:134`
  - def body `ingest/eval/goldset.py:151`
- def **reground** `ingest/eval/goldset.py:158`
- def **enforce_holdout** `ingest/eval/goldset.py:180`
- def **lint_span_coverage** `ingest/eval/goldset.py:190`

## ingest/eval/metrics.py
- class **Hit**() `ingest/eval/metrics.py:21`
- class **QueryScore**() `ingest/eval/metrics.py:29`
- def **reduce_ranking** `ingest/eval/metrics.py:39`
- def **_dcg** `ingest/eval/metrics.py:55`
- def **score_ranking** `ingest/eval/metrics.py:59`
- def **query_score** `ingest/eval/metrics.py:86`
- def **values** `ingest/eval/metrics.py:110`
- def **aggregate** `ingest/eval/metrics.py:114`
- def **breakdown** `ingest/eval/metrics.py:119`
- def **percentiles** `ingest/eval/metrics.py:127`

## ingest/eval/spanmap.py
imports: ingest/ingest/chunking.py
- def **_overlaps** `ingest/eval/spanmap.py:22`
- def **chunks_covering_span** `ingest/eval/spanmap.py:33`
- def **map_spans_to_chunks** `ingest/eval/spanmap.py:55`
- def **graded_relevant_chunks** `ingest/eval/spanmap.py:76`

## ingest/eval/stats.py
- class **CI**() `ingest/eval/stats.py:19`
- class **Comparison**() `ingest/eval/stats.py:26`
- def **bootstrap_ci** `ingest/eval/stats.py:37`
- def **paired_diff_ci** `ingest/eval/stats.py:55`
- def **paired_permutation_p** `ingest/eval/stats.py:78`
- def **compare** `ingest/eval/stats.py:106`

## ingest/eval/translations.py
imports: ingest/ingest/search.py
- def **load_query_translations** `ingest/eval/translations.py:20`

## ingest/scripts/backfill_consolidation.py
imports: ingest/ingest/config.py, ingest/ingest/pipeline.py, ingest/ingest/qdrant_store.py
- def **_flush** `ingest/scripts/backfill_consolidation.py:34`
- def **main** `ingest/scripts/backfill_consolidation.py:52`

## ingest/scripts/build_phase_c_report.py
- def **load_rows** `ingest/scripts/build_phase_c_report.py:25`
- def **cell** `ingest/scripts/build_phase_c_report.py:29`
- def **lat_cell** `ingest/scripts/build_phase_c_report.py:37`
- def **extra_knobs** `ingest/scripts/build_phase_c_report.py:42`
- def **rc_of** `ingest/scripts/build_phase_c_report.py:48`
- def **dedup_latest** `ingest/scripts/build_phase_c_report.py:52`
- def **metric_table** `ingest/scripts/build_phase_c_report.py:60`
- def **slice_table** `ingest/scripts/build_phase_c_report.py:81`
- def **parse_paired** `ingest/scripts/build_phase_c_report.py:103`
- def **paired_section** `ingest/scripts/build_phase_c_report.py:126`
- def **main** `ingest/scripts/build_phase_c_report.py:142`

## ingest/scripts/embed_delta.py
imports: ingest/ingest/config.py, ingest/ingest/embed_job.py, ingest/ingest/embedding.py, ingest/ingest/qdrant_store.py, ingest/ingest/sources.py
- def **_iter_items** `ingest/scripts/embed_delta.py:35`
- def **_resolve_paths** `ingest/scripts/embed_delta.py:48`
- def **main** `ingest/scripts/embed_delta.py:58`

## ingest/scripts/gen_code_map.py
- def **iter_py_files** `ingest/scripts/gen_code_map.py:57`
- def **resolve_import** `ingest/scripts/gen_code_map.py:72`
- def **module_section** `ingest/scripts/gen_code_map.py:93`
- def **generate** `ingest/scripts/gen_code_map.py:126`
- def **slugify** `ingest/scripts/gen_code_map.py:136`
- def **symbol_defined** `ingest/scripts/gen_code_map.py:141`
- def **lint** `ingest/scripts/gen_code_map.py:149`
- def **main** `ingest/scripts/gen_code_map.py:194`

## ingest/scripts/merge_delta_collection.py
imports: ingest/ingest/config.py, ingest/ingest/qdrant_store.py
- def **_to_struct** `ingest/scripts/merge_delta_collection.py:36`
- def **merge_collection** `ingest/scripts/merge_delta_collection.py:49`
- def **_count** `ingest/scripts/merge_delta_collection.py:69`
- def **dry_run** `ingest/scripts/merge_delta_collection.py:73`
- def **main** `ingest/scripts/merge_delta_collection.py:126`

## ingest/scripts/monitor_server.py
- def **_poll_once** `ingest/scripts/monitor_server.py:54`
- def **poll_loop** `ingest/scripts/monitor_server.py:88`
- def **_computed** `ingest/scripts/monitor_server.py:98`
- class **Handler**(BaseHTTPRequestHandler) `ingest/scripts/monitor_server.py:205`
  - def log_message `ingest/scripts/monitor_server.py:206`
  - def do_GET `ingest/scripts/monitor_server.py:209`
- def **main** `ingest/scripts/monitor_server.py:226`

## ingest/scripts/publish_snapshot.py
imports: ingest/ingest/config.py, ingest/ingest/remote_search.py
- def **_qdrant** `ingest/scripts/publish_snapshot.py:50`
- def **_sha256** `ingest/scripts/publish_snapshot.py:59`
- def **create** `ingest/scripts/publish_snapshot.py:77`
- def **_s3** `ingest/scripts/publish_snapshot.py:134`
- def **_load_manifest** `ingest/scripts/publish_snapshot.py:153`
- def **upload** `ingest/scripts/publish_snapshot.py:160`
- def **_abort_quietly** `ingest/scripts/publish_snapshot.py:254`
- def **_publish_manifest_object** `ingest/scripts/publish_snapshot.py:263`
- def **verify** `ingest/scripts/publish_snapshot.py:271`
- def **cleanup** `ingest/scripts/publish_snapshot.py:309`
- def **main** `ingest/scripts/publish_snapshot.py:336`

## ingest/scripts/rerank_latency_probe.py
imports: ingest/ingest/config.py, ingest/ingest/qdrant_store.py, ingest/ingest/rerank.py
- def **_golden_queries** `ingest/scripts/rerank_latency_probe.py:35`
- def **main** `ingest/scripts/rerank_latency_probe.py:48`

## ingest/scripts/runpod_orchestrate.py
imports: ingest/ingest/embed_job.py
- def **log** `ingest/scripts/runpod_orchestrate.py:67`
- def **_api_key** `ingest/scripts/runpod_orchestrate.py:78`
- def **gql** `ingest/scripts/runpod_orchestrate.py:89`
- def **run** `ingest/scripts/runpod_orchestrate.py:104`
- def **_ssh_base** `ingest/scripts/runpod_orchestrate.py:114`
- def **ssh_ok** `ingest/scripts/runpod_orchestrate.py:121`
- def **ssh_capture** `ingest/scripts/runpod_orchestrate.py:127`
- def **push_content** `ingest/scripts/runpod_orchestrate.py:133`
- def **ensure_pod_tools** `ingest/scripts/runpod_orchestrate.py:143`
- def **_remote_size** `ingest/scripts/runpod_orchestrate.py:158`
- def **push_file** `ingest/scripts/runpod_orchestrate.py:170`
- def **pull_file** `ingest/scripts/runpod_orchestrate.py:189`
- def **step_checksum_ref** `ingest/scripts/runpod_orchestrate.py:207`
- def **step_package** `ingest/scripts/runpod_orchestrate.py:217`
- def **step_keypair** `ingest/scripts/runpod_orchestrate.py:242`
- def **gpu_price** `ingest/scripts/runpod_orchestrate.py:251`
- def **step_provision** `ingest/scripts/runpod_orchestrate.py:262`
- def **step_wait_ssh** `ingest/scripts/runpod_orchestrate.py:296`
- def **step_transfer_in** `ingest/scripts/runpod_orchestrate.py:320`
- def **step_launch** `ingest/scripts/runpod_orchestrate.py:330`
- def **step_poll** `ingest/scripts/runpod_orchestrate.py:356`
- def **step_transfer_out** `ingest/scripts/runpod_orchestrate.py:379`
- def **step_verify_g2** `ingest/scripts/runpod_orchestrate.py:396`
- def **terminate** `ingest/scripts/runpod_orchestrate.py:407`
- def **_cleanup** `ingest/scripts/runpod_orchestrate.py:420`
- def **_sig** `ingest/scripts/runpod_orchestrate.py:426`
- def **step_restore** `ingest/scripts/runpod_orchestrate.py:432`
- def **main** `ingest/scripts/runpod_orchestrate.py:446`

## ingest/scripts/runpod_orchestrate_delta.py
imports: ingest/ingest/embed_job.py
- def **_retry** `ingest/scripts/runpod_orchestrate_delta.py:46`
- def **stage_delta_items** `ingest/scripts/runpod_orchestrate_delta.py:61`
- def **step_package_delta** `ingest/scripts/runpod_orchestrate_delta.py:87`
- def **step_launch_delta** `ingest/scripts/runpod_orchestrate_delta.py:113`
- def **step_poll_delta** `ingest/scripts/runpod_orchestrate_delta.py:138`
- def **step_transfer_out_delta** `ingest/scripts/runpod_orchestrate_delta.py:175`
- def **step_verify_g2_delta** `ingest/scripts/runpod_orchestrate_delta.py:196`
- def **step_restore_delta** `ingest/scripts/runpod_orchestrate_delta.py:207`
- def **main** `ingest/scripts/runpod_orchestrate_delta.py:228`

## ingest/scripts/runpod_orchestrate_multi.py
imports: ingest/ingest/embed_job.py
- def **log** `ingest/scripts/runpod_orchestrate_multi.py:54`
- def **gql** `ingest/scripts/runpod_orchestrate_multi.py:64`
- def **run** `ingest/scripts/runpod_orchestrate_multi.py:80`
- def **_ssh** `ingest/scripts/runpod_orchestrate_multi.py:88`
- def **ssh_ok** `ingest/scripts/runpod_orchestrate_multi.py:95`
- def **ssh_cap** `ingest/scripts/runpod_orchestrate_multi.py:100`
- def **push_content** `ingest/scripts/runpod_orchestrate_multi.py:105`
- def **pull_file** `ingest/scripts/runpod_orchestrate_multi.py:112`
- def **ensure_pod_tools** `ingest/scripts/runpod_orchestrate_multi.py:120`
- def **gpu_price** `ingest/scripts/runpod_orchestrate_multi.py:126`
- def **provision** `ingest/scripts/runpod_orchestrate_multi.py:134`
- def **wait_ssh** `ingest/scripts/runpod_orchestrate_multi.py:156`
- def **wait_corpus_ready** `ingest/scripts/runpod_orchestrate_multi.py:175`
- def **transfer_corpus** `ingest/scripts/runpod_orchestrate_multi.py:186`
- def **push_code** `ingest/scripts/runpod_orchestrate_multi.py:202`
- def **launch** `ingest/scripts/runpod_orchestrate_multi.py:219`
- def **poll** `ingest/scripts/runpod_orchestrate_multi.py:231`
- def **terminate** `ingest/scripts/runpod_orchestrate_multi.py:250`
- def **_cleanup** `ingest/scripts/runpod_orchestrate_multi.py:263`
- def **restore** `ingest/scripts/runpod_orchestrate_multi.py:268`
- def **main** `ingest/scripts/runpod_orchestrate_multi.py:279`

## ingest/scripts/runpod_rerank.py
- def **_terminate** `ingest/scripts/runpod_rerank.py:36`
- def **_sig** `ingest/scripts/runpod_rerank.py:46`
- def **_dep_check** `ingest/scripts/runpod_rerank.py:52`
- def **setup_pod** `ingest/scripts/runpod_rerank.py:60`
- def **open_tunnel** `ingest/scripts/runpod_rerank.py:103`
- def **up** `ingest/scripts/runpod_rerank.py:118`

## ingest/scripts/runpod_rerank_server.py
- def **score** `ingest/scripts/runpod_rerank_server.py:31`
- class **Handler**(BaseHTTPRequestHandler) `ingest/scripts/runpod_rerank_server.py:47`
  - def _send `ingest/scripts/runpod_rerank_server.py:48`
  - def do_GET `ingest/scripts/runpod_rerank_server.py:56`
  - def do_POST `ingest/scripts/runpod_rerank_server.py:59`
  - def log_message `ingest/scripts/runpod_rerank_server.py:68`

## ingest/scripts/sample_ingest_report.py
imports: ingest/ingest/config.py, ingest/ingest/pipeline.py
- def **main** `ingest/scripts/sample_ingest_report.py:33`

## ingest/scripts/session_monitor.py
- def **_tool_label** `ingest/scripts/session_monitor.py:55`
- def **parse_session** `ingest/scripts/session_monitor.py:78`
- def **sessions_state** `ingest/scripts/session_monitor.py:174`
- def **_ps_rows** `ingest/scripts/session_monitor.py:192`
- def **processes_state** `ingest/scripts/session_monitor.py:214`
- def **_short_name** `ingest/scripts/session_monitor.py:227`
- def **system_state** `ingest/scripts/session_monitor.py:242`
- def **_serverless_health** `ingest/scripts/session_monitor.py:279`
- def **_pod_health** `ingest/scripts/session_monitor.py:311`
- def **rag_state** `ingest/scripts/session_monitor.py:327`
- def **_qdrant_get** `ingest/scripts/session_monitor.py:359`
- def **qdrant_state** `ingest/scripts/session_monitor.py:374`
- def **_tail_line** `ingest/scripts/session_monitor.py:397`
- def **_script_alive** `ingest/scripts/session_monitor.py:409`
- def **coverage_state** `ingest/scripts/session_monitor.py:426`
- def **_log_epoch** `ingest/scripts/session_monitor.py:502`
- def **_env_value** `ingest/scripts/session_monitor.py:509`
- def **_runpod_account** `ingest/scripts/session_monitor.py:521`
- def **_remote_payload_bytes** `ingest/scripts/session_monitor.py:548`
- def **_pod_delta_points** `ingest/scripts/session_monitor.py:572`
- def **_seen_total** `ingest/scripts/session_monitor.py:596`
- def **scrape_state** `ingest/scripts/session_monitor.py:620`
- def **gpu_state** `ingest/scripts/session_monitor.py:656`
- def **build_state** `ingest/scripts/session_monitor.py:748`
- class **Handler**(BaseHTTPRequestHandler) `ingest/scripts/session_monitor.py:762`
  - def log_message `ingest/scripts/session_monitor.py:763`
  - def _send `ingest/scripts/session_monitor.py:766`
  - def do_GET `ingest/scripts/session_monitor.py:777`
- def **main** `ingest/scripts/session_monitor.py:792`

## ingest/scripts/verify_all_embedded.py
imports: ingest/ingest/config.py, ingest/ingest/qdrant_store.py, ingest/ingest/sources.py
- def **quarantined_ids** `ingest/scripts/verify_all_embedded.py:36`
- def **scraped_universe** `ingest/scripts/verify_all_embedded.py:58`
- def **no_text_ids** `ingest/scripts/verify_all_embedded.py:108`
- def **embedded_universe** `ingest/scripts/verify_all_embedded.py:147`
- def **main** `ingest/scripts/verify_all_embedded.py:176`

## ingest/scripts/verify_delta_embedded.py
imports: ingest/ingest/config.py, ingest/ingest/qdrant_store.py
- def **delta_doc_ids** `ingest/scripts/verify_delta_embedded.py:28`
- def **main** `ingest/scripts/verify_delta_embedded.py:51`

## ingest/scripts/verify_matsne_completeness.py
imports: ingest/ingest/config.py
- def **parse_last_page** `ingest/scripts/verify_matsne_completeness.py:60`
- def **expected_total** `ingest/scripts/verify_matsne_completeness.py:66`
- def **is_absent_page** `ingest/scripts/verify_matsne_completeness.py:73`
- def **residual_ids** `ingest/scripts/verify_matsne_completeness.py:81`
- def **fetch** `ingest/scripts/verify_matsne_completeness.py:89`
- def **load_seen_ids** `ingest/scripts/verify_matsne_completeness.py:106`
- def **referenced_ids** `ingest/scripts/verify_matsne_completeness.py:117`
- def **audit_advertised** `ingest/scripts/verify_matsne_completeness.py:138`
- def **audit_reference_closure** `ingest/scripts/verify_matsne_completeness.py:164`
- def **audit_id_enum** `ingest/scripts/verify_matsne_completeness.py:178`
- def **main** `ingest/scripts/verify_matsne_completeness.py:201`

## ingest/serverless/handler.py
imports: ingest/ingest/__init__.py
- def **_boot** `ingest/serverless/handler.py:46`
- def **_sysinfo** `ingest/serverless/handler.py:74`
- def **_warmup** `ingest/serverless/handler.py:105`
- def **handler** `ingest/serverless/handler.py:113`
- def **main** `ingest/serverless/handler.py:163`

## ingest/serverless/qdrant_boot.py
- def **storage_dir** `ingest/serverless/qdrant_boot.py:52`
- def **publish_dir** `ingest/serverless/qdrant_boot.py:56`
- def **_http** `ingest/serverless/qdrant_boot.py:60`
- def **_healthy** `ingest/serverless/qdrant_boot.py:80`
- def **_log_tail** `ingest/serverless/qdrant_boot.py:87`
- def **_spawn** `ingest/serverless/qdrant_boot.py:94`
- def **ensure_running** `ingest/serverless/qdrant_boot.py:108`
- def **_read_json** `ingest/serverless/qdrant_boot.py:140`
- def **_sha256** `ingest/serverless/qdrant_boot.py:147`
- def **needs_restore** `ingest/serverless/qdrant_boot.py:158`
- def **restore_pending** `ingest/serverless/qdrant_boot.py:171`
- def **_collection_points** `ingest/serverless/qdrant_boot.py:178`
- def **maybe_restore** `ingest/serverless/qdrant_boot.py:187`

## scraper/legal_scrapers/__init__.py

## scraper/legal_scrapers/extensions.py
- def **extract_title** `scraper/legal_scrapers/extensions.py:55`
- def **_truncate** `scraper/legal_scrapers/extensions.py:73`
- class **ProgressSnapshot**() `scraper/legal_scrapers/extensions.py:81`
- def **_format_elapsed** `scraper/legal_scrapers/extensions.py:99`
- def **_items_per_min** `scraper/legal_scrapers/extensions.py:108`
- def **_format_status_counts** `scraper/legal_scrapers/extensions.py:116`
- def **_status_glyph** `scraper/legal_scrapers/extensions.py:134`
- def **render_panel** `scraper/legal_scrapers/extensions.py:145`
- def **render_table** `scraper/legal_scrapers/extensions.py:191`
- class **_ProgressDashboard**() `scraper/legal_scrapers/extensions.py:254`
  - def __init__ `scraper/legal_scrapers/extensions.py:263`
  - def reset `scraper/legal_scrapers/extensions.py:266`
  - def declare `scraper/legal_scrapers/extensions.py:278`
  - def finish `scraper/legal_scrapers/extensions.py:284`
  - def register `scraper/legal_scrapers/extensions.py:318`
  - def on_spider_closed `scraper/legal_scrapers/extensions.py:322`
  - def _multi `scraper/legal_scrapers/extensions.py:332`
  - def _ensure_started `scraper/legal_scrapers/extensions.py:335`
  - def _render `scraper/legal_scrapers/extensions.py:351`
  - def _tick `scraper/legal_scrapers/extensions.py:380`
- def **declare_spiders** `scraper/legal_scrapers/extensions.py:394`
- def **finish_dashboard** `scraper/legal_scrapers/extensions.py:399`
- class **LiveProgressExtension**() `scraper/legal_scrapers/extensions.py:404`
  - def __init__ `scraper/legal_scrapers/extensions.py:407`
  - def from_crawler `scraper/legal_scrapers/extensions.py:416`
  - def spider_opened `scraper/legal_scrapers/extensions.py:432`
  - def item_scraped `scraper/legal_scrapers/extensions.py:437`
  - def spider_closed `scraper/legal_scrapers/extensions.py:442`
  - def current_snapshot `scraper/legal_scrapers/extensions.py:448`
  - def _snapshot `scraper/legal_scrapers/extensions.py:453`
  - def _total_items `scraper/legal_scrapers/extensions.py:484`

## scraper/legal_scrapers/items.py
imports: scraper/legal_scrapers/utils/markdown.py
- def **class_to_status** `scraper/legal_scrapers/items.py:12`
- class **MatsneItem**(scrapy.Item) `scraper/legal_scrapers/items.py:22`
- def **_strip** `scraper/legal_scrapers/items.py:58`
- def **_f** `scraper/legal_scrapers/items.py:67`
- def **_body** `scraper/legal_scrapers/items.py:72`
- def **_list** `scraper/legal_scrapers/items.py:78`
- class **EcdItem**(scrapy.Item) `scraper/legal_scrapers/items.py:84`
- class **ConstcourtItem**(scrapy.Item) `scraper/legal_scrapers/items.py:103`
- class **NaprItem**(scrapy.Item) `scraper/legal_scrapers/items.py:119`
- class **TbappealItem**(scrapy.Item) `scraper/legal_scrapers/items.py:136`
- class **SupremecourtItem**(scrapy.Item) `scraper/legal_scrapers/items.py:148`
- class **TasItem**(scrapy.Item) `scraper/legal_scrapers/items.py:163`

## scraper/legal_scrapers/middlewares.py
imports: scraper/legal_scrapers/utils/user_agents.py
- class **RotateUserAgentMiddleware**() `scraper/legal_scrapers/middlewares.py:11`
  - def process_request `scraper/legal_scrapers/middlewares.py:12`
- class **MatsneSpiderMiddleware**() `scraper/legal_scrapers/middlewares.py:16`
  - def from_crawler `scraper/legal_scrapers/middlewares.py:22`
  - def process_spider_input `scraper/legal_scrapers/middlewares.py:28`
  - def process_spider_output `scraper/legal_scrapers/middlewares.py:35`
  - def process_spider_exception `scraper/legal_scrapers/middlewares.py:43`
  - def process_start `scraper/legal_scrapers/middlewares.py:50`
  - def spider_opened `scraper/legal_scrapers/middlewares.py:56`
- class **MatsneDownloaderMiddleware**() `scraper/legal_scrapers/middlewares.py:60`
  - def from_crawler `scraper/legal_scrapers/middlewares.py:66`
  - def process_request `scraper/legal_scrapers/middlewares.py:72`
  - def process_response `scraper/legal_scrapers/middlewares.py:84`
  - def process_exception `scraper/legal_scrapers/middlewares.py:93`
  - def spider_opened `scraper/legal_scrapers/middlewares.py:103`

## scraper/legal_scrapers/pipelines.py
- class **MatsnePipeline**() `scraper/legal_scrapers/pipelines.py:9`
  - def process_item `scraper/legal_scrapers/pipelines.py:10`
- class **DedupPipeline**() `scraper/legal_scrapers/pipelines.py:14`
  - def process_item `scraper/legal_scrapers/pipelines.py:26`

## scraper/legal_scrapers/run.py
imports: scraper/legal_scrapers/extensions.py
- def **_bootstrap_project_dir** `scraper/legal_scrapers/run.py:35`
- def **parse_args** `scraper/legal_scrapers/run.py:50`
- def **select_spiders** `scraper/legal_scrapers/run.py:76`
- def **main** `scraper/legal_scrapers/run.py:95`

## scraper/legal_scrapers/settings.py

## scraper/legal_scrapers/spiders/__init__.py

## scraper/legal_scrapers/spiders/base.py
- class **BaseLegalSpider**(scrapy.Spider) `scraper/legal_scrapers/spiders/base.py:18`
  - def __init__ `scraper/legal_scrapers/spiders/base.py:38`
  - def from_crawler `scraper/legal_scrapers/spiders/base.py:55`
  - def request_failed `scraper/legal_scrapers/spiders/base.py:60`
  - def parse_date_arg `scraper/legal_scrapers/spiders/base.py:69`
  - def configure_run_outputs `scraper/legal_scrapers/spiders/base.py:80`
  - def open_dedup_store `scraper/legal_scrapers/spiders/base.py:109`
  - def dedup_key `scraper/legal_scrapers/spiders/base.py:142`
  - def is_seen `scraper/legal_scrapers/spiders/base.py:158`
  - def mark_seen `scraper/legal_scrapers/spiders/base.py:162`
  - def build_run_id `scraper/legal_scrapers/spiders/base.py:176`
  - def feed_options `scraper/legal_scrapers/spiders/base.py:190`
  - def write_run_metadata `scraper/legal_scrapers/spiders/base.py:198`

## scraper/legal_scrapers/spiders/constcourt_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/documents.py, scraper/legal_scrapers/utils/markdown.py
- class **ConstcourtSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/constcourt_spider.py:48`
  - def start `scraper/legal_scrapers/spiders/constcourt_spider.py:53`
  - def request_page `scraper/legal_scrapers/spiders/constcourt_spider.py:56`
  - def parse_list `scraper/legal_scrapers/spiders/constcourt_spider.py:70`
  - def parse_detail `scraper/legal_scrapers/spiders/constcourt_spider.py:100`
  - def parse_docx_body `scraper/legal_scrapers/spiders/constcourt_spider.py:132`
  - def load_item `scraper/legal_scrapers/spiders/constcourt_spider.py:141`

## scraper/legal_scrapers/spiders/ecd_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/json_api.py, scraper/legal_scrapers/utils/text.py
- class **EcdSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/ecd_spider.py:31`
  - def start `scraper/legal_scrapers/spiders/ecd_spider.py:36`
  - def parse_instances `scraper/legal_scrapers/spiders/ecd_spider.py:39`
  - def request_page `scraper/legal_scrapers/spiders/ecd_spider.py:47`
  - def parse_list `scraper/legal_scrapers/spiders/ecd_spider.py:63`
  - def parse_detail `scraper/legal_scrapers/spiders/ecd_spider.py:95`

## scraper/legal_scrapers/spiders/matsne_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/search_urls.py
- class **MatsneSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/matsne_spider.py:23`
  - def from_crawler `scraper/legal_scrapers/spiders/matsne_spider.py:32`
  - def start `scraper/legal_scrapers/spiders/matsne_spider.py:37`
  - def _load_seed_urls `scraper/legal_scrapers/spiders/matsne_spider.py:87`
  - def spider_idle `scraper/legal_scrapers/spiders/matsne_spider.py:117`
  - def start_phase `scraper/legal_scrapers/spiders/matsne_spider.py:136`
  - def build_request `scraper/legal_scrapers/spiders/matsne_spider.py:144`
  - def follow_request `scraper/legal_scrapers/spiders/matsne_spider.py:158`
  - def parse `scraper/legal_scrapers/spiders/matsne_spider.py:171`
  - def _split_requests `scraper/legal_scrapers/spiders/matsne_spider.py:217`
  - def parse_document `scraper/legal_scrapers/spiders/matsne_spider.py:257`

## scraper/legal_scrapers/spiders/napr_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/documents.py, scraper/legal_scrapers/utils/json_api.py
- def **decision_type_from_title** `scraper/legal_scrapers/spiders/napr_spider.py:65`
- class **NaprSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/napr_spider.py:75`
  - def from_crawler `scraper/legal_scrapers/spiders/napr_spider.py:81`
  - def start `scraper/legal_scrapers/spiders/napr_spider.py:86`
  - def spider_idle `scraper/legal_scrapers/spiders/napr_spider.py:91`
  - def request_page `scraper/legal_scrapers/spiders/napr_spider.py:99`
  - def parse_list `scraper/legal_scrapers/spiders/napr_spider.py:117`
  - def parse_pdf `scraper/legal_scrapers/spiders/napr_spider.py:158`
  - def load_item `scraper/legal_scrapers/spiders/napr_spider.py:168`

## scraper/legal_scrapers/spiders/supremecourt_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/markdown.py
- class **SupremecourtSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/supremecourt_spider.py:47`
  - def start `scraper/legal_scrapers/spiders/supremecourt_spider.py:59`
  - def request_page `scraper/legal_scrapers/spiders/supremecourt_spider.py:63`
  - def parse_list `scraper/legal_scrapers/spiders/supremecourt_spider.py:78`
  - def parse_detail `scraper/legal_scrapers/spiders/supremecourt_spider.py:118`

## scraper/legal_scrapers/spiders/tas_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/markdown.py
- def **_xml_tag** `scraper/legal_scrapers/spiders/tas_spider.py:150`
- def **_strip** `scraper/legal_scrapers/spiders/tas_spider.py:155`
- def **_clean_value** `scraper/legal_scrapers/spiders/tas_spider.py:159`
- def **_local_date** `scraper/legal_scrapers/spiders/tas_spider.py:166`
- def **_slash** `scraper/legal_scrapers/spiders/tas_spider.py:182`
- def **_field_labels** `scraper/legal_scrapers/spiders/tas_spider.py:192`
- def **_form_fields** `scraper/legal_scrapers/spiders/tas_spider.py:203`
- def **_request_text** `scraper/legal_scrapers/spiders/tas_spider.py:223`
- def **_parcels** `scraper/legal_scrapers/spiders/tas_spider.py:239`
- def **_primary_parcel** `scraper/legal_scrapers/spiders/tas_spider.py:257`
- def **_responses** `scraper/legal_scrapers/spiders/tas_spider.py:265`
- def **_response_to_markdown** `scraper/legal_scrapers/spiders/tas_spider.py:280`
- def **_nomenclature_full** `scraper/legal_scrapers/spiders/tas_spider.py:292`
- def **_full_name** `scraper/legal_scrapers/spiders/tas_spider.py:301`
- class **TasSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/tas_spider.py:308`
  - def start `scraper/legal_scrapers/spiders/tas_spider.py:333`
  - def parse_docs `scraper/legal_scrapers/spiders/tas_spider.py:340`
  - def _fetch_detail `scraper/legal_scrapers/spiders/tas_spider.py:376`
  - def build_item `scraper/legal_scrapers/spiders/tas_spider.py:394`
  - def _enrich `scraper/legal_scrapers/spiders/tas_spider.py:425`
  - def _list_body `scraper/legal_scrapers/spiders/tas_spider.py:517`
  - def _detail_body `scraper/legal_scrapers/spiders/tas_spider.py:528`

## scraper/legal_scrapers/spiders/tbappeal_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/markdown.py
- class **TbappealSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/tbappeal_spider.py:27`
  - def start `scraper/legal_scrapers/spiders/tbappeal_spider.py:31`
  - def request_page `scraper/legal_scrapers/spiders/tbappeal_spider.py:35`
  - def parse_list `scraper/legal_scrapers/spiders/tbappeal_spider.py:43`
  - def _in_window `scraper/legal_scrapers/spiders/tbappeal_spider.py:87`
  - def parse_detail `scraper/legal_scrapers/spiders/tbappeal_spider.py:93`

## scraper/legal_scrapers/utils/__init__.py
imports: scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/documents.py, scraper/legal_scrapers/utils/json_api.py, scraper/legal_scrapers/utils/markdown.py, scraper/legal_scrapers/utils/search_urls.py, scraper/legal_scrapers/utils/text.py, scraper/legal_scrapers/utils/user_agents.py

## scraper/legal_scrapers/utils/dates.py
- def **iso_to_dotted** `scraper/legal_scrapers/utils/dates.py:12`
- def **iso_to_slashed** `scraper/legal_scrapers/utils/dates.py:17`
- def **iso_to_year_slashed** `scraper/legal_scrapers/utils/dates.py:22`
- def **dotnet_date_to_iso** `scraper/legal_scrapers/utils/dates.py:30`
- def **date_part** `scraper/legal_scrapers/utils/dates.py:44`
- def **parse_dotted** `scraper/legal_scrapers/utils/dates.py:57`

## scraper/legal_scrapers/utils/documents.py
imports: scraper/legal_scrapers/utils/markdown.py, scraper/legal_scrapers/utils/text.py
- def **pdf_to_markdown** `scraper/legal_scrapers/utils/documents.py:23`
- def **_guard_docx** `scraper/legal_scrapers/utils/documents.py:34`
- def **docx_to_markdown** `scraper/legal_scrapers/utils/documents.py:48`

## scraper/legal_scrapers/utils/json_api.py
- def **json_post** `scraper/legal_scrapers/utils/json_api.py:26`
- def **form_post** `scraper/legal_scrapers/utils/json_api.py:41`
- def **loads_maybe_double** `scraper/legal_scrapers/utils/json_api.py:56`

## scraper/legal_scrapers/utils/markdown.py
- class **LegalConverter**(MarkdownConverter) `scraper/legal_scrapers/utils/markdown.py:63`
  - def convert_sup `scraper/legal_scrapers/utils/markdown.py:66`
  - def convert_sub `scraper/legal_scrapers/utils/markdown.py:69`
  - def convert_u `scraper/legal_scrapers/utils/markdown.py:72`
- def **_script** `scraper/legal_scrapers/utils/markdown.py:78`
- def **_has_untranslated** `scraper/legal_scrapers/utils/markdown.py:89`
- def **_strip_cruft** `scraper/legal_scrapers/utils/markdown.py:93`
- def **_clean_links** `scraper/legal_scrapers/utils/markdown.py:103`
- def **_absolutize_images** `scraper/legal_scrapers/utils/markdown.py:114`
- def **_section_path** `scraper/legal_scrapers/utils/markdown.py:122`
- def **_heading_level** `scraper/legal_scrapers/utils/markdown.py:131`
- def **_transform_section_tables** `scraper/legal_scrapers/utils/markdown.py:141`
- def **_replace_with_heading** `scraper/legal_scrapers/utils/markdown.py:172`
- def **_unwrap_into** `scraper/legal_scrapers/utils/markdown.py:183`
- def **_is_data_table** `scraper/legal_scrapers/utils/markdown.py:190`
- def **_flatten_layout_tables** `scraper/legal_scrapers/utils/markdown.py:209`
- def **_expand_spans** `scraper/legal_scrapers/utils/markdown.py:219`
- def **_normalize_data_tables** `scraper/legal_scrapers/utils/markdown.py:270`
- def **_strip_spans** `scraper/legal_scrapers/utils/markdown.py:283`
- def **_collapse_emphasis** `scraper/legal_scrapers/utils/markdown.py:288`
- def **_tidy** `scraper/legal_scrapers/utils/markdown.py:300`
- def **html_to_markdown** `scraper/legal_scrapers/utils/markdown.py:307`
- def **safe_html_to_markdown** `scraper/legal_scrapers/utils/markdown.py:333`

## scraper/legal_scrapers/utils/search_urls.py
- def **first_qs_value** `scraper/legal_scrapers/utils/search_urls.py:37`
- def **sub_windows** `scraper/legal_scrapers/utils/search_urls.py:42`
- def **build_search_url** `scraper/legal_scrapers/utils/search_urls.py:67`
- def **generate_start_url_batches** `scraper/legal_scrapers/utils/search_urls.py:74`
- def **generate_start_urls** `scraper/legal_scrapers/utils/search_urls.py:97`

## scraper/legal_scrapers/utils/text.py
- def **plain_text_to_markdown** `scraper/legal_scrapers/utils/text.py:11`

## scraper/legal_scrapers/utils/user_agents.py
- def **generate_random_user_agent** `scraper/legal_scrapers/utils/user_agents.py:18`

## run_all.py
- def **_pump** `run_all.py:32`
- def **main** `run_all.py:41`
