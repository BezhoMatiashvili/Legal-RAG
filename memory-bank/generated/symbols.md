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
- def **_cmd_watch** `ingest/ingest/__main__.py:79`
- def **_cmd_snapshot** `ingest/ingest/__main__.py:114`
- def **_cmd_embed** `ingest/ingest/__main__.py:126`
- def **_cmd_search** `ingest/ingest/__main__.py:207`
- def **main** `ingest/ingest/__main__.py:240`

## ingest/ingest/artifacts.py
- class **ArtifactSafetyError**(RuntimeError) `ingest/ingest/artifacts.py:51`
- class **RunIdentity**() `ingest/ingest/artifacts.py:56`
- class **VerifiedGenerationCoverage**() `ingest/ingest/artifacts.py:64`
- class **PruneCandidate**() `ingest/ingest/artifacts.py:73`
  - def to_dict `ingest/ingest/artifacts.py:86`
- class **RetainedRun**() `ingest/ingest/artifacts.py:101`
  - def to_dict `ingest/ingest/artifacts.py:109`
- class **PrunePlan**() `ingest/ingest/artifacts.py:119`
  - def to_dict `ingest/ingest/artifacts.py:129`
- def **_create_private_parents** `ingest/ingest/artifacts.py:143`
- def **_fsync_directory** `ingest/ingest/artifacts.py:165`
- def **atomic_write_text** `ingest/ingest/artifacts.py:174`
- def **atomic_write_json** `ingest/ingest/artifacts.py:210`
- def **_atomic_create_private_json** `ingest/ingest/artifacts.py:216`
- def **_validate_rotation_options** `ingest/ingest/artifacts.py:248`
- def **_regular_file_stat** `ingest/ingest/artifacts.py:255`
- def **_rotation_lock** `ingest/ingest/artifacts.py:266`
- def **_rotate_unlocked** `ingest/ingest/artifacts.py:281`
- def **rotate_file** `ingest/ingest/artifacts.py:307`
- def **append_rotating_text** `ingest/ingest/artifacts.py:323`
- def **parse_utc_datetime** `ingest/ingest/artifacts.py:360`
- def **_iso_utc** `ingest/ingest/artifacts.py:373`
- def **_read_json_object** `ingest/ingest/artifacts.py:379`
- def **_safe_component** `ingest/ingest/artifacts.py:393`
- def **_parse_covered_runs** `ingest/ingest/artifacts.py:401`
- def **_gate_ok** `ingest/ingest/artifacts.py:437`
- def **_matching_verification_report** `ingest/ingest/artifacts.py:446`
- def **load_verified_generation_coverage** `ingest/ingest/artifacts.py:480`
- def **_coverage_index** `ingest/ingest/artifacts.py:527`
- def **_protected_name** `ingest/ingest/artifacts.py:537`
- def **_contains_protected_evidence** `ingest/ingest/artifacts.py:542`
- def **_truthy_failure_signal** `ingest/ingest/artifacts.py:555`
- def **_metadata_has_failures** `ingest/ingest/artifacts.py:563`
- def **_required_true** `ingest/ingest/artifacts.py:577`
- def **_run_success_reason** `ingest/ingest/artifacts.py:592`
- def **_directory_size** `ingest/ingest/artifacts.py:625`
- def **_metadata_hash** `ingest/ingest/artifacts.py:636`
- def **_retention_days** `ingest/ingest/artifacts.py:644`
- def **_retained** `ingest/ingest/artifacts.py:652`
- def **_assess_run** `ingest/ingest/artifacts.py:661`
- def **build_prune_plan** `ingest/ingest/artifacts.py:718`
- def **_prune_plan_digest** `ingest/ingest/artifacts.py:775`
- def **prune_plan_document** `ingest/ingest/artifacts.py:782`
- def **write_prune_plan** `ingest/ingest/artifacts.py:788`
- def **_plan_component** `ingest/ingest/artifacts.py:793`
- def **_plan_timestamp** `ingest/ingest/artifacts.py:800`
- def **_load_prune_candidate** `ingest/ingest/artifacts.py:812`
- def **_load_retained_run** `ingest/ingest/artifacts.py:863`
- def **load_prune_plan** `ingest/ingest/artifacts.py:878`
- def **apply_prune_plan** `ingest/ingest/artifacts.py:967`

## ingest/ingest/chunking.py
- def **default_token_counter** `ingest/ingest/chunking.py:35`
- class **Chunk**() `ingest/ingest/chunking.py:41`
- def **_split_keep_pos** `ingest/ingest/chunking.py:53`
- def **_strip_span** `ingest/ingest/chunking.py:69`
- def **_split_sections** `ingest/ingest/chunking.py:76`
- def **heading_spans** `ingest/ingest/chunking.py:130`
- def **_split_oversized_run** `ingest/ingest/chunking.py:152`
- def **_atoms** `ingest/ingest/chunking.py:173`
- def **_pack** `ingest/ingest/chunking.py:234`
- def **chunk_document** `ingest/ingest/chunking.py:309`
- def **build_embed_text** `ingest/ingest/chunking.py:353`

## ingest/ingest/citations.py
imports: ingest/ingest/search.py
- class **CitationRef**() `ingest/ingest/citations.py:37`
- def **_nfc** `ingest/ingest/citations.py:75`
- def **_is_article_ref** `ingest/ingest/citations.py:79`
- def **_prefixed_alternates** `ingest/ingest/citations.py:83`
- def **extract_citation** `ingest/ingest/citations.py:89`
- def **_norm_alias_text** `ingest/ingest/citations.py:132`
- def **_cached_aliases** `ingest/ingest/citations.py:137`
- def **load_aliases** `ingest/ingest/citations.py:144`
- def **_match_alias** `ingest/ingest/citations.py:157`
- def **citation_lookup** `ingest/ingest/citations.py:184`
- def **_point_key** `ingest/ingest/citations.py:206`
- def **pin_points** `ingest/ingest/citations.py:214`

## ingest/ingest/collection_compatibility.py
imports: ingest/ingest/config.py, ingest/ingest/generation.py, ingest/ingest/qdrant_store.py
- class **CompatibilityIssue**() `ingest/ingest/collection_compatibility.py:20`
  - def to_dict `ingest/ingest/collection_compatibility.py:25`
- class **CollectionCompatibility**() `ingest/ingest/collection_compatibility.py:30`
  - def ok `ingest/ingest/collection_compatibility.py:39`
  - def coverage_issues `ingest/ingest/collection_compatibility.py:43`
  - def integrity_issues `ingest/ingest/collection_compatibility.py:47`
  - def to_dict `ingest/ingest/collection_compatibility.py:50`
- class **CollectionIncompatibleError**(RuntimeError) `ingest/ingest/collection_compatibility.py:62`
- def **config_manifest_issues** `ingest/ingest/collection_compatibility.py:66`
- def **require_config_manifest_compatibility** `ingest/ingest/collection_compatibility.py:127`
- def **expected_point_identity** `ingest/ingest/collection_compatibility.py:139`
- def **_value** `ingest/ingest/collection_compatibility.py:160`
- def **_distance** `ingest/ingest/collection_compatibility.py:166`
- def **_schema_issues** `ingest/ingest/collection_compatibility.py:173`
- def **_identity_filter** `ingest/ingest/collection_compatibility.py:262`
- def **check_collection_compatibility** `ingest/ingest/collection_compatibility.py:274`
- def **require_collection_compatibility** `ingest/ingest/collection_compatibility.py:392`

## ingest/ingest/config.py
- class **ConfigurationError**(ValueError) `ingest/ingest/config.py:21`
- def **_bool** `ingest/ingest/config.py:25`
- def **_int** `ingest/ingest/config.py:32`
- def **_str_opt** `ingest/ingest/config.py:37`
- def **_float_opt** `ingest/ingest/config.py:42`
- def **_device_opt** `ingest/ingest/config.py:53`
- def **_route_opt** `ingest/ingest/config.py:66`
- class **Config**() `ingest/ingest/config.py:73`
- def **_validate_revision** `ingest/ingest/config.py:133`
- def **_validate_model_name** `ingest/ingest/config.py:143`
- def **validate_production_config** `ingest/ingest/config.py:148`
- def **load_config** `ingest/ingest/config.py:176`
- def **_retrieval_fingerprint_material** `ingest/ingest/config.py:238`
- def **retrieval_fingerprint_sha256** `ingest/ingest/config.py:269`
- def **retrieval_fingerprint** `ingest/ingest/config.py:276`

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
- def **snapshot_doc_to_canonical** `ingest/ingest/embed_job.py:37`
- def **iter_snapshot_docs** `ingest/ingest/embed_job.py:77`
- def **load_snapshot_docs** `ingest/ingest/embed_job.py:92`
- def **dense_checksum** `ingest/ingest/embed_job.py:107`
- def **checksum_cosine** `ingest/ingest/embed_job.py:122`
- def **save_checksum_reference** `ingest/ingest/embed_job.py:132`
- def **_checkpoint_path** `ingest/ingest/embed_job.py:140`
- def **_load_ckpt** `ingest/ingest/embed_job.py:144`
- def **_save_ckpt** `ingest/ingest/embed_job.py:158`
- def **embed_docs** `ingest/ingest/embed_job.py:168`
- def **embed_source_resumable** `ingest/ingest/embed_job.py:194`

## ingest/ingest/embedding.py
imports: ingest/ingest/config.py
- class **Sparse**() `ingest/ingest/embedding.py:14`
- class **Embedded**() `ingest/ingest/embedding.py:20`
- def **_embedding_model_source** `ingest/ingest/embedding.py:25`
- class **BGEM3Embedder**() `ingest/ingest/embedding.py:40`
  - def __init__ `ingest/ingest/embedding.py:41`
  - def _encode `ingest/ingest/embedding.py:58`
  - def encode_passages `ingest/ingest/embedding.py:78`
  - def encode_query `ingest/ingest/embedding.py:81`
- def **make_token_counter** `ingest/ingest/embedding.py:85`

## ingest/ingest/generation.py
- class **GenerationFormatError**(ValueError) `ingest/ingest/generation.py:46`
- class **ChecksumMismatchError**(GenerationFormatError) `ingest/ingest/generation.py:50`
- def **_reject_duplicate_keys** `ingest/ingest/generation.py:54`
- def **_parse_json** `ingest/ingest/generation.py:63`
- def **_regular_file** `ingest/ingest/generation.py:72`
- def **_load_json** `ingest/ingest/generation.py:81`
- def **_object** `ingest/ingest/generation.py:91`
- def **_exact_keys** `ingest/ingest/generation.py:97`
- def **_string** `ingest/ingest/generation.py:110`
- def **_model_name** `ingest/ingest/generation.py:120`
- def **_optional_string** `ingest/ingest/generation.py:127`
- def **_integer** `ingest/ingest/generation.py:133`
- def **_boolean** `ingest/ingest/generation.py:139`
- def **_sha256** `ingest/ingest/generation.py:145`
- def **_optional_sha256** `ingest/ingest/generation.py:151`
- def **_revision** `ingest/ingest/generation.py:157`
- def **validate_generation_id** `ingest/ingest/generation.py:165`
- def **parse_rfc3339_utc** `ingest/ingest/generation.py:178`
- class **CorpusIdentity**() `ingest/ingest/generation.py:196`
  - def from_dict `ingest/ingest/generation.py:201`
- class **SourceIdentity**() `ingest/ingest/generation.py:213`
  - def from_dict `ingest/ingest/generation.py:218`
- class **ModelIdentity**() `ingest/ingest/generation.py:228`
  - def from_dict `ingest/ingest/generation.py:237`
- class **VectorSpaceIdentity**() `ingest/ingest/generation.py:274`
  - def from_dict `ingest/ingest/generation.py:282`
- class **ChunkingIdentity**() `ingest/ingest/generation.py:310`
  - def from_dict `ingest/ingest/generation.py:317`
- class **CoveredRun**() `ingest/ingest/generation.py:345`
  - def from_dict `ingest/ingest/generation.py:350`
- class **CodeIdentity**() `ingest/ingest/generation.py:360`
  - def from_dict `ingest/ingest/generation.py:365`
- class **DependencyIdentity**() `ingest/ingest/generation.py:382`
  - def from_dict `ingest/ingest/generation.py:387`
- class **CreationIdentity**() `ingest/ingest/generation.py:405`
  - def from_dict `ingest/ingest/generation.py:411`
- class **GenerationManifest**() `ingest/ingest/generation.py:424`
  - def from_dict `ingest/ingest/generation.py:444`
  - def to_dict `ingest/ingest/generation.py:539`
- class **DocumentRecord**() `ingest/ingest/generation.py:544`
  - def from_dict `ingest/ingest/generation.py:562`
  - def indexed `ingest/ingest/generation.py:673`
  - def to_dict `ingest/ingest/generation.py:676`
- class **SampleCheck**() `ingest/ingest/generation.py:681`
  - def from_dict `ingest/ingest/generation.py:691`
  - def to_dict `ingest/ingest/generation.py:744`
- class **ChecksumInventory**() `ingest/ingest/generation.py:749`
  - def from_dict `ingest/ingest/generation.py:755`
  - def to_dict `ingest/ingest/generation.py:790`
- def **_safe_artifact_name** `ingest/ingest/generation.py:798`
- def **load_manifest** `ingest/ingest/generation.py:811`
- def **load_checksums** `ingest/ingest/generation.py:815`
- def **_iter_jsonl** `ingest/ingest/generation.py:819`
- def **iter_document_records** `ingest/ingest/generation.py:846`
- def **iter_sample_checks** `ingest/ingest/generation.py:858`
- def **_file_sha256** `ingest/ingest/generation.py:870`
- def **verify_artifact_checksums** `ingest/ingest/generation.py:881`
- class **GenerationArtifacts**() `ingest/ingest/generation.py:930`
  - def iter_documents `ingest/ingest/generation.py:935`
  - def iter_samples `ingest/ingest/generation.py:941`
- def **load_generation** `ingest/ingest/generation.py:948`

## ingest/ingest/generation_scan.py
imports: ingest/ingest/__init__.py, ingest/ingest/config.py, ingest/ingest/generation.py, ingest/ingest/pipeline.py
- class **ScanIssue**() `ingest/ingest/generation_scan.py:46`
- class **_DocAccumulator**() `ingest/ingest/generation_scan.py:55`
- class **ScanResult**() `ingest/ingest/generation_scan.py:63`
- def **_source_identity** `ingest/ingest/generation_scan.py:70`
- def **_document_record_from_accumulator** `ingest/ingest/generation_scan.py:86`
- def **_sample_check** `ingest/ingest/generation_scan.py:144`
- def **aggregate_documents** `ingest/ingest/generation_scan.py:157`

## ingest/ingest/generation_snapshot.py
imports: ingest/ingest/generation.py
- class **GenerationPublishError**(RuntimeError) `ingest/ingest/generation_snapshot.py:50`
  - def __init__ `ingest/ingest/generation_snapshot.py:53`
- class **GenerationDestinationExists**(GenerationPublishError) `ingest/ingest/generation_snapshot.py:58`
- def **_lexists** `ingest/ingest/generation_snapshot.py:62`
- def **_rename_noreplace** `ingest/ingest/generation_snapshot.py:66`
- def **_create_private_directory** `ingest/ingest/generation_snapshot.py:106`
- def **_fsync_directory** `ingest/ingest/generation_snapshot.py:125`
- def **_publisher_lock** `ingest/ingest/generation_snapshot.py:134`
- def **_canonical_json_bytes** `ingest/ingest/generation_snapshot.py:148`
- def **_write_bytes** `ingest/ingest/generation_snapshot.py:171`
- def **_write_json** `ingest/ingest/generation_snapshot.py:189`
- def **_private_binary_writer** `ingest/ingest/generation_snapshot.py:194`
- def **_write_json_line** `ingest/ingest/generation_snapshot.py:208`
- def **_sha256_file** `ingest/ingest/generation_snapshot.py:212`
- def **source_state_sha256** `ingest/ingest/generation_snapshot.py:220`
- def **_as_manifest** `ingest/ingest/generation_snapshot.py:228`
- def **_as_document** `ingest/ingest/generation_snapshot.py:240`
- def **_as_sample** `ingest/ingest/generation_snapshot.py:252`
- def **_provenance** `ingest/ingest/generation_snapshot.py:264`
- def **_source_state_artifact** `ingest/ingest/generation_snapshot.py:282`
- def **_quarantine_record** `ingest/ingest/generation_snapshot.py:294`
- def **_open_validation_index** `ingest/ingest/generation_snapshot.py:314`
- def **_stream_documents** `ingest/ingest/generation_snapshot.py:328`
- def **_stream_samples** `ingest/ingest/generation_snapshot.py:380`
- def **_validate_observed_counts** `ingest/ingest/generation_snapshot.py:417`
- def **_checksum_inventory** `ingest/ingest/generation_snapshot.py:440`
- def **_validate_private_tree** `ingest/ingest/generation_snapshot.py:458`
- def **publish_generation** `ingest/ingest/generation_snapshot.py:472`

## ingest/ingest/hygiene.py
- def **strip_control** `ingest/ingest/hygiene.py:35`
- def **to_nfc** `ingest/ingest/hygiene.py:43`
- def **clean_text** `ingest/ingest/hygiene.py:48`
- class **DamageReport**() `ingest/ingest/hygiene.py:58`
  - def is_usable `ingest/ingest/hygiene.py:71`
- def **assess** `ingest/ingest/hygiene.py:75`

## ingest/ingest/integrity.py
imports: ingest/ingest/generation.py
- class **VerificationOutcome**() `ingest/ingest/integrity.py:33`
  - def to_dict `ingest/ingest/integrity.py:38`
- class **VerificationReport**() `ingest/ingest/integrity.py:47`
  - def ok `ingest/ingest/integrity.py:60`
  - def to_dict `ingest/ingest/integrity.py:71`
- class **_Issues**() `ingest/ingest/integrity.py:87`
  - def __init__ `ingest/ingest/integrity.py:88`
  - def add `ingest/ingest/integrity.py:93`
  - def outcome `ingest/ingest/integrity.py:98`
- def **_is_int** `ingest/ingest/integrity.py:106`
- def **_is_string** `ingest/ingest/integrity.py:110`
- def **_is_sha256** `ingest/ingest/integrity.py:114`
- def **_point_field** `ingest/ingest/integrity.py:120`
- def **_sparse_field** `ingest/ingest/integrity.py:126`
- def **_canonical_point_id** `ingest/ingest/integrity.py:132`
- def **_validate_vectors** `ingest/ingest/integrity.py:143`
- def **_expect_payload_value** `ingest/ingest/integrity.py:255`
- def **_create_tables** `ingest/ingest/integrity.py:275`
- def **_normalize_now** `ingest/ingest/integrity.py:311`
- def **_format_utc** `ingest/ingest/integrity.py:318`
- def **verify_generation_points** `ingest/ingest/integrity.py:322`
- def **verify_generation_artifacts** `ingest/ingest/integrity.py:1051`
- def **write_verification_report** `ingest/ingest/integrity.py:1074`

## ingest/ingest/mcp_server.py
imports: ingest/ingest/__init__.py, ingest/ingest/collection_compatibility.py, ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/generation.py, ingest/ingest/querylog.py, ingest/ingest/remote_search.py, ingest/ingest/rerank.py, ingest/ingest/retrieval.py, ingest/ingest/search.py, ingest/ingest/sources.py
- def **_get_cfg** `ingest/ingest/mcp_server.py:81`
- def **_configured_generation_manifest** `ingest/ingest/mcp_server.py:88`
- def **_install_verified_worker_runtime** `ingest/ingest/mcp_server.py:102`
- def **_local_readiness** `ingest/ingest/mcp_server.py:122`
- def **_require_local_readiness** `ingest/ingest/mcp_server.py:198`
- def **_cache_key** `ingest/ingest/mcp_server.py:219`
- def **_cache_get** `ingest/ingest/mcp_server.py:225`
- def **_cache_put** `ingest/ingest/mcp_server.py:237`
- def **_cacheable_search_result** `ingest/ingest/mcp_server.py:244`
- def **_get_client** `ingest/ingest/mcp_server.py:265`
- def **_get_embedder** `ingest/ingest/mcp_server.py:272`
- def **_get_reranker** `ingest/ingest/mcp_server.py:284`
- def **_is_remote_reranker** `ingest/ingest/mcp_server.py:307`
- def **_use_remote** `ingest/ingest/mcp_server.py:317`
- def **_get_remote_client** `ingest/ingest/mcp_server.py:321`
- def **_remote_op** `ingest/ingest/mcp_server.py:333`
- def **_remote_error_text** `ingest/ingest/mcp_server.py:346`
- def **_publish_manifest** `ingest/ingest/mcp_server.py:356`
- def **_handle_error** `ingest/ingest/mcp_server.py:365`
- class **ResponseFormat**(str, Enum) `ingest/ingest/mcp_server.py:377`
- def **_hit_dict** `ingest/ingest/mcp_server.py:384`
- def **_format_hit_md** `ingest/ingest/mcp_server.py:413`
- class **SearchInput**(BaseModel) `ingest/ingest/mcp_server.py:439`
- def **_retrieval_request** `ingest/ingest/mcp_server.py:511`
- def **legal_search** `ingest/ingest/mcp_server.py:545`
- def **_log_query_remote** `ingest/ingest/mcp_server.py:694`
- def **_log_query** `ingest/ingest/mcp_server.py:718`
- class **GetDocumentInput**(BaseModel) `ingest/ingest/mcp_server.py:739`
- def **_scroll_all** `ingest/ingest/mcp_server.py:756`
- def **_stitch_overlap** `ingest/ingest/mcp_server.py:775`
- def **legal_get_document** `ingest/ingest/mcp_server.py:813`
- def **_dedup_documents** `ingest/ingest/mcp_server.py:892`
- def **_condition_list** `ingest/ingest/mcp_server.py:954`
- def **_document_scroll_filter** `ingest/ingest/mcp_server.py:960`
- def **_iter_document_points** `ingest/ingest/mcp_server.py:973`
- def **_format_doc_line** `ingest/ingest/mcp_server.py:998`
- class **LookupInput**(BaseModel) `ingest/ingest/mcp_server.py:1014`
- def **legal_lookup** `ingest/ingest/mcp_server.py:1049`
- class **BrowseInput**(BaseModel) `ingest/ingest/mcp_server.py:1096`
- def **legal_browse** `ingest/ingest/mcp_server.py:1145`
- def **_source_counts** `ingest/ingest/mcp_server.py:1216`
- def **legal_collection_info** `ingest/ingest/mcp_server.py:1235`
- def **_latest_report** `ingest/ingest/mcp_server.py:1298`
- class **StatusInput**(BaseModel) `ingest/ingest/mcp_server.py:1308`
- def **ingest_status** `ingest/ingest/mcp_server.py:1328`
- class **GetVersionsInput**(BaseModel) `ingest/ingest/mcp_server.py:1392`
- def **legal_get_document_versions** `ingest/ingest/mcp_server.py:1414`
- def **legal_health** `ingest/ingest/mcp_server.py:1490`
- def **_is_our_server** `ingest/ingest/mcp_server.py:1551`
- def **_enforce_singleton** `ingest/ingest/mcp_server.py:1561`
- def **main** `ingest/ingest/mcp_server.py:1585`

## ingest/ingest/operational.py
- def **require_explicit_approval** `ingest/ingest/operational.py:20`
- def **require_run_scoped_delta_collection** `ingest/ingest/operational.py:36`
- def **refuse_legacy_operation** `ingest/ingest/operational.py:46`

## ingest/ingest/pipeline.py
imports: ingest/ingest/__init__.py, ingest/ingest/chunking.py, ingest/ingest/config.py, ingest/ingest/dedup.py, ingest/ingest/hygiene.py, ingest/ingest/sources.py
- class **DocumentQuarantined**(ValueError) `ingest/ingest/pipeline.py:38`
  - def __init__ `ingest/ingest/pipeline.py:41`
- def **_prepare_doc_for_index** `ingest/ingest/pipeline.py:48`
- def **_header_v2_kwargs** `ingest/ingest/pipeline.py:67`
- def **_validated_embeddings** `ingest/ingest/pipeline.py:80`
- def **_canonical_date** `ingest/ingest/pipeline.py:129`
- def **_document_state_hash** `ingest/ingest/pipeline.py:133`
- def **_indexed_document_state** `ingest/ingest/pipeline.py:179`
- def **_record_schema_drift** `ingest/ingest/pipeline.py:198`
- def **write_ingest_report** `ingest/ingest/pipeline.py:214`
- def **items_path** `ingest/ingest/pipeline.py:254`
- def **_iter_lines** `ingest/ingest/pipeline.py:258`
- def **_checkpoint_path** `ingest/ingest/pipeline.py:266`
- def **_load_checkpoint** `ingest/ingest/pipeline.py:270`
- def **_save_checkpoint** `ingest/ingest/pipeline.py:283`
- def **delete_checkpoint** `ingest/ingest/pipeline.py:293`
- def **ingest_source** `ingest/ingest/pipeline.py:297`
- def **resolve_sources** `ingest/ingest/pipeline.py:462`
- def **discover_runs** `ingest/ingest/pipeline.py:484`
- def **_read_complete_lines** `ingest/ingest/pipeline.py:502`
- def **_watch_state_path** `ingest/ingest/pipeline.py:532`
- def **_load_watch_state** `ingest/ingest/pipeline.py:536`
- def **_save_watch_state** `ingest/ingest/pipeline.py:545`
- def **delete_watch_state** `ingest/ingest/pipeline.py:554`
- def **_build_doc_points** `ingest/ingest/pipeline.py:558`
- def **watch_drain_source** `ingest/ingest/pipeline.py:605`
- def **watch_loop** `ingest/ingest/pipeline.py:785`

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

## ingest/ingest/promotion.py
imports: ingest/ingest/artifacts.py, ingest/ingest/generation.py
- class **PromotionError**(RuntimeError) `ingest/ingest/promotion.py:60`
- class **PromotionPreconditionError**(PromotionError) `ingest/ingest/promotion.py:64`
- class **PromotionLockedError**(PromotionError) `ingest/ingest/promotion.py:68`
- class **PromotionPlanExists**(PromotionError) `ingest/ingest/promotion.py:72`
- def **_require_string** `ingest/ingest/promotion.py:76`
- def **_require_model_name** `ingest/ingest/promotion.py:87`
- def **_require_sha256** `ingest/ingest/promotion.py:94`
- def **_require_revision** `ingest/ingest/promotion.py:100`
- def **_require_int** `ingest/ingest/promotion.py:108`
- def **_require_bool** `ingest/ingest/promotion.py:114`
- def **_require_timestamp** `ingest/ingest/promotion.py:120`
- def **_exact_keys** `ingest/ingest/promotion.py:128`
- def **_canonical_bytes** `ingest/ingest/promotion.py:136`
- def **_format_utc** `ingest/ingest/promotion.py:152`
- def **physical_collection_name** `ingest/ingest/promotion.py:160`
- class **ExpectedCollectionIdentity**() `ingest/ingest/promotion.py:166`
  - def from_dict `ingest/ingest/promotion.py:185`
  - def to_dict `ingest/ingest/promotion.py:246`
- class **PromotionPlan**() `ingest/ingest/promotion.py:251`
  - def from_dict `ingest/ingest/promotion.py:266`
  - def to_dict `ingest/ingest/promotion.py:325`
- class **CollectionInspection**() `ingest/ingest/promotion.py:343`
- class **PromotionBackend**(Protocol) `ingest/ingest/promotion.py:370`
  - def restore_candidate `ingest/ingest/promotion.py:373`
  - def wait_for_green `ingest/ingest/promotion.py:376`
  - def smoke `ingest/ingest/promotion.py:381`
  - def readiness `ingest/ingest/promotion.py:384`
  - def alias_target `ingest/ingest/promotion.py:387`
  - def switch_alias `ingest/ingest/promotion.py:390`
- def **create_promotion_plan** `ingest/ingest/promotion.py:394`
- def **promotion_plan_sha256** `ingest/ingest/promotion.py:475`
- def **write_promotion_plan** `ingest/ingest/promotion.py:479`
- def **_load_json** `ingest/ingest/promotion.py:505`
- def **load_promotion_plan** `ingest/ingest/promotion.py:523`
- def **stat_mode** `ingest/ingest/promotion.py:527`
- def **_require_private_regular_file** `ingest/ingest/promotion.py:531`
- def **_require_private_generation_tree** `ingest/ingest/promotion.py:544`
- def **_create_private_directories** `ingest/ingest/promotion.py:565`
- def **_fsync_directory** `ingest/ingest/promotion.py:588`
- def **_file_sha256** `ingest/ingest/promotion.py:596`
- class **PromotionState**() `ingest/ingest/promotion.py:605`
  - def from_dict `ingest/ingest/promotion.py:625`
  - def to_dict `ingest/ingest/promotion.py:746`
- def **load_promotion_state** `ingest/ingest/promotion.py:752`
- def **_write_state** `ingest/ingest/promotion.py:759`
- def **promotion_lock** `ingest/ingest/promotion.py:766`
- def **_initial_state** `ingest/ingest/promotion.py:786`
- def **_advance** `ingest/ingest/promotion.py:810`
- def **_candidate_mismatches** `ingest/ingest/promotion.py:828`
- def **_assert_alias** `ingest/ingest/promotion.py:889`
- def **_persist_advance** `ingest/ingest/promotion.py:897`
- def **_attempt_emergency_rollback** `ingest/ingest/promotion.py:911`
- def **execute_promotion** `ingest/ingest/promotion.py:970`

## ingest/ingest/qdrant_promotion.py
imports: ingest/ingest/artifacts.py, ingest/ingest/config.py, ingest/ingest/generation.py, ingest/ingest/integrity.py, ingest/ingest/promotion.py, ingest/ingest/qdrant_store.py
- class **CandidateVerificationProof**() `ingest/ingest/qdrant_promotion.py:39`
- def **_value** `ingest/ingest/qdrant_promotion.py:52`
- def **_normal** `ingest/ingest/qdrant_promotion.py:58`
- def **_sha256_file** `ingest/ingest/qdrant_promotion.py:63`
- def **_verified_snapshot_location** `ingest/ingest/qdrant_promotion.py:71`
- def **_stream_points** `ingest/ingest/qdrant_promotion.py:105`
- def **_generation_integrity_check** `ingest/ingest/qdrant_promotion.py:137`
- class **QdrantPromotionBackend**() `ingest/ingest/qdrant_promotion.py:190`
  - def __init__ `ingest/ingest/qdrant_promotion.py:193`
  - def restore_candidate `ingest/ingest/qdrant_promotion.py:219`
  - def _expected_payload `ingest/ingest/qdrant_promotion.py:255`
  - def _identity_count `ingest/ingest/qdrant_promotion.py:272`
  - def _inspect `ingest/ingest/qdrant_promotion.py:285`
  - def wait_for_green `ingest/ingest/qdrant_promotion.py:349`
  - def smoke `ingest/ingest/qdrant_promotion.py:409`
  - def readiness `ingest/ingest/qdrant_promotion.py:415`
  - def alias_target `ingest/ingest/qdrant_promotion.py:426`
  - def switch_alias `ingest/ingest/qdrant_promotion.py:440`
- def **make_qdrant_promotion_backend** `ingest/ingest/qdrant_promotion.py:467`

## ingest/ingest/qdrant_store.py
imports: ingest/ingest/chunking.py, ingest/ingest/config.py, ingest/ingest/dedup.py, ingest/ingest/embedding.py, ingest/ingest/generation.py, ingest/ingest/operational.py, ingest/ingest/sources.py
- def **_identity_sha256** `ingest/ingest/qdrant_store.py:79`
- def **vector_space_id** `ingest/ingest/qdrant_store.py:84`
- def **chunking_fingerprint** `ingest/ingest/qdrant_store.py:100`
- class **GenerationPointIdentity**() `ingest/ingest/qdrant_store.py:115`
  - def as_payload `ingest/ingest/qdrant_store.py:129`
- def **generation_point_identity** `ingest/ingest/qdrant_store.py:146`
- def **validate_generation_write_target** `ingest/ingest/qdrant_store.py:210`
- def **_refuse_serving_collection_mutation** `ingest/ingest/qdrant_store.py:256`
- def **make_client** `ingest/ingest/qdrant_store.py:264`
- def **point_id** `ingest/ingest/qdrant_store.py:279`
- def **_assert_dense_dim** `ingest/ingest/qdrant_store.py:284`
- def **ensure_collection** `ingest/ingest/qdrant_store.py:298`
- def **_rfc3339** `ingest/ingest/qdrant_store.py:370`
- def **build_payload** `ingest/ingest/qdrant_store.py:385`
- def **sparse_vector** `ingest/ingest/qdrant_store.py:449`
- def **point_struct** `ingest/ingest/qdrant_store.py:453`
- def **upsert_points** `ingest/ingest/qdrant_store.py:457`
- def **delete_doc_chunks_from** `ingest/ingest/qdrant_store.py:463`

## ingest/ingest/querylog.py
- def **build_query_record** `ingest/ingest/querylog.py:17`
- def **append_query_log** `ingest/ingest/querylog.py:45`

## ingest/ingest/remote_search.py
- class **RemoteSearchError**(RuntimeError) `ingest/ingest/remote_search.py:37`
- class **EndpointWarmingUp**(RemoteSearchError) `ingest/ingest/remote_search.py:41`
- class **RemoteOpError**(RemoteSearchError) `ingest/ingest/remote_search.py:45`
- class **RunPodQueueClient**() `ingest/ingest/remote_search.py:49`
  - def __init__ `ingest/ingest/remote_search.py:52`
  - def _request `ingest/ingest/remote_search.py:63`
  - def health `ingest/ingest/remote_search.py:87`
  - def call `ingest/ingest/remote_search.py:91`

## ingest/ingest/rerank.py
imports: ingest/ingest/config.py
- def **_revision_kwargs** `ingest/ingest/rerank.py:25`
- def **_auto_device** `ingest/ingest/rerank.py:30`
- def **_configure_cpu_threads** `ingest/ingest/rerank.py:38`
- class **BGEReranker**() `ingest/ingest/rerank.py:57`
  - def __init__ `ingest/ingest/rerank.py:60`
  - def score `ingest/ingest/rerank.py:79`
- class **ONNXBGEReranker**() `ingest/ingest/rerank.py:106`
  - def __init__ `ingest/ingest/rerank.py:116`
  - def score `ingest/ingest/rerank.py:137`
- def **make_reranker** `ingest/ingest/rerank.py:160`
- class **RemoteBGEReranker**() `ingest/ingest/rerank.py:167`
  - def __init__ `ingest/ingest/rerank.py:177`
  - def score `ingest/ingest/rerank.py:182`

## ingest/ingest/retrieval.py
imports: ingest/ingest/config.py, ingest/ingest/search.py
- class **TemporalContext**() `ingest/ingest/retrieval.py:22`
  - def as_filters `ingest/ingest/retrieval.py:29`
- class **RetrievalRequest**() `ingest/ingest/retrieval.py:41`
  - def __post_init__ `ingest/ingest/retrieval.py:52`
  - def search_kwargs `ingest/ingest/retrieval.py:59`
- class **RetrievalPolicy**() `ingest/ingest/retrieval.py:75`
  - def from_config `ingest/ingest/retrieval.py:91`
  - def candidate_depth `ingest/ingest/retrieval.py:98`
- class **RetrievalOutcome**() `ingest/ingest/retrieval.py:107`
  - def __post_init__ `ingest/ingest/retrieval.py:121`
- class **EvaluationProvenance**() `ingest/ingest/retrieval.py:127`
  - def __post_init__ `ingest/ingest/retrieval.py:153`
  - def validate_complete `ingest/ingest/retrieval.py:158`
  - def to_dict `ingest/ingest/retrieval.py:224`
- def **_is_remote_reranker** `ingest/ingest/retrieval.py:252`
- def **execute_retrieval** `ingest/ingest/retrieval.py:256`

## ingest/ingest/search.py
imports: ingest/ingest/citations.py, ingest/ingest/config.py
- def **detect_language** `ingest/ingest/search.py:18`
- def **_date_bound** `ingest/ingest/search.py:27`
- def **build_filter** `ingest/ingest/search.py:33`
- def **hybrid_search** `ingest/ingest/search.py:104`
- def **_point_dense** `ingest/ingest/search.py:218`
- def **_cosine** `ingest/ingest/search.py:226`
- def **diversify** `ingest/ingest/search.py:235`
- def **rerank_points** `ingest/ingest/search.py:305`

## ingest/ingest/snapshot.py
imports: ingest/ingest/__init__.py, ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/sources.py
- def **_run_files_desc** `ingest/ingest/snapshot.py:42`
- def **config_hash** `ingest/ingest/snapshot.py:55`
- class **SourceStats**() `ingest/ingest/snapshot.py:86`
  - def __post_init__ `ingest/ingest/snapshot.py:106`
- def **_snapshot_record** `ingest/ingest/snapshot.py:115`
- def **build_snapshot** `ingest/ingest/snapshot.py:157`
- def **_maybe_token_counter** `ingest/ingest/snapshot.py:274`
- def **_token_pctls** `ingest/ingest/snapshot.py:282`
- def **_near_dup_for_source** `ingest/ingest/snapshot.py:297`
- def **_pctl** `ingest/ingest/snapshot.py:308`
- def **_write_reports_and_manifest** `ingest/ingest/snapshot.py:315`

## ingest/ingest/sources.py
- def **normalize_status** `ingest/ingest/sources.py:49`
- def **_georgian_month** `ingest/ingest/sources.py:64`
- def **_valid_iso** `ingest/ingest/sources.py:73`
- def **_parse_date** `ingest/ingest/sources.py:82`
- class **CanonicalDoc**() `ingest/ingest/sources.py:114`
- class **SourceSpec**() `ingest/ingest/sources.py:156`
  - def declared_keys `ingest/ingest/sources.py:178`
  - def _first `ingest/ingest/sources.py:201`
  - def _parties `ingest/ingest/sources.py:208`
  - def build `ingest/ingest/sources.py:216`
- def **normalize** `ingest/ingest/sources.py:420`
- def **schema_drift** `ingest/ingest/sources.py:428`

## ingest/ingest/structure.py
- class **StructureInfo**() `ingest/ingest/structure.py:34`
- def **detect** `ingest/ingest/structure.py:44`
- def **article_spans** `ingest/ingest/structure.py:74`

## ingest/eval/.golden_v2_pending/add_holdout.py

## ingest/eval/.golden_v2_pending/assemble_batch.py
- def **nfc** `ingest/eval/.golden_v2_pending/assemble_batch.py:25`
- def **body_path** `ingest/eval/.golden_v2_pending/assemble_batch.py:29`

## ingest/eval/.golden_v2_pending/prepare_authoring.py

## ingest/eval/__init__.py

## ingest/eval/answer_eval.py
imports: ingest/eval/goldset.py, ingest/eval/metrics.py
- def **gold_doc_key** `ingest/eval/answer_eval.py:52`
- def **top1_doc** `ingest/eval/answer_eval.py:57`
- def **top1_score** `ingest/eval/answer_eval.py:63`
- def **confident_wrong** `ingest/eval/answer_eval.py:68`
- def **identity_at_1** `ingest/eval/answer_eval.py:82`
- def **identity_at_k** `ingest/eval/answer_eval.py:87`
- def **span_coverage_at_k** `ingest/eval/answer_eval.py:93`
- def **fully_grounded_at_k** `ingest/eval/answer_eval.py:107`
- class **AnswerScore**() `ingest/eval/answer_eval.py:113`
- def **score_answer** `ingest/eval/answer_eval.py:130`
- def **_mean** `ingest/eval/answer_eval.py:151`
- def **aggregate_answer_scores** `ingest/eval/answer_eval.py:155`
- def **breakdown_by** `ingest/eval/answer_eval.py:175`
- def **identity_failures** `ingest/eval/answer_eval.py:183`
- class **AbstentionResult**() `ingest/eval/answer_eval.py:196`
- def **abstention_correctness** `ingest/eval/answer_eval.py:208`

## ingest/eval/backend.py
imports: ingest/eval/bm25.py, ingest/eval/bm25_full.py, ingest/eval/metrics.py, ingest/ingest/citations.py, ingest/ingest/retrieval.py, ingest/ingest/search.py
- class **ChunkRecord**() `ingest/eval/backend.py:29`
- def **_rrf_fuse** `ingest/eval/backend.py:36`
- class **FakeBackend**() `ingest/eval/backend.py:45`
  - def __init__ `ingest/eval/backend.py:53`
  - def _to_dense `ingest/eval/backend.py:69`
  - def _hit `ingest/eval/backend.py:74`
  - def _dense_rank `ingest/eval/backend.py:78`
  - def _sparse_rank `ingest/eval/backend.py:87`
  - def search `ingest/eval/backend.py:96`
- class **ProductionBackend**() `ingest/eval/backend.py:148`
  - def __init__ `ingest/eval/backend.py:156`
  - def _hits `ingest/eval/backend.py:165`
  - def search `ingest/eval/backend.py:179`
- class **QdrantBackend**() `ingest/eval/backend.py:207`
  - def __init__ `ingest/eval/backend.py:214`
  - def _diversity_on `ingest/eval/backend.py:242`
  - def _candidate `ingest/eval/backend.py:245`
  - def _search_params `ingest/eval/backend.py:257`
  - def _fusion_query `ingest/eval/backend.py:265`
  - def _manual_fusion `ingest/eval/backend.py:269`
  - def _points_to_hits `ingest/eval/backend.py:303`
  - def _ensure_bm25 `ingest/eval/backend.py:312`
  - def search `ingest/eval/backend.py:348`

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

## ingest/eval/diagnose_confident_wrong.py
imports: ingest/eval/__init__.py, ingest/ingest/citations.py, ingest/ingest/config.py, ingest/ingest/qdrant_store.py
- def **_candidates** `ingest/eval/diagnose_confident_wrong.py:42`
- def **_carries_cited_id** `ingest/eval/diagnose_confident_wrong.py:47`
- def **classify** `ingest/eval/diagnose_confident_wrong.py:56`
- def **main** `ingest/eval/diagnose_confident_wrong.py:69`

## ingest/eval/dump_judge_batch.py
imports: ingest/eval/__init__.py, ingest/eval/evaluate.py, ingest/eval/translations.py, ingest/ingest/chunking.py, ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/qdrant_store.py
- def **_token_counter** `ingest/eval/dump_judge_batch.py:33`
- def **_stratified_sample** `ingest/eval/dump_judge_batch.py:43`
- def **main** `ingest/eval/dump_judge_batch.py:59`

## ingest/eval/eval_answer_quality.py
imports: ingest/eval/__init__.py, ingest/eval/answer_eval.py, ingest/eval/backend.py, ingest/eval/evaluate.py, ingest/eval/translations.py, ingest/ingest/chunking.py, ingest/ingest/config.py, ingest/ingest/embedding.py
- def **_token_counter** `ingest/eval/eval_answer_quality.py:42`
- def **_load_unanswerable** `ingest/eval/eval_answer_quality.py:52`
- def **_chunkable_gold** `ingest/eval/eval_answer_quality.py:58`
- def **main** `ingest/eval/eval_answer_quality.py:80`

## ingest/eval/evaluate.py
imports: ingest/eval/__init__.py, ingest/eval/backend.py, ingest/eval/metrics.py, ingest/eval/spanmap.py, ingest/eval/stats.py, ingest/eval/translations.py, ingest/ingest/artifacts.py, ingest/ingest/chunking.py, ingest/ingest/collection_compatibility.py, ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/generation.py, ingest/ingest/qdrant_store.py, ingest/ingest/rerank.py, ingest/ingest/retrieval.py
- def **_token_counter** `ingest/eval/evaluate.py:46`
- def **build_query_relevance** `ingest/eval/evaluate.py:54`
- def **build_fake_corpus** `ingest/eval/evaluate.py:73`
- def **eval_set_knob** `ingest/eval/evaluate.py:115`
- def **_file_sha256** `ingest/eval/evaluate.py:126`
- def **run_mode** `ingest/eval/evaluate.py:134`
- def **selected_modes** `ingest/eval/evaluate.py:156`
- def **_fmt_ci** `ingest/eval/evaluate.py:172`
- def **print_table** `ingest/eval/evaluate.py:176`
- def **print_breakdowns** `ingest/eval/evaluate.py:190`
- def **print_stage_latency** `ingest/eval/evaluate.py:200`
- def **print_degraded** `ingest/eval/evaluate.py:211`
- def **_value** `ingest/eval/evaluate.py:226`
- def **_resolve_physical_collection** `ingest/eval/evaluate.py:230`
- def **_header_identity** `ingest/eval/evaluate.py:249`
- def **build_evaluation_provenance** `ingest/eval/evaluate.py:256`
- def **qdrant_deps** `ingest/eval/evaluate.py:291`
- def **make_backend** `ingest/eval/evaluate.py:366`
- def **main** `ingest/eval/evaluate.py:377`

## ingest/eval/explog.py
- def **config_hash** `ingest/eval/explog.py:21`
- def **now_iso** `ingest/eval/explog.py:28`
- def **append_run** `ingest/eval/explog.py:32`
- def **read_log** `ingest/eval/explog.py:40`

## ingest/eval/goldset.py
imports: ingest/eval/spanmap.py
- class **EvalSetSpec**() `ingest/eval/goldset.py:38`
- def **_nfc** `ingest/eval/goldset.py:59`
- class **Relevance**() `ingest/eval/goldset.py:64`
- class **GoldQuery**() `ingest/eval/goldset.py:73`
- def **load_golden_set** `ingest/eval/goldset.py:87`
- def **load_holdout** `ingest/eval/goldset.py:124`
- def **eval_set_hash** `ingest/eval/goldset.py:129`
- def **gold_docs** `ingest/eval/goldset.py:139`
- class **SnapshotBodies**() `ingest/eval/goldset.py:144`
  - def __init__ `ingest/eval/goldset.py:158`
  - def source_files `ingest/eval/goldset.py:173`
  - def _merged_cache `ingest/eval/goldset.py:177`
  - def _load_source `ingest/eval/goldset.py:183`
  - def body `ingest/eval/goldset.py:222`
- def **reground** `ingest/eval/goldset.py:229`
- def **enforce_holdout** `ingest/eval/goldset.py:251`
- def **lint_span_coverage** `ingest/eval/goldset.py:261`

## ingest/eval/judge_eval.py
- class **JudgeVerdict**() `ingest/eval/judge_eval.py:37`
- def **_b** `ingest/eval/judge_eval.py:51`
- def **verdict_from_dict** `ingest/eval/judge_eval.py:58`
- def **load_verdicts** `ingest/eval/judge_eval.py:72`
- def **_mean** `ingest/eval/judge_eval.py:84`
- def **merge_panel** `ingest/eval/judge_eval.py:88`
- class **JudgeAggregate**() `ingest/eval/judge_eval.py:125`
- def **aggregate_verdicts** `ingest/eval/judge_eval.py:133`
- def **breakdown_verdicts** `ingest/eval/judge_eval.py:146`

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

## ingest/eval/score_judge.py
imports: ingest/eval/judge_eval.py
- def **main** `ingest/eval/score_judge.py:18`

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
imports: ingest/ingest/config.py, ingest/ingest/operational.py, ingest/ingest/pipeline.py, ingest/ingest/qdrant_store.py
- def **_flush** `ingest/scripts/backfill_consolidation.py:35`
- def **main** `ingest/scripts/backfill_consolidation.py:53`

## ingest/scripts/build_phase_c_report.py
- def **load_rows** `ingest/scripts/build_phase_c_report.py:25`
- def **cell** `ingest/scripts/build_phase_c_report.py:29`
- def **lat_cell** `ingest/scripts/build_phase_c_report.py:37`
- def **extra_knobs** `ingest/scripts/build_phase_c_report.py:42`
- def **rc_of** `ingest/scripts/build_phase_c_report.py:48`
- def **_index_key** `ingest/scripts/build_phase_c_report.py:52`
- def **dedup_latest** `ingest/scripts/build_phase_c_report.py:62`
- def **metric_table** `ingest/scripts/build_phase_c_report.py:78`
- def **slice_table** `ingest/scripts/build_phase_c_report.py:99`
- def **parse_paired** `ingest/scripts/build_phase_c_report.py:121`
- def **paired_section** `ingest/scripts/build_phase_c_report.py:144`
- def **main** `ingest/scripts/build_phase_c_report.py:160`

## ingest/scripts/build_snapshot_delta.py
imports: ingest/ingest/__init__.py, ingest/ingest/config.py, ingest/ingest/qdrant_store.py, ingest/ingest/snapshot.py, ingest/ingest/sources.py
- def **load_id_map** `ingest/scripts/build_snapshot_delta.py:58`
- def **already_in_v1** `ingest/scripts/build_snapshot_delta.py:66`
- def **fetch_payload_hashes** `ingest/scripts/build_snapshot_delta.py:85`
- def **build_delta** `ingest/scripts/build_snapshot_delta.py:102`
- def **main** `ingest/scripts/build_snapshot_delta.py:244`

## ingest/scripts/calibrate_min_score.py
imports: ingest/eval/__init__.py, ingest/eval/backend.py, ingest/eval/evaluate.py, ingest/eval/translations.py, ingest/ingest/config.py, ingest/ingest/embedding.py
- def **percentile** `ingest/scripts/calibrate_min_score.py:30`
- def **main** `ingest/scripts/calibrate_min_score.py:37`

## ingest/scripts/create_generation.py
imports: ingest/ingest/generation.py, ingest/ingest/generation_snapshot.py
- def **_reject_duplicate_keys** `ingest/scripts/create_generation.py:23`
- def **_reject_nonstandard_constant** `ingest/scripts/create_generation.py:34`
- def **_parser** `ingest/scripts/create_generation.py:38`
- def **main** `ingest/scripts/create_generation.py:51`

## ingest/scripts/embed_delta.py
imports: ingest/ingest/__init__.py, ingest/ingest/chunking.py, ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/operational.py, ingest/ingest/pipeline.py, ingest/ingest/qdrant_store.py, ingest/ingest/sources.py
- class **_SQLiteDocuments**() `ingest/scripts/embed_delta.py:64`
  - def __init__ `ingest/scripts/embed_delta.py:74`
  - def _serialize `ingest/scripts/embed_delta.py:97`
  - def store_last `ingest/scripts/embed_delta.py:104`
  - def store_first `ingest/scripts/embed_delta.py:116`
  - def finish `ingest/scripts/embed_delta.py:128`
  - def exclude `ingest/scripts/embed_delta.py:131`
  - def items `ingest/scripts/embed_delta.py:138`
  - def __len__ `ingest/scripts/embed_delta.py:154`
  - def close `ingest/scripts/embed_delta.py:160`
  - def __enter__ `ingest/scripts/embed_delta.py:169`
  - def __exit__ `ingest/scripts/embed_delta.py:172`
- class **_SQLiteDocumentCollection**() `ingest/scripts/embed_delta.py:176`
  - def __init__ `ingest/scripts/embed_delta.py:179`
  - def __iter__ `ingest/scripts/embed_delta.py:183`
  - def __len__ `ingest/scripts/embed_delta.py:188`
  - def close `ingest/scripts/embed_delta.py:191`
  - def __enter__ `ingest/scripts/embed_delta.py:198`
  - def __exit__ `ingest/scripts/embed_delta.py:201`
- def **_write_json_atomic** `ingest/scripts/embed_delta.py:205`
- def **_resolve_paths** `ingest/scripts/embed_delta.py:213`
- def **_input_sha256** `ingest/scripts/embed_delta.py:223`
- def **_load_items** `ingest/scripts/embed_delta.py:237`
- def **_analyze_docs** `ingest/scripts/embed_delta.py:312`
- def **validate_expected_manifest** `ingest/scripts/embed_delta.py:394`
- def **_embed_docs_exact** `ingest/scripts/embed_delta.py:407`
- def **_strict_manifest_check** `ingest/scripts/embed_delta.py:459`
- def **main** `ingest/scripts/embed_delta.py:469`

## ingest/scripts/export_onnx_reranker.py
imports: ingest/ingest/config.py
- def **_load_reranker** `ingest/scripts/export_onnx_reranker.py:28`
- def **main** `ingest/scripts/export_onnx_reranker.py:39`

## ingest/scripts/finetune_reranker.py
imports: ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/qdrant_store.py, ingest/ingest/search.py
- def **_load_pairs** `ingest/scripts/finetune_reranker.py:35`
- def **_excluded_docs** `ingest/scripts/finetune_reranker.py:46`
- def **_load_cross_encoder** `ingest/scripts/finetune_reranker.py:60`
- def **main** `ingest/scripts/finetune_reranker.py:69`

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
imports: ingest/ingest/config.py, ingest/ingest/operational.py, ingest/ingest/qdrant_store.py
- def **_to_struct** `ingest/scripts/merge_delta_collection.py:69`
- class **_DocVersion**() `ingest/scripts/merge_delta_collection.py:94`
- class **MergeExpectations**() `ingest/scripts/merge_delta_collection.py:100`
- class **DeltaAudit**() `ingest/scripts/merge_delta_collection.py:113`
- class **RollbackArtifact**() `ingest/scripts/merge_delta_collection.py:121`
- class **MergeOutcome**() `ingest/scripts/merge_delta_collection.py:133`
- def **_canonical_document_keys** `ingest/scripts/merge_delta_collection.py:144`
- def **_document_ids_sha256** `ingest/scripts/merge_delta_collection.py:150`
- def **_positive_int** `ingest/scripts/merge_delta_collection.py:155`
- def **_normalise_document_ids** `ingest/scripts/merge_delta_collection.py:161`
- def **_load_expected_ids** `ingest/scripts/merge_delta_collection.py:190`
- def **load_merge_expectations** `ingest/scripts/merge_delta_collection.py:203`
- def **_point_metadata** `ingest/scripts/merge_delta_collection.py:323`
- def **_audit_delta** `ingest/scripts/merge_delta_collection.py:360`
- def **_scan_complete_docs** `ingest/scripts/merge_delta_collection.py:502`
- def **_delete_stale_tails** `ingest/scripts/merge_delta_collection.py:509`
- def **merge_collection** `ingest/scripts/merge_delta_collection.py:545`
- def **_source_document_ids** `ingest/scripts/merge_delta_collection.py:631`
- def **verify_destination_coverage** `ingest/scripts/merge_delta_collection.py:664`
- def **_sha256_path** `ingest/scripts/merge_delta_collection.py:746`
- def **_fsync_directory** `ingest/scripts/merge_delta_collection.py:754`
- def **_write_json_atomic** `ingest/scripts/merge_delta_collection.py:762`
- def **_qdrant_snapshot_timeout** `ingest/scripts/merge_delta_collection.py:775`
- def **qdrant_write_lock** `ingest/scripts/merge_delta_collection.py:791`
- def **download_qdrant_snapshot** `ingest/scripts/merge_delta_collection.py:828`
- def **create_rollback_snapshot** `ingest/scripts/merge_delta_collection.py:858`
- def **restore_rollback_snapshot** `ingest/scripts/merge_delta_collection.py:938`
- def **_run_scoped_temp_collection** `ingest/scripts/merge_delta_collection.py:959`
- def **run_merge_workflow** `ingest/scripts/merge_delta_collection.py:968`
- def **_count** `ingest/scripts/merge_delta_collection.py:1065`
- def **dry_run** `ingest/scripts/merge_delta_collection.py:1069`
- def **main** `ingest/scripts/merge_delta_collection.py:1168`

## ingest/scripts/monitor_server.py
- def **_poll_once** `ingest/scripts/monitor_server.py:60`
- def **poll_loop** `ingest/scripts/monitor_server.py:94`
- def **_computed** `ingest/scripts/monitor_server.py:104`
- class **Handler**(BaseHTTPRequestHandler) `ingest/scripts/monitor_server.py:214`
  - def log_message `ingest/scripts/monitor_server.py:215`
  - def _host_allowed `ingest/scripts/monitor_server.py:218`
  - def do_GET `ingest/scripts/monitor_server.py:228`
- def **main** `ingest/scripts/monitor_server.py:250`

## ingest/scripts/promote_generation.py
imports: ingest/ingest/promotion.py
- def **_parser** `ingest/scripts/promote_generation.py:25`
- def **_backend_factory** `ingest/scripts/promote_generation.py:53`
- def **main** `ingest/scripts/promote_generation.py:65`

## ingest/scripts/prune_artifacts.py
imports: ingest/ingest/artifacts.py
- def **_parser** `ingest/scripts/prune_artifacts.py:29`
- def **main** `ingest/scripts/prune_artifacts.py:69`

## ingest/scripts/publish_snapshot.py
imports: ingest/ingest/config.py, ingest/ingest/remote_search.py
- class **ConditionalManifestActivator**(Protocol) `ingest/scripts/publish_snapshot.py:62`
  - def activate `ingest/scripts/publish_snapshot.py:67`
- class **PublisherLockedError**(RuntimeError) `ingest/scripts/publish_snapshot.py:79`
- def **_create_private_directories** `ingest/scripts/publish_snapshot.py:83`
- def **publisher_lock** `ingest/scripts/publish_snapshot.py:103`
- def **_environment** `ingest/scripts/publish_snapshot.py:123`
- def **_require_remote_approval** `ingest/scripts/publish_snapshot.py:127`
- def **_require_conditional_activator** `ingest/scripts/publish_snapshot.py:135`
- def **_sha256** `ingest/scripts/publish_snapshot.py:151`
- def **create** `ingest/scripts/publish_snapshot.py:171`
- def **_s3** `ingest/scripts/publish_snapshot.py:181`
- def **_reject_duplicate_keys** `ingest/scripts/publish_snapshot.py:215`
- def **_reject_nonstandard_constant** `ingest/scripts/publish_snapshot.py:226`
- def **_load_manifest** `ingest/scripts/publish_snapshot.py:230`
- def **_write_private_json** `ingest/scripts/publish_snapshot.py:259`
- def **_snapshot_metadata** `ingest/scripts/publish_snapshot.py:293`
- def **_remote_snapshot_matches** `ingest/scripts/publish_snapshot.py:303`
- def **_head_remote_snapshot** `ingest/scripts/publish_snapshot.py:313`
- def **_cold_restore_confirmed** `ingest/scripts/publish_snapshot.py:326`
- def **_require_cold_restore_confirmation** `ingest/scripts/publish_snapshot.py:330`
- def **upload** `ingest/scripts/publish_snapshot.py:346`
- def **_upload_locked** `ingest/scripts/publish_snapshot.py:360`
- def **_mark_upload_complete** `ingest/scripts/publish_snapshot.py:545`
- def **_abort_quietly** `ingest/scripts/publish_snapshot.py:558`
- def **_publish_manifest_object** `ingest/scripts/publish_snapshot.py:569`
- def **verify** `ingest/scripts/publish_snapshot.py:595`
- def **_verify_locked** `ingest/scripts/publish_snapshot.py:608`
- def **cleanup** `ingest/scripts/publish_snapshot.py:666`
- def **_cleanup_locked** `ingest/scripts/publish_snapshot.py:682`
- def **main** `ingest/scripts/publish_snapshot.py:721`

## ingest/scripts/reconcile_consolidated.py
imports: ingest/ingest/config.py, ingest/ingest/operational.py, ingest/ingest/qdrant_store.py
- def **resolve_ids_file** `ingest/scripts/reconcile_consolidated.py:59`
- def **sentinel_count** `ingest/scripts/reconcile_consolidated.py:80`
- def **load_listed_ids** `ingest/scripts/reconcile_consolidated.py:91`
- def **ensure_payload_indexes** `ingest/scripts/reconcile_consolidated.py:104`
- def **indexed_matsne_doc_ids** `ingest/scripts/reconcile_consolidated.py:120`
- def **count_consolidated** `ingest/scripts/reconcile_consolidated.py:150`
- def **set_consolidated** `ingest/scripts/reconcile_consolidated.py:163`
- def **main** `ingest/scripts/reconcile_consolidated.py:188`

## ingest/scripts/reembed_export.py
imports: ingest/ingest/config.py, ingest/ingest/qdrant_store.py
- def **main** `ingest/scripts/reembed_export.py:33`

## ingest/scripts/reembed_progress.py
- def **_count** `ingest/scripts/reembed_progress.py:26`
- def **main** `ingest/scripts/reembed_progress.py:38`

## ingest/scripts/reembed_v2.py
imports: ingest/ingest/chunking.py, ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/operational.py, ingest/ingest/qdrant_store.py
- def **embed_text_from_payload** `ingest/scripts/reembed_v2.py:43`
- def **iter_rows** `ingest/scripts/reembed_v2.py:63`
- def **main** `ingest/scripts/reembed_v2.py:71`

## ingest/scripts/rerank_latency_probe.py
imports: ingest/ingest/config.py, ingest/ingest/qdrant_store.py, ingest/ingest/rerank.py
- def **_golden_queries** `ingest/scripts/rerank_latency_probe.py:35`
- def **main** `ingest/scripts/rerank_latency_probe.py:48`

## ingest/scripts/rotating_tee.py
- class **_PrivateRotatingFileHandler**(RotatingFileHandler) `ingest/scripts/rotating_tee.py:19`
  - def _open `ingest/scripts/rotating_tee.py:20`
- def **_prepare_log** `ingest/scripts/rotating_tee.py:34`
- def **mirror_stream** `ingest/scripts/rotating_tee.py:52`
- def **main** `ingest/scripts/rotating_tee.py:85`

## ingest/scripts/runpod_orchestrate.py
imports: ingest/ingest/embed_job.py, ingest/ingest/operational.py
- def **log** `ingest/scripts/runpod_orchestrate.py:74`
- def **_api_key** `ingest/scripts/runpod_orchestrate.py:85`
- def **gql** `ingest/scripts/runpod_orchestrate.py:98`
- def **run** `ingest/scripts/runpod_orchestrate.py:117`
- def **_ssh_base** `ingest/scripts/runpod_orchestrate.py:127`
- def **ssh_ok** `ingest/scripts/runpod_orchestrate.py:134`
- def **ssh_capture** `ingest/scripts/runpod_orchestrate.py:140`
- def **push_content** `ingest/scripts/runpod_orchestrate.py:146`
- def **ensure_pod_tools** `ingest/scripts/runpod_orchestrate.py:156`
- def **_remote_size** `ingest/scripts/runpod_orchestrate.py:171`
- def **push_file** `ingest/scripts/runpod_orchestrate.py:183`
- def **pull_file** `ingest/scripts/runpod_orchestrate.py:202`
- def **step_checksum_ref** `ingest/scripts/runpod_orchestrate.py:220`
- def **step_package** `ingest/scripts/runpod_orchestrate.py:230`
- def **step_keypair** `ingest/scripts/runpod_orchestrate.py:255`
- def **gpu_price** `ingest/scripts/runpod_orchestrate.py:264`
- def **step_provision** `ingest/scripts/runpod_orchestrate.py:275`
- def **step_wait_ssh** `ingest/scripts/runpod_orchestrate.py:313`
- def **step_transfer_in** `ingest/scripts/runpod_orchestrate.py:337`
- def **step_launch** `ingest/scripts/runpod_orchestrate.py:347`
- def **step_poll** `ingest/scripts/runpod_orchestrate.py:373`
- def **step_transfer_out** `ingest/scripts/runpod_orchestrate.py:411`
- def **step_verify_g2** `ingest/scripts/runpod_orchestrate.py:428`
- def **_list_pods** `ingest/scripts/runpod_orchestrate.py:439`
- def **_reap_by_name** `ingest/scripts/runpod_orchestrate.py:450`
- def **terminate** `ingest/scripts/runpod_orchestrate.py:472`
- def **_cleanup** `ingest/scripts/runpod_orchestrate.py:503`
- def **_sig** `ingest/scripts/runpod_orchestrate.py:509`
- def **step_restore** `ingest/scripts/runpod_orchestrate.py:515`
- def **main** `ingest/scripts/runpod_orchestrate.py:530`

## ingest/scripts/runpod_orchestrate_delta.py
imports: ingest/ingest/config.py, ingest/ingest/embed_job.py, ingest/ingest/operational.py, ingest/ingest/qdrant_store.py, ingest/ingest/sources.py
- class **ReserveBudgetError**(RuntimeError) `ingest/scripts/runpod_orchestrate_delta.py:72`
- class **DeltaRun**() `ingest/scripts/runpod_orchestrate_delta.py:77`
  - def input_manifest `ingest/scripts/runpod_orchestrate_delta.py:87`
  - def snapshot `ingest/scripts/runpod_orchestrate_delta.py:91`
- class **SpendGate**() `ingest/scripts/runpod_orchestrate_delta.py:96`
- def **runpod_spend_lock** `ingest/scripts/runpod_orchestrate_delta.py:106`
- def **_safe_component** `ingest/scripts/runpod_orchestrate_delta.py:134`
- def **make_run** `ingest/scripts/runpod_orchestrate_delta.py:142`
- def **_new_run_id** `ingest/scripts/runpod_orchestrate_delta.py:160`
- def **_retry** `ingest/scripts/runpod_orchestrate_delta.py:165`
- def **stage_delta_items** `ingest/scripts/runpod_orchestrate_delta.py:181`
- def **build_input_manifest** `ingest/scripts/runpod_orchestrate_delta.py:223`
- def **estimate_gpu_hours** `ingest/scripts/runpod_orchestrate_delta.py:252`
- def **preflight_local_delta_restore** `ingest/scripts/runpod_orchestrate_delta.py:259`
- def **preflight_pod_delta_write** `ingest/scripts/runpod_orchestrate_delta.py:284`
- def **_account_state** `ingest/scripts/runpod_orchestrate_delta.py:313`
- def **runpod_spend_gate** `ingest/scripts/runpod_orchestrate_delta.py:329`
- def **enforce_reserve_budget** `ingest/scripts/runpod_orchestrate_delta.py:373`
- def **_budget_watchdog_seconds** `ingest/scripts/runpod_orchestrate_delta.py:394`
- def **paid_budget_watchdog** `ingest/scripts/runpod_orchestrate_delta.py:404`
- def **_cleanup_signal_shield** `ingest/scripts/runpod_orchestrate_delta.py:434`
- def **_named_active_pods** `ingest/scripts/runpod_orchestrate_delta.py:447`
- def **_secure_cloud_state** `ingest/scripts/runpod_orchestrate_delta.py:452`
- def **attest_provider_pod** `ingest/scripts/runpod_orchestrate_delta.py:483`
- def **step_provision_4090** `ingest/scripts/runpod_orchestrate_delta.py:520`
- def **attest_single_4090** `ingest/scripts/runpod_orchestrate_delta.py:579`
- def **_pod_gone** `ingest/scripts/runpod_orchestrate_delta.py:594`
- def **terminate_confirmed** `ingest/scripts/runpod_orchestrate_delta.py:605`
- def **_cleanup_delta** `ingest/scripts/runpod_orchestrate_delta.py:629`
- def **_atexit_cleanup** `ingest/scripts/runpod_orchestrate_delta.py:705`
- def **_signal_cleanup** `ingest/scripts/runpod_orchestrate_delta.py:709`
- def **step_package_delta** `ingest/scripts/runpod_orchestrate_delta.py:717`
- def **step_launch_delta** `ingest/scripts/runpod_orchestrate_delta.py:780`
- def **step_poll_delta** `ingest/scripts/runpod_orchestrate_delta.py:826`
- def **_sha256** `ingest/scripts/runpod_orchestrate_delta.py:879`
- def **step_verify_g2_delta** `ingest/scripts/runpod_orchestrate_delta.py:887`
- def **validate_run_manifest** `ingest/scripts/runpod_orchestrate_delta.py:905`
- def **step_transfer_out_delta** `ingest/scripts/runpod_orchestrate_delta.py:972`
- def **validate_restored_records** `ingest/scripts/runpod_orchestrate_delta.py:1033`
- def **step_restore_delta** `ingest/scripts/runpod_orchestrate_delta.py:1080`
- def **_source_and_paths** `ingest/scripts/runpod_orchestrate_delta.py:1158`
- def **_parse_args** `ingest/scripts/runpod_orchestrate_delta.py:1170`
- def **run_paid_workflow** `ingest/scripts/runpod_orchestrate_delta.py:1198`
- def **run_locked_paid_workflow** `ingest/scripts/runpod_orchestrate_delta.py:1323`
- def **main** `ingest/scripts/runpod_orchestrate_delta.py:1349`
- def **emergency_terminate** `ingest/scripts/runpod_orchestrate_delta.py:1440`

## ingest/scripts/runpod_orchestrate_delta_multi.py
imports: ingest/ingest/operational.py
- def **_retry** `ingest/scripts/runpod_orchestrate_delta_multi.py:41`
- def **step_package** `ingest/scripts/runpod_orchestrate_delta_multi.py:52`
- def **step_launch** `ingest/scripts/runpod_orchestrate_delta_multi.py:80`
- def **step_poll** `ingest/scripts/runpod_orchestrate_delta_multi.py:102`
- def **step_pull** `ingest/scripts/runpod_orchestrate_delta_multi.py:129`
- def **step_restore_and_merge** `ingest/scripts/runpod_orchestrate_delta_multi.py:141`
- def **main** `ingest/scripts/runpod_orchestrate_delta_multi.py:163`

## ingest/scripts/runpod_orchestrate_multi.py
imports: ingest/ingest/embed_job.py, ingest/ingest/operational.py
- def **log** `ingest/scripts/runpod_orchestrate_multi.py:65`
- def **gql** `ingest/scripts/runpod_orchestrate_multi.py:75`
- def **run** `ingest/scripts/runpod_orchestrate_multi.py:98`
- def **_ssh** `ingest/scripts/runpod_orchestrate_multi.py:106`
- def **ssh_ok** `ingest/scripts/runpod_orchestrate_multi.py:113`
- def **ssh_cap** `ingest/scripts/runpod_orchestrate_multi.py:118`
- def **push_content** `ingest/scripts/runpod_orchestrate_multi.py:123`
- def **pull_file** `ingest/scripts/runpod_orchestrate_multi.py:130`
- def **ensure_pod_tools** `ingest/scripts/runpod_orchestrate_multi.py:138`
- def **gpu_price** `ingest/scripts/runpod_orchestrate_multi.py:144`
- def **provision** `ingest/scripts/runpod_orchestrate_multi.py:152`
- def **wait_ssh** `ingest/scripts/runpod_orchestrate_multi.py:174`
- def **validated_source_endpoint** `ingest/scripts/runpod_orchestrate_multi.py:193`
- def **wait_corpus_ready** `ingest/scripts/runpod_orchestrate_multi.py:206`
- def **transfer_corpus** `ingest/scripts/runpod_orchestrate_multi.py:218`
- def **push_code** `ingest/scripts/runpod_orchestrate_multi.py:236`
- def **launch** `ingest/scripts/runpod_orchestrate_multi.py:253`
- def **poll** `ingest/scripts/runpod_orchestrate_multi.py:265`
- def **terminate** `ingest/scripts/runpod_orchestrate_multi.py:284`
- def **_cleanup** `ingest/scripts/runpod_orchestrate_multi.py:307`
- def **restore** `ingest/scripts/runpod_orchestrate_multi.py:312`
- def **main** `ingest/scripts/runpod_orchestrate_multi.py:324`

## ingest/scripts/runpod_orchestrate_reembed.py
imports: ingest/ingest/operational.py
- def **_load_refs** `ingest/scripts/runpod_orchestrate_reembed.py:76`
- def **_retry** `ingest/scripts/runpod_orchestrate_reembed.py:90`
- def **_kill_tunnel** `ingest/scripts/runpod_orchestrate_reembed.py:101`
- def **_cleanup** `ingest/scripts/runpod_orchestrate_reembed.py:108`
- def **step_package** `ingest/scripts/runpod_orchestrate_reembed.py:116`
- def **step_provision_cascade** `ingest/scripts/runpod_orchestrate_reembed.py:146`
- def **step_launch** `ingest/scripts/runpod_orchestrate_reembed.py:184`
- def **step_poll** `ingest/scripts/runpod_orchestrate_reembed.py:213`
- def **step_tunnel** `ingest/scripts/runpod_orchestrate_reembed.py:251`
- def **step_start_rerank_server** `ingest/scripts/runpod_orchestrate_reembed.py:274`
- def **_eval_env** `ingest/scripts/runpod_orchestrate_reembed.py:289`
- def **_run_eval** `ingest/scripts/runpod_orchestrate_reembed.py:297`
- def **step_eval** `ingest/scripts/runpod_orchestrate_reembed.py:312`
- def **step_gate** `ingest/scripts/runpod_orchestrate_reembed.py:322`
- def **step_pull_and_restore** `ingest/scripts/runpod_orchestrate_reembed.py:352`
- def **main** `ingest/scripts/runpod_orchestrate_reembed.py:374`

## ingest/scripts/runpod_rerank.py
imports: ingest/ingest/operational.py
- def **_terminate** `ingest/scripts/runpod_rerank.py:41`
- def **_sig** `ingest/scripts/runpod_rerank.py:51`
- def **_dep_check** `ingest/scripts/runpod_rerank.py:57`
- def **setup_pod** `ingest/scripts/runpod_rerank.py:65`
- def **open_tunnel** `ingest/scripts/runpod_rerank.py:108`
- def **up** `ingest/scripts/runpod_rerank.py:123`

## ingest/scripts/runpod_rerank_server.py
- def **_load_runtime** `ingest/scripts/runpod_rerank_server.py:35`
- def **score** `ingest/scripts/runpod_rerank_server.py:57`
- class **Handler**(BaseHTTPRequestHandler) `ingest/scripts/runpod_rerank_server.py:75`
  - def _send `ingest/scripts/runpod_rerank_server.py:76`
  - def do_GET `ingest/scripts/runpod_rerank_server.py:84`
  - def do_POST `ingest/scripts/runpod_rerank_server.py:87`
  - def log_message `ingest/scripts/runpod_rerank_server.py:96`
- def **main** `ingest/scripts/runpod_rerank_server.py:100`

## ingest/scripts/sample_goldset_candidates.py
imports: ingest/eval/__init__.py
- def **matsne_bucket** `ingest/scripts/sample_goldset_candidates.py:65`
- def **spread_sample** `ingest/scripts/sample_goldset_candidates.py:78`
- def **assign_pairs** `ingest/scripts/sample_goldset_candidates.py:103`
- def **main** `ingest/scripts/sample_goldset_candidates.py:164`

## ingest/scripts/scan_generation_candidate.py
imports: ingest/ingest/__init__.py, ingest/ingest/config.py, ingest/ingest/generation_scan.py
- def **_scroll_source** `ingest/scripts/scan_generation_candidate.py:43`
- def **_parser** `ingest/scripts/scan_generation_candidate.py:73`
- def **main** `ingest/scripts/scan_generation_candidate.py:86`

## ingest/scripts/session_monitor.py
- def **_tool_label** `ingest/scripts/session_monitor.py:65`
- def **parse_session** `ingest/scripts/session_monitor.py:88`
- def **sessions_state** `ingest/scripts/session_monitor.py:184`
- def **_ps_rows** `ingest/scripts/session_monitor.py:202`
- def **processes_state** `ingest/scripts/session_monitor.py:224`
- def **_short_name** `ingest/scripts/session_monitor.py:237`
- def **system_state** `ingest/scripts/session_monitor.py:252`
- def **_serverless_health** `ingest/scripts/session_monitor.py:289`
- def **_pod_health** `ingest/scripts/session_monitor.py:321`
- def **rag_state** `ingest/scripts/session_monitor.py:337`
- def **_qdrant_get** `ingest/scripts/session_monitor.py:369`
- def **qdrant_state** `ingest/scripts/session_monitor.py:384`
- def **_tail_line** `ingest/scripts/session_monitor.py:407`
- def **_script_alive** `ingest/scripts/session_monitor.py:419`
- def **coverage_state** `ingest/scripts/session_monitor.py:436`
- def **_log_epoch** `ingest/scripts/session_monitor.py:512`
- def **_env_value** `ingest/scripts/session_monitor.py:519`
- def **_runpod_account** `ingest/scripts/session_monitor.py:531`
- def **_remote_payload_bytes** `ingest/scripts/session_monitor.py:558`
- def **_pod_delta_points** `ingest/scripts/session_monitor.py:582`
- def **_seen_total** `ingest/scripts/session_monitor.py:606`
- def **scrape_state** `ingest/scripts/session_monitor.py:630`
- def **gpu_state** `ingest/scripts/session_monitor.py:666`
- def **_args_alive** `ingest/scripts/session_monitor.py:778`
- def **reembed_state** `ingest/scripts/session_monitor.py:793`
- def **build_state** `ingest/scripts/session_monitor.py:858`
- class **Handler**(BaseHTTPRequestHandler) `ingest/scripts/session_monitor.py:876`
  - def log_message `ingest/scripts/session_monitor.py:877`
  - def _host_allowed `ingest/scripts/session_monitor.py:880`
  - def _send `ingest/scripts/session_monitor.py:894`
  - def do_GET `ingest/scripts/session_monitor.py:905`
- def **main** `ingest/scripts/session_monitor.py:925`

## ingest/scripts/stage_missing_items.py
imports: ingest/ingest/config.py, ingest/ingest/sources.py
- def **main** `ingest/scripts/stage_missing_items.py:27`

## ingest/scripts/supply_chain.py
- class **SupplyChainError**(ValueError) `ingest/scripts/supply_chain.py:56`
- class **LockedRequirement**() `ingest/scripts/supply_chain.py:61`
- class **RequirementsLock**() `ingest/scripts/supply_chain.py:68`
- def **_normalise_name** `ingest/scripts/supply_chain.py:74`
- def **_require_sha256** `ingest/scripts/supply_chain.py:78`
- def **sha256_file** `ingest/scripts/supply_chain.py:84`
- def **validate_base_image_reference** `ingest/scripts/supply_chain.py:95`
- def **qdrant_archive_url** `ingest/scripts/supply_chain.py:112`
- def **validate_qdrant_release** `ingest/scripts/supply_chain.py:121`
- def **qdrant_checksum_from_evidence** `ingest/scripts/supply_chain.py:149`
- def **verify_file_sha256** `ingest/scripts/supply_chain.py:190`
- def **_logical_lock_lines** `ingest/scripts/supply_chain.py:200`
- def **parse_requirements_lock** `ingest/scripts/supply_chain.py:222`
- def **load_runtime_identity** `ingest/scripts/supply_chain.py:275`
- def **validate_runtime_identity** `ingest/scripts/supply_chain.py:285`
- def **validate_build_inputs** `ingest/scripts/supply_chain.py:367`
- def **verify_installed_runtime** `ingest/scripts/supply_chain.py:432`
- def **_run_command** `ingest/scripts/supply_chain.py:495`
- def **_atomic_write** `ingest/scripts/supply_chain.py:539`
- def **_atomic_json** `ingest/scripts/supply_chain.py:553`
- def **_severity_counts** `ingest/scripts/supply_chain.py:558`
- def **_local_image_identity** `ingest/scripts/supply_chain.py:570`
- def **emit_release_audit** `ingest/scripts/supply_chain.py:591`
- def **_parser** `ingest/scripts/supply_chain.py:683`
- def **main** `ingest/scripts/supply_chain.py:709`

## ingest/scripts/validate_golden_batch.py
imports: ingest/eval/__init__.py, ingest/ingest/config.py, ingest/ingest/embedding.py, ingest/ingest/qdrant_store.py, ingest/ingest/search.py
- def **_nfc** `ingest/scripts/validate_golden_batch.py:70`
- def **tokens** `ingest/scripts/validate_golden_batch.py:74`
- def **jaccard** `ingest/scripts/validate_golden_batch.py:81`
- def **citation_tokens** `ingest/scripts/validate_golden_batch.py:87`
- def **check_schema** `ingest/scripts/validate_golden_batch.py:97`
- def **check_reground** `ingest/scripts/validate_golden_batch.py:127`
- def **check_span_coverage** `ingest/scripts/validate_golden_batch.py:137`
- def **check_near_dup** `ingest/scripts/validate_golden_batch.py:147`
- def **check_paraphrase** `ingest/scripts/validate_golden_batch.py:161`
- def **_doc_metadata** `ingest/scripts/validate_golden_batch.py:174`
- def **check_citations** `ingest/scripts/validate_golden_batch.py:195`
- def **check_holdout** `ingest/scripts/validate_golden_batch.py:234`
- def **fetch_live_payloads** `ingest/scripts/validate_golden_batch.py:249`
- def **main** `ingest/scripts/validate_golden_batch.py:266`

## ingest/scripts/validate_supremecourt_partial.py
- class **ValidationError**(RuntimeError) `ingest/scripts/validate_supremecourt_partial.py:61`
  - def __init__ `ingest/scripts/validate_supremecourt_partial.py:64`
- class **_Record**() `ingest/scripts/validate_supremecourt_partial.py:72`
- def **_reject_json_constant** `ingest/scripts/validate_supremecourt_partial.py:81`
- def **_object_without_duplicate_keys** `ingest/scripts/validate_supremecourt_partial.py:85`
- def **_loads** `ingest/scripts/validate_supremecourt_partial.py:94`
- def **_open_private_regular** `ingest/scripts/validate_supremecourt_partial.py:106`
- def **_read_private_regular** `ingest/scripts/validate_supremecourt_partial.py:132`
- def **_iso_date** `ingest/scripts/validate_supremecourt_partial.py:137`
- def **_iso_datetime** `ingest/scripts/validate_supremecourt_partial.py:152`
- def **_canonical_fingerprint** `ingest/scripts/validate_supremecourt_partial.py:167`
- def **_is_explicitly_incomplete** `ingest/scripts/validate_supremecourt_partial.py:178`
- def **_record_from_item** `ingest/scripts/validate_supremecourt_partial.py:188`
- def **_read_jsonl** `ingest/scripts/validate_supremecourt_partial.py:275`
- def **_require_unique** `ingest/scripts/validate_supremecourt_partial.py:301`
- def **_manifest_int** `ingest/scripts/validate_supremecourt_partial.py:309`
- def **_resume_start_state** `ingest/scripts/validate_supremecourt_partial.py:319`
- def **_validate_completed_windows** `ingest/scripts/validate_supremecourt_partial.py:481`
- def **_absolute_lexical** `ingest/scripts/validate_supremecourt_partial.py:701`
- def **validate_run** `ingest/scripts/validate_supremecourt_partial.py:705`
- def **parse_args** `ingest/scripts/validate_supremecourt_partial.py:1100`
- def **main** `ingest/scripts/validate_supremecourt_partial.py:1111`

## ingest/scripts/verify_all_embedded.py
imports: ingest/ingest/config.py, ingest/ingest/qdrant_store.py, ingest/ingest/sources.py
- def **quarantined_ids** `ingest/scripts/verify_all_embedded.py:39`
- def **scraped_universe** `ingest/scripts/verify_all_embedded.py:61`
- def **no_text_ids** `ingest/scripts/verify_all_embedded.py:111`
- def **embedded_universe** `ingest/scripts/verify_all_embedded.py:150`
- def **main** `ingest/scripts/verify_all_embedded.py:179`

## ingest/scripts/verify_delta_embedded.py
imports: ingest/ingest/config.py, ingest/ingest/qdrant_store.py
- def **delta_doc_ids** `ingest/scripts/verify_delta_embedded.py:28`
- def **main** `ingest/scripts/verify_delta_embedded.py:51`

## ingest/scripts/verify_generation.py
imports: ingest/ingest/collection_compatibility.py, ingest/ingest/config.py, ingest/ingest/generation.py, ingest/ingest/integrity.py, ingest/ingest/qdrant_store.py
- def **sibling_report_path** `ingest/scripts/verify_generation.py:40`
- def **stream_collection_points** `ingest/scripts/verify_generation.py:48`
- def **_append_issues** `ingest/scripts/verify_generation.py:89`
- def **_merge_compatibility** `ingest/scripts/verify_generation.py:114`
- def **verify_loaded_generation** `ingest/scripts/verify_generation.py:141`
- def **verify_generation_directory** `ingest/scripts/verify_generation.py:171`
- def **_parser** `ingest/scripts/verify_generation.py:193`
- def **main** `ingest/scripts/verify_generation.py:210`

## ingest/scripts/verify_matsne_completeness.py
imports: ingest/ingest/config.py
- def **parse_last_page** `ingest/scripts/verify_matsne_completeness.py:60`
- def **expected_total** `ingest/scripts/verify_matsne_completeness.py:66`
- def **is_absent_page** `ingest/scripts/verify_matsne_completeness.py:73`
- def **residual_ids** `ingest/scripts/verify_matsne_completeness.py:81`
- def **fetch** `ingest/scripts/verify_matsne_completeness.py:89`
- def **load_seen_ids** `ingest/scripts/verify_matsne_completeness.py:113`
- def **referenced_ids** `ingest/scripts/verify_matsne_completeness.py:124`
- def **audit_advertised** `ingest/scripts/verify_matsne_completeness.py:145`
- def **audit_reference_closure** `ingest/scripts/verify_matsne_completeness.py:171`
- def **audit_id_enum** `ingest/scripts/verify_matsne_completeness.py:185`
- def **main** `ingest/scripts/verify_matsne_completeness.py:208`

## ingest/serverless/handler.py
imports: ingest/ingest/__init__.py, ingest/ingest/config.py, ingest/ingest/qdrant_store.py
- def **_configure_runtime_environment** `ingest/serverless/handler.py:48`
- def **_accept_restore** `ingest/serverless/handler.py:122`
- def **_boot** `ingest/serverless/handler.py:144`
- def **_worker_readiness** `ingest/serverless/handler.py:170`
- def **_sysinfo** `ingest/serverless/handler.py:192`
- def **_warmup** `ingest/serverless/handler.py:223`
- def **handler** `ingest/serverless/handler.py:237`
- def **main** `ingest/serverless/handler.py:332`

## ingest/serverless/qdrant_boot.py
- def **storage_dir** `ingest/serverless/qdrant_boot.py:95`
- def **publish_dir** `ingest/serverless/qdrant_boot.py:99`
- def **_ensure_container_disk_capacity** `ingest/serverless/qdrant_boot.py:103`
- def **_http** `ingest/serverless/qdrant_boot.py:116`
- def **_healthy** `ingest/serverless/qdrant_boot.py:142`
- def **_log_tail** `ingest/serverless/qdrant_boot.py:149`
- def **_spawn** `ingest/serverless/qdrant_boot.py:156`
- def **ensure_running** `ingest/serverless/qdrant_boot.py:185`
- def **_read_json** `ingest/serverless/qdrant_boot.py:225`
- def **_sha256** `ingest/serverless/qdrant_boot.py:232`
- def **validate_publish_manifest** `ingest/serverless/qdrant_boot.py:243`
- def **_restore_identity** `ingest/serverless/qdrant_boot.py:344`
- def **needs_restore** `ingest/serverless/qdrant_boot.py:352`
- def **restore_pending** `ingest/serverless/qdrant_boot.py:365`
- def **_collection_points** `ingest/serverless/qdrant_boot.py:374`
- def **_distance** `ingest/serverless/qdrant_boot.py:383`
- def **_collection_compatibility** `ingest/serverless/qdrant_boot.py:387`
- def **_write_active** `ingest/serverless/qdrant_boot.py:480`
- def **restore_allows_serving** `ingest/serverless/qdrant_boot.py:503`
- def **verified_runtime_manifest** `ingest/serverless/qdrant_boot.py:534`
- def **runtime_manifest_identity** `ingest/serverless/qdrant_boot.py:587`
- def **runtime_readiness** `ingest/serverless/qdrant_boot.py:595`
- def **maybe_restore** `ingest/serverless/qdrant_boot.py:634`

## scraper/legal_scrapers/__init__.py

## scraper/legal_scrapers/extensions.py
- class **_PrivateRotatingFileHandler**(RotatingFileHandler) `scraper/legal_scrapers/extensions.py:65`
  - def _open `scraper/legal_scrapers/extensions.py:66`
- class **_OnlySpider**(logging.Filter) `scraper/legal_scrapers/extensions.py:80`
  - def __init__ `scraper/legal_scrapers/extensions.py:81`
  - def filter `scraper/legal_scrapers/extensions.py:85`
- class **RotatingSpiderLogExtension**() `scraper/legal_scrapers/extensions.py:93`
  - def __init__ `scraper/legal_scrapers/extensions.py:96`
  - def from_crawler `scraper/legal_scrapers/extensions.py:101`
  - def spider_opened `scraper/legal_scrapers/extensions.py:107`
  - def spider_closed `scraper/legal_scrapers/extensions.py:135`
- class **DurableDedupCommitExtension**() `scraper/legal_scrapers/extensions.py:143`
  - def __init__ `scraper/legal_scrapers/extensions.py:146`
  - def from_crawler `scraper/legal_scrapers/extensions.py:150`
  - def _local_feed_path `scraper/legal_scrapers/extensions.py:158`
  - def _fsync_local_feed `scraper/legal_scrapers/extensions.py:168`
  - def feed_exporter_closed `scraper/legal_scrapers/extensions.py:183`
- def **extract_title** `scraper/legal_scrapers/extensions.py:241`
- def **_truncate** `scraper/legal_scrapers/extensions.py:259`
- class **ProgressSnapshot**() `scraper/legal_scrapers/extensions.py:267`
- def **_format_elapsed** `scraper/legal_scrapers/extensions.py:285`
- def **_items_per_min** `scraper/legal_scrapers/extensions.py:294`
- def **_format_status_counts** `scraper/legal_scrapers/extensions.py:302`
- def **_status_glyph** `scraper/legal_scrapers/extensions.py:320`
- def **render_panel** `scraper/legal_scrapers/extensions.py:331`
- def **render_table** `scraper/legal_scrapers/extensions.py:379`
- class **_ProgressDashboard**() `scraper/legal_scrapers/extensions.py:442`
  - def __init__ `scraper/legal_scrapers/extensions.py:451`
  - def reset `scraper/legal_scrapers/extensions.py:454`
  - def declare `scraper/legal_scrapers/extensions.py:466`
  - def finish `scraper/legal_scrapers/extensions.py:472`
  - def register `scraper/legal_scrapers/extensions.py:504`
  - def on_spider_closed `scraper/legal_scrapers/extensions.py:508`
  - def _multi `scraper/legal_scrapers/extensions.py:516`
  - def _ensure_started `scraper/legal_scrapers/extensions.py:519`
  - def _render `scraper/legal_scrapers/extensions.py:535`
  - def _tick `scraper/legal_scrapers/extensions.py:564`
- def **declare_spiders** `scraper/legal_scrapers/extensions.py:578`
- def **finish_dashboard** `scraper/legal_scrapers/extensions.py:583`
- class **LiveProgressExtension**() `scraper/legal_scrapers/extensions.py:588`
  - def __init__ `scraper/legal_scrapers/extensions.py:591`
  - def from_crawler `scraper/legal_scrapers/extensions.py:600`
  - def spider_opened `scraper/legal_scrapers/extensions.py:616`
  - def item_scraped `scraper/legal_scrapers/extensions.py:621`
  - def spider_closed `scraper/legal_scrapers/extensions.py:626`
  - def current_snapshot `scraper/legal_scrapers/extensions.py:632`
  - def _snapshot `scraper/legal_scrapers/extensions.py:637`
  - def _total_items `scraper/legal_scrapers/extensions.py:668`

## scraper/legal_scrapers/items.py
imports: scraper/legal_scrapers/utils/markdown.py
- def **class_to_status** `scraper/legal_scrapers/items.py:12`
- class **MatsneItem**(scrapy.Item) `scraper/legal_scrapers/items.py:22`
- def **_strip** `scraper/legal_scrapers/items.py:61`
- def **_f** `scraper/legal_scrapers/items.py:70`
- def **_body** `scraper/legal_scrapers/items.py:75`
- def **_list** `scraper/legal_scrapers/items.py:81`
- class **EcdItem**(scrapy.Item) `scraper/legal_scrapers/items.py:87`
- class **ConstcourtItem**(scrapy.Item) `scraper/legal_scrapers/items.py:106`
- class **NaprItem**(scrapy.Item) `scraper/legal_scrapers/items.py:126`
- class **TbappealItem**(scrapy.Item) `scraper/legal_scrapers/items.py:147`
- class **SupremecourtItem**(scrapy.Item) `scraper/legal_scrapers/items.py:164`
- class **TasItem**(scrapy.Item) `scraper/legal_scrapers/items.py:179`

## scraper/legal_scrapers/middlewares.py
*(unparseable: multiple exception types must be parenthesized at line 35)*

## scraper/legal_scrapers/pipelines.py
- class **MatsnePipeline**() `scraper/legal_scrapers/pipelines.py:9`
  - def process_item `scraper/legal_scrapers/pipelines.py:10`
- class **DedupPipeline**() `scraper/legal_scrapers/pipelines.py:14`
  - def process_item `scraper/legal_scrapers/pipelines.py:25`
- class **SupremecourtDurablePipeline**() `scraper/legal_scrapers/pipelines.py:50`
  - def process_item `scraper/legal_scrapers/pipelines.py:64`

## scraper/legal_scrapers/run.py
imports: scraper/legal_scrapers/extensions.py
- def **_bootstrap_project_dir** `scraper/legal_scrapers/run.py:35`
- def **parse_args** `scraper/legal_scrapers/run.py:50`
- def **_positive_int** `scraper/legal_scrapers/run.py:84`
- def **select_spiders** `scraper/legal_scrapers/run.py:91`
- def **crawl_quality_issues** `scraper/legal_scrapers/run.py:110`
- def **main** `scraper/legal_scrapers/run.py:137`

## scraper/legal_scrapers/settings.py

## scraper/legal_scrapers/spiders/__init__.py

## scraper/legal_scrapers/spiders/base.py
- class **BaseLegalSpider**(scrapy.Spider) `scraper/legal_scrapers/spiders/base.py:20`
  - def __init__ `scraper/legal_scrapers/spiders/base.py:49`
  - def from_crawler `scraper/legal_scrapers/spiders/base.py:66`
  - def request_failed `scraper/legal_scrapers/spiders/base.py:71`
  - def record_quality_failure `scraper/legal_scrapers/spiders/base.py:90`
  - def parse_date_arg `scraper/legal_scrapers/spiders/base.py:116`
  - def configure_run_outputs `scraper/legal_scrapers/spiders/base.py:127`
  - def open_dedup_store `scraper/legal_scrapers/spiders/base.py:157`
  - def _dedup_now `scraper/legal_scrapers/spiders/base.py:213`
  - def _parse_dedup_timestamp `scraper/legal_scrapers/spiders/base.py:217`
  - def _migrate_seen_schema `scraper/legal_scrapers/spiders/base.py:228`
  - def _load_seen_state `scraper/legal_scrapers/spiders/base.py:260`
  - def iter_refresh_keys `scraper/legal_scrapers/spiders/base.py:279`
  - def dedup_refresh_context `scraper/legal_scrapers/spiders/base.py:283`
  - def dedup_seen_count `scraper/legal_scrapers/spiders/base.py:303`
  - def dedup_key `scraper/legal_scrapers/spiders/base.py:311`
  - def is_seen `scraper/legal_scrapers/spiders/base.py:327`
  - def _explicitly_incomplete `scraper/legal_scrapers/spiders/base.py:341`
  - def _is_pending_material `scraper/legal_scrapers/spiders/base.py:350`
  - def _refresh_days `scraper/legal_scrapers/spiders/base.py:365`
  - def _dedup_record `scraper/legal_scrapers/spiders/base.py:372`
  - def stage_seen `scraper/legal_scrapers/spiders/base.py:427`
  - def _persist_dedup_records `scraper/legal_scrapers/spiders/base.py:445`
  - def commit_staged_seen `scraper/legal_scrapers/spiders/base.py:497`
  - def discard_staged_seen `scraper/legal_scrapers/spiders/base.py:505`
  - def mark_seen `scraper/legal_scrapers/spiders/base.py:511`
  - def build_run_id `scraper/legal_scrapers/spiders/base.py:526`
  - def feed_options `scraper/legal_scrapers/spiders/base.py:540`
  - def write_run_metadata `scraper/legal_scrapers/spiders/base.py:550`

## scraper/legal_scrapers/spiders/constcourt_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/documents.py, scraper/legal_scrapers/utils/markdown.py
- class **ConstcourtSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/constcourt_spider.py:48`
  - def start `scraper/legal_scrapers/spiders/constcourt_spider.py:53`
  - def request_page `scraper/legal_scrapers/spiders/constcourt_spider.py:56`
  - def parse_list `scraper/legal_scrapers/spiders/constcourt_spider.py:78`
  - def parse_detail `scraper/legal_scrapers/spiders/constcourt_spider.py:121`
  - def parse_docx_body `scraper/legal_scrapers/spiders/constcourt_spider.py:160`
  - def load_item `scraper/legal_scrapers/spiders/constcourt_spider.py:192`

## scraper/legal_scrapers/spiders/ecd_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/json_api.py, scraper/legal_scrapers/utils/pagination.py, scraper/legal_scrapers/utils/text.py
- class **EcdSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/ecd_spider.py:38`
  - def start `scraper/legal_scrapers/spiders/ecd_spider.py:44`
  - def pagination_request_failed `scraper/legal_scrapers/spiders/ecd_spider.py:56`
  - def _json_data `scraper/legal_scrapers/spiders/ecd_spider.py:59`
  - def parse_instances `scraper/legal_scrapers/spiders/ecd_spider.py:75`
  - def request_page `scraper/legal_scrapers/spiders/ecd_spider.py:117`
  - def parse_list `scraper/legal_scrapers/spiders/ecd_spider.py:142`
  - def parse_detail `scraper/legal_scrapers/spiders/ecd_spider.py:248`

## scraper/legal_scrapers/spiders/matsne_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/pagination.py, scraper/legal_scrapers/utils/search_urls.py
- def **status_from_effective_dates** `scraper/legal_scrapers/spiders/matsne_spider.py:32`
- class **MatsneSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/matsne_spider.py:49`
  - def from_crawler `scraper/legal_scrapers/spiders/matsne_spider.py:68`
  - def _doc_type `scraper/legal_scrapers/spiders/matsne_spider.py:74`
  - def pagination_scope `scraper/legal_scrapers/spiders/matsne_spider.py:98`
  - def _page_number `scraper/legal_scrapers/spiders/matsne_spider.py:114`
  - def _advertised_last_page `scraper/legal_scrapers/spiders/matsne_spider.py:127`
  - def _looks_waf_blocked `scraper/legal_scrapers/spiders/matsne_spider.py:147`
  - def _observe_page_number `scraper/legal_scrapers/spiders/matsne_spider.py:154`
  - def _listing_request_options `scraper/legal_scrapers/spiders/matsne_spider.py:174`
  - def pagination_request_failed `scraper/legal_scrapers/spiders/matsne_spider.py:193`
  - def _supersede_pagination_scope `scraper/legal_scrapers/spiders/matsne_spider.py:196`
  - def _record_main_listed_id `scraper/legal_scrapers/spiders/matsne_spider.py:231`
  - def start `scraper/legal_scrapers/spiders/matsne_spider.py:251`
  - def _load_seed_urls `scraper/legal_scrapers/spiders/matsne_spider.py:344`
  - def spider_closed `scraper/legal_scrapers/spiders/matsne_spider.py:374`
  - def spider_idle `scraper/legal_scrapers/spiders/matsne_spider.py:409`
  - def start_phase `scraper/legal_scrapers/spiders/matsne_spider.py:428`
  - def build_request `scraper/legal_scrapers/spiders/matsne_spider.py:436`
  - def follow_request `scraper/legal_scrapers/spiders/matsne_spider.py:451`
  - def parse `scraper/legal_scrapers/spiders/matsne_spider.py:469`
  - def _split_requests `scraper/legal_scrapers/spiders/matsne_spider.py:644`
  - def parse_document `scraper/legal_scrapers/spiders/matsne_spider.py:691`

## scraper/legal_scrapers/spiders/napr_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/documents.py, scraper/legal_scrapers/utils/json_api.py, scraper/legal_scrapers/utils/pagination.py
- def **decision_type_from_title** `scraper/legal_scrapers/spiders/napr_spider.py:72`
- class **NaprSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/napr_spider.py:82`
  - def from_crawler `scraper/legal_scrapers/spiders/napr_spider.py:89`
  - def start `scraper/legal_scrapers/spiders/napr_spider.py:94`
  - def spider_idle `scraper/legal_scrapers/spiders/napr_spider.py:99`
  - def pagination_scope `scraper/legal_scrapers/spiders/napr_spider.py:108`
  - def pagination_request_failed `scraper/legal_scrapers/spiders/napr_spider.py:115`
  - def request_page `scraper/legal_scrapers/spiders/napr_spider.py:118`
  - def parse_list `scraper/legal_scrapers/spiders/napr_spider.py:144`
  - def _parse_listing_record `scraper/legal_scrapers/spiders/napr_spider.py:258`
  - def parse_pdf `scraper/legal_scrapers/spiders/napr_spider.py:301`
  - def load_item `scraper/legal_scrapers/spiders/napr_spider.py:333`

## scraper/legal_scrapers/spiders/supremecourt_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/markdown.py
- class **DateWindow**() `scraper/legal_scrapers/spiders/supremecourt_spider.py:74`
  - def days `scraper/legal_scrapers/spiders/supremecourt_spider.py:96`
- class **NewestFirstPlanner**() `scraper/legal_scrapers/spiders/supremecourt_spider.py:100`
  - def __init__ `scraper/legal_scrapers/spiders/supremecourt_spider.py:103`
  - def _new_window `scraper/legal_scrapers/spiders/supremecourt_spider.py:111`
  - def seed `scraper/legal_scrapers/spiders/supremecourt_spider.py:140`
  - def seed_cursors `scraper/legal_scrapers/spiders/supremecourt_spider.py:143`
  - def peek_end `scraper/legal_scrapers/spiders/supremecourt_spider.py:155`
  - def pop_at_end `scraper/legal_scrapers/spiders/supremecourt_spider.py:160`
  - def split `scraper/legal_scrapers/spiders/supremecourt_spider.py:166`
  - def adapted_days `scraper/legal_scrapers/spiders/supremecourt_spider.py:189`
  - def schedule_older `scraper/legal_scrapers/spiders/supremecourt_spider.py:200`
- def **parse_authoritative_total** `scraper/legal_scrapers/spiders/supremecourt_spider.py:211`
- def **_atomic_write** `scraper/legal_scrapers/spiders/supremecourt_spider.py:219`
- class **SupremecourtSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/supremecourt_spider.py:238`
  - def __init__ `scraper/legal_scrapers/spiders/supremecourt_spider.py:260`
  - def configure_run_outputs `scraper/legal_scrapers/spiders/supremecourt_spider.py:292`
  - def start `scraper/legal_scrapers/spiders/supremecourt_spider.py:310`
  - def _partial_validator `scraper/legal_scrapers/spiders/supremecourt_spider.py:325`
  - def _discover_resume_state `scraper/legal_scrapers/spiders/supremecourt_spider.py:347`
  - def request_page `scraper/legal_scrapers/spiders/supremecourt_spider.py:443`
  - def request_window `scraper/legal_scrapers/spiders/supremecourt_spider.py:465`
  - def _advance_frontier `scraper/legal_scrapers/spiders/supremecourt_spider.py:488`
  - def parse_list `scraper/legal_scrapers/spiders/supremecourt_spider.py:596`
  - def _parse_case_card `scraper/legal_scrapers/spiders/supremecourt_spider.py:706`
  - def _parse_legacy_list `scraper/legal_scrapers/spiders/supremecourt_spider.py:736`
  - def _parse_retry `scraper/legal_scrapers/spiders/supremecourt_spider.py:757`
  - def _mark_terminal_ready `scraper/legal_scrapers/spiders/supremecourt_spider.py:767`
  - def parse_detail `scraper/legal_scrapers/spiders/supremecourt_spider.py:772`
  - def persist_item `scraper/legal_scrapers/spiders/supremecourt_spider.py:815`
  - def item_persisted `scraper/legal_scrapers/spiders/supremecourt_spider.py:839`
  - def item_failed `scraper/legal_scrapers/spiders/supremecourt_spider.py:848`
  - def _maybe_finish_detail_phase `scraper/legal_scrapers/spiders/supremecourt_spider.py:857`
  - def _settle_ready_windows `scraper/legal_scrapers/spiders/supremecourt_spider.py:868`
  - def _enqueue `scraper/legal_scrapers/spiders/supremecourt_spider.py:897`
  - def request_failed `scraper/legal_scrapers/spiders/supremecourt_spider.py:908`
  - def _window_failure `scraper/legal_scrapers/spiders/supremecourt_spider.py:929`
  - def _identity `scraper/legal_scrapers/spiders/supremecourt_spider.py:954`
  - def _collect_existing_items `scraper/legal_scrapers/spiders/supremecourt_spider.py:961`
  - def _reconcile_seen_store `scraper/legal_scrapers/spiders/supremecourt_spider.py:985`
  - def _contiguous_cursor `scraper/legal_scrapers/spiders/supremecourt_spider.py:1015`
  - def _sorted_cumulative `scraper/legal_scrapers/spiders/supremecourt_spider.py:1032`
  - def _materialize_items `scraper/legal_scrapers/spiders/supremecourt_spider.py:1043`
  - def _manifest `scraper/legal_scrapers/spiders/supremecourt_spider.py:1054`
  - def _write_manifest `scraper/legal_scrapers/spiders/supremecourt_spider.py:1171`
  - def closed `scraper/legal_scrapers/spiders/supremecourt_spider.py:1177`

## scraper/legal_scrapers/spiders/tas_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/markdown.py, scraper/legal_scrapers/utils/pagination.py
- def **_xml_tag** `scraper/legal_scrapers/spiders/tas_spider.py:157`
- def **_strip** `scraper/legal_scrapers/spiders/tas_spider.py:162`
- def **_clean_value** `scraper/legal_scrapers/spiders/tas_spider.py:166`
- def **_local_date** `scraper/legal_scrapers/spiders/tas_spider.py:173`
- def **_slash** `scraper/legal_scrapers/spiders/tas_spider.py:189`
- def **_field_labels** `scraper/legal_scrapers/spiders/tas_spider.py:199`
- def **_form_fields** `scraper/legal_scrapers/spiders/tas_spider.py:210`
- def **_request_text** `scraper/legal_scrapers/spiders/tas_spider.py:230`
- def **_parcels** `scraper/legal_scrapers/spiders/tas_spider.py:246`
- def **_primary_parcel** `scraper/legal_scrapers/spiders/tas_spider.py:264`
- def **_responses** `scraper/legal_scrapers/spiders/tas_spider.py:272`
- def **_response_to_markdown** `scraper/legal_scrapers/spiders/tas_spider.py:287`
- def **_nomenclature_full** `scraper/legal_scrapers/spiders/tas_spider.py:299`
- def **_full_name** `scraper/legal_scrapers/spiders/tas_spider.py:308`
- class **TasSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/tas_spider.py:315`
  - def start `scraper/legal_scrapers/spiders/tas_spider.py:341`
  - def pagination_scope `scraper/legal_scrapers/spiders/tas_spider.py:356`
  - def pagination_request_failed `scraper/legal_scrapers/spiders/tas_spider.py:362`
  - def parse_docs `scraper/legal_scrapers/spiders/tas_spider.py:365`
  - def _fetch_detail `scraper/legal_scrapers/spiders/tas_spider.py:517`
  - def build_item `scraper/legal_scrapers/spiders/tas_spider.py:547`
  - def _enrich `scraper/legal_scrapers/spiders/tas_spider.py:586`
  - def _list_body `scraper/legal_scrapers/spiders/tas_spider.py:691`
  - def _detail_body `scraper/legal_scrapers/spiders/tas_spider.py:702`

## scraper/legal_scrapers/spiders/tbappeal_spider.py
imports: scraper/legal_scrapers/items.py, scraper/legal_scrapers/spiders/base.py, scraper/legal_scrapers/utils/dates.py, scraper/legal_scrapers/utils/documents.py, scraper/legal_scrapers/utils/markdown.py, scraper/legal_scrapers/utils/pagination.py
- class **TbappealSpider**(BaseLegalSpider) `scraper/legal_scrapers/spiders/tbappeal_spider.py:36`
  - def start `scraper/legal_scrapers/spiders/tbappeal_spider.py:40`
  - def pagination_scope `scraper/legal_scrapers/spiders/tbappeal_spider.py:45`
  - def pagination_request_failed `scraper/legal_scrapers/spiders/tbappeal_spider.py:48`
  - def request_page `scraper/legal_scrapers/spiders/tbappeal_spider.py:51`
  - def _advertised_pages `scraper/legal_scrapers/spiders/tbappeal_spider.py:66`
  - def parse_list `scraper/legal_scrapers/spiders/tbappeal_spider.py:86`
  - def _in_window `scraper/legal_scrapers/spiders/tbappeal_spider.py:178`
  - def parse_detail `scraper/legal_scrapers/spiders/tbappeal_spider.py:184`
  - def parse_pdf `scraper/legal_scrapers/spiders/tbappeal_spider.py:223`
  - def _summary_fallback `scraper/legal_scrapers/spiders/tbappeal_spider.py:261`
  - def load_item `scraper/legal_scrapers/spiders/tbappeal_spider.py:273`

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
- class **ExtractionStatus**(StrEnum) `scraper/legal_scrapers/utils/documents.py:32`
- class **ExtractionLimits**() `scraper/legal_scrapers/utils/documents.py:41`
- class **ExtractionResult**() `scraper/legal_scrapers/utils/documents.py:51`
- def **_mime** `scraper/legal_scrapers/utils/documents.py:63`
- def **_malformed** `scraper/legal_scrapers/utils/documents.py:71`
- def **_validate_input** `scraper/legal_scrapers/utils/documents.py:80`
- def **_text_result** `scraper/legal_scrapers/utils/documents.py:127`
- def **_extract_pdf_payload** `scraper/legal_scrapers/utils/documents.py:155`
- def **_extract_docx_payload** `scraper/legal_scrapers/utils/documents.py:177`
- def **_apply_process_limits** `scraper/legal_scrapers/utils/documents.py:188`
- def **_extraction_worker** `scraper/legal_scrapers/utils/documents.py:197`
- def **_run_isolated_extraction** `scraper/legal_scrapers/utils/documents.py:219`
- def **pdf_to_markdown** `scraper/legal_scrapers/utils/documents.py:271`
- def **_guard_docx** `scraper/legal_scrapers/utils/documents.py:286`
- def **docx_to_markdown** `scraper/legal_scrapers/utils/documents.py:303`

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

## scraper/legal_scrapers/utils/pagination.py
- def **advertised_page_count** `scraper/legal_scrapers/utils/pagination.py:28`
- def **parse_advertised_count** `scraper/legal_scrapers/utils/pagination.py:37`
- class **PaginationOutcome**() `scraper/legal_scrapers/utils/pagination.py:60`
  - def to_dict `scraper/legal_scrapers/utils/pagination.py:79`
- class **PaginationReconciler**() `scraper/legal_scrapers/utils/pagination.py:101`
  - def __init__ `scraper/legal_scrapers/utils/pagination.py:104`
  - def advertised_pages_max `scraper/legal_scrapers/utils/pagination.py:154`
  - def _cursor `scraper/legal_scrapers/utils/pagination.py:158`
  - def _identifier `scraper/legal_scrapers/utils/pagination.py:163`
  - def _add_failure `scraper/legal_scrapers/utils/pagination.py:171`
  - def _advertised_value `scraper/legal_scrapers/utils/pagination.py:193`
  - def observe_page `scraper/legal_scrapers/utils/pagination.py:201`
  - def mark_failure `scraper/legal_scrapers/utils/pagination.py:301`
  - def mark_cap `scraper/legal_scrapers/utils/pagination.py:308`
  - def finalize `scraper/legal_scrapers/utils/pagination.py:315`
- def **get_pagination_reconciler** `scraper/legal_scrapers/utils/pagination.py:386`
- def **_append_private_jsonl** `scraper/legal_scrapers/utils/pagination.py:404`
- def **finalize_pagination_scope** `scraper/legal_scrapers/utils/pagination.py:425`
- def **handle_pagination_request_failure** `scraper/legal_scrapers/utils/pagination.py:479`

## scraper/legal_scrapers/utils/search_urls.py
- def **first_qs_value** `scraper/legal_scrapers/utils/search_urls.py:46`
- def **sub_windows** `scraper/legal_scrapers/utils/search_urls.py:51`
- def **build_search_url** `scraper/legal_scrapers/utils/search_urls.py:76`
- def **generate_start_url_batches** `scraper/legal_scrapers/utils/search_urls.py:91`
- def **generate_start_urls** `scraper/legal_scrapers/utils/search_urls.py:116`

## scraper/legal_scrapers/utils/text.py
- def **plain_text_to_markdown** `scraper/legal_scrapers/utils/text.py:11`

## scraper/legal_scrapers/utils/user_agents.py
- def **generate_random_user_agent** `scraper/legal_scrapers/utils/user_agents.py:18`

## run_all.py
- def **selected_ingest_sources** `run_all.py:34`
- def **build_watch_command** `run_all.py:49`
- def **_pump** `run_all.py:64`
- def **main** `run_all.py:73`
