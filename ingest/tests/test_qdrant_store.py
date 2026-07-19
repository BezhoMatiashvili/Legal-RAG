import dataclasses
import hashlib
from types import SimpleNamespace

import pytest

from ingest.chunking import Chunk
from ingest.config import (
    RETRIEVAL_FINGERPRINT_REVISION,
    ConfigurationError,
    load_config,
    retrieval_fingerprint_sha256,
)
from ingest.dedup import content_hash
from ingest.generation import GENERATION_SCHEMA_VERSION
from ingest.qdrant_store import (
    KEYWORD_FIELDS,
    TEXT_FIELDS,
    _rfc3339,
    build_payload,
    chunking_fingerprint,
    collection_configuration,
    collection_configuration_sha256,
    delete_doc_chunks_from,
    ensure_collection,
    prepare_embed_collection,
    point_id,
    refuse_aliased_write_target,
    upsert_points,
    validate_generation_write_target,
    vector_space_id,
)
from ingest.sources import CanonicalDoc, PageBoundary


def test_schema_v2_point_identity_separates_versions_and_preserves_legacy_id():
    legacy = point_id("matsne", "doc-1", 0)
    current = point_id("matsne", "doc-1", 0, version_id="current")
    repealed = point_id("matsne", "doc-1", 0, version_id="repealed")

    assert len({legacy, current, repealed}) == 3
    assert current == point_id("matsne", "doc-1", 0, version_id="current")


def _doc(**over):
    base = dict(
        source="ecd", document_id="5", title="T", date="2020-04-30", date_raw="2020-04-30",
        language="ka", document_type="court_decision", court="court", source_url="u",
        document_number=None, registration_code=None, parties=None,
        status=None, status_raw=None, in_force_date=None, expiry_date=None,
        body_markdown="b", extra={},
    )
    base.update(over)
    return CanonicalDoc(**base)


def test_rfc3339_validates_calendar_dates():
    assert _rfc3339("2020-04-30") == "2020-04-30T00:00:00Z"
    assert _rfc3339("2020-13-45") is None      # calendar-invalid -> dropped
    assert _rfc3339("2020-04-30 00:00:00") is None  # not a bare date
    assert _rfc3339("") is None
    assert _rfc3339(None) is None


def test_build_payload_shape():
    doc = _doc()
    chunk = Chunk(text="hello world", chunk_index=2, heading_path=["A", "B"], token_count=2)
    p = build_payload(doc, chunk)
    assert p["source"] == "ecd"
    assert p["document_id"] == "5"
    assert p["chunk_index"] == 2
    assert p["date"] == "2020-04-30T00:00:00Z"
    assert p["date_raw"] == "2020-04-30"
    assert p["heading"] == "A > B"
    assert p["text"] == "hello world"
    assert p["language"] == "ka"


def test_build_payload_carries_new_fields():
    doc = _doc(
        source="matsne", document_number="55", registration_code="140130000.22.034.017712",
        parties="ს. წიკლაური",
    )
    p = build_payload(doc, Chunk(text="t", chunk_index=0, heading_path=[], token_count=1))
    assert p["document_number"] == "55"
    assert p["registration_code"] == "140130000.22.034.017712"
    assert p["parties"] == "ს. წიკლაური"


def test_build_payload_keeps_raw_date_independent_of_iso():
    # date is the ISO (sortable) value; date_raw preserves the original scraped string.
    doc = _doc(date="2026-04-08", date_raw="08/04/2026")
    p = build_payload(doc, Chunk(text="t", chunk_index=0, heading_path=[], token_count=1))
    assert p["date"] == "2026-04-08T00:00:00Z"
    assert p["date_raw"] == "08/04/2026"


def test_build_payload_handles_bad_date():
    doc = _doc(date=None, date_raw="not-a-date", title=None, document_type=None,
               court=None, source_url=None)
    p = build_payload(doc, Chunk(text="t", chunk_index=0, heading_path=[], token_count=1))
    assert p["date"] is None          # invalid -> not indexed
    assert p["date_raw"] == "not-a-date"
    assert p["heading"] is None


def test_build_payload_carries_status_and_force_dates():
    doc = _doc(source="matsne", status="repealed", status_raw="ძალადაკარგული აქტები",
               in_force_date="2020-01-01", expiry_date="2024-06-03")
    p = build_payload(doc, Chunk(text="t", chunk_index=0, heading_path=[], token_count=1))
    assert p["status"] == "repealed"
    assert p["in_force_date"] == "2020-01-01T00:00:00Z"   # promoted to RFC3339 for the index
    assert p["expiry_date"] == "2024-06-03T00:00:00Z"


def test_new_index_fields_are_declared():
    # The fields the new lookup/browse tools filter on must be indexable.
    assert "document_number" in KEYWORD_FIELDS
    assert "registration_code" in KEYWORD_FIELDS
    assert "status" in KEYWORD_FIELDS
    assert {"text", "title", "parties", "article_summary"} <= set(TEXT_FIELDS)


def test_build_payload_carries_content_hash():
    # The doc-body identity is stamped on every chunk (same value across a doc's chunks)
    # and equals the corpus dedup hash of the cleaned body — this is what watch compares
    # to skip re-embedding unchanged docs.
    doc = _doc(body_markdown="the cleaned body text")
    p0 = build_payload(doc, Chunk(text="a", chunk_index=0, heading_path=[], token_count=1))
    p1 = build_payload(doc, Chunk(text="b", chunk_index=1, heading_path=[], token_count=1))
    assert p0["content_hash"] == content_hash("the cleaned body text")
    assert p0["content_hash"] == p1["content_hash"]
    assert "content_hash" in KEYWORD_FIELDS


def test_canonical_passage_payload_hashes_exact_document_slice():
    body = "შესავალი.\n\nმუხლი 7. წესი.\n\n1. ზუსტი ციტატა."
    from ingest.chunking import chunk_document

    doc = _doc(source="matsne", document_id="law-7", body_markdown=body)
    chunk = chunk_document(body, max_tokens=50, overlap=0, min_tokens=1)[1]
    payload = build_payload(doc, chunk)
    exact = body[chunk.char_start : chunk.char_end]
    assert payload["text"] == exact
    assert payload["canonical_text_exact"] is True
    assert payload["passage_hash"] == hashlib.sha256(exact.encode("utf-8")).hexdigest()
    assert payload["article_id"] == "7"
    assert payload["clause_id"] == "1"
    assert payload["heading_path"][-1].startswith("მუხლი 7")
    assert ":article:7:" in payload["parent_id"]
    assert payload["article_start_chunk_index"] == chunk.chunk_index
    assert payload["version_id"].startswith("derived:")
    assert payload["source_authority"] == "primary_official"


def test_page_mapping_is_stored_once_and_hash_bound_on_every_chunk():
    body = "abcdefghij"
    boundaries = (
        PageBoundary(page=1, char_start=0, char_end=5),
        PageBoundary(page=2, char_start=5, char_end=10),
    )
    doc = _doc(
        body_markdown=body,
        page_boundaries=boundaries,
        page_coordinate_reason="exact_pdf_text",
    )
    first = Chunk(
        text=body[:6],
        canonical_text=body[:6],
        chunk_index=0,
        heading_path=[],
        token_count=1,
        char_start=0,
        char_end=6,
        page_start=1,
        page_end=2,
        page_coordinate_reason="exact_pdf_text",
    )
    second = Chunk(
        text=body[6:],
        canonical_text=body[6:],
        chunk_index=1,
        heading_path=[],
        token_count=1,
        char_start=6,
        char_end=10,
        page_start=2,
        page_end=2,
        page_coordinate_reason="exact_pdf_text",
    )

    first_payload = build_payload(doc, first)
    second_payload = build_payload(doc, second)
    assert first_payload["page_boundaries"] == [
        {"page": 1, "char_start": 0, "char_end": 5},
        {"page": 2, "char_start": 5, "char_end": 10},
    ]
    assert second_payload["page_boundaries"] is None
    assert first_payload["page_boundary_mapping_sha256"] == second_payload[
        "page_boundary_mapping_sha256"
    ]
    assert (first_payload["page_start"], first_payload["page_end"]) == (1, 2)
    assert first_payload["page_coordinate_reason"] == "exact_pdf_text"
    assert first_payload["admissible"] is True


def test_canonical_passage_payload_rejects_false_hash_or_offsets():
    body = "ზუსტი ტექსტი"
    bad = Chunk(
        text=body,
        canonical_text="სხვა",
        passage_hash=hashlib.sha256("სხვა".encode("utf-8")).hexdigest(),
        chunk_index=0,
        heading_path=[],
        token_count=2,
        char_start=0,
        char_end=len(body),
    )
    with pytest.raises(ValueError, match="does not equal"):
        build_payload(_doc(body_markdown=body), bad)


def test_build_payload_carries_content_completeness_lineage():
    doc = _doc(
        content_kind="ruling_full_text",
        content_complete=True,
        extraction_status="full_text",
        source_binary_url="https://court.example/ruling.pdf",
        article_summary="summary",
    )
    payload = build_payload(
        doc, Chunk(text="ruling", chunk_index=0, heading_path=[], token_count=1)
    )
    assert payload["content_kind"] == "ruling_full_text"
    assert payload["content_complete"] is True
    assert payload["extraction_status"] == "full_text"
    assert payload["source_binary_url"].endswith("ruling.pdf")
    assert payload["article_summary"] == "summary"


def test_build_payload_carries_optional_document_chunk_count():
    chunk = Chunk(text="a", chunk_index=0, heading_path=[], token_count=1)
    assert "document_chunk_count" not in build_payload(_doc(), chunk)
    assert build_payload(_doc(), chunk, document_chunk_count=7)["document_chunk_count"] == 7


def test_build_payload_carries_optional_document_state_hash():
    chunk = Chunk(text="a", chunk_index=0, heading_path=[], token_count=1)
    assert "document_state_hash" not in build_payload(_doc(), chunk)
    assert build_payload(_doc(), chunk, document_state_hash="state")["document_state_hash"] == "state"


def test_build_payload_content_hash_changes_with_body():
    a = build_payload(_doc(body_markdown="v1"), Chunk(text="x", chunk_index=0, heading_path=[], token_count=1))
    b = build_payload(_doc(body_markdown="v2"), Chunk(text="x", chunk_index=0, heading_path=[], token_count=1))
    assert a["content_hash"] != b["content_hash"]


def _generation_cfg(**over):
    cfg = dataclasses.replace(
        load_config(),
        generation_id="gen_20260713_verified",
        collection_name="georgian_legal__gen_gen_20260713_verified",
        embedding_revision="a" * 40,
        tokenizer_revision="b" * 40,
        reranker_revision="c" * 40,
        rerank_enabled=False,
    )
    return dataclasses.replace(cfg, **over)


def test_generation_payload_carries_complete_cryptographic_identity():
    cfg = _generation_cfg(embed_header_v2=True)
    payload = build_payload(
        _doc(body_markdown="x", source_fingerprint="d" * 64, official_url="u"),
        Chunk(
            text="x",
            canonical_text="x",
            passage_hash=hashlib.sha256(b"x").hexdigest(),
            chunk_index=0,
            heading_path=[],
            token_count=1,
            char_start=0,
            char_end=1,
        ),
        document_chunk_count=1,
        document_state_hash="c" * 64,
        cfg=cfg,
    )
    assert payload["schema_version"] == GENERATION_SCHEMA_VERSION
    assert payload["generation_id"] == cfg.generation_id
    assert payload["embedding_model"] == cfg.embed_model
    assert payload["embedding_revision"] == "a" * 40
    assert payload["tokenizer_model"] == cfg.tokenizer_model
    assert payload["tokenizer_revision"] == "b" * 40
    assert payload["reranker_model"] == cfg.rerank_model
    assert payload["reranker_revision"] == "c" * 40
    assert payload["vector_space_id"] == vector_space_id(cfg)
    assert payload["chunking_fingerprint"] == chunking_fingerprint(cfg)
    assert payload["document_header"] is True
    assert payload["retrieval_fingerprint"] == retrieval_fingerprint_sha256(cfg)
    assert payload["retrieval_fingerprint_revision"] == RETRIEVAL_FINGERPRINT_REVISION
    assert len(payload["retrieval_fingerprint"]) == 64
    assert payload["canonical_text_exact"] is True


def test_generation_payload_rejects_unproven_canonical_text():
    with pytest.raises(ConfigurationError, match="proven equal"):
        build_payload(
            _doc(source_fingerprint="d" * 64),
            Chunk(text="x", chunk_index=0, heading_path=[], token_count=1),
            document_chunk_count=1,
            document_state_hash="c" * 64,
            cfg=_generation_cfg(),
        )


def test_generation_payload_rejects_partial_identity_and_document_markers():
    chunk = Chunk(text="x", chunk_index=0, heading_path=[], token_count=1)
    with pytest.raises(ConfigurationError, match="TOKENIZER_REVISION"):
        build_payload(
            _doc(),
            chunk,
            document_chunk_count=1,
            document_state_hash="c" * 64,
            cfg=_generation_cfg(tokenizer_revision=None),
        )
    with pytest.raises(ConfigurationError, match="RERANK_REVISION"):
        build_payload(
            _doc(),
            chunk,
            document_chunk_count=1,
            document_state_hash="c" * 64,
            cfg=_generation_cfg(reranker_revision=None),
        )
    with pytest.raises(ConfigurationError, match="immutable revisions"):
        build_payload(
            _doc(),
            chunk,
            document_chunk_count=1,
            document_state_hash="c" * 64,
            cfg=_generation_cfg(reranker_revision="main"),
        )
    with pytest.raises(ConfigurationError, match="document_chunk_count"):
        build_payload(_doc(), chunk, cfg=_generation_cfg())


def test_legacy_payload_remains_unversioned():
    payload = build_payload(
        _doc(), Chunk(text="x", chunk_index=0, heading_path=[], token_count=1)
    )
    assert "schema_version" not in payload
    assert "generation_id" not in payload


def test_generation_writer_refuses_stable_alias_before_client_access():
    class NoClientAccess:
        def collection_exists(self, _name):
            raise AssertionError("client must not be accessed")

    cfg = _generation_cfg(collection_name="georgian_legal")
    with pytest.raises(ConfigurationError, match="physical collection"):
        ensure_collection(NoClientAccess(), cfg)


def test_generation_writer_requires_explicit_generation_before_client_access():
    class NoClientAccess:
        def collection_exists(self, _name):
            raise AssertionError("client must not be accessed")

    cfg = dataclasses.replace(
        _generation_cfg(), generation_id=None, collection_name="georgian_legal"
    )
    with pytest.raises(ConfigurationError, match="explicit non-legacy GENERATION_ID"):
        ensure_collection(
            NoClientAccess(),
            cfg,
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )


def test_generation_writer_requires_apply_and_independent_approval():
    cfg = _generation_cfg()
    with pytest.raises(ConfigurationError, match="--apply"):
        validate_generation_write_target(
            cfg, apply=False, environ={"QDRANT_WRITE_APPROVED": "1"}
        )
    with pytest.raises(ConfigurationError, match="QDRANT_WRITE_APPROVED"):
        validate_generation_write_target(cfg, apply=True, environ={})

    validate_generation_write_target(
        cfg, apply=True, environ={"QDRANT_WRITE_APPROVED": "1"}
    )
    validate_generation_write_target(
        cfg, apply=True, environ={"RUNPOD_EPHEMERAL_QDRANT": "1"}
    )


def test_generation_writer_requires_exact_physical_name_not_suffix():
    cfg = _generation_cfg(
        collection_name="scratch__gen_gen_20260713_verified"
    )
    with pytest.raises(ConfigurationError, match="exact physical collection"):
        validate_generation_write_target(
            cfg, apply=True, environ={"QDRANT_WRITE_APPROVED": "1"}
        )


def test_generation_candidate_recreate_is_always_forbidden_before_client_access():
    class NoClientAccess:
        def collection_exists(self, _name):
            raise AssertionError("client must not be accessed")

    with pytest.raises(ConfigurationError, match="create-only"):
        ensure_collection(
            NoClientAccess(),
            _generation_cfg(),
            recreate=True,
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )


def _candidate_info(points_count=3):
    params = SimpleNamespace(
        vectors={
            "dense": SimpleNamespace(size=1024, distance="Cosine"),
        },
        sparse_vectors={"sparse": SimpleNamespace()},
    )
    return SimpleNamespace(
        config=SimpleNamespace(params=params),
        points_count=points_count,
    )


class _ExistingCandidate:
    def __init__(self, *, points_count=3, matching_count=3, count_error=None):
        self.info = _candidate_info(points_count)
        self.matching_count = matching_count
        self.count_error = count_error
        self.count_filter = None

    def collection_exists(self, _name):
        return True

    def get_collection(self, _name):
        return self.info

    def count(self, *, collection_name, count_filter, exact):
        assert collection_name == "georgian_legal__gen_gen_20260713_verified"
        assert exact is True
        self.count_filter = count_filter
        if self.count_error is not None:
            raise self.count_error
        return SimpleNamespace(count=self.matching_count)


def test_resume_requires_every_existing_point_to_match_full_identity():
    client = _ExistingCandidate()
    assert prepare_embed_collection(
        client,
        _generation_cfg(),
        resume=True,
        recreate=False,
        apply=True,
        environ={"QDRANT_WRITE_APPROVED": "1"},
    ) == 3
    fields = {condition.key for condition in client.count_filter.must}
    assert {
        "generation_id",
        "canonical_payload_revision",
        "embedding_revision",
        "tokenizer_revision",
        "reranker_revision",
        "vector_space_id",
        "chunking_fingerprint",
        "retrieval_fingerprint",
        "retrieval_fingerprint_revision",
    } <= fields


def test_resume_rejects_mismatched_or_unavailable_identity_proof():
    with pytest.raises(RuntimeError, match="2 of 3 points"):
        prepare_embed_collection(
            _ExistingCandidate(matching_count=2),
            _generation_cfg(),
            resume=True,
            recreate=False,
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )
    with pytest.raises(RuntimeError, match="cannot verify existing point identity"):
        prepare_embed_collection(
            _ExistingCandidate(count_error=OSError("offline")),
            _generation_cfg(),
            resume=True,
            recreate=False,
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )


def test_resume_rejects_collection_smaller_than_acknowledged_checkpoint():
    with pytest.raises(RuntimeError, match="below the 4 chunks acknowledged"):
        prepare_embed_collection(
            _ExistingCandidate(points_count=3, matching_count=3),
            _generation_cfg(),
            resume=True,
            recreate=False,
            apply=True,
            minimum_points=4,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )


def test_fresh_embed_refuses_any_existing_candidate_without_count_filter():
    client = _ExistingCandidate()
    with pytest.raises(RuntimeError, match="any pre-existing physical collection"):
        prepare_embed_collection(
            client,
            _generation_cfg(),
            resume=False,
            recreate=False,
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )
    assert client.count_filter is None


def test_fresh_embed_also_refuses_an_existing_empty_candidate():
    client = _ExistingCandidate(points_count=0, matching_count=0)
    with pytest.raises(RuntimeError, match="including an empty one"):
        prepare_embed_collection(
            client,
            _generation_cfg(),
            resume=False,
            recreate=False,
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )


def test_complete_collection_configuration_excludes_only_live_index_counts():
    base = SimpleNamespace(
        config={"params": {"vectors": {"dense": {"size": 1024}}}},
        payload_schema={
            "source": {"data_type": "keyword", "params": None, "points": 0}
        },
    )
    populated = SimpleNamespace(
        config=base.config,
        payload_schema={
            "source": {"data_type": "keyword", "params": None, "points": 999}
        },
    )
    configured_differently = SimpleNamespace(
        config=base.config,
        payload_schema={
            "source": {
                "data_type": "keyword",
                "params": {"on_disk": True},
                "points": 999,
            }
        },
    )
    base_value = collection_configuration(base)
    assert collection_configuration(populated) == base_value
    assert collection_configuration_sha256(
        collection_configuration(populated)
    ) == collection_configuration_sha256(base_value)
    assert collection_configuration(configured_differently) != base_value


def test_candidate_write_target_rejects_alias_reference_and_alias_uncertainty():
    with pytest.raises(RuntimeError, match="referenced by aliases"):
        refuse_aliased_write_target(
            SimpleNamespace(
                get_aliases=lambda: SimpleNamespace(
                    aliases=[
                        SimpleNamespace(
                            alias_name="candidate-live",
                            collection_name=_generation_cfg().collection_name,
                        )
                    ]
                )
            ),
            _generation_cfg().collection_name,
        )
    with pytest.raises(RuntimeError, match="cannot inspect Qdrant aliases"):
        refuse_aliased_write_target(SimpleNamespace(), _generation_cfg().collection_name)


@pytest.mark.parametrize("collection", ["georgian_legal", "georgian_legal_delta"])
def test_low_level_mutations_refuse_serving_collections(collection):
    class NoClientAccess:
        def upsert(self, **_kwargs):
            raise AssertionError("client must not be accessed")

        def delete(self, **_kwargs):
            raise AssertionError("client must not be accessed")

    with pytest.raises(ConfigurationError, match="strictly read-only"):
        upsert_points(NoClientAccess(), collection, [object()])
    with pytest.raises(ConfigurationError, match="strictly read-only"):
        delete_doc_chunks_from(NoClientAccess(), collection, "ecd", "1", 0)
