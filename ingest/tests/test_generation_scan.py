import hashlib

import pytest

from ingest.config import load_config
from ingest.generation_scan import aggregate_documents
from ingest.generation import CANONICAL_PAYLOAD_REVISION
from ingest import qdrant_store as store


@pytest.fixture
def cfg():
    return load_config()


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_aggregates_contiguous_document_into_one_record(cfg):
    body_hash = _chunk_hash("clean body")
    points = [
        {
            "source": "matsne", "document_id": "d1", "chunk_index": 0, "text": "chunk zero",
            "content_hash": body_hash,
        },
        {
            "source": "matsne", "document_id": "d1", "chunk_index": 1, "text": "chunk one",
            "content_hash": body_hash,
        },
    ]
    result = aggregate_documents(points, generation_id="legacy_2026_07_09", cfg=cfg)

    assert result.chunk_count == 2
    assert len(result.documents) == 1
    doc = result.documents[0]
    assert doc["source"] == "matsne"
    assert doc["document_id"] == "d1"
    assert doc["expected_chunk_count"] == 2
    assert doc["content_hash"] == body_hash
    assert doc["content_kind"] == "full_text"
    assert doc["extraction_status"] == "full_text"
    assert doc["content_complete"] is True
    assert doc["exclusion_reason"] is None
    assert not result.issues


def test_two_documents_are_kept_separate(cfg):
    points = [
        {"source": "matsne", "document_id": "a", "chunk_index": 0, "text": "a0", "content_hash": _chunk_hash("a")},
        {"source": "matsne", "document_id": "b", "chunk_index": 0, "text": "b0", "content_hash": _chunk_hash("b")},
    ]
    result = aggregate_documents(points, generation_id="legacy_2026_07_09", cfg=cfg)
    ids = sorted(d["document_id"] for d in result.documents)
    assert ids == ["a", "b"]


def test_two_versions_of_same_document_are_kept_separate(cfg):
    points = [
        {
            "source": "matsne", "document_id": "d1", "version_id": "v1",
            "canonical_payload_revision": CANONICAL_PAYLOAD_REVISION,
            "chunk_index": 0, "text": "same", "content_hash": _chunk_hash("same"),
        },
        {
            "source": "matsne", "document_id": "d1", "version_id": "v2",
            "canonical_payload_revision": CANONICAL_PAYLOAD_REVISION,
            "chunk_index": 0, "text": "same", "content_hash": _chunk_hash("same"),
        },
    ]

    result = aggregate_documents(points, generation_id="legacy_2026_07_09", cfg=cfg)

    assert {doc["version_id"] for doc in result.documents} == {"v1", "v2"}
    assert {sample["point_id"] for sample in result.samples} == {
        store.point_id("matsne", "d1", 0, version_id="v1"),
        store.point_id("matsne", "d1", 0, version_id="v2"),
    }


def test_out_of_order_scroll_still_aggregates_correctly(cfg):
    # Real Qdrant scroll makes no ordering promise across documents' chunks.
    points = [
        {"source": "matsne", "document_id": "d1", "chunk_index": 2, "text": "c2", "content_hash": _chunk_hash("x")},
        {"source": "matsne", "document_id": "d1", "chunk_index": 0, "text": "c0", "content_hash": _chunk_hash("x")},
        {"source": "matsne", "document_id": "d1", "chunk_index": 1, "text": "c1", "content_hash": _chunk_hash("x")},
    ]
    result = aggregate_documents(points, generation_id="legacy_2026_07_09", cfg=cfg)
    assert len(result.documents) == 1
    assert result.documents[0]["expected_chunk_count"] == 3
    assert not result.issues


def test_missing_chunk_flagged_as_issue_not_fatal(cfg):
    # chunk_index 0 and 2 present, 1 missing -> non-contiguous.
    points = [
        {"source": "matsne", "document_id": "d1", "chunk_index": 0, "text": "c0", "content_hash": _chunk_hash("x")},
        {"source": "matsne", "document_id": "d1", "chunk_index": 2, "text": "c2", "content_hash": _chunk_hash("x")},
    ]
    result = aggregate_documents(points, generation_id="legacy_2026_07_09", cfg=cfg)
    assert len(result.documents) == 1  # still produced a record, not dropped
    assert any("non-contiguous" in i.reason for i in result.issues)


def test_missing_chunk_zero_falls_back_and_flags_issue(cfg):
    points = [
        {"source": "matsne", "document_id": "d1", "chunk_index": 1, "text": "c1", "content_hash": _chunk_hash("x")},
    ]
    result = aggregate_documents(points, generation_id="legacy_2026_07_09", cfg=cfg)
    assert len(result.documents) == 1
    assert any("chunk_index 0 missing" in i.reason for i in result.issues)


def test_missing_content_hash_flagged_not_crashed(cfg):
    points = [{"source": "matsne", "document_id": "d1", "chunk_index": 0, "text": "c0"}]
    result = aggregate_documents(points, generation_id="legacy_2026_07_09", cfg=cfg)
    assert len(result.documents) == 1
    assert any("missing content_hash" in i.reason for i in result.issues)


def test_legacy_payload_without_document_state_hash_is_reconstructed(cfg):
    # No document_state_hash in the payload (true of every currently-live point) ->
    # reuses pipeline._document_state_hash, the same reconstruction watch already trusts.
    points = [{
        "source": "matsne", "document_id": "d1", "chunk_index": 0, "text": "c0",
        "content_hash": _chunk_hash("body"),
    }]
    result = aggregate_documents(points, generation_id="legacy_2026_07_09", cfg=cfg)
    doc = result.documents[0]
    assert len(doc["document_state_hash"]) == 64
    int(doc["document_state_hash"], 16)  # valid hex


def test_new_style_payload_fields_are_respected_when_present(cfg):
    # A hypothetical future point that DOES carry the newer content_kind fields (e.g. the
    # TB Appeals article-summary fallback) must not be overwritten with the legacy default.
    points = [{
        "source": "tbappeal", "document_id": "d1", "chunk_index": 0, "text": "summary only",
        "content_hash": _chunk_hash("s"), "content_kind": "article_summary",
        "extraction_status": "resource_limited", "content_complete": False,
    }]
    result = aggregate_documents(points, generation_id="legacy_2026_07_09", cfg=cfg)
    doc = result.documents[0]
    assert doc["content_kind"] == "article_summary"
    assert doc["extraction_status"] == "resource_limited"
    assert doc["content_complete"] is False


def test_sample_check_point_id_matches_deterministic_point_id(cfg):
    points = [{"source": "matsne", "document_id": "d1", "chunk_index": 0, "text": "hello",
               "content_hash": _chunk_hash("body")}]
    result = aggregate_documents(points, generation_id="legacy_2026_07_09", cfg=cfg)
    assert len(result.samples) == 1
    sample = result.samples[0]
    assert sample["point_id"] == store.point_id("matsne", "d1", 0)
    assert sample["text_sha256"] == hashlib.sha256(b"hello").hexdigest()
    assert sample["chunk_index"] == 0


def test_malformed_points_are_skipped_not_fatal(cfg):
    points = [
        {"source": "matsne", "chunk_index": 0, "text": "no document_id"},  # missing document_id
        {"document_id": "d1", "chunk_index": 0, "text": "no source"},  # missing source
        {"source": "matsne", "document_id": "d1", "text": "no chunk_index"},  # missing chunk_index
        {"source": "matsne", "document_id": "d2", "chunk_index": 0, "text": "ok",
         "content_hash": _chunk_hash("ok")},
    ]
    result = aggregate_documents(points, generation_id="legacy_2026_07_09", cfg=cfg)
    assert len(result.documents) == 1
    assert result.documents[0]["document_id"] == "d2"


def test_empty_scroll_yields_empty_result(cfg):
    result = aggregate_documents([], generation_id="legacy_2026_07_09", cfg=cfg)
    assert result.documents == []
    assert result.samples == []
    assert result.chunk_count == 0
    assert not result.issues


def test_rejects_invalid_generation_id(cfg):
    with pytest.raises(Exception):
        aggregate_documents([], generation_id="v1_not_allowed", cfg=cfg)
