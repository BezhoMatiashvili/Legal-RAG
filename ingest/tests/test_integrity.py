import hashlib
import json
import stat
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ingest.generation import (
    CANONICAL_PAYLOAD_REVISION,
    GENERATION_SCHEMA_VERSION,
    DocumentRecord,
    GenerationManifest,
    SampleCheck,
)
from ingest.integrity import verify_generation_points, write_verification_report
from ingest.qdrant_store import point_id

GENERATION_ID = "20260713t120000z_core"
MANIFEST_SHA256 = "f" * 64
CONTENT_HASH = "a" * 64
STATE_HASH = "b" * 64
VECTOR_SPACE_ID = "c" * 64
CHUNKING_FINGERPRINT = "d" * 64
RETRIEVAL_FINGERPRINT = "e" * 64
REVISION = "1" * 40
VERSION_ID = "derived:" + "8" * 64


def _manifest(*, chunks=2, samples=1, documents=1):
    return GenerationManifest.from_dict(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "document_count": documents,
            "indexed_document_count": documents,
            "excluded_document_count": 0,
            "chunk_count": chunks,
            "sample_count": samples,
            "corpus": {
                "name": "georgian_legal",
                "snapshot_sha256": "2" * 64,
            },
            "source": {"name": "snapshot-v3", "state_sha256": "3" * 64},
            "model": {
                "embedding_model": "BAAI/bge-m3",
                "embedding_revision": REVISION,
                "tokenizer_model": "BAAI/bge-m3",
                "tokenizer_revision": REVISION,
                "reranker_model": "BAAI/bge-reranker-v2-m3",
                "reranker_revision": REVISION,
            },
            "vector_space": {
                "id": VECTOR_SPACE_ID,
                "dense_name": "dense",
                "dense_dimension": 3,
                "distance": "cosine",
                "sparse_name": "sparse",
            },
            "chunking": {
                "fingerprint": CHUNKING_FINGERPRINT,
                "max_tokens": 512,
                "overlap_tokens": 64,
                "document_header": True,
            },
            "covered_runs": [{"source": "matsne", "run_id": "20260713t100000z"}],
            "retrieval_fingerprint_revision": 2,
            "retrieval_fingerprint": RETRIEVAL_FINGERPRINT,
            "code": {"git_sha": "4" * 40, "dirty_patch_sha256": None},
            "dependency": {"lock_sha256": "5" * 64, "image_digest": None},
            "creation": {
                "created_at": "2026-07-13T12:00:00Z",
                "run_id": "build-20260713",
                "actor": "release-worker",
            },
        }
    )


def _document(
    *, chunks=2, complete=True, deadline="2099-01-01T00:00:00Z",
    version_id=VERSION_ID,
):
    return DocumentRecord.from_dict(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "source": "matsne",
            "document_id": "doc-1",
            "version_id": version_id,
            "source_identity": "6" * 64,
            "content_hash": CONTENT_HASH,
            "document_state_hash": STATE_HASH,
            "expected_chunk_count": chunks,
            "content_kind": "full_text" if complete else "article_summary",
            "content_complete": complete,
            "extraction_status": "full_text" if complete else "malformed",
            "article_summary": None if complete else "article summary",
            "exclusion_reason": None,
            "refresh_deadline": deadline,
            "source_binary_url": None,
        }
    )


def _point_id(source, document_id, chunk_index, version_id=VERSION_ID):
    return point_id(source, document_id, chunk_index, version_id=version_id)


def _point(
    chunk_index,
    *,
    chunks=2,
    complete=True,
    source="matsne",
    document_id="doc-1",
    version_id=VERSION_ID,
    text=None,
    vectors=None,
):
    text = text if text is not None else f"chunk {chunk_index}"
    char_start = chunk_index * 100
    payload = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "canonical_payload_revision": CANONICAL_PAYLOAD_REVISION,
        "generation_id": GENERATION_ID,
        "source": source,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "document_chunk_count": chunks,
        "document_state_hash": STATE_HASH,
        "content_hash": CONTENT_HASH,
        "canonical_content_hash": CONTENT_HASH,
        "canonical_text_exact": True,
        "passage_id": f"passage:{chunk_index}",
        "passage_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "source_fingerprint": "9" * 64,
        "normalizer_revision": "canonical-source-normalizer-v2",
        "chunker_revision": "structural-article-clause-v2",
        "model_revision": REVISION,
        "article_id": None,
        "clause_id": None,
        "subarticle": None,
        "chapter": None,
        "heading_path": [],
        "parent_id": None,
        "article_start_chunk_index": None,
        "parent_chunk_index": None,
        "char_start": char_start,
        "char_end": char_start + len(text),
        "page_start": None,
        "page_end": None,
        "version_id": version_id,
        "supersedes": [],
        "effective_from": None,
        "effective_to": None,
        "repeal_date": None,
        "consolidation_status": None,
        "version_lineage_status": "partial",
        "version_lineage_complete": False,
        "official_url": "https://example.invalid/document",
        "official_binary_url": None,
        "source_authority": "primary_official",
        "freshness_sla_met": True,
        "content_kind": "full_text" if complete else "article_summary",
        "content_complete": complete,
        "extraction_status": "full_text" if complete else "malformed",
        "article_summary": None if complete else "article summary",
        "source_binary_url": None,
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": REVISION,
        "tokenizer_model": "BAAI/bge-m3",
        "tokenizer_revision": REVISION,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_revision": REVISION,
        "vector_space_id": VECTOR_SPACE_ID,
        "chunking_fingerprint": CHUNKING_FINGERPRINT,
        "document_header": True,
        "retrieval_fingerprint_revision": 2,
        "retrieval_fingerprint": RETRIEVAL_FINGERPRINT,
        "text": text,
    }
    return {
        "id": _point_id(source, document_id, chunk_index, version_id),
        "payload": payload,
        "vector": vectors
        if vectors is not None
        else {
            "dense": [0.1, 0.2, 0.3],
            "sparse": {"indices": [1, 4], "values": [0.25, 0.75]},
        },
    }


def _sample(*, text="chunk 0", version_id=VERSION_ID):
    return SampleCheck.from_dict(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "source": "matsne",
            "document_id": "doc-1",
            "version_id": version_id,
            "chunk_index": 0,
            "point_id": _point_id("matsne", "doc-1", 0, version_id),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    )


def _verify(
    *, manifest=None, document=None, documents=None, samples=None, points=None, **kwargs
):
    manifest = manifest or _manifest()
    document = document or _document(chunks=manifest.chunk_count)
    document_records = documents if documents is not None else [document]
    samples = samples if samples is not None else [_sample()]
    points = (
        points
        if points is not None
        else [
            _point(index, chunks=manifest.chunk_count)
            for index in range(manifest.chunk_count)
        ]
    )
    return verify_generation_points(
        manifest,
        MANIFEST_SHA256,
        document_records,
        samples,
        points,
        now=datetime(2026, 7, 13, tzinfo=timezone.utc),
        **kwargs,
    )


def _codes(outcome):
    return {example["code"] for example in outcome.examples}


def test_valid_qdrant_like_stream_passes_all_four_gates_and_cleans_sqlite(tmp_path):
    points = iter(
        [
            _point(0),
            SimpleNamespace(**_point(1)),
        ]
    )

    report = _verify(points=points, temp_dir=tmp_path)

    assert report.ok
    assert report.manifest_sha256 == MANIFEST_SHA256
    assert report.covered_runs == ({"source": "matsne", "run_id": "20260713t100000z"},)
    assert report.stats["observed_unique_chunks"] == 2
    assert report.coverage.ok
    assert report.integrity.ok
    assert report.freshness.ok
    assert report.quality.ok
    assert list(tmp_path.iterdir()) == []


def test_integrity_accounts_for_two_versions_of_same_document_independently():
    other_version = "derived:" + "7" * 64
    report = _verify(
        manifest=_manifest(chunks=2, samples=2, documents=2),
        documents=[
            _document(chunks=1, version_id=VERSION_ID),
            _document(chunks=1, version_id=other_version),
        ],
        samples=[
            _sample(version_id=VERSION_ID),
            _sample(version_id=other_version),
        ],
        points=[
            _point(0, chunks=1, version_id=VERSION_ID),
            _point(0, chunks=1, version_id=other_version),
        ],
    )

    assert report.ok
    assert report.stats["observed_unique_chunks"] == 2


def test_legacy_payload_fails_closed_with_bounded_examples():
    point = _point(0, chunks=1)
    point["payload"] = {
        "source": "matsne",
        "document_id": "doc-1",
        "chunk_index": 0,
        "text": "chunk 0",
    }

    report = _verify(
        manifest=_manifest(chunks=1),
        document=_document(chunks=1),
        points=[point],
        max_examples=2,
    )

    assert not report.coverage.ok
    assert not report.integrity.ok
    assert report.integrity.issue_count > 2
    assert len(report.integrity.examples) == 2
    assert "missing_payload_field" in _codes(report.integrity)


def test_exact_accounting_detects_duplicates_unexpected_points_and_chunk_holes():
    duplicate = _point(0)
    hole = _point(2)
    unexpected = _point(0, source="tas", document_id="other")

    report = _verify(points=[_point(0), duplicate, hole, unexpected])

    assert not report.ok
    assert not report.coverage.ok
    assert not report.integrity.ok
    assert "duplicate_logical_chunk" in _codes(report.integrity)
    assert "unexpected_point" in _codes(report.integrity)
    assert "non_contiguous_chunk_indexes" in _codes(report.integrity)
    assert "generation_chunk_count_mismatch" in _codes(report.coverage)
    assert report.stats["unaccounted_points"] == 1


@pytest.mark.parametrize(
    ("vectors", "expected_code"),
    [
        ({"dense": [0.1, 0.2, 0.3]}, "missing_named_vector"),
        (
            {
                "dense": [0.1, 0.2],
                "sparse": {"indices": [1], "values": [0.5]},
            },
            "dense_dimension_mismatch",
        ),
        (
            {
                "dense": [0.1, float("nan"), 0.3],
                "sparse": {"indices": [1], "values": [0.5]},
            },
            "dense_non_finite",
        ),
        (
            {
                "dense": [0.1, 0.2, 0.3],
                "sparse": {"indices": [1, 2], "values": [0.5]},
            },
            "sparse_cardinality_mismatch",
        ),
        (
            {
                "dense": [0.1, 0.2, 0.3],
                "sparse": {"indices": [1], "values": [float("inf")]},
            },
            "sparse_non_finite",
        ),
        (
            {
                "dense": [0.1, 0.2, 0.3],
                "sparse": {"indices": [2, 2], "values": [0.5, -0.1]},
            },
            "sparse_indexes_not_sorted_unique",
        ),
    ],
)
def test_dense_and_sparse_vector_integrity_is_strict(vectors, expected_code):
    report = _verify(
        manifest=_manifest(chunks=1),
        document=_document(chunks=1),
        points=[_point(0, chunks=1, vectors=vectors)],
    )

    assert not report.integrity.ok
    assert expected_code in _codes(report.integrity)
    if expected_code == "sparse_indexes_not_sorted_unique":
        assert "sparse_negative_value" in _codes(report.integrity)


def test_freshness_and_quality_fail_independently_of_valid_index_integrity():
    report = _verify(
        manifest=_manifest(chunks=1),
        document=_document(
            chunks=1,
            complete=False,
            deadline="2026-07-12T00:00:00Z",
        ),
        points=[_point(0, chunks=1, complete=False)],
    )

    assert report.coverage.ok
    assert report.integrity.ok
    assert not report.freshness.ok
    assert not report.quality.ok
    assert "refresh_deadline_expired" in _codes(report.freshness)
    assert "incomplete_document" in _codes(report.quality)


def test_required_payload_types_do_not_pass_via_python_equality():
    point = _point(0, chunks=1)
    point["payload"]["schema_version"] = True
    point["payload"]["document_header"] = 1

    report = _verify(
        manifest=_manifest(chunks=1),
        document=_document(chunks=1),
        points=[point],
    )

    invalid_fields = {
        example.get("field")
        for example in report.integrity.examples
        if example["code"] == "invalid_payload_field"
    }
    assert {"schema_version", "document_header"} <= invalid_fields


@pytest.mark.parametrize(
    "field",
    (
        "passage_hash",
        "canonical_text_exact",
        "source_fingerprint",
        "normalizer_revision",
        "version_id",
        "official_url",
    ),
)
def test_missing_canonical_evidence_field_fails_generation_integrity(field):
    point = _point(0, chunks=1)
    del point["payload"][field]

    report = _verify(
        manifest=_manifest(chunks=1),
        document=_document(chunks=1),
        points=[point],
    )

    assert not report.integrity.ok
    assert any(
        example["code"] == "missing_payload_field" and example.get("field") == field
        for example in report.integrity.examples
    )


def test_mismatched_passage_hash_and_offsets_fail_generation_integrity():
    point = _point(0, chunks=1)
    point["payload"]["passage_hash"] = "f" * 64
    point["payload"]["char_end"] += 1

    report = _verify(
        manifest=_manifest(chunks=1),
        document=_document(chunks=1),
        points=[point],
    )

    codes = _codes(report.integrity)
    assert {"passage_hash_mismatch", "invalid_canonical_offsets"} <= codes


@pytest.mark.parametrize(
    "field",
    (
        "tokenizer_model",
        "reranker_model",
        "reranker_revision",
        "retrieval_fingerprint_revision",
    ),
)
def test_full_model_identity_mismatch_fails_integrity(field):
    point = _point(0, chunks=1)
    point["payload"][field] = 1 if field.endswith("_fingerprint_revision") else "f" * 40

    report = _verify(
        manifest=_manifest(chunks=1),
        document=_document(chunks=1),
        points=[point],
    )

    assert not report.integrity.ok
    assert any(
        example["code"] == "payload_identity_mismatch"
        and example.get("field") == field
        for example in report.integrity.examples
    )


def test_report_is_atomic_owner_only_and_bound_to_exact_manifest(tmp_path):
    report = _verify()
    destination = tmp_path / "generation.verification.json"
    destination.write_text("old", encoding="utf-8")
    destination.chmod(0o644)

    written = write_verification_report(destination, report)

    assert written == destination
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    data = json.loads(destination.read_text(encoding="utf-8"))
    assert data["schema_version"] == GENERATION_SCHEMA_VERSION
    assert data["generation_id"] == GENERATION_ID
    assert data["manifest_sha256"] == MANIFEST_SHA256
    assert data["ok"] is True
    assert all(
        data[name]["ok"] for name in ("coverage", "integrity", "freshness", "quality")
    )
    assert list(tmp_path.glob(".*.tmp")) == []
