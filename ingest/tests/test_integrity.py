import hashlib
import json
import stat
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ingest.generation import (
    GENERATION_SCHEMA_VERSION,
    DocumentRecord,
    GenerationManifest,
    SampleCheck,
)
from ingest.integrity import verify_generation_points, write_verification_report

GENERATION_ID = "20260713t120000z_core"
MANIFEST_SHA256 = "f" * 64
CONTENT_HASH = "a" * 64
STATE_HASH = "b" * 64
VECTOR_SPACE_ID = "c" * 64
CHUNKING_FINGERPRINT = "d" * 64
RETRIEVAL_FINGERPRINT = "e" * 64
REVISION = "1" * 40


def _manifest(*, chunks=2, samples=1):
    return GenerationManifest.from_dict(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "document_count": 1,
            "indexed_document_count": 1,
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


def _document(*, chunks=2, complete=True, deadline="2099-01-01T00:00:00Z"):
    return DocumentRecord.from_dict(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "source": "matsne",
            "document_id": "doc-1",
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


def _point_id(source, document_id, chunk_index):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{document_id}:{chunk_index}"))


def _point(
    chunk_index,
    *,
    chunks=2,
    complete=True,
    source="matsne",
    document_id="doc-1",
    text=None,
    vectors=None,
):
    text = text if text is not None else f"chunk {chunk_index}"
    payload = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": GENERATION_ID,
        "source": source,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "document_chunk_count": chunks,
        "document_state_hash": STATE_HASH,
        "content_hash": CONTENT_HASH,
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
        "retrieval_fingerprint": RETRIEVAL_FINGERPRINT,
        "text": text,
    }
    return {
        "id": _point_id(source, document_id, chunk_index),
        "payload": payload,
        "vector": vectors
        if vectors is not None
        else {
            "dense": [0.1, 0.2, 0.3],
            "sparse": {"indices": [1, 4], "values": [0.25, 0.75]},
        },
    }


def _sample(*, text="chunk 0"):
    return SampleCheck.from_dict(
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "source": "matsne",
            "document_id": "doc-1",
            "chunk_index": 0,
            "point_id": _point_id("matsne", "doc-1", 0),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    )


def _verify(*, manifest=None, document=None, samples=None, points=None, **kwargs):
    manifest = manifest or _manifest()
    document = document or _document(chunks=manifest.chunk_count)
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
        [document],
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

    assert report.coverage.ok
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
    ("tokenizer_model", "reranker_model", "reranker_revision"),
)
def test_full_model_identity_mismatch_fails_integrity(field):
    point = _point(0, chunks=1)
    point["payload"][field] = "f" * 40

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
