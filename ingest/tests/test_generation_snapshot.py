"""Hermetic immutable-generation publication tests; snapshots/v1 is only a temp sentinel."""

from __future__ import annotations

import hashlib
import json
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingest import generation_snapshot
from ingest.generation import (
    CHECKSUM_FILENAME,
    DOCUMENTS_FILENAME,
    MANIFEST_FILENAME,
    SAMPLE_CHECKS_FILENAME,
    GENERATION_SCHEMA_VERSION,
    GenerationFormatError,
    GenerationManifest,
    load_generation,
)
from ingest.generation_snapshot import (
    PROVENANCE_FILENAME,
    QUARANTINE_FILENAME,
    SOURCE_STATE_FILENAME,
    GenerationDestinationExists,
    GenerationPublishError,
    publish_generation,
    source_state_sha256,
)
from ingest.integrity import (
    VerificationOutcome,
    VerificationReport,
    write_verification_report,
)
from ingest.promotion import (
    PromotionPreconditionError,
    create_promotion_plan,
    physical_collection_name,
)
from ingest.qdrant_store import point_id

GENERATION_ID = "gen-20260713"
VERSION_ONE = "derived:" + "1" * 64
VERSION_TWO = "derived:" + "2" * 64
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SOURCE_STATE = {
    "source": "full-corpus",
    "runs": [{"source": "matsne", "run_id": "20260713t100000z"}],
}


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _manifest(**overrides) -> GenerationManifest:
    data = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": GENERATION_ID,
        "document_count": 2,
        "indexed_document_count": 1,
        "excluded_document_count": 1,
        "chunk_count": 2,
        "sample_count": 1,
        "corpus": {"name": "georgian-legal-full", "snapshot_sha256": SHA_A},
        "source": {
            "name": "raw-crawl-state",
            "state_sha256": source_state_sha256(SOURCE_STATE),
        },
        "model": {
            "embedding_model": "BAAI/bge-m3",
            "embedding_revision": "a" * 40,
            "tokenizer_model": "BAAI/bge-m3",
            "tokenizer_revision": "b" * 40,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "reranker_revision": "c" * 40,
        },
        "vector_space": {
            "id": SHA_B,
            "dense_name": "dense",
            "dense_dimension": 1024,
            "distance": "cosine",
            "sparse_name": "sparse",
        },
        "chunking": {
            "fingerprint": SHA_C,
            "max_tokens": 450,
            "overlap_tokens": 80,
            "document_header": True,
        },
        "covered_runs": [{"source": "matsne", "run_id": "20260713t100000z"}],
        "retrieval_fingerprint_revision": 2,
        "retrieval_fingerprint": SHA_D,
        "code": {"git_sha": "c" * 40, "dirty_patch_sha256": None},
        "dependency": {"lock_sha256": SHA_E, "image_digest": None},
        "creation": {
            "created_at": "2026-07-13T12:00:00Z",
            "run_id": "build-20260713",
            "actor": "test-builder",
        },
    }
    data.update(overrides)
    return GenerationManifest.from_dict(data)


def _documents():
    return [
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "source": "matsne",
            "document_id": "doc-1",
            "version_id": VERSION_ONE,
            "source_identity": SHA_A,
            "content_hash": SHA_B,
            "document_state_hash": SHA_C,
            "expected_chunk_count": 2,
            "content_kind": "full_text",
            "content_complete": True,
            "extraction_status": "full_text",
            "article_summary": None,
            "exclusion_reason": None,
            "refresh_deadline": "2026-08-12T00:00:00Z",
            "source_binary_url": None,
        },
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "source": "matsne",
            "document_id": "doc-2",
            "version_id": VERSION_TWO,
            "source_identity": SHA_D,
            "content_hash": SHA_E,
            "document_state_hash": "f" * 64,
            "expected_chunk_count": 0,
            "content_kind": "article_summary",
            "content_complete": False,
            "extraction_status": "malformed",
            "article_summary": None,
            "exclusion_reason": "incomplete_content:article_summary:malformed",
            "refresh_deadline": "2026-07-14T00:00:00Z",
            "source_binary_url": "https://example.invalid/ruling.pdf",
        },
    ]


def _samples():
    return [
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "source": "matsne",
            "document_id": "doc-1",
            "version_id": VERSION_ONE,
            "chunk_index": 0,
            "point_id": point_id(
                "matsne", "doc-1", 0, version_id=VERSION_ONE
            ),
            "text_sha256": SHA_A,
        }
    ]


def _publish(root: Path, *, manifest=None, documents=None, samples=None):
    return publish_generation(
        root,
        GENERATION_ID,
        manifest or _manifest(),
        documents if documents is not None else _documents(),
        samples if samples is not None else _samples(),
        SOURCE_STATE,
    )


def test_generation_is_private_complete_checksummed_and_v1_untouched(tmp_path):
    root = tmp_path / "snapshots"
    v1 = root / "v1"
    v1.mkdir(parents=True)
    sentinel = v1 / "manifest.json"
    sentinel.write_text("frozen-v1", encoding="utf-8")

    destination = _publish(root)

    assert destination == root / GENERATION_ID
    assert sentinel.read_text(encoding="utf-8") == "frozen-v1"
    assert _mode(destination) == 0o700
    assert {path.name for path in destination.iterdir()} == {
        CHECKSUM_FILENAME,
        DOCUMENTS_FILENAME,
        MANIFEST_FILENAME,
        PROVENANCE_FILENAME,
        QUARANTINE_FILENAME,
        SAMPLE_CHECKS_FILENAME,
        SOURCE_STATE_FILENAME,
    }
    assert all(_mode(path) == 0o600 for path in destination.iterdir())
    assert not list(root.glob(f".{GENERATION_ID}.staging-*"))

    loaded = load_generation(destination)
    assert loaded.manifest == _manifest()
    assert len(list(loaded.iter_documents())) == 2
    assert len(list(loaded.iter_samples())) == 1
    assert set(loaded.checksums.files) == {
        DOCUMENTS_FILENAME,
        MANIFEST_FILENAME,
        PROVENANCE_FILENAME,
        QUARANTINE_FILENAME,
        SAMPLE_CHECKS_FILENAME,
        SOURCE_STATE_FILENAME,
    }

    provenance = json.loads((destination / PROVENANCE_FILENAME).read_text())
    assert provenance["generation_id"] == GENERATION_ID
    assert provenance["covered_runs"] == [
        dict(run) for run in _manifest().to_dict()["covered_runs"]
    ]
    state = json.loads((destination / SOURCE_STATE_FILENAME).read_text())
    assert state["state"] == SOURCE_STATE
    quarantine = [
        json.loads(line)
        for line in (destination / QUARANTINE_FILENAME).read_text().splitlines()
    ]
    assert quarantine == [
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "source": "matsne",
            "document_id": "doc-2",
            "version_id": VERSION_TWO,
            "source_identity": SHA_D,
            "content_hash": SHA_E,
            "document_state_hash": "f" * 64,
            "expected_chunk_count": 0,
            "content_kind": "article_summary",
            "extraction_status": "malformed",
            "article_summary": None,
            "exclusion_reason": "incomplete_content:article_summary:malformed",
            "content_complete": False,
            "refresh_deadline": "2026-07-14T00:00:00Z",
            "source_binary_url": "https://example.invalid/ruling.pdf",
        }
    ]


@pytest.mark.parametrize(
    "generation_id",
    ["v1", "v1-new-generation", "legacy", "SHORT", "../escape"],
)
def test_invalid_or_v1_generation_id_is_rejected_before_staging(
    tmp_path, generation_id
):
    root = tmp_path / "snapshots"
    with pytest.raises(GenerationFormatError):
        publish_generation(
            root,
            generation_id,
            _manifest(),
            _documents(),
            _samples(),
            SOURCE_STATE,
        )
    assert not root.exists()


def test_existing_destination_is_never_overwritten_or_consumes_input(tmp_path):
    root = tmp_path / "snapshots"
    destination = root / GENERATION_ID
    destination.mkdir(parents=True)
    sentinel = destination / "owned.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def must_not_iterate():
        raise AssertionError("existing destination must fail before input consumption")
        yield

    with pytest.raises(GenerationDestinationExists):
        publish_generation(
            root,
            GENERATION_ID,
            _manifest(),
            must_not_iterate(),
            _samples(),
            SOURCE_STATE,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_publish_rejects_existing_output_root_beneath_symlink_ancestor(tmp_path):
    real_parent = tmp_path / "real-parent"
    existing_root = real_parent / "snapshots"
    existing_root.mkdir(parents=True)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(GenerationPublishError, match="not a real directory"):
        _publish(linked_parent / "snapshots")

    assert not (existing_root / GENERATION_ID).exists()


def test_destination_race_is_atomically_rejected_without_overwrite(
    tmp_path, monkeypatch
):
    root = tmp_path / "snapshots"
    destination = root / GENERATION_ID
    original = generation_snapshot._rename_noreplace

    def raced_rename(source, target):
        assert target == destination
        target.write_text("raced-owner-data", encoding="utf-8")
        return original(source, target)

    monkeypatch.setattr(generation_snapshot, "_rename_noreplace", raced_rename)
    with pytest.raises(GenerationDestinationExists) as raised:
        _publish(root)

    assert destination.read_text(encoding="utf-8") == "raced-owner-data"
    assert raised.value.staging_path is not None
    assert raised.value.staging_path.is_dir()


def test_source_state_mismatch_fails_before_staging(tmp_path):
    root = tmp_path / "snapshots"
    with pytest.raises(GenerationFormatError, match="source_state"):
        publish_generation(
            root,
            GENERATION_ID,
            _manifest(),
            _documents(),
            _samples(),
            {"different": True},
        )
    assert not root.exists()


def test_frozen_v1_cannot_be_used_as_output_root(tmp_path):
    frozen = tmp_path / "snapshots" / "v1"
    frozen.mkdir(parents=True)
    sentinel = frozen / "manifest.json"
    sentinel.write_text("frozen", encoding="utf-8")
    with pytest.raises(GenerationPublishError, match="v1 is frozen"):
        _publish(frozen)
    assert sentinel.read_text(encoding="utf-8") == "frozen"
    assert not (frozen / GENERATION_ID).exists()


def test_failed_build_retains_private_staging_without_manifest(tmp_path):
    root = tmp_path / "snapshots"
    mismatched = _manifest(
        document_count=3,
        indexed_document_count=2,
        excluded_document_count=1,
        chunk_count=3,
    )
    with pytest.raises(GenerationPublishError, match="staging retained") as raised:
        _publish(root, manifest=mismatched)

    staging = raised.value.staging_path
    assert staging is not None and staging.is_dir()
    assert _mode(staging) == 0o700
    assert not (staging / MANIFEST_FILENAME).exists()
    assert not (root / GENERATION_ID).exists()
    assert all(_mode(path) == 0o600 for path in staging.iterdir())


def test_sample_for_excluded_document_fails_closed(tmp_path):
    samples = _samples()
    samples[0] = {
        **samples[0],
        "document_id": "doc-2",
        "version_id": VERSION_TWO,
    }
    with pytest.raises(GenerationPublishError, match="non-indexed chunk") as raised:
        _publish(tmp_path / "snapshots", samples=samples)
    assert raised.value.staging_path is not None
    assert not (tmp_path / "snapshots" / GENERATION_ID).exists()


def test_manifest_is_the_last_staged_artifact_before_atomic_rename(
    tmp_path, monkeypatch
):
    writes: list[str] = []
    original = generation_snapshot._write_bytes

    def recording_write(path, data):
        writes.append(path.name)
        return original(path, data)

    monkeypatch.setattr(generation_snapshot, "_write_bytes", recording_write)
    destination = _publish(tmp_path / "snapshots")
    assert writes[-1] == MANIFEST_FILENAME
    assert destination.is_dir()


def test_promotion_plan_requires_and_binds_four_gate_verification(tmp_path):
    root = tmp_path / "snapshots"
    destination = _publish(root)
    with pytest.raises(PromotionPreconditionError, match="verification sidecar"):
        create_promotion_plan(
            destination,
            snapshot_ref="immutable://gen.snapshot",
            snapshot_sha256=SHA_A,
            created_by="test-operator",
        )

    loaded = load_generation(destination)
    green = VerificationOutcome(ok=True, issue_count=0, examples=())
    report = VerificationReport(
        generation_id=GENERATION_ID,
        manifest_sha256=loaded.checksums.files[MANIFEST_FILENAME],
        physical_collection=physical_collection_name(GENERATION_ID),
        verified_at="2026-07-13T12:05:00Z",
        covered_runs=({"source": "matsne", "run_id": "20260713t100000z"},),
        stats={"observed_points": 2},
        coverage=green,
        integrity=green,
        freshness=green,
        quality=green,
    )
    sidecar = root / f"{GENERATION_ID}.verification.json"
    write_verification_report(sidecar, report)

    plan = create_promotion_plan(
        destination,
        snapshot_ref="immutable://gen.snapshot",
        snapshot_sha256=SHA_A,
        created_by="test-operator",
        created_at=datetime(2026, 7, 13, 12, 10, tzinfo=UTC),
        promotion_id="promotion-20260713",
    )
    assert plan.physical_collection == physical_collection_name(GENERATION_ID)
    assert plan.expected_collection.payload_schema_version == GENERATION_SCHEMA_VERSION
    assert plan.expected_collection.points_count == 2
    assert plan.expected_collection.document_header is True
    assert plan.manifest_sha256 == loaded.checksums.files[MANIFEST_FILENAME]
    assert (
        plan.verification_report_sha256
        == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    )

    (destination / DOCUMENTS_FILENAME).chmod(0o644)
    with pytest.raises(PromotionPreconditionError, match="not owner-only"):
        create_promotion_plan(
            destination,
            snapshot_ref="immutable://gen.snapshot",
            snapshot_sha256=SHA_A,
            created_by="test-operator",
        )
