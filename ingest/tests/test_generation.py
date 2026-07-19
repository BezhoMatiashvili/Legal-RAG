import hashlib
import json
from pathlib import Path

import pytest

from ingest.generation import (
    CHECKSUM_FILENAME,
    DOCUMENTS_FILENAME,
    GENERATION_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    SAMPLE_CHECKS_FILENAME,
    ChecksumInventory,
    ChecksumMismatchError,
    DocumentRecord,
    GenerationFormatError,
    GenerationManifest,
    load_generation,
    load_manifest,
    validate_generation_id,
)
from ingest.qdrant_store import point_id

GENERATION_ID = "20260713t120000z_core"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
REVISION = "d" * 40
VERSION_ID = "derived:" + "f" * 64
POINT_ID = point_id("matsne", "doc-1", 0, version_id=VERSION_ID)


def manifest_data(**overrides):
    value = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": GENERATION_ID,
        "document_count": 2,
        "indexed_document_count": 1,
        "excluded_document_count": 1,
        "chunk_count": 2,
        "sample_count": 1,
        "corpus": {"name": "georgian_legal", "snapshot_sha256": HASH_A},
        "source": {"name": "snapshot-v3", "state_sha256": HASH_B},
        "model": {
            "embedding_model": "BAAI/bge-m3",
            "embedding_revision": REVISION,
            "tokenizer_model": "BAAI/bge-m3",
            "tokenizer_revision": REVISION,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "reranker_revision": REVISION,
        },
        "vector_space": {
            "id": HASH_C,
            "dense_name": "dense",
            "dense_dimension": 3,
            "distance": "cosine",
            "sparse_name": "sparse",
        },
        "chunking": {
            "fingerprint": HASH_A,
            "max_tokens": 512,
            "overlap_tokens": 64,
            "document_header": True,
        },
        "covered_runs": [
            {"source": "matsne", "run_id": "20260713t100000z"},
            {"source": "tas", "run_id": "20260713t110000z"},
        ],
        "retrieval_fingerprint_revision": 2,
        "retrieval_fingerprint": HASH_B,
        "code": {"git_sha": "e" * 40, "dirty_patch_sha256": None},
        "dependency": {"lock_sha256": HASH_C, "image_digest": None},
        "creation": {
            "created_at": "2026-07-13T12:00:00Z",
            "run_id": "build-20260713",
            "actor": "release-worker",
        },
    }
    value.update(overrides)
    return value


def document_data(*, excluded=False):
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": GENERATION_ID,
        "source": "matsne",
        "document_id": "doc-1" if not excluded else "doc-excluded",
        "version_id": VERSION_ID if not excluded else "derived:" + "e" * 64,
        "source_identity": HASH_A if not excluded else HASH_B,
        "content_hash": HASH_B,
        "document_state_hash": HASH_C,
        "expected_chunk_count": 0 if excluded else 2,
        "content_kind": "source_binary" if excluded else "full_text",
        "content_complete": not excluded,
        "extraction_status": "scanned_no_text" if excluded else "full_text",
        "article_summary": None,
        "exclusion_reason": "scanned_no_text" if excluded else None,
        "refresh_deadline": "2026-08-12T12:00:00Z",
        "source_binary_url": None,
    }


def sample_data():
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": GENERATION_ID,
        "source": "matsne",
        "document_id": "doc-1",
        "version_id": VERSION_ID,
        "chunk_index": 0,
        "point_id": POINT_ID,
        "text_sha256": HASH_A,
    }


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_generation(root: Path) -> None:
    root.mkdir()
    _write_json(root / MANIFEST_FILENAME, manifest_data())
    docs = [document_data(), document_data(excluded=True)]
    (root / DOCUMENTS_FILENAME).write_text(
        "".join(json.dumps(doc, sort_keys=True) + "\n" for doc in docs),
        encoding="utf-8",
    )
    (root / SAMPLE_CHECKS_FILENAME).write_text(
        json.dumps(sample_data(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = {
        name: _sha256(root / name)
        for name in (MANIFEST_FILENAME, DOCUMENTS_FILENAME, SAMPLE_CHECKS_FILENAME)
    }
    _write_json(
        root / CHECKSUM_FILENAME,
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "algorithm": "sha256",
            "files": files,
        },
    )


def test_load_generation_verifies_and_streams_strict_artifacts(tmp_path):
    root = tmp_path / "generation"
    write_generation(root)

    artifacts = load_generation(root)

    assert artifacts.manifest.generation_id == GENERATION_ID
    assert [(run.source, run.run_id) for run in artifacts.manifest.covered_runs] == [
        ("matsne", "20260713t100000z"),
        ("tas", "20260713t110000z"),
    ]
    documents = list(artifacts.iter_documents())
    assert [record.indexed for record in documents] == [True, False]
    assert documents[0].expected_chunk_count == 2
    assert list(artifacts.iter_samples())[0].point_id == POINT_ID


def test_checksum_verification_rejects_tampering_and_unexplained_files(tmp_path):
    root = tmp_path / "generation"
    write_generation(root)
    with (root / DOCUMENTS_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(ChecksumMismatchError, match="checksum mismatch"):
        load_generation(root)

    write_generation(tmp_path / "other")
    (tmp_path / "other" / "untracked.txt").write_text(
        "not inventoried", encoding="utf-8"
    )
    with pytest.raises(ChecksumMismatchError, match="unexplained"):
        load_generation(tmp_path / "other")


def test_manifest_rejects_unknown_duplicate_and_mutable_identity_fields(tmp_path):
    unknown = manifest_data(unexpected=True)
    _write_json(tmp_path / "unknown.json", unknown)
    with pytest.raises(GenerationFormatError, match="unknown"):
        load_manifest(tmp_path / "unknown.json")

    duplicate = json.dumps(manifest_data())
    duplicate = duplicate[:-1] + ', "generation_id": "other_generation"}'
    (tmp_path / "duplicate.json").write_text(duplicate, encoding="utf-8")
    with pytest.raises(GenerationFormatError, match="duplicate JSON key"):
        load_manifest(tmp_path / "duplicate.json")

    mutable = manifest_data()
    mutable["model"]["embedding_revision"] = "main"
    with pytest.raises(GenerationFormatError, match="immutable"):
        GenerationManifest.from_dict(mutable)

    missing_model = manifest_data()
    del missing_model["model"]["tokenizer_model"]
    with pytest.raises(GenerationFormatError, match="missing"):
        GenerationManifest.from_dict(missing_model)

    mutable_reranker = manifest_data()
    mutable_reranker["model"]["reranker_revision"] = "latest"
    with pytest.raises(GenerationFormatError, match="immutable"):
        GenerationManifest.from_dict(mutable_reranker)

    blank_model = manifest_data()
    blank_model["model"]["reranker_model"] = "   "
    with pytest.raises(GenerationFormatError, match="non-empty model identity"):
        GenerationManifest.from_dict(blank_model)


def test_manifest_requires_canonical_covered_runs_and_exact_counts():
    legacy = manifest_data(schema_version=GENERATION_SCHEMA_VERSION - 1)
    with pytest.raises(GenerationFormatError, match="unsupported generation schema_version"):
        GenerationManifest.from_dict(legacy)

    missing_runs = manifest_data(covered_runs=[])
    with pytest.raises(GenerationFormatError, match="covered raw run"):
        GenerationManifest.from_dict(missing_runs)

    unsorted_runs = manifest_data(
        covered_runs=[
            {"source": "tas", "run_id": "b"},
            {"source": "matsne", "run_id": "a"},
        ]
    )
    with pytest.raises(GenerationFormatError, match="must be sorted"):
        GenerationManifest.from_dict(unsorted_runs)

    wrong_counts = manifest_data(document_count=3)
    with pytest.raises(GenerationFormatError, match="must equal"):
        GenerationManifest.from_dict(wrong_counts)


def test_public_generation_id_validator_is_safe_for_paths_and_collections():
    assert validate_generation_id(GENERATION_ID) == GENERATION_ID
    for invalid in ("v1", "legacy", "../escape", "UPPER_CASE", "short"):
        with pytest.raises(GenerationFormatError):
            validate_generation_id(invalid)


def test_document_and_checksum_path_semantics_fail_closed():
    included_without_chunks = document_data()
    included_without_chunks["expected_chunk_count"] = 0
    with pytest.raises(GenerationFormatError, match="indexed documents"):
        DocumentRecord.from_dict(included_without_chunks)

    excluded_with_chunks = document_data(excluded=True)
    excluded_with_chunks["expected_chunk_count"] = 1
    with pytest.raises(GenerationFormatError, match="excluded documents"):
        DocumentRecord.from_dict(excluded_with_chunks)

    with pytest.raises(GenerationFormatError, match="unsafe artifact path"):
        ChecksumInventory.from_dict(
            {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "algorithm": "sha256",
                "files": {
                    MANIFEST_FILENAME: HASH_A,
                    DOCUMENTS_FILENAME: HASH_A,
                    SAMPLE_CHECKS_FILENAME: HASH_A,
                    "../escape": HASH_A,
                },
            }
        )
