import hashlib
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import release_verification
from ingest import chunk_inventory
from ingest.chunking import STRUCTURAL_CHUNKER_REVISION
from ingest.generation import (
    CANONICAL_PAYLOAD_REVISION,
    CHECKSUM_FILENAME,
    COLLECTION_DIGEST_FILENAME,
    DOCUMENTS_FILENAME,
    GENERATION_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    SAMPLE_CHECKS_FILENAME,
    ChecksumMismatchError,
)
from ingest.generation_snapshot import (
    PROVENANCE_FILENAME,
    SOURCE_STATE_FILENAME,
    source_state_sha256,
)
from ingest.integrity import (
    COLLECTION_DENSE_ENCODING,
    COLLECTION_PAYLOAD_PROJECTION,
    COLLECTION_SPARSE_ENCODING,
    point_content_sha256,
    whole_collection_sha256,
)
from ingest.qdrant_store import (
    collection_configuration,
    collection_configuration_sha256,
    point_id,
)
from ingest.promotion import physical_collection_name
from scripts.verify_generation import (
    sibling_report_path,
    stream_collection_points,
    verify_generation_directory,
)
from scripts import verify_all_embedded

GENERATION_ID = "20260713t120000z_core"
REVISION = "1" * 40
VECTOR_SPACE_ID = "2" * 64
CHUNKING_FINGERPRINT = "3" * 64
RETRIEVAL_FINGERPRINT = "4" * 64
CONTENT_HASH = "5" * 64
STATE_HASH = "6" * 64
VERSION_ID = "derived:" + "d" * 64


def _manifest_data(*, source_state_sha256_value="8" * 64):
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": GENERATION_ID,
        "document_count": 1,
        "indexed_document_count": 1,
        "excluded_document_count": 0,
        "chunk_count": 2,
        "sample_count": 1,
        "corpus": {"name": "snapshot-v3", "snapshot_sha256": "7" * 64},
        "source": {
            "name": "snapshot-v3",
            "state_sha256": source_state_sha256_value,
        },
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
        "code": {"git_sha": "9" * 40, "dirty_patch_sha256": None},
        "dependency": {"lock_sha256": "a" * 64, "image_digest": None},
        "creation": {
            "created_at": "2026-07-13T12:00:00Z",
            "run_id": "build-20260713",
            "actor": "release-worker",
        },
    }


def _point_id(index):
    return point_id("matsne", "doc-1", index, version_id=VERSION_ID)


def _document_data():
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": GENERATION_ID,
        "source": "matsne",
        "document_id": "doc-1",
        "version_id": VERSION_ID,
        "source_identity": "b" * 64,
        "content_hash": CONTENT_HASH,
        "document_state_hash": STATE_HASH,
        "expected_chunk_count": 2,
        "content_kind": "full_text",
        "content_complete": True,
        "extraction_status": "full_text",
        "article_summary": None,
        "exclusion_reason": None,
        "refresh_deadline": "2099-01-01T00:00:00Z",
        "source_binary_url": None,
    }


def _point(index, *, legacy=False):
    text = f"chunk {index}"
    char_start = index * 100
    payload = {
        "source": "matsne",
        "document_id": "doc-1",
        "chunk_index": index,
        "text": text,
    }
    if not legacy:
        passage_sha = hashlib.sha256(text.encode()).hexdigest()
        passage_id = "passage:" + hashlib.sha256(
            (
                f"matsne\0doc-1\0{VERSION_ID}\0{char_start}\0"
                f"{char_start + len(text)}\0{passage_sha}"
            ).encode("utf-8")
        ).hexdigest()
        payload.update(
            {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "canonical_payload_revision": CANONICAL_PAYLOAD_REVISION,
                "generation_id": GENERATION_ID,
                "document_chunk_count": 2,
                "document_state_hash": STATE_HASH,
                "content_hash": CONTENT_HASH,
                "canonical_content_hash": CONTENT_HASH,
                "canonical_text_exact": True,
                "passage_id": passage_id,
                "passage_hash": passage_sha,
                "passage_content_hash": passage_sha,
                "source_fingerprint": "c" * 64,
                "normalizer_revision": "canonical-source-normalizer-v2",
                "chunker_revision": "structural-article-clause-v2",
                "model_revision": REVISION,
                "article_id": None,
                "article_label": None,
                "article_start": None,
                "clause": None,
                "clause_id": None,
                "clause_ids": [],
                "subarticle": None,
                "subarticle_ids": [],
                "chapter": None,
                "heading": None,
                "heading_path": [],
                "parent_id": None,
                "structural_parent_id": None,
                "article_start_chunk_index": None,
                "parent_chunk_index": None,
                "token_count": len(text.split()),
                "char_start": char_start,
                "char_end": char_start + len(text),
                "offset_unit": "unicode_codepoint",
                "page_start": None,
                "page_end": None,
                "page_coordinate_reason": "source_not_paginated",
                "page_boundaries": [] if index == 0 else None,
                "page_boundary_mapping_sha256": None,
                "admissible": True,
                "version_id": VERSION_ID,
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
                "content_kind": "full_text",
                "content_complete": True,
                "extraction_status": "full_text",
                "article_summary": None,
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
            }
        )
    return SimpleNamespace(
        id=_point_id(index),
        payload=payload,
        vector={
            "dense": [0.1, 0.2, 0.3],
            "sparse": {"indices": [1], "values": [0.5]},
        },
    )


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_line(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_snapshot(root: Path) -> dict:
    root.mkdir()
    points = [_point(0), _point(1)]
    chunks = []
    for point in points:
        payload = point.payload
        text = payload["text"]
        chunks.append(
            {
                "chunk_index": payload["chunk_index"],
                "document_chunk_count": 2,
                "canonical_passage_sha256": payload["passage_hash"],
                "canonical_passage_char_length": len(text),
                "canonical_passage_utf8_length": len(text.encode("utf-8")),
                "token_count": payload["token_count"],
                "char_start": payload["char_start"],
                "char_end": payload["char_end"],
                "structure": {
                    "heading_path": [],
                    "article_id": None,
                    "article_label": None,
                    "article_start": None,
                    "clause": None,
                    "clause_id": None,
                    "subarticle": None,
                    "chapter": None,
                    "parent_id": None,
                    "clause_ids": [],
                    "subarticle_ids": [],
                    "article_start_chunk_index": None,
                    "parent_chunk_index": None,
                    "chunker_revision": STRUCTURAL_CHUNKER_REVISION,
                },
                "page": {
                    "page_start": None,
                    "page_end": None,
                    "page_coordinate_reason": "source_not_paginated",
                    "page_boundary_mapping_sha256": None,
                },
                "embed_input": {
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "char_length": len(text),
                    "utf8_length": len(text.encode("utf-8")),
                    "token_count": len(text.split()),
                },
            }
        )
    identity = chunk_inventory.chunk_inventory_identity(
        sources=("matsne",),
        tokenizer_model="BAAI/bge-m3",
        tokenizer_revision=REVISION,
        max_tokens=512,
        overlap_tokens=64,
        min_tokens=64,
        document_header=True,
    )
    header = {
        "schema_version": chunk_inventory.CHUNK_INVENTORY_SCHEMA_VERSION,
        "kind": chunk_inventory.CHUNK_INVENTORY_KIND,
        "format": chunk_inventory.CHUNK_INVENTORY_FORMAT,
        "identity": identity,
        "identity_sha256": chunk_inventory.unavailable_manifest_entry(identity)[
            "identity_sha256"
        ],
    }
    document = {
        "schema_version": chunk_inventory.CHUNK_INVENTORY_SCHEMA_VERSION,
        "kind": chunk_inventory.CHUNK_INVENTORY_DOCUMENT_KIND,
        "source": "matsne",
        "document_id": "doc-1",
        "version_id": VERSION_ID,
        "doc_id": f"matsne:doc-1:{VERSION_ID}",
        "canonical_content_sha256": CONTENT_HASH,
        "canonical_body_char_length": 107,
        "canonical_body_utf8_length": 107,
        "page_boundaries": [],
        "page_boundary_mapping_sha256": None,
        "page_coordinate_reason": "source_not_paginated",
        "chunk_count": 2,
        "chunks": chunks,
    }
    inventory_bytes = _canonical_line(header) + _canonical_line(document)
    inventory_path = root / chunk_inventory.CHUNK_INVENTORY_FILENAME
    inventory_path.write_bytes(inventory_bytes)
    inventory_entry = {
        "schema_version": chunk_inventory.CHUNK_INVENTORY_SCHEMA_VERSION,
        "status": chunk_inventory.CHUNK_INVENTORY_STATUS_AVAILABLE,
        "reason": None,
        "format": chunk_inventory.CHUNK_INVENTORY_FORMAT,
        "path": chunk_inventory.CHUNK_INVENTORY_FILENAME,
        "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "size_bytes": len(inventory_bytes),
        "identity": identity,
        "identity_sha256": header["identity_sha256"],
        "record_count": 2,
        "document_count": 1,
        "chunk_count": 2,
        "source_counts": {"matsne": {"document_count": 1, "chunk_count": 2}},
    }
    manifest = {
        "snapshot_id": root.name,
        "snapshot_sha256": "7" * 64,
        "corpus_sha256": "d" * 64,
        "structural_chunk_inventory": inventory_entry,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def _write_generation(root: Path) -> Path:
    snapshot_root = root.parent / "snapshot-v3"
    snapshot_manifest = _write_snapshot(snapshot_root)
    inventory = snapshot_manifest["structural_chunk_inventory"]
    inventory_projection = {
        field: inventory[field]
        for field in (
            "sha256",
            "size_bytes",
            "identity_sha256",
            "record_count",
            "document_count",
            "chunk_count",
        )
    }
    source_state = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "snapshot_id": snapshot_manifest["snapshot_id"],
        "snapshot_sha256": snapshot_manifest["snapshot_sha256"],
        "corpus_sha256": snapshot_manifest["corpus_sha256"],
        "structural_chunk_inventory": inventory_projection,
    }
    state_sha = source_state_sha256(source_state)
    manifest_data = _manifest_data(source_state_sha256_value=state_sha)
    root.mkdir()
    _write_json(root / MANIFEST_FILENAME, manifest_data)
    _write_json(root / DOCUMENTS_FILENAME, _document_data())
    _write_json(
        root / SAMPLE_CHECKS_FILENAME,
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "source": "matsne",
            "document_id": "doc-1",
            "version_id": VERSION_ID,
            "chunk_index": 0,
            "point_id": _point_id(0),
            "text_sha256": hashlib.sha256(b"chunk 0").hexdigest(),
        },
    )
    configuration = collection_configuration(_collection_info())
    point_digests = [
        (
            str(point.id),
            point_content_sha256(
                str(point.id),
                point.payload,
                point.vector,
                dense_name="dense",
                sparse_name="sparse",
            ),
        )
        for point in (_point(0), _point(1))
    ]
    whole_sha, point_count = whole_collection_sha256(sorted(point_digests))
    _write_json(
        root / COLLECTION_DIGEST_FILENAME,
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "digest_revision": 1,
            "algorithm": "sha256",
            "point_count": point_count,
            "collection_sha256": whole_sha,
            "physical_collection": physical_collection_name(GENERATION_ID),
            "payload_projection": COLLECTION_PAYLOAD_PROJECTION,
            "dense_encoding": COLLECTION_DENSE_ENCODING,
            "sparse_encoding": COLLECTION_SPARSE_ENCODING,
            "collection_configuration_sha256": collection_configuration_sha256(
                configuration
            ),
            "collection_configuration": configuration,
            "vector_checksum_artifact_sha256": "e" * 64,
            "vector_probe_sha256": "f" * 64,
            "embed_binding_sha256": "0" * 64,
        },
    )
    _write_json(
        root / SOURCE_STATE_FILENAME,
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "source": manifest_data["source"]["name"],
            "state_sha256": state_sha,
            "state": source_state,
        },
    )
    _write_json(
        root / PROVENANCE_FILENAME,
        {
            "corpus": manifest_data["corpus"],
            "preparation": {
                "evidence": {
                    "snapshot": {
                        "snapshot_id": snapshot_manifest["snapshot_id"],
                        "snapshot_sha256": snapshot_manifest["snapshot_sha256"],
                        "corpus_sha256": snapshot_manifest["corpus_sha256"],
                        "structural_chunk_inventory": inventory_projection,
                    }
                }
            },
        },
    )
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            MANIFEST_FILENAME,
            DOCUMENTS_FILENAME,
            SAMPLE_CHECKS_FILENAME,
            COLLECTION_DIGEST_FILENAME,
            SOURCE_STATE_FILENAME,
            PROVENANCE_FILENAME,
        )
    }
    _write_json(
        root / CHECKSUM_FILENAME,
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "algorithm": "sha256",
            "files": files,
        },
    )
    return snapshot_root


def _collection_info():
    return SimpleNamespace(
        points_count=2,
        config=SimpleNamespace(
            params=SimpleNamespace(
                vectors={
                    "dense": SimpleNamespace(size=3, distance="Cosine"),
                },
                sparse_vectors={"sparse": SimpleNamespace()},
            )
        ),
        payload_schema={},
    )


@pytest.fixture(autouse=True)
def _stub_sealed_snapshot_verifier(monkeypatch):
    def verify(root):
        return json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))

    monkeypatch.setattr("ingest.integrity.verify_sealed_snapshot", verify)


class FakeReadOnlyClient:
    def __init__(self, points, *, identity_count=None):
        self.points = points
        self.identity_count = len(points) if identity_count is None else identity_count
        self.calls = []

    def get_collection(self, collection_name):
        self.calls.append(("get_collection", collection_name))
        return _collection_info()

    def count(self, **kwargs):
        self.calls.append(("count", kwargs))
        return SimpleNamespace(count=self.identity_count)

    def scroll(self, **kwargs):
        self.calls.append(("scroll", kwargs))
        start = kwargs["offset"] or 0
        end = min(start + kwargs["limit"], len(self.points))
        next_offset = end if end < len(self.points) else None
        return self.points[start:end], next_offset


def _codes(outcome):
    return {example["code"] for example in outcome.examples}


def test_adapter_streams_vectors_and_payloads_then_writes_owner_only_sibling(tmp_path):
    generation = tmp_path / "generation-1"
    snapshot_root = _write_generation(generation)
    client = FakeReadOnlyClient([_point(0), _point(1)])

    report, report_path = verify_generation_directory(
        client,
        generation,
        physical_collection_name(GENERATION_ID),
        snapshot_root=snapshot_root,
        page_size=1,
        verification_id="verify-a",
        count_tokens=lambda text: len(text.split()),
    )

    assert report.ok
    assert report_path == sibling_report_path(generation, "verify-a")
    assert report_path.parent == generation.parent
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["ok"] is True
    assert persisted["physical_collection"] == physical_collection_name(GENERATION_ID)
    proof = persisted["stats"]["structural_inventory_proof"]
    assert proof["artifact_exhausted"] is True
    assert proof["chunk_rows"] == 2
    assert proof["exact_point_matches"] == 2
    assert proof["embed_inputs_verified"] == 2
    assert (
        persisted["manifest_sha256"]
        == hashlib.sha256((generation / MANIFEST_FILENAME).read_bytes()).hexdigest()
    )
    assert [call[0] for call in client.calls] == [
        "get_collection",
        "count",
        "get_collection",
        "scroll",
        "scroll",
    ]
    for name, kwargs in client.calls[3:]:
        assert name == "scroll"
        assert kwargs["with_payload"] is True
        assert kwargs["with_vectors"] is True

    second_report, second_path = verify_generation_directory(
        FakeReadOnlyClient([_point(0), _point(1)]),
        generation,
        physical_collection_name(GENERATION_ID),
        snapshot_root=snapshot_root,
        verification_id="verify-b",
        count_tokens=lambda text: len(text.split()),
    )
    assert second_report.ok
    assert second_report.stats["structural_inventory_proof"][
        "structural_inventory_sha256"
    ] == proof["structural_inventory_sha256"]
    assert second_report.stats["structural_inventory_proof"][
        "artifact_exhausted"
    ] is True
    assert second_path != report_path
    assert second_path.exists()
    with pytest.raises(FileExistsError, match="will not be replaced"):
        verify_generation_directory(
            FakeReadOnlyClient([_point(0), _point(1)]),
            generation,
            physical_collection_name(GENERATION_ID),
            snapshot_root=snapshot_root,
            verification_id="verify-a",
            count_tokens=lambda text: len(text.split()),
        )


def test_release_pair_loader_requires_two_exact_clean_run_specific_reports(
    tmp_path, monkeypatch
):
    generation = tmp_path / GENERATION_ID
    snapshot_root = _write_generation(generation)
    monkeypatch.setattr(release_verification, "GENERATION_ID", GENERATION_ID)
    monkeypatch.setattr(
        release_verification,
        "PHYSICAL_COLLECTION",
        physical_collection_name(GENERATION_ID),
    )
    paths = []
    for verification_id in release_verification.VERIFICATION_IDS:
        report, path = verify_generation_directory(
            FakeReadOnlyClient([_point(0), _point(1)]),
            generation,
            physical_collection_name(GENERATION_ID),
            snapshot_root=snapshot_root,
            verification_id=verification_id,
            count_tokens=lambda text: len(text.split()),
        )
        assert report.ok
        paths.append(path)

    pair = release_verification.validate_physical_verification_pair(
        generation, (paths[0], paths[1])
    )
    assert pair.paths == (paths[0].absolute(), paths[1].absolute())
    assert pair.sha256[0] != pair.sha256[1]
    assert all(len(value) == 64 for value in pair.sha256)


def test_release_pair_loader_rejects_permission_or_structural_proof_drift(
    tmp_path, monkeypatch
):
    generation = tmp_path / GENERATION_ID
    snapshot_root = _write_generation(generation)
    monkeypatch.setattr(release_verification, "GENERATION_ID", GENERATION_ID)
    monkeypatch.setattr(
        release_verification,
        "PHYSICAL_COLLECTION",
        physical_collection_name(GENERATION_ID),
    )
    paths = [
        verify_generation_directory(
            FakeReadOnlyClient([_point(0), _point(1)]),
            generation,
            physical_collection_name(GENERATION_ID),
            snapshot_root=snapshot_root,
            verification_id=verification_id,
            count_tokens=lambda text: len(text.split()),
        )[1]
        for verification_id in release_verification.VERIFICATION_IDS
    ]
    paths[0].chmod(0o640)
    with pytest.raises(
        release_verification.ReleaseVerificationError, match="owner-only"
    ):
        release_verification.validate_physical_verification_pair(
            generation, (paths[0], paths[1])
        )

    paths[0].chmod(0o600)
    value = json.loads(paths[1].read_text(encoding="utf-8"))
    value["stats"]["structural_inventory_proof"]["exact_point_matches"] -= 1
    paths[1].write_text(json.dumps(value), encoding="utf-8")
    paths[1].chmod(0o600)
    with pytest.raises(
        release_verification.ReleaseVerificationError,
        match="structural inventory join is incomplete",
    ):
        release_verification.validate_physical_verification_pair(
            generation, (paths[0], paths[1])
        )


def test_adapter_refuses_legacy_points_and_persists_failed_four_gate_proof(tmp_path):
    generation = tmp_path / "generation-1"
    snapshot_root = _write_generation(generation)
    client = FakeReadOnlyClient(
        [_point(0, legacy=True), _point(1, legacy=True)],
        identity_count=0,
    )

    report, report_path = verify_generation_directory(
        client,
        generation,
        physical_collection_name(GENERATION_ID),
        snapshot_root=snapshot_root,
        verification_id="verify-failed",
        count_tokens=lambda text: len(text.split()),
    )

    assert not report.ok
    assert not report.coverage.ok
    assert not report.integrity.ok
    assert "missing_payload_field" in _codes(report.integrity)
    assert "identity_payload_count_mismatch" in _codes(report.integrity)
    assert json.loads(report_path.read_text(encoding="utf-8"))["ok"] is False


def test_physical_verifier_detects_exact_vector_byte_mutation(tmp_path):
    generation = tmp_path / "generation-1"
    snapshot_root = _write_generation(generation)
    points = [_point(0), _point(1)]
    points[1].vector["dense"][0] += 0.0001

    report, _ = verify_generation_directory(
        FakeReadOnlyClient(points),
        generation,
        physical_collection_name(GENERATION_ID),
        snapshot_root=snapshot_root,
        verification_id="verify-vector-mutation",
        count_tokens=lambda text: len(text.split()),
    )

    assert not report.ok
    assert "whole_collection_digest_mismatch" in _codes(report.integrity)


def test_physical_sidecar_rejects_same_count_altered_structural_boundary(tmp_path):
    generation = tmp_path / "generation-1"
    snapshot_root = _write_generation(generation)
    points = [_point(0), _point(1)]
    payload = points[0].payload
    payload["char_start"] += 1
    payload["char_end"] += 1
    payload["passage_id"] = "passage:" + hashlib.sha256(
        (
            f"matsne\0doc-1\0{VERSION_ID}\0{payload['char_start']}\0"
            f"{payload['char_end']}\0{payload['passage_hash']}"
        ).encode("utf-8")
    ).hexdigest()

    report, report_path = verify_generation_directory(
        FakeReadOnlyClient(points),
        generation,
        physical_collection_name(GENERATION_ID),
        snapshot_root=snapshot_root,
        verification_id="verify-altered-boundary",
        count_tokens=lambda text: len(text.split()),
    )

    assert not report.ok
    assert "structural_inventory_payload_mismatch" in _codes(report.integrity)
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    proof = persisted["stats"]["structural_inventory_proof"]
    assert proof["artifact_exhausted"] is True
    assert proof["chunk_rows"] == 2
    assert proof["exact_point_matches"] == 1


def test_physical_verifier_must_exhaust_inventory_before_writing_sidecar(tmp_path):
    generation = tmp_path / "generation-1"
    snapshot_root = _write_generation(generation)
    inventory_path = snapshot_root / chunk_inventory.CHUNK_INVENTORY_FILENAME
    header_raw, document_raw = inventory_path.read_bytes().splitlines()
    document = json.loads(document_raw)
    document["chunks"][1]["embed_input"]["token_count"] += 1
    inventory_path.write_bytes(header_raw + b"\n" + _canonical_line(document))
    client = FakeReadOnlyClient([_point(0), _point(1)])

    with pytest.raises(
        chunk_inventory.ChunkInventoryError,
        match="artifact mismatch",
    ):
        verify_generation_directory(
            client,
            generation,
            physical_collection_name(GENERATION_ID),
            snapshot_root=snapshot_root,
            verification_id="verify-unexhausted-inventory",
            count_tokens=lambda text: len(text.split()),
        )

    assert all(call[0] != "scroll" for call in client.calls)
    assert not sibling_report_path(
        generation, "verify-unexhausted-inventory"
    ).exists()


def test_checksum_failure_happens_before_any_collection_read_or_report(tmp_path):
    generation = tmp_path / "generation-1"
    snapshot_root = _write_generation(generation)
    (generation / DOCUMENTS_FILENAME).write_text("{}\n", encoding="utf-8")
    client = FakeReadOnlyClient([_point(0), _point(1)])

    with pytest.raises(ChecksumMismatchError):
        verify_generation_directory(
            client,
            generation,
            physical_collection_name(GENERATION_ID),
            snapshot_root=snapshot_root,
            verification_id="verify-checksum-failed",
            count_tokens=lambda text: len(text.split()),
        )

    assert client.calls == []
    assert not sibling_report_path(generation, "verify-checksum-failed").exists()


def test_verification_rejects_alias_or_wrong_collection_before_qdrant_read(tmp_path):
    generation = tmp_path / "generation-1"
    snapshot_root = _write_generation(generation)
    client = FakeReadOnlyClient([_point(0), _point(1)])

    with pytest.raises(ValueError, match="exact physical collection"):
        verify_generation_directory(
            client,
            generation,
            "georgian_legal",
            snapshot_root=snapshot_root,
            verification_id="verify-wrong-target",
            count_tokens=lambda text: len(text.split()),
        )

    assert client.calls == []
    assert not sibling_report_path(generation, "verify-wrong-target").exists()


def test_scroll_rejects_nonadvancing_or_empty_continuations():
    class BrokenClient:
        def scroll(self, **kwargs):
            return [], "still-more"

    with pytest.raises(RuntimeError, match="empty page"):
        list(stream_collection_points(BrokenClient(), "candidate"))


def test_legacy_coverage_checker_requires_explicit_opt_in(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["verify_all_embedded.py"])

    with pytest.raises(SystemExit) as exc:
        verify_all_embedded.main()

    assert exc.value.code == 2
    assert "--coverage-only" in capsys.readouterr().err
