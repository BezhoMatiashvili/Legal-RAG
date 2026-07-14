import hashlib
import json
import stat
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from ingest.generation import (
    CHECKSUM_FILENAME,
    DOCUMENTS_FILENAME,
    GENERATION_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    SAMPLE_CHECKS_FILENAME,
    ChecksumMismatchError,
)
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


def _manifest_data():
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": GENERATION_ID,
        "document_count": 1,
        "indexed_document_count": 1,
        "excluded_document_count": 0,
        "chunk_count": 2,
        "sample_count": 1,
        "corpus": {"name": "georgian_legal", "snapshot_sha256": "7" * 64},
        "source": {"name": "snapshot-v3", "state_sha256": "8" * 64},
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
        "code": {"git_sha": "9" * 40, "dirty_patch_sha256": None},
        "dependency": {"lock_sha256": "a" * 64, "image_digest": None},
        "creation": {
            "created_at": "2026-07-13T12:00:00Z",
            "run_id": "build-20260713",
            "actor": "release-worker",
        },
    }


def _point_id(index):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"matsne:doc-1:{index}"))


def _document_data():
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "generation_id": GENERATION_ID,
        "source": "matsne",
        "document_id": "doc-1",
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
    payload = {
        "source": "matsne",
        "document_id": "doc-1",
        "chunk_index": index,
        "text": text,
    }
    if not legacy:
        payload.update(
            {
                "schema_version": GENERATION_SCHEMA_VERSION,
                "generation_id": GENERATION_ID,
                "document_chunk_count": 2,
                "document_state_hash": STATE_HASH,
                "content_hash": CONTENT_HASH,
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


def _write_generation(root: Path) -> None:
    root.mkdir()
    _write_json(root / MANIFEST_FILENAME, _manifest_data())
    _write_json(root / DOCUMENTS_FILENAME, _document_data())
    _write_json(
        root / SAMPLE_CHECKS_FILENAME,
        {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "generation_id": GENERATION_ID,
            "source": "matsne",
            "document_id": "doc-1",
            "chunk_index": 0,
            "point_id": _point_id(0),
            "text_sha256": hashlib.sha256(b"chunk 0").hexdigest(),
        },
    )
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
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
    )


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
    _write_generation(generation)
    client = FakeReadOnlyClient([_point(0), _point(1)])

    report, report_path = verify_generation_directory(
        client,
        generation,
        "candidate",
        page_size=1,
    )

    assert report.ok
    assert report_path == sibling_report_path(generation)
    assert report_path.parent == generation.parent
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["ok"] is True
    assert (
        persisted["manifest_sha256"]
        == hashlib.sha256((generation / MANIFEST_FILENAME).read_bytes()).hexdigest()
    )
    assert [call[0] for call in client.calls] == [
        "get_collection",
        "count",
        "scroll",
        "scroll",
    ]
    for name, kwargs in client.calls[2:]:
        assert name == "scroll"
        assert kwargs["with_payload"] is True
        assert kwargs["with_vectors"] is True


def test_adapter_refuses_legacy_points_and_persists_failed_four_gate_proof(tmp_path):
    generation = tmp_path / "generation-1"
    _write_generation(generation)
    client = FakeReadOnlyClient(
        [_point(0, legacy=True), _point(1, legacy=True)],
        identity_count=0,
    )

    report, report_path = verify_generation_directory(
        client,
        generation,
        "legacy",
    )

    assert not report.ok
    assert report.coverage.ok
    assert not report.integrity.ok
    assert "missing_payload_field" in _codes(report.integrity)
    assert "identity_payload_count_mismatch" in _codes(report.integrity)
    assert json.loads(report_path.read_text(encoding="utf-8"))["ok"] is False


def test_checksum_failure_happens_before_any_collection_read_or_report(tmp_path):
    generation = tmp_path / "generation-1"
    _write_generation(generation)
    (generation / DOCUMENTS_FILENAME).write_text("{}\n", encoding="utf-8")
    client = FakeReadOnlyClient([_point(0), _point(1)])

    with pytest.raises(ChecksumMismatchError):
        verify_generation_directory(client, generation, "candidate")

    assert client.calls == []
    assert not sibling_report_path(generation).exists()


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
