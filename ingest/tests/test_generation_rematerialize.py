from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import models

from ingest import generation_rematerialize as remat
from ingest.chunking import chunk_document
from ingest.config import load_config
from ingest.pipeline import _document_state_hash
from ingest.qdrant_store import build_payload, point_id
from ingest.release_inputs import GENERATION_ID as FROZEN_CANDIDATE_GENERATION_ID
from ingest.sources import CanonicalDoc, derived_version_id
from scripts import rematerialize_generation as rematerialize_cli


def _report(**updates):
    sha = "a" * 64
    value = {
        "schema_version": 1,
        "kind": remat.REPORT_KIND,
        "created_at": "2026-07-17T00:00:00Z",
        "binding_sha256": sha,
        "source_collection": "georgian_legal_delta_remat_run123",
        "target_collection": "georgian_legal__gen_generation123",
        "generation_id": "generation123",
        "document_count": 1,
        "point_count": 1,
        "source_logical_vector_sha256": sha,
        "target_logical_vector_sha256": sha,
        "old_id_set_sha256": sha,
        "new_id_set_sha256": sha,
        "rekey_map_sha256": sha,
        "target_configuration_sha256": sha,
        "retrieval_fingerprint": sha,
        "court_extractor_revision": "court-extract-v1",
        "provenance_strength": "legacy-empirically-attested",
    }
    value.update(updates)
    return value


def test_logical_vector_digest_is_id_independent_and_float32_stable():
    one = remat.logical_vector_sha256("ecd\tdoc\tversion\t0", [1, 2], [9], [0.25])
    two = remat.logical_vector_sha256("ecd\tdoc\tversion\t0", [1.0, 2.0], [9], [0.25])
    changed = remat.logical_vector_sha256("ecd\tdoc\tversion\t1", [1, 2], [9], [0.25])
    assert one == two
    assert one != changed


def test_validate_rematerialization_report_rejects_vector_mismatch(tmp_path: Path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    value, file_sha = remat.validate_rematerialization_report(
        path,
        generation_id="generation123",
        physical_collection="georgian_legal__gen_generation123",
    )
    assert value["point_count"] == 1
    assert len(file_sha) == 64

    path.write_text(
        json.dumps(_report(target_logical_vector_sha256="b" * 64)),
        encoding="utf-8",
    )
    with pytest.raises(remat.RematerializationError, match="identity is invalid"):
        remat.validate_rematerialization_report(
            path,
            generation_id="generation123",
            physical_collection="georgian_legal__gen_generation123",
        )


def test_source_evidence_rejects_mutable_model_revision(tmp_path: Path):
    snapshot = tmp_path / "legacy.snapshot"
    snapshot.write_bytes(b"qdrant")
    checksum = remat._file_sha256(snapshot)
    value = {
        "schema_version": 1,
        "kind": remat.SOURCE_EVIDENCE_KIND,
        "snapshot_path": str(snapshot),
        "snapshot_sha256": checksum,
        "source_collection": "legacy",
        "source_points_count": 1,
        "source_configuration_sha256": "a" * 64,
        "legacy_id_scheme": "uuid5-source-document-chunk-v1",
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "main",
        "tokenizer_model": "BAAI/bge-m3",
        "tokenizer_revision": "1" * 40,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_revision": "2" * 40,
        "dense_name": "dense",
        "dense_dimension": 1024,
        "sparse_name": "sparse",
        "chunk_tokens": 512,
        "chunk_overlap": 80,
        "chunk_min_tokens": 64,
        "document_header": False,
        "retrieval_fingerprint": "b" * 64,
        "vector_checksum_artifact_sha256": "c" * 64,
        "vector_probe_sha256": "d" * 64,
    }
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(remat.RematerializationError, match="immutable hexadecimal"):
        remat.load_source_evidence(evidence)


def test_frozen_candidate_rematerialization_refuses_before_inputs_or_client(tmp_path):
    class NoClientAccess:
        def __getattr__(self, name):
            pytest.fail(f"frozen-candidate refusal must precede client access: {name}")

    with pytest.raises(remat.RematerializationError, match="cannot be rematerialized"):
        remat.rematerialize_generation(
            NoClientAccess(),
            None,
            source_evidence_path=tmp_path / "missing-evidence.json",
            snapshot_root=tmp_path / "missing-snapshot",
            generation_id=FROZEN_CANDIDATE_GENERATION_ID,
            vector_checksum_path=tmp_path / "missing-checksum.json",
            actor="operator",
            run_id="run-1",
            state_dir=tmp_path / "state",
        )


def test_rematerialization_cli_refuses_frozen_candidate_before_config(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rematerialize_cli,
        "load_config",
        lambda: pytest.fail("frozen-candidate refusal must precede configuration/client access"),
    )
    with pytest.raises(SystemExit, match="cannot be rematerialized"):
        rematerialize_cli.main(
            [
                "production",
                "--source-evidence",
                str(tmp_path / "missing-evidence.json"),
                "--snapshot",
                str(tmp_path / "missing-snapshot"),
                "--generation-id",
                FROZEN_CANDIDATE_GENERATION_ID,
                "--vector-checksum",
                str(tmp_path / "missing-checksum.json"),
                "--actor",
                "operator",
                "--run-id",
                "run-1",
            ]
        )


class _FakeClient:
    def __init__(self, source_name, source_points):
        self.collections = {source_name: {str(point.id): point for point in source_points}}
        self.write_targets = []

    def collection_exists(self, name):
        return name in self.collections

    def get_aliases(self):
        return SimpleNamespace(aliases=[])

    def retrieve(self, *, collection_name, ids, **_kwargs):
        table = self.collections[collection_name]
        return [table[str(point_id)] for point_id in ids if str(point_id) in table]

    def scroll(self, *, collection_name, offset=None, limit=256, **_kwargs):
        values = sorted(self.collections[collection_name].values(), key=lambda point: str(point.id))
        start = int(offset or 0)
        page = values[start : start + limit]
        next_offset = start + len(page) if start + len(page) < len(values) else None
        return page, next_offset

    def upsert(self, *, collection_name, points, wait):
        assert wait is True
        self.write_targets.append(collection_name)
        table = self.collections[collection_name]
        for point in points:
            table[str(point.id)] = SimpleNamespace(
                id=point.id, payload=point.payload, vector=point.vector
            )

    def get_collection(self, name):
        return SimpleNamespace(points_count=len(self.collections[name]))


def _doc() -> CanonicalDoc:
    body = (
        "მოსამართლეები: ლაშა ქოჩიაშვილი\n\n"
        "გ ა დ ა წ ყ ვ ი ტ ა:\n1. საჩივარი არ დაკმაყოფილდეს."
    )
    doc = CanonicalDoc(
        source="ecd",
        document_id="doc-1",
        title="საქმე",
        date="2026-01-01",
        date_raw="2026-01-01",
        language="ka",
        document_type="decision",
        court="თბილისის სააპელაციო სასამართლო",
        source_url="https://example.test/doc-1",
        document_number="1",
        registration_code=None,
        parties=None,
        status=None,
        status_raw=None,
        in_force_date=None,
        expiry_date=None,
        body_markdown=body,
        extra={},
        source_fingerprint="1" * 64,
        official_url="https://example.test/doc-1",
    )
    return dataclasses.replace(doc, version_id=derived_version_id(doc))


def test_scratch_copy_preserves_ids_and_never_writes_source(tmp_path: Path, monkeypatch):
    cfg = dataclasses.replace(
        load_config(),
        dense_dim=2,
        chunk_tokens=512,
        chunk_overlap=0,
        chunk_min_tokens=1,
        generation_id=None,
        production_mode=False,
    )
    doc = _doc()
    chunks = chunk_document(
        doc.body_markdown,
        max_tokens=512,
        overlap=0,
        min_tokens=1,
        count_tokens=lambda value: len(value.split()),
    )
    assert len(chunks) == 1
    state_hash = _document_state_hash(cfg, doc=doc)
    payload = build_payload(
        doc,
        chunks[0],
        document_chunk_count=1,
        document_state_hash=state_hash,
    )
    legacy_id = point_id("ecd", "doc-1", 0)
    source = "georgian_legal"
    source_point = SimpleNamespace(
        id=legacy_id,
        payload=payload,
        vector={
            "dense": [0.25, -0.5],
            "sparse": models.SparseVector(indices=[7], values=[0.75]),
        },
    )
    client = _FakeClient(source, [source_point])
    target = "georgian_legal_delta_court_unit_run"

    def fake_ensure(_client, target_cfg, **_kwargs):
        assert target_cfg.collection_name == target
        client.collections[target] = {}
        return True

    monkeypatch.setattr(remat, "ensure_collection", fake_ensure)
    monkeypatch.setattr(remat, "collection_configuration", lambda _info: {"dense": 2})
    monkeypatch.setattr(remat, "refuse_aliased_write_target", lambda *_args: None)
    binding = {
        "schema_version": 1,
        "kind": remat.BINDING_KIND,
        "mode": "scratch",
        "run_id": "unit_run",
        "source_collection": source,
        "target_collection": target,
        "sources": ["ecd"],
        "retrieval_fingerprint": "a" * 64,
        "court_extractor_revision": "court-extract-v1",
        "provenance_strength": "live-read-validation-only",
        "plan": None,
    }
    result = remat._run_copy(
        client,
        cfg,
        source_collection=source,
        target_collection=target,
        sources=["ecd"],
        docs=lambda: iter([doc]),
        binding=binding,
        run_id="unit_run",
        state_dir=tmp_path,
        batch_size=10,
        resume=False,
        apply=True,
        allow_run_scoped_delta=True,
        count_tokens=lambda value: len(value.split()),
        environ={"QDRANT_WRITE_APPROVED": "1"},
    )
    assert result.point_count == 1
    assert set(client.collections[target]) == {legacy_id}
    assert client.collections[target][legacy_id].vector["dense"] == [0.25, -0.5]
    assert client.write_targets == [target]
    assert result.source_logical_vector_sha256 == result.target_logical_vector_sha256
