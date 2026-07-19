from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import models

from ingest import generation_rematerialize as remat
from ingest.config import load_config
from ingest.qdrant_store import point_id
from ingest.sources import COURT_CANONICAL_FIELDS


class _FakeClient:
    def __init__(self, source_name: str, source_points: list[SimpleNamespace]):
        self.collections = {
            source_name: {str(point.id): point for point in source_points}
        }
        self.created_targets: list[str] = []
        self.upsert_targets: list[str] = []

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def get_aliases(self):
        return SimpleNamespace(aliases=[])

    def retrieve(self, *, collection_name: str, ids, **_kwargs):
        table = self.collections[collection_name]
        return [table[str(point_id_value)] for point_id_value in ids if str(point_id_value) in table]

    def scroll(self, *, collection_name: str, offset=None, limit=256, **_kwargs):
        values = sorted(
            self.collections[collection_name].values(), key=lambda point: str(point.id)
        )
        start = int(offset or 0)
        page = values[start : start + limit]
        next_offset = start + len(page) if start + len(page) < len(values) else None
        return page, next_offset

    def upsert(self, *, collection_name: str, points, wait: bool):
        assert wait is True
        self.upsert_targets.append(collection_name)
        table = self.collections[collection_name]
        for point in points:
            table[str(point.id)] = SimpleNamespace(
                id=point.id,
                payload=point.payload,
                vector=point.vector,
            )

    def get_collection(self, name: str):
        return SimpleNamespace(points_count=len(self.collections[name]))


def _cfg():
    # These settings cannot reproduce the deliberately legacy-shaped chunks below.
    # Scratch overlay must copy their boundaries and never consult a tokenizer.
    return dataclasses.replace(
        load_config(),
        dense_dim=2,
        chunk_tokens=1,
        chunk_overlap=0,
        chunk_min_tokens=1,
        generation_id=None,
        production_mode=False,
    )


def _write_jsonl(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _ecd_raw(document_id: str = "ecd-1") -> dict:
    return {
        "decision_document_id": document_id,
        "case_no": "საქმე-1",
        "court_name": "თბილისის სააპელაციო სასამართლო",
        "decision_date": "2026-01-01",
        "body_markdown": (
            "მოსამართლეები: ლაშა ქოჩიაშვილი\n\n"
            "გ ა დ ა წ ყ ვ ი ტ ა:\n1. საჩივარი არ დაკმაყოფილდეს."
        ),
    }


def _supreme_raw() -> dict:
    return {
        "case_id": "sc-1",
        "chamber": "სამოქალაქო საქმეთა პალატა",
        "case_number": "ას-1-2026",
        "date": "2026-01-02",
        "result": "გაუქმდა და მიღებულია ახალი გადაწყვეტილება",
        "appeal_type": "საკასაციო საჩივარი",
        "body_markdown": (
            "თავმჯდომარე: ნუგზარ სხირტლაძე\n"
            "მოსამართლეები - ლაშა ქოჩიაშვილი, პ. სილაგაძე\n\n"
            "დ ა ა დ გ ი ნ ა:\n1. საჩივარი არ დაკმაყოფილდეს."
        ),
    }


def _payload(
    overlay,
    *,
    chunk_index: int,
    chunk_count: int,
    legacy_text: str,
) -> dict:
    return {
        "source": overlay.source,
        "document_id": overlay.document_id,
        "chunk_index": chunk_index,
        "document_chunk_count": chunk_count,
        "content_hash": overlay.content_hash,
        "canonical_content_hash": overlay.content_hash,
        "source_fingerprint": overlay.source_fingerprint,
        "version_id": overlay.version_id,
        "text": legacy_text,
        "char_start": 17 + chunk_index,
        "char_end": 41 + chunk_index,
        "heading_path": ["legacy", str(chunk_index)],
        "token_count": 777,
        "document_state_hash": "legacy-state-hash",
        "custom_payload": {"must": "survive"},
        "judges": ["ძველი. მოსამართლე"],
        "judges_raw": ["ძველი მოსამართლე"],
        "reporting_judge": "ძ. მოსამართლე",
        "judge_extraction_confidence": "low",
        "disposition": "unknown",
        "disposition_source": "none",
        "disposition_confidence": "low",
        "disposition_mixed": True,
        "court_extractor_revision": "stale-revision",
    }


def _point(payload: dict, *, dense: list[float], sparse_index: int):
    return SimpleNamespace(
        id=point_id(payload["source"], payload["document_id"], payload["chunk_index"]),
        payload=payload,
        vector={
            "dense": dense,
            "sparse": models.SparseVector(indices=[sparse_index], values=[0.75]),
        },
    )


def _patch_target_creation(monkeypatch, client: _FakeClient) -> None:
    def fake_ensure(_client, target_cfg, **_kwargs):
        client.created_targets.append(target_cfg.collection_name)
        client.collections[target_cfg.collection_name] = {}
        return True

    monkeypatch.setattr(remat, "ensure_collection", fake_ensure)
    monkeypatch.setattr(remat, "collection_configuration", lambda _info: {"dense": 2})


def test_public_scratch_overlay_preserves_live_chunks_and_updates_court_payload(
    tmp_path: Path, monkeypatch
):
    ecd_path = _write_jsonl(tmp_path / "ecd.jsonl", _ecd_raw())
    supreme_path = _write_jsonl(tmp_path / "supreme.jsonl", _supreme_raw())
    source_files = {"ecd": ecd_path, "supremecourt": supreme_path}
    overlays = remat._load_scratch_overlays(source_files)

    ecd_overlay = overlays[("ecd", "ecd-1")]
    supreme_key = ("supremecourt", "sc-1:სამოქალაქო საქმეთა პალატა")
    supreme_overlay = overlays[supreme_key]
    ecd_payload = _payload(
        ecd_overlay,
        chunk_index=0,
        chunk_count=1,
        legacy_text="legacy ECD fragment with historical boundaries",
    )
    supreme_payload = _payload(
        supreme_overlay,
        chunk_index=0,
        chunk_count=1,
        legacy_text="legacy Supreme fragment with historical boundaries",
    )
    supreme_payload.update({"result": "stale result", "appeal_type": "stale type"})
    source_points = [
        _point(ecd_payload, dense=[0.25, -0.5], sparse_index=7),
        _point(supreme_payload, dense=[0.5, -0.25], sparse_index=9),
    ]
    source_name = "georgian_legal"
    client = _FakeClient(source_name, source_points)
    original_payloads = {
        str(point.id): copy.deepcopy(point.payload) for point in source_points
    }
    _patch_target_creation(monkeypatch, client)

    def forbidden_tokenizer(_text: str) -> int:
        raise AssertionError("scratch overlay called the tokenizer")

    result = remat.rematerialize_scratch(
        client,
        _cfg(),
        source_collection=source_name,
        source_files=source_files,
        run_id="overlay_success",
        state_dir=tmp_path / "state",
        batch_size=2,
        apply=True,
        count_tokens=forbidden_tokenizer,
        environ={"QDRANT_WRITE_APPROVED": "1"},
    )

    target = "georgian_legal_delta_court_overlay_success"
    assert result.target_collection == target
    assert result.point_count == 2
    assert result.document_count == 2
    assert result.source_logical_vector_sha256 == result.target_logical_vector_sha256
    assert client.created_targets == [target]
    assert client.upsert_targets == [target]
    assert set(client.collections[target]) == set(client.collections[source_name])

    allowlist = set(COURT_CANONICAL_FIELDS) | {"result", "appeal_type"}
    for point_id_value, source_point in client.collections[source_name].items():
        target_point = client.collections[target][point_id_value]
        assert source_point.payload == original_payloads[point_id_value]
        for field, value in original_payloads[point_id_value].items():
            if field not in allowlist:
                assert target_point.payload[field] == value
        assert target_point.vector["dense"] == source_point.vector["dense"]
        assert target_point.vector["sparse"].indices == source_point.vector["sparse"].indices
        assert target_point.vector["sparse"].values == source_point.vector["sparse"].values

    ecd_target = client.collections[target][str(source_points[0].id)].payload
    assert ecd_target["judges"] == ["ლ. ქოჩიაშვილი"]
    assert ecd_target["disposition"] == "upheld"
    assert ecd_target["court_extractor_revision"] == remat.EXTRACTOR_REVISION

    supreme_target = client.collections[target][str(source_points[1].id)].payload
    assert supreme_target["judges"] == [
        "ლ. ქოჩიაშვილი",
        "ნ. სხირტლაძე",
        "პ. სილაგაძე",
    ]
    assert supreme_target["disposition"] == "overturned"
    assert supreme_target["disposition_source"] == "scraped_result"
    assert supreme_target["result"] == _supreme_raw()["result"]
    assert supreme_target["appeal_type"] == _supreme_raw()["appeal_type"]


def test_scratch_overlay_content_hash_mismatch_refuses_before_target_creation(
    tmp_path: Path, monkeypatch
):
    source_path = _write_jsonl(tmp_path / "ecd.jsonl", _ecd_raw())
    source_files = {"ecd": source_path}
    overlay = remat._load_scratch_overlays(source_files)[("ecd", "ecd-1")]
    payload = _payload(
        overlay,
        chunk_index=0,
        chunk_count=1,
        legacy_text="legacy fragment",
    )
    payload["content_hash"] = "0" * 64
    source_name = "georgian_legal"
    client = _FakeClient(
        source_name, [_point(payload, dense=[0.25, -0.5], sparse_index=7)]
    )
    _patch_target_creation(monkeypatch, client)

    with pytest.raises(remat.RematerializationError, match="canonical body differs"):
        remat.rematerialize_scratch(
            client,
            _cfg(),
            source_collection=source_name,
            source_files=source_files,
            run_id="hash_mismatch",
            state_dir=tmp_path / "state",
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )

    assert client.created_targets == []
    assert client.upsert_targets == []
    assert "georgian_legal_delta_court_hash_mismatch" not in client.collections


def test_scratch_overlay_noncontiguous_chunks_refuses_before_target_creation(
    tmp_path: Path, monkeypatch
):
    source_path = _write_jsonl(tmp_path / "ecd.jsonl", _ecd_raw())
    source_files = {"ecd": source_path}
    overlay = remat._load_scratch_overlays(source_files)[("ecd", "ecd-1")]
    chunk_zero = _payload(
        overlay,
        chunk_index=0,
        chunk_count=3,
        legacy_text="legacy first fragment",
    )
    chunk_two = _payload(
        overlay,
        chunk_index=2,
        chunk_count=3,
        legacy_text="legacy third fragment",
    )
    source_name = "georgian_legal"
    client = _FakeClient(
        source_name,
        [
            _point(chunk_zero, dense=[0.25, -0.5], sparse_index=7),
            _point(chunk_two, dense=[0.75, -0.25], sparse_index=8),
        ],
    )
    _patch_target_creation(monkeypatch, client)

    with pytest.raises(remat.RematerializationError, match="chunks are not contiguous"):
        remat.rematerialize_scratch(
            client,
            _cfg(),
            source_collection=source_name,
            source_files=source_files,
            run_id="chunk_gap",
            state_dir=tmp_path / "state",
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )

    assert client.created_targets == []
    assert client.upsert_targets == []
    assert "georgian_legal_delta_court_chunk_gap" not in client.collections
