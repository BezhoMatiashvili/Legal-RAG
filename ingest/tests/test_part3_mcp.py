"""Offline tests for the new MCP tools (ingest_status, legal_get_document_versions, health)."""

import asyncio
import dataclasses
import json
from types import SimpleNamespace

import pytest

from ingest import mcp_server
from ingest.config import load_config
from ingest.mcp_server import (
    GetVersionsInput,
    ResponseFormat,
    StatusInput,
    ingest_status,
    legal_get_document_versions,
    legal_health,
    legal_search,
    SearchInput,
)


def _pt(doc, reg, status, date, ci=0):
    return SimpleNamespace(payload={
        "source": "matsne", "document_id": doc, "registration_code": reg, "status": status,
        "date": date + "T00:00:00Z", "date_raw": date, "title": f"Act {doc}", "chunk_index": ci,
        "document_type": "legislation", "document_number": None, "parties": None,
        "court": None, "language": "ka", "source_url": None,
    })


class VersionClient:
    def __init__(self, points):
        self._points = points

    def scroll(self, collection_name, scroll_filter=None, with_payload=True,
               with_vectors=False, limit=256, offset=None):
        return list(self._points), None


class StatusClient:
    def get_collection(self, name):
        return SimpleNamespace(points_count=12345)

    def facet(self, collection_name, key, limit=50):
        return SimpleNamespace(hits=[SimpleNamespace(value="matsne", count=9000),
                                     SimpleNamespace(value="ecd", count=3345)])


@pytest.fixture
def _tmp_cfg(tmp_path):
    cfg = dataclasses.replace(load_config(), collection_name="test", state_dir=tmp_path / "state")
    old_cfg, old_client = mcp_server._cfg, mcp_server._client
    old_manifest = mcp_server._generation_manifest
    old_root = mcp_server._generation_root
    mcp_server._cfg = cfg
    mcp_server._generation_manifest = None
    mcp_server._generation_root = None
    yield cfg
    mcp_server._cfg, mcp_server._client = old_cfg, old_client
    mcp_server._generation_manifest = old_manifest
    mcp_server._generation_root = old_root


def test_get_document_versions_groups_by_registration_code(_tmp_cfg):
    pts = [_pt("v1", "REG1", "repealed", "2019-01-01"),
           _pt("v2", "REG1", "in_force", "2022-06-01")]
    mcp_server._client = VersionClient(pts)
    out = asyncio.run(legal_get_document_versions(
        GetVersionsInput(source="matsne", document_id="v1", response_format=ResponseFormat.JSON)))
    data = json.loads(out)
    assert data["registration_code"] == "REG1" and data["count"] == 2
    versions = {v["document_id"]: v for v in data["versions"]}
    assert versions["v2"]["is_current"] is True        # in_force = current
    assert versions["v1"]["is_current"] is False
    assert data["versions"][0]["document_id"] == "v2"  # newest first


def test_get_document_versions_single_when_no_reg_code(_tmp_cfg):
    mcp_server._client = VersionClient([_pt("d", None, None, "2020-01-01")])
    out = asyncio.run(legal_get_document_versions(
        GetVersionsInput(source="ecd", document_id="d", response_format=ResponseFormat.JSON)))
    data = json.loads(out)
    assert data["registration_code"] is None and data["count"] == 1


def test_ingest_status_reports_points_and_no_watcher(_tmp_cfg):
    mcp_server._client = StatusClient()
    out = asyncio.run(ingest_status(StatusInput(response_format=ResponseFormat.JSON)))
    data = json.loads(out)
    assert data["points"] == 12345
    assert data["per_source"]["matsne"] == 9000
    assert data["watchers"] == {}                      # no watcher state on a fresh box


def test_legacy_health_fails_without_generation_manifest(_tmp_cfg):
    mcp_server._client = StatusClient()
    data = json.loads(asyncio.run(legal_health()))
    assert data["ok"] is False
    assert data["code"] == "generation_manifest_not_configured"
    assert data["points"] == 12345 and "fingerprint" in data


def test_exact_generation_health_returns_compatibility_payload(_tmp_cfg, monkeypatch):
    mcp_server._client = StatusClient()
    monkeypatch.setattr(
        mcp_server,
        "_local_readiness",
        lambda cfg, client: {
            "ok": True,
            "collection": cfg.collection_name,
            "generation_id": "gen_verified",
            "points": 12345,
            "issues": [],
        },
    )
    data = json.loads(asyncio.run(legal_health()))
    assert data["ok"] is True
    assert data["generation_id"] == "gen_verified"


def test_production_search_abstains_before_model_load(_tmp_cfg, monkeypatch):
    mcp_server._cfg = dataclasses.replace(_tmp_cfg, production_mode=True)
    mcp_server._client = StatusClient()

    def incompatible(_cfg, _client):
        raise RuntimeError("generation identity mismatch")

    async def model_must_not_load():
        raise AssertionError("model loaded before readiness")

    monkeypatch.setattr(mcp_server, "_require_local_readiness", incompatible)
    monkeypatch.setattr(mcp_server, "_get_embedder", model_must_not_load)
    out = asyncio.run(legal_search(SearchInput(query="test query")))
    assert "generation identity mismatch" in out


def test_legal_health_reports_error(_tmp_cfg):
    class Broken:
        def get_collection(self, name):
            raise RuntimeError("qdrant down")

    mcp_server._client = Broken()
    data = json.loads(asyncio.run(legal_health()))
    assert data["ok"] is False and "qdrant down" in data["error"]


def test_browse_scroll_is_streamed_metadata_only_and_chunk_zero_filtered():
    class PagingClient:
        def __init__(self):
            self.calls = []
            self.points = [
                _pt("a", "R1", "in_force", "2024-01-01", 0),
                _pt("b", "R2", "in_force", "2023-01-01", 0),
            ]

        def scroll(self, **kwargs):
            self.calls.append(kwargs)
            start = kwargs.get("offset") or 0
            batch = self.points[start:start + 1]
            return batch, (start + 1 if start + 1 < len(self.points) else None)

    client = PagingClient()
    docs = mcp_server._dedup_documents(mcp_server._iter_document_points(
        client, "test", None, chunk_zero_only=True, page=1))

    assert [doc["document_id"] for doc in docs] == ["a", "b"]
    assert len(client.calls) == 2
    assert client.calls[0]["with_payload"] == mcp_server._DOCUMENT_PAYLOAD_FIELDS
    conditions = client.calls[0]["scroll_filter"].must
    assert any(condition.key == "chunk_index" for condition in conditions)
