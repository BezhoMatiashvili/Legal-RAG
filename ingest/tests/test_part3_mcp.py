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
    mcp_server._cfg = cfg
    yield cfg
    mcp_server._cfg, mcp_server._client = old_cfg, old_client


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


def test_legal_health_ok(_tmp_cfg):
    mcp_server._client = StatusClient()
    data = json.loads(asyncio.run(legal_health()))
    assert data["ok"] is True and data["points"] == 12345 and "fingerprint" in data


def test_legal_health_reports_error(_tmp_cfg):
    class Broken:
        def get_collection(self, name):
            raise RuntimeError("qdrant down")

    mcp_server._client = Broken()
    data = json.loads(asyncio.run(legal_health()))
    assert data["ok"] is False and "qdrant down" in data["error"]
