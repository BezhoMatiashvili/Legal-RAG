"""Remote-backend tests: the RunPod queue client, MCP tool routing, and the worker's
publish/restore bookkeeping. All offline — no network, no models, no Qdrant."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ingest import mcp_server as srv
from ingest.config import load_config
from ingest.remote_search import (
    EndpointWarmingUp,
    RemoteOpError,
    RemoteSearchError,
    RunPodQueueClient,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serverless"))
import qdrant_boot  # noqa: E402


# --- RunPodQueueClient ----------------------------------------------------------


def _client(responses, **kwargs):
    """Client whose _request pops canned responses: [(expected_method_path_prefix, resp), ...]."""
    c = RunPodQueueClient("ep-test", "key-test", poll_interval=0.0, **kwargs)
    seq = list(responses)

    def fake_request(method, path, body=None, timeout=30):
        assert seq, f"unexpected extra request {method} {path}"
        resp = seq[0]
        if callable(resp):
            return resp(method, path)
        seq.pop(0)
        return resp

    c._request = fake_request
    return c


def test_call_polls_to_completion():
    c = _client([
        {"id": "j1", "status": "IN_QUEUE"},
        {"status": "IN_PROGRESS"},
        {"status": "COMPLETED", "output": {"result": "hello"}},
    ])
    assert c.call("search", {"query": "x"}) == {"result": "hello"}


def test_call_raises_on_worker_error():
    c = _client([
        {"id": "j1"},
        {"status": "COMPLETED", "output": {"error": "boom", "traceback": []}},
    ])
    with pytest.raises(RemoteOpError, match="boom"):
        c.call("search", {})


def test_call_raises_on_platform_failure():
    c = _client([{"id": "j1"}, {"status": "FAILED", "error": "worker exited"}])
    with pytest.raises(RemoteOpError, match="FAILED"):
        c.call("search", {})


def test_call_budget_expiry_is_warming_up():
    c = _client([{"id": "j1", "status": "IN_QUEUE"}, lambda m, p: {"status": "IN_QUEUE"}],
                timeout=0.05)
    with pytest.raises(EndpointWarmingUp, match="cold-start"):
        c.call("search", {})


def test_missing_credentials_fail_fast():
    with pytest.raises(RemoteSearchError, match="RUNPOD_ENDPOINT_ID"):
        RunPodQueueClient("", "")


# --- MCP tool routing in remote mode ---------------------------------------------


class FakeRemote:
    def __init__(self):
        self.calls: list[tuple] = []

    def call(self, op, params=None, timeout=None):
        self.calls.append((op, params))
        return {"result": f"REMOTE:{op}"}

    def health(self):
        return {"workers": {"idle": 1, "running": 0}, "jobs": {"completed": 3}}


@pytest.fixture
def remote_mode(monkeypatch):
    cfg = replace(load_config(), search_backend="remote",
                  runpod_endpoint_id="ep-test", runpod_api_key="key-test")
    monkeypatch.setattr(srv, "_cfg", cfg)
    fake = FakeRemote()
    monkeypatch.setattr(srv, "_remote_client", fake)
    return fake


def test_search_routes_remotely(remote_mode):
    out = asyncio.run(srv.legal_search(srv.SearchInput(query="იარაღი", top_k=5)))
    assert out == "REMOTE:search"
    op, params = remote_mode.calls[0]
    assert op == "search"
    assert params["query"] == "იარაღი"
    assert params["top_k"] == 5
    assert params["response_format"] == "markdown"  # enum serialized for the wire


def test_search_params_revalidate_on_worker(remote_mode):
    """The wire dict must round-trip through the same pydantic model the worker uses."""
    asyncio.run(srv.legal_search(srv.SearchInput(query="x", status="in_force")))
    _, params = remote_mode.calls[0]
    again = srv.SearchInput(**params)  # what handler.py does
    assert again.query == "x" and again.status == "in_force"


def test_get_document_browse_lookup_versions_route(remote_mode):
    asyncio.run(srv.legal_get_document(srv.GetDocumentInput(source="matsne", document_id="d1")))
    asyncio.run(srv.legal_browse(srv.BrowseInput(source="matsne")))
    asyncio.run(srv.legal_lookup(srv.LookupInput(document_number="55")))
    asyncio.run(srv.legal_get_document_versions(
        srv.GetVersionsInput(source="matsne", document_id="d1")))
    assert [c[0] for c in remote_mode.calls] == ["get_document", "browse", "lookup", "versions"]


def test_collection_info_routes(remote_mode):
    assert asyncio.run(srv.legal_collection_info()) == "REMOTE:collection_info"


def test_lookup_without_identifier_never_rpcs(remote_mode):
    out = asyncio.run(srv.legal_lookup(srv.LookupInput()))
    assert "Provide at least one identifier" in out
    assert remote_mode.calls == []


def test_health_remote_uses_endpoint_health_not_a_worker(remote_mode):
    out = json.loads(asyncio.run(srv.legal_health()))
    assert out["ok"] is True and out["backend"] == "remote"
    assert out["workers"] == {"idle": 1, "running": 0}
    assert remote_mode.calls == []  # /health only — no job submitted, no worker woken


def test_remote_error_message_names_the_env_flip(remote_mode, monkeypatch):
    def boom(op, params=None, timeout=None):
        raise RemoteOpError("worker op failed")
    monkeypatch.setattr(remote_mode, "call", boom)
    out = asyncio.run(srv.legal_search(srv.SearchInput(query="x")))
    assert "SEARCH_BACKEND=local" in out


def test_local_mode_is_default(monkeypatch):
    # Hermetic: ignore whatever SEARCH_BACKEND the live ingest/.env sets (it is 'remote'
    # once the RunPod backend is wired up) and assert the code default when it is unset.
    monkeypatch.delenv("SEARCH_BACKEND", raising=False)
    assert load_config().search_backend == "local"


# --- worker restore bookkeeping (qdrant_boot) -------------------------------------


def test_needs_restore_matrix():
    m = {"snapshot": "s1", "sha256": "aa", "collection": "georgian_legal"}
    assert qdrant_boot.needs_restore(m, None) is True
    assert qdrant_boot.needs_restore(m, dict(m)) is False
    assert qdrant_boot.needs_restore(m, dict(m, applied_at="later")) is False  # bookkeeping ignored
    assert qdrant_boot.needs_restore(m, dict(m, sha256="bb")) is True
    assert qdrant_boot.needs_restore(None, dict(m)) is False


def test_maybe_restore_without_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    assert qdrant_boot.maybe_restore()["status"] == "no_manifest"


def test_maybe_restore_refuses_bad_sha(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    pub = tmp_path / "publish"
    pub.mkdir(parents=True)
    (pub / "x.snapshot").write_bytes(b"partial upload")
    (pub / "manifest.json").write_text(json.dumps({
        "snapshot": "x.snapshot", "sha256": "0" * 64,
        "collection": "georgian_legal", "points_count": 1}))
    out = qdrant_boot.maybe_restore()
    assert out["status"] == "error" and "sha256 mismatch" in out["detail"]


def _publish_fixture(tmp_path, blob=b"snapshot bytes"):
    pub = tmp_path / "publish"
    pub.mkdir(parents=True)
    manifest = {"snapshot": "x.snapshot", "sha256": hashlib.sha256(blob).hexdigest(),
                "collection": "georgian_legal", "points_count": 1}
    (pub / "x.snapshot").write_bytes(blob)
    (pub / "manifest.json").write_text(json.dumps(manifest))
    return pub, manifest


def test_maybe_restore_up_to_date_reality_checks_the_collection(monkeypatch, tmp_path):
    """ACTIVE == manifest → up_to_date, but only after confirming the collection exists."""
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    pub, manifest = _publish_fixture(tmp_path)
    (pub / "ACTIVE").write_text(json.dumps(dict(manifest, applied_at="t")))
    calls = []

    def fake_http(method, path, body=None, timeout=10):
        calls.append((method, path))
        return {"result": {"points_count": 1}}

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)
    out = qdrant_boot.maybe_restore()
    assert out["status"] == "up_to_date" and out["points"] == 1
    assert calls == [("GET", "/collections/georgian_legal")]  # no recover PUT


def test_maybe_restore_re_restores_when_collection_vanished(monkeypatch, tmp_path):
    """A stale ACTIVE (wiped storage / swapped volume) must not be trusted blindly."""
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    pub, manifest = _publish_fixture(tmp_path)
    (pub / "ACTIVE").write_text(json.dumps(dict(manifest, applied_at="t")))
    calls = []

    def fake_http(method, path, body=None, timeout=10):
        calls.append((method, path))
        if method == "GET" and path == "/collections/georgian_legal" and len(calls) == 1:
            raise RuntimeError("collection missing")  # reality check fails
        return {"result": {"points_count": 1}}

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)
    out = qdrant_boot.maybe_restore()
    assert out["status"] == "restored"
    assert any("snapshots/recover" in p for _, p in calls)


def test_maybe_restore_refuses_to_replace_live_collection(monkeypatch, tmp_path):
    """A warm refresh must preserve last-good data until blue/green restore exists."""
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    _publish_fixture(tmp_path)
    calls = []

    def fake_http(method, path, body=None, timeout=10):
        calls.append((method, path))
        if method == "GET" and path == "/collections/georgian_legal":
            return {"result": {"points_count": 123}}
        raise AssertionError(f"destructive restore attempted: {method} {path}")

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)
    out = qdrant_boot.maybe_restore(force=True)

    assert out["status"] == "error"
    assert out["code"] == "unsafe_warm_restore"
    assert out["points"] == 123
    assert calls == [("GET", "/collections/georgian_legal")]


def test_restore_pending_is_a_pure_file_check(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    assert qdrant_boot.restore_pending() is False  # no manifest at all
    pub, manifest = _publish_fixture(tmp_path)
    assert qdrant_boot.restore_pending() is True   # manifest, no ACTIVE yet
    (pub / "ACTIVE").write_text(json.dumps(dict(manifest, applied_at="t")))
    assert qdrant_boot.restore_pending() is False


def test_poll_survives_transient_status_errors(monkeypatch):
    """A single 502 mid-poll must not abandon a job that is still cold-starting."""
    import ingest.remote_search as rs
    from ingest.remote_search import RemoteSearchError

    monkeypatch.setattr(rs.time, "sleep", lambda s: None)  # skip the error backoff
    flaky = iter([
        {"id": "j1", "status": "IN_QUEUE"},
        RemoteSearchError("Cloudflare 502"),
        RemoteSearchError("Cloudflare 502"),
        {"status": "COMPLETED", "output": {"result": "ok"}},
    ])
    c = RunPodQueueClient("ep", "k", poll_interval=0.0)

    def fake_request(method, path, body=None, timeout=30):
        item = next(flaky)
        if isinstance(item, Exception):
            raise item
        return item

    c._request = fake_request
    assert c.call("search", {}) == {"result": "ok"}
