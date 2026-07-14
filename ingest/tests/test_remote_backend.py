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


def test_remote_control_plane_health_never_claims_corpus_readiness(remote_mode):
    out = json.loads(asyncio.run(srv.legal_health()))
    assert out["ok"] is False and out["platform_ok"] is True
    assert out["backend"] == "remote"
    assert out["code"] == "generation_publish_manifest_missing_or_legacy"
    assert out["workers"] == {"idle": 1, "running": 0}
    assert remote_mode.calls == []  # /health only — no job submitted, no worker woken


def test_remote_publish_intent_still_requires_worker_generation_probe(
    remote_mode, monkeypatch, tmp_path
):
    publish = tmp_path / "publish"
    publish.mkdir()
    (publish / "manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "generation_id": "gen_20260713_test",
        "generation_manifest_sha256": "1" * 64,
    }))
    monkeypatch.setattr(srv, "_cfg", replace(srv._cfg, state_dir=tmp_path))

    out = json.loads(asyncio.run(srv.legal_health()))

    assert out["ok"] is False and out["platform_ok"] is True
    assert out["code"] == "worker_generation_not_probed"


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


def _generation_publish_manifest(blob=b"snapshot bytes"):
    generation_id = "gen_20260713_test"
    return {
        "schema_version": 2,
        "snapshot": "x.snapshot",
        "sha256": hashlib.sha256(blob).hexdigest(),
        "collection": f"georgian_legal__gen_{generation_id}",
        "points_count": 1,
        "generation_id": generation_id,
        "generation_manifest_sha256": "1" * 64,
        "point_identity": {
            "schema_version": 1,
            "generation_id": generation_id,
            "embedding_model": "BAAI/bge-m3",
            "embedding_revision": "a" * 40,
            "tokenizer_model": "BAAI/bge-m3",
            "tokenizer_revision": "b" * 40,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "reranker_revision": "c" * 40,
            "vector_space_id": "2" * 64,
            "chunking_fingerprint": "3" * 64,
            "document_header": True,
            "retrieval_fingerprint": "4" * 64,
        },
        "vector_space": {
            "dense_name": "dense",
            "dense_dimension": 1024,
            "distance": "cosine",
            "sparse_name": "sparse",
        },
    }


@pytest.mark.parametrize(
    "field",
    ("tokenizer_model", "reranker_model", "reranker_revision"),
)
def test_publish_manifest_requires_full_point_model_identity(field):
    manifest = _generation_publish_manifest()
    del manifest["point_identity"][field]

    error = qdrant_boot.validate_publish_manifest(manifest)

    assert error is not None
    assert "point_identity is missing" in error
    assert field in error


@pytest.mark.parametrize("field", ("tokenizer_model", "reranker_model"))
def test_publish_manifest_rejects_blank_point_model_names(field):
    manifest = _generation_publish_manifest()
    manifest["point_identity"][field] = "   "

    assert qdrant_boot.validate_publish_manifest(manifest) == (
        f"point_identity.{field} must be non-empty"
    )


def test_publish_manifest_rejects_mutable_reranker_revision():
    manifest = _generation_publish_manifest()
    manifest["point_identity"]["reranker_revision"] = "main"

    assert qdrant_boot.validate_publish_manifest(manifest) == (
        "point_identity.reranker_revision must be an immutable hexadecimal revision"
    )


def _compatible_collection_info(manifest, *, points=1, status="green"):
    vectors = manifest["vector_space"]
    return {
        "result": {
            "points_count": points,
            "status": status,
            "optimizer_status": "ok",
            "config": {
                "params": {
                    "vectors": {
                        vectors["dense_name"]: {
                            "size": vectors["dense_dimension"],
                            "distance": vectors["distance"].title(),
                        }
                    },
                    "sparse_vectors": {vectors["sparse_name"]: {}},
                }
            },
        }
    }


def test_needs_restore_matrix():
    m = _generation_publish_manifest()
    assert qdrant_boot.needs_restore(m, None) is True
    assert qdrant_boot.needs_restore(m, dict(m)) is False
    assert (
        qdrant_boot.needs_restore(m, dict(m, applied_at="later")) is False
    )  # bookkeeping ignored
    assert qdrant_boot.needs_restore(m, dict(m, sha256="bb")) is True
    changed = json.loads(json.dumps(m))
    changed["point_identity"]["retrieval_fingerprint"] = "5" * 64
    assert qdrant_boot.needs_restore(m, changed) is True
    assert qdrant_boot.needs_restore(None, dict(m)) is False


def test_worker_serving_gate_requires_exact_verified_generation_result():
    verified = {
        "status": "up_to_date",
        "snapshot": "x.snapshot",
        "collection": "georgian_legal__gen_gen_20260713_test",
        "generation_id": "gen_20260713_test",
        "generation_manifest_sha256": "1" * 64,
        "points": 7,
        "expected_points": 7,
        "identity_matched_points": 7,
    }
    assert qdrant_boot.restore_allows_serving(verified) is True
    for change in (
        {"status": "error"},
        {"generation_id": "legacy"},
        {"collection": "georgian_legal"},
        {"snapshot": "../x.snapshot"},
        {"generation_manifest_sha256": None},
        {"points": 6},
        {"expected_points": True, "points": 1, "identity_matched_points": 1},
        {"identity_matched_points": 6},
    ):
        assert qdrant_boot.restore_allows_serving({**verified, **change}) is False


def test_maybe_restore_without_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    assert qdrant_boot.maybe_restore()["status"] == "no_manifest"


def test_maybe_restore_refuses_bad_sha(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    pub = tmp_path / "publish"
    pub.mkdir(parents=True)
    (pub / "x.snapshot").write_bytes(b"partial upload")
    manifest = _generation_publish_manifest(b"partial upload")
    manifest["sha256"] = "0" * 64
    (pub / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(qdrant_boot, "_collection_points", lambda _collection: None)
    out = qdrant_boot.maybe_restore()
    assert out["status"] == "error" and "sha256 mismatch" in out["detail"]


def test_maybe_restore_rejects_legacy_manifest_before_qdrant(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    pub = tmp_path / "publish"
    pub.mkdir(parents=True)
    (pub / "manifest.json").write_text(
        json.dumps(
            {
                "snapshot": "x.snapshot",
                "sha256": "0" * 64,
                "collection": "georgian_legal",
                "points_count": 1,
            }
        )
    )
    monkeypatch.setattr(
        qdrant_boot,
        "_http",
        lambda *_args, **_kwargs: pytest.fail("legacy manifest reached Qdrant"),
    )

    out = qdrant_boot.maybe_restore()

    assert out["status"] == "error"
    assert out["code"] == "invalid_manifest"
    assert "legacy/incomplete" in out["detail"]


def _publish_fixture(tmp_path, blob=b"snapshot bytes"):
    pub = tmp_path / "publish"
    pub.mkdir(parents=True)
    manifest = _generation_publish_manifest(blob)
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
        if method == "GET":
            return _compatible_collection_info(manifest)
        assert method == "POST" and path.endswith("/points/count")
        assert body["exact"] is True
        return {"result": {"count": 1}}

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)
    out = qdrant_boot.maybe_restore()
    assert out["status"] == "up_to_date" and out["points"] == 1
    assert qdrant_boot.restore_allows_serving(out) is True
    collection = manifest["collection"]
    assert calls == [
        ("GET", f"/collections/{collection}"),
        ("POST", f"/collections/{collection}/points/count"),
    ]  # no recover PUT


def test_runtime_binding_rechecks_active_files_and_live_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    pub, manifest = _publish_fixture(tmp_path)
    (pub / "ACTIVE").write_text(json.dumps(dict(manifest, applied_at="t")))

    def fake_http(method, path, body=None, timeout=10):
        if method == "GET":
            return _compatible_collection_info(manifest)
        assert method == "POST" and path.endswith("/points/count")
        return {"result": {"count": 1}}

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)
    restored = qdrant_boot.maybe_restore()
    bound = qdrant_boot.verified_runtime_manifest(restored)

    ready = qdrant_boot.runtime_readiness(restored, bound)

    assert ready["ok"] is True
    assert ready["collection"] == manifest["collection"]
    assert ready["generation_id"] == manifest["generation_id"]


def test_runtime_binding_rejects_warm_generation_change(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    pub, manifest = _publish_fixture(tmp_path)
    (pub / "ACTIVE").write_text(json.dumps(dict(manifest, applied_at="t")))

    def fake_http(method, path, body=None, timeout=10):
        if method == "GET":
            return _compatible_collection_info(manifest)
        return {"result": {"count": 1}}

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)
    restored = qdrant_boot.maybe_restore()
    bound = qdrant_boot.verified_runtime_manifest(restored)
    changed = json.loads(json.dumps(manifest))
    changed["point_identity"]["retrieval_fingerprint"] = "5" * 64
    (pub / "manifest.json").write_text(json.dumps(changed))
    (pub / "ACTIVE").write_text(json.dumps(dict(changed, applied_at="later")))

    ready = qdrant_boot.runtime_readiness(restored, bound)

    assert ready["ok"] is False
    assert ready["code"] == "worker_runtime_binding_invalid"
    assert "active generation changed after search runtime import" in ready["error"]


def test_verified_runtime_manifest_rejects_stale_active(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    pub, manifest = _publish_fixture(tmp_path)
    stale = json.loads(json.dumps(manifest))
    stale["generation_manifest_sha256"] = "9" * 64
    (pub / "ACTIVE").write_text(json.dumps(stale))
    restored = {
        "status": "up_to_date",
        "snapshot": manifest["snapshot"],
        "collection": manifest["collection"],
        "generation_id": manifest["generation_id"],
        "generation_manifest_sha256": manifest["generation_manifest_sha256"],
        "points": 1,
        "expected_points": 1,
        "identity_matched_points": 1,
    }
    monkeypatch.setattr(
        qdrant_boot,
        "_http",
        lambda *_args, **_kwargs: pytest.fail("stale ACTIVE reached Qdrant"),
    )

    with pytest.raises(RuntimeError, match="manifest and ACTIVE"):
        qdrant_boot.verified_runtime_manifest(restored)


def test_maybe_restore_does_not_trust_count_without_payload_identity(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    pub, manifest = _publish_fixture(tmp_path)
    (pub / "ACTIVE").write_text(json.dumps(dict(manifest, applied_at="t")))

    def fake_http(method, path, body=None, timeout=10):
        if method == "GET":
            return _compatible_collection_info(manifest)
        assert method == "POST" and body["exact"] is True
        return {"result": {"count": 0}}

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)

    out = qdrant_boot.maybe_restore()

    assert out["status"] == "error"
    assert out["code"] == "identity_payload_count_mismatch"


def test_maybe_restore_re_restores_when_collection_vanished(monkeypatch, tmp_path):
    """A stale ACTIVE (wiped storage / swapped volume) must not be trusted blindly."""
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    pub, manifest = _publish_fixture(tmp_path)
    (pub / "ACTIVE").write_text(json.dumps(dict(manifest, applied_at="t")))
    calls = []

    def fake_http(method, path, body=None, timeout=10):
        calls.append((method, path))
        collection_path = f"/collections/{manifest['collection']}"
        if method == "GET" and path == collection_path and len(calls) == 1:
            raise RuntimeError("collection missing")  # reality check fails
        if method == "GET":
            return _compatible_collection_info(manifest)
        if method == "POST":
            return {"result": {"count": 1}}
        return {"result": {}}

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)
    out = qdrant_boot.maybe_restore()
    assert out["status"] == "restored"
    assert qdrant_boot.restore_allows_serving(out) is True
    assert any("snapshots/recover" in p for _, p in calls)
    assert (pub / "ACTIVE").stat().st_mode & 0o777 == 0o600


def test_maybe_restore_requires_green_collection_after_recovery(monkeypatch, tmp_path):
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    pub, manifest = _publish_fixture(tmp_path)
    get_calls = 0

    def fake_http(method, path, body=None, timeout=10):
        nonlocal get_calls
        if method == "GET":
            get_calls += 1
            if get_calls == 1:
                raise RuntimeError("collection is absent before cold restore")
            return _compatible_collection_info(manifest, status="yellow")
        if method == "PUT":
            return {"result": {}}
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)

    out = qdrant_boot.maybe_restore()

    assert out["status"] == "error"
    assert out["code"] == "collection_not_green"
    assert not (pub / "ACTIVE").exists()


def test_maybe_restore_refuses_to_replace_live_collection(monkeypatch, tmp_path):
    """A warm refresh must preserve last-good data until blue/green restore exists."""
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    _publish_fixture(tmp_path)
    calls = []

    def fake_http(method, path, body=None, timeout=10):
        calls.append((method, path))
        collection = _generation_publish_manifest()["collection"]
        if method == "GET" and path == f"/collections/{collection}":
            return {"result": {"points_count": 123}}
        raise AssertionError(f"destructive restore attempted: {method} {path}")

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)
    out = qdrant_boot.maybe_restore(force=True)

    assert out["status"] == "error"
    assert out["code"] == "unsafe_warm_restore"
    assert out["points"] == 123
    collection = _generation_publish_manifest()["collection"]
    assert calls == [("GET", f"/collections/{collection}")]


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
