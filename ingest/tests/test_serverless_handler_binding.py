"""Hermetic cold-boot-to-search binding tests; no Qdrant, network, or models."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
import uuid
from dataclasses import replace
from pathlib import Path

from ingest import mcp_server
from ingest.config import load_config
from ingest.qdrant_store import generation_point_identity

HANDLER_PATH = Path(__file__).resolve().parents[1] / "serverless" / "handler.py"
_HANDLER_ENV = (
    "SEARCH_BACKEND",
    "QUERY_LOG_ENABLED",
    "QDRANT_URL",
    "COLLECTION_NAME",
    "GENERATION_ID",
    "GENERATION_DIR",
    "EMBED_MODEL",
    "EMBED_REVISION",
    "TOKENIZER_MODEL",
    "TOKENIZER_REVISION",
    "RERANK_MODEL",
    "RERANK_REVISION",
    "RERANK_REMOTE_URL",
    "DENSE_DIM",
    "EMBED_HEADER_V2",
    "PRODUCTION_MODE",
    "VERIFIED_WORKER_BINDING",
    "RETRIEVER_LICENSE_ATTESTATION_PATH",
    "RERANKER_LICENSE_ATTESTATION_PATH",
    "HF_HOME",
)


def _manifest() -> dict:
    generation_id = "gen_20260713_binding"
    cfg = replace(
        load_config(),
        qdrant_url="http://127.0.0.1:6333",
        collection_name=f"georgian_legal__gen_{generation_id}",
        search_backend="local",
        embed_model="BAAI/bge-m3",
        embedding_revision="a" * 40,
        tokenizer_model="BAAI/bge-m3",
        tokenizer_revision="b" * 40,
        rerank_model="BAAI/bge-reranker-v2-m3",
        reranker_revision="c" * 40,
        dense_dim=1024,
        embed_header_v2=False,
        generation_id=generation_id,
        generation_dir=None,
        production_mode=True,
        verified_worker_binding=True,
    )
    identity = generation_point_identity(cfg)
    assert identity is not None
    return {
        "schema_version": 2,
        "snapshot": "binding.snapshot",
        "sha256": "0" * 64,
        "collection": cfg.collection_name,
        "points_count": 11,
        "generation_id": generation_id,
        "generation_manifest_sha256": "1" * 64,
        "point_identity": identity.as_payload(),
        "vector_space": {
            "dense_name": "dense",
            "dense_dimension": 1024,
            "distance": "cosine",
            "sparse_name": "sparse",
        },
    }


def _restore(manifest: dict) -> dict:
    return {
        "status": "up_to_date",
        "snapshot": manifest["snapshot"],
        "collection": manifest["collection"],
        "generation_id": manifest["generation_id"],
        "generation_manifest_sha256": manifest["generation_manifest_sha256"],
        "points": manifest["points_count"],
        "expected_points": manifest["points_count"],
        "identity_matched_points": manifest["points_count"],
    }


def _load_handler(monkeypatch, tmp_path, manifest: dict | None):
    # Register every handler-owned env key with monkeypatch so module-level hard assigns
    # are restored after the test as well.
    original = dict(os.environ)
    for key in _HANDLER_ENV:
        if key in original:
            monkeypatch.setenv(key, original[key])
        else:
            # Record an undo even though the key starts absent; handler.py assigns it
            # directly rather than through this fixture.
            monkeypatch.setenv(key, "")
            monkeypatch.delenv(key, raising=False)

    state = {"manifest": manifest}
    if manifest is not None:
        identity = manifest["point_identity"]
        for label, role, model_key, revision_key in (
            ("retriever", "embedder", "embedding_model", "embedding_revision"),
            ("reranker", "reranker", "reranker_model", "reranker_revision"),
        ):
            model_id = identity[model_key]
            revision = identity[revision_key]
            path = tmp_path / f"{label}-license.json"
            path.write_text(json.dumps({
                "model_id": model_id,
                "revision": revision,
                "version": f"{model_id}@{revision}",
                "role": role,
                "license_id": "Apache-2.0",
                "commercial_use_allowed": True,
                "weights_private_deployment_allowed": True,
                "training_data_use_allowed": True,
                "reviewed_by": "legal-reviewer",
                "reviewed_at": "2026-07-15",
                "authoritative_source_url": "https://example.test/license",
            }), encoding="utf-8")
            monkeypatch.setenv(
                f"{label.upper()}_LICENSE_ATTESTATION_PATH", str(path)
            )
    fake_boot = types.ModuleType("qdrant_boot")
    fake_boot.QDRANT_URL = "http://127.0.0.1:6333"
    fake_boot.VOLUME_ROOT = tmp_path
    fake_boot.ensure_running = lambda: None
    fake_boot.maybe_restore = lambda force=False: (
        _restore(state["manifest"])
        if state["manifest"] is not None
        else {"status": "no_manifest"}
    )
    fake_boot.verified_runtime_manifest = lambda result: json.loads(
        json.dumps(state["manifest"])
    )
    fake_boot.runtime_manifest_identity = lambda value: (
        json.dumps(value, sort_keys=True, separators=(",", ":")),
    )
    fake_boot.runtime_readiness = lambda result, bound: (
        {
            "ok": bound == state["manifest"],
            "collection": state["manifest"]["collection"],
            "generation_id": state["manifest"]["generation_id"],
            "issues": [],
        }
        if state["manifest"] is not None and bound is not None
        else {
            "ok": False,
            "code": "worker_runtime_binding_invalid",
            "issues": [{"gate": "integrity", "code": "no_binding"}],
        }
    )
    fake_boot.restore_pending = lambda: False

    calls: dict[str, object] = {"health": 0}
    fake_srv = types.ModuleType("ingest.mcp_server")

    class Input:
        def __init__(self, **values):
            self.values = values

    async def operation(*_args, **_kwargs):
        return "ok"

    async def health():
        calls["health"] = int(calls["health"]) + 1
        return json.dumps({"ok": True})

    def install(cfg, probe, **kwargs):
        calls["cfg"] = cfg
        calls["probe"] = probe
        calls["attestations"] = kwargs

    fake_srv._install_verified_worker_runtime = install
    fake_srv._result_cache = {}
    fake_srv._get_embedder = operation
    fake_srv._get_reranker = operation
    fake_srv.legal_search = operation
    fake_srv.legal_ask = operation
    fake_srv.legal_get_context = operation
    fake_srv.legal_get_document = operation
    fake_srv.legal_lookup = operation
    fake_srv.legal_browse = operation
    fake_srv.legal_get_document_versions = operation
    fake_srv.legal_collection_info = operation
    fake_srv.legal_health = health
    fake_srv.SearchInput = Input
    fake_srv.LegalAskInput = Input
    fake_srv.LegalGetContextInput = Input
    fake_srv.GetDocumentInput = Input
    fake_srv.LookupInput = Input
    fake_srv.BrowseInput = Input
    fake_srv.GetVersionsInput = Input

    import ingest as ingest_package

    monkeypatch.setitem(sys.modules, "qdrant_boot", fake_boot)
    monkeypatch.setitem(sys.modules, "ingest.mcp_server", fake_srv)
    monkeypatch.setattr(ingest_package, "mcp_server", fake_srv)
    module_name = f"_test_serverless_handler_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, HANDLER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, state, calls


def test_cold_boot_binds_exact_runtime_before_mcp_install(monkeypatch, tmp_path):
    manifest = _manifest()
    module, _state, calls = _load_handler(monkeypatch, tmp_path, manifest)

    cfg = calls["cfg"]
    assert module._BOOT_ERROR is None
    assert cfg.production_mode is True
    assert cfg.verified_worker_binding is True
    assert cfg.generation_dir is None
    assert cfg.collection_name == manifest["collection"]
    assert cfg.generation_id == manifest["generation_id"]
    assert cfg.rerank_remote_url is None
    assert generation_point_identity(cfg).as_payload() == manifest["point_identity"]
    assert calls["probe"]()["ok"] is True
    assert set(calls["attestations"]) == {
        "retriever_attestation",
        "reranker_attestation",
    }


def test_prepublication_health_is_available_but_never_calls_mcp(monkeypatch, tmp_path):
    module, _state, calls = _load_handler(monkeypatch, tmp_path, None)

    out = asyncio.run(module.handler({"input": {"op": "health"}}))
    health = json.loads(out["result"])

    assert module._BOOT_ERROR is None
    assert health["ok"] is False
    assert health["code"] == "worker_runtime_binding_invalid"
    assert calls["health"] == 0
    assert "cfg" not in calls


def test_cold_boot_rejects_ambient_retrieval_policy_mismatch(monkeypatch, tmp_path):
    manifest = _manifest()
    manifest["point_identity"]["retrieval_fingerprint"] = "f" * 64
    module, _state, calls = _load_handler(monkeypatch, tmp_path, manifest)

    assert "retrieval_fingerprint" in module._BOOT_ERROR
    assert "cfg" not in calls
    out = asyncio.run(module.handler({"input": {"op": "health"}}))
    assert "worker boot failed" in out["error"]
    assert calls["health"] == 0


def test_warm_generation_change_cannot_rebind_imported_search(monkeypatch, tmp_path):
    manifest = _manifest()
    module, state, _calls = _load_handler(monkeypatch, tmp_path, manifest)
    changed = json.loads(json.dumps(manifest))
    changed["generation_id"] = "gen_20260713_changed"
    changed["collection"] = "georgian_legal__gen_gen_20260713_changed"
    changed["point_identity"]["generation_id"] = changed["generation_id"]
    state["manifest"] = changed

    try:
        module._accept_restore(_restore(changed), allow_initial_bind=False)
    except RuntimeError as exc:
        assert "cold restart required" in str(exc)
    else:  # pragma: no cover - explicit fail message is clearer than a bare assert
        raise AssertionError("warm generation unexpectedly rebound imported search")

    out = asyncio.run(
        module.handler({"input": {"op": "search", "params": {"query": "x"}}})
    )
    assert "worker abstained" in out["error"]


def test_verified_worker_env_without_handler_probe_fails_closed():
    cfg = replace(
        load_config(),
        collection_name="georgian_legal__gen_gen_20260713_binding",
        search_backend="local",
        generation_id="gen_20260713_binding",
        generation_dir=None,
        production_mode=True,
        verified_worker_binding=True,
    )
    old_probe = mcp_server._worker_readiness_probe
    mcp_server._worker_readiness_probe = None
    try:
        readiness = mcp_server._local_readiness(cfg, object())
    finally:
        mcp_server._worker_readiness_probe = old_probe

    assert readiness["ok"] is False
    assert readiness["code"] == "worker_readiness_probe_not_installed"


def test_worker_probe_cannot_claim_a_different_generation():
    cfg = replace(
        load_config(),
        collection_name="georgian_legal__gen_gen_20260713_binding",
        search_backend="local",
        generation_id="gen_20260713_binding",
        generation_dir=None,
        production_mode=True,
        verified_worker_binding=True,
    )
    old_probe = mcp_server._worker_readiness_probe
    mcp_server._worker_readiness_probe = lambda: {
        "ok": True,
        "collection": "georgian_legal__gen_gen_20260713_other",
        "generation_id": "gen_20260713_other",
        "issues": [],
    }
    try:
        readiness = mcp_server._local_readiness(cfg, object())
    finally:
        mcp_server._worker_readiness_probe = old_probe

    assert readiness["ok"] is False
    assert readiness["code"] == "worker_readiness_identity_mismatch"
