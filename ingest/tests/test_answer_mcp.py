import asyncio
import dataclasses
import json
from types import SimpleNamespace

import pytest

from ingest import mcp_server
from ingest.config import load_config
from ingest.legal_answer import EvidencePack
from ingest.mcp_server import (
    LegalAskInput,
    LegalGetContextInput,
    legal_ask,
    legal_get_context,
)
from ingest.model_policy import LicenseAttestation, ModelRole
from ingest.risk_calibration import FEATURE_NAMES, HeldOutRiskCalibrator


@pytest.fixture(autouse=True)
def _reset_runtime(monkeypatch):
    for name in (
        "_answer_translator",
        "_answer_composer",
        "_answer_calibrator",
        "_answer_freshness_guard",
        "_retriever_license_attestation",
        "_reranker_license_attestation",
    ):
        monkeypatch.setattr(mcp_server, name, None)


def _cfg(*, production=False, worker=False):
    return dataclasses.replace(
        load_config(),
        generation_id="generation-20260715",
        generation_dir=None,
        production_mode=production,
        verified_worker_binding=worker,
        embedding_revision="a" * 40,
        reranker_revision="b" * 40,
    )


def _attestation(role, model_id, revision, **overrides):
    values = {
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
    }
    values.update(overrides)
    return LicenseAttestation(**values)


def _retrieval_attestations(cfg):
    return {
        "retriever_attestation": _attestation(
            ModelRole.EMBEDDER, cfg.embed_model, cfg.embedding_revision
        ),
        "reranker_attestation": _attestation(
            ModelRole.RERANKER, cfg.rerank_model, cfg.reranker_revision
        ),
    }


def _provider(role, model_id, revision, version):
    provider = SimpleNamespace(
        role=role,
        model_id=model_id,
        revision=revision,
        version=version,
    )
    provider.license_attestation = _attestation(
        role,
        model_id,
        revision,
        version=version,
    )
    return provider


def _calibrator(tmp_path, binding):
    payload = {
        "schema_version": "heldout-selective-risk/v2",
        "calibration_set_sha256": "c" * 64,
        "document_version_split_sha256": "d" * 64,
        "pipeline_binding": dataclasses.asdict(binding),
        "feature_weights": {name: 0.0 for name in FEATURE_NAMES},
        "intercept": 0.5,
        "missing_margin_value": 0.0,
        "bands": [{
            "lower_score": 0.0,
            "upper_score": 1.0,
            "answered": 1000,
            "severe_errors": 0,
            "material_errors": 0,
            "answered_clusters": 1000,
            "severe_error_clusters": 0,
            "material_error_clusters": 0,
        }],
        "minimum_band_answered": 1000,
        "confidence": 0.95,
        "severe_error_upper_limit": 0.01,
        "material_error_upper_limit": 0.03,
        "held_out": True,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return HeldOutRiskCalibrator.from_path(path)


def test_legal_ask_returns_full_structured_abstention_without_private_runtime(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(mcp_server, "_cfg", cfg)
    monkeypatch.setattr(mcp_server, "_use_remote", lambda: False)
    result = json.loads(asyncio.run(legal_ask(LegalAskInput(question="რა ამბობს კანონი?"))))
    assert result["outcome"] == "abstain"
    assert result["abstention_reason"] == "generator_unavailable"
    assert result["claims"] == []
    assert result["versions"]["corpus_generation"] == cfg.generation_id
    assert result["trace_id"] == result["trace"]["trace_id"]


def test_legal_get_context_returns_untruncated_evidence_pack(monkeypatch):
    cfg = _cfg()
    pack = EvidencePack("p" * 64, cfg.generation_id, "r" * 64, (), 0, 12000)

    class Service:
        def get_context(self, evidence_id, **kwargs):
            assert evidence_id == "ev1.locator.hash"
            assert kwargs == {"neighbor_chunks": 1, "max_tokens": 12000}
            return pack

    monkeypatch.setattr(mcp_server, "_cfg", cfg)
    monkeypatch.setattr(mcp_server, "_client", object())
    monkeypatch.setattr(mcp_server, "_use_remote", lambda: False)
    monkeypatch.setattr(mcp_server, "_answer_service", lambda *a, **k: Service())
    result = json.loads(asyncio.run(
        legal_get_context(LegalGetContextInput(evidence_id="ev1.locator.hash"))
    ))
    assert result["pack_id"] == "p" * 64
    assert result["items"] == []


def test_answer_runtime_requires_explicit_provider_versions(monkeypatch):
    monkeypatch.setattr(mcp_server, "_cfg", _cfg())
    with pytest.raises(ValueError, match="pinned non-empty version"):
        mcp_server._install_answer_runtime(composer=object())
    pinned = SimpleNamespace(version="private-revision-1")
    mcp_server._install_answer_runtime(
        translator=pinned, composer=pinned, calibrator=pinned
    )


def test_production_runtime_requires_exact_retrieval_attestations(monkeypatch):
    cfg = _cfg(production=True)
    monkeypatch.setattr(mcp_server, "_cfg", cfg)
    with pytest.raises(ValueError, match="retriever and reranker"):
        mcp_server._install_answer_runtime()
    wrong = _retrieval_attestations(cfg)
    wrong["retriever_attestation"] = dataclasses.replace(
        wrong["retriever_attestation"], model_id="wrong/model"
    )
    with pytest.raises(ValueError, match="does not match configured"):
        mcp_server._install_answer_runtime(**wrong)


def test_production_runtime_verifies_provider_artifact_and_pipeline_binding(
    monkeypatch, tmp_path
):
    cfg = _cfg(production=True)
    translator = _provider(ModelRole.TRANSLATOR, "private/mt", "1" * 40, "mt-v1")
    composer = _provider(ModelRole.GENERATOR, "private/llm", "2" * 40, "llm-v1")
    binding = mcp_server._expected_calibration_binding(
        cfg, translator=translator, composer=composer
    )
    calibrator = _calibrator(tmp_path, binding)
    monkeypatch.setattr(mcp_server, "_cfg", cfg)
    mcp_server._install_answer_runtime(
        translator=translator,
        composer=composer,
        calibrator=calibrator,
        **_retrieval_attestations(cfg),
    )
    assert mcp_server._answer_calibrator is calibrator


def test_production_runtime_rejects_duck_type_unverified_hash_and_wrong_binding(
    monkeypatch, tmp_path
):
    cfg = _cfg(production=True)
    translator = _provider(ModelRole.TRANSLATOR, "private/mt", "1" * 40, "mt-v1")
    composer = _provider(ModelRole.GENERATOR, "private/llm", "2" * 40, "llm-v1")
    kwargs = _retrieval_attestations(cfg)
    monkeypatch.setattr(mcp_server, "_cfg", cfg)
    duck = SimpleNamespace(version="heldout-selective-risk/v2:" + "f" * 64)
    with pytest.raises(ValueError, match="HeldOutRiskCalibrator"):
        mcp_server._install_answer_runtime(
            translator=translator, composer=composer, calibrator=duck, **kwargs
        )
    valid = _calibrator(
        tmp_path,
        mcp_server._expected_calibration_binding(
            cfg, translator=translator, composer=composer
        ),
    )
    unverified = HeldOutRiskCalibrator(valid.artifact, valid.artifact_sha256)
    with pytest.raises(ValueError, match="hash must be verified"):
        mcp_server._install_answer_runtime(
            translator=translator,
            composer=composer,
            calibrator=unverified,
            **kwargs,
        )
    wrong_cfg = dataclasses.replace(cfg, generation_id="generation-20260716")
    wrong = _calibrator(
        tmp_path,
        mcp_server._expected_calibration_binding(
            wrong_cfg, translator=translator, composer=composer
        ),
    )
    with pytest.raises(ValueError, match="pipeline binding"):
        mcp_server._install_answer_runtime(
            translator=translator, composer=composer, calibrator=wrong, **kwargs
        )


def test_production_runtime_rejects_provider_attestation_identity_mismatch(
    monkeypatch, tmp_path
):
    cfg = _cfg(production=True)
    translator = _provider(ModelRole.TRANSLATOR, "private/mt", "1" * 40, "mt-v1")
    composer = _provider(ModelRole.GENERATOR, "private/llm", "2" * 40, "llm-v1")
    composer.license_attestation = dataclasses.replace(
        composer.license_attestation, version="different"
    )
    calibrator = _calibrator(
        tmp_path,
        mcp_server._expected_calibration_binding(
            cfg, translator=translator, composer=composer
        ),
    )
    monkeypatch.setattr(mcp_server, "_cfg", cfg)
    with pytest.raises(ValueError, match="identity mismatch"):
        mcp_server._install_answer_runtime(
            translator=translator,
            composer=composer,
            calibrator=calibrator,
            **_retrieval_attestations(cfg),
        )


@pytest.mark.parametrize("remote", [False, True])
def test_legal_ask_failures_always_return_complete_deterministic_answer_result(
    monkeypatch, remote
):
    cfg = _cfg()
    monkeypatch.setattr(mcp_server, "_cfg", cfg)
    monkeypatch.setattr(mcp_server, "_use_remote", lambda: remote)
    if remote:
        async def failed_remote(*args, **kwargs):
            return "Error (remote backend): unavailable"

        monkeypatch.setattr(mcp_server, "_remote_op", failed_remote)
        reason = "remote_answer_pipeline_error"
    else:
        def failed_service(*args, **kwargs):
            raise RuntimeError("unexpected local failure")

        monkeypatch.setattr(mcp_server, "_answer_service", failed_service)
        reason = "answer_pipeline_error"
    question = LegalAskInput(question="რა ამბობს კანონი?")
    first = json.loads(asyncio.run(legal_ask(question)))
    second = json.loads(asyncio.run(legal_ask(question)))
    assert first["outcome"] == "abstain"
    assert first["abstention_reason"] == reason
    assert first["claims"] == [] and first["evidence"] is None
    assert set(first["versions"]) == {
        "corpus_generation", "retriever", "reranker", "translator",
        "generator", "prompt", "calibrator",
    }
    assert first["trace"] is not None
    assert first["trace_id"] == first["trace"]["trace_id"] == second["trace_id"]


def test_answer_runtime_installs_and_passes_freshness_guard(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(mcp_server, "_cfg", cfg)

    class Guard:
        generation_id = cfg.generation_id
        active_manifest_sha256 = "b" * 64

    monkeypatch.setattr(mcp_server, "VerifiedFreshnessGuard", Guard)
    guard = Guard()
    mcp_server._install_answer_runtime(freshness_guard=guard)
    captured = {}

    class Service:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(mcp_server, "LegalAnswerService", Service)
    mcp_server._answer_service(cfg, None)
    assert captured["freshness_guard"] is guard


def test_production_freshness_guard_must_match_worker_manifest(monkeypatch):
    cfg = _cfg(production=True, worker=True)
    manifest_sha256 = "e" * 64
    monkeypatch.setattr(mcp_server, "_cfg", cfg)
    monkeypatch.setattr(
        mcp_server,
        "_worker_readiness_probe",
        lambda: {
            "ok": True,
            "generation_id": cfg.generation_id,
            "generation_manifest_sha256": manifest_sha256,
        },
    )

    class Guard:
        generation_id = cfg.generation_id

        def __init__(self, active_manifest_sha256):
            self.active_manifest_sha256 = active_manifest_sha256

    monkeypatch.setattr(mcp_server, "VerifiedFreshnessGuard", Guard)
    with pytest.raises(ValueError, match="manifest does not match"):
        mcp_server._install_answer_runtime(
            freshness_guard=Guard("f" * 64),
            **_retrieval_attestations(cfg),
        )
    valid = Guard(manifest_sha256)
    mcp_server._install_answer_runtime(
        freshness_guard=valid,
        **_retrieval_attestations(cfg),
    )
    assert mcp_server._answer_freshness_guard is valid
