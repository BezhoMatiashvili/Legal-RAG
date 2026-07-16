import json
from dataclasses import FrozenInstanceError

import pytest

from ingest.legal_answer import CalibrationFeatures
from ingest.risk_calibration import (
    FEATURE_NAMES,
    CalibrationArtifact,
    HeldOutRiskCalibrator,
)


def _payload(**overrides):
    payload = {
        "schema_version": "heldout-selective-risk/v2",
        "calibration_set_sha256": "a" * 64,
        "document_version_split_sha256": "b" * 64,
        "pipeline_binding": {
            "generation_id": "generation-20260715",
            "retriever": f"BAAI/bge-m3@{'1' * 40}",
            "reranker": f"BAAI/bge-reranker-v2-m3@{'2' * 40}",
            "translator": "translator-v1",
            "generator": "generator-v1",
            "prompt": "prompt-v1",
            "config_sha256": "c" * 64,
        },
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
    payload.update(overrides)
    return payload


def _features(**overrides):
    values = {
        "degraded": False,
        "identity_confidence": 1.0,
        "top_result_margin": 0.5,
        "route_agreement": 1.0,
        "translation_agreement": 1.0,
        "version_certainty": 1.0,
        "evidence_coverage": 1.0,
        "validator_passed": True,
    }
    values.update(overrides)
    return CalibrationFeatures(**values)


def test_pinned_heldout_band_allows_only_empirically_safe_features(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_payload(), sort_keys=True), encoding="utf-8")
    calibrator = HeldOutRiskCalibrator.from_path(path)
    assert calibrator.held_out is True
    assert calibrator.artifact_hash_verified is True
    assert len(calibrator.artifact_sha256) == 64
    assert calibrator.assess(_features()).allow_answer
    assert not calibrator.assess(_features(degraded=True)).allow_answer
    assert not calibrator.assess(_features(validator_passed=False)).allow_answer


def test_empirical_error_bound_and_underpowered_band_reject():
    unsafe = CalibrationArtifact.from_dict(_payload(bands=[{
        "lower_score": 0.0,
        "upper_score": 1.0,
        "answered": 1000,
        "severe_errors": 10,
        "material_errors": 20,
        "answered_clusters": 1000,
        "severe_error_clusters": 10,
        "material_error_clusters": 20,
    }]))
    decision = HeldOutRiskCalibrator(unsafe, artifact_sha256="c" * 64).assess(_features())
    assert not decision.allow_answer
    assert decision.reason == "calibrated_severe_risk_above_limit"

    underpowered = CalibrationArtifact.from_dict(_payload(
        minimum_band_answered=1200,
    ))
    decision = HeldOutRiskCalibrator(
        underpowered, artifact_sha256="d" * 64
    ).assess(_features())
    assert not decision.allow_answer
    assert decision.reason == "calibration_band_underpowered"


def test_empirical_band_uses_cluster_conservative_error_bound():
    correlated = CalibrationArtifact.from_dict(_payload(bands=[{
        "lower_score": 0.0,
        "upper_score": 1.0,
        "answered": 1000,
        "severe_errors": 0,
        "material_errors": 0,
        "answered_clusters": 1,
        "severe_error_clusters": 0,
        "material_error_clusters": 0,
    }]))
    decision = HeldOutRiskCalibrator(
        correlated, artifact_sha256="9" * 64
    ).assess(_features())
    assert not decision.allow_answer
    assert decision.reason == "calibrated_severe_risk_above_limit"


def test_artifact_rejects_missing_features_overlap_and_nonheldout():
    missing = _payload()
    missing["feature_weights"].pop("version_certainty")
    with pytest.raises(ValueError, match="complete feature schema"):
        CalibrationArtifact.from_dict(missing)
    with pytest.raises(ValueError, match="overlap"):
        CalibrationArtifact.from_dict(_payload(bands=[
            {"lower_score": 0.0, "upper_score": 0.6, "answered": 1000,
             "severe_errors": 0, "material_errors": 0,
             "answered_clusters": 1000, "severe_error_clusters": 0,
             "material_error_clusters": 0},
            {"lower_score": 0.5, "upper_score": 1.0, "answered": 1000,
             "severe_errors": 0, "material_errors": 0,
             "answered_clusters": 1000, "severe_error_clusters": 0,
             "material_error_clusters": 0},
        ]))
    with pytest.raises(ValueError, match="held-out"):
        CalibrationArtifact.from_dict(_payload(held_out=False))


def test_artifact_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":"one","schema_version":"two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate calibration artifact key"):
        HeldOutRiskCalibrator.from_path(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("confidence", 0.951, "confidence must equal 0.95"),
        ("severe_error_upper_limit", 0.011, "must equal 0.01"),
        ("material_error_upper_limit", 0.031, "must equal 0.03"),
    ],
)
def test_strict_release_limits_are_not_artifact_configurable(field, value, message):
    with pytest.raises(ValueError, match=message):
        CalibrationArtifact.from_dict(_payload(**{field: value}))


def test_artifact_and_calibrator_are_immutable_and_hash_proven(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(_payload(), sort_keys=True), encoding="utf-8")
    calibrator = HeldOutRiskCalibrator.from_path(path)
    with pytest.raises(TypeError):
        calibrator.artifact.feature_weights["identity_confidence"] = 1.0
    with pytest.raises(FrozenInstanceError):
        calibrator.artifact.confidence = 0.99
    with pytest.raises(FrozenInstanceError):
        calibrator.artifact_sha256 = "f" * 64
    direct = HeldOutRiskCalibrator(
        calibrator.artifact, artifact_sha256=calibrator.artifact_sha256
    )
    assert direct.artifact_hash_verified is False


def test_artifact_requires_complete_exact_pipeline_binding():
    missing = _payload()
    missing["pipeline_binding"].pop("prompt")
    with pytest.raises(ValueError, match="calibration binding fields mismatch"):
        CalibrationArtifact.from_dict(missing)
    invalid_hash = _payload()
    invalid_hash["pipeline_binding"]["config_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="config_sha256"):
        CalibrationArtifact.from_dict(invalid_hash)


def test_out_of_support_and_out_of_distribution_fail_closed():
    artifact = CalibrationArtifact.from_dict(_payload(bands=[{
        "lower_score": 0.6,
        "upper_score": 1.0,
        "answered": 1000,
        "severe_errors": 0,
        "material_errors": 0,
        "answered_clusters": 1000,
        "severe_error_clusters": 0,
        "material_error_clusters": 0,
    }]))
    calibrator = HeldOutRiskCalibrator(artifact, artifact_sha256="e" * 64)
    assert calibrator.assess(_features()).reason == "calibration_score_out_of_support"
    assert calibrator.assess(_features(route_agreement=1.5)).reason == (
        "calibration_feature_out_of_distribution"
    )

    weights = {name: 0.0 for name in FEATURE_NAMES}
    weights["identity_confidence"] = 2.0
    high_score = CalibrationArtifact.from_dict(_payload(feature_weights=weights))
    decision = HeldOutRiskCalibrator(
        high_score, artifact_sha256="f" * 64
    ).assess(_features())
    assert decision.reason == "calibration_score_out_of_support"
