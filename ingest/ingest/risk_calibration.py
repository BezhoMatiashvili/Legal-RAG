"""Pinned held-out selective-risk calibration for strict legal answers.

The reranker sigmoid is deliberately absent from this module.  A calibration artifact
combines the retrieval/identity/version/evidence/validator features defined by the answer
contract into a scalar selective-risk score, then permits answers only in empirical score
bands whose one-sided Wilson error bounds satisfy the legal release limits.  No artifact is
shipped as "good": deployments must fit and review one on leakage-free held-out data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import InitVar, dataclass, field
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from .legal_answer import CalibrationDecision, CalibrationFeatures

SCHEMA_VERSION = "heldout-selective-risk/v2"
FEATURE_NAMES = (
    "identity_confidence",
    "top_result_margin",
    "route_agreement",
    "translation_agreement",
    "version_certainty",
    "evidence_coverage",
    "validator_passed",
    "not_degraded",
)
_SHA256_LENGTH = 64
STRICT_CONFIDENCE = 0.95
STRICT_SEVERE_ERROR_UPPER_LIMIT = 0.01
STRICT_MATERIAL_ERROR_UPPER_LIMIT = 0.03


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _one_sided_upper(errors: int, total: int, confidence: float) -> float:
    if total <= 0:
        return 1.0
    z = NormalDist().inv_cdf(confidence)
    proportion = errors / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half = z * sqrt(
        proportion * (1.0 - proportion) / total
        + z2 / (4.0 * total * total)
    ) / denominator
    return min(1.0, center + half)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, where: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise ValueError(f"{where} fields mismatch: missing={missing}, unknown={unknown}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate calibration artifact key: {key!r}")
        value[key] = item
    return value


@dataclass(frozen=True)
class CalibrationBinding:
    """Exact production tuple on which held-out calibration was measured."""

    generation_id: str
    retriever: str
    reranker: str
    translator: str
    generator: str
    prompt: str
    config_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationBinding:
        expected = {
            "generation_id",
            "retriever",
            "reranker",
            "translator",
            "generator",
            "prompt",
            "config_sha256",
        }
        _exact_keys(value, expected, where="calibration binding")
        binding = cls(**value)
        binding.validate()
        return binding

    def validate(self) -> None:
        for name in (
            "generation_id",
            "retriever",
            "reranker",
            "translator",
            "generator",
            "prompt",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"calibration binding {name} must be a non-empty exact string")
        if not _is_sha256(self.config_sha256):
            raise ValueError("calibration binding config_sha256 must be lowercase SHA-256")


@dataclass(frozen=True)
class EmpiricalRiskBand:
    lower_score: float
    upper_score: float
    answered: int
    severe_errors: int
    material_errors: int
    answered_clusters: int
    severe_error_clusters: int
    material_error_clusters: int

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EmpiricalRiskBand:
        _exact_keys(
            value,
            {
                "lower_score",
                "upper_score",
                "answered",
                "severe_errors",
                "material_errors",
                "answered_clusters",
                "severe_error_clusters",
                "material_error_clusters",
            },
            where="risk band",
        )
        band = cls(**value)
        band.validate()
        return band

    def validate(self) -> None:
        if not all(
            type(value) in (int, float) and math.isfinite(value)
            for value in (self.lower_score, self.upper_score)
        ):
            raise ValueError("risk band score bounds must be finite")
        if self.upper_score <= self.lower_score:
            raise ValueError("risk band upper_score must exceed lower_score")
        if (
            type(self.answered) is not int
            or type(self.severe_errors) is not int
            or type(self.material_errors) is not int
            or not 0 <= self.severe_errors <= self.material_errors <= self.answered
        ):
            raise ValueError("risk band error counts are invalid")
        if (
            type(self.answered_clusters) is not int
            or type(self.severe_error_clusters) is not int
            or type(self.material_error_clusters) is not int
            or not 0
            <= self.severe_error_clusters
            <= self.material_error_clusters
            <= self.answered_clusters
            <= self.answered
        ):
            raise ValueError("risk band cluster error counts are invalid")


@dataclass(frozen=True)
class CalibrationArtifact:
    calibration_set_sha256: str
    document_version_split_sha256: str
    pipeline_binding: CalibrationBinding
    feature_weights: Mapping[str, float]
    intercept: float
    missing_margin_value: float
    bands: tuple[EmpiricalRiskBand, ...]
    minimum_band_answered: int = 1000
    confidence: float = STRICT_CONFIDENCE
    severe_error_upper_limit: float = STRICT_SEVERE_ERROR_UPPER_LIMIT
    material_error_upper_limit: float = STRICT_MATERIAL_ERROR_UPPER_LIMIT
    held_out: bool = True
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline_binding, CalibrationBinding):
            raise TypeError("pipeline_binding must be CalibrationBinding")
        if not isinstance(self.feature_weights, Mapping):
            raise TypeError("feature_weights must be a mapping")
        object.__setattr__(
            self,
            "feature_weights",
            MappingProxyType(dict(self.feature_weights)),
        )
        object.__setattr__(self, "bands", tuple(self.bands))
        self.validate()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CalibrationArtifact:
        expected = {
            "schema_version",
            "calibration_set_sha256",
            "document_version_split_sha256",
            "pipeline_binding",
            "feature_weights",
            "intercept",
            "missing_margin_value",
            "bands",
            "minimum_band_answered",
            "confidence",
            "severe_error_upper_limit",
            "material_error_upper_limit",
            "held_out",
        }
        _exact_keys(value, expected, where="calibration artifact")
        bands = value["bands"]
        if not isinstance(bands, list) or any(not isinstance(item, Mapping) for item in bands):
            raise ValueError("calibration artifact bands must be an array of objects")
        binding = value["pipeline_binding"]
        if not isinstance(binding, Mapping):
            raise ValueError("calibration artifact pipeline_binding must be an object")
        artifact = cls(
            schema_version=value["schema_version"],
            calibration_set_sha256=value["calibration_set_sha256"],
            document_version_split_sha256=value["document_version_split_sha256"],
            pipeline_binding=CalibrationBinding.from_dict(binding),
            feature_weights=dict(value["feature_weights"]),
            intercept=value["intercept"],
            missing_margin_value=value["missing_margin_value"],
            bands=tuple(EmpiricalRiskBand.from_dict(item) for item in bands),
            minimum_band_answered=value["minimum_band_answered"],
            confidence=value["confidence"],
            severe_error_upper_limit=value["severe_error_upper_limit"],
            material_error_upper_limit=value["material_error_upper_limit"],
            held_out=value["held_out"],
        )
        return artifact

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported calibration schema: {self.schema_version!r}")
        if not _is_sha256(self.calibration_set_sha256) or not _is_sha256(
            self.document_version_split_sha256
        ):
            raise ValueError("calibration and split hashes must be full lowercase SHA-256")
        if self.held_out is not True:
            raise ValueError("calibration artifact must attest leakage-free held-out fitting")
        self.pipeline_binding.validate()
        if set(self.feature_weights) != set(FEATURE_NAMES):
            raise ValueError("calibration artifact must bind the complete feature schema")
        numeric = (
            self.intercept,
            self.missing_margin_value,
            self.confidence,
            self.severe_error_upper_limit,
            self.material_error_upper_limit,
            *self.feature_weights.values(),
        )
        if not all(
            type(value) in (int, float) and math.isfinite(value) for value in numeric
        ):
            raise ValueError("calibration numeric values must be finite")
        if self.confidence != STRICT_CONFIDENCE:
            raise ValueError("strict calibration confidence must equal 0.95")
        if self.severe_error_upper_limit != STRICT_SEVERE_ERROR_UPPER_LIMIT:
            raise ValueError("strict severe-error upper limit must equal 0.01")
        if self.material_error_upper_limit != STRICT_MATERIAL_ERROR_UPPER_LIMIT:
            raise ValueError("strict material-error upper limit must equal 0.03")
        if type(self.minimum_band_answered) is not int or self.minimum_band_answered < 1000:
            raise ValueError("minimum_band_answered must be at least 1000")
        if not self.bands:
            raise ValueError("calibration artifact has no empirical risk bands")
        previous_upper: float | None = None
        for band in self.bands:
            band.validate()
            if previous_upper is not None and band.lower_score < previous_upper:
                raise ValueError("calibration risk bands overlap")
            previous_upper = band.upper_score


def _artifact_from_raw(raw: bytes) -> CalibrationArtifact:
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid calibration artifact JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("calibration artifact must be a JSON object")
    return CalibrationArtifact.from_dict(value)


@dataclass(frozen=True)
class HeldOutRiskCalibrator:
    """Deterministic calibrator backed by one checksum-pinned held-out artifact."""

    artifact: CalibrationArtifact
    artifact_sha256: str
    artifact_bytes: InitVar[bytes | None] = None
    artifact_hash_verified: bool = field(init=False)
    held_out: ClassVar[bool] = True

    def __post_init__(self, artifact_bytes: bytes | None) -> None:
        if not isinstance(self.artifact, CalibrationArtifact):
            raise TypeError("artifact must be CalibrationArtifact")
        self.artifact.validate()
        if not _is_sha256(self.artifact_sha256):
            raise ValueError("artifact_sha256 must be a full lowercase SHA-256")
        verified = False
        if artifact_bytes is not None:
            if not isinstance(artifact_bytes, bytes):
                raise TypeError("artifact_bytes must be bytes")
            if _sha256_bytes(artifact_bytes) != self.artifact_sha256:
                raise ValueError("artifact bytes do not match artifact_sha256")
            if _artifact_from_raw(artifact_bytes) != self.artifact:
                raise ValueError("artifact bytes do not encode the supplied artifact")
            verified = True
        object.__setattr__(self, "artifact_hash_verified", verified)

    @property
    def version(self) -> str:
        return f"{SCHEMA_VERSION}:{self.artifact_sha256}"

    @classmethod
    def from_path(cls, path: str | Path) -> HeldOutRiskCalibrator:
        raw = Path(path).read_bytes()
        return cls(
            _artifact_from_raw(raw),
            artifact_sha256=_sha256_bytes(raw),
            artifact_bytes=raw,
        )

    def _features(self, features: CalibrationFeatures) -> dict[str, float] | None:
        margin = (
            self.artifact.missing_margin_value
            if features.top_result_margin is None
            else features.top_result_margin
        )
        values = {
            "identity_confidence": features.identity_confidence,
            "top_result_margin": margin,
            "route_agreement": features.route_agreement,
            "translation_agreement": features.translation_agreement,
            "version_certainty": features.version_certainty,
            "evidence_coverage": features.evidence_coverage,
            "validator_passed": float(features.validator_passed),
            "not_degraded": float(not features.degraded),
        }
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in values.values()
        ):
            return None
        return values

    def assess(self, features: CalibrationFeatures) -> CalibrationDecision:
        if features.degraded:
            return CalibrationDecision(False, "calibration_rejects_degraded_retrieval")
        if not features.validator_passed:
            return CalibrationDecision(False, "calibration_rejects_failed_validation")
        values = self._features(features)
        if values is None:
            return CalibrationDecision(False, "calibration_feature_out_of_distribution")
        score = self.artifact.intercept + sum(
            self.artifact.feature_weights[name] * values[name] for name in FEATURE_NAMES
        )
        matching = [
            band
            for band in self.artifact.bands
            if band.lower_score <= score
            and score < band.upper_score
        ]
        if len(matching) != 1:
            return CalibrationDecision(False, "calibration_score_out_of_support")
        band = matching[0]
        if band.answered < self.artifact.minimum_band_answered:
            return CalibrationDecision(False, "calibration_band_underpowered")
        severe_upper = _one_sided_upper(
            band.severe_errors, band.answered, self.artifact.confidence
        )
        material_upper = _one_sided_upper(
            band.material_errors, band.answered, self.artifact.confidence
        )
        severe_upper = max(
            severe_upper,
            _one_sided_upper(
                band.severe_error_clusters,
                band.answered_clusters,
                self.artifact.confidence,
            ),
        )
        material_upper = max(
            material_upper,
            _one_sided_upper(
                band.material_error_clusters,
                band.answered_clusters,
                self.artifact.confidence,
            ),
        )
        if severe_upper >= self.artifact.severe_error_upper_limit:
            return CalibrationDecision(False, "calibrated_severe_risk_above_limit")
        if material_upper >= self.artifact.material_error_upper_limit:
            return CalibrationDecision(False, "calibrated_material_risk_above_limit")
        return CalibrationDecision(True)
