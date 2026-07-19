"""Fail-closed validation of the checked-in court-extraction trust attestation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TRUST_SCHEMA_VERSION = "court-extraction-trust/v1"
EXPECTED_THRESHOLDS = {
    "disposition_accuracy_confident": 0.95,
    "disposition_coverage": 0.80,
    "judge_f1": 0.97,
}
DEFAULT_TRUST_PATH = Path(__file__).resolve().parent / "data" / "court_extraction_trust.json"
_EXTRACTOR_PATH = Path(__file__).resolve().with_name("court_extract.py")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOP_LEVEL_KEYS = {
    "schema_version",
    "extractor_revision",
    "extractor_source_sha256",
    "eval_set_hash",
    "result_hash",
    "covered_sources",
    "thresholds",
    "metrics",
    "passed",
}


@dataclass(frozen=True)
class CourtExtractionTrustDecision:
    trusted: bool
    reasons: tuple[str, ...]
    extractor_revision: str
    eval_set_hash: str | None = None
    result_hash: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


def _strict_json(raw: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        raw,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision(
    revision: str,
    *reasons: str,
    value: dict[str, Any] | None = None,
) -> CourtExtractionTrustDecision:
    raw_metrics = (value or {}).get("metrics", {})
    return CourtExtractionTrustDecision(
        trusted=not reasons,
        reasons=tuple(reasons),
        extractor_revision=revision,
        eval_set_hash=(value or {}).get("eval_set_hash"),
        result_hash=(value or {}).get("result_hash"),
        metrics=dict(raw_metrics) if isinstance(raw_metrics, dict) else {},
    )


def evaluate_court_extraction_trust(
    *,
    collection_revision: str | None,
    path: Path | None = None,
    expected_eval_set_hash: str | None = None,
    extractor_path: Path | None = None,
) -> CourtExtractionTrustDecision:
    """Validate runtime code, collection identity, and fixed metric thresholds.

    This boundary never raises for a missing or corrupt artifact: statistics remain usable
    as explicitly untrusted diagnostics and receive stable machine-readable reason codes.
    """

    try:
        from .court_extract import EXTRACTOR_REVISION
    except Exception:
        return _decision("unknown", "extractor_unavailable")

    trust_path = Path(path) if path is not None else DEFAULT_TRUST_PATH
    source_path = Path(extractor_path) if extractor_path is not None else _EXTRACTOR_PATH
    if trust_path.is_symlink():
        return _decision(EXTRACTOR_REVISION, "attestation_symlink")
    if not trust_path.is_file():
        return _decision(EXTRACTOR_REVISION, "attestation_missing")
    try:
        value = _strict_json(trust_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return _decision(EXTRACTOR_REVISION, "attestation_invalid_json")
    if not isinstance(value, dict) or set(value) != _TOP_LEVEL_KEYS:
        return _decision(EXTRACTOR_REVISION, "attestation_invalid_schema")

    reasons: list[str] = []
    if value.get("schema_version") != TRUST_SCHEMA_VERSION:
        reasons.append("attestation_schema_version_mismatch")
    if value.get("extractor_revision") != EXTRACTOR_REVISION:
        reasons.append("attestation_extractor_revision_mismatch")
    if collection_revision is not None and collection_revision != EXTRACTOR_REVISION:
        reasons.append("collection_extractor_revision_mismatch")
    if expected_eval_set_hash is not None and value.get("eval_set_hash") != expected_eval_set_hash:
        reasons.append("eval_set_hash_mismatch")

    for field_name in (
        "extractor_source_sha256",
        "eval_set_hash",
        "result_hash",
    ):
        if not isinstance(value.get(field_name), str) or not _SHA256_RE.fullmatch(
            value[field_name]
        ):
            reasons.append(f"invalid_{field_name}")
    try:
        observed_source_hash = _sha256(source_path)
    except OSError:
        reasons.append("extractor_source_unreadable")
    else:
        if value.get("extractor_source_sha256") != observed_source_hash:
            reasons.append("extractor_source_hash_mismatch")

    if value.get("covered_sources") != ["ecd", "supremecourt"]:
        reasons.append("covered_sources_mismatch")
    if value.get("thresholds") != EXPECTED_THRESHOLDS:
        reasons.append("thresholds_mismatch")

    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(EXPECTED_THRESHOLDS):
        reasons.append("metrics_invalid")
    else:
        for name, threshold in EXPECTED_THRESHOLDS.items():
            metric = metrics.get(name)
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or not math.isfinite(float(metric))
                or not 0.0 <= float(metric) <= 1.0
            ):
                reasons.append(f"metric_invalid:{name}")
            elif float(metric) < threshold:
                reasons.append(f"metric_below_threshold:{name}")
    if value.get("passed") is not True:
        reasons.append("attestation_not_passed")

    return _decision(EXTRACTOR_REVISION, *dict.fromkeys(reasons), value=value)


__all__ = [
    "DEFAULT_TRUST_PATH",
    "EXPECTED_THRESHOLDS",
    "TRUST_SCHEMA_VERSION",
    "CourtExtractionTrustDecision",
    "evaluate_court_extraction_trust",
]
