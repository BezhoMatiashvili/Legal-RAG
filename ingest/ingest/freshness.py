"""Offline freshness and completeness audits for immutable corpus generations.

The auditor deliberately has no crawler, HTTP, Qdrant, or alias-promotion code.  It
combines a checksum-verified immutable generation with an immutable observation file
written by the ingestion workflow.  Reports are content-bound to both inputs and are
created without replacement outside the generation directory.

The observation file is strict JSON with this shape::

    {
      "schema_version": 1,
      "generation_id": "gen-20260715",
      "captured_at": "2026-07-15T04:00:00Z",
      "sources": [{
        "source": "matsne",
        "covered_run_id": "matsne-20260715t030000z",
        "official_last_success_at": "2026-07-15T03:30:00Z",
        "official_document_count": 100,
        "versioned_document_count": 100,
        "version_lineage_complete_document_count": 100
      }]
    }

``official_document_count`` is the number of authoritative version records eligible for
this corpus: distinct ``(source, document_id, version_id)`` identities, not unique logical
documents, pages, or an estimate.  Full-text and quarantine counts are never trusted from
this file: they are derived from the generation's document and quarantine ledgers.
Version-lineage counts remain observation fields because the current generation document
ledger does not carry per-document lineage completeness metadata; their exact input bytes
are hashed into every report.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .generation import (
    MANIFEST_FILENAME,
    GenerationArtifacts,
    GenerationFormatError,
    load_generation,
    parse_rfc3339_utc,
    validate_generation_id,
)
from .generation_snapshot import QUARANTINE_FILENAME

AUDIT_INPUT_SCHEMA_VERSION = 1
AUDIT_REPORT_SCHEMA_VERSION = 1
RATIO_SCALE = 10_000
PRIVATE_FILE_MODE = 0o600


class FreshnessAuditError(ValueError):
    """An audit input, immutable ledger, or report is invalid or ambiguous."""


@dataclass(frozen=True)
class SourceSLA:
    """Maximum source age and minimum canonical-corpus completeness."""

    source: str
    max_official_success_age_hours: int
    minimum_full_text_coverage_basis_points: int = RATIO_SCALE
    maximum_quarantine_rate_basis_points: int = 0
    minimum_version_lineage_coverage_basis_points: int | None = None

    def __post_init__(self) -> None:
        if not self.source or not self.source.strip():
            raise ValueError("source SLA requires a non-empty source")
        if (
            isinstance(self.max_official_success_age_hours, bool)
            or not isinstance(self.max_official_success_age_hours, int)
            or self.max_official_success_age_hours < 1
        ):
            raise ValueError("source SLA max age must be at least one hour")
        for name in (
            "minimum_full_text_coverage_basis_points",
            "maximum_quarantine_rate_basis_points",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= RATIO_SCALE
            ):
                raise ValueError(f"{name} must be between 0 and {RATIO_SCALE}")
        lineage = self.minimum_version_lineage_coverage_basis_points
        if lineage is not None and (
            isinstance(lineage, bool)
            or not isinstance(lineage, int)
            or not 0 <= lineage <= RATIO_SCALE
        ):
            raise ValueError(
                "minimum_version_lineage_coverage_basis_points must be null or "
                f"between 0 and {RATIO_SCALE}"
            )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# Nightly ingestion is the operating target.  The extra allowance prevents a crawl that
# straddles midnight from immediately failing while still refusing multi-day staleness for
# legislation and high-change administrative sources.  Completeness is deliberately exact:
# an incomplete official record belongs in quarantine, and a source-wide current-law answer
# cannot silently assume that the missing record is irrelevant.
DEFAULT_SOURCE_SLAS: Mapping[str, SourceSLA] = MappingProxyType(
    {
        "matsne": SourceSLA(
            "matsne", 36, minimum_version_lineage_coverage_basis_points=RATIO_SCALE
        ),
        "napr": SourceSLA("napr", 36),
        "tas": SourceSLA("tas", 36),
        "constcourt": SourceSLA("constcourt", 48),
        "supremecourt": SourceSLA("supremecourt", 48),
        "ecd": SourceSLA("ecd", 72),
        "tbappeal": SourceSLA("tbappeal", 72),
    }
)


@dataclass(frozen=True)
class SourceObservation:
    source: str
    covered_run_id: str | None
    official_last_success_at: str | None
    official_document_count: int
    versioned_document_count: int
    version_lineage_complete_document_count: int

    @classmethod
    def from_dict(cls, value: Any, *, index: int) -> SourceObservation:
        data = _object(value, field=f"sources[{index}]")
        _exact_keys(
            data,
            {
                "source",
                "covered_run_id",
                "official_last_success_at",
                "official_document_count",
                "versioned_document_count",
                "version_lineage_complete_document_count",
            },
            field=f"sources[{index}]",
        )
        source = _nonempty_string(data["source"], field=f"sources[{index}].source")
        covered_run_id = _optional_string(
            data["covered_run_id"], field=f"sources[{index}].covered_run_id"
        )
        last_success = data["official_last_success_at"]
        if last_success is not None:
            _timestamp(last_success, field=f"sources[{index}].official_last_success_at")
        if (covered_run_id is None) != (last_success is None):
            raise FreshnessAuditError(
                f"sources[{index}] must set covered_run_id and "
                "official_last_success_at together"
            )
        official_count = _integer(
            data["official_document_count"],
            field=f"sources[{index}].official_document_count",
        )
        versioned_count = _integer(
            data["versioned_document_count"],
            field=f"sources[{index}].versioned_document_count",
        )
        complete_count = _integer(
            data["version_lineage_complete_document_count"],
            field=f"sources[{index}].version_lineage_complete_document_count",
        )
        if versioned_count > official_count:
            raise FreshnessAuditError(
                f"sources[{index}].versioned_document_count exceeds official inventory"
            )
        if complete_count > versioned_count:
            raise FreshnessAuditError(
                f"sources[{index}].version_lineage_complete_document_count exceeds "
                "versioned_document_count"
            )
        return cls(
            source=source,
            covered_run_id=covered_run_id,
            official_last_success_at=last_success,
            official_document_count=official_count,
            versioned_document_count=versioned_count,
            version_lineage_complete_document_count=complete_count,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FreshnessAuditInput:
    schema_version: int
    generation_id: str
    captured_at: str
    sources: tuple[SourceObservation, ...]

    @classmethod
    def from_dict(cls, value: Any) -> FreshnessAuditInput:
        data = _object(value, field="audit_input")
        _exact_keys(
            data,
            {"schema_version", "generation_id", "captured_at", "sources"},
            field="audit_input",
        )
        schema_version = _integer(data["schema_version"], field="schema_version")
        if schema_version != AUDIT_INPUT_SCHEMA_VERSION:
            raise FreshnessAuditError(
                f"unsupported audit input schema_version {schema_version}; "
                f"expected {AUDIT_INPUT_SCHEMA_VERSION}"
            )
        generation_id = validate_generation_id(data["generation_id"])
        captured_at = data["captured_at"]
        captured = _timestamp(captured_at, field="captured_at")
        raw_sources = data["sources"]
        if not isinstance(raw_sources, list):
            raise FreshnessAuditError("sources must be a JSON array")
        sources = tuple(
            SourceObservation.from_dict(item, index=index)
            for index, item in enumerate(raw_sources)
        )
        names = [source.source for source in sources]
        if len(names) != len(set(names)):
            raise FreshnessAuditError("sources contains duplicate source names")
        for source in sources:
            if source.official_last_success_at is not None:
                succeeded = _timestamp(
                    source.official_last_success_at,
                    field=f"sources[{source.source}].official_last_success_at",
                )
                if succeeded > captured:
                    raise FreshnessAuditError(
                        f"{source.source} official_last_success_at is after captured_at"
                    )
        return cls(
            schema_version=schema_version,
            generation_id=generation_id,
            captured_at=captured_at,
            sources=tuple(sorted(sources, key=lambda item: item.source)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "captured_at": self.captured_at,
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass(frozen=True)
class AuditBreach:
    code: str
    source: str | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any, *, index: int) -> AuditBreach:
        data = _object(value, field=f"breaches[{index}]")
        _exact_keys(data, {"code", "source", "detail"}, field=f"breaches[{index}]")
        source = data["source"]
        if source is not None:
            source = _nonempty_string(source, field=f"breaches[{index}].source")
        return cls(
            code=_nonempty_string(data["code"], field=f"breaches[{index}].code"),
            source=source,
            detail=_nonempty_string(data["detail"], field=f"breaches[{index}].detail"),
        )


@dataclass(frozen=True)
class SourceAuditResult:
    source: str
    ok: bool
    covered_run_id: str | None
    official_last_success_at: str | None
    freshness_deadline: str | None
    age_seconds: int | None
    official_document_count: int | None
    generation_document_count: int
    full_text_document_count: int
    full_text_coverage: float | None
    quarantine_document_count: int
    quarantine_rate: float | None
    versioned_document_count: int | None
    version_lineage_complete_document_count: int | None
    version_lineage_coverage: float | None
    sla: SourceSLA | None
    breach_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["breach_codes"] = list(self.breach_codes)
        return data

    @classmethod
    def from_dict(cls, value: Any, *, index: int) -> SourceAuditResult:
        data = _object(value, field=f"report.sources[{index}]")
        expected = {
            "source",
            "ok",
            "covered_run_id",
            "official_last_success_at",
            "freshness_deadline",
            "age_seconds",
            "official_document_count",
            "generation_document_count",
            "full_text_document_count",
            "full_text_coverage",
            "quarantine_document_count",
            "quarantine_rate",
            "versioned_document_count",
            "version_lineage_complete_document_count",
            "version_lineage_coverage",
            "sla",
            "breach_codes",
        }
        _exact_keys(data, expected, field=f"report.sources[{index}]")
        last_success = _optional_timestamp(
            data["official_last_success_at"],
            field=f"report.sources[{index}].official_last_success_at",
        )
        deadline = _optional_timestamp(
            data["freshness_deadline"],
            field=f"report.sources[{index}].freshness_deadline",
        )
        raw_sla = data["sla"]
        sla = None if raw_sla is None else _sla_from_dict(raw_sla, index=index)
        raw_codes = data["breach_codes"]
        if not isinstance(raw_codes, list):
            raise FreshnessAuditError(
                f"report.sources[{index}].breach_codes must be an array"
            )
        codes = tuple(
            _nonempty_string(code, field=f"report.sources[{index}].breach_codes")
            for code in raw_codes
        )
        result = cls(
            source=_nonempty_string(
                data["source"], field=f"report.sources[{index}].source"
            ),
            ok=_boolean(data["ok"], field=f"report.sources[{index}].ok"),
            covered_run_id=_optional_string(
                data["covered_run_id"],
                field=f"report.sources[{index}].covered_run_id",
            ),
            official_last_success_at=last_success,
            freshness_deadline=deadline,
            age_seconds=_optional_integer(
                data["age_seconds"], field=f"report.sources[{index}].age_seconds"
            ),
            official_document_count=_optional_integer(
                data["official_document_count"],
                field=f"report.sources[{index}].official_document_count",
            ),
            generation_document_count=_integer(
                data["generation_document_count"],
                field=f"report.sources[{index}].generation_document_count",
            ),
            full_text_document_count=_integer(
                data["full_text_document_count"],
                field=f"report.sources[{index}].full_text_document_count",
            ),
            full_text_coverage=_optional_ratio(
                data["full_text_coverage"],
                field=f"report.sources[{index}].full_text_coverage",
            ),
            quarantine_document_count=_integer(
                data["quarantine_document_count"],
                field=f"report.sources[{index}].quarantine_document_count",
            ),
            quarantine_rate=_optional_ratio(
                data["quarantine_rate"],
                field=f"report.sources[{index}].quarantine_rate",
            ),
            versioned_document_count=_optional_integer(
                data["versioned_document_count"],
                field=f"report.sources[{index}].versioned_document_count",
            ),
            version_lineage_complete_document_count=_optional_integer(
                data["version_lineage_complete_document_count"],
                field=f"report.sources[{index}].version_lineage_complete_document_count",
            ),
            version_lineage_coverage=_optional_ratio(
                data["version_lineage_coverage"],
                field=f"report.sources[{index}].version_lineage_coverage",
            ),
            sla=sla,
            breach_codes=codes,
        )
        if result.ok != (not result.breach_codes):
            raise FreshnessAuditError(
                f"report.sources[{index}].ok disagrees with breach_codes"
            )
        if result.sla is not None and result.sla.source != result.source:
            raise FreshnessAuditError(
                f"report.sources[{index}].sla source does not match result source"
            )
        return result


@dataclass(frozen=True)
class FreshnessAuditReport:
    schema_version: int
    audit_id: str
    ok: bool
    generation_id: str
    generation_manifest_sha256: str
    audit_input_sha256: str
    captured_at: str
    audited_at: str
    sources: tuple[SourceAuditResult, ...]
    breaches: tuple[AuditBreach, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "audit_id": self.audit_id,
            "ok": self.ok,
            "generation_id": self.generation_id,
            "generation_manifest_sha256": self.generation_manifest_sha256,
            "audit_input_sha256": self.audit_input_sha256,
            "captured_at": self.captured_at,
            "audited_at": self.audited_at,
            "sources": [source.to_dict() for source in self.sources],
            "breaches": [breach.to_dict() for breach in self.breaches],
        }

    @classmethod
    def from_dict(cls, value: Any) -> FreshnessAuditReport:
        data = _object(value, field="report")
        expected = {
            "schema_version",
            "audit_id",
            "ok",
            "generation_id",
            "generation_manifest_sha256",
            "audit_input_sha256",
            "captured_at",
            "audited_at",
            "sources",
            "breaches",
        }
        _exact_keys(data, expected, field="report")
        schema_version = _integer(data["schema_version"], field="report.schema_version")
        if schema_version != AUDIT_REPORT_SCHEMA_VERSION:
            raise FreshnessAuditError(
                f"unsupported audit report schema_version {schema_version}; "
                f"expected {AUDIT_REPORT_SCHEMA_VERSION}"
            )
        raw_sources = data["sources"]
        raw_breaches = data["breaches"]
        if not isinstance(raw_sources, list) or not isinstance(raw_breaches, list):
            raise FreshnessAuditError("report sources and breaches must be arrays")
        sources = tuple(
            SourceAuditResult.from_dict(item, index=index)
            for index, item in enumerate(raw_sources)
        )
        if [item.source for item in sources] != sorted(item.source for item in sources):
            raise FreshnessAuditError("report sources must be sorted by source")
        if len({item.source for item in sources}) != len(sources):
            raise FreshnessAuditError("report contains duplicate sources")
        breaches = tuple(
            AuditBreach.from_dict(item, index=index)
            for index, item in enumerate(raw_breaches)
        )
        report = cls(
            schema_version=schema_version,
            audit_id=_sha256(data["audit_id"], field="report.audit_id"),
            ok=_boolean(data["ok"], field="report.ok"),
            generation_id=validate_generation_id(data["generation_id"]),
            generation_manifest_sha256=_sha256(
                data["generation_manifest_sha256"],
                field="report.generation_manifest_sha256",
            ),
            audit_input_sha256=_sha256(
                data["audit_input_sha256"], field="report.audit_input_sha256"
            ),
            captured_at=_timestamp_text(
                data["captured_at"], field="report.captured_at"
            ),
            audited_at=_timestamp_text(data["audited_at"], field="report.audited_at"),
            sources=sources,
            breaches=breaches,
        )
        if report.ok != (not report.breaches):
            raise FreshnessAuditError("report.ok disagrees with breaches")
        _validate_report_consistency(report)
        expected_id = _compute_audit_id(report.to_dict())
        if report.audit_id != expected_id:
            raise FreshnessAuditError("report audit_id does not match report content")
        return report


@dataclass(frozen=True)
class CurrentLawDecision:
    """Fail-closed eligibility decision; it never represents answer confidence."""

    outcome: Literal["eligible", "abstain"]
    generation_id: str
    audit_id: str
    relevant_sources: tuple[str, ...]
    reasons: tuple[AuditBreach, ...]

    @property
    def allowed(self) -> bool:
        return self.outcome == "eligible"

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "generation_id": self.generation_id,
            "audit_id": self.audit_id,
            "relevant_sources": list(self.relevant_sources),
            "reasons": [reason.to_dict() for reason in self.reasons],
        }


@dataclass(frozen=True)
class VerifiedFreshnessGuard:
    """Bind one verified audit report to the exact active generation manifest."""

    report: FreshnessAuditReport
    active_generation_id: str
    active_manifest_sha256: str

    def __post_init__(self) -> None:
        validate_generation_id(self.active_generation_id)
        _sha256(self.active_manifest_sha256, field="active_manifest_sha256")
        if self.report.generation_id != self.active_generation_id:
            raise FreshnessAuditError(
                "freshness report generation_id does not match active generation"
            )
        if self.report.generation_manifest_sha256 != self.active_manifest_sha256:
            raise FreshnessAuditError(
                "freshness report manifest hash does not match active generation"
            )

    @classmethod
    def from_files(
        cls,
        generation_dir: str | Path,
        report_path: str | Path,
    ) -> VerifiedFreshnessGuard:
        """Checksum-load both immutable artifacts and bind them for serving."""

        artifacts = load_generation(generation_dir)
        manifest_sha256 = artifacts.checksums.files.get(MANIFEST_FILENAME)
        if manifest_sha256 is None:
            raise FreshnessAuditError(
                "active generation checksum inventory omits manifest.json"
            )
        report = load_audit_report(report_path)
        return cls(
            report=report,
            active_generation_id=artifacts.manifest.generation_id,
            active_manifest_sha256=manifest_sha256,
        )

    @property
    def audit_id(self) -> str:
        return self.report.audit_id

    @property
    def generation_id(self) -> str:
        return self.active_generation_id

    def decision(
        self,
        relevant_sources: tuple[str, ...] | list[str] | set[str],
        *,
        now: datetime | None = None,
    ) -> CurrentLawDecision:
        """Re-evaluate bound source deadlines at request time."""

        return current_law_decision(self.report, relevant_sources, now=now)


@dataclass
class _Inventory:
    document_count: int = 0
    full_text_count: int = 0
    quarantine_count: int = 0


def load_audit_input(path: str | Path) -> tuple[FreshnessAuditInput, str]:
    """Load strict JSON and return the parsed snapshot plus its exact SHA-256."""

    raw = _read_regular_file(Path(path))
    value = _parse_json(raw, origin=str(path))
    return FreshnessAuditInput.from_dict(value), hashlib.sha256(raw).hexdigest()


def load_audit_report(path: str | Path) -> FreshnessAuditReport:
    """Load a strict, self-hash-verified audit report."""

    raw = _read_regular_file(Path(path))
    return FreshnessAuditReport.from_dict(_parse_json(raw, origin=str(path)))


def write_audit_input(
    path: str | Path,
    audit_input: FreshnessAuditInput,
) -> Path:
    """Atomically create an owner-only observation snapshot without replacement."""

    value = audit_input.to_dict()
    FreshnessAuditInput.from_dict(value)
    return _write_immutable_json(path, value, artifact="audit input")


def audit_generation(
    artifacts: GenerationArtifacts,
    audit_input: FreshnessAuditInput,
    *,
    audit_input_sha256: str,
    audited_at: datetime,
    slas: Mapping[str, SourceSLA] = DEFAULT_SOURCE_SLAS,
) -> FreshnessAuditReport:
    """Audit one already-loaded generation without changing it or external state."""

    if audit_input.generation_id != artifacts.manifest.generation_id:
        raise FreshnessAuditError(
            "audit input generation_id does not match immutable generation"
        )
    _sha256(audit_input_sha256, field="audit_input_sha256")
    now = _aware_utc(audited_at, field="audited_at")
    captured = _timestamp(audit_input.captured_at, field="captured_at")
    if captured > now:
        raise FreshnessAuditError("audit input captured_at is after audited_at")

    inventories: dict[str, _Inventory] = {}
    document_identities: set[tuple[str, str, str]] = set()
    excluded_identities: set[tuple[str, str, str]] = set()
    for document in artifacts.iter_documents():
        key = (document.source, document.document_id, document.version_id)
        if key in document_identities:
            raise FreshnessAuditError(f"duplicate generation document identity: {key}")
        document_identities.add(key)
        inventory = inventories.setdefault(document.source, _Inventory())
        inventory.document_count += 1
        if (
            document.indexed
            and document.content_complete
            and document.extraction_status == "full_text"
        ):
            inventory.full_text_count += 1
        if not document.indexed:
            inventory.quarantine_count += 1
            excluded_identities.add(key)

    quarantine_identities = _quarantine_identities(
        artifacts.root / QUARANTINE_FILENAME,
        generation_id=artifacts.manifest.generation_id,
    )
    observations = {item.source: item for item in audit_input.sources}
    source_names = set(inventories) | set(observations)
    source_names.update(run.source for run in artifacts.manifest.covered_runs)
    covered_run_ids: dict[str, set[str]] = {}
    for run in artifacts.manifest.covered_runs:
        covered_run_ids.setdefault(run.source, set()).add(run.run_id)

    breaches: list[AuditBreach] = []
    if len(document_identities) != artifacts.manifest.document_count:
        breaches.append(
            AuditBreach(
                "manifest_document_count_mismatch",
                None,
                f"manifest={artifacts.manifest.document_count}, ledger={len(document_identities)}",
            )
        )
    if len(excluded_identities) != artifacts.manifest.excluded_document_count:
        breaches.append(
            AuditBreach(
                "manifest_quarantine_count_mismatch",
                None,
                f"manifest={artifacts.manifest.excluded_document_count}, "
                f"document_ledger={len(excluded_identities)}",
            )
        )
    if quarantine_identities != excluded_identities:
        breaches.append(
            AuditBreach(
                "quarantine_ledger_mismatch",
                None,
                f"quarantine_only={len(quarantine_identities - excluded_identities)}, "
                f"documents_only={len(excluded_identities - quarantine_identities)}",
            )
        )

    source_results: list[SourceAuditResult] = []
    for source in sorted(source_names):
        inventory = inventories.get(source, _Inventory())
        observation = observations.get(source)
        sla = slas.get(source)
        result, source_breaches = _audit_source(
            source=source,
            inventory=inventory,
            observation=observation,
            sla=sla,
            audited_at=now,
            covered_run_ids=covered_run_ids.get(source, set()),
        )
        source_results.append(result)
        breaches.extend(source_breaches)

    manifest_sha256 = artifacts.checksums.files.get(MANIFEST_FILENAME)
    if manifest_sha256 is None:
        raise FreshnessAuditError("generation checksum inventory omits manifest.json")
    base = FreshnessAuditReport(
        schema_version=AUDIT_REPORT_SCHEMA_VERSION,
        audit_id="0" * 64,
        ok=not breaches,
        generation_id=artifacts.manifest.generation_id,
        generation_manifest_sha256=manifest_sha256,
        audit_input_sha256=audit_input_sha256,
        captured_at=audit_input.captured_at,
        audited_at=_format_utc(now),
        sources=tuple(source_results),
        breaches=tuple(breaches),
    )
    return replace(base, audit_id=_compute_audit_id(base.to_dict()))


def audit_generation_directory(
    generation_dir: str | Path,
    audit_input_path: str | Path,
    *,
    audited_at: datetime | None = None,
    slas: Mapping[str, SourceSLA] = DEFAULT_SOURCE_SLAS,
) -> FreshnessAuditReport:
    """Checksum both audit inputs and return a deterministic report."""

    artifacts = load_generation(generation_dir)
    audit_input, input_sha256 = load_audit_input(audit_input_path)
    return audit_generation(
        artifacts,
        audit_input,
        audit_input_sha256=input_sha256,
        audited_at=audited_at or datetime.now(timezone.utc),
        slas=slas,
    )


def current_law_decision(
    report: FreshnessAuditReport,
    relevant_sources: tuple[str, ...] | list[str] | set[str],
    *,
    now: datetime | None = None,
) -> CurrentLawDecision:
    """Allow current-law composition only while every relevant source remains valid.

    Source age is recomputed against ``now``.  This prevents a report that passed on
    Sunday from authorizing answers after its underlying source SLA expires.  A caller
    must provide the sources selected by the intent/entity planner; an empty or unknown
    set abstains rather than silently treating the audit as global permission.
    """

    checked_at = _aware_utc(now or datetime.now(timezone.utc), field="now")
    audited_at = _timestamp(report.audited_at, field="report.audited_at")
    reasons: list[AuditBreach] = []
    normalized_sources: set[str] = set()
    for source_name in relevant_sources:
        if not isinstance(source_name, str) or not source_name.strip():
            reasons.append(
                AuditBreach(
                    "relevant_source_invalid",
                    None,
                    "planner supplied an empty or non-string relevant source",
                )
            )
        else:
            normalized_sources.add(source_name)
    sources = tuple(sorted(normalized_sources))
    if not sources:
        reasons.append(
            AuditBreach(
                "relevant_sources_missing",
                None,
                "current-law eligibility requires at least one planner-selected source",
            )
        )
    if checked_at < audited_at:
        reasons.append(
            AuditBreach(
                "audit_from_future",
                None,
                "serving clock precedes the report audit timestamp",
            )
        )

    by_source = {source.source: source for source in report.sources}
    for breach in report.breaches:
        if breach.source is None:
            reasons.append(breach)
    for source_name in sources:
        source = by_source.get(source_name)
        if source is None:
            reasons.append(
                AuditBreach(
                    "source_not_audited",
                    source_name,
                    "relevant source is absent from the bound freshness report",
                )
            )
            continue
        reasons.extend(
            breach for breach in report.breaches if breach.source == source_name
        )
        if source.sla is None:
            # ``source_sla_missing`` should already be a report breach, but retain this
            # independent guard for reports constructed by older callers.
            if not any(
                reason.source == source_name and reason.code == "source_sla_missing"
                for reason in reasons
            ):
                reasons.append(
                    AuditBreach(
                        "source_sla_missing",
                        source_name,
                        "no current-law freshness policy is bound to this source",
                    )
                )
            continue
        if source.official_last_success_at is None:
            continue
        last_success = _timestamp(
            source.official_last_success_at,
            field=f"{source_name}.official_last_success_at",
        )
        deadline = last_success + timedelta(
            hours=source.sla.max_official_success_age_hours
        )
        if checked_at > deadline and not any(
            reason.source == source_name and reason.code == "official_success_stale"
            for reason in reasons
        ):
            reasons.append(
                AuditBreach(
                    "official_success_stale",
                    source_name,
                    f"last official-source success expired at {_format_utc(deadline)}",
                )
            )

    unique_reasons = tuple(
        dict.fromkeys((reason.code, reason.source, reason.detail) for reason in reasons)
    )
    normalized_reasons = tuple(
        AuditBreach(code=code, source=source, detail=detail)
        for code, source, detail in unique_reasons
    )
    return CurrentLawDecision(
        outcome="abstain" if normalized_reasons else "eligible",
        generation_id=report.generation_id,
        audit_id=report.audit_id,
        relevant_sources=sources,
        reasons=normalized_reasons,
    )


def write_audit_report(path: str | Path, report: FreshnessAuditReport) -> Path:
    """Atomically create an owner-only report; an existing path is never replaced."""

    value = report.to_dict()
    FreshnessAuditReport.from_dict(value)
    return _write_immutable_json(path, value, artifact="audit report")


def _write_immutable_json(
    path: str | Path,
    value: Mapping[str, object],
    *,
    artifact: str,
) -> Path:
    destination = Path(path)
    parent = destination.parent
    try:
        parent_mode = parent.lstat().st_mode
    except OSError as exc:
        raise FreshnessAuditError(
            f"cannot stat report directory {parent}: {exc}"
        ) from exc
    if stat.S_ISLNK(parent_mode) or not stat.S_ISDIR(parent_mode):
        raise FreshnessAuditError(
            f"{artifact} parent is not a real directory: {parent}"
        )
    if os.path.lexists(destination):
        raise FileExistsError(f"{artifact} already exists: {destination}")

    payload = _canonical_json_bytes(value, pretty=True)
    temporary = parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, PRIVATE_FILE_MODE)
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write to {temporary}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        # A hard-link publication is atomic and fails if destination appeared after the
        # pre-check.  It also avoids a replace-capable rename primitive.
        os.link(temporary, destination, follow_symlinks=False)
        os.unlink(temporary)
        _fsync_directory(parent)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return destination


def report_filename(report: FreshnessAuditReport) -> str:
    """Return a collision-resistant filename for a weekly immutable audit report."""

    timestamp = report.audited_at.replace("-", "").replace(":", "").replace(".", "")
    return (
        f"{report.generation_id}.freshness-audit.{timestamp}."
        f"{report.audit_id[:12]}.json"
    )


def _audit_source(
    *,
    source: str,
    inventory: _Inventory,
    observation: SourceObservation | None,
    sla: SourceSLA | None,
    audited_at: datetime,
    covered_run_ids: set[str],
) -> tuple[SourceAuditResult, list[AuditBreach]]:
    breaches: list[AuditBreach] = []

    def breach(code: str, detail: str) -> None:
        breaches.append(AuditBreach(code=code, source=source, detail=detail))

    if observation is None:
        breach(
            "source_observation_missing",
            "generation source has no official-source observation",
        )
    elif (
        observation.covered_run_id is not None
        and observation.covered_run_id not in covered_run_ids
    ):
        breach(
            "covered_run_not_in_generation",
            f"official success run {observation.covered_run_id!r} is not in manifest.covered_runs",
        )
    if sla is None:
        breach(
            "source_sla_missing", "source has no configured freshness/completeness SLA"
        )

    last_success = None if observation is None else observation.official_last_success_at
    deadline: str | None = None
    age_seconds: int | None = None
    if last_success is None:
        breach(
            "official_success_missing",
            "no successful official-source crawl is recorded",
        )
    else:
        succeeded = _timestamp(last_success, field=f"{source}.official_last_success_at")
        age_seconds = max(0, int((audited_at - succeeded).total_seconds()))
        if sla is not None:
            deadline_dt = succeeded + timedelta(
                hours=sla.max_official_success_age_hours
            )
            deadline = _format_utc(deadline_dt)
            if audited_at > deadline_dt:
                breach(
                    "official_success_stale",
                    f"last success {_format_utc(succeeded)} exceeded deadline {deadline}",
                )

    official_count = (
        None if observation is None else observation.official_document_count
    )
    full_coverage = _ratio(inventory.full_text_count, official_count)
    quarantine_rate = _ratio(inventory.quarantine_count, official_count)
    if official_count is not None:
        if official_count == 0:
            breach(
                "official_inventory_empty",
                "official eligible document inventory is empty",
            )
        if inventory.document_count != official_count:
            breach(
                "generation_inventory_mismatch",
                f"official={official_count}, generation={inventory.document_count}",
            )
        if sla is not None and not _meets_minimum(
            inventory.full_text_count,
            official_count,
            sla.minimum_full_text_coverage_basis_points,
        ):
            breach(
                "full_text_coverage_below_sla",
                f"full_text={inventory.full_text_count}/{official_count}, "
                f"minimum_basis_points={sla.minimum_full_text_coverage_basis_points}",
            )
        if sla is not None and not _meets_maximum(
            inventory.quarantine_count,
            official_count,
            sla.maximum_quarantine_rate_basis_points,
        ):
            breach(
                "quarantine_rate_above_sla",
                f"quarantined={inventory.quarantine_count}/{official_count}, "
                f"maximum_basis_points={sla.maximum_quarantine_rate_basis_points}",
            )

    versioned_count = (
        None if observation is None else observation.versioned_document_count
    )
    complete_count = (
        None
        if observation is None
        else observation.version_lineage_complete_document_count
    )
    lineage_coverage = _ratio(complete_count, versioned_count)
    if (
        sla is not None
        and sla.minimum_version_lineage_coverage_basis_points is not None
    ):
        if versioned_count in {None, 0}:
            breach(
                "version_lineage_missing",
                "source requires version lineage but no versioned documents are recorded",
            )
        elif complete_count is not None and not _meets_minimum(
            complete_count,
            versioned_count,
            sla.minimum_version_lineage_coverage_basis_points,
        ):
            breach(
                "version_lineage_coverage_below_sla",
                f"complete={complete_count}/{versioned_count}, "
                "minimum_basis_points="
                f"{sla.minimum_version_lineage_coverage_basis_points}",
            )

    return (
        SourceAuditResult(
            source=source,
            ok=not breaches,
            covered_run_id=None if observation is None else observation.covered_run_id,
            official_last_success_at=last_success,
            freshness_deadline=deadline,
            age_seconds=age_seconds,
            official_document_count=official_count,
            generation_document_count=inventory.document_count,
            full_text_document_count=inventory.full_text_count,
            full_text_coverage=full_coverage,
            quarantine_document_count=inventory.quarantine_count,
            quarantine_rate=quarantine_rate,
            versioned_document_count=versioned_count,
            version_lineage_complete_document_count=complete_count,
            version_lineage_coverage=lineage_coverage,
            sla=sla,
            breach_codes=tuple(item.code for item in breaches),
        ),
        breaches,
    )


def _quarantine_identities(
    path: Path, *, generation_id: str
) -> set[tuple[str, str, str]]:
    raw = _read_regular_file(path)
    identities: set[tuple[str, str, str]] = set()
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            raise FreshnessAuditError(f"{path}:{line_number}: blank JSONL line")
        value = _parse_json(line, origin=f"{path}:{line_number}")
        data = _object(value, field=f"{path}:{line_number}")
        if data.get("generation_id") != generation_id:
            raise FreshnessAuditError(
                f"{path}:{line_number}: quarantine generation_id mismatch"
            )
        source = _nonempty_string(
            data.get("source"), field=f"{path}:{line_number}.source"
        )
        document_id = _nonempty_string(
            data.get("document_id"), field=f"{path}:{line_number}.document_id"
        )
        version_id = _nonempty_string(
            data.get("version_id"), field=f"{path}:{line_number}.version_id"
        )
        if data.get("exclusion_reason") in {None, ""}:
            raise FreshnessAuditError(
                f"{path}:{line_number}: quarantined document lacks exclusion_reason"
            )
        identity = (source, document_id, version_id)
        if identity in identities:
            raise FreshnessAuditError(f"duplicate quarantine identity: {identity}")
        identities.add(identity)
    return identities


def _sla_from_dict(value: Any, *, index: int) -> SourceSLA:
    data = _object(value, field=f"report.sources[{index}].sla")
    _exact_keys(
        data,
        {
            "source",
            "max_official_success_age_hours",
            "minimum_full_text_coverage_basis_points",
            "maximum_quarantine_rate_basis_points",
            "minimum_version_lineage_coverage_basis_points",
        },
        field=f"report.sources[{index}].sla",
    )
    lineage = data["minimum_version_lineage_coverage_basis_points"]
    if lineage is not None:
        lineage = _integer(lineage, field=f"report.sources[{index}].sla.lineage")
    return SourceSLA(
        source=_nonempty_string(
            data["source"], field=f"report.sources[{index}].sla.source"
        ),
        max_official_success_age_hours=_integer(
            data["max_official_success_age_hours"],
            field=f"report.sources[{index}].sla.max_age",
        ),
        minimum_full_text_coverage_basis_points=_integer(
            data["minimum_full_text_coverage_basis_points"],
            field=f"report.sources[{index}].sla.min_full_text",
        ),
        maximum_quarantine_rate_basis_points=_integer(
            data["maximum_quarantine_rate_basis_points"],
            field=f"report.sources[{index}].sla.max_quarantine",
        ),
        minimum_version_lineage_coverage_basis_points=lineage,
    )


def _ratio(numerator: int | None, denominator: int | None) -> float | None:
    if numerator is None or denominator in {None, 0}:
        return None
    # Inventory mismatch is reported separately.  Coverage/rate remain valid ratios even
    # when a malformed observation claims a denominator smaller than the generation.
    return round(min(numerator / denominator, 1.0), 8)


def _meets_minimum(numerator: int, denominator: int, basis_points: int) -> bool:
    return denominator > 0 and numerator * RATIO_SCALE >= denominator * basis_points


def _meets_maximum(numerator: int, denominator: int, basis_points: int) -> bool:
    return denominator > 0 and numerator * RATIO_SCALE <= denominator * basis_points


def _compute_audit_id(value: Mapping[str, object]) -> str:
    material = dict(value)
    material.pop("audit_id", None)
    return hashlib.sha256(_canonical_json_bytes(material, pretty=False)).hexdigest()


def _validate_report_consistency(report: FreshnessAuditReport) -> None:
    captured = _timestamp(report.captured_at, field="report.captured_at")
    audited = _timestamp(report.audited_at, field="report.audited_at")
    if captured > audited:
        raise FreshnessAuditError("report captured_at is after audited_at")

    for source in report.sources:
        expected_codes = tuple(
            breach.code for breach in report.breaches if breach.source == source.source
        )
        if source.breach_codes != expected_codes:
            raise FreshnessAuditError(
                f"report source {source.source!r} breach_codes disagree with breaches"
            )
        if source.full_text_document_count > source.generation_document_count:
            raise FreshnessAuditError(
                f"report source {source.source!r} full-text count exceeds generation count"
            )
        if source.quarantine_document_count > source.generation_document_count:
            raise FreshnessAuditError(
                f"report source {source.source!r} quarantine count exceeds generation count"
            )
        if source.full_text_coverage != _ratio(
            source.full_text_document_count, source.official_document_count
        ):
            raise FreshnessAuditError(
                f"report source {source.source!r} full-text coverage is inconsistent"
            )
        if source.quarantine_rate != _ratio(
            source.quarantine_document_count, source.official_document_count
        ):
            raise FreshnessAuditError(
                f"report source {source.source!r} quarantine rate is inconsistent"
            )
        if (
            source.version_lineage_complete_document_count is not None
            and source.versioned_document_count is not None
            and source.version_lineage_complete_document_count
            > source.versioned_document_count
        ):
            raise FreshnessAuditError(
                f"report source {source.source!r} complete lineage count exceeds versioned count"
            )
        if source.version_lineage_coverage != _ratio(
            source.version_lineage_complete_document_count,
            source.versioned_document_count,
        ):
            raise FreshnessAuditError(
                f"report source {source.source!r} lineage coverage is inconsistent"
            )

        if source.official_last_success_at is None:
            if source.covered_run_id is not None:
                raise FreshnessAuditError(
                    f"report source {source.source!r} has a run without a success timestamp"
                )
            if source.freshness_deadline is not None or source.age_seconds is not None:
                raise FreshnessAuditError(
                    f"report source {source.source!r} missing success has derived freshness values"
                )
            continue
        if source.covered_run_id is None:
            raise FreshnessAuditError(
                f"report source {source.source!r} success lacks a covered run"
            )
        succeeded = _timestamp(
            source.official_last_success_at,
            field=f"report.sources[{source.source}].official_last_success_at",
        )
        if succeeded > captured:
            raise FreshnessAuditError(
                f"report source {source.source!r} success is after captured_at"
            )
        expected_age = max(0, int((audited - succeeded).total_seconds()))
        if source.age_seconds != expected_age:
            raise FreshnessAuditError(
                f"report source {source.source!r} age_seconds is inconsistent"
            )
        expected_deadline = (
            None
            if source.sla is None
            else _format_utc(
                succeeded + timedelta(hours=source.sla.max_official_success_age_hours)
            )
        )
        if source.freshness_deadline != expected_deadline:
            raise FreshnessAuditError(
                f"report source {source.source!r} freshness deadline is inconsistent"
            )


def _canonical_json_bytes(value: object, *, pretty: bool) -> bytes:
    try:
        if pretty:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        else:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
    except (TypeError, ValueError) as exc:
        raise FreshnessAuditError(f"value is not strict JSON: {exc}") from exc
    return f"{encoded}\n".encode("utf-8")


def _read_regular_file(path: Path) -> bytes:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise FreshnessAuditError(f"cannot stat {path}: {exc}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise FreshnessAuditError(f"expected a regular non-symlink file: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise FreshnessAuditError(f"cannot read {path}: {exc}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FreshnessAuditError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _parse_json(raw: bytes, *, origin: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except FreshnessAuditError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise FreshnessAuditError(f"{origin}: invalid JSON: {exc}") from exc


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FreshnessAuditError(f"{field} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise FreshnessAuditError(
            f"{field} has invalid keys (missing={missing}, unknown={extra})"
        )


def _nonempty_string(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(ord(char) < 32 for char in value)
    ):
        raise FreshnessAuditError(f"{field} must be a non-empty control-free string")
    return value


def _optional_string(value: Any, *, field: str) -> str | None:
    return None if value is None else _nonempty_string(value, field=field)


def _integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FreshnessAuditError(f"{field} must be an integer >= 0")
    return value


def _optional_integer(value: Any, *, field: str) -> int | None:
    return None if value is None else _integer(value, field=field)


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise FreshnessAuditError(f"{field} must be a boolean")
    return value


def _optional_ratio(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 <= value <= 1
    ):
        raise FreshnessAuditError(f"{field} must be null or a number between 0 and 1")
    return float(value)


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise FreshnessAuditError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _timestamp(value: Any, *, field: str) -> datetime:
    try:
        return parse_rfc3339_utc(value, field=field)
    except GenerationFormatError as exc:
        raise FreshnessAuditError(str(exc)) from exc


def _timestamp_text(value: Any, *, field: str) -> str:
    _timestamp(value, field=field)
    return value


def _optional_timestamp(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _timestamp_text(value, field=field)


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise FreshnessAuditError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    normalized = _aware_utc(value, field="timestamp")
    return normalized.isoformat().replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "AUDIT_INPUT_SCHEMA_VERSION",
    "AUDIT_REPORT_SCHEMA_VERSION",
    "AuditBreach",
    "CurrentLawDecision",
    "DEFAULT_SOURCE_SLAS",
    "FreshnessAuditError",
    "FreshnessAuditInput",
    "FreshnessAuditReport",
    "SourceAuditResult",
    "SourceObservation",
    "SourceSLA",
    "VerifiedFreshnessGuard",
    "audit_generation",
    "audit_generation_directory",
    "current_law_decision",
    "load_audit_input",
    "load_audit_report",
    "report_filename",
    "write_audit_report",
    "write_audit_input",
]
