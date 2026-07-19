import hashlib
import importlib.util
import json
import stat
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ingest.freshness import (
    FreshnessAuditError,
    FreshnessAuditInput,
    VerifiedFreshnessGuard,
    audit_generation,
    current_law_decision,
    load_audit_input,
    load_audit_report,
    write_audit_input,
    write_audit_report,
)
from ingest.generation import MANIFEST_FILENAME


GENERATION_ID = "gen-20260715"
AUDITED_AT = datetime(2026, 7, 15, 12, tzinfo=timezone.utc)


class _Artifacts:
    def __init__(self, root: Path, documents: list[SimpleNamespace]):
        self.root = root
        self._documents = documents
        excluded = sum(not document.indexed for document in documents)
        self.manifest = SimpleNamespace(
            generation_id=GENERATION_ID,
            document_count=len(documents),
            excluded_document_count=excluded,
            covered_runs=(SimpleNamespace(source="matsne", run_id="matsne-run-1"),),
        )
        self.checksums = SimpleNamespace(files={MANIFEST_FILENAME: "b" * 64})

    def iter_documents(self):
        return iter(self._documents)


def _document(
    document_id: str,
    *,
    version_id: str = "v1",
    indexed: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        source="matsne",
        document_id=document_id,
        version_id=version_id,
        indexed=indexed,
        content_complete=indexed,
        extraction_status="full_text" if indexed else "malformed",
    )


def _artifacts(tmp_path: Path, documents: list[SimpleNamespace]) -> _Artifacts:
    root = tmp_path / "generation"
    root.mkdir(parents=True)
    quarantine = []
    for document in documents:
        if not document.indexed:
            quarantine.append(
                {
                    "generation_id": GENERATION_ID,
                    "source": document.source,
                    "document_id": document.document_id,
                    "version_id": document.version_id,
                    "exclusion_reason": "incomplete_content",
                }
            )
    (root / "quarantine.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in quarantine),
        encoding="utf-8",
    )
    return _Artifacts(root, documents)


def _input(
    *,
    official_count: int = 2,
    complete_lineages: int = 2,
    last_success: str | None = "2026-07-15T03:00:00Z",
) -> FreshnessAuditInput:
    return FreshnessAuditInput.from_dict(
        {
            "schema_version": 1,
            "generation_id": GENERATION_ID,
            "captured_at": "2026-07-15T04:00:00Z",
            "sources": [
                {
                    "source": "matsne",
                    "covered_run_id": None if last_success is None else "matsne-run-1",
                    "official_last_success_at": last_success,
                    "official_document_count": official_count,
                    "versioned_document_count": official_count,
                    "version_lineage_complete_document_count": complete_lineages,
                }
            ],
        }
    )


def _report(tmp_path: Path, *, incomplete: bool = False, stale: bool = False):
    documents = [_document("one"), _document("two", indexed=not incomplete)]
    audit_input = _input(
        complete_lineages=1 if incomplete else 2,
        last_success="2026-07-13T00:00:00Z" if stale else "2026-07-15T03:00:00Z",
    )
    return audit_generation(
        _artifacts(tmp_path, documents),
        audit_input,
        audit_input_sha256="a" * 64,
        audited_at=AUDITED_AT,
    )


def test_passing_audit_binds_inputs_and_reports_exact_metrics(tmp_path):
    report = _report(tmp_path)

    assert report.ok is True
    assert report.generation_manifest_sha256 == "b" * 64
    assert report.audit_input_sha256 == "a" * 64
    assert len(report.audit_id) == 64
    source = report.sources[0]
    assert source.source == "matsne"
    assert source.full_text_document_count == 2
    assert source.full_text_coverage == 1.0
    assert source.quarantine_document_count == 0
    assert source.version_lineage_coverage == 1.0
    assert source.freshness_deadline == "2026-07-16T15:00:00Z"


def test_multiple_versions_of_one_document_are_distinct_inventory_records(tmp_path):
    report = audit_generation(
        _artifacts(
            tmp_path,
            [
                _document("same-law", version_id="version-2025"),
                _document("same-law", version_id="version-2026"),
            ],
        ),
        _input(official_count=2, complete_lineages=2),
        audit_input_sha256="a" * 64,
        audited_at=AUDITED_AT,
    )

    assert report.ok is True
    assert report.sources[0].generation_document_count == 2
    assert report.sources[0].full_text_document_count == 2


def test_stale_incomplete_generation_reports_each_fail_closed_breach(tmp_path):
    report = _report(tmp_path, incomplete=True, stale=True)
    codes = {breach.code for breach in report.breaches}

    assert report.ok is False
    assert {
        "official_success_stale",
        "full_text_coverage_below_sla",
        "quarantine_rate_above_sla",
        "version_lineage_coverage_below_sla",
    } <= codes
    source = report.sources[0]
    assert source.full_text_coverage == 0.5
    assert source.quarantine_document_count == 1
    assert source.quarantine_rate == 0.5


def test_official_success_must_name_a_run_covered_by_generation(tmp_path):
    value = _input().to_dict()
    value["sources"][0]["covered_run_id"] = "newer-run-not-in-generation"
    report = audit_generation(
        _artifacts(tmp_path, [_document("one"), _document("two")]),
        FreshnessAuditInput.from_dict(value),
        audit_input_sha256="a" * 64,
        audited_at=AUDITED_AT,
    )

    assert report.ok is False
    assert [breach.code for breach in report.breaches] == [
        "covered_run_not_in_generation"
    ]


def test_current_law_rechecks_age_instead_of_trusting_old_pass(tmp_path):
    report = _report(tmp_path)

    passing = current_law_decision(
        report,
        ["matsne"],
        now=datetime(2026, 7, 16, 14, 59, tzinfo=timezone.utc),
    )
    expired = current_law_decision(
        report,
        ["matsne"],
        now=datetime(2026, 7, 16, 15, 1, tzinfo=timezone.utc),
    )

    assert passing.allowed is True
    assert expired.allowed is False
    assert [(reason.source, reason.code) for reason in expired.reasons] == [
        ("matsne", "official_success_stale")
    ]


def test_verified_guard_binds_report_to_active_generation_tuple(tmp_path):
    report = _report(tmp_path)
    guard = VerifiedFreshnessGuard(
        report=report,
        active_generation_id=GENERATION_ID,
        active_manifest_sha256="b" * 64,
    )

    assert guard.generation_id == GENERATION_ID
    assert guard.audit_id == report.audit_id
    assert guard.decision(["matsne"], now=AUDITED_AT).allowed is True

    with pytest.raises(FreshnessAuditError, match="generation_id"):
        VerifiedFreshnessGuard(
            report=report,
            active_generation_id="different-generation",
            active_manifest_sha256="b" * 64,
        )
    with pytest.raises(FreshnessAuditError, match="manifest hash"):
        VerifiedFreshnessGuard(
            report=report,
            active_generation_id=GENERATION_ID,
            active_manifest_sha256="c" * 64,
        )


def test_current_law_refuses_empty_or_unknown_relevant_sources(tmp_path):
    report = _report(tmp_path)

    empty = current_law_decision(report, [], now=AUDITED_AT)
    unknown = current_law_decision(report, ["new-source"], now=AUDITED_AT)

    assert empty.outcome == "abstain"
    assert empty.reasons[0].code == "relevant_sources_missing"
    assert unknown.outcome == "abstain"
    assert unknown.reasons[0].code == "source_not_audited"


def test_report_is_owner_only_self_verifying_and_never_replaced(tmp_path):
    report = _report(tmp_path)
    destination = tmp_path / "report.json"

    write_audit_report(destination, report)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert load_audit_report(destination) == report
    with pytest.raises(FileExistsError):
        write_audit_report(destination, report)

    tampered = report.to_dict()
    tampered["ok"] = False
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(FreshnessAuditError, match="disagrees|audit_id"):
        load_audit_report(tampered_path)


def test_input_loader_hashes_exact_bytes_and_rejects_duplicate_keys(tmp_path):
    value = _input(official_count=1, complete_lineages=1).to_dict()
    path = tmp_path / "input.json"
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)

    loaded, digest = load_audit_input(path)

    assert loaded.generation_id == GENERATION_ID
    assert digest == hashlib.sha256(raw).hexdigest()

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"schema_version":1,"generation_id":"gen-20260715",'
        '"captured_at":"2026-07-15T04:00:00Z","sources":[]}',
        encoding="utf-8",
    )
    with pytest.raises(FreshnessAuditError, match="duplicate JSON key"):
        load_audit_input(duplicate)


def test_audit_input_writer_is_owner_only_and_create_only(tmp_path):
    path = tmp_path / "observation.json"
    write_audit_input(path, _input())

    loaded, _ = load_audit_input(path)
    assert loaded == _input()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        write_audit_input(path, _input())


def test_cli_returns_one_and_still_writes_report_on_sla_breach(tmp_path, monkeypatch):
    report = _report(tmp_path / "audit", incomplete=True)
    script_path = Path(__file__).parents[1] / "scripts" / "audit_corpus_freshness.py"
    spec = importlib.util.spec_from_file_location("audit_corpus_freshness", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "audit_generation_directory", lambda *args, **kwargs: report
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()

    exit_code = module.main(
        ["unused-generation", "unused-input", "--report-dir", str(report_dir)]
    )

    assert exit_code == 1
    written = list(report_dir.glob("*.freshness-audit.*.json"))
    assert len(written) == 1
    assert load_audit_report(written[0]) == report
