"""Focused tests for create-only source-state evidence and strict run consumption."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ingest import snapshot, source_state
from ingest.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scraper"))

from legal_scrapers import completion as scraper_completion  # noqa: E402

RUN_ID = "20260713T100000Z_start-1900-01-01_end-2026-07-13"
ADMIN = "ადმინისტრაციულ საქმეთა პალატა"
CIVIL = "სამოქალაქო საქმეთა პალატა"
CRIMINAL = "სისხლის სამართლის საქმეთა პალატა"
CHAMBERS = (ADMIN, CIVIL, CRIMINAL)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _completion_record(run_dir: Path, source: str) -> dict[str, object]:
    items_path = run_dir / "items.jsonl"
    latest_path = run_dir.parents[1] / "latest" / "items.jsonl"
    items_sha = _sha256(items_path)
    items_size = items_path.stat().st_size
    finish_reason = "finished"
    validation: dict[str, object] = {"kind": "generic", "passed": True}
    if source == "supremecourt":
        finish_reason = "closespider_timeout"
        manifest = run_dir / "partial_manifest.json"
        journal = run_dir / "items.journal.jsonl"
        validation = {
            "kind": "supremecourt_partial_v1",
            "passed": True,
            "validator_schema_version": 1,
            "run_dir": str(run_dir.absolute()),
            "run_id": run_dir.name,
            "finish_reason": finish_reason,
            "items_sha256": items_sha,
            "manifest_path": str(manifest.absolute()),
            "manifest_sha256": _sha256(manifest),
            "journal_path": str(journal.absolute()),
            "journal_sha256": _sha256(journal),
            "unresolved_failure_count": 0,
        }
    files = [
        {
            "role": role,
            "configured_uri": str(path.absolute()),
            "path": str(path.absolute()),
            "size_bytes": items_size,
            "sha256": items_sha,
        }
        for role, path in (("run", items_path), ("latest", latest_path))
    ]
    files.sort(key=lambda row: (row["role"], row["configured_uri"], row["path"]))
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "source": source,
        "spider": source,
        "start_date": "1900-01-01",
        "end_date": "2026-07-13",
        "started_at": "2026-07-13T10:00:00Z",
        "items_path": str(items_path.absolute()),
        "latest_items_path": str(latest_path.absolute()),
        "log_path": str((run_dir / "spider.log").absolute()),
        "outcome": "success",
        "quality_passed": True,
        "feeds_durable": True,
        "failure_count": 0,
        "finish_reason": finish_reason,
        "completed_at": "2026-07-13T14:00:03Z",
        "feed_outputs": {
            "durable": True,
            "configured_count": 2,
            "success_count": 2,
            "failure_count": 0,
            "files": files,
        },
        "quality": {
            "passed": True,
            "quality_failures": 0,
            "spider_errors": 0,
            "spider_exceptions": 0,
            "item_errors": 0,
            "pagination_reconcilers": 1,
            "pagination_reconciled": 1,
        },
        "source_validation": validation,
    }


def test_scraper_producer_round_trips_through_strict_run_consumer(tmp_path):
    artifacts = tmp_path / "artifacts"
    source_root = artifacts / "ecd"
    run_dir = source_root / "runs" / RUN_ID
    latest_dir = source_root / "latest"
    for directory in (source_root, source_root / "runs", run_dir, latest_dir):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
    items_path = run_dir / "items.jsonl"
    latest_items_path = latest_dir / "items.jsonl"
    for path in (items_path, latest_items_path):
        path.write_bytes(b'{"decision_document_id":"one"}\n')
        path.chmod(0o600)
    spider = SimpleNamespace(
        name="ecd",
        run_id=RUN_ID,
        scraping_start_date=date(1900, 1, 1),
        scraping_end_date=date(2026, 7, 13),
        started_at=datetime(2026, 7, 13, 10, tzinfo=UTC),
        source_root=source_root,
        run_dir=run_dir,
        latest_dir=latest_dir,
        items_path=items_path,
        latest_items_path=latest_items_path,
        log_path=run_dir / "spider.log",
        run_metadata_path=run_dir / "run.json",
        latest_metadata_path=latest_dir / "run.json",
    )
    startup = scraper_completion.build_startup_record(spider)
    scraper_completion.publish_startup_metadata(
        spider.run_metadata_path,
        spider.latest_metadata_path,
        startup,
    )
    feeds = scraper_completion.attest_feed_outputs(
        {str(items_path): {}, str(latest_items_path): {}},
        {"feedexport/success_count/FileFeedStorage": 2},
        items_path,
        latest_items_path,
    )
    quality = scraper_completion.evaluate_crawl_quality({}, "ecd", "finished")
    terminal = scraper_completion.build_terminal_record(
        spider,
        finish_reason="finished",
        quality=quality,
        feeds=feeds,
        source_validation=scraper_completion.generic_source_validation(),
        completed_at=datetime(2026, 7, 13, 14, tzinfo=UTC),
    )
    scraper_completion.publish_terminal_record(spider, terminal)

    validated = source_state.validate_selected_run(
        artifacts,
        "ecd",
        RUN_ID,
        now=datetime(2026, 7, 14, tzinfo=UTC),
    )

    assert validated.items.sha256 == feeds.files[0]["sha256"]
    assert validated.completion.sha256 == _sha256(run_dir / "run.json")


def _write_attested_run(
    artifacts: Path,
    source: str,
    *,
    run_id: str = RUN_ID,
    items: bytes = b"",
    strict_supreme: bool = False,
) -> Path:
    run_dir = artifacts / source / "runs" / run_id
    run_dir.mkdir(mode=0o700, parents=True)
    run_dir.chmod(0o700)
    items_path = run_dir / "items.jsonl"
    items_path.write_bytes(items)
    items_path.chmod(0o600)
    if source == "supremecourt":
        if strict_supreme:
            _write_strict_supreme_files(run_dir)
        else:
            (run_dir / "partial_manifest.json").write_text("{}\n", encoding="utf-8")
            (run_dir / "items.journal.jsonl").write_bytes(b"")
            for path in (
                run_dir / "partial_manifest.json",
                run_dir / "items.journal.jsonl",
            ):
                path.chmod(0o600)
    record = _completion_record(run_dir, source)
    payload = source_state.canonical_json_bytes(record)
    run_path = run_dir / "run.json"
    candidate_path = run_dir / source_state.TERMINAL_CANDIDATE_FILENAME
    for path in (run_path, candidate_path):
        path.write_bytes(payload)
        path.chmod(0o600)
    authorization = {
        "schema_version": 1,
        "state": "terminal_authorized",
        "source": source,
        "run_id": run_dir.name,
        "candidate_filename": source_state.TERMINAL_CANDIDATE_FILENAME,
        "terminal_size_bytes": len(payload),
        "terminal_sha256": hashlib.sha256(payload).hexdigest(),
    }
    authorization_path = run_dir / source_state.FINALIZATION_CLAIM_FILENAME
    authorization_path.write_bytes(source_state.canonical_json_bytes(authorization))
    authorization_path.chmod(0o600)
    return run_dir


def _rewrite_terminal_wal(
    run_dir: Path,
    record: dict[str, object],
    *,
    canonical: bool = True,
) -> None:
    payload = (
        source_state.canonical_json_bytes(record)
        if canonical
        else (json.dumps(record, sort_keys=True) + "\n").encode()
    )
    for path in (
        run_dir / "run.json",
        run_dir / source_state.TERMINAL_CANDIDATE_FILENAME,
    ):
        path.write_bytes(payload)
        path.chmod(0o600)
    authorization_path = run_dir / source_state.FINALIZATION_CLAIM_FILENAME
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization["source"] = record.get("source", authorization["source"])
    authorization["run_id"] = record.get("run_id", authorization["run_id"])
    authorization["terminal_size_bytes"] = len(payload)
    authorization["terminal_sha256"] = hashlib.sha256(payload).hexdigest()
    authorization_path.write_bytes(source_state.canonical_json_bytes(authorization))
    authorization_path.chmod(0o600)


def _write_all(artifacts: Path, *, strict_supreme: bool = False) -> list[Path]:
    return [
        _write_attested_run(
            artifacts,
            source,
            strict_supreme=strict_supreme and source == "supremecourt",
        )
        for source in source_state.PRODUCTION_SOURCES
    ]


def _selections(run_dirs: list[Path]) -> list[str]:
    return [f"{path.parents[1].name}:{path.name}" for path in run_dirs]


def _fake_supreme_validator(monkeypatch) -> None:
    def validate_run(run_dir):
        run_dir = Path(run_dir).absolute()
        return {
            "schema_version": 1,
            "valid": True,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "finish_reason": "closespider_timeout",
            "items_sha256": _sha256(run_dir / "items.jsonl"),
            "journal_sha256": _sha256(run_dir / "items.journal.jsonl"),
            "unresolved_failure_count": 0,
            "errors": [],
        }

    monkeypatch.setattr(
        source_state,
        "_load_supremecourt_validator",
        lambda: SimpleNamespace(validate_run=validate_run),
    )


def _write_strict_supreme_files(run_dir: Path) -> None:
    def item(case_id: str, chamber: str, decision_date: str, *, target=False):
        palata = {ADMIN: "0", CIVIL: "1", CRIMINAL: "2"}[chamber]
        body = "საქართველოს უზენაესი სასამართლოს სრული გადაწყვეტილება. " * 8
        if target:
            body += " პირველი ინსტანციის საქმის ნომერია 330100122006207137."
        return {
            "case_id": case_id,
            "case_number": f"ას-{case_id}-2026",
            "chamber": chamber,
            "date": decision_date,
            "source_url": f"https://www.supremecourt.ge/ka/fullcase/{case_id}/{palata}",
            "body_markdown": body,
        }

    rows = [
        item("300", ADMIN, "2026-06-01"),
        item("100", CIVIL, "2026-07-03"),
        item("099", CIVIL, "2026-05-20"),
        item("49251", CRIMINAL, "2026-07-06", target=True),
    ]
    rows.sort(key=lambda row: (row["date"], row["chamber"], row["case_id"]), reverse=True)
    journal = [row for row in rows if row["case_id"] in {"49251", "100"}]
    items_path = run_dir / "items.jsonl"
    journal_path = run_dir / "items.journal.jsonl"
    items_path.write_bytes(_jsonl(rows))
    journal_path.write_bytes(_jsonl(journal))
    journal_ids = {(row["case_id"], row["chamber"]) for row in journal}
    per_chamber = {}
    for chamber in CHAMBERS:
        chamber_rows = [row for row in rows if row["chamber"] == chamber]
        dates = [row["date"] for row in chamber_rows]
        new = sum(
            (row["case_id"], row["chamber"]) in journal_ids for row in chamber_rows
        )
        per_chamber[chamber] = {
            "known_items": len(chamber_rows) - new,
            "new_items": new,
            "total_items": len(chamber_rows),
            "newest_date": max(dates) if dates else None,
            "oldest_date": min(dates) if dates else None,
            "resume_cursor": "2026-05-31",
        }
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "started_at": "2026-07-13T10:00:00+00:00",
        "finished_at": "2026-07-13T14:00:03+00:00",
        "finish_reason": "closespider_timeout",
        "partial_by_design": True,
        "date_order": "newest_first",
        "frontier_start_date": "2026-07-13",
        "lower_bound": "1900-01-01",
        "max_runtime_seconds": 14400,
        "elapsed_time_seconds": 14403.0,
        "items_file": str(items_path.absolute()),
        "journal_file": str(journal_path.absolute()),
        "items_sha256": _sha256(items_path),
        "known_items": len(rows) - len(journal),
        "new_items": len(journal),
        "total_items": len(rows),
        "known_encountered": 1,
        "per_chamber": per_chamber,
        "oldest_fully_completed_global_date_frontier": "2026-06-01",
        "per_chamber_resume_cursors": {
            chamber: values["resume_cursor"] for chamber, values in per_chamber.items()
        },
        "completed_windows": [
            {
                "id": f"w{index:06d}",
                "chamber": chamber,
                "start": "2026-06-01",
                "end": "2026-07-13",
                "status": "completed",
                "authoritative_total": 1,
                "known": 1 if chamber == ADMIN else 0,
                "new": 0 if chamber == ADMIN else 1,
                "failures": 0,
            }
            for index, chamber in enumerate(CHAMBERS, 1)
        ],
        "retries": {"total": 0, "retry_after": 0, "parse": 0, "detail_parse": 0},
        "unresolved_failure_count": 0,
        "unresolved_failures_truncated": False,
        "unresolved_failures": [],
    }
    manifest_path = run_dir / "partial_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in (items_path, journal_path, manifest_path):
        path.chmod(0o600)


def test_builder_is_deterministic_private_and_loadable_with_multiple_runs(
    tmp_path, monkeypatch
):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    extra = _write_attested_run(
        artifacts,
        "matsne",
        run_id="20260714T100000Z_start-2026-07-14_end-2026-07-14",
    )
    selections = list(reversed(_selections([*runs, extra])))
    first = tmp_path / "review" / "first.json"
    second = tmp_path / "review" / "second.json"

    built = source_state.build_source_state_evidence(artifacts, first, selections)
    source_state.build_source_state_evidence(artifacts, second, selections)

    assert len(built["runs"]) == 8
    assert first.read_bytes() == second.read_bytes() == source_state.canonical_json_bytes(built)
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    candidate, authorization = source_state._companion_paths(first)
    assert stat.S_IMODE(candidate.stat().st_mode) == 0o600
    assert stat.S_IMODE(authorization.stat().st_mode) == 0o600
    assert first.stat().st_ino == candidate.stat().st_ino
    loaded = source_state.load_source_state_evidence(artifacts, first)
    assert [(run.source, run.run_id) for run in loaded.runs] == sorted(
        (path.parents[1].name, path.name) for path in [*runs, extra]
    )


@pytest.mark.parametrize(
    ("selections", "reason"),
    [
        ([], "at least one explicit"),
        (["matsne:latest"], "latest"),
        (["unknown:run"], "unknown source"),
        (["matsne:not/a/run"], "unsafe run_id"),
    ],
)
def test_builder_rejects_nonexact_or_incomplete_selections(tmp_path, selections, reason):
    with pytest.raises(source_state.SourceStateError, match=reason):
        source_state.build_source_state_evidence(
            tmp_path / "artifacts", tmp_path / "out.json", selections
        )


def test_builder_rejects_missing_source_duplicate_and_identity_mismatch(
    tmp_path, monkeypatch
):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    selections = _selections(runs)
    with pytest.raises(source_state.SourceStateError, match="all seven"):
        source_state.build_source_state_evidence(
            artifacts, tmp_path / "missing.json", selections[:-1]
        )
    with pytest.raises(source_state.SourceStateError, match="duplicate"):
        source_state.build_source_state_evidence(
            artifacts, tmp_path / "duplicate.json", [*selections, selections[0]]
        )

    record_path = runs[0] / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["source"] = "ecd"
    _rewrite_terminal_wal(runs[0], record)
    with pytest.raises(source_state.SourceStateError, match="identity mismatch"):
        source_state.build_source_state_evidence(
            artifacts, tmp_path / "mismatch.json", selections
        )


def test_builder_rejects_startup_only_symlink_and_insecure_modes(tmp_path, monkeypatch):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    selections = _selections(runs)
    _rewrite_terminal_wal(
        runs[0],
        {"schema_version": 1, "run_id": runs[0].name, "outcome": "started"},
    )
    with pytest.raises(source_state.SourceStateError, match="identity mismatch|keys mismatch"):
        source_state.build_source_state_evidence(
            artifacts, tmp_path / "startup.json", selections
        )

    runs = _write_all(tmp_path / "other-artifacts")
    selections = _selections(runs)
    target = tmp_path / "private-items"
    target.write_bytes(b"")
    target.chmod(0o600)
    items = runs[0] / "items.jsonl"
    items.unlink()
    items.symlink_to(target)
    with pytest.raises(source_state.SourceStateError, match="symlink"):
        source_state.build_source_state_evidence(
            tmp_path / "other-artifacts", tmp_path / "symlink.json", selections
        )

    runs = _write_all(tmp_path / "mode-artifacts")
    (runs[0] / "run.json").chmod(0o644)
    with pytest.raises(source_state.SourceStateError, match="mode 0600"):
        source_state.build_source_state_evidence(
            tmp_path / "mode-artifacts", tmp_path / "mode.json", _selections(runs)
        )


def test_builder_rejects_terminal_publication_recovery_guard(tmp_path, monkeypatch):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    guard = runs[0] / source_state.TERMINAL_RECOVERY_GUARD_FILENAME
    guard.write_text("recovery required\n", encoding="utf-8")
    guard.chmod(0o600)

    with pytest.raises(source_state.SourceStateError, match="recovery guard"):
        source_state.build_source_state_evidence(
            artifacts, tmp_path / "guarded.json", _selections(runs)
        )

    assert not (tmp_path / "guarded.json").exists()


@pytest.mark.parametrize(
    "missing_name",
    [
        source_state.FINALIZATION_CLAIM_FILENAME,
        source_state.TERMINAL_CANDIDATE_FILENAME,
    ],
)
def test_builder_requires_both_permanent_terminal_wal_files(
    tmp_path, monkeypatch, missing_name
):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    (runs[0] / missing_name).unlink()

    with pytest.raises(source_state.SourceStateError, match="terminal authorization|candidate"):
        source_state.build_source_state_evidence(
            artifacts, tmp_path / "missing-wal.json", _selections(runs)
        )


def test_builder_rejects_terminal_candidate_hash_drift_and_symlink(
    tmp_path, monkeypatch
):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    candidate = runs[0] / source_state.TERMINAL_CANDIDATE_FILENAME
    candidate.write_bytes(candidate.read_bytes() + b" ")
    candidate.chmod(0o600)
    with pytest.raises(source_state.SourceStateError, match="does not match"):
        source_state.build_source_state_evidence(
            artifacts, tmp_path / "drifted-wal.json", _selections(runs)
        )

    other_artifacts = tmp_path / "other-artifacts"
    other_runs = _write_all(other_artifacts)
    candidate = other_runs[0] / source_state.TERMINAL_CANDIDATE_FILENAME
    target = tmp_path / "candidate-target"
    target.write_bytes(candidate.read_bytes())
    target.chmod(0o600)
    candidate.unlink()
    candidate.symlink_to(target)
    with pytest.raises(source_state.SourceStateError, match="symlink"):
        source_state.build_source_state_evidence(
            other_artifacts,
            tmp_path / "symlinked-wal.json",
            _selections(other_runs),
        )


def test_builder_final_rehash_rejects_late_recovery_guard(tmp_path, monkeypatch):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    real_rehash = source_state._rehash_validated

    def add_guard_then_rehash(validated):
        guard = runs[0] / source_state.TERMINAL_RECOVERY_GUARD_FILENAME
        guard.write_text("late guard\n", encoding="utf-8")
        guard.chmod(0o600)
        real_rehash(validated)

    monkeypatch.setattr(source_state, "_rehash_validated", add_guard_then_rehash)
    with pytest.raises(source_state.SourceStateError, match="recovery guard"):
        source_state.build_source_state_evidence(
            artifacts, tmp_path / "late-guard.json", _selections(runs)
        )
    assert not (tmp_path / "late-guard.json").exists()


def test_builder_rejects_noncanonical_completion_bytes(tmp_path, monkeypatch):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    path = runs[0] / "run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    _rewrite_terminal_wal(runs[0], record, canonical=False)

    with pytest.raises(source_state.SourceStateError, match="canonically serialized"):
        source_state.build_source_state_evidence(
            artifacts, tmp_path / "noncanonical.json", _selections(runs)
        )


def test_builder_detects_changed_inputs_and_never_replaces_output(tmp_path, monkeypatch):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    selections = _selections(runs)
    output = tmp_path / "evidence.json"
    source_state.build_source_state_evidence(artifacts, output, selections)
    original = output.read_bytes()
    with pytest.raises(source_state.SourceStateError, match="already exists"):
        source_state.build_source_state_evidence(artifacts, output, selections)
    assert output.read_bytes() == original

    real_rehash = source_state._rehash_validated

    def mutate_then_rehash(validated):
        validated[0].items.path.write_bytes(b"changed\n")
        real_rehash(validated)

    monkeypatch.setattr(source_state, "_rehash_validated", mutate_then_rehash)
    with pytest.raises(source_state.SourceStateError, match="changed before publication"):
        source_state.build_source_state_evidence(
            artifacts, tmp_path / "changed.json", selections
        )
    assert not (tmp_path / "changed.json").exists()


def _assert_failed_publication_left_no_evidence(output: Path) -> None:
    assert not os.path.lexists(output)
    assert not list(output.parent.glob(".*.tmp"))


def test_one_shot_postlink_fsync_failure_recovers_and_returns_success(
    tmp_path, monkeypatch
):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    output = tmp_path / "review" / "recovered-fsync.json"
    real_fsync = source_state._fsync_publication_parent
    calls = 0

    def fail_once(descriptor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected one-shot parent fsync")
        return real_fsync(descriptor)

    monkeypatch.setattr(source_state, "_fsync_publication_parent", fail_once)
    built = source_state.build_source_state_evidence(
        artifacts, output, _selections(runs)
    )

    assert built["schema_version"] == 1
    assert calls >= 2
    assert len(source_state.load_source_state_evidence(artifacts, output).runs) == 7


def test_compound_postlink_failure_is_authorized_pending_not_fail_open(
    tmp_path, monkeypatch
):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    output = tmp_path / "review" / "fsync-failure.json"
    real_fchmod = source_state.os.fchmod
    real_unlink = source_state.os.unlink
    real_parent_fsync = source_state._fsync_publication_parent

    def fail_fsync(_descriptor):
        raise OSError("injected publication parent fsync")

    def refuse_rollback_mode(descriptor, mode):
        if mode == 0:
            raise OSError("injected rollback fchmod refusal")
        return real_fchmod(descriptor, mode)

    def refuse_rollback_unlink(path, *args, **kwargs):
        if path == output.name:
            raise OSError("injected rollback unlink refusal")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(source_state, "_fsync_publication_parent", fail_fsync)
    monkeypatch.setattr(source_state.os, "fchmod", refuse_rollback_mode)
    monkeypatch.setattr(source_state.os, "unlink", refuse_rollback_unlink)
    with pytest.raises(
        source_state.SourceStatePublicationPending,
        match="authorized.*pending",
    ):
        source_state.build_source_state_evidence(
            artifacts, output, _selections(runs)
        )

    # The exact output is positively authorized, but the loader still requires a working
    # parent fsync before it will consume the pending hard link.
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(
        source_state.SourceStateError, match="durability is still pending"
    ):
        source_state.load_source_state_evidence(artifacts, output)
    monkeypatch.setattr(
        source_state, "_fsync_publication_parent", real_parent_fsync
    )
    assert len(source_state.load_source_state_evidence(artifacts, output).runs) == 7


def test_missing_tampered_and_symlinked_authorization_are_rejected(
    tmp_path, monkeypatch
):
    _fake_supreme_validator(monkeypatch)

    def built_bundle(label: str):
        artifacts = tmp_path / f"{label}-artifacts"
        runs = _write_all(artifacts)
        output = tmp_path / label / "evidence.json"
        source_state.build_source_state_evidence(
            artifacts, output, _selections(runs)
        )
        candidate, authorization = source_state._companion_paths(output)
        return artifacts, output, candidate, authorization

    artifacts, output, _candidate, authorization = built_bundle("missing-marker")
    authorization.unlink()
    with pytest.raises(source_state.SourceStateError, match="authorization"):
        source_state.load_source_state_evidence(artifacts, output)

    artifacts, output, _candidate, authorization = built_bundle("tampered-marker")
    marker = json.loads(authorization.read_text(encoding="utf-8"))
    marker["sha256"] = "0" * 64
    authorization.write_bytes(source_state.canonical_json_bytes(marker))
    authorization.chmod(0o600)
    with pytest.raises(source_state.SourceStateError, match="hash/size mismatch"):
        source_state.load_source_state_evidence(artifacts, output)

    artifacts, output, _candidate, authorization = built_bundle("symlink-marker")
    target = tmp_path / "authorization-target"
    target.write_bytes(authorization.read_bytes())
    target.chmod(0o600)
    authorization.unlink()
    authorization.symlink_to(target)
    with pytest.raises(source_state.SourceStateError, match="symlink"):
        source_state.load_source_state_evidence(artifacts, output)


def test_missing_or_tampered_candidate_is_rejected(tmp_path, monkeypatch):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "missing-artifacts"
    runs = _write_all(artifacts)
    output = tmp_path / "missing-candidate" / "evidence.json"
    source_state.build_source_state_evidence(artifacts, output, _selections(runs))
    candidate, _authorization = source_state._companion_paths(output)
    candidate.unlink()
    with pytest.raises(source_state.SourceStateError, match="candidate"):
        source_state.load_source_state_evidence(artifacts, output)

    artifacts = tmp_path / "tampered-artifacts"
    runs = _write_all(artifacts)
    output = tmp_path / "tampered-candidate" / "evidence.json"
    source_state.build_source_state_evidence(artifacts, output, _selections(runs))
    candidate, _authorization = source_state._companion_paths(output)
    candidate.write_bytes(candidate.read_bytes() + b" ")
    candidate.chmod(0o600)
    with pytest.raises(source_state.SourceStateError, match="hash/size mismatch"):
        source_state.load_source_state_evidence(artifacts, output)


def test_loader_requires_private_companions_and_same_candidate_inode(
    tmp_path, monkeypatch
):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "mode-artifacts"
    runs = _write_all(artifacts)
    output = tmp_path / "mode-companion" / "evidence.json"
    source_state.build_source_state_evidence(artifacts, output, _selections(runs))
    _candidate, authorization = source_state._companion_paths(output)
    authorization.chmod(0o644)
    with pytest.raises(source_state.SourceStateError, match="mode 0600"):
        source_state.load_source_state_evidence(artifacts, output)

    artifacts = tmp_path / "inode-artifacts"
    runs = _write_all(artifacts)
    output = tmp_path / "inode-companion" / "evidence.json"
    source_state.build_source_state_evidence(artifacts, output, _selections(runs))
    candidate, _authorization = source_state._companion_paths(output)
    payload = output.read_bytes()
    output.unlink()
    output.write_bytes(payload)
    output.chmod(0o600)
    assert output.stat().st_ino != candidate.stat().st_ino
    with pytest.raises(source_state.SourceStateError, match="candidate inode"):
        source_state.load_source_state_evidence(artifacts, output)


def test_candidate_only_is_not_authorization(tmp_path, monkeypatch):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    _write_all(artifacts)
    output = tmp_path / "candidate-only" / "evidence.json"
    output.parent.mkdir(mode=0o700)
    candidate, _authorization = source_state._companion_paths(output)
    candidate.write_bytes(b"candidate without authorization\n")
    candidate.chmod(0o600)

    with pytest.raises(source_state.SourceStateError, match="source-state evidence"):
        source_state.load_source_state_evidence(artifacts, output)
    assert not output.exists()


def test_retry_recovers_exact_authorized_sidecars_when_output_is_absent(
    tmp_path, monkeypatch
):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    output = tmp_path / "review" / "recover.json"
    real_link = source_state.os.link

    def refuse_link(*_args, **_kwargs):
        raise OSError("injected link refusal")

    monkeypatch.setattr(source_state.os, "link", refuse_link)
    with pytest.raises(
        source_state.SourceStatePublicationPending, match="stabilization is pending"
    ):
        source_state.build_source_state_evidence(
            artifacts, output, _selections(runs)
        )
    assert not output.exists()
    candidate, authorization = source_state._companion_paths(output)
    assert candidate.is_file() and authorization.is_file()

    monkeypatch.setattr(source_state.os, "link", real_link)
    built = source_state.build_source_state_evidence(
        artifacts, output, _selections(runs)
    )

    assert built["schema_version"] == 1
    assert output.stat().st_ino == candidate.stat().st_ino
    assert len(source_state.load_source_state_evidence(artifacts, output).runs) == 7


def test_loader_rehashes_output_candidate_and_authorization(tmp_path, monkeypatch):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    output = tmp_path / "review" / "late-marker-change.json"
    source_state.build_source_state_evidence(artifacts, output, _selections(runs))
    _candidate, authorization = source_state._companion_paths(output)
    real_rehash = source_state._rehash_validated

    def mutate_marker_then_rehash(validated):
        real_rehash(validated)
        marker = json.loads(authorization.read_text(encoding="utf-8"))
        marker["state"] = "tampered_after_initial_read"
        authorization.write_bytes(source_state.canonical_json_bytes(marker))
        authorization.chmod(0o600)

    monkeypatch.setattr(source_state, "_rehash_validated", mutate_marker_then_rehash)
    with pytest.raises(source_state.SourceStateError, match="authorization"):
        source_state.load_source_state_evidence(artifacts, output)


def test_loader_rejects_noncanonical_evidence_even_with_matching_wal(
    tmp_path, monkeypatch
):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    output = tmp_path / "review" / "noncanonical-evidence.json"
    source_state.build_source_state_evidence(artifacts, output, _selections(runs))
    candidate, authorization = source_state._companion_paths(output)
    value = json.loads(output.read_text(encoding="utf-8"))
    noncanonical = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    output.write_bytes(noncanonical)
    output.chmod(0o600)
    assert output.stat().st_ino == candidate.stat().st_ino
    marker = json.loads(authorization.read_text(encoding="utf-8"))
    marker["size_bytes"] = len(noncanonical)
    marker["sha256"] = hashlib.sha256(noncanonical).hexdigest()
    authorization.write_bytes(source_state.canonical_json_bytes(marker))
    authorization.chmod(0o600)

    with pytest.raises(source_state.SourceStateError, match="not canonically serialized"):
        source_state.load_source_state_evidence(artifacts, output)


def test_publication_rejects_uncontrolled_ancestor(tmp_path, monkeypatch):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    uncontrolled = tmp_path / "uncontrolled"
    uncontrolled.mkdir(mode=0o700)
    uncontrolled.chmod(0o777)
    parent = uncontrolled / "owned"
    parent.mkdir(mode=0o700)
    output = parent / "evidence.json"

    with pytest.raises(source_state.SourceStateError, match="without sticky protection"):
        source_state.build_source_state_evidence(
            artifacts, output, _selections(runs)
        )

    _assert_failed_publication_left_no_evidence(output)


def test_detached_publication_parent_is_detected_as_pending(
    tmp_path, monkeypatch
):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    parent = tmp_path / "review"
    parent.mkdir(mode=0o700)
    detached = tmp_path / "detached-review"
    output = parent / "evidence.json"
    real_parent_fsync = source_state._fsync_publication_parent

    def detach_parent(descriptor):
        os.fsync(descriptor)
        parent.rename(detached)
        parent.mkdir(mode=0o700)

    monkeypatch.setattr(source_state, "_fsync_publication_parent", detach_parent)
    with pytest.raises(
        source_state.SourceStatePublicationPending, match="stabilization is pending"
    ):
        source_state.build_source_state_evidence(
            artifacts, output, _selections(runs)
        )

    _assert_failed_publication_left_no_evidence(output)
    assert os.path.lexists(detached / output.name)
    monkeypatch.setattr(
        source_state, "_fsync_publication_parent", real_parent_fsync
    )
    with pytest.raises(source_state.SourceStateError):
        source_state.load_source_state_evidence(artifacts, output)


def test_snapshot_adapter_uses_shared_strict_loader(tmp_path, monkeypatch):
    _fake_supreme_validator(monkeypatch)
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    evidence = tmp_path / "evidence.json"
    source_state.build_source_state_evidence(artifacts, evidence, _selections(runs))
    cfg = dataclasses.replace(load_config(), artifacts_root=artifacts)

    selected, attestation = snapshot._load_source_state_evidence(cfg, evidence)

    assert len(selected) == 7
    assert all(run.success_verified for run in selected)
    assert attestation == {
        "sha256": _sha256(evidence),
        "size_bytes": evidence.stat().st_size,
    }


def test_real_supremecourt_validator_is_required_and_proof_bound(tmp_path):
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts, strict_supreme=True)
    output = tmp_path / "strict.json"

    source_state.build_source_state_evidence(artifacts, output, _selections(runs))

    supreme = next(path for path in runs if path.parents[1].name == "supremecourt")
    record_path = supreme / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["source_validation"]["unresolved_failure_count"] = 1
    _rewrite_terminal_wal(supreme, record)
    with pytest.raises(source_state.SourceStateError, match="failure_recorded"):
        source_state.validate_selected_run(artifacts, "supremecourt", supreme.name)


def test_supremecourt_validator_report_rejects_type_confusion_and_unavailability(
    tmp_path, monkeypatch
):
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    supreme = next(path for path in runs if path.parents[1].name == "supremecourt")

    def confused_report(run_dir):
        run_dir = Path(run_dir).absolute()
        return {
            "schema_version": True,
            "valid": 1,
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "finish_reason": "closespider_timeout",
            "items_sha256": _sha256(run_dir / "items.jsonl"),
            "journal_sha256": _sha256(run_dir / "items.journal.jsonl"),
            "unresolved_failure_count": False,
        }

    monkeypatch.setattr(
        source_state,
        "_load_supremecourt_validator",
        lambda: SimpleNamespace(validate_run=confused_report),
    )
    with pytest.raises(source_state.SourceStateError, match="report mismatch"):
        source_state.validate_selected_run(artifacts, "supremecourt", supreme.name)

    def unavailable():
        raise source_state.SourceStateError("strict validator unavailable")

    monkeypatch.setattr(source_state, "_load_supremecourt_validator", unavailable)
    with pytest.raises(source_state.SourceStateError, match="validator unavailable"):
        source_state.validate_selected_run(artifacts, "supremecourt", supreme.name)


def test_bounded_read_stops_concurrent_growth(tmp_path, monkeypatch):
    path = tmp_path / "bounded.json"
    path.write_bytes(b"x")
    path.chmod(0o600)
    real_read = source_state.os.read
    injected = False

    def growing_read(descriptor, count):
        nonlocal injected
        block = real_read(descriptor, count)
        if not block and not injected:
            injected = True
            return b"y"
        return block

    monkeypatch.setattr(source_state.os, "read", growing_read)
    with pytest.raises(source_state.SourceStateError, match="while reading"):
        source_state._read_private_regular(
            path,
            label="bounded fixture",
            max_bytes=1,
        )


def test_publication_refuses_uncontrolled_parent_and_fsync_failure(
    tmp_path, monkeypatch
):
    artifacts = tmp_path / "artifacts"
    runs = _write_all(artifacts)
    _fake_supreme_validator(monkeypatch)
    parent = tmp_path / "uncontrolled"
    parent.mkdir(mode=0o777)
    parent.chmod(0o777)
    with pytest.raises(source_state.SourceStateError, match="writable by other users"):
        source_state.build_source_state_evidence(
            artifacts, parent / "evidence.json", _selections(runs)
        )

    output = tmp_path / "durable" / "evidence.json"
    real_fsync = source_state.os.fsync
    calls = 0

    def fail_first_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("fsync refused")
        return real_fsync(descriptor)

    monkeypatch.setattr(source_state.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="fsync refused"):
        source_state.build_source_state_evidence(
            artifacts, output, _selections(runs)
        )
    assert not output.exists()


def test_cli_prints_independent_review_warning(tmp_path, monkeypatch, capsys):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_source_state_evidence.py"
    spec = importlib.util.spec_from_file_location("build_source_state_evidence_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "build_source_state_evidence",
        lambda *_args, **_kwargs: {"schema_version": 1, "runs": [{}] * 7},
    )

    assert module.main(
        [
            "--artifacts-root",
            str(tmp_path / "artifacts"),
            "--output",
            str(tmp_path / "review.json"),
            "--select",
            "matsne:run",
        ]
    ) == 0
    output = capsys.readouterr().out
    assert "AWAITING INDEPENDENT OPERATOR REVIEW" in output
    assert "inseparable same-directory trio" in output
    assert "never copy or move the JSON alone" in output
