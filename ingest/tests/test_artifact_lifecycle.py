"""Hermetic artifact lifecycle tests; every deletion target lives under tmp_path."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingest.artifacts import (
    DEFAULT_ROTATE_BACKUPS,
    DEFAULT_ROTATE_BYTES,
    ArtifactSafetyError,
    append_rotating_text,
    apply_prune_plan,
    atomic_write_json,
    atomic_write_text,
    build_prune_plan,
    load_prune_plan,
    load_verified_generation_coverage,
    write_prune_plan,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import prune_artifacts  # noqa: E402

NOW = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _write_run(
    root: Path,
    source: str,
    run_id: str,
    *,
    completed_at: str,
    outcome: str = "success",
    quality_passed: bool = True,
    feeds_durable: bool = True,
) -> Path:
    run_dir = root / source / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "items.jsonl").write_text('{"document_id":"1"}\n', encoding="utf-8")
    metadata = {
        "run_id": run_id,
        "spider": source,
        "outcome": outcome,
        "quality_passed": quality_passed,
        "feeds_durable": feeds_durable,
        "completed_at": completed_at,
        "failure_count": 0,
    }
    (run_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir


def _write_generation(
    root: Path,
    generation_id: str,
    covered_runs: object,
    *,
    verified: bool = True,
) -> Path:
    directory = root / generation_id
    directory.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "generation_id": generation_id,
        "covered_runs": covered_runs,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    if verified:
        report = {
            "schema_version": 1,
            "generation_id": generation_id,
            "verified_at": "2026-07-13T11:00:00Z",
            "ok": True,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "covered_runs": covered_runs,
            "stats": {},
            "coverage": {"ok": True, "issue_count": 0, "examples": []},
            "integrity": {"ok": True, "issue_count": 0, "examples": []},
            "freshness": {"ok": True, "issue_count": 0, "examples": []},
            "quality": {"ok": True, "issue_count": 0, "examples": []},
        }
        report_path = directory.with_name(f"{directory.name}.verification.json")
        report_path.write_text(json.dumps(report), encoding="utf-8")
    return directory


def _candidate_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifacts = tmp_path / "artifacts"
    generations = tmp_path / "generations"
    artifacts.mkdir()
    old = _write_run(
        artifacts,
        "matsne",
        "old-success",
        completed_at="2025-01-01T00:00:00Z",
    )
    _write_generation(
        generations,
        "gen-001",
        [{"source": "matsne", "run_id": "old-success"}],
    )
    return artifacts, generations, old


def test_atomic_writes_are_private_stable_and_replace(tmp_path):
    text_path = tmp_path / "private" / "nested" / "note.txt"
    atomic_write_text(text_path, "first\n")
    assert text_path.read_text(encoding="utf-8") == "first\n"
    assert _mode(text_path) == 0o600
    assert _mode(text_path.parent) == 0o700
    assert _mode(text_path.parent.parent) == 0o700

    text_path.chmod(0o644)
    atomic_write_text(text_path, "replacement\n")
    assert text_path.read_text(encoding="utf-8") == "replacement\n"
    assert _mode(text_path) == 0o600
    assert not list(text_path.parent.glob(f".{text_path.name}.*.tmp"))

    json_path = tmp_path / "private" / "report.json"
    atomic_write_json(json_path, {"z": "თბილისი", "a": 1})
    assert json_path.read_text(encoding="utf-8") == (
        '{\n  "a": 1,\n  "z": "თბილისი"\n}\n'
    )
    assert _mode(json_path) == 0o600


def test_atomic_write_preserves_original_if_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    path.write_text("original", encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr("ingest.artifacts.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        atomic_write_text(path, "new")
    assert path.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_rotation_defaults_and_retained_archive_bound(tmp_path):
    assert DEFAULT_ROTATE_BYTES == 50 * 1024 * 1024
    assert DEFAULT_ROTATE_BACKUPS == 10

    log = tmp_path / "logs" / "crawler.log"
    for label in ("aaaa\n", "bbbb\n", "cccc\n", "dddd\n"):
        append_rotating_text(log, label, max_bytes=8, backups=2)

    assert log.read_text(encoding="utf-8") == "dddd\n"
    assert (log.parent / "crawler.log.1").read_text(encoding="utf-8") == "cccc\n"
    assert (log.parent / "crawler.log.2").read_text(encoding="utf-8") == "bbbb\n"
    assert not (log.parent / "crawler.log.3").exists()
    assert _mode(log) == 0o600
    assert _mode(log.parent / "crawler.log.1") == 0o600
    assert _mode(log.parent) == 0o700


def test_verified_coverage_accepts_mapping_and_records_but_fails_closed(tmp_path):
    generations = tmp_path / "generations"
    _write_generation(
        generations,
        "gen-a",
        {"matsne": ["run-2", "run-1"]},
    )
    _write_generation(
        generations,
        "gen-b",
        [{"source": "tas", "run_id": "run-3"}],
    )
    _write_generation(generations, "unverified", {"ecd": ["run-4"]}, verified=False)
    inline_only = generations / "inline-only"
    inline_only.mkdir(parents=True)
    (inline_only / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_id": "inline-only",
                "covered_runs": {"ecd": ["run-5"]},
                "verified": True,
            }
        ),
        encoding="utf-8",
    )
    bad = generations / "missing-coverage"
    bad.mkdir(parents=True)
    (bad / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "generation_id": "missing-coverage"}),
        encoding="utf-8",
    )

    loaded = load_verified_generation_coverage(generations)
    assert [item.generation_id for item in loaded] == ["gen-a", "gen-b"]
    assert [(run.source, run.run_id) for run in loaded[0].covered_runs] == [
        ("matsne", "run-1"),
        ("matsne", "run-2"),
    ]


def test_verification_report_requires_all_gates_and_matching_coverage(tmp_path):
    generations = tmp_path / "generations"
    directory = _write_generation(
        generations,
        "gen-report",
        [{"source": "matsne", "run_id": "run-1"}],
        verified=False,
    )
    report = {
        "schema_version": 1,
        "generation_id": "gen-report",
        "verified_at": "2026-07-13T11:00:00Z",
        "ok": True,
        "manifest_sha256": hashlib.sha256(
            (directory / "manifest.json").read_bytes()
        ).hexdigest(),
        "covered_runs": [{"source": "matsne", "run_id": "run-1"}],
        "stats": {},
        "coverage": {"ok": True, "issue_count": 0, "examples": []},
        "integrity": {"ok": True, "issue_count": 0, "examples": []},
        "freshness": {"ok": True, "issue_count": 0, "examples": []},
        "quality": {"ok": True, "issue_count": 0, "examples": []},
    }
    report_path = directory.with_name(f"{directory.name}.verification.json")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert [
        item.generation_id for item in load_verified_generation_coverage(generations)
    ] == ["gen-report"]

    report["quality"] = {"ok": False, "issue_count": 1, "examples": ["bad"]}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert load_verified_generation_coverage(generations) == ()

    report["quality"] = {"ok": True, "issue_count": 0, "examples": []}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    assert load_verified_generation_coverage(generations) == ()


def test_plan_is_deterministic_and_applies_source_retention_windows(tmp_path):
    artifacts = tmp_path / "artifacts"
    generations = tmp_path / "generations"
    artifacts.mkdir()
    _write_run(
        artifacts,
        "matsne",
        "eligible-at-180",
        completed_at="2026-01-14T12:00:00Z",
    )
    _write_run(
        artifacts,
        "matsne",
        "recent",
        completed_at="2026-07-01T00:00:00Z",
    )
    _write_run(
        artifacts,
        "tas",
        "eligible-at-30",
        completed_at="2026-06-13T12:00:00Z",
    )
    _write_generation(
        generations,
        "gen-001",
        {
            "matsne": ["eligible-at-180", "recent"],
            "tas": ["eligible-at-30"],
        },
    )

    first = build_prune_plan(artifacts, generations, now=NOW)
    second = build_prune_plan(artifacts, generations, now=NOW)
    assert first.to_dict() == second.to_dict()
    assert [item.relative_path for item in first.candidates] == [
        "matsne/runs/eligible-at-180",
        "tas/runs/eligible-at-30",
    ]
    assert [item.retention_days for item in first.candidates] == [180, 30]
    assert [(item.run_id, item.reason) for item in first.retained] == [
        ("recent", "retention_period_not_elapsed")
    ]


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("bad-outcome", "run_not_successful"),
        ("bad-quality", "quality_not_passed"),
        ("feeds-not-durable", "feeds_not_durable"),
        ("uncovered", "not_covered_by_verified_generation"),
    ],
)
def test_plan_retains_failed_incomplete_or_uncovered_runs(tmp_path, case, reason):
    artifacts = tmp_path / "artifacts"
    generations = tmp_path / "generations"
    artifacts.mkdir()
    kwargs = {
        "completed_at": "2025-01-01T00:00:00Z",
        "outcome": "failed" if case == "bad-outcome" else "success",
        "quality_passed": case != "bad-quality",
        "feeds_durable": case != "feeds-not-durable",
    }
    _write_run(artifacts, "matsne", case, **kwargs)
    covered = [] if case == "uncovered" else [{"source": "matsne", "run_id": case}]
    _write_generation(generations, "gen-001", covered)

    plan = build_prune_plan(artifacts, generations, now=NOW)
    assert plan.candidates == ()
    assert [(item.run_id, item.reason) for item in plan.retained] == [(case, reason)]


@pytest.mark.parametrize(
    "evidence_name",
    [
        "failures.jsonl",
        "repair_manifest.jsonl",
        "quarantine.jsonl",
        "raw_failed/source.pdf",
        "snapshots/copy.bin",
    ],
)
def test_plan_never_selects_failure_repair_quarantine_or_snapshot_evidence(
    tmp_path, evidence_name
):
    artifacts, generations, run_dir = _candidate_tree(tmp_path)
    evidence = run_dir / evidence_name
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("evidence", encoding="utf-8")

    plan = build_prune_plan(artifacts, generations, now=NOW)
    assert plan.candidates == ()
    assert plan.retained[0].reason == "protected_evidence_present"


def test_apply_requires_approval_and_revalidates_before_any_deletion(tmp_path):
    artifacts, generations, old = _candidate_tree(tmp_path)
    plan = build_prune_plan(artifacts, generations, now=NOW)
    plan_path = write_prune_plan(tmp_path / "reviewed-plan.json", plan)
    with pytest.raises(PermissionError, match="explicit approval"):
        apply_prune_plan(plan_path)
    assert old.exists()

    (old / "failures.jsonl").write_text("late failure\n", encoding="utf-8")
    with pytest.raises(ArtifactSafetyError, match="changed after planning"):
        apply_prune_plan(plan_path, approved=True)
    assert old.exists()


def test_apply_deletes_only_preflighted_tmp_path_candidate(tmp_path):
    artifacts, generations, old = _candidate_tree(tmp_path)
    recent = _write_run(
        artifacts,
        "matsne",
        "recent",
        completed_at="2026-07-01T00:00:00Z",
    )
    _write_generation(
        generations,
        "gen-002",
        [{"source": "matsne", "run_id": "recent"}],
    )
    plan = build_prune_plan(artifacts, generations, now=NOW)
    plan_path = write_prune_plan(tmp_path / "reviewed-plan.json", plan)

    assert apply_prune_plan(plan_path, approved=True) == (
        "matsne/runs/old-success",
    )
    assert not old.exists()
    assert recent.exists()


def test_cli_requires_separate_plan_then_flag_and_environment(tmp_path, capsys):
    artifacts, generations, old = _candidate_tree(tmp_path)
    common = [
        "--artifacts-root",
        str(artifacts),
        "--generations-root",
        str(generations),
        "--now",
        "2026-07-13T12:00:00Z",
    ]

    with pytest.raises(SystemExit):
        prune_artifacts.main(common, environ={})
    assert old.exists()
    capsys.readouterr()

    plan_path = tmp_path / "reviewed-plan.json"
    assert (
        prune_artifacts.main(
            [*common, "--plan-output", str(plan_path)], environ={}
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["mode"] == "dry-run"
    assert old.exists()
    assert load_prune_plan(plan_path).candidates[0].run_id == "old-success"

    with pytest.raises(SystemExit):
        prune_artifacts.main(
            ["--apply", "--plan", str(plan_path)], environ={}
        )
    assert old.exists()
    capsys.readouterr()

    assert (
        prune_artifacts.main(
            ["--apply", "--plan", str(plan_path)],
            environ={"ARTIFACT_PRUNE_APPROVED": "1"},
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["deleted"] == ["matsne/runs/old-success"]
    assert not old.exists()


def test_cli_plan_report_is_owner_only(tmp_path, capsys):
    artifacts, generations, old = _candidate_tree(tmp_path)
    report_path = tmp_path / "reports" / "plan.json"
    args = [
        "--artifacts-root",
        str(artifacts),
        "--generations-root",
        str(generations),
        "--now",
        "2026-07-13T12:00:00Z",
        "--plan-output",
        str(report_path),
    ]
    assert prune_artifacts.main(args, environ={}) == 0
    capsys.readouterr()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 2
    assert len(persisted["plan_sha256"]) == 64
    assert _mode(report_path) == 0o600
    assert _mode(report_path.parent) == 0o700
    assert old.exists()


def test_plan_is_digest_bound_owner_only_and_never_overwritten(tmp_path):
    artifacts, generations, _old = _candidate_tree(tmp_path)
    plan = build_prune_plan(artifacts, generations, now=NOW)
    path = write_prune_plan(tmp_path / "plans" / "review.json", plan)

    assert _mode(path) == 0o600
    with pytest.raises(FileExistsError):
        write_prune_plan(path, plan)

    value = json.loads(path.read_text(encoding="utf-8"))
    value["candidates"][0]["size_bytes"] += 1
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ArtifactSafetyError, match="digest mismatch"):
        load_prune_plan(path)
