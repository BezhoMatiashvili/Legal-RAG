"""Frozen schema-v2 crawl evidence and seven-row candidate-ledger gates."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingest import snapshot, source_state
from ingest.config import load_config


def _ordinary_completion(
    root: Path, source: str = "ecd"
) -> tuple[dict[str, object], source_state.FileAttestation]:
    run_id = "20260717T100000Z_start-1900-01-01_end-2026-07-15"
    run_dir = root / source / "runs" / run_id
    run_dir.mkdir(mode=0o700, parents=True)
    run_dir.chmod(0o700)
    items = run_dir / "items.jsonl"
    identity_fields = {
        "ecd": "decision_document_id",
        "constcourt": "legal_id",
    }
    field = identity_fields[source]
    items.write_bytes(f'{{"{field}":"one"}}\n'.encode())
    items.chmod(0o600)
    digest = hashlib.sha256(items.read_bytes()).hexdigest()
    proof = source_state.FileAttestation(items, digest, items.stat().st_size)
    run_output = {
        "role": "run",
        "path": str(items.absolute()),
        "size_bytes": proof.size_bytes,
        "sha256": proof.sha256,
    }
    identity = hashlib.sha256(b"one").hexdigest()
    journal = run_dir / "identity.journal.jsonl"
    journal.write_bytes(
        source_state.canonical_json_bytes(
            {
                "identity_sha256": identity,
                "outcome": "unique",
                "sequence": 1,
            }
        )
    )
    journal.chmod(0o600)
    journal_output = {
        "role": "identity_journal",
        "path": str(journal.absolute()),
        "size_bytes": journal.stat().st_size,
        "sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
    }
    record: dict[str, object] = {
        "schema_version": 2,
        "run_id": run_id,
        "source": source,
        "spider": source,
        "start_date": "1900-01-01",
        "end_date": "2026-07-15",
        "started_at": "2026-07-17T10:00:00Z",
        "items_path": str(items.absolute()),
        "latest_items_path": None,
        "log_path": str((run_dir / "spider.log").absolute()),
        "outcome": "success",
        "quality_passed": True,
        "feeds_durable": True,
        "failure_count": 0,
        "finish_reason": "finished",
        "completed_at": "2026-07-17T11:00:00Z",
        "feed_outputs": {
            "durable": True,
            "configured_count": 1,
            "success_count": 1,
            "failure_count": 0,
            "files": [
                {
                    "role": "run",
                    "configured_uri": str(items.absolute()),
                    "path": str(items.absolute()),
                    "size_bytes": proof.size_bytes,
                    "sha256": proof.sha256,
                }
            ],
        },
        "quality": {
            "passed": True,
            "quality_failures": 0,
            "spider_errors": 0,
            "spider_exceptions": 0,
            "item_errors": 0,
            "pagination_reconcilers": 0,
            "pagination_reconciled": 0,
        },
        "source_validation": {"kind": "generic", "passed": True},
        "crawl_contract": {
            "kind": "immutable_evidence_crawl_v1",
            "start_date": "1900-01-01",
            "end_date": "2026-07-15",
            "source_arguments": {},
            "code_revision": "a" * 40,
            "code_identity_sha256": "b" * 64,
            "artifact_root": str(root.absolute()),
            "http_cache_enabled": False,
            "cross_run_dedup_enabled": False,
            "shared_seen_sqlite_accessed": False,
            "within_run_duplicates": {
                "observed": 1,
                "unique": 1,
                "duplicates": 0,
                "missing_identity": 0,
                "reconciled": True,
            },
            "discovery_limits": (
                {"max_pages": 5_000, "page_size": 50}
                if source == "constcourt"
                else {"max_pages": 20_000, "page_size": 50}
            ),
            "terminal_status": "finished",
            "durable_outputs": [journal_output, run_output],
            "completed_at": "2026-07-17T11:00:00Z",
            "quality_passed": True,
            "feeds_durable": True,
            "failure_signals": {
                "quality_failures": 0,
                "spider_errors": 0,
                "spider_exceptions": 0,
                "item_errors": 0,
                "feed_failures": 0,
            },
        },
    }
    return record, proof


def test_schema_v2_completion_independently_binds_frozen_contract(
    tmp_path, monkeypatch
):
    root = (tmp_path / "source-evidence").absolute()
    monkeypatch.setattr(source_state, "CANDIDATE_ARTIFACT_ROOT", root)
    record, items = _ordinary_completion(root)

    completed_at = source_state.validate_completion_record(
        record,
        source="ecd",
        run_id=str(record["run_id"]),
        run_dir=items.path.parent,
        items=items,
        now=datetime(2026, 7, 18, tzinfo=UTC),
    )

    assert completed_at == "2026-07-17T11:00:00Z"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("http_cache_enabled", True, "must be false"),
        ("cross_run_dedup_enabled", True, "must be false"),
        ("shared_seen_sqlite_accessed", True, "must be false"),
    ],
)
def test_schema_v2_completion_rejects_shared_or_cross_run_state(
    tmp_path, monkeypatch, field, value, match
):
    root = (tmp_path / "source-evidence").absolute()
    monkeypatch.setattr(source_state, "CANDIDATE_ARTIFACT_ROOT", root)
    record, items = _ordinary_completion(root)
    record["crawl_contract"][field] = value

    with pytest.raises(source_state.SourceStateError, match=match):
        source_state.validate_completion_record(
            record,
            source="ecd",
            run_id=str(record["run_id"]),
            run_dir=items.path.parent,
            items=items,
            now=datetime(2026, 7, 18, tzinfo=UTC),
        )


def test_schema_v2_recomputes_counts_instead_of_trusting_declared_values(
    tmp_path, monkeypatch
):
    root = (tmp_path / "source-evidence").absolute()
    monkeypatch.setattr(source_state, "CANDIDATE_ARTIFACT_ROOT", root)
    record, items = _ordinary_completion(root)
    counts = record["crawl_contract"]["within_run_duplicates"]
    counts.update({"observed": 2, "duplicates": 1})

    with pytest.raises(source_state.SourceStateError, match="durable identity evidence"):
        source_state.validate_completion_record(
            record,
            source="ecd",
            run_id=str(record["run_id"]),
            run_dir=items.path.parent,
            items=items,
            now=datetime(2026, 7, 18, tzinfo=UTC),
        )


def test_constcourt_completion_requires_the_explicit_pagination_cap(
    tmp_path, monkeypatch
):
    root = (tmp_path / "source-evidence").absolute()
    monkeypatch.setattr(source_state, "CANDIDATE_ARTIFACT_ROOT", root)
    record, items = _ordinary_completion(root, source="constcourt")
    del record["crawl_contract"]["discovery_limits"]["max_pages"]

    with pytest.raises(source_state.SourceStateError, match="discovery limits"):
        source_state.validate_completion_record(
            record,
            source="constcourt",
            run_id=str(record["run_id"]),
            run_dir=items.path.parent,
            items=items,
            now=datetime(2026, 7, 18, tzinfo=UTC),
        )


def _fake_run(source: str, run_id: str) -> source_state.ValidatedRun:
    file = source_state.FileAttestation(
        Path(f"/{source}/runs/{run_id}/items.jsonl"), "1" * 64, 1
    )
    completion = source_state.FileAttestation(
        Path(f"/{source}/runs/{run_id}/run.json"), "2" * 64, 1
    )
    return source_state.ValidatedRun(
        source=source,
        run_id=run_id,
        items=file,
        completion=completion,
        terminal_candidate=completion,
        terminal_authorization=completion,
        completed_at="2026-07-17T11:00:00Z",
        completion_schema_version=2,
    )


def test_candidate_rejects_extra_ordinary_run_directories(tmp_path):
    root = tmp_path / "source-evidence"
    runs = []
    for source in source_state._ORDINARY_IDENTITY_FIELDS:
        run = _fake_run(source, f"{source}-01")
        runs.append(run)
        run_dir = root / source / "runs" / run.run_id
        run_dir.mkdir(mode=0o700, parents=True)
        run_dir.chmod(0o700)
        run_dir.parent.chmod(0o700)
    extra = root / "ecd" / "runs" / "ecd-02"
    extra.mkdir(mode=0o700)
    extra.chmod(0o700)

    with pytest.raises(source_state.SourceStateError, match="exactly one"):
        source_state._validate_ordinary_run_directories(root, runs)


def test_candidate_ledger_requires_contiguous_supreme_chain_and_final_feed(monkeypatch):
    first = _fake_run("supremecourt", "supreme-01")
    final = _fake_run("supremecourt", "supreme-02")
    runs = [
        final if source == "supremecourt" else _fake_run(source, f"{source}-01")
        for source in source_state.PRODUCTION_SOURCES
    ]
    completions = {
        run.run_id: {
            "finish_reason": "finished" if run is final else "closespider_timeout",
            "crawl_contract": {
                "code_revision": "a" * 40,
                "code_identity_sha256": "b" * 64,
                "source_arguments": (
                    {
                        "initial_window_days": 7,
                        "max_runtime_seconds": 14_400,
                        "parent_run_id": "supreme-01",
                    }
                    if run is final
                    else {}
                ),
            },
            "source_validation": {},
        }
        for run in runs
    }
    completions[first.run_id] = {
        "finish_reason": "closespider_timeout",
        "crawl_contract": {
            "code_revision": "a" * 40,
            "code_identity_sha256": "b" * 64,
            "source_arguments": {
                "initial_window_days": 7,
                "max_runtime_seconds": 14_400,
                "parent_run_id": "none",
            },
        },
        "source_validation": {},
    }
    monkeypatch.setattr(
        source_state, "_completion_value", lambda run: completions[run.run_id]
    )
    terminal_calls: list[bool] = []

    def validated(_value, *, run_dir, require_terminal_coverage, **_kwargs):
        terminal_calls.append(require_terminal_coverage)
        return {
            "resume_parent": (
                None if run_dir.name == first.run_id else {"run_id": first.run_id}
            )
        }

    monkeypatch.setattr(source_state, "_validate_supremecourt_source", validated)

    assert source_state._validate_candidate_run_set(runs, [first, final]) == (
        "a" * 40,
        "b" * 64,
    )
    assert terminal_calls == [False, True]

    completions[final.run_id]["crawl_contract"]["source_arguments"]["parent_run_id"] = (
        "none"
    )
    with pytest.raises(source_state.SourceStateError, match="contiguous parent chain"):
        source_state._validate_candidate_run_set(runs, [first, final])


def test_exact_candidate_snapshot_refuses_legacy_source_state(monkeypatch, tmp_path):
    evidence = source_state.FileAttestation(tmp_path / "legacy.json", "3" * 64, 1)
    monkeypatch.setattr(
        snapshot,
        "load_source_state_evidence",
        lambda *_args, **_kwargs: source_state.LoadedEvidence((), evidence),
    )

    with pytest.raises(snapshot.SnapshotSafetyError, match="schema v2"):
        snapshot._load_source_state_evidence(
            load_config(),
            evidence.path,
            expected_snapshot_id=source_state.CANDIDATE_SNAPSHOT_ID,
        )
