"""Fail-closed immutable clean-corpus snapshot tests."""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from ingest import snapshot
from ingest import source_state
from ingest.config import load_config

BODY = "მუხლი 1. სასამართლომ დაადგინა შემდეგი გარემოებანი საქმეზე. " * 3
RUN_ID = "20260701T000000Z_fixture"


def _cfg(tmp_path):
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        artifacts_root=tmp_path / "artifacts",
        state_dir=tmp_path / "ingest" / ".state",
    )


def _write_run(
    tmp_path: Path,
    source: str,
    items: list[dict],
    *,
    run_id: str = RUN_ID,
    completion: bool = False,
    **completion_overrides,
) -> Path:
    run_dir = tmp_path / "artifacts" / source / "runs" / run_id
    run_dir.mkdir(parents=True)
    run_dir.chmod(0o700)
    with (run_dir / "items.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    items_path = run_dir / "items.jsonl"
    items_path.chmod(0o600)
    if completion:
        latest_items_path = tmp_path / "artifacts" / source / "latest" / "items.jsonl"
        items_sha256 = _sha256(items_path)
        item_size = items_path.stat().st_size
        source_validation = {"kind": "generic", "passed": True}
        finish_reason = "finished"
        if source == "supremecourt":
            finish_reason = "closespider_timeout"
            manifest_path = run_dir / "partial_manifest.json"
            journal_path = run_dir / "items.journal.jsonl"
            manifest_path.write_text("{}\n", encoding="utf-8")
            journal_path.write_bytes(b"")
            manifest_path.chmod(0o600)
            journal_path.chmod(0o600)
            source_validation = {
                "kind": "supremecourt_partial_v1",
                "passed": True,
                "validator_schema_version": 1,
                "run_dir": str(run_dir.absolute()),
                "run_id": run_id,
                "finish_reason": finish_reason,
                "items_sha256": items_sha256,
                "manifest_path": str(manifest_path.absolute()),
                "manifest_sha256": _sha256(manifest_path),
                "journal_path": str(journal_path.absolute()),
                "journal_sha256": _sha256(journal_path),
                "unresolved_failure_count": 0,
            }
        record = {
            "schema_version": 1,
            "run_id": run_id,
            "source": source,
            "spider": source,
            "start_date": "2026-07-01",
            "end_date": "2026-07-01",
            "started_at": "2026-07-01T00:00:00Z",
            "items_path": str(items_path.absolute()),
            "latest_items_path": str(latest_items_path.absolute()),
            "log_path": str((run_dir / "spider.log").absolute()),
            "outcome": "success",
            "quality_passed": True,
            "feeds_durable": True,
            "completed_at": "2026-07-01T01:00:00Z",
            "failure_count": 0,
            "finish_reason": finish_reason,
            "feed_outputs": {
                "durable": True,
                "configured_count": 2,
                "success_count": 2,
                "failure_count": 0,
                "files": sorted(
                    [
                        {
                            "role": "run",
                            "configured_uri": str(items_path.absolute()),
                            "path": str(items_path.absolute()),
                            "size_bytes": item_size,
                            "sha256": items_sha256,
                        },
                        {
                            "role": "latest",
                            "configured_uri": str(latest_items_path.absolute()),
                            "path": str(latest_items_path.absolute()),
                            "size_bytes": item_size,
                            "sha256": items_sha256,
                        },
                    ],
                    key=lambda row: (row["role"], row["configured_uri"], row["path"]),
                ),
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
            "source_validation": source_validation,
        }
        record.update(completion_overrides)
        payload = source_state.canonical_json_bytes(record)
        for path in (
            run_dir / "run.json",
            run_dir / source_state.TERMINAL_CANDIDATE_FILENAME,
        ):
            path.write_bytes(payload)
            path.chmod(0o600)
        authorization = {
            "schema_version": 1,
            "state": "terminal_authorized",
            "source": source,
            "run_id": run_id,
            "candidate_filename": source_state.TERMINAL_CANDIDATE_FILENAME,
            "terminal_size_bytes": len(payload),
            "terminal_sha256": hashlib.sha256(payload).hexdigest(),
        }
        authorization_path = run_dir / source_state.FINALIZATION_CLAIM_FILENAME
        authorization_path.write_bytes(source_state.canonical_json_bytes(authorization))
        authorization_path.chmod(0o600)
    return run_dir


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_evidence(tmp_path: Path, run_dirs: list[Path]) -> Path:
    runs = []
    for run_dir in run_dirs:
        source = run_dir.parents[1].name
        runs.append(
            {
                "source": source,
                "run_id": run_dir.name,
                "items_sha256": _sha256(run_dir / "items.jsonl"),
                "completion_record_sha256": _sha256(run_dir / "run.json"),
            }
        )
    path = tmp_path / "source-state.json"
    runs.sort(key=lambda row: (row["source"], row["run_id"]))
    source_state._publish_create_only(
        path,
        source_state.canonical_json_bytes(
            {
                "schema_version": snapshot.SOURCE_STATE_EVIDENCE_SCHEMA_VERSION,
                "runs": runs,
            }
        ),
    )
    return path


def _patch_supreme_validator(monkeypatch) -> None:
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


def _preflight_root(cfg) -> Path:
    return cfg.state_dir / "v3" / "snapshots"


def _ecd_items() -> list[dict]:
    return [
        {
            "decision_document_id": "E1",
            "case_no": "c1",
            "decision_type_name": "t",
            "body_markdown": BODY,
        },
        {
            "decision_document_id": "E2",
            "case_no": "c2",
            "decision_type_name": "t",
            "body_markdown": "",
        },
        {
            "decision_document_id": "E3",
            "case_no": "c3",
            "decision_type_name": "t",
            "body_markdown": "დად\x00გენ\x00ილება " + BODY,
        },
        {
            "decision_document_id": "E4",
            "case_no": "c4",
            "decision_type_name": "t",
            "body_markdown": BODY,
        },
    ]


def test_preflight_cleans_quarantines_seals_all_sources_and_is_offline(
    tmp_path, monkeypatch
):
    cfg = dataclasses.replace(_cfg(tmp_path), tokenizer_revision="a" * 40)
    _write_run(tmp_path, "ecd", _ecd_items())

    real_import = builtins.__import__

    def no_embedding_import(name, *args, **kwargs):
        if name in {"ingest.embedding", ".embedding"} or name.endswith(".embedding"):
            raise AssertionError("token_sample=0 imported the tokenizer stack")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_embedding_import)
    snapshot_id = "v3_test_clean"
    manifest = snapshot.build_snapshot(
        cfg,
        snapshot_id=snapshot_id,
        output_root=_preflight_root(cfg),
        preflight=True,
        near_dup=False,
        token_sample=0,
    )
    root = _preflight_root(cfg) / snapshot_id

    assert manifest["preflight"] is True
    assert manifest["build"]["tokenizer"]["revision"] == "a" * 40
    assert manifest["build"]["sources"] == list(snapshot.SOURCES_PRESENT)
    assert manifest["structural_chunk_inventory"]["status"] == "unavailable"
    assert not (root / "structural_chunk_inventory.jsonl").exists()
    assert set(manifest["sources"]) == set(snapshot.SOURCES_PRESENT)
    assert manifest["totals"] == {"clean": 3, "quarantined": 1, "malformed": 0}
    assert len([entry for entry in manifest["files"] if entry["path"].startswith("docs/")]) == 7
    assert (root / "docs" / "supremecourt.jsonl").is_file()
    assert snapshot.verify_sealed_snapshot(root, allow_preflight=True) == manifest
    with pytest.raises(snapshot.SnapshotSafetyError, match="preflight"):
        snapshot.verify_sealed_snapshot(root)

    quarantine = [
        json.loads(line)
        for line in (root / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(quarantine) == 1
    assert quarantine[0]["document_id"] == "E2"
    assert quarantine[0]["reason"] == "empty_body"
    docs = [
        json.loads(line)
        for line in (root / "docs" / "ecd.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {doc["document_id"]: doc for doc in docs}
    assert "\x00" not in by_id["E3"]["body_markdown"]
    assert by_id["E1"]["content_hash"] == by_id["E4"]["content_hash"]
    assert by_id["E1"]["snapshot_id"] == snapshot_id
    assert by_id["E1"]["doc_id"] == f"ecd:E1:{by_id['E1']['version_id']}"


def test_snapshot_keeps_each_canonical_version_of_same_document(tmp_path):
    cfg = _cfg(tmp_path)
    old = "ძველი ვერსია საკმარისი სიგრძის ტექსტით დოკუმენტისთვის აქ. " * 3
    new = "ახალი ვერსია საკმარისი სიგრძის ტექსტით დოკუმენტისთვის აქ. " * 3
    _write_run(
        tmp_path,
        "ecd",
        [
            {
                "decision_document_id": "E1",
                "case_no": "c",
                "decision_type_name": "t",
                "body_markdown": old,
            }
        ],
        run_id="20260101T000000Z_r1",
    )
    _write_run(
        tmp_path,
        "ecd",
        [
            {
                "decision_document_id": "E1",
                "case_no": "c",
                "decision_type_name": "t",
                "body_markdown": new,
            }
        ],
        run_id="20260202T000000Z_r2",
    )
    snapshot.build_snapshot(
        cfg,
        snapshot_id="v3_versions",
        output_root=_preflight_root(cfg),
        preflight=True,
        near_dup=False,
        token_sample=0,
    )
    docs = [
        json.loads(line)
        for line in (
            _preflight_root(cfg) / "v3_versions" / "docs" / "ecd.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert len(docs) == 2
    assert {doc["body_markdown"] for doc in docs} == {old, new}
    assert len({doc["version_id"] for doc in docs}) == 2
    assert docs[0]["source_run"].startswith("20260202")


def test_production_evidence_selects_exact_runs_and_preserves_incomplete_quarantine(
    tmp_path, monkeypatch,
):
    _patch_supreme_validator(monkeypatch)
    cfg = _cfg(tmp_path)
    source_items = {source: [] for source in snapshot.SOURCES_PRESENT}
    source_items["ecd"] = _ecd_items()[:1]
    source_items["supremecourt"] = [
        {
            "case_id": "SC1",
            "chamber": "civil",
            "case_number": "N-1",
            "subject": "დავა",
            "body_markdown": BODY,
        }
    ]
    source_items["tas"] = [
        {
            "document_id": "TAS1",
            "document_no": "AR1",
            "body_markdown": BODY,
        }
    ]
    source_items["tbappeal"] = [
        {"slug": "TB1", "title": "summary", "body_markdown": BODY}
    ]
    run_dirs = [
        _write_run(tmp_path, source, source_items[source], completion=True)
        for source in snapshot.SOURCES_PRESENT
    ]
    evidence = _write_evidence(tmp_path, run_dirs)
    output_root = tmp_path / "releases" / "snapshots"
    manifest = snapshot.build_snapshot(
        cfg,
        snapshot_id="v3_attested_candidate",
        output_root=output_root,
        source_state_evidence=evidence,
        near_dup=False,
        token_sample=0,
    )
    root = output_root / "v3_attested_candidate"

    assert manifest["preflight"] is False
    assert len(manifest["runs"]) == 7
    assert all(run["success_verified"] for run in manifest["runs"])
    assert all(run["completion_record"] for run in manifest["runs"])
    assert manifest["source_state_evidence"] == {
        "sha256": _sha256(evidence),
        "size_bytes": evidence.stat().st_size,
    }
    assert manifest["sources"]["supremecourt"]["clean"] == 1
    assert manifest["sources"]["tas"]["quarantined"] == {
        "inadmissible_source:non_authoritative_summary:source_not_paginated": 1
    }
    assert manifest["sources"]["tbappeal"]["quarantined"] == {
        "inadmissible_source:non_authoritative_summary:source_not_paginated": 1
    }
    assert snapshot.verify_sealed_snapshot(root) == manifest
    quarantine_sources = {
        json.loads(line)["source"]
        for line in (root / "quarantine.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert {"tas", "tbappeal"}.issubset(quarantine_sources)


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"outcome": "failed"}, "run_not_successful"),
        ({"quality_passed": False}, "quality_not_passed"),
        ({"feeds_durable": False}, "feeds_not_durable"),
        ({"completed_at": "not-a-timestamp"}, "completion.completed_at"),
    ],
)
def test_production_rejects_unattested_completion(
    tmp_path, monkeypatch, override, reason
):
    _patch_supreme_validator(monkeypatch)
    cfg = _cfg(tmp_path)
    run_dirs = [
        _write_run(
            tmp_path,
            source,
            [],
            completion=True,
            **(override if source == "ecd" else {}),
        )
        for source in snapshot.SOURCES_PRESENT
    ]
    evidence = _write_evidence(tmp_path, run_dirs)
    with pytest.raises(snapshot.SnapshotSafetyError, match=reason):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_rejected_evidence",
            output_root=tmp_path / "releases",
            source_state_evidence=evidence,
            near_dup=False,
            token_sample=0,
        )
    assert not (tmp_path / "releases" / "v3_rejected_evidence").exists()


def test_production_rejects_items_hash_tampering_before_publication(
    tmp_path, monkeypatch
):
    _patch_supreme_validator(monkeypatch)
    cfg = _cfg(tmp_path)
    run_dirs = [
        _write_run(tmp_path, source, [], completion=True)
        for source in snapshot.SOURCES_PRESENT
    ]
    evidence = _write_evidence(tmp_path, run_dirs)
    (run_dirs[0] / "items.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(snapshot.SnapshotSafetyError, match="items.jsonl hash mismatch"):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_tampered_input",
            output_root=tmp_path / "releases",
            source_state_evidence=evidence,
            near_dup=False,
            token_sample=0,
        )


def test_production_consumes_the_exact_post_validation_items_bytes(
    tmp_path, monkeypatch
):
    """A path swap cannot be hidden by restoring the attested file before rehash."""

    _patch_supreme_validator(monkeypatch)
    cfg = _cfg(tmp_path)
    run_dirs = [
        _write_run(
            tmp_path,
            source,
            _ecd_items()[:1] if source == "ecd" else [],
            completion=True,
        )
        for source in snapshot.SOURCES_PRESENT
    ]
    evidence = _write_evidence(tmp_path, run_dirs)
    ecd_items = next(
        run / "items.jsonl" for run in run_dirs if run.parents[1].name == "ecd"
    )
    original = ecd_items.read_bytes()
    changed = original.replace(b'E1', b'X1', 1)
    assert changed != original and len(changed) == len(original)

    real_load = snapshot._load_source_state_evidence

    def load_then_swap(*args, **kwargs):
        loaded = real_load(*args, **kwargs)
        ecd_items.write_bytes(changed)
        ecd_items.chmod(0o600)
        return loaded

    real_assert = snapshot._assert_inputs_unchanged

    def restore_before_legacy_rehash(*args, **kwargs):
        ecd_items.write_bytes(original)
        ecd_items.chmod(0o600)
        return real_assert(*args, **kwargs)

    monkeypatch.setattr(snapshot, "_load_source_state_evidence", load_then_swap)
    monkeypatch.setattr(snapshot, "_assert_inputs_unchanged", restore_before_legacy_rehash)
    output_root = tmp_path / "releases"
    with pytest.raises(snapshot.SnapshotSafetyError, match="bytes changed"):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_post_validation_swap",
            output_root=output_root,
            source_state_evidence=evidence,
            near_dup=False,
            token_sample=0,
        )
    assert not (output_root / "v3_post_validation_swap").exists()


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ('{"schema_version":1,"schema_version":1,"runs":[]}', "duplicate JSON key"),
        ('{"schema_version":NaN,"runs":[]}', "non-finite JSON number"),
        ('{"schema_version":1,"runs":[}', "invalid JSON object"),
    ],
)
def test_source_state_evidence_requires_strict_json(tmp_path, payload, reason):
    cfg = _cfg(tmp_path)
    evidence = tmp_path / "bad-source-state.json"
    source_state._publish_create_only(evidence, payload.encode("utf-8"))
    with pytest.raises(snapshot.SnapshotSafetyError, match=reason):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_bad_evidence_json",
            output_root=tmp_path / "releases",
            source_state_evidence=evidence,
            near_dup=False,
            token_sample=0,
        )


def test_seal_detects_checksum_tampering(tmp_path):
    cfg = _cfg(tmp_path)
    _write_run(tmp_path, "ecd", _ecd_items()[:1])
    snapshot.build_snapshot(
        cfg,
        snapshot_id="v3_tamper_check",
        output_root=_preflight_root(cfg),
        preflight=True,
        near_dup=False,
        token_sample=0,
    )
    root = _preflight_root(cfg) / "v3_tamper_check"
    with (root / "docs" / "ecd.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(snapshot.SnapshotSafetyError, match="inventory"):
        snapshot.verify_sealed_snapshot(root, allow_preflight=True)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ('{"schema_version":2,"schema_version":2}', "duplicate JSON key"),
        ('{"schema_version":NaN}', "non-finite JSON number"),
        ('{"schema_version":2', "invalid JSON object"),
    ],
)
def test_sealed_manifest_requires_strict_json(tmp_path, payload, reason):
    cfg = _cfg(tmp_path)
    snapshot.build_snapshot(
        cfg,
        snapshot_id="v3_strict_manifest",
        output_root=_preflight_root(cfg),
        preflight=True,
        near_dup=False,
        token_sample=0,
    )
    root = _preflight_root(cfg) / "v3_strict_manifest"
    (root / "manifest.json").write_text(payload, encoding="utf-8")
    with pytest.raises(snapshot.SnapshotSafetyError, match=reason):
        snapshot.verify_sealed_snapshot(root, allow_preflight=True)


def test_production_manifest_requires_at_least_one_run_per_source(tmp_path, monkeypatch):
    _patch_supreme_validator(monkeypatch)
    cfg = _cfg(tmp_path)
    run_dirs = [
        _write_run(tmp_path, source, [], completion=True)
        for source in snapshot.SOURCES_PRESENT
    ]
    evidence = _write_evidence(tmp_path, run_dirs)
    root = tmp_path / "releases" / "v3_missing_run"
    manifest = snapshot.build_snapshot(
        cfg,
        snapshot_id=root.name,
        output_root=root.parent,
        source_state_evidence=evidence,
        near_dup=False,
        token_sample=0,
    )
    manifest["runs"] = [
        run for run in manifest["runs"] if run["source"] != "supremecourt"
    ]
    manifest["sources"]["supremecourt"]["run_ids"] = []
    manifest["snapshot_sha256"] = snapshot.snapshot_manifest_sha256(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(snapshot.SnapshotSafetyError, match="missing=.*supremecourt"):
        snapshot.verify_sealed_snapshot(root)


def test_create_only_destination_and_symlinks_fail_closed(tmp_path):
    cfg = _cfg(tmp_path)
    output_root = _preflight_root(cfg)
    destination = output_root / "v3_existing"
    destination.mkdir(parents=True)
    sentinel = destination / "owned.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(snapshot.SnapshotSafetyError, match="already exists"):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_existing",
            output_root=output_root,
            preflight=True,
            near_dup=False,
            token_sample=0,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    real_root = cfg.state_dir / "v3" / "real"
    real_root.mkdir(parents=True)
    linked_root = cfg.state_dir / "v3" / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(snapshot.SnapshotSafetyError, match="symlink"):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_symlinked",
            output_root=linked_root,
            preflight=True,
            near_dup=False,
            token_sample=0,
        )


def test_legacy_frozen_and_preflight_isolation_guards(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(snapshot.SnapshotSafetyError, match="legacy"):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v1_forbidden",
            output_root=_preflight_root(cfg),
            preflight=True,
            near_dup=False,
            token_sample=0,
        )
    with pytest.raises(snapshot.SnapshotSafetyError, match="below ingest/.state/v3"):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_wrong_root",
            output_root=tmp_path / "elsewhere",
            preflight=True,
            near_dup=False,
            token_sample=0,
        )
    with pytest.raises(snapshot.SnapshotSafetyError, match="only --preflight"):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_production_in_preflight_tree",
            output_root=_preflight_root(cfg),
            source_state_evidence=tmp_path / "unused-evidence.json",
            near_dup=False,
            token_sample=0,
        )
    with pytest.raises(snapshot.SnapshotSafetyError, match="frozen"):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_below_v1",
            output_root=snapshot.FROZEN_V1_ROOT / "nested",
            preflight=True,
            near_dup=False,
            token_sample=0,
        )


def test_limit_and_success_evidence_are_mode_bound(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(snapshot.SnapshotSafetyError, match="only with --preflight"):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_limited_prod",
            output_root=tmp_path / "releases",
            limit=1,
            token_sample=0,
        )
    with pytest.raises(snapshot.SnapshotSafetyError, match="source-state-evidence"):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_no_evidence",
            output_root=tmp_path / "releases",
            token_sample=0,
        )


def test_positive_token_sampling_requires_revision_and_propagates_failure(
    tmp_path, monkeypatch
):
    cfg = _cfg(tmp_path)
    with pytest.raises(snapshot.SnapshotSafetyError, match="TOKENIZER_REVISION"):
        snapshot.build_snapshot(
            cfg,
            snapshot_id="v3_unpinned_tokenizer",
            output_root=_preflight_root(cfg),
            preflight=True,
            near_dup=False,
            token_sample=1,
        )

    pinned = dataclasses.replace(cfg, tokenizer_revision="a" * 40)
    from ingest import embedding

    def fail_counter(*_args, **_kwargs):
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr(embedding, "make_token_counter", fail_counter)
    with pytest.raises(RuntimeError, match="tokenizer unavailable"):
        snapshot.build_snapshot(
            pinned,
            snapshot_id="v3_tokenizer_failure",
            output_root=_preflight_root(pinned),
            preflight=True,
            near_dup=False,
            token_sample=1,
        )
    assert not (_preflight_root(pinned) / "v3_tokenizer_failure").exists()


def test_tokenizer_backed_snapshot_seals_full_structural_chunk_inventory(
    tmp_path, monkeypatch
):
    cfg = dataclasses.replace(_cfg(tmp_path), tokenizer_revision="a" * 40)
    _write_run(tmp_path, "ecd", _ecd_items())
    from ingest import embedding

    monkeypatch.setattr(
        embedding,
        "make_token_counter",
        lambda *_args, **_kwargs: lambda text: len(text.split()),
    )
    manifest = snapshot.build_snapshot(
        cfg,
        snapshot_id="v3_chunk_inventory",
        output_root=_preflight_root(cfg),
        preflight=True,
        near_dup=False,
        token_sample=1,
    )
    root = _preflight_root(cfg) / "v3_chunk_inventory"
    inventory = manifest["structural_chunk_inventory"]

    assert inventory["status"] == "available"
    assert inventory["document_count"] == manifest["totals"]["clean"] == 3
    assert inventory["chunk_count"] >= inventory["document_count"]
    artifact = root / inventory["path"]
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == inventory["sha256"]
    assert snapshot.verify_sealed_snapshot(root, allow_preflight=True) == manifest


def test_frozen_candidate_requires_tokenizer_backed_chunk_inventory(tmp_path):
    cfg = dataclasses.replace(
        _cfg(tmp_path),
        artifacts_root=source_state.CANDIDATE_ARTIFACT_ROOT,
        tokenizer_revision="a" * 40,
        chunk_tokens=512,
        chunk_overlap=80,
        chunk_min_tokens=64,
    )
    candidate_output = Path(snapshot.__file__).resolve().parents[1] / "snapshots/v3"
    with pytest.raises(snapshot.SnapshotSafetyError, match="chunk inventory"):
        snapshot._validate_build_request(
            cfg,
            snapshot_id=source_state.CANDIDATE_SNAPSHOT_ID,
            output_root=candidate_output,
            sources=snapshot.SOURCES_PRESENT,
            source_state_evidence=tmp_path / "source-state.json",
            preflight=False,
            limit=None,
            token_sample=0,
        )


def test_published_modes_are_private(tmp_path):
    cfg = _cfg(tmp_path)
    snapshot.build_snapshot(
        cfg,
        snapshot_id="v3_private_modes",
        output_root=_preflight_root(cfg),
        preflight=True,
        near_dup=False,
        token_sample=0,
    )
    root = _preflight_root(cfg) / "v3_private_modes"
    assert stat_mode(root) == 0o700
    assert all(stat_mode(path) == 0o700 for path in root.rglob("*") if path.is_dir())
    assert all(stat_mode(path) == 0o600 for path in root.rglob("*") if path.is_file())


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_atomic_publisher_uses_no_replace(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    with pytest.raises(snapshot.SnapshotSafetyError, match="already exists"):
        snapshot._rename_noreplace(source, destination)
    assert source.is_dir()
    assert destination.is_dir()
    assert os.path.lexists(source)
