from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ingest import candidate_handoff as ch
from ingest.candidate_handoff import (
    CandidateHandoffError,
    create_inconclusive_handoff,
)
from ingest.release_inputs import GENERATION_ID


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "presentation", repo / "presentation")
    symbols = repo / "memory-bank/generated/symbols.md"
    symbols.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "memory-bank/generated/symbols.md", symbols)
    git_dir = repo / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(ch.INITIAL_REPOSITORY_HEAD + "\n", encoding="ascii")
    safety_root = repo / "ingest/.state/v3/release-safety" / GENERATION_ID
    safety_root.mkdir(parents=True)
    validation = safety_root / "release-input-validation-01.json"
    validation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generation_id": GENERATION_ID,
                "status": "inconclusive",
                "ceiling": "no_network_or_paid_work",
                "checked_at": "2026-07-17T12:53:13Z",
                "bundle": ch.RELEASE_BUNDLE_PATH,
                "reason": "release bundle is absent",
            }
        ),
        encoding="utf-8",
    )
    inventory = safety_root / "initial-worktree-inventory.json"
    shutil.copy2(
        REPO_ROOT
        / "ingest/.state/v3/release-safety"
        / GENERATION_ID
        / "initial-worktree-inventory.json",
        inventory,
    )
    return repo, validation, inventory


def _symbols_sha256(repo: Path) -> str:
    return hashlib.sha256(
        (repo / "memory-bank/generated/symbols.md").read_bytes()
    ).hexdigest()


def _checks(repo: Path) -> list[dict[str, str]]:
    symbols_sha = _symbols_sha256(repo)
    return [
        {
            "name": ch.CODE_MAP_CHECK_NAME,
            "command": (
                "python3 ingest/scripts/gen_code_map.py && "
                "python3 ingest/scripts/gen_code_map.py --check"
            ),
            "status": "passed",
            "result": (
                f"current_symbols_sha256={symbols_sha}; "
                "regeneration=passed; check=passed"
            ),
        },
        {
            "name": "focused tests",
            "command": "pytest -q tests/test_candidate_handoff.py",
            "status": "passed",
            "result": "1 passed",
        },
    ]


def test_inconclusive_handoff_is_create_only_and_explicit(tmp_path):
    repo, validation, inventory = _write_inputs(tmp_path)
    output = tmp_path / "handoff.json"
    checks = _checks(repo)
    symbols_sha = _symbols_sha256(repo)

    create_inconclusive_handoff(
        validation_report=validation,
        safety_inventory=inventory,
        checks=checks,
        output=output,
        repo_root=repo,
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
        expected_current_symbols_sha256=symbols_sha,
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["verdict"] == "inconclusive"
    assert value["external_resource_usage"]["paid_compute_usd"] == 0.0
    assert (
        value["external_resource_usage"]["external_state_observability"]
        == "not_observable_from_supplied_local_evidence"
    )
    assert (
        value["external_resource_usage"]["scope"] == "actions_performed_in_this_session"
    )
    assert value["promotion"]["alias_operations"] == 0
    assert value["promotion"]["plan_created"] is False
    assert (
        value["promotion"]["external_alias_state_observability"]
        == "not_observable_from_supplied_local_evidence"
    )
    assert value["next_experiment"]["selected_experiment"] is None
    assert value["safety"]["aggregate_inventory_validated"] is True
    assert len(value["safety"]["verified_absent_local_candidate_paths"]) == 9
    assert value["safety"]["current_generated_symbols_sha256"] == symbols_sha
    assert (
        value["safety"]["initial_accepted_generated_symbols_sha256"]
        == ch.INITIAL_SYMBOLS_SHA256
    )
    assert value["safety"]["preservation_basis"]
    with pytest.raises(FileExistsError):
        create_inconclusive_handoff(
            validation_report=validation,
            safety_inventory=inventory,
            checks=checks,
            output=output,
            repo_root=repo,
            expected_current_symbols_sha256=symbols_sha,
        )


def test_inconclusive_handoff_refuses_presentation_drift(tmp_path):
    repo, validation, inventory = _write_inputs(tmp_path)
    (repo / "presentation/accuracy-first-demo/README.md").write_text(
        "changed\n", encoding="utf-8"
    )

    with pytest.raises(CandidateHandoffError, match="presentation tree bytes changed"):
        create_inconclusive_handoff(
            validation_report=validation,
            safety_inventory=inventory,
            checks=[],
            output=tmp_path / "handoff.json",
            repo_root=repo,
        )


def test_inconclusive_handoff_refuses_fabricated_minimal_gate_report(tmp_path):
    repo, validation, inventory = _write_inputs(tmp_path)
    validation.write_text(
        json.dumps(
            {
                "generation_id": GENERATION_ID,
                "status": "inconclusive",
                "ceiling": "no_network_or_paid_work",
                "reason": "fabricated",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CandidateHandoffError, match="report keys mismatch"):
        create_inconclusive_handoff(
            validation_report=validation,
            safety_inventory=inventory,
            checks=[],
            output=tmp_path / "handoff.json",
            repo_root=repo,
        )


def test_inconclusive_handoff_refuses_nonofficial_inventory_bytes(tmp_path):
    repo, validation, inventory = _write_inputs(tmp_path)
    value = json.loads(inventory.read_text(encoding="utf-8"))
    value["git_status_entry_count"] += 1
    inventory.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(CandidateHandoffError, match="bytes are not official"):
        create_inconclusive_handoff(
            validation_report=validation,
            safety_inventory=inventory,
            checks=[],
            output=tmp_path / "handoff.json",
            repo_root=repo,
        )


@pytest.mark.parametrize(
    "relative",
    (
        f"ingest/snapshots/v3/{ch.SNAPSHOT_ID}",
        f"ingest/.state/v3/prepared/{GENERATION_ID}",
        f"ingest/.state/embed/{GENERATION_ID}",
        (f"artifacts/generations/{GENERATION_ID}.physical-v3-512-01.verification.json"),
    ),
)
def test_inconclusive_handoff_refuses_existing_candidate_artifact(tmp_path, relative):
    repo, validation, inventory = _write_inputs(tmp_path)
    candidate = repo / relative
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if candidate.suffix:
        candidate.write_text("candidate evidence\n", encoding="utf-8")
    else:
        candidate.mkdir()

    with pytest.raises(CandidateHandoffError, match="candidate artifact exists"):
        create_inconclusive_handoff(
            validation_report=validation,
            safety_inventory=inventory,
            checks=[],
            output=tmp_path / "handoff.json",
            repo_root=repo,
        )


def test_inconclusive_handoff_requires_passed_code_map_check(tmp_path):
    repo, validation, inventory = _write_inputs(tmp_path)

    with pytest.raises(CandidateHandoffError, match="exactly one.*code map"):
        create_inconclusive_handoff(
            validation_report=validation,
            safety_inventory=inventory,
            checks=[],
            output=tmp_path / "handoff.json",
            repo_root=repo,
            expected_current_symbols_sha256=_symbols_sha256(repo),
        )


def test_inconclusive_handoff_accepts_digest_bound_code_map_regeneration(tmp_path):
    repo, validation, inventory = _write_inputs(tmp_path)
    symbols = repo / "memory-bank/generated/symbols.md"
    symbols.write_text("intentionally regenerated code map\n", encoding="utf-8")
    current_sha = _symbols_sha256(repo)
    output = tmp_path / "handoff.json"

    create_inconclusive_handoff(
        validation_report=validation,
        safety_inventory=inventory,
        checks=_checks(repo),
        output=output,
        repo_root=repo,
        expected_current_symbols_sha256=current_sha,
    )

    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["safety"]["generated_symbols_changed_since_initial"] is True
    assert value["safety"]["current_generated_symbols_sha256"] == current_sha
    assert value["safety"]["code_map_regeneration_check"]["status"] == "passed"


def test_inconclusive_handoff_rejects_unexplained_generated_symbols_drift(tmp_path):
    repo, validation, inventory = _write_inputs(tmp_path)
    (repo / "memory-bank/generated/symbols.md").write_text(
        "unreviewed drift\n", encoding="utf-8"
    )

    with pytest.raises(CandidateHandoffError, match="explicit expected current"):
        create_inconclusive_handoff(
            validation_report=validation,
            safety_inventory=inventory,
            checks=_checks(repo),
            output=tmp_path / "handoff.json",
            repo_root=repo,
        )


def test_inconclusive_handoff_rejects_wrong_expected_symbols_digest(tmp_path):
    repo, validation, inventory = _write_inputs(tmp_path)

    with pytest.raises(CandidateHandoffError, match="do not match the reviewed"):
        create_inconclusive_handoff(
            validation_report=validation,
            safety_inventory=inventory,
            checks=_checks(repo),
            output=tmp_path / "handoff.json",
            repo_root=repo,
            expected_current_symbols_sha256="f" * 64,
        )
