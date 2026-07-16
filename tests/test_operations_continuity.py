"""Offline regression tests for fail-closed operational entry points."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAILY_INGEST = ROOT / "ingest" / "scripts" / "daily_ingest.sh"
MONITOR_SERVER = ROOT / "ingest" / "scripts" / "monitor_server.py"
MULTI_ORCHESTRATOR = ROOT / "ingest" / "scripts" / "runpod_orchestrate_multi.py"
IMMUTABLE_BUILD_RUNBOOK = ROOT / "ingest" / "docs" / "immutable-512-build.md"


def _without(env: dict[str, str], *names: str) -> dict[str, str]:
    cleaned = env.copy()
    for name in names:
        cleaned.pop(name, None)
    return cleaned


def test_non_dry_daily_ingest_always_requires_explicit_approval() -> None:
    env = _without(os.environ, "DAILY_INGEST_APPROVED")
    env["DAILY_INGEST_REQUIRE_APPROVAL"] = "0"

    result = subprocess.run(
        ["bash", str(DAILY_INGEST)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 78
    assert "daily ingest refused" in result.stderr


def test_daily_ingest_default_includes_supreme_court() -> None:
    script = DAILY_INGEST.read_text(encoding="utf-8")
    assert (
        "matsne ecd constcourt napr supremecourt tas tbappeal" in script
    )


def test_monitor_requires_explicit_source_pod_configuration() -> None:
    env = _without(os.environ, "MON_POD_IP", "MON_POD_PORT")

    result = subprocess.run(
        [sys.executable, str(MONITOR_SERVER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode != 0
    assert "MON_POD_IP and MON_POD_PORT are required" in result.stderr


def test_emergency_terminate_does_not_require_source_pod_config(tmp_path: Path) -> None:
    env = _without(os.environ, "RUNPOD_SOURCE_IP", "RUNPOD_SOURCE_PORT")
    env["HOME"] = str(tmp_path)
    env["GPU_WORKDIR"] = str(tmp_path / "gpu-work")

    result = subprocess.run(
        [sys.executable, str(MULTI_ORCHESTRATOR), "terminate"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0
    assert "no pod_multi.id" in result.stdout


def test_operational_files_have_no_historical_machine_or_pod_defaults() -> None:
    paths = [
        ROOT / "ingest" / "scripts" / name
        for name in (
            "finish_load.sh",
            "monitor_server.py",
            "phase_c.sh",
            "phase_c_full.sh",
            "runpod_orchestrate_multi.py",
            "session_monitor.py",
        )
    ]
    contents = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "/home/bezhomatiashvili" not in contents
    assert "209.170.80.132" not in contents
    assert "47.47.180.65" not in contents


def test_ci_covers_serving_branches_and_keeps_symbol_map_gate() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "branches: [main, dev, master]" in workflow
    assert "scripts/gen_code_map.py --check" in workflow


def test_legacy_optional_approval_switch_is_absent() -> None:
    script = DAILY_INGEST.read_text(encoding="utf-8")
    service = (ROOT / "ingest" / "systemd" / "legal-ingest.service").read_text(
        encoding="utf-8"
    )

    assert "DAILY_INGEST_REQUIRE_APPROVAL" not in script
    assert "DAILY_INGEST_REQUIRE_APPROVAL" not in service
    assert '${DAILY_INGEST_APPROVED:-0}' in script


def test_systemd_unit_requires_a_configured_repository_root() -> None:
    service = (ROOT / "ingest" / "systemd" / "legal-ingest.service").read_text(
        encoding="utf-8"
    )

    assert "EnvironmentFile=-%h/.config/georgia-legal-search.env" in service
    assert "${LEGAL_SEARCH_REPO:?" in service
    assert "$LEGAL_SEARCH_REPO/ingest/scripts/daily_ingest.sh" in service
    assert "Desktop/Projects" not in service
    assert "WorkingDirectory=" not in service


def test_immutable_build_runbook_requires_new_exact_crawl_evidence() -> None:
    runbook = IMMUTABLE_BUILD_RUNBOOK.read_text(encoding="utf-8")

    assert "--only matsne ecd constcourt napr tas tbappeal" in runbook
    assert "--only supremecourt" in runbook
    assert "--max-runtime-seconds 14400" in runbook
    assert "-s CLOSESPIDER_TIMEOUT=14400" in runbook
    assert "scripts/build_source_state_evidence.py" in runbook
    for source in (
        "matsne",
        "napr",
        "ecd",
        "constcourt",
        "supremecourt",
        "tas",
        "tbappeal",
    ):
        assert f"--select {source}:EXACT_" in runbook
    assert "AWAITING INDEPENDENT OPERATOR REVIEW" in runbook
    assert "Historical startup-only records must never" in runbook
    assert "REVIEWED_SOURCE_STATE_EVIDENCE_SHA256" in runbook
