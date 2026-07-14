"""Offline regressions for fail-closed operational entry points."""

# ruff: noqa: E402 -- operational scripts intentionally live outside the package.

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

INGEST_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INGEST_ROOT.parent
SCRIPTS_ROOT = INGEST_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import ingest.config as config_module
import ingest.__main__ as primary_cli
from ingest.config import ConfigurationError
from ingest.operational import (
    QDRANT_WRITE_APPROVAL_ENV,
    RUNPOD_SPEND_APPROVAL_ENV,
    require_explicit_approval,
    require_run_scoped_delta_collection,
)
import ingest.qdrant_store as qdrant_store_module
import scripts.backfill_consolidation as backfill
import scripts.embed_delta as embed_delta
import scripts.merge_delta_collection as merge_delta
import scripts.reconcile_consolidated as reconcile
import scripts.reembed_v2 as reembed_v2
import runpod_orchestrate as full_orchestrator
import runpod_orchestrate_delta as delta_orchestrator
import runpod_orchestrate_delta_multi as delta_multi_orchestrator
import runpod_orchestrate_multi as multi_orchestrator
import runpod_orchestrate_reembed as reembed_orchestrator
import runpod_rerank as rerank_orchestrator


def test_explicit_approval_requires_flag_and_environment() -> None:
    with pytest.raises(SystemExit, match="--apply.*QDRANT_WRITE_APPROVED=1"):
        require_explicit_approval(
            apply=False,
            approval_env=QDRANT_WRITE_APPROVAL_ENV,
            operation="test write",
            environ={QDRANT_WRITE_APPROVAL_ENV: "1"},
        )
    with pytest.raises(SystemExit, match="--apply.*QDRANT_WRITE_APPROVED=1"):
        require_explicit_approval(
            apply=True,
            approval_env=QDRANT_WRITE_APPROVAL_ENV,
            operation="test write",
            environ={},
        )


@pytest.mark.parametrize("collection", ["georgian_legal", "georgian_legal_delta"])
def test_stable_or_live_delta_target_is_refused(collection: str) -> None:
    with pytest.raises(SystemExit, match="stable/live"):
        require_run_scoped_delta_collection(collection)


def test_run_scoped_delta_target_is_accepted() -> None:
    require_run_scoped_delta_collection(
        "georgian_legal_delta_supremecourt_20260713T120000Z"
    )


@pytest.mark.parametrize(
    ("entrypoint", "args"),
    (
        (
            primary_cli._cmd_ingest,
            SimpleNamespace(
                collection=None,
                recreate=False,
                resume=False,
                apply=False,
                source="ecd",
            ),
        ),
        (
            primary_cli._cmd_watch,
            SimpleNamespace(
                collection=None,
                recreate=False,
                apply=False,
                source="ecd",
            ),
        ),
        (
            primary_cli._cmd_embed,
            SimpleNamespace(
                collection=None,
                recreate=False,
                apply=False,
                checksum=False,
            ),
        ),
    ),
)
def test_primary_mutating_cli_refuses_live_default_before_client_or_model(
    monkeypatch, entrypoint, args
) -> None:
    cfg = dataclasses.replace(
        config_module.load_config(),
        generation_id=None,
        collection_name="georgian_legal",
    )
    monkeypatch.setattr(primary_cli, "_resolved_cfg", lambda _args: cfg)
    monkeypatch.setattr(
        qdrant_store_module,
        "make_client",
        lambda _cfg: pytest.fail("client must not be created"),
    )

    with pytest.raises(ConfigurationError, match="GENERATION_ID"):
        entrypoint(args)


@pytest.mark.parametrize(
    "entrypoint",
    [
        full_orchestrator.main,
        multi_orchestrator.main,
        delta_multi_orchestrator.main,
        reembed_orchestrator.main,
        merge_delta.main,
        backfill.main,
        reconcile.main,
        reembed_v2.main,
    ],
)
def test_legacy_mutating_entrypoints_are_disabled(entrypoint) -> None:
    with pytest.raises(SystemExit, match="Legacy .* disabled before configuration"):
        entrypoint()


@pytest.mark.parametrize(
    "operation,args",
    [
        (full_orchestrator.step_restore, (1,)),
        (multi_orchestrator.restore, (1,)),
        (delta_multi_orchestrator.step_restore_and_merge, (1,)),
        (reembed_orchestrator.step_pull_and_restore, ("127.0.0.1", 22, 1)),
    ],
)
def test_legacy_restore_helpers_are_disabled(operation, args) -> None:
    with pytest.raises(SystemExit, match="Legacy .* disabled before configuration"):
        operation(*args)


def test_delta_cli_requires_write_and_spend_approvals_before_input_access(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        delta_orchestrator,
        "_source_and_paths",
        lambda _args: pytest.fail("approval must fail before input access"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["runpod_orchestrate_delta.py", "--source", "supremecourt", "--items", "x"],
    )
    monkeypatch.delenv(QDRANT_WRITE_APPROVAL_ENV, raising=False)
    monkeypatch.delenv(RUNPOD_SPEND_APPROVAL_ENV, raising=False)
    with pytest.raises(SystemExit, match="QDRANT_WRITE_APPROVED=1"):
        delta_orchestrator.main()

    monkeypatch.setenv(QDRANT_WRITE_APPROVAL_ENV, "1")
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--apply"])
    with pytest.raises(SystemExit, match="RUNPOD_SPEND_APPROVED=1"):
        delta_orchestrator.main()


def test_reranker_requires_spend_approval_before_provisioning(monkeypatch) -> None:
    monkeypatch.setattr(
        rerank_orchestrator.O,
        "step_keypair",
        lambda: pytest.fail("approval must fail before provisioning setup"),
    )
    monkeypatch.delenv(RUNPOD_SPEND_APPROVAL_ENV, raising=False)
    with pytest.raises(SystemExit, match="--apply.*RUNPOD_SPEND_APPROVED=1"):
        rerank_orchestrator.up(apply=False)
    with pytest.raises(SystemExit, match="--apply.*RUNPOD_SPEND_APPROVED=1"):
        rerank_orchestrator.up(apply=True)


def test_missing_checksum_reference_is_never_regenerated(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(full_orchestrator, "CPU_REF", tmp_path / "missing.json")
    monkeypatch.setattr(
        full_orchestrator,
        "run",
        lambda *_args, **_kwargs: pytest.fail("checksum must not be regenerated"),
    )
    with pytest.raises(RuntimeError, match="refusing to regenerate"):
        full_orchestrator.step_checksum_ref()


def test_direct_delta_restore_requires_gate_before_configuration() -> None:
    with pytest.raises(SystemExit, match="QDRANT_WRITE_APPROVED=1"):
        delta_orchestrator.step_restore_delta(None, {}, {})


def test_delta_restore_never_deletes_an_existing_staging_collection(
    monkeypatch,
) -> None:
    ctx = SimpleNamespace(collection="georgian_legal_delta_tas_run_1")
    cfg = SimpleNamespace(
        qdrant_url="http://127.0.0.1:6333",
        qdrant_api_key=None,
    )

    class ExistingClient:
        def collection_exists(self, _collection):
            return True

        def delete_collection(self, _collection):
            pytest.fail("an existing staging collection must never be deleted")

    monkeypatch.setattr(config_module, "load_config", lambda: cfg)
    monkeypatch.setattr(qdrant_store_module, "make_client", lambda _cfg: ExistingClient())
    monkeypatch.setenv(QDRANT_WRITE_APPROVAL_ENV, "1")
    with pytest.raises(RuntimeError, match="refusing to delete or overwrite"):
        delta_orchestrator.step_restore_delta(ctx, {}, {}, apply=True)


def test_embed_delta_refuses_live_target_before_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        embed_delta,
        "load_config",
        lambda: pytest.fail("target guard must fail before configuration"),
    )
    monkeypatch.setenv(QDRANT_WRITE_APPROVAL_ENV, "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embed_delta.py",
            "--items",
            "x.jsonl",
            "--collection",
            "georgian_legal",
            "--apply",
        ],
    )
    with pytest.raises(SystemExit, match="stable/live"):
        embed_delta.main()


@pytest.mark.parametrize(
    ("ip", "port"),
    [("host; touch /tmp/x", "22"), ("127.0.0.1", "-1"), ("127.0.0.1", "abc")],
)
def test_source_endpoint_rejects_unsafe_values(ip: str, port: str) -> None:
    with pytest.raises(ValueError):
        multi_orchestrator.validated_source_endpoint(ip, port)


def test_source_endpoint_normalizes_valid_values() -> None:
    assert multi_orchestrator.validated_source_endpoint("127.0.0.1", "2222") == (
        "127.0.0.1",
        2222,
    )


def test_finish_load_refuses_before_source_or_restore_access() -> None:
    environment = os.environ.copy()
    for name in ("POD_IP", "POD_PORT"):
        environment.pop(name, None)
    result = subprocess.run(
        ["bash", str(INGEST_ROOT / "scripts" / "finish_load.sh")],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 78
    assert "Legacy direct restore" in result.stderr
    assert "POD_IP" not in result.stderr


@pytest.mark.parametrize(
    "script",
    [
        "runpod_embed.sh",
        "runpod_embed_multi.sh",
        "runpod_delta_multi.sh",
        "runpod_reembed_v2.sh",
    ],
)
def test_legacy_pod_writer_requires_ephemeral_qdrant_attestation(script: str) -> None:
    environment = os.environ.copy()
    environment.pop("RUNPOD_EPHEMERAL_QDRANT", None)
    result = subprocess.run(
        ["bash", str(INGEST_ROOT / "scripts" / script)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 78
    assert "ephemeral RunPod Qdrant" in result.stderr


def test_approved_daily_ingest_still_refuses_in_place_serving_write() -> None:
    environment = os.environ.copy()
    environment["DAILY_INGEST_APPROVED"] = "1"
    result = subprocess.run(
        ["bash", str(INGEST_ROOT / "scripts" / "daily_ingest.sh")],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 78
    assert "direct writes to a serving corpus remain disabled" in result.stderr


def test_operational_paths_are_repository_relative_or_environment_driven() -> None:
    scripts = [
        INGEST_ROOT / "scripts" / "runpod_orchestrate.py",
        INGEST_ROOT / "scripts" / "runpod_orchestrate_multi.py",
        INGEST_ROOT / "scripts" / "monitor_server.py",
        INGEST_ROOT / "scripts" / "session_monitor.py",
        INGEST_ROOT / "scripts" / "finish_load.sh",
    ]
    contents = "\n".join(path.read_text(encoding="utf-8") for path in scripts)
    assert "Path.home() / \"gpu_embed_work\"" not in contents
    assert "$HOME/gpu_embed_work" not in contents

    service = (INGEST_ROOT / "systemd" / "legal-monitor.service").read_text(
        encoding="utf-8"
    )
    assert "${LEGAL_SEARCH_REPO:?" in service
    assert "Desktop/Projects" not in service
    assert "WorkingDirectory=" not in service
