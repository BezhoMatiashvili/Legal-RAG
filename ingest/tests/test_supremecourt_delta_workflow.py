"""Offline gates for the run-scoped Supreme Court RunPod delta workflow."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import embed_delta  # noqa: E402
import runpod_orchestrate_delta as orch  # noqa: E402
import ingest.config as config_module  # noqa: E402
import ingest.qdrant_store as qdrant_store_module  # noqa: E402
from ingest.qdrant_store import point_id  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_orchestrator_globals():
    orch._created_pod_ids.clear()  # noqa: SLF001
    orch._created_pod_name = None  # noqa: SLF001
    orch.O._pod_id = None
    orch.O._ip = None
    orch.O._port = None
    orch.O._provisioned_at = None
    orch.O._terminated = False
    yield
    orch._created_pod_ids.clear()  # noqa: SLF001
    orch._created_pod_name = None  # noqa: SLF001


def _supreme_item(case_id: str, chamber: str | None = None) -> dict:
    item = {
        "case_id": case_id,
        "case_number": f"ას-{case_id}-2026",
        "body_markdown": (
            "საქართველოს უზენაესი სასამართლოს სრული გადაწყვეტილება. " * 12
        ),
    }
    if chamber is not None:
        item["chamber"] = chamber
    return item


def test_run_scoped_collection_is_source_aware_and_sanitized(tmp_path):
    first = orch.make_run("supremecourt", "2026/07/13 12:00Z", workdir=tmp_path)
    second = orch.make_run("supremecourt", "2026/07/13 12:01Z", workdir=tmp_path)

    assert first.collection == "georgian_legal_delta_supremecourt_2026_07_13_12_00Z"
    assert first.collection != second.collection
    assert first.out != second.out
    assert first.collection != "georgian_legal_delta"
    assert "/" not in first.collection and " " not in first.collection


def test_run_scoped_staging_guard_is_narrow_and_preserves_generation_guards():
    base = config_module.load_config()
    delta = dataclasses.replace(
        base,
        generation_id=None,
        collection_name="georgian_legal_delta_supremecourt_run_20260713",
    )
    approvals = {
        "QDRANT_WRITE_APPROVED": "1",
        "QDRANT_RECREATE_APPROVED": "1",
        "RUNPOD_EPHEMERAL_QDRANT": "1",
    }

    qdrant_store_module.validate_generation_write_target(
        delta,
        apply=True,
        recreate=True,
        allow_run_scoped_delta=True,
        environ=approvals,
    )

    with pytest.raises(config_module.ConfigurationError, match="GENERATION_ID"):
        qdrant_store_module.validate_generation_write_target(
            delta, apply=True, recreate=True, environ=approvals
        )
    with pytest.raises(config_module.ConfigurationError, match="run-scoped collection"):
        qdrant_store_module.validate_generation_write_target(
            dataclasses.replace(delta, collection_name="georgian_legal"),
            apply=True,
            allow_run_scoped_delta=True,
            environ=approvals,
        )
    with pytest.raises(config_module.ConfigurationError, match="must not claim"):
        qdrant_store_module.validate_generation_write_target(
            dataclasses.replace(delta, generation_id="gen_20260713_verified"),
            apply=True,
            allow_run_scoped_delta=True,
            environ=approvals,
        )

    generation = dataclasses.replace(
        base,
        generation_id="gen_20260713_verified",
        collection_name="georgian_legal__gen_gen_20260713_verified",
    )
    qdrant_store_module.validate_generation_write_target(
        generation,
        apply=True,
        environ={"QDRANT_WRITE_APPROVED": "1"},
    )
    with pytest.raises(config_module.ConfigurationError, match="physical collection"):
        qdrant_store_module.validate_generation_write_target(
            dataclasses.replace(generation, collection_name="georgian_legal"),
            apply=True,
            environ={"QDRANT_WRITE_APPROVED": "1"},
        )


def test_pod_write_preflight_requires_recreate_approval_before_spend(
    monkeypatch, tmp_path
):
    ctx = orch.make_run("supremecourt", "run-1", workdir=tmp_path)
    base = dataclasses.replace(config_module.load_config(), generation_id=None)
    monkeypatch.setattr(config_module, "load_config", lambda: base)
    monkeypatch.setattr(orch.O, "log", lambda _message: None)

    orch.preflight_pod_delta_write(
        ctx,
        environ={
            "QDRANT_WRITE_APPROVED": "1",
            "QDRANT_RECREATE_APPROVED": "1",
        },
    )

    with pytest.raises(config_module.ConfigurationError, match="RECREATE"):
        orch.preflight_pod_delta_write(ctx, environ={"QDRANT_WRITE_APPROVED": "1"})
    with pytest.raises(config_module.ConfigurationError, match="WRITE"):
        orch.preflight_pod_delta_write(ctx, environ={"QDRANT_RECREATE_APPROVED": "1"})


def test_explicit_items_fail_closed_when_any_file_is_missing(tmp_path):
    valid = tmp_path / "run" / "items.jsonl"
    valid.parent.mkdir()
    valid.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing/empty"):
        orch.stage_delta_items(
            tmp_path / "unused",
            None,
            tmp_path / "stage",
            explicit=[valid, tmp_path / "missing" / "items.jsonl"],
        )


def test_explicit_new_item_journal_is_a_valid_delta_input(tmp_path):
    chamber = "სამოქალაქო საქმეთა პალატა"
    journal = tmp_path / "crawl-run" / "items.journal.jsonl"
    journal.parent.mkdir()
    journal.write_text(
        json.dumps(_supreme_item("101", chamber), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    staged = orch.stage_delta_items(
        tmp_path / "unused",
        None,
        tmp_path / "stage",
        explicit=[journal],
    )

    assert len(staged) == 1
    assert staged[0].read_bytes() == journal.read_bytes()


def test_supremecourt_manifest_has_exact_composite_ids_chunks_and_skips(tmp_path):
    chamber = "სამოქალაქო საქმეთა პალატა"
    path = tmp_path / "items.jsonl"
    rows = [_supreme_item("101", chamber), _supreme_item("102", chamber)]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    cfg = SimpleNamespace(chunk_tokens=64, chunk_overlap=8, chunk_min_tokens=1)

    docs, manifest = embed_delta._analyze_docs(  # noqa: SLF001 - direct offline contract test
        cfg,
        [("supremecourt", [path])],
        lambda text: max(1, len(text.split())),
    )

    assert len(docs) == manifest["documents"] == 2
    assert manifest["chunks"] > 0
    assert manifest["skipped"] == 0
    assert manifest["document_ids"] == [
        f"supremecourt\t101:{chamber}",
        f"supremecourt\t102:{chamber}",
    ]
    assert len(manifest["input_sha256"]) == 64
    assert len(manifest["document_ids_sha256"]) == 64


def test_supremecourt_manifest_rejects_missing_chamber_identity(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(
        json.dumps(_supreme_item("101"), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    cfg = SimpleNamespace(chunk_tokens=64, chunk_overlap=8, chunk_min_tokens=1)

    _docs, manifest = embed_delta._analyze_docs(  # noqa: SLF001
        cfg,
        [("supremecourt", [path])],
        lambda text: max(1, len(text.split())),
    )

    assert manifest["documents"] == 0
    assert manifest["skipped"] == 1
    assert "missing identity field(s): chamber" in manifest["failures"][0]["reason"]
    with pytest.raises(RuntimeError, match="no embeddable documents"):
        embed_delta._strict_manifest_check(manifest)  # noqa: SLF001


def test_supremecourt_manifest_rejects_unofficial_or_duplicate_identity(tmp_path):
    official = "სამოქალაქო საქმეთა პალატა"
    rows = [
        _supreme_item("101", official),
        _supreme_item("101", official),
        _supreme_item("102", "not an official chamber"),
    ]
    path = tmp_path / "items.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    cfg = SimpleNamespace(chunk_tokens=64, chunk_overlap=8, chunk_min_tokens=1)

    _docs, manifest = embed_delta._analyze_docs(  # noqa: SLF001
        cfg,
        [("supremecourt", [path])],
        lambda text: max(1, len(text.split())),
    )

    assert manifest["documents"] == 1
    assert manifest["skipped"] == 2
    reasons = " ".join(failure["reason"] for failure in manifest["failures"])
    assert "duplicate Supreme Court identity" in reasons
    assert "unofficial Supreme Court chamber" in reasons


def test_spend_gate_requires_no_active_pods_and_preserves_two_dollar_reserve(
    monkeypatch,
):
    monkeypatch.setattr(orch, "_account_state", lambda: (10.0, []))
    monkeypatch.setattr(orch.O, "gpu_price", lambda gpu: (0.69, "Low"))

    gate = orch.runpod_spend_gate(120_000)

    assert gate.price == 0.69
    cleanup_cost = orch.BILLING_CLEANUP_MARGIN_S / 3600 * gate.price
    assert gate.max_compute_cost == pytest.approx(8.0 - cleanup_cost)
    assert gate.estimated_cost + cleanup_cost + orch.RESERVE_USD <= gate.balance

    monkeypatch.setattr(
        orch,
        "_account_state",
        lambda: (
            10.0,
            [{"id": "other", "name": "billing", "desiredStatus": "RUNNING"}],
        ),
    )
    with pytest.raises(RuntimeError, match="active pod"):
        orch.runpod_spend_gate(1)


def test_spend_gate_fails_when_conservative_cost_plus_reserve_exceeds_balance(
    monkeypatch,
):
    monkeypatch.setattr(orch, "_account_state", lambda: (2.25, []))
    monkeypatch.setattr(orch.O, "gpu_price", lambda gpu: (0.69, "Medium"))

    with pytest.raises(RuntimeError, match=r"cleanup margin, and \$2.00 reserve"):
        orch.runpod_spend_gate(120_000)


def test_local_restore_preflight_rejects_remote_or_existing_target(
    monkeypatch, tmp_path
):
    ctx = orch.make_run("supremecourt", "run-1", workdir=tmp_path)
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: SimpleNamespace(
            qdrant_url="https://qdrant.example", qdrant_api_key="x"
        ),
    )
    monkeypatch.setattr(
        qdrant_store_module,
        "make_client",
        lambda _cfg: pytest.fail("remote Qdrant must fail before client construction"),
    )
    with pytest.raises(RuntimeError, match="local Qdrant"):
        orch.preflight_local_delta_restore(ctx)

    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: SimpleNamespace(
            qdrant_url="http://127.0.0.1:6333", qdrant_api_key=None
        ),
    )
    monkeypatch.setattr(
        qdrant_store_module,
        "make_client",
        lambda _cfg: SimpleNamespace(collection_exists=lambda _name: True),
    )
    with pytest.raises(RuntimeError, match="paid work.*already exists"):
        orch.preflight_local_delta_restore(ctx)


def test_runpod_spend_lock_is_exclusive_and_reusable(tmp_path):
    lock = tmp_path / "runpod-spend.lock"

    with orch.runpod_spend_lock(lock):
        with pytest.raises(RuntimeError, match="another local RunPod spend workflow"):
            with orch.runpod_spend_lock(lock):
                pytest.fail("the second holder must never enter")

    with orch.runpod_spend_lock(lock):
        pass


def test_locked_paid_workflow_holds_lock_across_gate_and_cleanup_return(
    monkeypatch, tmp_path
):
    ctx = orch.make_run("supremecourt", "run-1", workdir=tmp_path)
    gate = orch.SpendGate(10.0, 0.69, "High", 1.0, 0.69, 8.0)
    events = []

    class Lock:
        def __enter__(self):
            events.append("lock-enter")

        def __exit__(self, *_args):
            events.append("lock-exit")

    monkeypatch.setattr(orch, "runpod_spend_lock", lambda: Lock())
    monkeypatch.setattr(
        orch,
        "preflight_pod_delta_write",
        lambda _ctx: events.append("pod-write-preflight"),
    )
    monkeypatch.setattr(
        orch, "runpod_spend_gate", lambda _chunks: events.append("gate") or gate
    )
    monkeypatch.setattr(
        orch, "step_package_delta", lambda _ctx: events.append("package") or "secret"
    )
    monkeypatch.setattr(
        orch.O, "step_keypair", lambda: events.append("keypair") or "public-key"
    )
    monkeypatch.setattr(
        orch,
        "run_paid_workflow",
        lambda *_args, **_kwargs: (
            events.append("paid-cleanup-returned") or {"ok": True}
        ),
    )

    assert orch.run_locked_paid_workflow(ctx, {"chunks": 7}, apply=True) == {"ok": True}
    assert events == [
        "pod-write-preflight",
        "lock-enter",
        "gate",
        "package",
        "keypair",
        "gate",
        "paid-cleanup-returned",
        "lock-exit",
    ]


def test_provider_attestation_requires_one_named_pod_below_gated_price(
    monkeypatch, tmp_path
):
    ctx = orch.make_run("supremecourt", "run-1", workdir=tmp_path)
    gate = orch.SpendGate(10.0, 0.69, "High", 1.0, 0.69, 8.0)
    pod = {
        "id": "pod-4090",
        "name": ctx.pod_name,
        "desiredStatus": "RUNNING",
        "costPerHr": 0.68,
    }
    monkeypatch.setattr(orch, "_account_state", lambda: (10.0, [pod]))
    monkeypatch.setattr(orch, "_secure_cloud_state", lambda _pod_id: True)

    assert orch.attest_provider_pod(ctx, "pod-4090", gate) == 0.68

    pod["costPerHr"] = 0.70
    with pytest.raises(RuntimeError, match="exceeds gated price"):
        orch.attest_provider_pod(ctx, "pod-4090", gate)

    pod["costPerHr"] = None
    with pytest.raises(RuntimeError, match="unavailable costPerHr"):
        orch.attest_provider_pod(ctx, "pod-4090", gate)


def test_exposed_secure_cloud_false_is_rejected(monkeypatch, tmp_path):
    ctx = orch.make_run("supremecourt", "run-1", workdir=tmp_path)
    gate = orch.SpendGate(10.0, 0.69, "High", 1.0, 0.69, 8.0)
    monkeypatch.setattr(
        orch,
        "_account_state",
        lambda: (
            10.0,
            [
                {
                    "id": "pod-4090",
                    "name": ctx.pod_name,
                    "desiredStatus": "RUNNING",
                    "costPerHr": 0.68,
                }
            ],
        ),
    )
    monkeypatch.setattr(orch, "_secure_cloud_state", lambda _pod_id: False)

    with pytest.raises(RuntimeError, match="Secure Cloud is false"):
        orch.attest_provider_pod(ctx, "pod-4090", gate)


def test_unexposed_secure_cloud_field_is_recorded_as_optional(monkeypatch):
    monkeypatch.setattr(
        orch.O,
        "gql",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError(
                'GraphQL error: Cannot query field "secureCloud" on type "Machine"'
            )
        ),
    )

    assert orch._secure_cloud_state("pod-4090") is None  # noqa: SLF001


def test_hardware_attestation_requires_exactly_one_rtx_4090(monkeypatch):
    monkeypatch.setattr(
        orch.O,
        "ssh_capture",
        lambda *_args, **_kwargs: "NVIDIA GeForce RTX 4090\nNVIDIA GeForce RTX 4090\n",
    )
    with pytest.raises(RuntimeError, match="expected exactly"):
        orch.attest_single_4090("127.0.0.1", 22)


def test_reserve_budget_is_checked_from_provision_attempt(monkeypatch):
    monkeypatch.setattr(orch.time, "time", lambda: 3_601.0)

    with pytest.raises(RuntimeError, match=r"during after setup.*\$2.00 reserve"):
        orch.enforce_reserve_budget(
            price=1.0,
            provisioned_at=0.0,
            max_compute_cost=1.0,
            phase="after setup",
        )


def test_provision_requests_exactly_one_secure_4090_without_fallback(
    monkeypatch, tmp_path
):
    calls = []

    def fake_gql(query, variables=None):
        calls.append((query, variables))
        return {"podFindAndDeployOnDemand": {"id": "pod-4090"}}

    monkeypatch.setattr(orch.O, "gql", fake_gql)
    monkeypatch.setattr(orch.O, "WORKDIR", tmp_path)
    orch._created_pod_ids.clear()  # noqa: SLF001
    ctx = orch.make_run("supremecourt", "run-1", workdir=tmp_path)

    assert orch.step_provision_4090("ssh-ed25519 test", ctx, 0.69) == "pod-4090"

    request = calls[0][1]["input"]
    assert request["cloudType"] == "SECURE"
    assert request["gpuCount"] == 1
    assert request["gpuTypeId"] == "NVIDIA GeForce RTX 4090"
    assert len(calls) == 1


def test_checksum_is_a_separate_gate_before_corpus_embedding():
    script = (SCRIPTS / "runpod_embed_delta.sh").read_text(encoding="utf-8")
    checksum = script.index("# 4. G2 gate BEFORE corpus embedding")
    actual_embed = script.index("--batch-size 256 --strict --recreate")

    assert "checksum_gpu.json" in script
    assert "cosine < 0.999" in script
    assert checksum < actual_embed
    assert 'RERANK_ENABLED="false"' in script
    assert 'EMBED_DEVICE="cuda"' in script
    assert 'EMBED_USE_FP16="true"' in script
    assert "${RUNPOD_EPHEMERAL_QDRANT:-0}" in script
    assert "${QDRANT_WRITE_APPROVED:-0}" in script
    assert "${QDRANT_RECREATE_APPROVED:-0}" in script
    assert "unset GENERATION_ID GENERATION_DIR" in script
    assert 'PRODUCTION_MODE="false"' in script


def test_launch_propagates_ephemeral_write_and_recreate_attestations(
    monkeypatch, tmp_path
):
    ctx = orch.make_run("supremecourt", "run-1", workdir=tmp_path)
    pushed = {}
    remote_commands = []

    def capture(_ip, _port, path, content, *, mode):
        pushed[path] = (content, mode)

    monkeypatch.setattr(orch.O, "push_content", capture)

    def capture_remote(_ip, _port, command, **_kwargs):
        remote_commands.append(command)
        return "LAUNCHED\n"

    monkeypatch.setattr(orch.O, "ssh_capture", capture_remote)

    orch.step_launch_delta("127.0.0.1", 22, "passphrase", ctx)

    launch, mode = pushed["/workspace/launch_delta.sh"]
    assert mode == "755"
    assert "QDRANT_WRITE_APPROVED=1" in launch
    assert "QDRANT_RECREATE_APPROVED=1" in launch
    assert "RUNPOD_EPHEMERAL_QDRANT=1" in launch
    assert "GENERATION_ID=" not in launch
    assert "launch.pid" in remote_commands[0]


def test_poll_uses_exact_launch_pid_instead_of_self_matching_pgrep(
    monkeypatch,
):
    commands = []
    checks = iter([False, False, False])

    def ssh_ok(_ip, _port, command, **_kwargs):
        commands.append(command)
        return next(checks)

    monkeypatch.setattr(orch.O, "ssh_ok", ssh_ok)
    monkeypatch.setattr(orch.O, "ssh_capture", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(orch, "enforce_reserve_budget", lambda **_kwargs: None)
    monkeypatch.setattr(orch, "DEAD_CHECKS", 1)
    monkeypatch.setattr(orch.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="ended without DONE"):
        orch.step_poll_delta(
            "127.0.0.1",
            22,
            price=0.5,
            provisioned_at=orch.time.monotonic(),
            max_compute_cost=10.0,
        )

    liveness = [command for command in commands if "kill -0" in command]
    assert liveness
    assert all("launch.pid" in command for command in liveness)
    assert all("pgrep" not in command for command in commands)


def test_run_manifest_requires_exact_snapshot_hash_documents_chunks_and_points(
    tmp_path,
):
    ctx = orch.make_run("supremecourt", "run-1", workdir=tmp_path)
    ctx.out.mkdir(parents=True)
    ctx.snapshot.write_bytes(b"valid snapshot bytes")
    document_ids = ["supremecourt\tdoc-1", "supremecourt\tdoc-2"]
    document_ids_sha256 = hashlib.sha256(
        "\n".join(document_ids).encode("utf-8")
    ).hexdigest()
    expected = {
        "input_sha256": "a" * 64,
        "document_ids": document_ids,
        "document_ids_sha256": document_ids_sha256,
        "documents": 2,
        "chunks": 3,
    }
    run = {
        "source": "supremecourt",
        "run_id": "run-1",
        "collection": ctx.collection,
        "input_sha256": "a" * 64,
        "document_ids": document_ids,
        "document_ids_sha256": document_ids_sha256,
        "expected_documents": 2,
        "documents": 2,
        "expected_chunks": 3,
        "chunks": 3,
        "points_count": 3,
        "skipped": 0,
        "snapshot": ctx.snapshot.name,
        "snapshot_size_bytes": ctx.snapshot.stat().st_size,
        "snapshot_sha256": hashlib.sha256(ctx.snapshot.read_bytes()).hexdigest(),
        "checksum_cosine": 0.9999,
        "gpu": orch.GPU,
    }

    orch.validate_run_manifest(ctx, expected, run, ctx.snapshot)

    bad = dict(run, points_count=2)
    with pytest.raises(RuntimeError, match="points_count"):
        orch.validate_run_manifest(ctx, expected, bad, ctx.snapshot)
    bad = dict(run, snapshot_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="snapshot_sha256"):
        orch.validate_run_manifest(ctx, expected, bad, ctx.snapshot)
    bad = dict(run, document_ids_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="document_ids_sha256 is not canonical"):
        orch.validate_run_manifest(ctx, expected, bad, ctx.snapshot)


def test_restored_points_require_uuid5_exact_ids_and_contiguous_chunks(tmp_path):
    ctx = orch.make_run("supremecourt", "run-1", workdir=tmp_path)
    document_id = "101:სამოქალაქო საქმეთა პალატა"
    expected = {
        "document_ids": [f"supremecourt\t{document_id}"],
        "chunks": 2,
    }
    records = [
        SimpleNamespace(
            id=point_id("supremecourt", document_id, index),
            payload={
                "source": "supremecourt",
                "document_id": document_id,
                "chunk_index": index,
                "document_chunk_count": 2,
            },
        )
        for index in (0, 1)
    ]

    orch.validate_restored_records(records, ctx, expected)

    records[1].id = "00000000-0000-0000-0000-000000000000"
    with pytest.raises(RuntimeError, match="UUIDv5"):
        orch.validate_restored_records(records, ctx, expected)


def test_paid_workflow_runs_strict_cleanup_even_when_provision_raises(
    monkeypatch, tmp_path
):
    ctx = orch.make_run("supremecourt", "run-1", workdir=tmp_path)
    gate = orch.SpendGate(10.0, 0.69, "High", 1.0, 0.69, 8.0)
    cleanup = []
    monkeypatch.setattr(
        orch,
        "step_provision_4090",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("deploy response lost")
        ),
    )
    monkeypatch.setattr(
        orch, "_cleanup_delta", lambda *, strict: cleanup.append(strict)
    )
    orch.O._ip = None
    orch.O._port = None
    orch.O._provisioned_at = None
    monkeypatch.setenv("RUNPOD_SPEND_APPROVED", "1")
    monkeypatch.setenv("QDRANT_WRITE_APPROVED", "1")
    monkeypatch.setenv("QDRANT_RECREATE_APPROVED", "1")

    with pytest.raises(RuntimeError, match="deploy response lost"):
        orch.run_paid_workflow(ctx, {}, gate, "passphrase", "pubkey", apply=True)

    assert cleanup == [True]


def test_paid_workflow_checks_budget_around_setup_launch_and_transfer(
    monkeypatch, tmp_path
):
    ctx = orch.make_run("supremecourt", "run-1", workdir=tmp_path)
    gate = orch.SpendGate(10.0, 0.69, "High", 1.0, 0.69, 8.0)
    events = []

    monkeypatch.setenv("RUNPOD_SPEND_APPROVED", "1")
    monkeypatch.setenv("QDRANT_WRITE_APPROVED", "1")
    monkeypatch.setenv("QDRANT_RECREATE_APPROVED", "1")
    monkeypatch.setattr(orch, "step_provision_4090", lambda *_args: "pod-4090")
    monkeypatch.setattr(
        orch,
        "enforce_reserve_budget",
        lambda **kwargs: events.append(kwargs["phase"]),
    )
    monkeypatch.setattr(orch, "_retry", lambda fn, **_kwargs: fn())
    monkeypatch.setattr(
        orch,
        "attest_provider_pod",
        lambda *_args: events.append("provider-attestation"),
    )
    monkeypatch.setattr(orch.O, "step_wait_ssh", lambda _pod_id: ("127.0.0.1", 22))
    monkeypatch.setattr(
        orch,
        "attest_single_4090",
        lambda *_args: events.append("hardware-attestation"),
    )
    monkeypatch.setattr(orch.O, "ensure_pod_tools", lambda *_args: None)
    monkeypatch.setattr(orch.O, "push_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        orch,
        "step_launch_delta",
        lambda *_args: events.append("launch"),
    )
    monkeypatch.setattr(orch, "step_poll_delta", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        orch,
        "step_transfer_out_delta",
        lambda *_args: events.append("transfer") or {"ok": True},
    )
    monkeypatch.setattr(orch.O, "ssh_ok", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        orch,
        "_cleanup_delta",
        lambda *, strict: events.append(f"cleanup:{strict}"),
    )

    assert orch.run_paid_workflow(
        ctx, {}, gate, "passphrase", "pubkey", apply=True
    ) == {"ok": True}
    assert events == [
        "before setup",
        "provider-attestation",
        "hardware-attestation",
        "after setup",
        "before launch",
        "launch",
        "after launch",
        "before output transfer",
        "transfer",
        "after output transfer",
        "cleanup:True",
    ]


def test_terminate_requires_confirmed_absence(monkeypatch, tmp_path):
    states = iter(["RUNNING", "TERMINATED"])
    mutations = []

    def fake_gql(query, variables=None):
        if "podTerminate" in query:
            mutations.append(variables["id"])
            return {"podTerminate": True}
        return {"pod": {"id": "pod-1", "desiredStatus": next(states)}}

    monkeypatch.setattr(orch.O, "gql", fake_gql)
    monkeypatch.setattr(orch.O, "WORKDIR", tmp_path)
    monkeypatch.setattr(orch.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(orch.O, "log", lambda _message: None)
    orch._created_pod_ids.clear()  # noqa: SLF001
    orch._created_pod_ids.add("pod-1")  # noqa: SLF001

    orch.terminate_confirmed("pod-1")

    assert mutations == ["pod-1", "pod-1"]
    assert not orch._created_pod_ids  # noqa: SLF001


def test_non_strict_cleanup_never_forgets_uncertain_deploy_name(monkeypatch):
    orch._created_pod_name = "georgian-legal-delta-uncertain"  # noqa: SLF001
    monkeypatch.setattr(orch, "_named_active_pods", lambda _name: [])

    orch._cleanup_delta(strict=False)  # noqa: SLF001

    assert orch._created_pod_name == "georgian-legal-delta-uncertain"  # noqa: SLF001


def test_strict_cleanup_requires_sustained_successful_absence(monkeypatch):
    orch._created_pod_name = "georgian-legal-delta-uncertain"  # noqa: SLF001
    queries = []

    def no_match(name):
        queries.append(name)
        return []

    monkeypatch.setattr(orch, "_named_active_pods", no_match)
    monkeypatch.setattr(orch.time, "sleep", lambda _seconds: None)

    orch._cleanup_delta(strict=True)  # noqa: SLF001

    assert len(queries) == orch.UNCERTAIN_DEPLOY_RECONCILE_ATTEMPTS
    assert orch._created_pod_name is None  # noqa: SLF001


def test_strict_cleanup_does_not_count_account_errors_as_absence(monkeypatch):
    orch._created_pod_name = "georgian-legal-delta-uncertain"  # noqa: SLF001
    monkeypatch.setattr(
        orch,
        "_named_active_pods",
        lambda _name: (_ for _ in ()).throw(RuntimeError("account unavailable")),
    )
    monkeypatch.setattr(orch.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="could not confirm sustained absence"):
        orch._cleanup_delta(strict=True)  # noqa: SLF001

    assert orch._created_pod_name == "georgian-legal-delta-uncertain"  # noqa: SLF001


def test_signal_handler_delegates_cleanup_to_paid_finally(monkeypatch):
    cleanup_calls = []
    monkeypatch.setattr(
        orch, "_cleanup_delta", lambda *, strict: cleanup_calls.append(strict)
    )

    with pytest.raises(SystemExit) as exc:
        orch._signal_cleanup(orch.signal.SIGTERM, None)  # noqa: SLF001

    assert exc.value.code == 128 + int(orch.signal.SIGTERM)
    assert cleanup_calls == []


def test_budget_error_is_never_retried():
    calls = []

    def exhausted():
        calls.append("attempt")
        raise orch.ReserveBudgetError("stop billing")

    with pytest.raises(orch.ReserveBudgetError, match="stop billing"):
        orch._retry(exhausted, attempts=6, delay=0, what="paid transfer")  # noqa: SLF001

    assert calls == ["attempt"]


def test_paid_watchdog_arms_and_raises_non_retryable_budget_error(monkeypatch):
    armed = []
    handlers = {}

    monkeypatch.setattr(orch.signal, "getsignal", lambda _signum: "previous")
    monkeypatch.setattr(orch.signal, "getitimer", lambda _which: (0.0, 0.0))
    monkeypatch.setattr(
        orch.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(
        orch.signal,
        "setitimer",
        lambda which, seconds, interval=0.0: armed.append((which, seconds, interval)),
    )

    with pytest.raises(orch.ReserveBudgetError, match="paid-work deadline"):
        with orch.paid_budget_watchdog(price=1.0, max_compute_cost=0.5):
            handlers[orch.signal.SIGALRM](orch.signal.SIGALRM, None)

    assert armed[0][1] == pytest.approx(1800.0)
    assert armed[-1][1] == 0
