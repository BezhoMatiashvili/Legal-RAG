#!/usr/bin/env python3
"""Run one validated, source-aware delta embed on one Secure Cloud GPU pod.

The workflow is deliberately fail-closed: explicit raw items are normalized/chunked on
the host first, a run-scoped collection is derived from ``source`` + ``run_id``, the pod
must pass the CPU/GPU checksum gate *before* embedding, and the downloaded snapshot is
accepted only when its SHA-256, document identities, chunks, UUIDv5 point ids, and exact
point count match the host manifest.  The paid pod is terminated and confirmed absent
before the local restore starts.

GPU selection (2026-07-14): tries ``GPU_PRIMARY`` (RTX 4090) first, then each
``GPU_FALLBACKS`` candidate in order if the primary has no capacity — user-authorized
override of the prior single-GPU-only policy after repeated zero-cost "no capacity"
failures. The G2 CPU/GPU cosine-similarity gate (``O.COS_GATE``) validates numerical
correctness regardless of which candidate is used; budget/reserve math still gates on
``GPU_PRIMARY``'s price as a conservative ceiling.

Supreme Court example (from ``ingest/``)::

    .venv/bin/python scripts/runpod_orchestrate_delta.py \
        --source supremecourt --run-id 20260713T120000Z \
        --items ../artifacts/supremecourt/runs/<run>/items.jsonl

The legacy Matsne selection remains available with ``--runs-since``.  Merge into the
main collection remains a separate operator step.
"""

from __future__ import annotations

import argparse
import atexit
from collections import Counter
from contextlib import contextmanager
import dataclasses
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import stat
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_orchestrate as O  # noqa: E402 - shared GraphQL/SSH/transfer helpers
from ingest.operational import (  # noqa: E402
    QDRANT_RECREATE_APPROVAL_ENV,
    QDRANT_WRITE_APPROVAL_ENV,
    RUNPOD_EPHEMERAL_QDRANT_ENV,
    RUNPOD_SPEND_APPROVAL_ENV,
    require_explicit_approval,
)

EMBED_SH = Path(__file__).resolve().parent / "runpod_embed_delta.sh"
POLL_DEADLINE_S = 4 * 3600
POLL_S = 30
DEAD_CHECKS = 6
RESERVE_USD = 2.0
BILLING_CLEANUP_MARGIN_S = 35 * 60
UNCERTAIN_DEPLOY_RECONCILE_ATTEMPTS = 12
CONFIRMED_ABSENCE_CHECKS = 3
RECONCILE_INTERVAL_S = 5
# Primary GPU, tried first; fallbacks tried in order only after the primary is exhausted
# (2026-07-14, user-authorized override of the prior single-GPU-only policy after 6
# consecutive zero-cost "no capacity" failures on RTX 4090 Secure Cloud spanning ~3h real
# time — same candidate list already used and tested in runpod_orchestrate.py). Budget/
# reserve math still gates on GPU_PRIMARY's price as a conservative ceiling: fallbacks are
# typically cheaper, and attest_provider_pod already enforces actual cost <= gate.price
# regardless of which candidate is used, so this never underestimates spend risk.
GPU_PRIMARY = "NVIDIA GeForce RTX 4090"
GPU_FALLBACKS = ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
                 "NVIDIA RTX 4000 Ada Generation", "NVIDIA L4"]
ALL_CANDIDATE_GPUS = (GPU_PRIMARY, *GPU_FALLBACKS)
GPU = GPU_PRIMARY  # backward-compat alias (existing tests reference orch.GPU directly)
COLLECTION_PREFIX = "georgian_legal_delta"
POD_PREFIX = "georgian-legal-delta"
RUNPOD_SPEND_LOCK = O.INGEST / ".state" / "runpod-spend.lock"
_ACTIVE_FINAL_STATES = {"TERMINATED", "EXITED"}

_created_pod_ids: set[str] = set()
_created_pod_name: str | None = None
_provisioned_gpu: str | None = None  # which candidate actually got a pod, set by step_provision_4090


class ReserveBudgetError(RuntimeError):
    """The paid workflow reached its fail-safe compute deadline."""


@dataclasses.dataclass(frozen=True)
class DeltaRun:
    source: str
    run_id: str
    collection: str
    stage_root: Path
    out: Path
    payload: Path
    pod_name: str

    @property
    def input_manifest(self) -> Path:
        return self.stage_root / "delta_input_manifest.json"

    @property
    def snapshot(self) -> Path:
        return self.out / f"{self.collection}.snapshot"


@dataclasses.dataclass(frozen=True)
class SpendGate:
    balance: float
    price: float
    stock: str
    estimated_hours: float
    estimated_cost: float
    max_compute_cost: float


@contextmanager
def runpod_spend_lock(path: Path | None = None):
    """Exclude concurrent local RunPod spend workflows from gate through cleanup."""
    lock_path = Path(path or RUNPOD_SPEND_LOCK)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode) or stat.S_IMODE(current.st_mode) & 0o077:
            raise PermissionError(
                f"RunPod spend lock must be an owner-only regular file: {lock_path}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another local RunPod spend workflow holds {lock_path}"
            ) from exc
        O.log(f"acquired exclusive RunPod spend lock {lock_path}")
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _safe_component(value: str, *, max_length: int = 72) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_-")
    safe = re.sub(r"_+", "_", safe)[:max_length].rstrip("_-")
    if not safe:
        raise ValueError(f"unsafe/empty run component: {value!r}")
    return safe


def make_run(source: str, run_id: str, *, workdir: Path | None = None) -> DeltaRun:
    source_safe = _safe_component(source.lower(), max_length=32)
    run_safe = _safe_component(run_id)
    root = Path(workdir or O.WORKDIR)
    collection = f"{COLLECTION_PREFIX}_{source_safe}_{run_safe}"
    suffix = f"{source_safe}_{run_safe}"
    pod_name = f"{POD_PREFIX}-{source_safe}-{run_safe}"[:63].rstrip("-")
    return DeltaRun(
        source=source,
        run_id=run_safe,
        collection=collection,
        stage_root=root / f"delta_stage_{suffix}",
        out=root / f"out_delta_{suffix}",
        payload=root / f"payload_delta_{suffix}.tar.gz.enc",
        pod_name=pod_name,
    )


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _retry(fn, *, attempts: int = 6, delay: int = 20, what: str = ""):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except ReserveBudgetError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport errors have several types
            if attempt == attempts:
                raise
            O.log(
                f"{what}: attempt {attempt}/{attempts} failed "
                f"({str(exc)[-160:]}); retrying in {delay}s"
            )
            time.sleep(delay)


def stage_delta_items(
    runs_dir: Path,
    since: str | None,
    stage_dir: Path,
    explicit: list[Path] | None = None,
) -> list[Path]:
    """Copy selected items in deterministic last-run-wins order.

    The legacy filenames are retained when unique, preserving existing Matsne behavior.
    A numeric prefix is used only when two explicit paths share a parent name.
    """
    if explicit:
        paths = [Path(path) for path in explicit]
        invalid = [
            str(path)
            for path in paths
            if not path.is_file() or path.stat().st_size == 0
        ]
        if invalid:
            raise ValueError(f"explicit delta input file(s) missing/empty: {invalid}")
    else:
        paths = sorted(Path(runs_dir).glob("*/items.jsonl"))
        if since:
            paths = [path for path in paths if path.parent.name >= since]
        paths = [path for path in paths if path.exists() and path.stat().st_size > 0]
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    ordered = sorted(paths, key=lambda item: item.parent.name)
    parent_counts = Counter(path.parent.name for path in ordered)
    staged: list[Path] = []
    for index, path in enumerate(ordered):
        name = f"{path.parent.name}.jsonl"
        if parent_counts[path.parent.name] > 1:
            name = f"{index:04d}_{path.parent.name}.jsonl"
        destination = stage_dir / name
        shutil.copyfile(path, destination)
        staged.append(destination)
    return staged


def build_input_manifest(ctx: DeltaRun, staged: list[Path]) -> dict:
    """Run the exact local tokenizer/chunker dry-run and load its strict manifest."""
    command = [
        "env",
        "RERANK_ENABLED=false",
        str(O.INGEST / ".venv/bin/python"),
        str(O.INGEST / "scripts/embed_delta.py"),
        "--source",
        ctx.source,
        "--items",
        *[str(path) for path in staged],
        "--collection",
        ctx.collection,
        "--dry-run",
        "--strict",
        "--manifest-out",
        str(ctx.input_manifest),
    ]
    O.run(command, timeout=7200)
    manifest = json.loads(ctx.input_manifest.read_text(encoding="utf-8"))
    if manifest.get("source") != ctx.source:
        raise RuntimeError(
            f"input manifest source {manifest.get('source')!r} != {ctx.source!r}"
        )
    if manifest.get("documents", 0) <= 0 or manifest.get("skipped") != 0:
        raise RuntimeError(f"invalid strict input manifest: {manifest}")
    return manifest


def estimate_gpu_hours(chunks: int) -> float:
    """Conservative 4090 estimate including dependency/model/snapshot overhead."""
    if chunks <= 0:
        raise ValueError("chunk estimate must be positive")
    return max(1.0, 0.75 + chunks / 120_000)


def preflight_local_delta_restore(ctx: DeltaRun) -> None:
    """Prove the eventual restore target is reachable, local, and still absent."""
    from urllib.parse import urlparse

    from ingest.config import load_config
    from ingest.qdrant_store import make_client

    cfg = load_config()
    parsed = urlparse(cfg.qdrant_url)
    if (parsed.hostname or "") not in {"", "localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(
            f"delta snapshots must restore to local Qdrant, not {cfg.qdrant_url!r}"
        )
    client = make_client(cfg)
    if client.collection_exists(ctx.collection):
        raise RuntimeError(
            f"refusing paid work because staging collection {ctx.collection!r} "
            "already exists; use a new run id"
        )
    O.log(
        f"local Qdrant preflight PASS: reachable at {cfg.qdrant_url}, "
        f"target {ctx.collection!r} is absent"
    )


def preflight_pod_delta_write(
    ctx: DeltaRun, *, environ: dict[str, str] | None = None
) -> None:
    """Exercise the exact pod-local staging guard before any paid provisioning."""
    from ingest.config import ConfigurationError, load_config
    from ingest.qdrant_store import validate_generation_write_target

    cfg = dataclasses.replace(
        load_config(), collection_name=ctx.collection, generation_id=None
    )
    pod_environment = dict(os.environ if environ is None else environ)
    if pod_environment.get(QDRANT_WRITE_APPROVAL_ENV) != "1":
        raise ConfigurationError(
            f"pod-local delta staging requires explicit {QDRANT_WRITE_APPROVAL_ENV}=1"
        )
    pod_environment[RUNPOD_EPHEMERAL_QDRANT_ENV] = "1"
    validate_generation_write_target(
        cfg,
        apply=True,
        recreate=True,
        allow_run_scoped_delta=True,
        environ=pod_environment,
    )
    O.log(
        "pod-local Qdrant write preflight PASS: unversioned run-scoped staging, "
        "ephemeral Qdrant attested, recreation explicitly approved"
    )


def _account_state() -> tuple[float, list[dict]]:
    data = O.gql(
        "query{ myself{ clientBalance pods{ id name desiredStatus costPerHr } } }"
    )
    myself = data.get("myself") or {}
    balance = myself.get("clientBalance")
    if not isinstance(balance, (int, float)):
        raise RuntimeError(f"RunPod returned invalid clientBalance={balance!r}")
    active = [
        pod
        for pod in myself.get("pods") or []
        if str(pod.get("desiredStatus") or "").upper() not in _ACTIVE_FINAL_STATES
    ]
    return float(balance), active


def runpod_spend_gate(chunks: int) -> SpendGate:
    """Gate paid work on live balance, no active pods, secure 4090 stock, and reserve."""
    balance, active = _account_state()
    if active:
        summary = [
            (pod.get("id"), pod.get("name"), pod.get("desiredStatus")) for pod in active
        ]
        raise RuntimeError(
            f"RunPod already has active pod(s); refusing a second pod: {summary}"
        )
    price, stock = O.gpu_price(
        GPU_PRIMARY
    )  # secureCloud=true inside the browser-UA GraphQL helper; conservative price ceiling
    # for ALL candidates (fallbacks are typically cheaper) — see GPU_PRIMARY comment above.
    if not isinstance(price, (int, float)) or float(price) <= 0:
        raise RuntimeError("no live Secure Cloud RTX 4090 price/capacity")
    if str(stock or "").strip().lower() not in {"low", "medium", "high"}:
        raise RuntimeError(f"Secure Cloud RTX 4090 stock is unavailable: {stock!r}")
    hours = estimate_gpu_hours(chunks)
    estimated = hours * float(price)
    cleanup_margin_cost = BILLING_CLEANUP_MARGIN_S / 3600 * float(price)
    required = estimated + cleanup_margin_cost + RESERVE_USD
    if required > balance:
        raise RuntimeError(
            f"RunPod balance ${balance:.2f} cannot cover conservative cost "
            f"${estimated:.2f}, ${cleanup_margin_cost:.2f} cleanup margin, "
            f"and ${RESERVE_USD:.2f} reserve"
        )
    gate = SpendGate(
        balance=balance,
        price=float(price),
        stock=str(stock),
        estimated_hours=hours,
        estimated_cost=estimated,
        max_compute_cost=balance - RESERVE_USD - cleanup_margin_cost,
    )
    O.log(
        f"RunPod gate PASS: balance=${balance:.2f}, active_pods=0, {GPU_PRIMARY} Secure "
        f"stock={stock}, price=${float(price):.2f}/hr, conservative={hours:.2f}h/"
        f"${estimated:.2f}, cleanup_margin={BILLING_CLEANUP_MARGIN_S / 60:.0f}min/"
        f"${cleanup_margin_cost:.2f}, reserve=${RESERVE_USD:.2f}"
    )
    return gate


def enforce_reserve_budget(
    *,
    price: float,
    provisioned_at: float,
    max_compute_cost: float,
    phase: str,
) -> float:
    """Fail once wall-clock compute at the gated price would consume the reserve."""
    now = time.time()
    cost = max(0.0, now - provisioned_at) / 3600 * price
    if cost >= max_compute_cost:
        raise ReserveBudgetError(
            f"RunPod compute reached ${cost:.2f} during {phase}; "
            f"preserving ${RESERVE_USD:.2f} reserve"
        )
    O.log(
        f"RunPod wall-clock budget checkpoint {phase}: ${cost:.3f} spent-equivalent, "
        f"${max_compute_cost - cost:.3f} before reserve"
    )
    return cost


def _budget_watchdog_seconds(*, price: float, max_compute_cost: float) -> float:
    if price <= 0 or max_compute_cost <= 0:
        raise ReserveBudgetError("RunPod compute budget is not positive")
    seconds = max_compute_cost / price * 3600
    if seconds <= 0:
        raise ReserveBudgetError("RunPod compute budget is already exhausted")
    return seconds


@contextmanager
def paid_budget_watchdog(*, price: float, max_compute_cost: float):
    """Interrupt any blocking paid operation before cleanup or the $2 reserve is spent."""
    seconds = _budget_watchdog_seconds(
        price=price, max_compute_cost=max_compute_cost
    )
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise RuntimeError("paid RunPod workflow requires a POSIX billing watchdog")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0:
        raise RuntimeError(
            "refusing paid workflow while another process alarm/watchdog is active"
        )

    def _budget_alarm(_signum, _frame) -> None:  # noqa: ANN001
        raise ReserveBudgetError(
            "RunPod paid-work deadline reached; terminating the created pod before "
            "the cleanup margin or $2 reserve is consumed"
        )

    signal.signal(signal.SIGALRM, _budget_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def _cleanup_signal_shield():
    """Do not let a second operator signal interrupt strict billing cleanup."""
    previous: dict[int, object] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, signal.SIG_IGN)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _named_active_pods(name: str) -> list[dict]:
    _, pods = _account_state()
    return [pod for pod in pods if pod.get("name") == name]


def _secure_cloud_state(pod_id: str) -> bool | None:
    """Return provider secure-cloud state, or None when that field is unsupported."""
    try:
        data = O.gql(
            "query($id:String!){ pod(input:{podId:$id}){ machine{ secureCloud } } }",
            {"id": pod_id},
        )
    except RuntimeError as exc:
        message = str(exc)
        unsupported = "secureCloud" in message and any(
            marker in message
            for marker in ("Cannot query field", "Unknown field", "not defined")
        )
        if unsupported:
            O.log(
                "RunPod API does not expose machine.secureCloud; "
                "retaining the SECURE deploy-request attestation"
            )
            return None
        raise
    pod = data.get("pod") or {}
    machine = pod.get("machine")
    if not isinstance(machine, dict) or not isinstance(
        machine.get("secureCloud"), bool
    ):
        raise RuntimeError(
            f"RunPod exposed secureCloud without an attestable boolean for pod {pod_id}"
        )
    return machine["secureCloud"]


def attest_provider_pod(ctx: DeltaRun, pod_id: str, gate: SpendGate) -> float:
    """Recheck account exclusivity, billed price, name, and exposed cloud security."""
    _, active = _account_state()
    matches = [pod for pod in active if str(pod.get("id") or "") == pod_id]
    if len(active) != 1 or len(matches) != 1:
        summary = [
            (pod.get("id"), pod.get("name"), pod.get("desiredStatus")) for pod in active
        ]
        raise RuntimeError(
            f"expected exactly the created RunPod pod after provision; active={summary}"
        )
    pod = matches[0]
    if pod.get("name") != ctx.pod_name:
        raise RuntimeError(
            f"created pod name mismatch: expected {ctx.pod_name!r}, "
            f"got {pod.get('name')!r}"
        )
    cost = pod.get("costPerHr")
    if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost <= 0:
        raise RuntimeError(f"created pod has unavailable costPerHr: {cost!r}")
    if float(cost) > gate.price:
        raise RuntimeError(
            f"created pod costPerHr ${float(cost):.4f} exceeds gated "
            f"price ${gate.price:.4f}"
        )
    secure = _secure_cloud_state(pod_id)
    if secure is False:
        raise RuntimeError(
            "created pod provider attestation says Secure Cloud is false"
        )
    O.log(
        f"provider pod attestation PASS: id={pod_id}, name={ctx.pod_name}, "
        f"cost=${float(cost):.4f}/hr, secure={secure if secure is not None else 'requested'}"
    )
    return float(cost)


def step_provision_4090(pubkey: str, ctx: DeltaRun, price: float) -> str:
    """Provision exactly one Secure Cloud GPU pod: try GPU_PRIMARY, then each
    GPU_FALLBACKS candidate in order (2026-07-14, user-authorized override of the prior
    single-GPU-only policy — see the GPU_PRIMARY constant comment for why). A candidate is
    skipped if its live price exceeds the gated ceiling `price`, so this never provisions
    above what was budgeted. Records which candidate actually got a pod in the
    module-level `_provisioned_gpu` global so attestation/manifest checks verify against
    the real GPU used, not an assumption."""
    global _created_pod_name, _provisioned_gpu
    _created_pod_name = ctx.pod_name
    mutation = (
        "mutation($input:PodFindAndDeployOnDemandInput!){ "
        "podFindAndDeployOnDemand(input:$input){ id imageName machineId } }"
    )
    for gpu_id in ALL_CANDIDATE_GPUS:
        if gpu_id != GPU_PRIMARY:
            live_price, live_stock = O.gpu_price(gpu_id)
            if live_price is None or float(live_price) > price:
                O.log(f"{gpu_id}: no price within gated ceiling ${price:.2f}/hr — skipping")
                continue
            O.log(f"{gpu_id}: ${live_price}/hr stock={live_stock}")
        variables = {
            "input": {
                "cloudType": "SECURE",
                "gpuCount": 1,
                "gpuTypeId": gpu_id,
                "minMemoryInGb": 20,
                "minVcpuCount": 4,
                "name": ctx.pod_name,
                "imageName": O.IMAGE,
                "dockerArgs": "",
                "ports": "22/tcp",
                "volumeInGb": 80,
                "containerDiskInGb": 60,
                "volumeMountPath": "/workspace",
                "supportPublicIp": True,
                "startSsh": True,
                "env": [{"key": "PUBLIC_KEY", "value": pubkey}],
            }
        }
        for attempt in range(1, 4):
            try:
                data = O.gql(mutation, variables)
            except RuntimeError as exc:
                # A clean GraphQL error response (e.g. RunPod's "does not have the
                # resources to deploy" capacity message) — the request definitely reached
                # RunPod and got a definitive no; no pod was created, no reconciliation
                # needed, safe to retry/fall back like a null "no capacity" response.
                O.log(f"Secure 1x {gpu_id} deploy attempt {attempt}/3 error: {exc}")
                time.sleep(20)
                continue
            except (
                Exception
            ):  # a lost response may still have created the uniquely named pod
                matches = []
                for _ in range(3):
                    matches = _named_active_pods(ctx.pod_name)
                    for match in matches:
                        if match.get("id"):
                            _created_pod_ids.add(match["id"])
                    if matches:
                        break
                    time.sleep(5)
                if len(matches) == 1:
                    pod_id = str(matches[0]["id"])
                    (O.WORKDIR / "pod.id").write_text(pod_id, encoding="utf-8")
                    _provisioned_gpu = gpu_id
                    O.log(f"adopted pod {pod_id} on {gpu_id} after lost deploy response")
                    return pod_id
                raise
            pod = data.get("podFindAndDeployOnDemand")
            if pod and pod.get("id"):
                pod_id = str(pod["id"])
                _created_pod_ids.add(pod_id)
                (O.WORKDIR / "pod.id").write_text(pod_id, encoding="utf-8")
                _provisioned_gpu = gpu_id
                O.log(f"provisioned Secure 1x {gpu_id} pod {pod_id} (${price:.2f}/hr)")
                return pod_id
            O.log(f"Secure 1x {gpu_id} deploy attempt {attempt}/3 returned no capacity")
            time.sleep(20)
        O.log(f"{gpu_id}: exhausted 3 attempts, trying next candidate")
    raise RuntimeError(f"could not provision any GPU (tried {list(ALL_CANDIDATE_GPUS)})")


def attest_single_4090(ip: str, port: int) -> None:
    """Verify the pod's hardware matches the candidate step_provision_4090 actually used
    (`_provisioned_gpu`, set as a side effect) — falls back to GPU_PRIMARY if that global
    is somehow unset, preserving the original fail-closed default."""
    expected = _provisioned_gpu or GPU_PRIMARY
    names = [
        line.strip()
        for line in O.ssh_capture(
            ip, port, "nvidia-smi --query-gpu=name --format=csv,noheader", timeout=60
        ).splitlines()
        if line.strip()
    ]
    if names != [expected]:
        raise RuntimeError(
            f"GPU attestation failed; expected exactly [{expected!r}], got {names!r}"
        )
    O.log(f"GPU attestation PASS: exactly one {expected}")


def _pod_gone(pod_id: str) -> bool:
    data = O.gql(
        "query($id:String!){ pod(input:{podId:$id}){ id desiredStatus } }",
        {"id": pod_id},
    )
    pod = data.get("pod")
    return (
        not pod or str(pod.get("desiredStatus") or "").upper() in _ACTIVE_FINAL_STATES
    )


def terminate_confirmed(pod_id: str) -> None:
    """Terminate one created pod and require RunPod to confirm it is gone."""
    last_error: Exception | None = None
    for attempt in range(1, 7):
        try:
            O.gql(
                "mutation($id:String!){ podTerminate(input:{podId:$id}) }",
                {"id": pod_id},
            )
            if _pod_gone(pod_id):
                _created_pod_ids.discard(pod_id)
                O._terminated = True
                (O.WORKDIR / "pod.id").unlink(missing_ok=True)
                O.log(f"pod {pod_id} termination confirmed")
                return
        except Exception as exc:  # noqa: BLE001 - keep the billing backstop armed
            last_error = exc
        O.log(f"pod {pod_id} termination not confirmed ({attempt}/6); retrying")
        time.sleep(min(attempt * 5, 20))
    raise RuntimeError(
        f"pod {pod_id} may still be billing; termination unconfirmed: {last_error}"
    )


def _cleanup_delta(*, strict: bool) -> None:
    global _created_pod_name
    errors: list[str] = []
    name = _created_pod_name
    name_absence_confirmed = name is None
    if name:
        # A deploy response can be lost before RunPod exposes the new pod in account state.
        # When the outcome is uncertain, require a full run of successful empty account
        # observations; API errors never count as absence. Once a pod is observed/known,
        # terminate every matching id and then require consecutive successful empty reads.
        outcome_uncertain = not _created_pod_ids
        required_empty = (
            UNCERTAIN_DEPLOY_RECONCILE_ATTEMPTS
            if strict and outcome_uncertain
            else CONFIRMED_ABSENCE_CHECKS
        )
        max_queries = required_empty + (6 if strict else 0)
        empty_checks = 0
        last_reconcile_error: Exception | None = None
        for reconcile_attempt in range(max_queries if strict else 1):
            try:
                matches = _named_active_pods(name)
                last_reconcile_error = None
                for pod in matches:
                    if pod.get("id"):
                        _created_pod_ids.add(str(pod["id"]))
                if matches:
                    outcome_uncertain = False
                    required_empty = CONFIRMED_ABSENCE_CHECKS
                    empty_checks = 0
                else:
                    empty_checks += 1
            except Exception as exc:  # noqa: BLE001 - strict main cleanup reports this
                last_reconcile_error = exc
                empty_checks = 0

            for pod_id in list(_created_pod_ids):
                try:
                    terminate_confirmed(pod_id)
                except Exception as exc:  # noqa: BLE001 - attempt every known created pod
                    errors.append(str(exc))

            if strict and not _created_pod_ids and empty_checks >= required_empty:
                name_absence_confirmed = True
                break
            if not strict:
                break
            if reconcile_attempt + 1 < max_queries:
                time.sleep(RECONCILE_INTERVAL_S)

        if strict and not name_absence_confirmed:
            detail = (
                f"; last account error: {last_reconcile_error}"
                if last_reconcile_error is not None
                else ""
            )
            errors.append(
                f"could not confirm sustained absence of created pod name {name!r}"
                f"{detail}"
            )
    else:
        for pod_id in list(_created_pod_ids):
            try:
                terminate_confirmed(pod_id)
            except Exception as exc:  # noqa: BLE001 - attempt every known created pod
                errors.append(str(exc))

    if errors:
        message = "RunPod cleanup failed: " + "; ".join(errors)
        if strict:
            raise RuntimeError(message)
        O.log("WARNING: " + message)
    elif strict and name_absence_confirmed and not _created_pod_ids:
        _created_pod_name = None


def _atexit_cleanup() -> None:
    _cleanup_delta(strict=False)


def _signal_cleanup(signum, _frame) -> None:  # noqa: ANN001
    O.log(
        f"signal {signum} received; strict pod cleanup is delegated to the paid "
        "workflow's finally block"
    )
    raise SystemExit(128 + int(signum))


def step_package_delta(ctx: DeltaRun) -> str:
    """Encrypt code, exact inputs, host manifest, and CPU checksum reference."""
    import os
    import secrets

    passphrase = secrets.token_urlsafe(36)
    os.environ["PASSPHRASE"] = passphrase
    plain = ctx.payload.with_suffix("")
    ctx.payload.parent.mkdir(parents=True, exist_ok=True)
    O.log("packaging source-aware delta payload...")
    O.run(
        [
            "tar",
            "czf",
            str(plain),
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=*.pyo",
            "--exclude=.pytest_cache",
            "--exclude=.ruff_cache",
            "-C",
            str(O.REPO),
            "ingest/ingest",
            "ingest/pyproject.toml",
            "ingest/scripts",
            "ingest/snapshots/v1/checksum_cpu.json",
            "-C",
            str(ctx.stage_root),
            "delta_items",
            "delta_input_manifest.json",
        ],
        timeout=1800,
    )
    listing = O.run(["tar", "tzf", str(plain)]).stdout.decode()
    leaked = [
        line
        for line in listing.splitlines()
        if "/.env" in line or ".venv" in line or "/.state" in line
    ]
    if leaked:
        raise RuntimeError(f"payload would leak secrets/state: {leaked[:5]}")
    O.run(
        [
            "openssl",
            "enc",
            "-aes-256-cbc",
            "-pbkdf2",
            "-pass",
            "env:PASSPHRASE",
            "-in",
            str(plain),
            "-out",
            str(ctx.payload),
        ],
        timeout=600,
    )
    plain.unlink()
    O.log(
        f"delta payload encrypted: {ctx.payload.name} ({ctx.payload.stat().st_size / 1e6:.1f} MB)"
    )
    return passphrase


def step_launch_delta(ip: str, port: int, passphrase: str, ctx: DeltaRun) -> None:
    O.push_content(
        ip,
        port,
        "/dev/shm/p.env",
        "export PASSPHRASE=%s\n" % shlex.quote(passphrase),
        mode="600",
    )
    exports = " ".join(
        [
            f"COLLECTION_NAME={shlex.quote(ctx.collection)}",
            f"SOURCE_NAME={shlex.quote(ctx.source)}",
            f"RUN_ID={shlex.quote(ctx.run_id)}",
            f"QDRANT_VER={shlex.quote(O.QDRANT_VER)}",
            "EMBED_BATCH_SIZE=256",
            "RERANK_ENABLED=false",
            f"{QDRANT_WRITE_APPROVAL_ENV}=1",
            f"{QDRANT_RECREATE_APPROVAL_ENV}=1",
            f"{RUNPOD_EPHEMERAL_QDRANT_ENV}=1",
            f"EXPECTED_GPU={shlex.quote(_provisioned_gpu or GPU_PRIMARY)}",
            "WORK=/workspace",
        ]
    )
    launch = (
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        "mkdir -p /workspace/out\n"
        "cd /workspace\n"
        ". /dev/shm/p.env\n"
        "rm -f /dev/shm/p.env\n"
        f"export {exports}\n"
        "bash /workspace/runpod_embed_delta.sh\n"
        'echo "EXIT=$?" >> /workspace/out/embed.log\n'
    )
    O.push_content(ip, port, "/workspace/launch_delta.sh", launch, mode="755")
    remote = (
        "mkdir -p /workspace/out; "
        "setsid bash /workspace/launch_delta.sh >/workspace/out/launch.out 2>&1 </dev/null & "
        "printf '%s\\n' $! > /workspace/out/launch.pid; "
        "echo LAUNCHED"
    )
    output = O.ssh_capture(ip, port, remote, timeout=60)
    if "LAUNCHED" not in output:
        raise RuntimeError(f"failed to launch delta embed: {output!r}")
    O.log("delta embed launched (setsid, detached)")


def step_poll_delta(
    ip: str,
    port: int,
    *,
    price: float,
    provisioned_at: float,
    max_compute_cost: float,
) -> None:
    deadline = time.time() + POLL_DEADLINE_S
    dead_checks = 0
    while True:
        enforce_reserve_budget(
            price=price,
            provisioned_at=provisioned_at,
            max_compute_cost=max_compute_cost,
            phase="embed polling",
        )
        if time.time() >= deadline:
            raise TimeoutError("delta embed exceeded deadline")
        if O.ssh_ok(ip, port, "test -f /workspace/out/DONE"):
            enforce_reserve_budget(
                price=price,
                provisioned_at=provisioned_at,
                max_compute_cost=max_compute_cost,
                phase="embed completion",
            )
            O.log("DONE marker found")
            return
        tail = O.ssh_capture(
            ip,
            port,
            "tail -n 3 /workspace/out/embed.log /workspace/out/launch.out 2>/dev/null",
        ).strip()
        if tail:
            O.log("  delta: " + tail.replace("\n", " | ")[-300:])
        alive = O.ssh_ok(
            ip,
            port,
            "pid=$(cat /workspace/out/launch.pid 2>/dev/null) "
            "&& case $pid in (*[!0-9]*|'') false;; (*) kill -0 \"$pid\" 2>/dev/null;; esac",
        )
        dead_checks = 0 if alive else dead_checks + 1
        if dead_checks >= DEAD_CHECKS:
            if O.ssh_ok(ip, port, "test -f /workspace/out/DONE"):
                continue
            full = O.ssh_capture(
                ip,
                port,
                "tail -n 60 /workspace/out/launch.out /workspace/out/embed.log 2>/dev/null",
            )
            raise RuntimeError("delta embed ended without DONE:\n" + full)
        time.sleep(POLL_S)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def step_verify_g2_delta(out: Path) -> float:
    from ingest.embed_job import CHECKSUM_SENTENCE, checksum_cosine

    cpu = json.loads(O.CPU_REF.read_text(encoding="utf-8"))
    gpu = json.loads((out / "checksum_gpu.json").read_text(encoding="utf-8"))
    if gpu.get("sentence") != CHECKSUM_SENTENCE:
        raise RuntimeError("GPU checksum sentence mismatch")
    if len(cpu.get("dense") or []) != 1024 or len(gpu.get("dense") or []) != 1024:
        raise RuntimeError("CPU/GPU checksum vectors must both have 1024 dimensions")
    cosine = checksum_cosine(cpu["dense"], gpu["dense"])
    if cosine < O.COS_GATE:
        raise RuntimeError(
            f"VECTOR-SPACE MISMATCH cosine={cosine:.6f} < {O.COS_GATE:.6f}"
        )
    O.log(f"G2 checksum verified again on host: cosine={cosine:.6f}")
    return cosine


def validate_run_manifest(
    ctx: DeltaRun,
    expected: dict,
    run_manifest: dict,
    snapshot: Path,
) -> None:
    checks = {
        "source": ctx.source,
        "run_id": ctx.run_id,
        "collection": ctx.collection,
        "input_sha256": expected["input_sha256"],
        "document_ids": expected["document_ids"],
        "document_ids_sha256": expected["document_ids_sha256"],
        "expected_documents": expected["documents"],
        "documents": expected["documents"],
        "expected_chunks": expected["chunks"],
        "chunks": expected["chunks"],
        "points_count": expected["chunks"],
        "skipped": 0,
        "snapshot": snapshot.name,
    }
    errors = [
        f"{field}: expected {value!r}, got {run_manifest.get(field)!r}"
        for field, value in checks.items()
        if run_manifest.get(field) != value
    ]
    # "gpu" is checked separately: when this process itself provisioned the pod
    # (_provisioned_gpu set), require an exact match to what was live-attested via SSH;
    # otherwise (e.g. --skip-pod validating a prior run's artifacts) accept any of the
    # known candidate GPUs rather than assuming a single fixed one.
    reported_gpu = run_manifest.get("gpu")
    if _provisioned_gpu is not None:
        if reported_gpu != _provisioned_gpu:
            errors.append(f"gpu: expected {_provisioned_gpu!r}, got {reported_gpu!r}")
    elif reported_gpu not in ALL_CANDIDATE_GPUS:
        errors.append(
            f"gpu: {reported_gpu!r} is not one of the allowed GPU types {list(ALL_CANDIDATE_GPUS)!r}"
        )
    expected_ids = expected.get("document_ids")
    if (
        not isinstance(expected_ids, list)
        or not all(isinstance(value, str) and value for value in expected_ids)
        or expected_ids != sorted(set(expected_ids))
    ):
        errors.append("host input document_ids are not sorted and unique")
    else:
        expected_ids_sha256 = hashlib.sha256(
            "\n".join(expected_ids).encode("utf-8")
        ).hexdigest()
        if expected.get("document_ids_sha256") != expected_ids_sha256:
            errors.append("host input document_ids_sha256 is not canonical")
    run_ids = run_manifest.get("document_ids")
    if (
        not isinstance(run_ids, list)
        or not all(isinstance(value, str) and value for value in run_ids)
        or run_ids != sorted(set(run_ids))
    ):
        errors.append("run manifest document_ids are not sorted and unique")
    else:
        run_ids_sha256 = hashlib.sha256("\n".join(run_ids).encode("utf-8")).hexdigest()
        if run_manifest.get("document_ids_sha256") != run_ids_sha256:
            errors.append("run manifest document_ids_sha256 is not canonical")
    actual_size = snapshot.stat().st_size
    if run_manifest.get("snapshot_size_bytes") != actual_size:
        errors.append(
            f"snapshot_size_bytes: expected local {actual_size}, "
            f"got {run_manifest.get('snapshot_size_bytes')!r}"
        )
    actual_sha = _sha256(snapshot)
    if run_manifest.get("snapshot_sha256") != actual_sha:
        errors.append("snapshot_sha256 differs from downloaded snapshot")
    cosine = run_manifest.get("checksum_cosine")
    if not isinstance(cosine, (int, float)) or cosine < O.COS_GATE:
        errors.append(f"checksum_cosine below gate: {cosine!r}")
    if errors:
        raise RuntimeError("invalid delta run manifest: " + "; ".join(errors))


def step_transfer_out_delta(ctx: DeltaRun, ip: str, port: int, expected: dict) -> dict:
    ctx.out.mkdir(parents=True, exist_ok=True)
    required = (
        "checksum_gpu.json",
        "input_manifest_gpu.json",
        "embed_report.json",
        "run_manifest.json",
        "embed.log",
        "qdrant.log",
        "launch.out",
    )
    for name in required:
        _retry(
            lambda name=name: O.pull_file(
                f"/workspace/out/{name}", ctx.out / name, ip, port, timeout=600
            ),
            attempts=4,
            delay=10,
            what=f"{name} pull",
        )
    _retry(
        lambda: O.pull_file(
            f"/workspace/out/{ctx.collection}.snapshot",
            ctx.snapshot,
            ip,
            port,
            timeout=1800,
        ),
        attempts=8,
        delay=30,
        what="snapshot pull",
    )
    if not ctx.snapshot.exists() or ctx.snapshot.stat().st_size < 100_000:
        raise RuntimeError(f"delta snapshot missing/too small at {ctx.snapshot}")
    O.run(["tar", "tf", str(ctx.snapshot)], timeout=600)
    run_manifest = json.loads(
        (ctx.out / "run_manifest.json").read_text(encoding="utf-8")
    )
    from embed_delta import validate_expected_manifest

    gpu_input = json.loads(
        (ctx.out / "input_manifest_gpu.json").read_text(encoding="utf-8")
    )
    embed_report = json.loads(
        (ctx.out / "embed_report.json").read_text(encoding="utf-8")
    )
    validate_expected_manifest(expected, gpu_input)
    validate_expected_manifest(expected, embed_report)
    if embed_report.get("collection") != ctx.collection:
        raise RuntimeError(
            f"embed report collection {embed_report.get('collection')!r} != {ctx.collection!r}"
        )
    validate_run_manifest(ctx, expected, run_manifest, ctx.snapshot)
    step_verify_g2_delta(ctx.out)
    O.log(
        f"downloaded and validated {ctx.snapshot.name} "
        f"({ctx.snapshot.stat().st_size / 1e6:.1f} MB)"
    )
    return run_manifest


def validate_restored_records(records: list, ctx: DeltaRun, expected: dict) -> None:
    """Validate exact source/doc set, UUIDv5 ids, and contiguous chunk indexes."""
    from ingest.qdrant_store import point_id

    expected_docs = set(expected["document_ids"])
    indices: dict[str, set[int]] = {}
    chunk_counts: dict[str, int] = {}
    for record in records:
        payload = record.payload or {}
        source = payload.get("source")
        document_id = payload.get("document_id")
        chunk_index = payload.get("chunk_index")
        chunk_count = payload.get("document_chunk_count")
        key = f"{source}\t{document_id}"
        if source != ctx.source or key not in expected_docs:
            raise RuntimeError(f"unexpected restored delta document {key!r}")
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
            raise RuntimeError(f"invalid chunk_index for {key}: {chunk_index!r}")
        # Deliberately legacy: ``preflight_pod_delta_write`` pins this run-scoped staging
        # collection to generation_id=None. Immutable v3 publishes never use this workflow.
        if str(record.id) != point_id(source, document_id, chunk_index):
            raise RuntimeError(
                f"UUIDv5 point id mismatch for {key} chunk {chunk_index}"
            )
        if not isinstance(chunk_count, int) or chunk_count <= 0:
            raise RuntimeError(
                f"invalid document_chunk_count for {key}: {chunk_count!r}"
            )
        previous = chunk_counts.setdefault(key, chunk_count)
        if previous != chunk_count:
            raise RuntimeError(f"inconsistent document_chunk_count for {key}")
        indices.setdefault(key, set()).add(chunk_index)

    if set(indices) != expected_docs:
        missing = sorted(expected_docs - set(indices))[:10]
        extra = sorted(set(indices) - expected_docs)[:10]
        raise RuntimeError(
            f"restored document identity mismatch: missing={missing}, extra={extra}"
        )
    for key, actual in indices.items():
        wanted = set(range(chunk_counts[key]))
        if actual != wanted:
            raise RuntimeError(f"non-contiguous chunks for {key}: {sorted(actual)}")
    if len(records) != expected["chunks"]:
        raise RuntimeError(
            f"restored records {len(records)} != expected chunks {expected['chunks']}"
        )


def step_restore_delta(
    ctx: DeltaRun,
    expected: dict,
    run_manifest: dict,
    *,
    apply: bool = False,
) -> None:
    require_explicit_approval(
        apply=apply,
        approval_env=QDRANT_WRITE_APPROVAL_ENV,
        operation="run-scoped delta restore",
    )
    from dotenv import dotenv_values
    from ingest.config import load_config
    from ingest.qdrant_store import make_client
    from urllib.parse import urlparse

    cfg = load_config()
    parsed = urlparse(cfg.qdrant_url)
    if (parsed.hostname or "") not in {"", "localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(
            f"delta snapshots must restore to local Qdrant, not {cfg.qdrant_url!r}"
        )
    key = (
        cfg.qdrant_api_key
        or dotenv_values(O.INGEST / ".env").get("QDRANT_API_KEY")
        or ""
    )
    qdrant_url = cfg.qdrant_url.rstrip("/")
    client = make_client(cfg)
    if client.collection_exists(ctx.collection):
        raise RuntimeError(
            f"refusing to delete or overwrite existing staging collection {ctx.collection!r}; "
            "use a new run id"
        )
    O.log(f"restoring snapshot into run-scoped local collection {ctx.collection!r}...")
    O.run(
        [
            "curl",
            "-sf",
            "-X",
            "POST",
            f"{qdrant_url}/collections/{ctx.collection}/snapshots/upload?priority=snapshot",
            "-H",
            f"api-key: {key}",
            "-H",
            "Content-Type: multipart/form-data",
            "-F",
            f"snapshot=@{ctx.snapshot}",
        ],
        timeout=3600,
    )
    info = client.get_collection(ctx.collection)
    if int(info.points_count) != run_manifest["points_count"]:
        raise RuntimeError(
            f"restored point count {info.points_count} != {run_manifest['points_count']}"
        )

    records = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=ctx.collection,
            limit=512,
            with_vectors=False,
            with_payload=True,
            offset=offset,
        )
        records.extend(page)
        if offset is None:
            break
    validate_restored_records(records, ctx, expected)
    O.log(
        f"restored {ctx.collection!r}: {len(expected['document_ids'])} documents / "
        f"{len(records)} points validated; merge remains an explicit next step"
    )


def _source_and_paths(args) -> tuple[str, Path, list[Path] | None]:
    if args.items:
        if not args.source:
            raise SystemExit("explicit --items requires explicit --source")
        return args.source, O.REPO / "artifacts" / args.source / "runs", args.items
    if not args.runs_since:
        raise SystemExit("pass explicit --source + --items, or Matsne --runs-since")
    if args.source not in (None, "matsne"):
        raise SystemExit("--runs-since is the legacy Matsne-only selector")
    return "matsne", O.REPO / "artifacts" / "matsne" / "runs", None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", help="normalization source; required with explicit --items"
    )
    parser.add_argument(
        "--items", nargs="+", type=Path, help="explicit items.jsonl path(s)"
    )
    parser.add_argument("--runs-since", help="legacy Matsne run-id lower bound")
    parser.add_argument("--run-id", help="stable run id; generated when omitted")
    parser.add_argument(
        "--skip-pod",
        action="store_true",
        help="validate/restore already-downloaded artifacts for this source/run-id",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "permit approved external actions; Qdrant writes additionally require "
            f"{QDRANT_WRITE_APPROVAL_ENV}=1, pod-local recreation requires "
            f"{QDRANT_RECREATE_APPROVAL_ENV}=1, and paid runs require "
            f"{RUNPOD_SPEND_APPROVAL_ENV}=1"
        ),
    )
    return parser.parse_args()


def run_paid_workflow(
    ctx: DeltaRun,
    expected: dict,
    gate: SpendGate,
    passphrase: str,
    pubkey: str,
    *,
    apply: bool = False,
) -> dict:
    """Execute paid steps and unconditionally confirm cleanup before returning."""
    require_explicit_approval(
        apply=apply,
        approval_env=RUNPOD_SPEND_APPROVAL_ENV,
        operation="paid RunPod delta workflow",
    )
    preflight_pod_delta_write(ctx)
    O._price = gate.price
    provision_attempted_at = time.time()
    O._provisioned_at = provision_attempted_at
    watchdog = paid_budget_watchdog(
        price=gate.price,
        max_compute_cost=gate.max_compute_cost,
    )
    watchdog.__enter__()
    try:
        O._pod_id = step_provision_4090(pubkey, ctx, gate.price)
        enforce_reserve_budget(
            price=gate.price,
            provisioned_at=provision_attempted_at,
            max_compute_cost=gate.max_compute_cost,
            phase="before setup",
        )
        _retry(
            lambda: attest_provider_pod(ctx, O._pod_id, gate),
            attempts=6,
            delay=5,
            what="provider pod attestation",
        )
        O._ip, O._port = O.step_wait_ssh(O._pod_id)
        attest_single_4090(O._ip, O._port)
        O.ensure_pod_tools(O._ip, O._port)
        _retry(
            lambda: O.push_file(
                ctx.payload,
                "/workspace/payload.tar.gz.enc",
                O._ip,
                O._port,
                timeout=14400,
            ),
            attempts=8,
            delay=30,
            what="payload push",
        )
        _retry(
            lambda: O.push_file(
                EMBED_SH,
                "/workspace/runpod_embed_delta.sh",
                O._ip,
                O._port,
                timeout=120,
            ),
            what="embed script push",
        )
        enforce_reserve_budget(
            price=gate.price,
            provisioned_at=provision_attempted_at,
            max_compute_cost=gate.max_compute_cost,
            phase="after setup",
        )
        enforce_reserve_budget(
            price=gate.price,
            provisioned_at=provision_attempted_at,
            max_compute_cost=gate.max_compute_cost,
            phase="before launch",
        )
        step_launch_delta(O._ip, O._port, passphrase, ctx)
        enforce_reserve_budget(
            price=gate.price,
            provisioned_at=provision_attempted_at,
            max_compute_cost=gate.max_compute_cost,
            phase="after launch",
        )
        step_poll_delta(
            O._ip,
            O._port,
            price=gate.price,
            provisioned_at=provision_attempted_at,
            max_compute_cost=gate.max_compute_cost,
        )
        enforce_reserve_budget(
            price=gate.price,
            provisioned_at=provision_attempted_at,
            max_compute_cost=gate.max_compute_cost,
            phase="before output transfer",
        )
        run_manifest = step_transfer_out_delta(ctx, O._ip, O._port, expected)
        enforce_reserve_budget(
            price=gate.price,
            provisioned_at=provision_attempted_at,
            max_compute_cost=gate.max_compute_cost,
            phase="after output transfer",
        )
        return run_manifest
    finally:
        with _cleanup_signal_shield():
            watchdog.__exit__(None, None, None)
            if O._ip and O._port:
                try:
                    O.ssh_ok(
                        O._ip,
                        O._port,
                        "rm -rf /workspace/* /dev/shm/p.env 2>/dev/null; sync",
                        timeout=60,
                    )
                except Exception:  # noqa: BLE001 - termination is the real cleanup
                    pass
            _cleanup_delta(strict=True)
            hours = max(0.0, time.time() - provision_attempted_at) / 3600
            O.log(
                "COST: wall-clock/conservative "
                f"elapsed={hours * 60:.1f}min price=${gate.price:.2f}/hr "
                f"est=${hours * gate.price:.3f}"
            )


def run_locked_paid_workflow(
    ctx: DeltaRun,
    expected: dict,
    *,
    apply: bool = False,
) -> dict:
    """Hold the repository spend lock from the live gate through strict pod cleanup."""
    preflight_pod_delta_write(ctx)
    with runpod_spend_lock():
        gate = runpod_spend_gate(expected["chunks"])
        passphrase = step_package_delta(ctx)
        pubkey = O.step_keypair()
        # Packaging and key generation can take long enough for account balance, active
        # pods, capacity, or price to change. Re-run the complete live gate immediately
        # before the only function that can provision.
        gate = runpod_spend_gate(expected["chunks"])
        return run_paid_workflow(
            ctx,
            expected,
            gate,
            passphrase,
            pubkey,
            apply=apply,
        )


def main() -> None:
    global _created_pod_name
    args = _parse_args()
    require_explicit_approval(
        apply=args.apply,
        approval_env=QDRANT_WRITE_APPROVAL_ENV,
        operation="run-scoped delta restore",
    )
    if not args.skip_pod:
        require_explicit_approval(
            apply=args.apply,
            approval_env=RUNPOD_SPEND_APPROVAL_ENV,
            operation="paid RunPod delta workflow",
        )
        require_explicit_approval(
            apply=args.apply,
            approval_env=QDRANT_RECREATE_APPROVAL_ENV,
            operation="run-scoped pod-local delta recreation",
        )
    source, runs_dir, explicit = _source_and_paths(args)
    if source == "supremecourt" and not explicit:
        raise SystemExit("Supreme Court delta requires explicit --items")
    if args.skip_pod and not args.run_id:
        raise SystemExit("--skip-pod requires the original --run-id")
    ctx = make_run(source, args.run_id or _new_run_id())

    from ingest.sources import SOURCES

    if source not in SOURCES:
        raise SystemExit(
            f"unknown source {source!r}; expected one of {sorted(SOURCES)}"
        )

    atexit.register(_atexit_cleanup)
    signal.signal(signal.SIGINT, _signal_cleanup)
    signal.signal(signal.SIGTERM, _signal_cleanup)
    O._terminated = False
    O._pod_id = None
    O._ip = None
    O._port = None
    O._provisioned_at = None
    O._price = None
    _created_pod_ids.clear()
    _created_pod_name = None

    staged = stage_delta_items(
        runs_dir,
        args.runs_since,
        ctx.stage_root / "delta_items",
        explicit=explicit,
    )
    if not staged:
        raise SystemExit("no non-empty items.jsonl matched the delta selection")
    O.step_checksum_ref()
    expected = build_input_manifest(ctx, staged)
    O.log(
        f"validated {expected['documents']} {source} documents / {expected['chunks']} chunks; "
        f"collection={ctx.collection}"
    )
    # A remote/missing Qdrant or an already-used run id must fail before a pod can bill.
    # step_restore_delta repeats the target-absence check after the paid phase to close the
    # preflight-to-restore race.
    preflight_local_delta_restore(ctx)

    run_manifest: dict
    if args.skip_pod:
        run_manifest = json.loads(
            (ctx.out / "run_manifest.json").read_text(encoding="utf-8")
        )
        validate_run_manifest(ctx, expected, run_manifest, ctx.snapshot)
        step_verify_g2_delta(ctx.out)
    else:
        # Hold one repository-wide OS lock from the live zero-pod/price gate until the
        # created pod has been termination-confirmed in run_paid_workflow's finally block.
        run_manifest = run_locked_paid_workflow(
            ctx,
            expected,
            apply=args.apply,
        )

    if _created_pod_ids:
        raise RuntimeError(
            f"created RunPod pod(s) still present: {sorted(_created_pod_ids)}"
        )
    step_restore_delta(ctx, expected, run_manifest, apply=args.apply)
    O.log(
        f"DELTA COMPLETE: {ctx.collection} is validated locally; "
        f"merge with --src {ctx.collection}"
    )


def emergency_terminate() -> None:
    global _created_pod_name
    pid_file = O.WORKDIR / "pod.id"
    if pid_file.exists():
        _created_pod_ids.add(pid_file.read_text(encoding="utf-8").strip())
    _created_pod_name = None
    # Reconcile every active run-scoped delta pod; never touch unrelated account pods.
    _, active = _account_state()
    for pod in active:
        if str(pod.get("name") or "").startswith(POD_PREFIX + "-") and pod.get("id"):
            _created_pod_ids.add(str(pod["id"]))
    _cleanup_delta(strict=True)
    O.log("all run-scoped delta pods are confirmed terminated")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "terminate":
        emergency_terminate()
    else:
        main()
