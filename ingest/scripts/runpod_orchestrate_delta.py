#!/usr/bin/env python3
"""Local orchestrator for the INCREMENTAL RunPod GPU delta embed (the sweep's new docs).

The one-time full-corpus embed (runpod_orchestrate.py) shipped the whole clean snapshot and
brought back a 24 GB collection snapshot. This sibling ships ONLY the newly-scraped matsne
run items (delta_items/*.jsonl inside the same encrypted-payload scheme — PII never travels
in the clear), drives ``runpod_embed_delta.sh`` on the pod, and brings back a small
``georgian_legal_delta`` snapshot which it restores into LOCAL Qdrant under that name.

Merging into the main collection stays a separate, explicit step so the delta can be
inspected first:
    .venv/bin/python scripts/merge_delta_collection.py --dry-run
    .venv/bin/python scripts/merge_delta_collection.py --src georgian_legal_delta

Pod-termination guarantees are inherited from runpod_orchestrate: try/finally + atexit +
SIGINT/SIGTERM all call podTerminate, and a poll deadline hard-stops a hung embed.

Usage (from ingest/):
    .venv/bin/python scripts/runpod_orchestrate_delta.py [--runs-since 20260709T080007Z]
        [--items <items.jsonl> ...] [--skip-pod]   # --skip-pod: restore+verify an already-pulled snapshot
    .venv/bin/python scripts/runpod_orchestrate_delta.py terminate
"""
from __future__ import annotations

import argparse
import atexit
import json
import shlex
import shutil
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_orchestrate as O  # noqa: E402 — reuse gql/ssh/provision/terminate helpers

DELTA_COLLECTION = "georgian_legal_delta"
EMBED_SH = Path(__file__).resolve().parent / "runpod_embed_delta.sh"
OUT = O.WORKDIR / "out_delta"
POLL_DEADLINE_S = 90 * 60  # ~35 min embed expected; hard watchdog well above it
POLL_S = 30
DEAD_CHECKS = 6  # consecutive failed liveness probes before declaring the embed dead


def _retry(fn, *, attempts: int = 6, delay: int = 20, what: str = ""):
    """Retry a transfer through transient link drops. rsync runs with --partial
    --append-verify, so each retry RESUMES the file instead of restarting — on a flaky
    uplink (this box is often on a phone hotspot) that turns a fatal mid-push socket
    error into a pause."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - link errors surface as many types
            if i == attempts:
                raise
            O.log(f"{what}: attempt {i}/{attempts} failed ({str(exc)[-160:]}); retrying in {delay}s")
            time.sleep(delay)


def stage_delta_items(runs_dir: Path, since: str | None, stage_dir: Path,
                      explicit: list[Path] | None = None) -> list[Path]:
    """Copy the delta run items into ``stage_dir`` with run-id-sortable names.

    Selection mirrors embed_delta._resolve_paths (every runs/*/items.jsonl with run id >=
    ``since``); ordering matters because embed_delta dedups document_id last-wins, so files
    are named by their run id (ascending = newest last = newest wins).
    """
    if explicit:
        paths = [Path(p) for p in explicit]
    else:
        paths = sorted(runs_dir.glob("*/items.jsonl"))
        if since:
            paths = [p for p in paths if p.parent.name >= since]
    paths = [p for p in paths if p.exists() and p.stat().st_size > 0]
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)
    staged = []
    for p in sorted(paths, key=lambda p: p.parent.name):
        dst = stage_dir / f"{p.parent.name}.jsonl"
        shutil.copyfile(p, dst)
        staged.append(dst)
    return staged


def step_package_delta(stage_root: Path) -> str:
    """Tar {ingest pkg, pyproject, scripts, delta_items/} + encrypt (passphrase in memory only)."""
    import os
    import secrets
    passphrase = secrets.token_urlsafe(36)
    os.environ["PASSPHRASE"] = passphrase
    plain = O.WORKDIR / "payload_delta.tar.gz"
    enc = O.WORKDIR / "payload_delta.tar.gz.enc"
    O.log("packaging delta payload (ingest pkg + scripts + delta_items)...")
    O.run(["tar", "czf", str(plain),
           "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=*.pyo",
           "--exclude=.pytest_cache", "--exclude=.ruff_cache",
           "-C", str(O.REPO), "ingest/ingest", "ingest/pyproject.toml", "ingest/scripts",
           "-C", str(stage_root), "delta_items"],
          timeout=1800)
    listing = O.run(["tar", "tzf", str(plain)]).stdout.decode()
    leaked = [ln for ln in listing.splitlines() if "/.env" in ln or ".venv" in ln or "/.state" in ln]
    if leaked:
        raise RuntimeError(f"payload would leak secrets/state: {leaked[:5]}")
    O.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-pass", "env:PASSPHRASE",
           "-in", str(plain), "-out", str(enc)], timeout=600)
    plain.unlink()
    O.log(f"delta payload encrypted: {enc.name} ({enc.stat().st_size/1e6:.1f} MB)")
    return passphrase


def step_launch_delta(ip: str, port: int, passphrase: str) -> None:
    O.push_content(ip, port, "/dev/shm/p.env",
                   "export PASSPHRASE=%s\n" % shlex.quote(passphrase), mode="600")
    launch = (
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        "mkdir -p /workspace/out\n"
        "cd /workspace\n"
        ". /dev/shm/p.env\n"
        "rm -f /dev/shm/p.env\n"
        f"export COLLECTION_NAME={DELTA_COLLECTION} QDRANT_VER={O.QDRANT_VER} "
        "EMBED_BATCH_SIZE=256 WORK=/workspace\n"
        "bash /workspace/runpod_embed_delta.sh\n"
        'echo "EXIT=$?" >> /workspace/out/embed.log\n'
    )
    O.push_content(ip, port, "/workspace/launch_delta.sh", launch, mode="755")
    remote = ("mkdir -p /workspace/out; "
              "setsid bash /workspace/launch_delta.sh >/workspace/out/launch.out 2>&1 </dev/null & "
              "echo LAUNCHED")
    out = O.ssh_capture(ip, port, remote, timeout=60)
    if "LAUNCHED" not in out:
        raise RuntimeError(f"failed to launch delta embed (setsid): {out!r}")
    O.log("delta embed launched (setsid, detached)")


def step_poll_delta(ip: str, port: int) -> int:
    """Poll for the DONE marker, tolerating transient link drops.

    The embed runs setsid-detached ON the pod, so a local network blip must not be read
    as "the embed died" — only ``DEAD_CHECKS`` consecutive failed liveness probes (each a
    separate ssh) count as death, and death is only declared after a last DONE re-check.
    """
    deadline = time.time() + POLL_DEADLINE_S
    dead_checks = 0
    while time.time() < deadline:
        if O.ssh_ok(ip, port, "test -f /workspace/out/DONE"):
            pts = O.ssh_capture(ip, port, "cat /workspace/out/DONE").strip()
            O.log(f"DONE marker found: points={pts}")
            return int(pts or "0")
        tail = O.ssh_capture(
            ip, port, "tail -n 3 /workspace/out/embed.log /workspace/out/launch.out 2>/dev/null").strip()
        if tail:
            O.log("  delta: " + tail.replace("\n", " | ")[-300:])
        alive = O.ssh_ok(ip, port, "pgrep -f runpod_embed_delta.sh >/dev/null 2>&1 "
                                   "|| pgrep -f 'bash /workspace/launch_delta.sh' >/dev/null 2>&1")
        if alive:
            dead_checks = 0
        else:
            dead_checks += 1
            O.log(f"  liveness probe failed ({dead_checks}/{DEAD_CHECKS}) — "
                  "embed process not seen (may be a link blip)")
            if dead_checks >= DEAD_CHECKS:
                if O.ssh_ok(ip, port, "test -f /workspace/out/DONE"):
                    continue
                full = O.ssh_capture(
                    ip, port,
                    "tail -n 40 /workspace/out/launch.out /workspace/out/embed.log 2>/dev/null")
                raise RuntimeError("delta embed ended without DONE:\n" + full)
        time.sleep(POLL_S)
    raise TimeoutError("delta embed exceeded POLL_DEADLINE")


def step_transfer_out_delta(ip: str, port: int) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    O.log("transferring delta results out (snapshot + checksum + logs)...")
    for name in ("checksum_gpu.json", "embed.log", "qdrant.log"):
        try:
            O.pull_file(f"/workspace/out/{name}", OUT / name, ip, port, timeout=600)
        except RuntimeError as e:
            O.log(f"  (optional {name} pull failed: {e})")
    snap = OUT / f"{DELTA_COLLECTION}.snapshot"
    # 30 min ceiling: a delta snapshot is ~0.2 GB; a stalled link must fail into the retry
    # loop, not hold the billing pod for hours (2026-07-09: a dead pod's proxy stalled a
    # pull for 2 h at timeout=14400 before the retry could fire).
    _retry(lambda: O.pull_file(f"/workspace/out/{DELTA_COLLECTION}.snapshot", snap, ip, port,
                               timeout=1800),
           attempts=8, delay=30, what="snapshot pull")
    if not snap.exists() or snap.stat().st_size < 1_000_000:
        raise RuntimeError(f"delta snapshot missing/too small at {snap}")
    O.log(f"got delta snapshot {snap.name} ({snap.stat().st_size/1e9:.2f} GB)")
    return snap


def step_verify_g2_delta() -> float:
    from ingest.embed_job import checksum_cosine
    cpu = json.loads(O.CPU_REF.read_text())
    gpu = json.loads((OUT / "checksum_gpu.json").read_text())
    cos = checksum_cosine(cpu["dense"], gpu["dense"])
    O.log(f"G2 cosine(CPU-fp32, GPU-fp16) = {cos:.6f}  (gate {O.COS_GATE})")
    if cos < O.COS_GATE:
        raise RuntimeError(f"VECTOR-SPACE MISMATCH cos={cos:.6f} — refusing to restore these vectors")
    return cos


def step_restore_delta(expected_points: int) -> None:
    """Restore the pulled snapshot into LOCAL Qdrant as `georgian_legal_delta` (API-keyed)."""
    from dotenv import dotenv_values
    key = dotenv_values(O.INGEST / ".env").get("QDRANT_API_KEY") or ""
    snap = OUT / f"{DELTA_COLLECTION}.snapshot"
    O.log(f"restoring delta snapshot into LOCAL Qdrant as `{DELTA_COLLECTION}`...")
    O.run(["curl", "-sf", "-X", "POST",
           f"http://127.0.0.1:6333/collections/{DELTA_COLLECTION}/snapshots/upload?priority=snapshot",
           "-H", f"api-key: {key}",
           "-H", "Content-Type: multipart/form-data",
           "-F", f"snapshot=@{snap}"], timeout=3600)
    info = O.run(["curl", "-sf", "-H", f"api-key: {key}",
                  f"http://127.0.0.1:6333/collections/{DELTA_COLLECTION}"]).stdout.decode()
    pts = json.loads(info)["result"]["points_count"]
    O.log(f"restored `{DELTA_COLLECTION}`: points_count={pts} (embed reported {expected_points})")
    if expected_points and pts != expected_points:
        raise RuntimeError(f"restored point count {pts} != embed-reported {expected_points}")
    O.log("next: .venv/bin/python scripts/merge_delta_collection.py --dry-run   # then --src "
          + DELTA_COLLECTION)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-since", default=None,
                    help="include matsne run dirs whose id >= this (e.g. 20260709T080007Z)")
    ap.add_argument("--items", nargs="*", help="explicit items.jsonl path(s) (overrides --runs-since)")
    ap.add_argument("--skip-pod", action="store_true",
                    help="skip provisioning; just G2-verify + restore an already-pulled snapshot")
    args = ap.parse_args()

    atexit.register(O._cleanup)
    signal.signal(signal.SIGINT, O._sig)
    signal.signal(signal.SIGTERM, O._sig)

    points = 0
    if not args.skip_pod:
        if not args.runs_since and not args.items:
            raise SystemExit("pass --runs-since or --items (which delta to embed)")
        runs_dir = O.REPO / "artifacts" / "matsne" / "runs"
        stage_root = O.WORKDIR / "delta_stage"
        staged = stage_delta_items(runs_dir, args.runs_since, stage_root / "delta_items",
                                   explicit=args.items)
        if not staged:
            raise SystemExit("no non-empty items.jsonl matched the delta selection")
        n_lines = sum(1 for p in staged for _ in p.open(encoding="utf-8"))
        O.log(f"staged {len(staged)} run file(s), {n_lines} item lines total")

        O.step_checksum_ref()
        passphrase = step_package_delta(stage_root)
        pubkey = O.step_keypair()
        try:
            O._pod_id, gpu_used, O._price = O.step_provision(pubkey)
            O._provisioned_at = time.time()
            O._ip, O._port = O.step_wait_ssh(O._pod_id)
            O.ensure_pod_tools(O._ip, O._port)
            _retry(lambda: O.push_file(O.WORKDIR / "payload_delta.tar.gz.enc",
                                       "/workspace/payload.tar.gz.enc", O._ip, O._port,
                                       timeout=14400),
                   attempts=8, delay=30, what="payload push")
            _retry(lambda: O.push_file(EMBED_SH, "/workspace/runpod_embed_delta.sh",
                                       O._ip, O._port, timeout=120),
                   what="embed script push")
            step_launch_delta(O._ip, O._port, passphrase)
            points = step_poll_delta(O._ip, O._port)
            step_transfer_out_delta(O._ip, O._port)
        finally:
            if O._pod_id:
                if O._ip and O._port:  # best-effort wipe (terminate destroys the volume anyway)
                    try:
                        O.ssh_ok(O._ip, O._port, "rm -rf /workspace/* /dev/shm/p.env 2>/dev/null; sync",
                                 timeout=60)
                    except Exception:  # noqa: BLE001
                        pass
                O.terminate(O._pod_id)
                if O._provisioned_at:
                    hrs = (time.time() - O._provisioned_at) / 3600
                    cost = hrs * (O._price or 0)
                    O.log(f"COST: up={hrs*60:.1f}min price=${O._price}/hr est=${cost:.3f}")

    step_verify_g2_delta()
    step_restore_delta(points)
    O.log("DELTA EMBED + LOCAL RESTORE COMPLETE ✓ (merge is the next explicit step)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "terminate":
        pid_file = O.WORKDIR / "pod.id"
        if pid_file.exists():
            O.terminate(pid_file.read_text().strip())
        else:
            O.log("no pod.id to terminate")
        sys.exit(0)
    main()
