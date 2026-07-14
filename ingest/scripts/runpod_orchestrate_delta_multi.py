#!/usr/bin/env python3
"""Multi-source GPU delta: embed the staged missing docs → snapshot → merge into LIVE.

Completes the live ``georgian_legal`` collection after a scrape, on GPU (the CPU embed of
long court docs is ~14 h). Ships ``.state/delta_stage/<source>.jsonl`` (from
``stage_missing_items.py``) to a single cheap pod, runs ``runpod_delta_multi.sh``
(embed_delta --items-dir → ``georgian_legal_delta``, v1 headers), pulls the small snapshot,
restores it locally, and merges into live via ``merge_delta_collection.py``.

Pod-termination guarantees inherited from runpod_orchestrate (try/finally + atexit + signals
+ poll deadline). One GPU (cheapest available) — the delta is tiny.

    .venv/bin/python scripts/runpod_orchestrate_delta_multi.py            # full run
    .venv/bin/python scripts/runpod_orchestrate_delta_multi.py terminate  # emergency
"""
from __future__ import annotations

import atexit
import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_orchestrate as O  # noqa: E402 — reuse gql/ssh/provision/terminate helpers
from ingest.operational import refuse_legacy_operation  # noqa: E402

DELTA_COLLECTION = "georgian_legal_delta"
STAGE_SRC = O.INGEST / ".state" / "delta_stage"
EMBED_SH = Path(__file__).resolve().parent / "runpod_delta_multi.sh"
OUT = O.WORKDIR / "out_delta_multi"
POLL_DEADLINE_S = 3 * 3600   # 4,370 docs on 1 GPU ≈ 20-40 min; generous watchdog
POLL_S = 30
DEAD_CHECKS = 6
BUDGET_CEILING = 2.0


def _retry(fn, *, attempts=8, delay=30, what=""):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if i == attempts:
                raise
            O.log(f"{what}: attempt {i}/{attempts} failed ({str(exc)[-160:]}); retry in {delay}s")
            time.sleep(delay)


def step_package(passphrase_holder: list) -> str:
    import os
    import secrets
    files = sorted(STAGE_SRC.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no staged items in {STAGE_SRC} — run stage_missing_items.py first")
    total = sum(sum(1 for _ in f.open(encoding='utf-8')) for f in files)
    O.log(f"staged sources: {[f.stem for f in files]} · {total} docs")
    passphrase = secrets.token_urlsafe(36)
    passphrase_holder.append(passphrase)
    os.environ["PASSPHRASE"] = passphrase
    plain = O.WORKDIR / "payload_delta_multi.tar.gz"
    enc = O.WORKDIR / "payload_delta_multi.tar.gz.enc"
    O.run(["tar", "czf", str(plain), "--exclude=__pycache__", "--exclude=*.pyc",
           "--exclude=.pytest_cache", "--exclude=.ruff_cache",
           "-C", str(O.REPO), "ingest/ingest", "ingest/pyproject.toml", "ingest/scripts",
           "-C", str(STAGE_SRC.parent), STAGE_SRC.name], timeout=1800)
    listing = O.run(["tar", "tzf", str(plain)]).stdout.decode()
    leaked = [ln for ln in listing.splitlines() if "/.env" in ln or ".venv" in ln]
    if leaked:
        raise RuntimeError(f"payload would leak secrets: {leaked[:5]}")
    O.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-pass", "env:PASSPHRASE",
           "-in", str(plain), "-out", str(enc)], timeout=600)
    plain.unlink()
    O.log(f"payload encrypted: {enc.name} ({enc.stat().st_size/1e6:.1f} MB)")
    return passphrase


def step_launch(ip: str, port: int, passphrase: str) -> None:
    O.push_content(ip, port, "/dev/shm/p.env",
                   "export PASSPHRASE=%s\n" % shlex.quote(passphrase), mode="600")
    launch = (
        "#!/usr/bin/env bash\nset -uo pipefail\nmkdir -p /workspace/out\ncd /workspace\n"
        ". /dev/shm/p.env\nrm -f /dev/shm/p.env\n"
        "openssl enc -d -aes-256-cbc -pbkdf2 -pass env:PASSPHRASE -in payload.tar.enc | tar xzf -\n"
        "mv delta_stage delta_items 2>/dev/null || true\n"
        f"export COLLECTION_NAME={DELTA_COLLECTION} QDRANT_VER={O.QDRANT_VER} "
        "EMBED_BATCH_SIZE=256 WORK=/workspace\n"
        "bash /workspace/runpod_delta_multi.sh\n"
        'echo "EXIT=$?" >> /workspace/out/embed.log\n'
    )
    O.push_content(ip, port, "/workspace/launch.sh", launch, mode="755")
    out = O.ssh_capture(ip, port,
                        "mkdir -p /workspace/out; setsid bash /workspace/launch.sh "
                        ">/workspace/out/launch.out 2>&1 </dev/null & echo LAUNCHED", timeout=60)
    if "LAUNCHED" not in out:
        raise RuntimeError(f"failed to launch: {out!r}")
    O.log("multi-source delta launched (setsid)")


def step_poll(ip: str, port: int, price: float, t0: float) -> int:
    deadline = time.time() + POLL_DEADLINE_S
    dead = 0
    while time.time() < deadline:
        if (time.time() - t0) / 3600 * price > BUDGET_CEILING:
            raise RuntimeError(f"budget ceiling ${BUDGET_CEILING} exceeded")
        if O.ssh_ok(ip, port, "test -f /workspace/out/DONE"):
            return int(O.ssh_capture(ip, port, "cat /workspace/out/DONE").strip() or "0")
        tail = O.ssh_capture(ip, port, "tail -n 2 /workspace/out/embed.log 2>/dev/null").strip()
        if tail:
            O.log("  " + tail.replace("\n", " | ")[-260:])
        alive = O.ssh_ok(ip, port, "pgrep -f runpod_delta_multi.sh >/dev/null 2>&1 "
                                   "|| pgrep -f embed_delta.py >/dev/null 2>&1 "
                                   "|| pgrep -f 'bash /workspace/launch.sh' >/dev/null 2>&1")
        if alive:
            dead = 0
        else:
            dead += 1
            if dead >= DEAD_CHECKS:
                if O.ssh_ok(ip, port, "test -f /workspace/out/DONE"):
                    continue
                raise RuntimeError("delta ended without DONE:\n" + O.ssh_capture(
                    ip, port, "tail -n 40 /workspace/out/launch.out /workspace/out/embed.log 2>/dev/null"))
        time.sleep(POLL_S)
    raise TimeoutError("delta exceeded deadline")


def step_pull(ip: str, port: int) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    snap = OUT / f"{DELTA_COLLECTION}.snapshot"
    _retry(lambda: O.pull_file(f"/workspace/out/{DELTA_COLLECTION}.snapshot", snap, ip, port,
                               timeout=1800), attempts=8, delay=30, what="snapshot pull")
    O.run(["tar", "tf", str(snap)], timeout=300)  # truncated-tar guard
    if snap.stat().st_size < 100_000:
        raise RuntimeError("delta snapshot too small")
    O.log(f"pulled + tar-verified delta snapshot ({snap.stat().st_size/1e6:.0f} MB)")
    return snap


def step_restore_and_merge(expected: int) -> None:
    refuse_legacy_operation("delta restore followed by an in-place live merge")
    from dotenv import dotenv_values
    key = dotenv_values(O.INGEST / ".env").get("QDRANT_API_KEY") or ""
    snap = OUT / f"{DELTA_COLLECTION}.snapshot"
    O.run(["curl", "-sf", "-X", "POST",
           f"http://127.0.0.1:6333/collections/{DELTA_COLLECTION}/snapshots/upload?priority=snapshot",
           "-H", f"api-key: {key}", "-F", f"snapshot=@{snap}"], timeout=3600)
    info = json.loads(O.run(["curl", "-sf", "-H", f"api-key: {key}",
                             f"http://127.0.0.1:6333/collections/{DELTA_COLLECTION}"]).stdout.decode())
    pts = info["result"]["points_count"]
    O.log(f"restored `{DELTA_COLLECTION}`: {pts} points (pod reported {expected})")
    if expected and pts != expected:
        raise RuntimeError(f"restored {pts} != pod-reported {expected}")
    O.log("merging delta into LIVE georgian_legal ...")
    r = subprocess.run([str(O.INGEST / ".venv/bin/python"), "scripts/merge_delta_collection.py",
                        "--src", DELTA_COLLECTION], cwd=O.INGEST, capture_output=True, timeout=3600)
    O.log(r.stdout.decode()[-600:])
    if r.returncode:
        raise RuntimeError("merge failed:\n" + r.stderr.decode()[-800:])


def main() -> None:
    refuse_legacy_operation("multi-source delta embed and in-place live merge")
    atexit.register(O._cleanup)
    signal.signal(signal.SIGINT, O._sig)
    signal.signal(signal.SIGTERM, O._sig)
    holder: list = []
    try:
        passphrase = step_package(holder)
        pubkey = O.step_keypair()
        O._pod_id, gpu_used, O._price = O.step_provision(pubkey)
        O._provisioned_at = time.time()
        O._ip, O._port = O.step_wait_ssh(O._pod_id)
        O.ensure_pod_tools(O._ip, O._port)
        _retry(lambda: O.push_file(O.WORKDIR / "payload_delta_multi.tar.gz.enc",
                                   "/workspace/payload.tar.enc", O._ip, O._port, timeout=3600),
               what="payload push")
        _retry(lambda: O.push_file(EMBED_SH, "/workspace/runpod_delta_multi.sh",
                                   O._ip, O._port, timeout=120), what="script push")
        step_launch(O._ip, O._port, passphrase)
        points = step_poll(O._ip, O._port, O._price or 0.0, O._provisioned_at)
        O.log(f"pod delta embed done: {points} points")
        step_pull(O._ip, O._port)
    finally:
        if O._pod_id:
            if O._ip and O._port:
                try:
                    O.ssh_ok(O._ip, O._port, "rm -rf /workspace/* /dev/shm/p.env 2>/dev/null; sync",
                             timeout=60)
                except Exception:  # noqa: BLE001
                    pass
            O.terminate(O._pod_id)
            if O._provisioned_at:
                hrs = (time.time() - O._provisioned_at) / 3600
                O.log(f"COST: up={hrs*60:.1f}min price=${O._price}/hr est=${hrs*(O._price or 0):.3f}")
    step_restore_and_merge(points)
    O.log("MULTI-SOURCE DELTA COMPLETE ✓ — live georgian_legal updated; re-verify next")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "terminate":
        pid_file = O.WORKDIR / "pod.id"
        if pid_file.exists():
            O.terminate(pid_file.read_text().strip())
        else:
            O.log("no pod.id to terminate")
        sys.exit(0)
    main()
