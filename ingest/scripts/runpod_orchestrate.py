#!/usr/bin/env python3
"""Local orchestrator for the one-time RunPod GPU full-corpus embed (Part 3 linchpin).

Drives the on-pod bootstrap ``scripts/runpod_embed.sh`` end to end and GUARANTEES the pod is
terminated (try/finally + atexit + SIGINT/SIGTERM), so a crash can never leak paid GPU time.

Flow:
  0. assert the local CPU checksum reference exists (guardrail G2 baseline)
  1. package {ingest pkg, pyproject, scripts, snapshots/v1/docs} → tar → openssl-encrypt
     (random passphrase held only in memory; never on disk or in logs)
  2. provision a Secure-Cloud RTX 4090 (browser-UA GraphQL; live price/stock; ephemeral SSH key)
  3. wait for the public 22/tcp mapping + sshd, transfer the encrypted payload in
  4. run runpod_embed.sh in tmux (COLLECTION_NAME=georgian_legal, EMBED_DEVICE=cuda FP16,
     QDRANT_VER matched to LOCAL Qdrant so the snapshot restores cleanly)
  5. poll for out/DONE, transfer out the snapshot + GPU checksum
  6. verify checksum_cosine(CPU, GPU) >= COS_GATE  (fail loud; do not restore on mismatch)
  7. finally: wipe volume + podTerminate; log pod-hours × price vs the $15 budget
  8. restore the snapshot into LOCAL Qdrant as `georgian_legal`; assert points_count

Run (from ingest/):   .venv/bin/python scripts/runpod_orchestrate.py
Emergency cleanup:    .venv/bin/python scripts/runpod_orchestrate.py terminate
"""
from __future__ import annotations

import atexit
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# --- paths / constants --------------------------------------------------------
INGEST = Path(__file__).resolve().parents[1]                 # .../Georgia-Legal-Search/ingest
REPO = INGEST.parent
WORKDIR = Path.home() / "gpu_embed_work"                     # under /home (not tmpfs): holds keys, enc payload, OUT snapshot
CPU_REF = INGEST / "snapshots" / "v1" / "checksum_cpu.json"
sys.path.insert(0, str(INGEST))                              # import the local `ingest` package

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/125.0.0.0 Safari/537.36")
GPU_PRIMARY = "NVIDIA GeForce RTX 4090"
GPU_FALLBACKS = ["NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
                 "NVIDIA RTX 4000 Ada Generation", "NVIDIA L4"]
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
COLLECTION = "georgian_legal"
QDRANT_VER = "v1.18.2"                                       # match LOCAL Qdrant → clean restore
COS_GATE = 0.999                                             # G2: below this = different vector space
SSH_DEADLINE_S = 15 * 60                                     # image pull + boot + sshd
EMBED_DEADLINE_S = 120 * 60                                  # hard wall-clock; watchdog force-terminates
POLL_S = 30
BUDGET = 15.00

LOGFILE = WORKDIR / "orchestrate.log"
_pod_id: str | None = None
_provisioned_at: float | None = None
_price: float | None = None
_terminated = False
_ip: str | None = None
_port: int | None = None


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc):%FT%TZ}] {msg}"
    print(line, flush=True)
    try:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        with LOGFILE.open("a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _api_key() -> str:
    from dotenv import dotenv_values
    key = dotenv_values(INGEST / ".env").get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY")
    if not key:
        sys.exit("RUNPOD_API_KEY not found in ingest/.env")
    return key


_KEY = None


def gql(query: str, variables: dict | None = None) -> dict:
    global _KEY
    if _KEY is None:
        _KEY = _api_key()
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    # Auth via `Authorization: Bearer` rather than a `?api_key=` query string: a secret in the
    # URL is captured verbatim by CDN/proxy/server access logs, while a header is not. RunPod's
    # GraphQL API accepts the Bearer header (same scheme remote_search / session_monitor use).
    req = urllib.request.Request(
        "https://api.runpod.io/graphql", data=body,
        headers={"Content-Type": "application/json", "User-Agent": UA,
                 "Authorization": f"Bearer {_KEY}"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        out = json.loads(resp.read().decode())
    if out.get("errors"):
        raise RuntimeError("GraphQL error: " + json.dumps(out["errors"]))
    return out["data"]


def run(cmd: list[str], *, input: bytes | None = None, timeout: int | None = None,
        check: bool = True, ok_codes: tuple = (0,)) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, input=input, capture_output=True, timeout=timeout)
    if check and r.returncode not in ok_codes:
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(cmd[:3])}...\n"
                           f"{r.stderr.decode(errors='replace')[-2000:]}")
    return r


# --- ssh helpers --------------------------------------------------------------
def _ssh_base(ip: str, port: int) -> list[str]:
    return ["ssh", "-p", str(port), "-i", str(WORKDIR / "id_ed25519"),
            "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={WORKDIR / 'known_hosts'}",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20"]


def ssh_ok(ip: str, port: int, remote: str, timeout: int = 30) -> bool:
    r = subprocess.run(_ssh_base(ip, port) + [f"root@{ip}", remote],
                       capture_output=True, timeout=timeout)
    return r.returncode == 0


def ssh_capture(ip: str, port: int, remote: str, timeout: int = 30) -> str:
    r = subprocess.run(_ssh_base(ip, port) + [f"root@{ip}", remote],
                       capture_output=True, timeout=timeout)
    return r.stdout.decode(errors="replace")


def push_content(ip: str, port: int, remote_path: str, content: str, mode: str | None = None) -> None:
    run(_ssh_base(ip, port) + [f"root@{ip}", f"cat > {shlex.quote(remote_path)}"],
        input=content.encode(), timeout=60)
    if mode:
        run(_ssh_base(ip, port) + [f"root@{ip}", f"chmod {mode} {shlex.quote(remote_path)}"], timeout=30)


_HAVE_RSYNC = [False]  # whether the pod has rsync (best-effort install); else stream via `cat`


def ensure_pod_tools(ip: str, port: int) -> None:
    """The runpod/pytorch image lacks several CLI tools. Install what the run needs:
    rsync (resumable transfers; else we stream via `cat`), and curl + openssl which
    ``runpod_embed.sh`` uses to fetch Qdrant and decrypt the payload. tmux is deliberately
    NOT required — the embed is launched with setsid (see step_launch)."""
    have = ssh_ok(ip, port,
                  "command -v rsync >/dev/null && command -v curl >/dev/null && command -v openssl >/dev/null",
                  timeout=30)
    if not have:
        log("installing pod tools (rsync curl openssl)...")
        ssh_ok(ip, port, "apt-get update -qq && apt-get install -y -qq rsync curl openssl", timeout=300)
    _HAVE_RSYNC[0] = ssh_ok(ip, port, "command -v rsync >/dev/null 2>&1", timeout=30)
    log(f"pod tools ready (rsync available: {_HAVE_RSYNC[0]})")


def _remote_size(ip: str, port: int, path: str) -> int:
    out = ssh_capture(ip, port, f"stat -c%s {shlex.quote(path)} 2>/dev/null", timeout=30).strip()
    return int(out) if out.isdigit() else -1


# Rsync WITHOUT -a: no owner/group/perm preservation → avoids the chown that RunPod's volume
# rejects (exit 23). --inplace resumes into the dest directly. Size match is the real success
# check, so benign attr codes (23/24) are tolerated.
_RSYNC_FLAGS = ["--partial", "--append-verify", "--inplace"]
_RSYNC_OK = (0, 23, 24)


def push_file(local: Path, remote_path: str, ip: str, port: int, timeout: int = 3600) -> None:
    local = Path(local)
    want = local.stat().st_size
    if _HAVE_RSYNC[0]:
        ssh_e = " ".join(shlex.quote(a) for a in _ssh_base(ip, port))
        run(["rsync", *_RSYNC_FLAGS, "-e", ssh_e, str(local), f"root@{ip}:{remote_path}"],
            timeout=timeout, ok_codes=_RSYNC_OK)
        if _remote_size(ip, port, remote_path) == want:
            return
        log("rsync push size mismatch — falling back to cat stream")
    with open(local, "rb") as fh:  # stream stdin → remote `cat` (no whole-file buffering)
        r = subprocess.run(_ssh_base(ip, port) + [f"root@{ip}", f"cat > {shlex.quote(remote_path)}"],
                           stdin=fh, stderr=subprocess.PIPE, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"push_file failed: {r.stderr.decode(errors='replace')[-500:]}")
    if _remote_size(ip, port, remote_path) != want:
        raise RuntimeError(f"push_file size mismatch: remote != {want}")


def pull_file(remote_path: str, local: Path, ip: str, port: int, timeout: int = 7200) -> None:
    local = Path(local)
    if _HAVE_RSYNC[0]:
        ssh_e = " ".join(shlex.quote(a) for a in _ssh_base(ip, port))
        run(["rsync", *_RSYNC_FLAGS, "-e", ssh_e, f"root@{ip}:{remote_path}", str(local)],
            timeout=timeout, ok_codes=_RSYNC_OK)
        rs = _remote_size(ip, port, remote_path)
        if rs >= 0 and local.exists() and local.stat().st_size == rs:
            return
        log("rsync pull size mismatch — falling back to cat stream")
    with open(local, "wb") as fh:  # stream remote `cat` → local file
        r = subprocess.run(_ssh_base(ip, port) + [f"root@{ip}", f"cat {shlex.quote(remote_path)}"],
                           stdout=fh, stderr=subprocess.PIPE, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"pull_file failed: {r.stderr.decode(errors='replace')[-500:]}")


# --- steps --------------------------------------------------------------------
def step_checksum_ref() -> None:
    if not CPU_REF.exists():
        log("CPU checksum reference missing — generating (CPU embed of the pin sentence)...")
        run([str(INGEST / ".venv/bin/python"), "-m", "ingest", "embed", "--checksum"],
            timeout=1200)
    ref = json.loads(CPU_REF.read_text())
    assert len(ref["dense"]) == 1024, "checksum_cpu.json dense dim != 1024"
    log(f"CPU checksum ref ok: sha={ref['sha']} dims={len(ref['dense'])}")


def step_package() -> str:
    import secrets
    WORKDIR.mkdir(parents=True, exist_ok=True)
    passphrase = secrets.token_urlsafe(36)
    os.environ["PASSPHRASE"] = passphrase  # inherited by the openssl subprocess; never on disk/logs
    plain = WORKDIR / "payload.tar.gz"
    enc = WORKDIR / "payload.tar.gz.enc"
    log("packaging payload (ingest pkg + pyproject + scripts + snapshots/v1/docs)...")
    run(["tar", "czf", str(plain), "-C", str(REPO),
         "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=*.pyo",
         "--exclude=.pytest_cache", "--exclude=.ruff_cache",
         "ingest/ingest", "ingest/pyproject.toml", "ingest/scripts", "ingest/snapshots/v1/docs"],
        timeout=1800)
    listing = run(["tar", "tzf", str(plain)]).stdout.decode()
    leaked = [ln for ln in listing.splitlines() if "/.env" in ln or ".venv" in ln or "/.state" in ln]
    if leaked:
        raise RuntimeError(f"payload would leak secrets/state: {leaked[:5]}")
    run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-pass", "env:PASSPHRASE",
         "-in", str(plain), "-out", str(enc)], timeout=600)
    plain.unlink()
    size_gb = enc.stat().st_size / 1e9
    log(f"payload encrypted: {enc.name} ({size_gb:.2f} GB); plaintext removed")
    return passphrase


def step_keypair() -> str:
    key = WORKDIR / "id_ed25519"
    for p in (key, WORKDIR / "id_ed25519.pub", WORKDIR / "known_hosts"):
        p.unlink(missing_ok=True)
    run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "georgian-legal-embed", "-f", str(key)],
        timeout=60)
    return (WORKDIR / "id_ed25519.pub").read_text().strip()


def gpu_price(gpu_id: str) -> tuple[float | None, str | None]:
    q = ('query($id:String!){ gpuTypes(input:{id:$id}){ '
         'lowestPrice(input:{gpuCount:1,minMemoryInGb:20,secureCloud:true}){ '
         'uninterruptablePrice stockStatus } } }')
    types = gql(q, {"id": gpu_id}).get("gpuTypes") or []
    if not types or not types[0].get("lowestPrice"):
        return None, None
    lp = types[0]["lowestPrice"]
    return lp.get("uninterruptablePrice"), lp.get("stockStatus")


def step_provision(pubkey: str) -> tuple[str, str, float]:
    mutation = ("mutation($input:PodFindAndDeployOnDemandInput!){ "
                "podFindAndDeployOnDemand(input:$input){ id imageName machineId } }")
    for gpu_id in [GPU_PRIMARY, *GPU_FALLBACKS]:
        price, stock = gpu_price(gpu_id)
        if price is None:
            log(f"  {gpu_id}: no secure price/stock — skipping")
            continue
        log(f"  {gpu_id}: ${price}/hr stock={stock}")
        for attempt in range(1, 4):
            variables = {"input": {
                "cloudType": "SECURE", "gpuCount": 1, "gpuTypeId": gpu_id,
                "minMemoryInGb": 20, "minVcpuCount": 4, "name": "georgian-legal-embed",
                "imageName": IMAGE, "dockerArgs": "", "ports": "22/tcp",
                "volumeInGb": 80, "containerDiskInGb": 60, "volumeMountPath": "/workspace",
                "supportPublicIp": True, "startSsh": True,
                "env": [{"key": "PUBLIC_KEY", "value": pubkey}],
            }}
            try:
                data = gql(mutation, variables)
            except Exception as e:  # noqa: BLE001 - transport faults (HTTPError/URLError/
                # TimeoutError) are NOT RuntimeError; catching only RuntimeError let them escape
                # step_provision uncaught, so a lost response after RunPod created the pod left it
                # billing with its id recorded nowhere. Catch broadly so the loop retries; the
                # `terminate` subcommand's name-reconciliation reaps any pod a lost response made.
                log(f"  deploy attempt {attempt} error: {e}")
                time.sleep(20)
                continue
            pod = data.get("podFindAndDeployOnDemand")
            if pod and pod.get("id"):
                (WORKDIR / "pod.id").write_text(pod["id"])
                log(f"provisioned pod {pod['id']} on {gpu_id} (${price}/hr)")
                return pod["id"], gpu_id, price
            log(f"  deploy attempt {attempt}: null (no capacity), retrying...")
            time.sleep(20)
    raise RuntimeError("could not provision any GPU (primary + fallbacks exhausted)")


def step_wait_ssh(pod_id: str) -> tuple[str, int]:
    q = ("query($id:String!){ pod(input:{podId:$id}){ desiredStatus "
         "runtime{ uptimeInSeconds ports{ ip isIpPublic privatePort publicPort type } } } }")
    deadline = time.time() + SSH_DEADLINE_S
    ip = port = None
    while time.time() < deadline:
        pod = gql(q, {"id": pod_id}).get("pod") or {}
        rt = pod.get("runtime")
        if rt and rt.get("ports"):
            for p in rt["ports"]:
                if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("type") == "tcp":
                    ip, port = p["ip"], p["publicPort"]
                    break
        if ip:
            log(f"public SSH mapped: {ip}:{port} — waiting for sshd...")
            for _ in range(20):
                if ssh_ok(ip, port, "true", timeout=20):
                    log("sshd accepting connections")
                    return ip, port
                time.sleep(10)
        time.sleep(15)
    raise TimeoutError("SSH not ready within deadline")


def step_transfer_in(ip: str, port: int) -> None:
    ensure_pod_tools(ip, port)
    size_gb = (WORKDIR / "payload.tar.gz.enc").stat().st_size / 1e9
    log(f"transferring encrypted payload ({size_gb:.2f} GB) + bootstrap to pod "
        f"(upload-bound; may take a while)...")
    # Generous timeout: a slow home upstream (~1-2 Mbps) can need well over an hour for ~1 GB.
    push_file(WORKDIR / "payload.tar.gz.enc", "/workspace/payload.tar.gz.enc", ip, port, timeout=14400)
    push_file(INGEST / "scripts" / "runpod_embed.sh", "/workspace/runpod_embed.sh", ip, port, timeout=120)


def step_launch(ip: str, port: int, passphrase: str) -> None:
    push_content(ip, port, "/dev/shm/p.env",
                 "export PASSPHRASE=%s\n" % shlex.quote(passphrase), mode="600")
    launch = (
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        "mkdir -p /workspace/out\n"
        "cd /workspace\n"
        ". /dev/shm/p.env\n"
        "rm -f /dev/shm/p.env\n"
        f"export COLLECTION_NAME={COLLECTION} QDRANT_VER={QDRANT_VER} EMBED_BATCH_SIZE=256 WORK=/workspace\n"
        "bash /workspace/runpod_embed.sh\n"
        'echo "EXIT=$?" >> /workspace/out/embed.log\n'
    )
    push_content(ip, port, "/workspace/launch.sh", launch, mode="755")
    # Detach with setsid (no tmux dependency): survives the SSH session closing. Redirect all
    # fds off the ssh channel so the ssh command returns immediately.
    remote = ("mkdir -p /workspace/out; "
              "setsid bash /workspace/launch.sh >/workspace/out/launch.out 2>&1 </dev/null & "
              "echo LAUNCHED")
    out = ssh_capture(ip, port, remote, timeout=60)
    if "LAUNCHED" not in out:
        raise RuntimeError(f"failed to launch embed (setsid): {out!r}")
    log("embed launched (setsid, detached)")


def step_poll(ip: str, port: int) -> int:
    deadline = time.time() + EMBED_DEADLINE_S
    # A single ssh_ok()==False is ambiguous: the embed may be dead, OR the SSH round-trip
    # just blipped (ConnectTimeout, transient network, a busy sshd). Declaring death on one
    # blip terminates the pod and throws away the multi-hour, most-expensive embed. Require
    # several CONSECUTIVE not-alive readings (DONE re-checked each time) before giving up.
    _DEAD_CONFIRM = 3
    dead_polls = 0
    while time.time() < deadline:
        if ssh_ok(ip, port, "test -f /workspace/out/DONE"):
            pts = ssh_capture(ip, port, "cat /workspace/out/DONE").strip()
            log(f"DONE marker found: points={pts}")
            return int(pts or "0")
        tail = ssh_capture(
            ip, port, "tail -n 3 /workspace/out/embed.log /workspace/out/launch.out 2>/dev/null").strip()
        if tail:
            log("  embed: " + tail.replace("\n", " | ")[-300:])
        alive = ssh_ok(ip, port, "pgrep -f runpod_embed.sh >/dev/null 2>&1 "
                                 "|| pgrep -f 'bash /workspace/launch.sh' >/dev/null 2>&1")
        if not alive:
            if ssh_ok(ip, port, "test -f /workspace/out/DONE"):
                continue
            dead_polls += 1
            if dead_polls >= _DEAD_CONFIRM:
                full = ssh_capture(
                    ip, port,
                    "tail -n 40 /workspace/out/launch.out /workspace/out/embed.log 2>/dev/null")
                raise RuntimeError(
                    f"embed process not running for {dead_polls} consecutive polls without a "
                    f"DONE marker:\n" + full)
            log(f"  aliveness check failed ({dead_polls}/{_DEAD_CONFIRM}) — transient SSH blip? "
                "re-checking next poll")
        else:
            dead_polls = 0
        time.sleep(POLL_S)
    raise TimeoutError("embed exceeded EMBED_DEADLINE")


def step_transfer_out(ip: str, port: int) -> Path:
    out = WORKDIR / "out"
    out.mkdir(exist_ok=True)
    log("transferring results out (snapshot + checksum + logs)...")
    for name in ("checksum_gpu.json", "embed.log", "qdrant.log", "checksum_stdout.txt"):
        try:
            pull_file(f"/workspace/out/{name}", out / name, ip, port, timeout=600)
        except RuntimeError as e:
            log(f"  (optional {name} pull failed: {e})")
    pull_file(f"/workspace/out/{COLLECTION}.snapshot", out / f"{COLLECTION}.snapshot", ip, port, timeout=14400)
    snap = out / f"{COLLECTION}.snapshot"
    if not snap.exists() or snap.stat().st_size < 1_000_000:
        raise RuntimeError(f"snapshot missing/too small at {snap}")
    log(f"got snapshot {snap.name} ({snap.stat().st_size/1e9:.2f} GB)")
    return snap


def step_verify_g2() -> float:
    from ingest.embed_job import checksum_cosine
    cpu = json.loads(CPU_REF.read_text())
    gpu = json.loads((WORKDIR / "out" / "checksum_gpu.json").read_text())
    cos = checksum_cosine(cpu["dense"], gpu["dense"])
    log(f"G2 cosine(CPU-fp32, GPU-fp16) = {cos:.6f}  (gate {COS_GATE})")
    if cos < COS_GATE:
        raise RuntimeError(f"VECTOR-SPACE MISMATCH cos={cos:.6f} — refusing to restore these vectors")
    return cos


def _list_pods() -> list[dict]:
    """Every pod on the account (id, name, desiredStatus, costPerHr). Best-effort — returns
    [] on any API failure so reconciliation never masks the real error."""
    try:
        data = gql("query{ myself{ pods{ id name desiredStatus costPerHr } } }")
    except Exception as e:  # noqa: BLE001
        log(f"could not list account pods: {e}")
        return []
    return ((data.get("myself") or {}).get("pods")) or []


def _reap_by_name(name: str = "georgian-legal-embed") -> int:
    """Terminate every non-terminated pod carrying this job's name and return the count.

    Reconciles two orphan classes the on-disk ``pod.id`` cannot: a pod created by a deploy
    call whose HTTP response was lost (its id is recorded nowhere), and a ``pod.id`` clobbered
    by a concurrent orchestrator (the shared file holds only the last writer's id). Invoked
    ONLY by the operator via the ``terminate`` subcommand — never during a normal run — so it
    cannot race a concurrent same-name provision mid-flight.
    """
    n = 0
    for p in _list_pods():
        if p.get("name") == name and (p.get("desiredStatus") or "").upper() != "TERMINATED":
            log(f"reaping pod {p.get('id')} (name={name}, status={p.get('desiredStatus')}, "
                f"${p.get('costPerHr')}/hr)")
            try:
                gql("mutation($id:String!){ podTerminate(input:{podId:$id}) }", {"id": p["id"]})
                n += 1
            except Exception as e:  # noqa: BLE001
                log(f"  reap terminate failed for {p.get('id')}: {e}")
    return n


def terminate(pod_id: str | None) -> None:
    """Terminate the pod, retrying transient API failures.

    Sets the ``_terminated`` backstop flag ONLY after ``podTerminate`` actually succeeds. The
    old code set it *before* the call, so a single transient Cloudflare 403 / network blip
    permanently short-circuited both the ``finally`` block and the ``atexit`` cleanup (both
    gate on ``_terminated``) — the run then printed COMPLETE and exited 0 while the GPU pod
    kept billing (the realized orphaned-pod incident class). Leaving the flag False on failure
    lets those backstops retry. ``pod.id`` is unlinked only on success so the emergency
    ``terminate`` subcommand can still find the pod after a failed cleanup.
    """
    global _terminated
    if not pod_id or _terminated:
        return
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            gql("mutation($id:String!){ podTerminate(input:{podId:$id}) }", {"id": pod_id})
            _terminated = True
            log(f"pod {pod_id} terminated")
            (WORKDIR / "pod.id").unlink(missing_ok=True)
            return
        except Exception as e:  # noqa: BLE001 - never let cleanup failure mask the real error
            last_err = e
            log(f"WARNING: podTerminate attempt {attempt}/3 failed for {pod_id}: {e}")
            time.sleep(min(5 * attempt, 15))
    log(f"WARNING: podTerminate FAILED for {pod_id} after 3 attempts: {last_err}  — the pod may "
        f"still be BILLING. Run `python scripts/runpod_orchestrate.py terminate` or kill it in "
        f"the RunPod console. (_terminated left False so finally/atexit will retry.)")


def _cleanup() -> None:
    if _pod_id and not _terminated:
        log("cleanup: terminating pod (atexit/signal)")
        terminate(_pod_id)


def _sig(signum, frame):  # noqa: ANN001
    log(f"signal {signum} received")
    _cleanup()
    sys.exit(1)


def step_restore(expected_points: int) -> None:
    snap = WORKDIR / "out" / f"{COLLECTION}.snapshot"
    log(f"restoring snapshot into LOCAL Qdrant as `{COLLECTION}` (may take minutes)...")
    run(["curl", "-sf", "-X", "POST",
         f"http://127.0.0.1:6333/collections/{COLLECTION}/snapshots/upload?priority=snapshot",
         "-H", "Content-Type: multipart/form-data",
         "-F", f"snapshot=@{snap}"], timeout=3600)
    info = run(["curl", "-sf", f"http://127.0.0.1:6333/collections/{COLLECTION}"]).stdout.decode()
    pts = json.loads(info)["result"]["points_count"]
    log(f"restored: points_count={pts} (embed reported {expected_points})")
    if not (400_000 <= pts <= 700_000):
        log(f"WARNING: points_count {pts} outside expected ~500-600k range — inspect")


def main() -> None:
    global _pod_id, _provisioned_at, _price, _ip, _port
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    step_checksum_ref()
    passphrase = step_package()
    pubkey = step_keypair()

    cos = None
    points = 0
    try:
        _pod_id, gpu_used, _price = step_provision(pubkey)
        _provisioned_at = time.time()
        _ip, _port = step_wait_ssh(_pod_id)
        step_transfer_in(_ip, _port)
        step_launch(_ip, _port, passphrase)
        points = step_poll(_ip, _port)
        step_transfer_out(_ip, _port)
        cos = step_verify_g2()
    finally:
        if _pod_id:
            if _ip and _port:  # best-effort wipe before terminate (terminate destroys the volume anyway)
                try:
                    ssh_ok(_ip, _port, "rm -rf /workspace/* /dev/shm/p.env 2>/dev/null; sync", timeout=60)
                except Exception:  # noqa: BLE001
                    pass
            terminate(_pod_id)
            if _provisioned_at:
                hrs = (time.time() - _provisioned_at) / 3600
                cost = hrs * (_price or 0)
                log(f"COST: up={hrs*60:.1f}min price=${_price}/hr est=${cost:.3f} "
                    f"remaining≈${BUDGET - cost:.2f}")

    log(f"embed verified (cos={cos:.6f}); restoring locally")
    step_restore(points)
    log("PART-3 GPU EMBED COMPLETE ✓")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "terminate":
        pid = (WORKDIR / "pod.id").read_text().strip() if (WORKDIR / "pod.id").exists() else None
        if pid:
            terminate(pid)
        else:
            log("no pod.id on disk")
        # Reconcile by job name too: catches an orphan from a lost deploy response and any pod
        # whose id was clobbered in the shared pod.id by a concurrent orchestrator.
        reaped = _reap_by_name()
        log(f"name-reconciliation terminated {reaped} additional pod(s)")
        sys.exit(0)
    main()
