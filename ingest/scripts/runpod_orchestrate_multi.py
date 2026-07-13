#!/usr/bin/env python3
"""Multi-GPU embed orchestrator: provision a 4x RTX 4090 pod, pull the corpus POD-TO-POD from
the current single-GPU pod (no slow home re-upload), run N sharded embed processes (one per
GPU) into one Qdrant, then verify + restore locally. try/finally always terminates the NEW pod.

The source pod (SRC_*) is left running as a fallback; terminate it manually once this succeeds.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

INGEST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INGEST))
WORKDIR = Path.home() / "gpu_embed_work"
KEY = str(WORKDIR / "id_ed25519")
CPU_REF = INGEST / "snapshots" / "v1" / "checksum_cpu.json"

# Source pod (has the unpacked corpus + corpus.tgz); supplied per run, never committed.
SRC_IP = os.environ.get("RUNPOD_SOURCE_IP", "")
SRC_PORT = os.environ.get("RUNPOD_SOURCE_PORT", "")

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/125.0.0.0 Safari/537.36")
GPU = "NVIDIA GeForce RTX 4090"
GPU_COUNT = 4
N_SHARDS = 4
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
COLLECTION = "georgian_legal"
QDRANT_VER = "v1.18.2"
COS_GATE = 0.999
SSH_DEADLINE_S = 15 * 60
EMBED_DEADLINE_S = 5 * 3600   # generous: multi-hour embed
POLL_S = 60
BUDGET = 15.0

LOGFILE = WORKDIR / "orchestrate_multi.log"
_pod_id = None
_ip = _port = None
_provisioned_at = None
_price = None
_terminated = False
_KEY = None


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc):%FT%TZ}] {msg}"
    print(line, flush=True)
    try:
        WORKDIR.mkdir(parents=True, exist_ok=True)
        LOGFILE.open("a").write(line + "\n")
    except OSError:
        pass


def gql(query: str, variables: dict | None = None) -> dict:
    global _KEY
    if _KEY is None:
        from dotenv import dotenv_values
        _KEY = dotenv_values(INGEST / ".env").get("RUNPOD_API_KEY")
    # Auth via `Authorization: Bearer` header, not a `?api_key=` query string (a URL secret is
    # logged verbatim by CDN/proxy access logs; a header is not). RunPod's GraphQL accepts it.
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={"Content-Type": "application/json", "User-Agent": UA,
                 "Authorization": f"Bearer {_KEY}"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read().decode())
    if out.get("errors"):
        raise RuntimeError("GraphQL: " + json.dumps(out["errors"]))
    return out["data"]


def run(cmd, *, input=None, timeout=None, check=True, ok=(0,)):
    r = subprocess.run(cmd, input=input, capture_output=True, timeout=timeout)
    if check and r.returncode not in ok:
        raise RuntimeError(f"cmd failed ({r.returncode}): {' '.join(cmd[:3])}\n"
                           f"{r.stderr.decode(errors='replace')[-1500:]}")
    return r


def _ssh(ip, port):
    return ["ssh", "-p", str(port), "-i", KEY, "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={WORKDIR / 'known_hosts'}",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20"]


def ssh_ok(ip, port, remote, timeout=40):
    return subprocess.run(_ssh(ip, port) + [f"root@{ip}", remote],
                          capture_output=True, timeout=timeout).returncode == 0


def ssh_cap(ip, port, remote, timeout=60):
    return subprocess.run(_ssh(ip, port) + [f"root@{ip}", remote],
                          capture_output=True, timeout=timeout).stdout.decode(errors="replace")


def push_content(ip, port, path, content, mode=None):
    run(_ssh(ip, port) + [f"root@{ip}", f"cat > {shlex.quote(path)}"], input=content.encode(),
        timeout=60)
    if mode:
        run(_ssh(ip, port) + [f"root@{ip}", f"chmod {mode} {shlex.quote(path)}"], timeout=30)


def pull_file(ip, port, remote_path, local: Path, timeout=14400):
    with open(local, "wb") as fh:
        r = subprocess.run(_ssh(ip, port) + [f"root@{ip}", f"cat {shlex.quote(remote_path)}"],
                           stdout=fh, stderr=subprocess.PIPE, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"pull failed: {r.stderr.decode(errors='replace')[-400:]}")


def ensure_pod_tools(ip, port):
    if not ssh_ok(ip, port, "command -v curl >/dev/null && command -v openssl >/dev/null", 30):
        log("installing pod tools...")
        ssh_ok(ip, port, "apt-get update -qq && apt-get install -y -qq curl openssl", 300)


def gpu_price():
    q = ('query($id:String!){ gpuTypes(input:{id:$id}){ lowestPrice(input:{gpuCount:%d,'
         'secureCloud:true}){ uninterruptablePrice stockStatus } } }' % GPU_COUNT)
    t = gql(q, {"id": GPU}).get("gpuTypes") or []
    lp = (t[0].get("lowestPrice") or {}) if t else {}
    return lp.get("uninterruptablePrice"), lp.get("stockStatus")


def provision(pubkey):
    price, stock = gpu_price()
    log(f"{GPU} x{GPU_COUNT}: ${price}/gpu-hr stock={stock}")
    mut = ("mutation($i:PodFindAndDeployOnDemandInput!){ podFindAndDeployOnDemand(input:$i){ id } }")
    var = {"i": {"cloudType": "SECURE", "gpuCount": GPU_COUNT, "gpuTypeId": GPU,
                 "minMemoryInGb": 80, "minVcpuCount": 16, "name": "georgian-legal-embed-4x",
                 "imageName": IMAGE, "ports": "22/tcp", "volumeInGb": 120,
                 "containerDiskInGb": 80, "volumeMountPath": "/workspace",
                 "supportPublicIp": True, "startSsh": True,
                 "env": [{"key": "PUBLIC_KEY", "value": pubkey}]}}
    for attempt in range(1, 5):
        pod = gql(mut, var).get("podFindAndDeployOnDemand")
        if pod and pod.get("id"):
            (WORKDIR / "pod_multi.id").write_text(pod["id"])
            # RunPod lowestPrice(gpuCount:N) is already the TOTAL pod price, not per-GPU.
            log(f"provisioned 4x pod {pod['id']} (${price}/hr total for {GPU_COUNT} GPUs)")
            return pod["id"], price
        log(f"deploy attempt {attempt}: no capacity, retry...")
        time.sleep(20)
    raise RuntimeError("could not provision 4x pod")


def wait_ssh(pod_id):
    q = ("query($id:String!){ pod(input:{podId:$id}){ runtime{ ports{ ip isIpPublic privatePort "
         "publicPort type } } } }")
    deadline = time.time() + SSH_DEADLINE_S
    while time.time() < deadline:
        rt = (gql(q, {"id": pod_id}).get("pod") or {}).get("runtime")
        if rt and rt.get("ports"):
            for p in rt["ports"]:
                if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("type") == "tcp":
                    ip, port = p["ip"], p["publicPort"]
                    for _ in range(20):
                        if ssh_ok(ip, port, "true", 20):
                            log(f"SSH ready {ip}:{port}")
                            return ip, port
                        time.sleep(10)
        time.sleep(15)
    raise TimeoutError("SSH not ready")


def wait_corpus_ready():
    log("waiting for corpus.tgz on the source pod...")
    for _ in range(60):
        if ssh_ok(SRC_IP, SRC_PORT, "test -f /workspace/out/tar.done && test -f /workspace/corpus.tgz", 30):
            sz = ssh_cap(SRC_IP, SRC_PORT, "stat -c%s /workspace/corpus.tgz 2>/dev/null").strip()
            log(f"corpus.tgz ready ({int(sz)/1e9:.2f} GB) on source pod")
            return
        time.sleep(15)
    raise TimeoutError("corpus.tgz not ready on source pod")


def transfer_corpus(ip, port):
    log("delivering corpus pod-to-pod (source pod → new pod, datacenter speed)...")
    push_content(ip, port, "/root/.ssh/srckey", Path(KEY).read_text(), mode="600")
    ensure_pod_tools(ip, port)
    ssh_ok(ip, port, "command -v scp >/dev/null || (apt-get install -y -qq openssh-client)", 180)
    remote = (f"scp -i /root/.ssh/srckey -P {SRC_PORT} -o StrictHostKeyChecking=accept-new "
              f"-o UserKnownHostsFile=/root/.ssh/known_hosts "
              f"root@{SRC_IP}:/workspace/corpus.tgz /workspace/corpus.tgz && "
              f"cd /workspace && tar xzf corpus.tgz --no-same-owner --no-same-permissions && "
              f"rm -f corpus.tgz /root/.ssh/srckey && echo TRANSFER_OK")
    out = ssh_cap(ip, port, remote, timeout=1800)
    if "TRANSFER_OK" not in out:
        raise RuntimeError(f"corpus transfer failed: {out[-500:]}")
    log("corpus delivered + unpacked on new pod")


def push_code(ip, port):
    """Overwrite the (stale) ingest package on the pod with the CURRENT local code — the corpus
    tarball came from the source pod, whose code may predate local changes (e.g. --shard)."""
    tar = WORKDIR / "code.tgz"
    run(["tar", "czf", str(tar), "-C", str(INGEST.parent),
         "--exclude=__pycache__", "--exclude=*.pyc",
         "ingest/ingest", "ingest/pyproject.toml"], timeout=120)
    r = subprocess.run(
        _ssh(ip, port) + [f"root@{ip}",
            "cat > /workspace/code.tgz && cd /workspace && tar xzf code.tgz --no-same-owner "
            "--no-same-permissions && rm -f code.tgz && echo CODE_OK"],
        input=tar.read_bytes(), capture_output=True, timeout=120)
    if b"CODE_OK" not in r.stdout:
        raise RuntimeError(f"push_code failed: {r.stderr.decode(errors='replace')[-400:]}")
    log("pushed current local ingest code over the stale copy (adds --shard)")


def launch(ip, port):
    push_content(ip, port, "/workspace/runpod_embed_multi.sh",
                 (INGEST / "scripts" / "runpod_embed_multi.sh").read_text(), mode="755")
    remote = (f"mkdir -p /workspace/out; setsid env COLLECTION_NAME={COLLECTION} "
              f"QDRANT_VER={QDRANT_VER} SHARDS={N_SHARDS} EMBED_BATCH_SIZE=256 WORK=/workspace "
              f"bash /workspace/runpod_embed_multi.sh >/workspace/out/launch.out 2>&1 </dev/null & "
              f"echo LAUNCHED")
    if "LAUNCHED" not in ssh_cap(ip, port, remote, 60):
        raise RuntimeError("failed to launch multi embed")
    log(f"multi embed launched ({N_SHARDS} shards, setsid)")


def poll(ip, port):
    deadline = time.time() + EMBED_DEADLINE_S
    while time.time() < deadline:
        if ssh_ok(ip, port, "test -f /workspace/out/DONE"):
            return int(ssh_cap(ip, port, "cat /workspace/out/DONE").strip() or "0")
        tail = ssh_cap(ip, port, "tail -n 2 /workspace/out/embed.log 2>/dev/null").strip()
        pts = ssh_cap(ip, port, "curl -sf http://127.0.0.1:6333/collections/georgian_legal 2>/dev/null").strip()
        m = re.search(r'"points_count":(\d+)', pts) if pts else None
        log(f"  embed running · points={m.group(1) if m else '?'} · {tail[-160:]}")
        # bracket-safe liveness (avoids matching our own ssh command)
        if not ssh_ok(ip, port, "pgrep -f '[r]unpod_embed_multi' >/dev/null 2>&1"):
            if ssh_ok(ip, port, "test -f /workspace/out/DONE"):
                continue
            raise RuntimeError("multi embed ended without DONE:\n"
                               + ssh_cap(ip, port, "tail -n 30 /workspace/out/embed.log /workspace/out/shard*.log 2>/dev/null"))
        time.sleep(POLL_S)
    raise TimeoutError("embed exceeded deadline")


def terminate(pod_id):
    """Terminate the pod, retrying transient failures; set the ``_terminated`` backstop flag
    ONLY after podTerminate succeeds. The old code set it before the call, so one transient
    API failure permanently disabled the finally/atexit cleanup and leaked a billing pod."""
    global _terminated
    if not pod_id or _terminated:
        return
    last_err = None
    for attempt in range(1, 4):
        try:
            gql("mutation($id:String!){ podTerminate(input:{podId:$id}) }", {"id": pod_id})
            _terminated = True
            log(f"new pod {pod_id} terminated")
            (WORKDIR / "pod_multi.id").unlink(missing_ok=True)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            log(f"WARNING: terminate attempt {attempt}/3 failed {pod_id}: {e}")
            time.sleep(min(5 * attempt, 15))
    log(f"WARNING: terminate FAILED {pod_id} after 3 attempts: {last_err} — pod may still be "
        f"BILLING; run `python scripts/runpod_orchestrate_multi.py terminate` or use the console.")


def _cleanup():
    if _pod_id and not _terminated:
        terminate(_pod_id)


def restore(points):
    snap = WORKDIR / "out_multi" / f"{COLLECTION}.snapshot"
    log("restoring snapshot into LOCAL Qdrant...")
    run(["curl", "-sf", "-X", "POST",
         f"http://127.0.0.1:6333/collections/{COLLECTION}/snapshots/upload?priority=snapshot",
         "-H", "Content-Type: multipart/form-data", "-F", f"snapshot=@{snap}"], timeout=3600)
    info = run(["curl", "-sf", f"http://127.0.0.1:6333/collections/{COLLECTION}"]).stdout.decode()
    pts = json.loads(info)["result"]["points_count"]
    log(f"restored: points_count={pts} (embed reported {points})")


def main():
    global _pod_id, _ip, _port, _provisioned_at, _price
    if not SRC_IP or not SRC_PORT:
        raise SystemExit("Set RUNPOD_SOURCE_IP and RUNPOD_SOURCE_PORT before starting")
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *a: (_cleanup(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda *a: (_cleanup(), sys.exit(1)))
    assert CPU_REF.exists(), "checksum_cpu.json missing"
    for p in (WORKDIR / "id_ed25519.pub",):
        assert p.exists(), f"missing {p} (need the ephemeral pubkey)"
    pubkey = (WORKDIR / "id_ed25519.pub").read_text().strip()

    wait_corpus_ready()
    points = 0
    cos = None
    try:
        _pod_id, _price = provision(pubkey)
        _provisioned_at = time.time()
        _ip, _port = wait_ssh(_pod_id)
        transfer_corpus(_ip, _port)
        push_code(_ip, _port)
        launch(_ip, _port)
        points = poll(_ip, _port)
        out = WORKDIR / "out_multi"
        out.mkdir(exist_ok=True)
        for name in ("checksum_gpu.json", "embed.log"):
            try:
                pull_file(_ip, _port, f"/workspace/out/{name}", out / name, 600)
            except RuntimeError:
                pass
        pull_file(_ip, _port, f"/workspace/out/{COLLECTION}.snapshot", out / f"{COLLECTION}.snapshot")
        from ingest.embed_job import checksum_cosine
        cpu = json.loads(CPU_REF.read_text())
        gpu = json.loads((out / "checksum_gpu.json").read_text())
        cos = checksum_cosine(cpu["dense"], gpu["dense"])
        log(f"G2 cosine = {cos:.6f}")
        if cos < COS_GATE:
            raise RuntimeError(f"vector mismatch cos={cos}")
    finally:
        if _pod_id:
            terminate(_pod_id)
            if _provisioned_at:
                hrs = (time.time() - _provisioned_at) / 3600
                log(f"COST: {hrs*60:.1f}min x ${_price}/hr = ${hrs*(_price or 0):.2f}")
    restore(points)
    log(f"MULTI-GPU EMBED COMPLETE ✓ (cos={cos:.6f}, points={points})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "terminate":
        pid = (WORKDIR / "pod_multi.id").read_text().strip() if (WORKDIR / "pod_multi.id").exists() else None
        terminate(pid) if pid else log("no pod_multi.id")
        sys.exit(0)
    main()
