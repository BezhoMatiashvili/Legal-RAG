#!/usr/bin/env python3
"""Autonomous I6 orchestrator: full v2-header re-embed → tunnel eval → gate → conditional pull.

One detached process does the whole experiment with zero human-in-the-loop billing
exposure (pod-termination guarantees inherited from runpod_orchestrate: try/finally +
atexit + signals + poll deadline):

  1. package the ``reembed_export.py`` rows + ingest code (encrypted; no .env/.state),
  2. provision one GPU pod (4090 primary), push, run ``runpod_reembed_v2.sh``
     (v2 headers, same point ids/payloads → the cleanest possible A/B vs the live index),
  3. eval OVER AN SSH TUNNEL against the pod's Qdrant — hybrid (recall stage) and
     rerank@50 via the pod's own GPU rerank server — logging normal experiment rows,
  4. gate vs the 2026-07-10 v1 rows at the SAME corpus state (nDCG@10 +>=0.02 on the
     production rerank mode, no slice -0.02), and ONLY on PASS pull the ~24 GB snapshot
     (the historical money-burner) and restore it locally as ``georgian_legal_v2``,
  5. always terminate + cost-report.

Usage (from ingest/):
    nohup .venv/bin/python scripts/runpod_orchestrate_reembed.py > ~/gpu_embed_work/reembed_v2.log 2>&1 &
    .venv/bin/python scripts/runpod_orchestrate_reembed.py terminate   # emergency
"""
from __future__ import annotations

import atexit
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_orchestrate as O  # noqa: E402 — gql/ssh/provision/terminate helpers

V2_COLLECTION = "georgian_legal_v2"
ROWS_SRC = O.INGEST / ".state" / "reembed_v2" / "rows"
STAGE = O.WORKDIR / "reembed_stage"
OUT = O.WORKDIR / "out_reembed_v2"
EMBED_SH = Path(__file__).resolve().parent / "runpod_reembed_v2.sh"
RERANK_SERVER = Path(__file__).resolve().parent / "runpod_rerank_server.py"
TRANSLATIONS = "eval/query_translations_v2.json"  # covers all 60 v2 EN pairs
GOLDEN_SET = "v2"  # frozen 337-pair superset of v1; has temporal + more citation (I6's targets)

# Sharded embed (user asked for 4 GPUs to speed up). Multi-GPU secure pods are supply-
# constrained, so cascade the count 4→2→1: grab the most parallelism actually available.
# Per-GPU cost is ~flat, so fewer GPUs = same $, just slower wall-clock.
# Single GPU (2026-07-11): balance is tight ($4.74) so run one cheap-first GPU; the batch=256
# fix makes a single 4090 finish in ~5-6h. Cascade kept as a list for reuse but 1-only here.
GPU_COUNTS = [4, 2, 1]  # user asked to speed up: try a 4-GPU pod, fall back 4->2->1 on capacity
# 4090-first (user's choice): cheap GPUs are 3-4x slower (13-23h — impractical) for this
# workload, so the 4090 (~5.8h at batch 256) is the only practical single-GPU option.
GPU_PREFERENCE = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"]
ACTUAL_GPUS = 1  # set by step_provision_cascade to the count actually obtained
# Single-GPU wall-clock for the full 2.65M chunks; deadline = this / actual_count + buffer.
DEADLINE_1GPU_S = {"NVIDIA GeForce RTX 4090": 6 * 3600,
                   "NVIDIA GeForce RTX 3090": 8 * 3600,
                   "NVIDIA RTX A5000": 8 * 3600,
                   "NVIDIA RTX 4000 Ada Generation": 12 * 3600}
DEADLINE_1GPU_DEFAULT = 12 * 3600
POLL_S = 60
DEAD_CHECKS = 6
TUNNEL_QDRANT = 16333
TUNNEL_RERANK = 18900
BUDGET_CEILING = float(os.environ.get("REEMBED_BUDGET_CEILING", "4.4"))  # env-overridable; default clean-aborts just under the old $4.74 balance

# References = v1-header LIVE collection at the CURRENT corpus on the v2 golden set (337)
# with v2 translations, written by scripts/scratchpad/ref_v2_driver.sh. (v2 ⊇ v1, so these
# guard the v1 pairs too; temporal+citation slices measure I6's benefit.)
REF_FILE = O.INGEST / ".state" / "ref_v2.json"


def _load_refs() -> tuple[dict, dict]:
    ref = json.loads(REF_FILE.read_text(encoding="utf-8"))
    hyb = ref["hybrid"]
    rr = {"ndcg10": ref["rerank"]["ndcg10"], "recall10": ref["rerank"]["recall10"],
          "slices": {k: tuple(v) for k, v in ref["rerank"]["slices"].items()}}
    return hyb, rr


GATE_MIN_GAIN = 0.02
GATE_SLICE_TOL = 0.02

_tunnel: subprocess.Popen | None = None


def _retry(fn, *, attempts=8, delay=30, what=""):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if i == attempts:
                raise
            O.log(f"{what}: attempt {i}/{attempts} failed ({str(exc)[-160:]}); retrying in {delay}s")
            time.sleep(delay)


def _kill_tunnel() -> None:
    global _tunnel
    if _tunnel and _tunnel.poll() is None:
        _tunnel.terminate()
    _tunnel = None


def _cleanup() -> None:
    _kill_tunnel()
    O._cleanup()


# --- packaging -----------------------------------------------------------------


def step_package() -> str:
    import secrets
    manifest = json.loads((ROWS_SRC / "manifest.json").read_text())
    O.log(f"rows manifest: {manifest['rows']:,} rows, {manifest['files']} files "
          f"(collection {manifest['collection']})")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copytree(ROWS_SRC, STAGE / "rows")  # keep the original export as the local record
    passphrase = secrets.token_urlsafe(36)
    os.environ["PASSPHRASE"] = passphrase
    plain = O.WORKDIR / "payload_reembed.tar"
    enc = O.WORKDIR / "payload_reembed.tar.enc"
    O.log("packaging payload (ingest pkg + scripts + rows)...")
    O.run(["tar", "cf", str(plain),  # rows are already gzip — no second compression pass
           "--exclude=__pycache__", "--exclude=*.pyc", "--exclude=.pytest_cache",
           "--exclude=.ruff_cache", "--exclude=.ckpt-*",
           "-C", str(O.REPO), "ingest/ingest", "ingest/pyproject.toml", "ingest/scripts",
           "-C", str(STAGE), "rows"], timeout=3600)
    listing = O.run(["tar", "tf", str(plain)]).stdout.decode()
    leaked = [ln for ln in listing.splitlines() if "/.env" in ln or ".venv" in ln or "/.state" in ln]
    if leaked:
        raise RuntimeError(f"payload would leak secrets/state: {leaked[:5]}")
    O.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-pass", "env:PASSPHRASE",
           "-in", str(plain), "-out", str(enc)], timeout=1800)
    plain.unlink()
    O.log(f"payload encrypted: {enc.name} ({enc.stat().st_size/1e9:.2f} GB)")
    return passphrase


def step_provision_cascade(pubkey: str) -> tuple[str, str, float]:
    """Provision the most GPUs actually available: try counts 4→2→1, each across the GPU
    preference list. Sets the module-level ACTUAL_GPUS to what was obtained (drives shards
    + deadline). Cost is ~flat per GPU, so a smaller pod is the same $, just slower."""
    global ACTUAL_GPUS
    mutation = ("mutation($input:PodFindAndDeployOnDemandInput!){ "
                "podFindAndDeployOnDemand(input:$input){ id imageName machineId } }")
    for count in GPU_COUNTS:
        for gpu_id in GPU_PREFERENCE:
            price, stock = O.gpu_price(gpu_id)
            if price is None:
                continue
            O.log(f"  trying {gpu_id} ×{count}: ${price}/GPU/hr (${price * count:.2f}/hr) stock={stock}")
            variables = {"input": {
                "cloudType": "SECURE", "gpuCount": count, "gpuTypeId": gpu_id,
                "minMemoryInGb": 20, "minVcpuCount": max(4, 2 * count),
                "name": "georgian-legal-reembed-v2",
                "imageName": O.IMAGE, "dockerArgs": "", "ports": "22/tcp",
                "volumeInGb": 120, "containerDiskInGb": 60, "volumeMountPath": "/workspace",
                "supportPublicIp": True, "startSsh": True,
                "env": [{"key": "PUBLIC_KEY", "value": pubkey}],
            }}
            try:
                data = O.gql(mutation, variables)
            except RuntimeError as e:
                O.log(f"    {gpu_id}×{count}: {str(e)[-120:]}")
                continue
            pod = data.get("podFindAndDeployOnDemand")
            if pod and pod.get("id"):
                (O.WORKDIR / "pod.id").write_text(pod["id"])
                ACTUAL_GPUS = count
                O.log(f"provisioned {count}x{gpu_id} pod {pod['id']} (${price * count:.2f}/hr)")
                return pod["id"], gpu_id, price * count
            O.log(f"    {gpu_id}×{count}: no capacity")
        O.log(f"  no {count}-GPU capacity on any type; trying fewer GPUs...")
    raise RuntimeError("could not provision any pod (4/2/1 GPU all exhausted)")


def step_launch(ip: str, port: int, passphrase: str) -> None:
    O.push_content(ip, port, "/dev/shm/p.env",
                   "export PASSPHRASE=%s\n" % shlex.quote(passphrase), mode="600")
    launch = (
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        "mkdir -p /workspace/out\n"
        "cd /workspace\n"
        ". /dev/shm/p.env\n"
        "rm -f /dev/shm/p.env\n"
        # extracts to /workspace/ingest/{ingest,pyproject.toml,scripts} + /workspace/rows —
        # exactly the layout runpod_reembed_v2.sh expects ($WORK/ingest project root, $WORK/rows)
        "openssl enc -d -aes-256-cbc -pbkdf2 -pass env:PASSPHRASE "
        "-in payload.tar.enc | tar xf -\n"
        f"export COLLECTION_NAME={V2_COLLECTION} QDRANT_VER={O.QDRANT_VER} "
        f"EMBED_BATCH_SIZE=256 WORK=/workspace SHARDS={ACTUAL_GPUS}\n"
        "bash /workspace/runpod_reembed_v2.sh\n"
        'echo "EXIT=$?" >> /workspace/out/embed.log\n'
    )
    O.push_content(ip, port, "/workspace/launch_reembed.sh", launch, mode="755")
    remote = ("mkdir -p /workspace/out; "
              "setsid bash /workspace/launch_reembed.sh >/workspace/out/launch.out 2>&1 </dev/null & "
              "echo LAUNCHED")
    out = O.ssh_capture(ip, port, remote, timeout=60)
    if "LAUNCHED" not in out:
        raise RuntimeError(f"failed to launch (setsid): {out!r}")
    O.log("reembed launched (setsid, detached)")


def step_poll(ip: str, port: int, deadline_s: int, price: float, t_start: float) -> int:
    deadline = time.time() + deadline_s
    dead = 0
    while time.time() < deadline:
        est = (time.time() - t_start) / 3600 * price
        if est > BUDGET_CEILING:
            raise RuntimeError(f"budget ceiling ${BUDGET_CEILING} exceeded (est ${est:.2f})")
        if O.ssh_ok(ip, port, "test -f /workspace/out/DONE"):
            pts = O.ssh_capture(ip, port, "cat /workspace/out/DONE").strip()
            O.log(f"DONE marker: points={pts}")
            return int(pts or "0")
        tail = O.ssh_capture(
            ip, port,
            "tail -n 2 /workspace/out/embed.log /workspace/out/shard0.log 2>/dev/null").strip()
        if tail:
            O.log("  pod: " + tail.replace("\n", " | ")[-280:])
        alive = O.ssh_ok(ip, port, "pgrep -f runpod_reembed_v2.sh >/dev/null 2>&1 "
                                   "|| pgrep -f reembed_v2.py >/dev/null 2>&1 "
                                   "|| pgrep -f 'bash /workspace/launch_reembed.sh' >/dev/null 2>&1")
        if alive:
            dead = 0
        else:
            dead += 1
            O.log(f"  liveness probe failed ({dead}/{DEAD_CHECKS})")
            if dead >= DEAD_CHECKS:
                if O.ssh_ok(ip, port, "test -f /workspace/out/DONE"):
                    continue
                full = O.ssh_capture(ip, port,
                                     "tail -n 40 /workspace/out/launch.out "
                                     "/workspace/out/embed.log /workspace/out/shard0.log 2>/dev/null")
                raise RuntimeError("reembed ended without DONE:\n" + full)
        time.sleep(POLL_S)
    raise TimeoutError("reembed exceeded deadline")


# --- tunnel eval ------------------------------------------------------------------


def step_tunnel(ip: str, port: int) -> None:
    global _tunnel
    _tunnel = subprocess.Popen(
        ["ssh", "-i", O.KEY, "-p", str(port), "-N",
         "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
         "-o", "ExitOnForwardFailure=yes",
         "-L", f"{TUNNEL_QDRANT}:127.0.0.1:6333",
         "-L", f"{TUNNEL_RERANK}:127.0.0.1:8900",
         f"root@{ip}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(30):
        try:
            out = O.run(["curl", "-sf", f"http://127.0.0.1:{TUNNEL_QDRANT}/collections/{V2_COLLECTION}"],
                        timeout=10).stdout.decode()
            pts = json.loads(out)["result"]["points_count"]
            O.log(f"tunnel up: {V2_COLLECTION} points={pts}")
            return
        except Exception:  # noqa: BLE001
            time.sleep(5)
    raise RuntimeError("qdrant tunnel failed")


def step_start_rerank_server(ip: str, port: int) -> None:
    O.push_file(RERANK_SERVER, "/workspace/rerank_server.py", ip, port, timeout=120)
    O.ssh_capture(ip, port,
                  "setsid /workspace/venv/bin/python /workspace/rerank_server.py "
                  ">/workspace/out/rerank_server.log 2>&1 </dev/null & echo OK", timeout=30)
    for _ in range(60):  # model download + load
        try:
            out = O.run(["curl", "-sf", f"http://127.0.0.1:{TUNNEL_RERANK}/health"], timeout=10)
            O.log(f"pod rerank server healthy: {out.stdout.decode()[:120]}")
            return
        except Exception:  # noqa: BLE001
            time.sleep(10)
    raise RuntimeError("pod rerank server did not become healthy")


def _eval_env(**over) -> dict:
    env = dict(os.environ)
    env.pop("QDRANT_API_KEY", None)  # pod qdrant is keyless; tunnel is localhost
    env.update({"QDRANT_URL": f"http://127.0.0.1:{TUNNEL_QDRANT}",
                "COLLECTION_NAME": V2_COLLECTION, "OMP_NUM_THREADS": "8"}, **over)
    return env


def _run_eval(args: list[str], env: dict, log_path: Path, timeout: int) -> dict:
    n_before = sum(1 for _ in open(log_path, encoding="utf-8")) if log_path.exists() else 0
    cmd = [str(O.INGEST / ".venv/bin/python"), "-m", "eval.evaluate",
           "--backend", "qdrant", "--relevance", "chunk", "--golden-set", GOLDEN_SET,
           "--translate-queries", TRANSLATIONS, "--log", "--log-path", str(log_path), *args]
    O.log("eval: " + " ".join(args))
    r = subprocess.run(cmd, cwd=O.INGEST, env=env, capture_output=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError("eval failed:\n" + r.stderr.decode(errors="replace")[-1500:])
    rows = [json.loads(ln) for ln in open(log_path, encoding="utf-8")]
    if len(rows) <= n_before:
        raise RuntimeError("eval logged no new row")
    return rows[-1]


def step_eval() -> tuple[dict, dict]:
    hyb = _run_eval(["--mode", "hybrid"], _eval_env(RERANK_ENABLED="false"),
                    O.INGEST / "eval" / "experiments.jsonl", timeout=3600)
    rr = _run_eval(["--mode", "rerank", "--rerank-candidates", "50"],
                   _eval_env(RERANK_ENABLED="true",
                             RERANK_REMOTE_URL=f"http://127.0.0.1:{TUNNEL_RERANK}"),
                   O.INGEST / "eval" / "experiments_gpu.jsonl", timeout=4 * 3600)
    return hyb, rr


def step_gate(hyb: dict, rr: dict) -> bool:
    ref_hyb, ref_rr = _load_refs()  # written by ref_v2_driver.sh; raises if absent
    ok = True
    h_ndcg = hyb["metrics"]["ndcg10"]
    r_ndcg = rr["metrics"]["ndcg10"]
    O.log(f"GATE hybrid  nDCG {h_ndcg:.3f} vs ref {ref_hyb['ndcg10']:.3f} "
          f"(Δ{h_ndcg - ref_hyb['ndcg10']:+.3f})")
    O.log(f"GATE rerank  nDCG {r_ndcg:.3f} vs ref {ref_rr['ndcg10']:.3f} "
          f"(Δ{r_ndcg - ref_rr['ndcg10']:+.3f}, need +{GATE_MIN_GAIN})")
    if r_ndcg < ref_rr["ndcg10"] + GATE_MIN_GAIN:
        ok = False
    slices = {**rr["per_query_type"], **rr["per_language"]}
    for name, ref in ref_rr["slices"].items():
        if ref is None:
            continue
        ref_n, ref_r = ref
        got = slices.get(name, {})
        dn = got.get("ndcg10", 0) - ref_n
        dr = got.get("recall10", 0) - ref_r
        flag = "" if (dn >= -GATE_SLICE_TOL and dr >= -GATE_SLICE_TOL) else "  ⚠ REGRESSION"
        O.log(f"  slice {name:<17} ndcg Δ{dn:+.3f}  r10 Δ{dr:+.3f}{flag}")
        if flag:
            ok = False
    O.log(f"GATE verdict: {'PASS — pulling snapshot' if ok else 'FAIL — vectors stay on the pod'}")
    (OUT / "verdict.json").write_text(json.dumps(
        {"pass": ok, "hybrid": hyb["metrics"], "rerank": rr["metrics"],
         "rerank_slices": slices}, indent=1))
    return ok


def step_pull_and_restore(ip: str, port: int, expected_points: int) -> None:
    snap = OUT / f"{V2_COLLECTION}.snapshot"
    _retry(lambda: O.pull_file(f"/workspace/out/{V2_COLLECTION}.snapshot", snap, ip, port,
                               timeout=1800),
           attempts=12, delay=30, what="snapshot pull")
    O.run(["tar", "tf", str(snap)], timeout=600)  # integrity: a truncated tar fails here
    O.log(f"snapshot pulled + tar-verified ({snap.stat().st_size/1e9:.2f} GB)")
    from dotenv import dotenv_values
    key = dotenv_values(O.INGEST / ".env").get("QDRANT_API_KEY") or ""
    O.log(f"restoring into LOCAL Qdrant as `{V2_COLLECTION}`...")
    O.run(["curl", "-sf", "-X", "POST",
           f"http://127.0.0.1:6333/collections/{V2_COLLECTION}/snapshots/upload?priority=snapshot",
           "-H", f"api-key: {key}", "-F", f"snapshot=@{snap}"], timeout=7200)
    info = O.run(["curl", "-sf", "-H", f"api-key: {key}",
                  f"http://127.0.0.1:6333/collections/{V2_COLLECTION}"]).stdout.decode()
    pts = json.loads(info)["result"]["points_count"]
    O.log(f"restored `{V2_COLLECTION}`: points_count={pts} (pod reported {expected_points})")
    if expected_points and pts != expected_points:
        raise RuntimeError(f"restored count {pts} != pod-reported {expected_points}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not REF_FILE.exists():  # fail before spending pod money
        raise SystemExit(f"reference file {REF_FILE} missing — run ref_v2_driver.sh first")
    _load_refs()  # validate it parses now, not at gate time (after $ spent)
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, O._sig)
    signal.signal(signal.SIGTERM, O._sig)

    passphrase = step_package()
    pubkey = O.step_keypair()
    verdict = False
    try:
        O._pod_id, gpu_used, O._price = step_provision_cascade(pubkey)
        O._provisioned_at = time.time()
        O._ip, O._port = O.step_wait_ssh(O._pod_id)
        O.ensure_pod_tools(O._ip, O._port)
        _retry(lambda: O.push_file(O.WORKDIR / "payload_reembed.tar.enc",
                                   "/workspace/payload.tar.enc", O._ip, O._port, timeout=14400),
               what="payload push")
        _retry(lambda: O.push_file(EMBED_SH, "/workspace/runpod_reembed_v2.sh",
                                   O._ip, O._port, timeout=120), what="embed script push")
        step_launch(O._ip, O._port, passphrase)
        # deadline = single-GPU wall-clock / actual shard count, + 2h fixed buffer. The buffer is
        # GPU-count-independent overhead (fresh torch-cu124 install + 4x model download ~40min +
        # count-settle up to 10min); the +1h buffer starved finalization on the 4x3090 run (embed
        # ate the whole 3h deadline, snapshot never ran, pod wiped). DONE now lands before the
        # snapshot too, so this only has to cover setup+embed+verify.
        deadline = DEADLINE_1GPU_S.get(gpu_used, DEADLINE_1GPU_DEFAULT) / ACTUAL_GPUS + 7200
        points = step_poll(O._ip, O._port, deadline, O._price or 0.0, O._provisioned_at)
        step_tunnel(O._ip, O._port)
        step_start_rerank_server(O._ip, O._port)
        hyb, rr = step_eval()
        verdict = step_gate(hyb, rr)
        if verdict and os.environ.get("REEMBED_SKIP_PULL") == "1":
            O.log("gate PASS — but REEMBED_SKIP_PULL=1: verdict recorded, SKIPPING the ~24GB pull "
                  "(insufficient balance to adopt now; top up then pull separately)")
        elif verdict:
            step_pull_and_restore(O._ip, O._port, points)
    finally:
        _kill_tunnel()
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
                O.log(f"COST: up={hrs*60:.1f}min price=${O._price}/hr est=${hrs*(O._price or 0):.2f}")
    O.log(f"I6 REEMBED COMPLETE — gate {'PASS (v2 restored locally)' if verdict else 'FAIL (recorded, nothing restored)'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "terminate":
        pid_file = O.WORKDIR / "pod.id"
        if pid_file.exists():
            O.terminate(pid_file.read_text().strip())
        else:
            O.log("no pod.id to terminate")
        sys.exit(0)
    main()
