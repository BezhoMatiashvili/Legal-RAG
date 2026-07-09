#!/usr/bin/env python
"""Stand up the cross-encoder reranker on a RunPod GPU as an HTTP service over an SSH tunnel,
and ALWAYS terminate the pod on exit. Lets the Phase-C rerank sweep offload scoring to GPU while
retrieval + metrics stay local (set RERANK_REMOTE_URL=http://localhost:8900 for the eval runs).

Transport is the SSH tunnel (encrypted); the pod is wiped by termination. Reuses the embed
orchestrator's RunPod/SSH helpers (runpod_orchestrate.py).

    python scripts/runpod_rerank.py up     # provision + serve; blocks holding the tunnel;
                                           #   writes ~/gpu_embed_work/rerank_ready with the URL
    python scripts/runpod_rerank.py down   # emergency terminate (reads ~/gpu_embed_work/pod.id)

To stop cleanly: `touch ~/gpu_embed_work/rerank_stop` (or SIGTERM the `up` process).
"""

import atexit
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_orchestrate as O  # noqa: E402  — reuse gql/ssh/provision/terminate helpers

PORT = 8900
READY = O.WORKDIR / "rerank_ready"
STOP = O.WORKDIR / "rerank_stop"
SERVER_LOCAL = Path(__file__).resolve().parent / "runpod_rerank_server.py"
MAX_UPTIME_S = 120 * 60  # hard watchdog — never bill past this even if the caller vanishes

_tunnel: subprocess.Popen | None = None


def _terminate() -> None:
    global _tunnel
    try:
        if _tunnel and _tunnel.poll() is None:
            _tunnel.terminate()
    except Exception:  # noqa: BLE001
        pass
    O.terminate(O._pod_id)  # idempotent (guarded by O._terminated)


def _sig(signum, frame):  # noqa: ANN001
    O.log(f"rerank orchestrator: signal {signum}")
    _terminate()
    sys.exit(0)


def _dep_check(ip: str, port: int) -> str:
    return O.ssh_capture(
        ip, port,
        "python -c 'import torch;from transformers import AutoModelForSequenceClassification,"
        "AutoTokenizer;print(\"DEPS_OK cuda=\"+str(torch.cuda.is_available()))' 2>&1",
        timeout=180).strip()


def setup_pod(ip: str, port: int) -> None:
    O.ensure_pod_tools(ip, port)  # curl etc. (used for on-pod health check)
    O.push_content(ip, port, "/workspace/rerank_server.py", SERVER_LOCAL.read_text(), mode="644")
    out = _dep_check(ip, port)
    O.log(f"pod deps (image transformers): {out[-160:]}")
    if "DEPS_OK cuda=True" not in out:
        # RunPod image's pre-installed pkgs can break transformers lazy imports (old peft, etc.).
        O.log("fixing pod deps (drop peft; install transformers stack)...")
        O.ssh_capture(ip, port,
                      "pip -q uninstall -y peft 2>/dev/null; "
                      "pip -q install -U transformers tokenizers safetensors sentencepiece "
                      "2>&1 | tail -3", timeout=900)
        out = _dep_check(ip, port)
        O.log(f"pod deps (after fix): {out[-160:]}")
        if "DEPS_OK cuda=True" not in out:
            raise RuntimeError(f"pod dependency check failed:\n{out[-1200:]}")
    # Launch detached via a script + setsid — the proven pattern from step_launch: redirect ALL
    # fds off the ssh channel (</dev/null, >log 2>&1) so the launch call returns immediately
    # instead of the ssh session hanging on the backgrounded process.
    launch_sh = (
        "#!/usr/bin/env bash\n"
        "cd /workspace\n"
        f"export RERANK_MODEL=BAAI/bge-reranker-v2-m3 PORT={PORT}\n"
        "exec python rerank_server.py\n"
    )
    O.push_content(ip, port, "/workspace/launch_rerank.sh", launch_sh, mode="755")
    out = O.ssh_capture(
        ip, port,
        "setsid bash /workspace/launch_rerank.sh >/workspace/rerank.log 2>&1 </dev/null & echo LAUNCHED",
        timeout=60)
    if "LAUNCHED" not in out:
        raise RuntimeError(f"failed to launch rerank server (setsid): {out!r}")
    O.log("waiting for pod rerank server (model download + load)...")
    for _ in range(90):  # up to ~15 min
        h = O.ssh_capture(ip, port, f"curl -s -m 5 localhost:{PORT}/health 2>/dev/null", timeout=20)
        if '"ok": true' in h or '"ok":true' in h:
            O.log(f"pod rerank server healthy: {h.strip()[:120]}")
            return
        time.sleep(10)
    tail = O.ssh_capture(ip, port, "tail -20 /workspace/rerank.log 2>/dev/null", timeout=30)
    raise RuntimeError(f"pod rerank server never became healthy. log tail:\n{tail}")


def open_tunnel(ip: str, port: int) -> None:
    global _tunnel
    cmd = O._ssh_base(ip, port) + ["-N", "-L", f"{PORT}:localhost:{PORT}", f"root@{ip}"]
    _tunnel = subprocess.Popen(cmd)
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    O.log(f"SSH tunnel up: http://localhost:{PORT} -> pod:{PORT}")
                    return
        except Exception:  # noqa: BLE001
            time.sleep(2)
    raise RuntimeError("SSH tunnel health check failed")


def up() -> None:
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    atexit.register(_terminate)
    READY.unlink(missing_ok=True)
    STOP.unlink(missing_ok=True)
    try:
        pubkey = O.step_keypair()
        O._pod_id, gpu, price = O.step_provision(pubkey)
        O._provisioned_at = time.time()
        ip, port = O.step_wait_ssh(O._pod_id)
        O._ip, O._port = ip, port
        setup_pod(ip, port)
        open_tunnel(ip, port)
        READY.write_text(f"http://localhost:{PORT}\n")
        O.log(f"READY  http://localhost:{PORT}  (pod {O._pod_id} {gpu} ${price}/hr) — "
              f"set RERANK_REMOTE_URL and run the sweep; `touch {STOP}` to stop")
        started = time.time()
        while True:
            if STOP.exists():
                O.log("stop-file seen — terminating pod")
                break
            if time.time() - started > MAX_UPTIME_S:
                O.log("watchdog: max uptime reached — terminating pod")
                break
            if _tunnel is None or _tunnel.poll() is not None:
                O.log("tunnel down — reopening")
                open_tunnel(ip, port)
            time.sleep(15)
    finally:
        _terminate()
        READY.unlink(missing_ok=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "up"
    if cmd == "down":
        pid_file = O.WORKDIR / "pod.id"
        O.terminate(pid_file.read_text().strip() if pid_file.exists() else None)
    elif cmd == "up":
        up()
    else:
        sys.exit(f"usage: {sys.argv[0]} [up|down]")
