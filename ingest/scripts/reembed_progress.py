#!/usr/bin/env python
"""Poll the re-embed pod's Qdrant point count → ``$GPU_WORKDIR/reembed_progress.json``.

Runs alongside (not inside) runpod_orchestrate_reembed.py so the dashboard can show live
"X / TOTAL embedded (Y%)" during the ~3h GPU re-embed. Reuses the orchestrator's RunPod/SSH
helpers; reads the pod id from ``$GPU_WORKDIR/pod.id``. Exits when the pod is gone or the
collection is complete.

    .venv/bin/python scripts/reembed_progress.py --total 2654818 --collection georgian_legal_v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runpod_orchestrate as O  # noqa: E402

OUT = O.WORKDIR / "reembed_progress.json"


def _count(ip: str, port: int, collection: str) -> int | None:
    try:
        out = O.ssh_capture(
            ip, port,
            f"curl -sf http://127.0.0.1:6333/collections/{collection} "
            "| python3 -c 'import sys,json; print(json.load(sys.stdin)[\"result\"][\"points_count\"])'",
            timeout=30).strip()
        return int(out) if out.isdigit() else None
    except Exception:  # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--collection", default="georgian_legal_v2")
    ap.add_argument("--interval", type=int, default=45)
    args = ap.parse_args()

    pid_file = O.WORKDIR / "pod.id"
    ip = port = None
    # Resolve the pod's SSH endpoint (retry: ports appear a bit after provisioning).
    for _ in range(40):
        if not pid_file.exists():
            time.sleep(10)
            continue
        pod_id = pid_file.read_text().strip()
        q = ("query($id:String!){ pod(input:{podId:$id}){ runtime{ ports{ ip isIpPublic "
             "privatePort publicPort type } } } }")
        try:
            pod = O.gql(q, {"id": pod_id}).get("pod") or {}
            rt = pod.get("runtime") or {}
            for p in rt.get("ports") or []:
                if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("type") == "tcp":
                    ip, port = p["ip"], p["publicPort"]
                    break
        except Exception:  # noqa: BLE001
            pass
        if ip:
            break
        time.sleep(15)
    if not ip:
        OUT.write_text(json.dumps({"error": "pod SSH endpoint not found"}))
        return

    seen = False   # has the v2 collection appeared yet? (it won't until embed starts)
    stale = 0
    while True:
        n = _count(ip, port, args.collection)
        if n is None:
            # Before the collection exists, only give up if the POD itself is gone; after
            # it has appeared, repeated None means embed finished and the pod was torn down.
            pod_alive = O.ssh_ok(ip, port, "true", timeout=20)
            if not pod_alive:
                stale += 1
                if stale > 5:
                    break
            elif not seen:
                OUT.write_text(json.dumps({
                    "embedded": 0, "total": args.total, "remaining": args.total, "pct": 0,
                    "collection": args.collection, "stage": "uploading / starting embed",
                    "ts": time.strftime("%H:%M:%S", time.gmtime())}))
        else:
            seen = True
            stale = 0
            pct = round(100 * n / args.total, 1) if args.total else 0
            OUT.write_text(json.dumps({
                "embedded": n, "total": args.total, "remaining": max(0, args.total - n),
                "pct": pct, "collection": args.collection,
                "ts": time.strftime("%H:%M:%S", time.gmtime()),
            }))
            if n >= args.total:
                break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
