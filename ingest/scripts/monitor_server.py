#!/usr/bin/env python3
"""Tiny local live dashboard for the RunPod full-corpus embed.

READ-ONLY: it only runs curl/nvidia-smi/ps/cat on the pod over SSH — it never touches the
embed process. Run it, then open the URL it prints:

    .venv/bin/python scripts/monitor_server.py
    # → http://localhost:8765

Set ``MON_POD_IP`` and ``MON_POD_PORT`` to the active pod before starting the monitor.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

POD_IP = os.environ.get("MON_POD_IP", "")
POD_PORT = os.environ.get("MON_POD_PORT", "")
KEY = str(Path.home() / "gpu_embed_work" / "id_ed25519")
KH = str(Path.home() / "gpu_embed_work" / "known_hosts")
PORT = int(os.environ.get("MON_PORT", "8765"))
POLL_S = 15

# Total docs per source (clean snapshot v1) — the accurate denominator for progress.
SRC_TOTALS = {"matsne": 132787, "napr": 25210, "ecd": 21827, "constcourt": 3069,
              "tas": 1344, "tbappeal": 81}
TOTAL_DOCS = sum(SRC_TOTALS.values())  # 184318

SSH = ["ssh", "-p", POD_PORT, "-i", KEY, "-o", "IdentitiesOnly=yes",
       "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={KH}",
       "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15"]

REMOTE = r"""
curl -sf http://127.0.0.1:6333/collections/georgian_legal 2>/dev/null | grep -oE '"points_count":[0-9]+' | head -1 | sed 's/.*://;s/^/CHUNKS:/'
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1 | sed 's/^/GPU:/'
echo "ALIVE:$(ps aux | grep -cE '[r]unpod_embed')"
echo "DONE:$(test -f /workspace/out/DONE && cat /workspace/out/DONE || echo no)"
for f in /workspace/ingest/.state/*.embed.json; do [ -f "$f" ] && s=$(basename "$f" .embed.json) && echo "SRC:${s%%.shard*}:$(grep -oE '"docs": [0-9]+' "$f" | grep -oE '[0-9]+')"; done 2>/dev/null
"""

_lock = threading.Lock()
_state: dict = {"chunks": 0, "gpu_util": None, "gpu_mem": None, "gpu_memtot": None,
                "alive": True, "done": None, "sources": {}, "ts": 0.0, "error": None,
                "history": []}  # history: list of [ts, chunks]


def _poll_once() -> None:
    r = subprocess.run(SSH + [f"root@{POD_IP}", "bash -s"], input=REMOTE.encode(),
                       capture_output=True, timeout=25)
    out = r.stdout.decode(errors="replace")
    chunks, gpu_util, gpu_mem, gpu_memtot, alive, done = 0, None, None, None, True, None
    sources: dict = {}
    for line in out.splitlines():
        if line.startswith("CHUNKS:"):
            v = line[7:].strip()
            chunks = int(v) if v.isdigit() else 0
        elif line.startswith("GPU:"):
            parts = [p.strip() for p in line[4:].split(",")]
            m = re.findall(r"\d+", " ".join(parts))
            if len(m) >= 3:
                gpu_util, gpu_mem, gpu_memtot = int(m[0]), int(m[1]), int(m[2])
        elif line.startswith("ALIVE:"):
            alive = line[6:].strip() not in ("0", "")
        elif line.startswith("DONE:"):
            done = None if line[5:].strip() == "no" else line[5:].strip()
        elif line.startswith("SRC:"):
            try:
                _, name, docs = line.split(":", 2)
                if docs.strip().isdigit():  # sum shard checkpoints back into their source
                    sources[name] = sources.get(name, 0) + int(docs)
            except ValueError:
                pass
    now = time.time()
    with _lock:
        _state.update(chunks=chunks, gpu_util=gpu_util, gpu_mem=gpu_mem, gpu_memtot=gpu_memtot,
                      alive=alive, done=done, sources=sources, ts=now, error=None)
        _state["history"].append([now, chunks])
        _state["history"] = _state["history"][-240:]  # ~1h of 15s samples


def poll_loop() -> None:
    while True:
        try:
            _poll_once()
        except Exception as e:  # noqa: BLE001
            with _lock:
                _state["error"] = f"{type(e).__name__}: {e}"
        time.sleep(POLL_S)


def _computed() -> dict:
    with _lock:
        s = dict(_state)
        hist = list(_state["history"])
    total_docs = sum(s["sources"].values())
    doc_pct = 100.0 * total_docs / TOTAL_DOCS if TOTAL_DOCS else 0.0
    # rate over the last <=10 min of history
    rate = 0.0
    if len(hist) >= 2:
        window = [h for h in hist if h[0] >= hist[-1][0] - 600] or hist[-2:]
        dt = window[-1][0] - window[0][0]
        if dt > 0:
            rate = (window[-1][1] - window[0][1]) / dt
    # self-calibrating chunk target from docs progress
    est_total_chunks = int(s["chunks"] / (doc_pct / 100)) if doc_pct > 1 else None
    eta_min = None
    if rate > 0 and est_total_chunks:
        eta_min = max(0, (est_total_chunks - s["chunks"]) / rate / 60)
    spark = [h[1] for h in hist[-60:]]
    return {**s, "total_docs": total_docs, "total_docs_target": TOTAL_DOCS,
            "doc_pct": round(doc_pct, 2), "rate_cps": round(rate, 1),
            "est_total_chunks": est_total_chunks, "eta_min": round(eta_min, 1) if eta_min else None,
            "src_totals": SRC_TOTALS, "spark": spark, "age_s": round(time.time() - s["ts"], 1)}


PAGE = """<!doctype html><html><head><meta charset=utf-8>
<title>Embed monitor</title><meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:#0b0f17;color:#e6edf3}
.wrap{max-width:820px;margin:0 auto;padding:22px 18px}
h1{font-size:19px;margin:0 0 2px;font-weight:600}
.sub{color:#7d8590;font-size:13px;margin-bottom:18px}
.card{background:#11161f;border:1px solid #222b38;border-radius:12px;padding:18px;margin-bottom:14px}
.big{font-size:40px;font-weight:700;letter-spacing:-1px}
.big .u{font-size:16px;color:#7d8590;font-weight:500;margin-left:6px}
.row{display:flex;flex-wrap:wrap;gap:14px}
.row .card{flex:1;min-width:150px;margin-bottom:0}
.lbl{color:#7d8590;font-size:12px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}
.bar{height:12px;background:#1b2230;border-radius:6px;overflow:hidden;margin-top:10px}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,#2f81f7,#3fb950);transition:width .6s}
.srcbar{height:8px;background:#1b2230;border-radius:4px;overflow:hidden;margin-top:4px}
.srcbar>i{display:block;height:100%;background:#2f81f7}
.srcrow{margin:10px 0}
.srcrow .t{display:flex;justify-content:space-between;font-size:13px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:1px}
.ok{background:#3fb950}.warn{background:#d29922}.bad{background:#f85149}
svg{width:100%;height:44px;display:block}
.muted{color:#7d8590;font-size:12px}
a{color:#2f81f7}
</style></head><body><div class=wrap>
<h1>Georgian Legal — GPU embed monitor</h1>
<div class=sub id=status>connecting…</div>

<div class=card>
  <div class=lbl>Documents embedded</div>
  <div class=big><span id=docs>–</span><span class=u>/ 184,318 docs</span></div>
  <div class=bar><i id=docbar style=width:0%></i></div>
  <div class=muted style=margin-top:8px><span id=docpct>–</span>% · <span id=chunks>–</span> chunks · <span id=rate>–</span> chunks/sec · ETA <span id=eta>–</span></div>
  <svg id=spark viewBox="0 0 300 44" preserveAspectRatio=none><polyline id=sparkline fill=none stroke=#3fb950 stroke-width=2 points=""/></svg>
</div>

<div class=row>
  <div class=card><div class=lbl>GPU utilization</div><div class=big><span id=gpu>–</span><span class=u>%</span></div></div>
  <div class=card><div class=lbl>GPU memory</div><div class=big style=font-size:26px><span id=gmem>–</span></div></div>
  <div class=card><div class=lbl>Status</div><div class=big style=font-size:22px id=st>–</div></div>
</div>

<div class=card>
  <div class=lbl>Per source</div>
  <div id=sources></div>
</div>
<div class=muted>Auto-updates every 4s · data age <span id=age>–</span>s · read-only (does not affect the embed)</div>
</div>
<script>
const TOT={matsne:132787,napr:25210,ecd:21827,constcourt:3069,tas:1344,tbappeal:81};
const fmt=n=>n==null?'–':n.toLocaleString();
async function tick(){
 let d;try{d=await (await fetch('/data',{cache:'no-store'})).json()}catch(e){document.getElementById('status').textContent='monitor unreachable';return}
 const done=d.done, dead=!d.alive&&!done;
 document.getElementById('docs').textContent=fmt(d.total_docs);
 document.getElementById('docpct').textContent=d.doc_pct;
 document.getElementById('docbar').style.width=Math.min(100,d.doc_pct)+'%';
 document.getElementById('chunks').textContent=fmt(d.chunks);
 document.getElementById('rate').textContent=d.rate_cps;
 document.getElementById('eta').textContent=done?'done':(dead?'—':(d.eta_min!=null?d.eta_min+' min':'…'));
 document.getElementById('gpu').textContent=d.gpu_util==null?'–':d.gpu_util;
 document.getElementById('gmem').textContent=d.gpu_mem==null?'–':(fmt(d.gpu_mem)+' / '+fmt(d.gpu_memtot)+' MiB');
 document.getElementById('age').textContent=d.age_s;
 const st=document.getElementById('st'), stt=document.getElementById('status');
 if(done){st.innerHTML='<span class="dot ok"></span>DONE';stt.textContent='Embed complete — '+fmt(+done)+' chunks. The pipeline will verify + restore locally.';}
 else if(dead){st.innerHTML='<span class="dot bad"></span>stopped';stt.textContent='Embed process not running (no DONE marker) — may need a resume.';}
 else{st.innerHTML='<span class="dot ok"></span>embedding';stt.textContent='Live · pod RTX 4090 · updated '+d.age_s+'s ago'+(d.error?(' · last poll error: '+d.error):'');}
 // sources
 let h='';for(const k of Object.keys(TOT)){const done=(d.sources&&d.sources[k])||0;const p=Math.min(100,100*done/TOT[k]);
   h+=`<div class=srcrow><div class=t><span>${k}</span><span>${fmt(done)} / ${fmt(TOT[k])}</span></div><div class=srcbar><i style=width:${p}%></i></div></div>`;}
 document.getElementById('sources').innerHTML=h;
 // sparkline
 const sp=d.spark||[];if(sp.length>1){const mn=Math.min(...sp),mx=Math.max(...sp),r=(mx-mn)||1;
   const pts=sp.map((v,i)=>`${(300*i/(sp.length-1)).toFixed(1)},${(42-40*(v-mn)/r).toFixed(1)}`).join(' ');
   document.getElementById('sparkline').setAttribute('points',pts);}
}
tick();setInterval(tick,4000);
</script></body></html>"""


_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _host_allowed(self) -> bool:
        # Loopback-bound, but a DNS-rebinding page can still read /data cross-origin; require a
        # loopback Host header. Real local browser access sends Host: localhost:PORT.
        host = (self.headers.get("Host") or "").strip().lower()
        if host.startswith("["):
            hostname = host[: host.find("]") + 1] if "]" in host else host
        else:
            hostname = host.rsplit(":", 1)[0] if ":" in host else host
        return hostname in _ALLOWED_HOSTS

    def do_GET(self):
        if not self._host_allowed():
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/data"):
            body = json.dumps(_computed()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> None:
    if not POD_IP or not POD_PORT:
        raise SystemExit("Set MON_POD_IP and MON_POD_PORT before starting the monitor")
    threading.Thread(target=poll_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Embed monitor → http://localhost:{PORT}   (Ctrl-C to stop; read-only, safe)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
