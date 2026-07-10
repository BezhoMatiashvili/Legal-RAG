#!/usr/bin/env python3
"""Live monitor for Claude Code sessions on this box + the RAG stack's health.

A browser can't read your session transcripts directly (file:// is sandboxed), so this
tiny **stdlib-only** server reads them for you and serves a live dashboard.

    python scripts/session_monitor.py            # serves http://localhost:8770
    python scripts/session_monitor.py 9000       # custom port

Then open the URL. It shows, refreshing every few seconds:
  * every recent Claude Code session for this project — title (slug), git branch, whether it's
    WORKING / WAITING-FOR-YOU / IDLE, seconds since last activity, its live to-do list
    (reconstructed from TaskCreate/TaskUpdate), and a feed of its last tool calls;
  * system health — RAM + swap gauges (the thing that's been killing us), load, top RAM hogs;
  * the RAG stack — Qdrant RAM, how many MCP servers are running, the GPU rerank pod's health,
    and the most recent legal_search latencies from the query log.

Reads only; never writes. Binds to loopback (not exposed). No third-party deps.
"""

from __future__ import annotations

import calendar
import json
import re
import subprocess
import sys
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# --- Locations (this project) -------------------------------------------------
REPO = Path("/home/bezhomatiashvili/Desktop/Projects/Georgia-Legal-Search")
_ESCAPED = str(REPO).replace("/", "-")          # ~/.claude/projects/<escaped-cwd>/
PROJECT_DIR = Path.home() / ".claude" / "projects" / _ESCAPED
QUERY_LOG = REPO / "ingest" / ".state" / "queries.jsonl"
POD_HEALTH_URL = "http://localhost:8900/health"
HTML_FILE = Path(__file__).resolve().parent / "session_monitor.html"
GPU_WORKDIR = Path.home() / "gpu_embed_work"    # delta_orch*.log + ssh key live here
ENV_FILE = REPO / "ingest" / ".env"
RUNPOD_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
             "Chrome/125.0.0.0 Safari/537.36")

SESSION_MAX_AGE_S = 6 * 3600     # hide transcripts older than this
WORKING_AGE_S = 45               # < this since last write ⇒ actively working
IDLE_AGE_S = 30 * 60             # > this ⇒ idle regardless of last action

# --- tiny time-based caches (avoid re-parsing / re-shelling on every poll) -----
_session_cache: dict[str, tuple[float, dict]] = {}   # path -> (mtime, parsed)
_pod_cache: tuple[float, dict] | None = None          # (checked_at, result)
_ps_cache: tuple[float, list] | None = None           # (checked_at, rows)


def _tool_label(name: str, inp: dict) -> str:
    inp = inp or {}
    if name == "Bash":
        return inp.get("description") or (inp.get("command", "")[:60])
    if name in ("Read", "Edit", "Write", "NotebookEdit"):
        return f"{name} {Path(inp.get('file_path', '')).name}"
    if name == "TaskCreate":
        return f"+ task: {inp.get('subject', '')[:44]}"
    if name == "TaskUpdate":
        return f"task {inp.get('taskId', '?')} → {inp.get('status', '?')}"
    if name == "AskUserQuestion":
        return "asked you a question"
    if name in ("Grep", "Glob"):
        return f"{name} {inp.get('pattern', '')[:30]}"
    if name == "Workflow":
        return "launched a workflow"
    if name == "Agent":
        return f"spawned agent: {inp.get('description', '')[:34]}"
    if name.startswith("mcp__"):
        return name.split("__")[-1]
    return name


def parse_session(path: Path) -> dict:
    """Extract title, activity, to-do list and recent tool feed from a session .jsonl."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    cached = _session_cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]

    lines: list[dict] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    try:
                        lines.append(json.loads(ln))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return {}
    if not lines:
        return {}

    last = lines[-1]
    meta = next((d for d in reversed(lines) if d.get("slug")), last)
    slug = meta.get("slug") or "(untitled)"
    branch = meta.get("gitBranch") or "?"

    # Replay TaskCreate/TaskUpdate into the current to-do list.
    tasks: list[dict] = []
    feed: list[dict] = []
    last_text = ""
    for d in lines:
        ts = d.get("timestamp")
        content = (d.get("message") or {}).get("content")
        if not isinstance(content, list):
            if isinstance(content, str) and d.get("type") == "assistant" and content.strip():
                last_text = content.strip()
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and d.get("type") == "assistant" and b.get("text", "").strip():
                last_text = b["text"].strip()
            elif b.get("type") == "tool_use":
                name, inp = b.get("name", "?"), b.get("input") or {}
                if name == "TaskCreate":
                    tasks.append({"id": len(tasks) + 1, "subject": inp.get("subject", ""),
                                  "status": "pending"})
                elif name == "TaskUpdate":
                    tid, st = str(inp.get("taskId", "")), inp.get("status")
                    for t in tasks:
                        if str(t["id"]) == tid and st:
                            t["status"] = st
                feed.append({"t": ts, "label": _tool_label(name, inp), "tool": name})

    tasks = [t for t in tasks if t["status"] != "deleted"]
    done = sum(1 for t in tasks if t["status"] == "completed")

    lt = last.get("type")
    lc = (last.get("message") or {}).get("content")
    has_tool = isinstance(lc, list) and any(
        isinstance(x, dict) and x.get("type") == "tool_use" for x in lc)
    waiting = (lt == "assistant" and not has_tool)

    age = time.time() - mtime
    if age > IDLE_AGE_S:
        status = "idle"
    elif waiting:
        status = "waiting"
    elif age < WORKING_AGE_S or not waiting:
        status = "working"
    else:
        status = "idle"

    parsed = {
        "id": path.stem,
        "short": path.stem[:8],
        "slug": slug,
        "branch": branch,
        "status": status,
        "age_s": round(age),
        "mtime": mtime,
        "messages": len(lines),
        "last_text": last_text[:240],
        "tasks": tasks,
        "task_done": done,
        "task_total": len(tasks),
        "feed": feed[-8:][::-1],   # most recent first
    }
    _session_cache[str(path)] = (mtime, parsed)
    return parsed


def sessions_state() -> list[dict]:
    if not PROJECT_DIR.is_dir():
        return []
    out = []
    now = time.time()
    for f in PROJECT_DIR.glob("*.jsonl"):
        try:
            if now - f.stat().st_mtime > SESSION_MAX_AGE_S:
                continue
        except OSError:
            continue
        p = parse_session(f)
        if p:
            out.append(p)
    out.sort(key=lambda s: s["mtime"], reverse=True)
    return out[:10]


def _ps_rows() -> list[dict]:
    global _ps_cache
    if _ps_cache and time.time() - _ps_cache[0] < 2.0:
        return _ps_cache[1]
    rows: list[dict] = []
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,ppid,pcpu,rss,etime,args", "--no-headers", "--sort=-rss"],
            capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            parts = line.split(None, 5)
            if len(parts) < 6:
                continue
            pid, ppid, cpu, rss, etime, args = parts
            rows.append({"pid": int(pid), "ppid": int(ppid), "cpu": float(cpu),
                         "rss_mb": int(rss) / 1024, "etime": etime, "args": args})
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    _ps_cache = (time.time(), rows)
    return rows


def processes_state() -> list[dict]:
    """Every non-kernel process, RSS-sorted — the full 'what is actually running' view."""
    out = []
    for r in _ps_rows():
        args = r["args"]
        if not args or args.startswith("["):   # skip kernel threads
            continue
        out.append({"pid": r["pid"], "ppid": r.get("ppid"), "cpu": r["cpu"],
                    "rss_mb": round(r["rss_mb"]), "etime": r.get("etime", ""),
                    "cmd": args[:160], "name": _short_name(args)})
    return out[:300]


def _short_name(args: str) -> str:
    if "ingest.mcp_server" in args:
        return "mcp_server (legal_rag)"
    if "qdrant" in args.lower():
        return "qdrant"
    if "session_monitor" in args:
        return "session_monitor"
    if "runpod_rerank" in args:
        return "runpod pod tunnel"
    if "eval.evaluate" in args:
        return "eval.evaluate"
    base = args.split()[0].split("/")[-1] if args.split() else args
    return base[:24]


def system_state() -> dict:
    mem: dict[str, int] = {}
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = ln.partition(":")
            mem[k.strip()] = int(v.strip().split()[0])  # kB
    except OSError:
        pass
    total = mem.get("MemTotal", 1)
    avail = mem.get("MemAvailable", 0)
    used = total - avail
    stot = mem.get("SwapTotal", 0)
    sfree = mem.get("SwapFree", 0)
    sused = stot - sfree
    try:
        load = Path("/proc/loadavg").read_text().split()[0]
    except OSError:
        load = "?"
    rows = _ps_rows()
    top = [{"name": _short_name(r["args"]), "rss_gb": round(r["rss_mb"] / 1024, 2),
            "cpu": r["cpu"], "pid": r["pid"]}
           for r in rows[:8]]
    return {
        "ram_used_gb": round(used / 1024 / 1024, 1),
        "ram_total_gb": round(total / 1024 / 1024, 1),
        "ram_pct": round(100 * used / total),
        "swap_used_gb": round(sused / 1024 / 1024, 1),
        "swap_total_gb": round(stot / 1024 / 1024, 1),
        "swap_pct": round(100 * sused / stot) if stot else 0,
        "load": load,
        "top": top,
    }


_sls_cache: tuple[float, dict] | None = None   # (checked_at, serverless endpoint state)


def _serverless_health() -> dict:
    """Serverless endpoint state via GET /v2/{id}/health — cheap and never wakes (= never
    bills) a worker. Cached 30 s. Adds the publish manifest so the card shows what data
    the worker is expected to serve."""
    global _sls_cache
    if _sls_cache and time.time() - _sls_cache[0] < 30.0:
        return _sls_cache[1]
    result: dict = {"configured": False}
    eid, key = _env_value("RUNPOD_ENDPOINT_ID"), _env_value("RUNPOD_API_KEY")
    if eid and key:
        result = {"configured": True, "up": False,
                  "backend": _env_value("SEARCH_BACKEND") or "local"}
        try:
            req = urllib.request.Request(
                f"https://api.runpod.ai/v2/{eid}/health",
                headers={"Authorization": f"Bearer {key}", "User-Agent": RUNPOD_UA})
            with urllib.request.urlopen(req, timeout=5) as r:
                d = json.loads(r.read().decode())
            result.update(up=True, workers=d.get("workers") or {}, jobs=d.get("jobs") or {})
        except Exception:  # noqa: BLE001 - endpoint absent/unreachable is a normal state
            pass
        try:
            m = json.loads((REPO / "ingest" / ".state" / "publish" / "manifest.json")
                           .read_text(encoding="utf-8"))
            result["published"] = {"points": m.get("points_count"),
                                   "at": m.get("created_at")}
        except (OSError, json.JSONDecodeError):
            pass
    _sls_cache = (time.time(), result)
    return result


def _pod_health() -> dict:
    global _pod_cache
    if _pod_cache and time.time() - _pod_cache[0] < 5.0:
        return _pod_cache[1]
    result = {"up": False}
    try:
        with urllib.request.urlopen(POD_HEALTH_URL, timeout=1.5) as r:
            d = json.loads(r.read().decode())
            result = {"up": bool(d.get("ok")), "device": d.get("device"),
                      "model": d.get("model")}
    except Exception:  # noqa: BLE001 - pod down / no tunnel is a normal state
        pass
    _pod_cache = (time.time(), result)
    return result


def rag_state() -> dict:
    rows = _ps_rows()
    mcp = [r for r in rows if "ingest.mcp_server" in r["args"] and "-m ingest.mcp_server" in r["args"]]
    qdrant = next((r for r in rows if r["args"].strip().endswith("qdrant") or "/qdrant" in r["args"]), None)
    queries = []
    try:
        tail = QUERY_LOG.read_text(encoding="utf-8").splitlines()[-5:]
        for ln in tail:
            try:
                d = json.loads(ln)
                queries.append({"latency_s": round(d.get("latency_ms", 0) / 1000, 1),
                                "route": d.get("route"),
                                "q": (d.get("query") or "")[:42]})
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return {
        "pod": _pod_health(),
        "serverless": _serverless_health(),
        "mcp_count": len(mcp),
        "mcp_rss_gb": round(sum(r["rss_mb"] for r in mcp) / 1024, 2),
        "mcp_cpu": round(sum(r["cpu"] for r in mcp)),
        "qdrant_gb": round(qdrant["rss_mb"] / 1024, 1) if qdrant else None,
        "queries": queries[::-1],
    }


# --- Qdrant / embedding-coverage panel -------------------------------------------
_qdrant_cache: tuple[float, dict] | None = None    # (checked_at, collections state)


def _qdrant_get(path: str) -> dict | None:
    """GET against the local Qdrant REST API with the .env api-key (short timeout)."""
    base = (_env_value("QDRANT_URL") or "http://localhost:6333").rstrip("/")
    headers = {}
    key = _env_value("QDRANT_API_KEY")
    if key:
        headers["api-key"] = key
    try:
        req = urllib.request.Request(base + path, headers=headers)
        with urllib.request.urlopen(req, timeout=4) as r:
            return json.loads(r.read().decode()).get("result")
    except Exception:  # noqa: BLE001 - qdrant busy/down is a normal state
        return None


def qdrant_state() -> dict:
    """Live point counts + index status for the main and delta collections (cached 10 s)."""
    global _qdrant_cache
    if _qdrant_cache and time.time() - _qdrant_cache[0] < 10.0:
        return _qdrant_cache[1]
    st: dict = {"up": False}
    main = _qdrant_get("/collections/georgian_legal")
    if main:
        st = {"up": True,
              "points": main.get("points_count"),
              "status": main.get("status"),
              "indexed_vectors": main.get("indexed_vectors_count")}
        delta = _qdrant_get("/collections/georgian_legal_delta")
        if delta:
            st["delta_points"] = delta.get("points_count")
            st["delta_status"] = delta.get("status")
    _qdrant_cache = (time.time(), st)
    return st


_BATCH2_TAIL_RE = re.compile(r"tail to embed: (\d+) docs")


def _tail_line(path: Path) -> str | None:
    """Last non-empty line of a log (reads only the final 4 KB)."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 4096))
            lines = [ln for ln in fh.read().decode(errors="replace").splitlines() if ln.strip()]
        return lines[-1][:220] if lines else None
    except OSError:
        return None


def _script_alive(script: str) -> bool:
    """True iff a real python invocation of ``script`` is running.

    Bare substring matching on ps args also hits editors, `tail -f`, greps and shell
    wrappers whose command string mentions the filename — only count processes whose
    executable is python and whose arguments (not argv[0]) name the script.
    """
    for r in _ps_rows():
        parts = r["args"].split()
        if len(parts) < 2:
            continue
        exe = parts[0].rsplit("/", 1)[-1]
        if exe.startswith("python") and any(script in p for p in parts[1:]):
            return True
    return False


def coverage_state() -> dict:
    """Last verify_all_embedded.py report + the batch-2 embed pipeline's current stage."""
    st: dict = {}
    try:
        st["report"] = json.loads(
            (REPO / "ingest" / ".state" / "embed_coverage.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st["report"] = None

    b2: dict = {"active": False}
    log = GPU_WORKDIR / "batch2.log"
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    if lines:
        b2 = {"active": True, "stage": "waiting for crawl", "tail_docs": None}
        for ln in lines:
            m = _BATCH2_TAIL_RE.search(ln)
            if m:
                b2["tail_docs"] = int(m.group(1))
                b2["stage"] = "embedding on GPU"
            elif "nothing new to embed" in ln:
                b2["stage"] = "complete (nothing to embed)"
            elif "orchestrator v3 FAILED" in ln:
                b2["stage"] = "FAILED (orchestrator)"
            elif "merging batch 2" in ln:
                b2["stage"] = "merging into georgian_legal"
            elif "BATCH 2 COMPLETE" in ln:
                b2["stage"] = "complete"
        b2["alive"] = _script_alive("run_batch2.py")
        if not b2["alive"] and b2["stage"] not in ("complete", "complete (nothing to embed)") \
                and not b2["stage"].startswith("FAILED"):
            b2["stage"] = "FAILED (driver exited)"
        # A recovery re-run supersedes a FAILED batch2.log verdict: report the live stage,
        # or "recovered" once a clean coverage report postdates the failure.
        if b2["stage"].startswith("FAILED"):
            if _script_alive("runpod_orchestrate_delta.py"):
                b2["stage"] = "re-running GPU embed (recovery)"
            elif _script_alive("merge_delta_collection.py"):
                b2["stage"] = "merging into georgian_legal (recovery)"
            else:
                rep = st.get("report") or {}
                try:
                    rep_t = calendar.timegm(time.strptime(
                        rep.get("generated_at", ""), "%Y-%m-%dT%H:%M:%SZ"))
                    if rep.get("total_missing") == 0 and rep_t > log.stat().st_mtime:
                        b2["stage"] = "complete (recovered)"
                except (ValueError, OSError):
                    pass
    st["batch2"] = b2

    st["verify_running"] = _script_alive("verify_all_embedded.py")
    # Live detail lines for the post-embed phases, so the dashboard tracks the whole
    # chain (embed → pull → restore → merge → verify) in real time, not just its name.
    st["merge_running"] = _script_alive("merge_delta_collection.py")
    if st["merge_running"]:
        logs = sorted(GPU_WORKDIR.glob("merge*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        st["merge_detail"] = _tail_line(logs[0]) if logs else None
    if st["verify_running"]:
        st["verify_detail"] = _tail_line(GPU_WORKDIR / "verify_final.log")
    return st


# --- GPU delta-embed panel ------------------------------------------------------
_upload_cache: tuple[float, int] | None = None    # (checked_at, remote_bytes)
_runpod_cache: tuple[float, dict] | None = None   # (checked_at, {balance, pods})
_delta_pts_cache: tuple[float, int] | None = None  # (checked_at, pod-side delta points)
_seen_cache: tuple[float, int] | None = None       # (checked_at, seen.sqlite rows)
CHUNKS_PER_DOC = 13.3                              # corpus average → embed-progress estimate
ADVERTISED_TOTAL = 156_800                         # matsne audit-A advertised doc count (2026-07-09)

_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z\]")


def _log_epoch(line: str) -> float | None:
    m = _TS_RE.match(line)
    if not m:
        return None
    return calendar.timegm(time.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S"))


def _env_value(key: str) -> str | None:
    """Read one KEY=VALUE from ingest/.env without third-party deps (never logged)."""
    try:
        for ln in ENV_FILE.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln.startswith(f"{key}="):
                return ln.split("=", 1)[1].split("#")[0].strip() or None
    except OSError:
        pass
    return None


def _runpod_account() -> dict:
    """Balance + open pods via GraphQL (browser UA — Cloudflare), cached 60 s."""
    global _runpod_cache
    if _runpod_cache and time.time() - _runpod_cache[0] < 60.0:
        return _runpod_cache[1]
    result: dict = {}
    key = _env_value("RUNPOD_API_KEY")
    if key:
        try:
            body = json.dumps({"query": "query{ myself{ clientBalance pods{ id costPerHr } } }"}).encode()
            # Bearer header, not ?api_key= — keeps the key out of URL/proxy logs (verified
            # live: GraphQL accepts it; the browser UA is still the Cloudflare workaround).
            req = urllib.request.Request(
                "https://api.runpod.io/graphql", data=body,
                headers={"Content-Type": "application/json", "User-Agent": RUNPOD_UA,
                         "Authorization": f"Bearer {key}"}, method="POST")
            with urllib.request.urlopen(req, timeout=8) as r:
                me = json.loads(r.read().decode()).get("data", {}).get("myself") or {}
            result = {"balance": me.get("clientBalance"),
                      "open_pods": len(me.get("pods") or []),
                      "pod_cost_hr": sum(p.get("costPerHr") or 0 for p in (me.get("pods") or []))}
        except Exception:  # noqa: BLE001 - offline / API down is a normal state
            pass
    _runpod_cache = (time.time(), result)
    return result


def _remote_payload_bytes(ip: str, port: str) -> int | None:
    """Uploaded-so-far byte count on the pod (cached 30 s; one cheap ssh stat)."""
    global _upload_cache
    if _upload_cache and time.time() - _upload_cache[0] < 30.0:
        return _upload_cache[1]
    size = None
    keyfile = GPU_WORKDIR / "id_ed25519"
    if keyfile.exists():
        try:
            out = subprocess.run(
                ["ssh", "-p", port, "-i", str(keyfile), "-o", "IdentitiesOnly=yes",
                 "-o", "StrictHostKeyChecking=accept-new",
                 "-o", f"UserKnownHostsFile={GPU_WORKDIR / 'known_hosts'}",
                 "-o", "ConnectTimeout=5", f"root@{ip}",
                 "stat -c%s /workspace/payload.tar.gz.enc 2>/dev/null || echo 0"],
                capture_output=True, text=True, timeout=12).stdout.strip()
            if out.isdigit():
                size = int(out)
        except (subprocess.SubprocessError, OSError):
            pass
    _upload_cache = (time.time(), size)
    return size


def _pod_delta_points(ip: str, port: str) -> int | None:
    """Point count of the pod-side delta collection (cached 20 s; ssh → pod-local curl)."""
    global _delta_pts_cache
    if _delta_pts_cache and time.time() - _delta_pts_cache[0] < 20.0:
        return _delta_pts_cache[1]
    pts = None
    keyfile = GPU_WORKDIR / "id_ed25519"
    if keyfile.exists():
        try:
            out = subprocess.run(
                ["ssh", "-p", port, "-i", str(keyfile), "-o", "IdentitiesOnly=yes",
                 "-o", "StrictHostKeyChecking=accept-new",
                 "-o", f"UserKnownHostsFile={GPU_WORKDIR / 'known_hosts'}",
                 "-o", "ConnectTimeout=5", f"root@{ip}",
                 "curl -s -m 4 localhost:6333/collections/georgian_legal_delta 2>/dev/null"],
                capture_output=True, text=True, timeout=12).stdout
            d = json.loads(out or "{}")
            pts = (d.get("result") or {}).get("points_count")
        except Exception:  # noqa: BLE001 - pod busy/unreachable is a normal state
            pass
    _delta_pts_cache = (time.time(), pts)
    return pts


def _seen_total() -> int | None:
    """seen.sqlite row count (cached 10 s; read-only URI so we never touch the crawler's lock)."""
    global _seen_cache
    if _seen_cache and time.time() - _seen_cache[0] < 10.0:
        return _seen_cache[1]
    n = None
    try:
        import sqlite3
        db = REPO / "artifacts" / "matsne" / "seen.sqlite"
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.0)
        try:
            n = conn.execute("SELECT COUNT(*) FROM seen").fetchone()[0]
        finally:
            conn.close()
    except Exception:  # noqa: BLE001 - db busy/absent is a normal state
        pass
    _seen_cache = (time.time(), n)
    return n


_LOGSTAT_RE = re.compile(
    r"Crawled (\d+) pages \(at (\d+) pages/min\), scraped (\d+) items \(at (\d+) items/min\)")


def scrape_state() -> dict:
    """Live view of the newest matsne crawl: items so far, rate, alive/finished, seen total."""
    runs = REPO / "artifacts" / "matsne" / "runs"
    try:
        newest = max(runs.iterdir(), key=lambda p: p.stat().st_mtime)
    except (OSError, ValueError):
        return {"active": False}
    st: dict = {"active": True, "run": newest.name}
    items = newest / "items.jsonl"
    try:
        with items.open("rb") as fh:
            st["items"] = sum(1 for _ in fh)
    except OSError:
        st["items"] = 0
    log = newest / "spider.log"
    try:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
    except OSError:
        tail = []
    for ln in reversed(tail):
        m = _LOGSTAT_RE.search(ln)
        if m:
            st.update(pages=int(m.group(1)), pages_min=int(m.group(2)),
                      items_min=int(m.group(4)))
            break
    st["finished"] = any("Spider closed (finished)" in ln for ln in tail)
    st["alive"] = any("scrapy crawl matsne" in r["args"] for r in _ps_rows())
    st["errors"] = sum(1 for ln in tail if " ERROR" in ln)
    seen = _seen_total()
    if seen is not None:
        st["seen_total"] = seen
        st["advertised"] = ADVERTISED_TOTAL
        st["coverage_pct"] = round(100 * seen / ADVERTISED_TOTAL, 1)
    return st


def gpu_state() -> dict:
    """Parse the newest delta-embed orchestrator log into a live phase/progress view."""
    logs = sorted(GPU_WORKDIR.glob("delta_orch*.log"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return {"active": False}
    try:
        lines = logs[0].read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {"active": False}

    st: dict = {"active": True, "log": logs[0].name, "phase": "starting", "detail": ""}
    provisioned_at = None
    ip = port = None
    payload_mb = None
    for ln in lines:
        if "staged" in ln and "item lines" in ln:
            m = re.search(r"staged (\d+) run file\(s\), (\d+) item lines", ln)
            if m:
                st["runs"], st["items"] = int(m.group(1)), int(m.group(2))
        elif "delta payload encrypted" in ln:
            m = re.search(r"\(([\d.]+) MB\)", ln)
            payload_mb = float(m.group(1)) if m else None
            st["phase"], st["payload_mb"] = "packaged", payload_mb
        elif "provisioned pod" in ln:
            m = re.search(r"provisioned pod (\S+) on (.+?) \(\$([\d.]+)/hr\)", ln)
            if m:
                st.update(pod=m.group(1), gpu=m.group(2), price_hr=float(m.group(3)))
            provisioned_at = _log_epoch(ln)
            st["phase"] = "booting pod"
        elif "public SSH mapped" in ln:
            m = re.search(r"public SSH mapped: ([\d.]+):(\d+)", ln)
            if m:
                ip, port = m.group(1), m.group(2)
        elif "pod tools ready" in ln:
            st["phase"] = "uploading payload"
        elif "payload push: attempt" in ln:
            st["phase"], st["detail"] = "uploading payload", ln.split("] ", 1)[-1]
        elif "delta embed launched" in ln:
            st["phase"], st["detail"] = "embedding on GPU", "installing deps + model load first"
        elif ln.strip().startswith("delta:") or "  delta: " in ln:
            st["phase"] = "embedding on GPU"
            st["detail"] = ln.split("delta:", 1)[-1].strip()[:220]
        elif "DONE marker found" in ln:
            m = re.search(r"points=(\d+)", ln)
            st["phase"] = "snapshotting"
            if m:
                st["points"] = int(m.group(1))
        elif "transferring delta results" in ln:
            st["phase"] = "downloading snapshot"
        elif "got delta snapshot" in ln:
            st["phase"], st["detail"] = "verifying + restoring", ln.split("] ", 1)[-1]
        elif "G2 cosine" in ln:
            st["g2"] = ln.split("] ", 1)[-1]
        elif "restoring delta snapshot" in ln:
            st["phase"] = "restoring locally"
        elif "COST:" in ln:
            st["cost_line"] = ln.split("] ", 1)[-1]
        elif "terminated" in ln and "pod" in ln:
            st["pod_terminated"] = True
        elif "RESTORE COMPLETE" in ln:
            st["phase"], st["done"] = "complete", True
        elif "liveness probe failed" in ln:
            st["detail"] = ln.split("] ", 1)[-1]

    orch_alive = _script_alive("runpod_orchestrate_delta.py")
    st["orchestrator_alive"] = orch_alive
    if not orch_alive and not st.get("done"):
        tb = any("Traceback" in ln or "rsync error" in ln for ln in lines[-30:])
        st["phase"] = "FAILED (orchestrator exited)" if tb else "exited"
    if provisioned_at and not st.get("pod_terminated"):
        up_s = max(0, time.time() - provisioned_at)
        st["pod_uptime_min"] = round(up_s / 60, 1)
        if st.get("price_hr"):
            st["cost_so_far"] = round(up_s / 3600 * st["price_hr"], 3)
    if st["phase"] == "uploading payload" and ip and port and payload_mb:
        got = _remote_payload_bytes(ip, port)
        if got is not None:
            st["upload_pct"] = min(100, round(100 * got / (payload_mb * 1e6)))
            st["upload_mb"] = round(got / 1e6, 1)
    if st["phase"] == "embedding on GPU" and ip and port and orch_alive:
        pts = _pod_delta_points(ip, port)
        if pts is not None:
            st["points_so_far"] = pts
            if st.get("items"):
                est = int(st["items"] * CHUNKS_PER_DOC)
                st["est_total_points"] = est
                st["embed_pct"] = min(99, round(100 * pts / est))
    st.update(_runpod_account())
    return st


# --- I6 re-embed pipeline panel (2026-07-10) --------------------------------------
# Stage markers emitted by scripts/runpod_orchestrate_reembed.py (O.log lines) — ordered;
# the LAST marker seen wins, so the panel tracks the pipeline monotonically.
_REEMBED_STAGES = [
    ("packaging payload", "packaging rows"),
    ("provisioned pod", "pod provisioned"),
    ("payload push", "uploading rows to pod"),
    ("reembed launched", "embedding on GPU (v2 headers)"),
    ("DONE marker", "embed complete"),
    ("tunnel up", "eval over tunnel (hybrid)"),
    ("rerank server healthy", "eval over tunnel (rerank@50)"),
    ("GATE verdict: PASS", "GATE PASS — pulling snapshot"),
    ("GATE verdict: FAIL", "GATE FAIL — nothing restored"),
    ("snapshot pulled", "restoring locally as georgian_legal_v2"),
    ("restored `georgian_legal_v2`", "restored locally"),
    ("I6 REEMBED COMPLETE", "complete"),
]


def _args_alive(substr: str) -> bool:
    """True iff a python/bash process's FULL argument string contains ``substr``.

    Unlike ``_script_alive`` (per-token match), this handles multi-word needles like
    ``-m ingest watch``; tail/grep watchers are excluded by the exe check."""
    for r in _ps_rows():
        parts = r["args"].split()
        if len(parts) < 2:
            continue
        exe = parts[0].rsplit("/", 1)[-1]
        if exe.startswith(("python", "bash")) and substr in " ".join(parts[1:]):
            return True
    return False


def reembed_state() -> dict:
    """The post-scrape I6 pipeline: local delta embed → orchestrated v2 re-embed."""
    st: dict = {"delta": None, "orch": None}

    dlog = GPU_WORKDIR / "delta_embed_all.log"
    try:
        lines = dlog.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    if lines:
        stage = "starting"
        for ln in lines:
            if "embedding source:" in ln:
                stage = "embedding " + ln.split("embedding source:")[-1].strip(" =")
            elif "re-verify coverage" in ln:
                stage = "verifying coverage"
            elif "DELTA EMBED ALL DONE" in ln:
                stage = "complete"
            elif "MISSING" in ln and " 0 MISSING" not in ln:
                stage = "verify: " + ln.strip()[:80]
        alive = _args_alive("delta_embed_all.sh") or _args_alive("-m ingest watch")
        if not alive and stage not in ("complete",) and not stage.startswith("verify"):
            stage += " (process gone?)"
        st["delta"] = {"stage": stage, "tail": lines[-1][-160:], "alive": alive}

    olog = GPU_WORKDIR / "reembed_v2.log"
    try:
        olines = olog.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        olines = []
    if olines:
        stage = "starting"
        cost = None
        for ln in olines:
            for marker, label in _REEMBED_STAGES:
                if marker in ln:
                    stage = label
            if "COST:" in ln:
                cost = ln.split("COST:")[-1].strip()
        alive = _script_alive("runpod_orchestrate_reembed.py")
        if not alive and stage != "complete":
            stage += " (orchestrator gone?)"
        orch = {"stage": stage, "tail": olines[-1][-160:], "alive": alive, "cost": cost}
        try:
            orch["verdict"] = json.loads(
                (GPU_WORKDIR / "out_reembed_v2" / "verdict.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            orch["verdict"] = None
        st["orch"] = orch
    return st


def build_state() -> dict:
    return {
        "now": time.strftime("%H:%M:%S"),
        "sessions": sessions_state(),
        "system": system_state(),
        "rag": rag_state(),
        "gpu": gpu_state(),
        "scrape": scrape_state(),
        "qdrant": qdrant_state(),
        "coverage": coverage_state(),
        "reembed": reembed_state(),
        "processes": processes_state(),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request logging
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        # No CORS header: the dashboard is same-origin, and /api/state carries account
        # data (balance, publish state) that a random website's JS must not be able to
        # read off localhost.
        self.send_header("Cache-Control", "no-store")  # always serve the freshest page + state
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            try:
                body = json.dumps(build_state()).encode()
            except Exception as e:  # noqa: BLE001 - never 500 the dashboard
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self._send(body, "application/json")
        else:
            try:
                html = HTML_FILE.read_bytes()
            except OSError:
                html = b"<h1>session_monitor.html not found next to the script</h1>"
            self._send(html, "text/html; charset=utf-8")


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"session monitor → http://localhost:{port}   (project: {PROJECT_DIR.name})")
    print("  Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
