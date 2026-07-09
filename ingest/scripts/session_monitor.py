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

import json
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
        "mcp_count": len(mcp),
        "mcp_rss_gb": round(sum(r["rss_mb"] for r in mcp) / 1024, 2),
        "mcp_cpu": round(sum(r["cpu"] for r in mcp)),
        "qdrant_gb": round(qdrant["rss_mb"] / 1024, 1) if qdrant else None,
        "queries": queries[::-1],
    }


def build_state() -> dict:
    return {
        "now": time.strftime("%H:%M:%S"),
        "sessions": sessions_state(),
        "system": system_state(),
        "rag": rag_state(),
        "processes": processes_state(),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request logging
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
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
