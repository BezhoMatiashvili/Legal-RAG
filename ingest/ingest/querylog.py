"""Local, append-only JSONL log of served retrieval queries (operational analytics).

Written ONLY to a local file under the state dir. A query string may itself contain PII;
per project policy PII may live in LOCAL logs and the LOCAL index but must never leave the
machine — so this module does no network I/O whatsoever, and for each hit it records only
identifiers/score (``document_id``/``source``/``chunk_index``/``score``), never the full
chunk text. Logging is best-effort: a caller must never let a log failure break a search.
"""

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path


def build_query_record(
    *, query: str, filters: dict | None, top_k: int, hits, latency_ms: float,
    fingerprint: str, mode: str = "hybrid", route=None,
) -> dict:
    """A JSON-serialisable record for one served query. ``hits`` are hit dicts (as returned
    by the MCP ``_hit_dict``); only id/source/chunk_index/score are kept, not the text."""
    return {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "query": query,
        "filters": {k: v for k, v in (filters or {}).items() if v not in (None, "")},
        "top_k": top_k,
        "mode": mode,
        "route": route,
        "fingerprint": fingerprint,
        "latency_ms": round(float(latency_ms), 1),
        "n_hits": len(hits),
        "hits": [
            {
                "document_id": h.get("document_id"),
                "source": h.get("source"),
                "chunk_index": h.get("chunk_index"),
                "score": round(float(h.get("score") or 0.0), 4),
            }
            for h in hits
        ],
    }


def append_query_log(record: dict, path: Path) -> None:
    """Append one record as a JSON line (atomic per-line; local file only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        current = path.lstat()
    except FileNotFoundError:
        current = None
    if current is not None and (
        not stat.S_ISREG(current.st_mode) or stat.S_IMODE(current.st_mode) & 0o077
    ):
        raise PermissionError(
            "query log is not an owner-only regular file; refusing to chmod existing evidence"
        )
    line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(line)
        while view:
            written = os.write(fd, view)
            view = view[written:]
    finally:
        os.close(fd)
