"""Boot and refresh the worker-local Qdrant inside a RunPod serverless worker.

The worker runs Qdrant as a side process (static binary baked into the image) with its
storage directory on the shared network volume. Publishing new data is a two-sided
protocol with ``scripts/publish_snapshot.py``:

  local box                                network volume            worker (this module)
  ---------                                --------------            --------------------
  snapshot local collection  ──upload──►   publish/<name>.snapshot
  write manifest.json LAST   ──upload──►   publish/manifest.json ──► maybe_restore() compares
                                                                     manifest to publish/ACTIVE;
                                                                     if different: sha256-verify,
                                                                     snapshot-recover via file://,
                                                                     check point count, write ACTIVE

The manifest is uploaded last so a half-finished upload can never trigger a restore.
Restores therefore happen exactly once per publish; every later cold start just opens the
existing storage directory (no restore, ~1–3 min to ready).

Concurrency: Qdrant holds an exclusive lock on its storage dir. The endpoint MUST run with
max workers = 1, but RunPod can briefly overlap an old and a new worker during a rollout —
the newcomer's Qdrant then fails to acquire the lock. ``ensure_running`` retries the spawn
until the old worker drains instead of failing the cold start.

Import has no side effects (unit-testable); the handler drives the module explicitly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("serverless.qdrant_boot")

QDRANT_URL = "http://127.0.0.1:6333"
# Overridable so the image can be rehearsed locally with a bind-mounted spare dir.
VOLUME_ROOT = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
QDRANT_BIN = os.getenv("QDRANT_BIN", "/opt/qdrant/qdrant")
QDRANT_LOG = Path(os.getenv("QDRANT_LOG", "/tmp/qdrant.log"))

_proc: subprocess.Popen | None = None


def storage_dir() -> Path:
    return VOLUME_ROOT / "qdrant_storage"


def publish_dir() -> Path:
    return VOLUME_ROOT / "publish"


def _http(method: str, path: str, body: dict | None = None, timeout: float = 10) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{QDRANT_URL}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        # Qdrant puts the actual reason in the body (e.g. the 403 'must be inside the
        # snapshots directory') — an opaque status code is undebuggable in RunPod logs.
        detail = ""
        try:
            detail = e.read().decode()[:500]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"qdrant {method} {path} → HTTP {e.code}: {detail or e.reason}") from e


def _healthy() -> bool:
    try:
        return bool(_http("GET", "/").get("version"))
    except Exception:  # noqa: BLE001 - not up yet is the normal early state
        return False


def _log_tail(n: int = 40) -> str:
    try:
        return "\n".join(QDRANT_LOG.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no qdrant log)"


def _spawn() -> subprocess.Popen:
    storage_dir().mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        QDRANT__STORAGE__STORAGE_PATH=str(storage_dir()),
        QDRANT__STORAGE__SNAPSHOTS_PATH=str(VOLUME_ROOT / "qdrant_snapshots"),
        QDRANT__SERVICE__HOST="127.0.0.1",  # never exposed; the queue API is the only ingress
        QDRANT__SERVICE__MAX_REQUEST_SIZE_MB="1024",
        QDRANT__TELEMETRY_DISABLED="true",
    )
    logfh = QDRANT_LOG.open("ab")
    return subprocess.Popen([QDRANT_BIN], env=env, stdout=logfh, stderr=subprocess.STDOUT)


def ensure_running(deadline_s: float = 300) -> None:
    """Start Qdrant against the volume and block until it serves, or raise with the log tail.

    Retries the spawn while the deadline allows: during an endpoint rollout the outgoing
    worker may still hold the storage lock for a short window, which makes the fresh
    Qdrant exit immediately — that is a wait-and-retry, not a failure.
    """
    global _proc
    if _healthy():
        return
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        if _proc is None or _proc.poll() is not None:
            if _proc is not None:
                logger.warning("qdrant exited rc=%s; retrying (storage lock still held by the "
                               "previous worker?)\n%s", _proc.returncode, _log_tail(10))
                time.sleep(3)
            _proc = _spawn()
        if _healthy():
            logger.info("qdrant ready in %.1fs (storage=%s)",
                        time.monotonic() - started, storage_dir())
            return
        time.sleep(1)
    raise RuntimeError(
        f"qdrant did not become healthy within {deadline_s}s.\n--- qdrant log tail ---\n"
        + _log_tail()
    )


# --- publish/restore protocol ---------------------------------------------------


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _sha256(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def needs_restore(manifest: dict | None, active: dict | None) -> bool:
    """A restore is due when a manifest exists and differs from what was last applied.

    Compared on the identity fields only, so re-writing ACTIVE with extra bookkeeping
    (applied_at) never re-triggers a restore.
    """
    if not manifest:
        return False
    if not active:
        return True
    return any(manifest.get(k) != active.get(k) for k in ("snapshot", "sha256", "collection"))


def restore_pending() -> bool:
    """Cheap two-file check — used per job so a warm/FlashBoot-revived worker (whose
    module import long predates the job) still notices a publish that landed since."""
    return needs_restore(_read_json(publish_dir() / "manifest.json"),
                         _read_json(publish_dir() / "ACTIVE"))


def _collection_points(collection: str) -> int | None:
    """Live point count, or None when the collection doesn't exist / qdrant is unreachable."""
    try:
        info = _http("GET", f"/collections/{collection}", timeout=10)
        return (info.get("result") or {}).get("points_count")
    except Exception:  # noqa: BLE001
        return None


def maybe_restore(force: bool = False) -> dict:
    """Apply the volume's published snapshot if it is new (or ``force``). Returns a status dict.

    Blocking and potentially slow (sha256 of ~24GB + snapshot recover ≈ 10–20 min on the
    first boot after a publish) — the endpoint's execution timeout must cover it (1800s).
    """
    manifest = _read_json(publish_dir() / "manifest.json")
    active = _read_json(publish_dir() / "ACTIVE")
    if manifest is None:
        return {"status": "no_manifest", "detail": f"nothing published at {publish_dir()}"}

    name = manifest["snapshot"]
    collection = manifest.get("collection") or os.getenv("COLLECTION_NAME", "georgian_legal")
    if not force and not needs_restore(manifest, active):
        # ACTIVE describes what SHOULD be present — verify reality before trusting it
        # (a wiped storage dir or swapped volume can carry a stale ACTIVE marker).
        points = _collection_points(collection)
        if points is not None:
            return {"status": "up_to_date", "snapshot": name, "points": points}
        logger.warning("ACTIVE claims %s is applied but collection %s is missing — "
                       "re-restoring", name, collection)

    snap = publish_dir() / name
    if not snap.exists():
        return {"status": "error", "detail": f"manifest names {name} but {snap} is missing"}

    t0 = time.monotonic()
    digest = _sha256(snap)
    if digest != manifest.get("sha256"):
        return {"status": "error",
                "detail": f"sha256 mismatch for {name}: volume={digest} manifest={manifest.get('sha256')} "
                          "(incomplete upload?) — refusing to restore"}
    logger.info("restoring %s into collection %s (sha ok, %.0fs)", name, collection,
                time.monotonic() - t0)

    # Qdrant refuses (403) file:// recovery from outside its snapshots directory, so stage
    # the file there — a hardlink is instant and free on the same volume; copy is the
    # cross-filesystem fallback.
    snapshots_dir = VOLUME_ROOT / "qdrant_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    staged = snapshots_dir / name
    if not staged.exists():
        try:
            os.link(snap, staged)
        except OSError:
            shutil.copyfile(snap, staged)
    try:
        _http("PUT", f"/collections/{collection}/snapshots/recover?wait=true",
              {"location": f"file://{staged}", "priority": "snapshot"}, timeout=1500)
    finally:
        staged.unlink(missing_ok=True)

    info = _http("GET", f"/collections/{collection}", timeout=30)
    points = (info.get("result") or {}).get("points_count")
    expected = manifest.get("points_count")
    if expected is not None and points != expected:
        return {"status": "error",
                "detail": f"restored point count {points} != manifest {expected}"}

    applied = dict(manifest, applied_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (publish_dir() / "ACTIVE").write_text(json.dumps(applied, indent=2), encoding="utf-8")
    took = time.monotonic() - t0
    logger.info("restore complete: %s points in %.0fs", points, took)
    return {"status": "restored", "snapshot": name, "points": points, "seconds": round(took)}
