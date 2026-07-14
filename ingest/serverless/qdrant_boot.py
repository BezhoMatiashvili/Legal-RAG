"""Boot and refresh the worker-local Qdrant inside a RunPod serverless worker.

The worker runs Qdrant as a side process (static binary baked into the image) with its
storage directory on the fast ephemeral container disk. The network volume holds immutable
published snapshots and the model cache. Publishing new data is a two-sided
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
Because container storage is ephemeral, a true cold start restores the active snapshot again;
FlashBoot/warm revivals can reuse the restored directory.

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
import re
import shutil
import stat
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger("serverless.qdrant_boot")

QDRANT_URL = "http://127.0.0.1:6333"
# Overridable so the image can be rehearsed locally with a bind-mounted spare dir.
VOLUME_ROOT = Path(os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume"))
# Qdrant storage lives on the EPHEMERAL CONTAINER DISK, not the network volume. The volume
# reliably serves large-file READS (the sha256 of the 26GB snapshot passes) but chokes on
# the sustained large-file WRITES that snapshot recovery does — unpacking a 5GB segment tar
# onto the volume fails with a File IO error, every time, in every datacenter. Recovering
# onto the fast local container disk sidesteps that entirely. The trade-off: storage does
# not persist across cold starts, so every cold boot re-restores from the volume snapshot
# (~2-5 min) — which the self-heal path in maybe_restore already handles (empty storage +
# stale ACTIVE ⇒ re-restore).
STORAGE_ROOT = Path(os.getenv("QDRANT_STORAGE_ROOT", "/qdrant-data"))
QDRANT_BIN = os.getenv("QDRANT_BIN", "/opt/qdrant/qdrant")
QDRANT_LOG = Path(os.getenv("QDRANT_LOG", "/tmp/qdrant.log"))
MIN_CONTAINER_DISK_GB = float(os.getenv("QDRANT_MIN_CONTAINER_DISK_GB", "64"))

_proc: subprocess.Popen | None = None

PUBLISH_SCHEMA_VERSION = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_GENERATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{7,127}$")
_POINT_IDENTITY_FIELDS = (
    "schema_version",
    "generation_id",
    "embedding_model",
    "embedding_revision",
    "tokenizer_model",
    "tokenizer_revision",
    "reranker_model",
    "reranker_revision",
    "vector_space_id",
    "chunking_fingerprint",
    "document_header",
    "retrieval_fingerprint",
)
_RESTORE_IDENTITY_FIELDS = (
    "schema_version",
    "snapshot",
    "sha256",
    "collection",
    "points_count",
    "generation_id",
    "generation_manifest_sha256",
    "point_identity",
    "vector_space",
)


def storage_dir() -> Path:
    return STORAGE_ROOT / "qdrant_storage"


def publish_dir() -> Path:
    return VOLUME_ROOT / "publish"


def _ensure_container_disk_capacity() -> None:
    """Fail before boot when the ephemeral disk cannot hold restored Qdrant + recovery temp."""
    probe = STORAGE_ROOT if STORAGE_ROOT.exists() else STORAGE_ROOT.parent
    usage = shutil.disk_usage(probe)
    total_gb = usage.total / 1e9
    if total_gb < MIN_CONTAINER_DISK_GB:
        raise RuntimeError(
            f"container disk is {total_gb:.1f} GB; Qdrant restore requires at least "
            f"{MIN_CONTAINER_DISK_GB:.0f} GB (configure the RunPod endpoint container disk, "
            "or lower QDRANT_MIN_CONTAINER_DISK_GB only after measuring peak restore usage)"
        )


def _http(
    method: str, path: str, body: dict | None = None, timeout: float = 10
) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{QDRANT_URL}{path}",
        data=data,
        method=method,
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
        raise RuntimeError(
            f"qdrant {method} {path} → HTTP {e.code}: {detail or e.reason}"
        ) from e


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
    # Temp MUST be on the same filesystem as storage (both container disk): recovery
    # unpacks segment tars into temp then rename()s them into storage — a cross-device
    # rename is EXDEV. Keeping both on the fast container disk also means the multi-GB
    # unpack writes never touch the flaky network volume.
    temp_dir = storage_dir().parent / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    # Qdrant's snapshots dir IS the publish dir where the uploaded snapshot already lives.
    # Recovery requires the file:// path to be inside the snapshots dir; pointing the dir
    # here means recover reads the published snapshot IN PLACE — no staging copy. (The old
    # staging hardlink fell back to a 26GB copy because network volumes don't support
    # hardlinks, which alone forced the volume to be ~2x oversized.) The worker is a
    # read-only replica so it never writes snapshots here.
    env = dict(
        os.environ,
        QDRANT__STORAGE__STORAGE_PATH=str(storage_dir()),
        QDRANT__STORAGE__SNAPSHOTS_PATH=str(publish_dir()),
        QDRANT__STORAGE__TEMP_PATH=str(temp_dir),
        QDRANT__SERVICE__HOST="127.0.0.1",  # never exposed; the queue API is the only ingress
        QDRANT__SERVICE__MAX_REQUEST_SIZE_MB="1024",
        QDRANT__TELEMETRY_DISABLED="true",
    )
    logfh = QDRANT_LOG.open("ab")
    return subprocess.Popen(
        [QDRANT_BIN], env=env, stdout=logfh, stderr=subprocess.STDOUT
    )


def ensure_running(deadline_s: float = 300) -> None:
    """Start Qdrant against the volume and block until it serves, or raise with the log tail.

    Retries the spawn while the deadline allows: during an endpoint rollout the outgoing
    worker may still hold the storage lock for a short window, which makes the fresh
    Qdrant exit immediately — that is a wait-and-retry, not a failure.
    """
    global _proc
    if _healthy():
        return
    _ensure_container_disk_capacity()
    started = time.monotonic()
    while time.monotonic() - started < deadline_s:
        if _proc is None or _proc.poll() is not None:
            if _proc is not None:
                logger.warning(
                    "qdrant exited rc=%s; retrying (storage lock still held by the "
                    "previous worker?)\n%s",
                    _proc.returncode,
                    _log_tail(10),
                )
                time.sleep(3)
            _proc = _spawn()
        if _healthy():
            logger.info(
                "qdrant ready in %.1fs (storage=%s)",
                time.monotonic() - started,
                storage_dir(),
            )
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


def validate_publish_manifest(manifest: dict) -> str | None:
    """Return a bounded error when a publish cannot prove generation identity."""
    if not isinstance(manifest, dict):
        return "manifest must be a JSON object"
    missing = [key for key in _RESTORE_IDENTITY_FIELDS if key not in manifest]
    if missing:
        return f"legacy/incomplete publish manifest; missing {missing}"
    if manifest.get("schema_version") != PUBLISH_SCHEMA_VERSION:
        return f"schema_version must be {PUBLISH_SCHEMA_VERSION}"

    snapshot = manifest.get("snapshot")
    if (
        not isinstance(snapshot, str)
        or not snapshot.endswith(".snapshot")
        or Path(snapshot).name != snapshot
        or snapshot in {".", ".."}
    ):
        return "snapshot must be a safe .snapshot basename"
    for key in ("sha256", "generation_manifest_sha256"):
        value = manifest.get(key)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            return f"{key} must be a lowercase SHA-256 digest"

    generation_id = manifest.get("generation_id")
    if (
        not isinstance(generation_id, str)
        or not _GENERATION_ID_RE.fullmatch(generation_id)
        or generation_id in {"legacy", "snapshot_v1"}
        or generation_id.startswith("v1")
    ):
        return "generation_id is invalid or legacy"
    if manifest.get("collection") != f"georgian_legal__gen_{generation_id}":
        return "collection must be the physical collection for generation_id"
    expected = manifest.get("points_count")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        return "points_count must be a non-negative integer"

    identity = manifest.get("point_identity")
    if not isinstance(identity, dict):
        return "point_identity must be an object"
    missing_identity = [key for key in _POINT_IDENTITY_FIELDS if key not in identity]
    if missing_identity:
        return f"point_identity is missing {missing_identity}"
    if identity.get("schema_version") != 1:
        return "point_identity.schema_version must be 1"
    if identity.get("generation_id") != generation_id:
        return "point_identity.generation_id does not match generation_id"
    for key in ("embedding_model", "tokenizer_model", "reranker_model"):
        model = identity.get(key)
        if not isinstance(model, str) or not model.strip():
            return f"point_identity.{key} must be non-empty"
    for key in (
        "embedding_revision",
        "tokenizer_revision",
        "reranker_revision",
    ):
        value = identity.get(key)
        if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
            return f"point_identity.{key} must be an immutable hexadecimal revision"
    for key in (
        "vector_space_id",
        "chunking_fingerprint",
        "retrieval_fingerprint",
    ):
        value = identity.get(key)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            return f"point_identity.{key} must be a lowercase SHA-256 digest"
    if not isinstance(identity.get("document_header"), bool):
        return "point_identity.document_header must be boolean"

    vector_space = manifest.get("vector_space")
    required_vectors = {"dense_name", "dense_dimension", "distance", "sparse_name"}
    if not isinstance(vector_space, dict) or not required_vectors.issubset(
        vector_space
    ):
        return f"vector_space must contain {sorted(required_vectors)}"
    dense_name = vector_space.get("dense_name")
    sparse_name = vector_space.get("sparse_name")
    if not isinstance(dense_name, str) or not dense_name:
        return "vector_space.dense_name must be non-empty"
    if not isinstance(sparse_name, str) or not sparse_name:
        return "vector_space.sparse_name must be non-empty"
    if dense_name == sparse_name:
        return "dense and sparse vector names must differ"
    dimension = vector_space.get("dense_dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        return "vector_space.dense_dimension must be a positive integer"
    if str(vector_space.get("distance", "")).lower() not in {
        "cosine",
        "dot",
        "euclid",
        "manhattan",
    }:
        return "vector_space.distance is unsupported"
    return None


# Compatibility for any pre-hardening local callers; new publisher code uses the public name.
_validate_publish_manifest = validate_publish_manifest


def _restore_identity(manifest: dict) -> tuple[str, ...]:
    """Canonical immutable identity used to compare manifest and ACTIVE."""
    return tuple(
        json.dumps(manifest.get(key), sort_keys=True, separators=(",", ":"))
        for key in _RESTORE_IDENTITY_FIELDS
    )


def needs_restore(manifest: dict | None, active: dict | None) -> bool:
    """A restore is due when a manifest exists and differs from what was last applied.

    Compared on the identity fields only, so re-writing ACTIVE with extra bookkeeping
    (applied_at) never re-triggers a restore.
    """
    if not manifest:
        return False
    if not active:
        return True
    return _restore_identity(manifest) != _restore_identity(active)


def restore_pending() -> bool:
    """Cheap two-file check — used per job so a warm/FlashBoot-revived worker (whose
    module import long predates the job) still notices a publish that landed since."""
    return needs_restore(
        _read_json(publish_dir() / "manifest.json"),
        _read_json(publish_dir() / "ACTIVE"),
    )


def _collection_points(collection: str) -> int | None:
    """Live point count, or None when the collection doesn't exist / qdrant is unreachable."""
    try:
        info = _http("GET", f"/collections/{collection}", timeout=10)
        return (info.get("result") or {}).get("points_count")
    except Exception:  # noqa: BLE001
        return None


def _distance(value: object) -> str:
    return str(value or "").rsplit(".", maxsplit=1)[-1].lower()


def _collection_compatibility(collection: str, manifest: dict) -> tuple[bool, dict]:
    """Verify exact count, green schema, and immutable payload identity read-only."""
    try:
        response = _http("GET", f"/collections/{collection}", timeout=30)
        info = response.get("result") if isinstance(response, dict) else None
    except Exception as exc:  # noqa: BLE001 - health uncertainty must fail closed
        return False, {
            "code": "collection_unavailable",
            "error_type": type(exc).__name__,
        }
    if not isinstance(info, dict):
        return False, {"code": "invalid_collection_info"}

    expected = manifest["points_count"]
    points = info.get("points_count")
    if isinstance(points, bool) or not isinstance(points, int) or points != expected:
        return False, {
            "code": "point_count_mismatch",
            "points": points,
            "expected_points": expected,
        }
    if (
        str(info.get("status", "")).lower() != "green"
        or _distance(info.get("optimizer_status")) != "ok"
    ):
        return False, {
            "code": "collection_not_green",
            "collection_status": info.get("status"),
            "optimizer_status": info.get("optimizer_status"),
        }

    config = info.get("config")
    params = config.get("params") if isinstance(config, dict) else None
    if not isinstance(params, dict):
        return False, {"code": "invalid_collection_config"}
    vectors = params.get("vectors")
    sparse_vectors = params.get("sparse_vectors")
    expected_vectors = manifest["vector_space"]
    dense_name = expected_vectors["dense_name"]
    sparse_name = expected_vectors["sparse_name"]
    if not isinstance(vectors, dict) or set(vectors) != {dense_name}:
        return False, {
            "code": "dense_vector_names_mismatch",
            "actual": sorted(vectors) if isinstance(vectors, dict) else None,
        }
    dense = vectors[dense_name]
    if (
        not isinstance(dense, dict)
        or dense.get("size") != expected_vectors["dense_dimension"]
        or _distance(dense.get("distance")) != expected_vectors["distance"].lower()
    ):
        return False, {"code": "dense_vector_schema_mismatch"}
    if not isinstance(sparse_vectors, dict) or set(sparse_vectors) != {sparse_name}:
        return False, {
            "code": "sparse_vector_names_mismatch",
            "actual": sorted(sparse_vectors)
            if isinstance(sparse_vectors, dict)
            else None,
        }

    conditions = [
        {"key": key, "match": {"value": manifest["point_identity"][key]}}
        for key in _POINT_IDENTITY_FIELDS
    ]
    try:
        count = _http(
            "POST",
            f"/collections/{collection}/points/count",
            {"filter": {"must": conditions}, "exact": True},
            timeout=60,
        )
        count_result = count.get("result") if isinstance(count, dict) else None
        identity_count = (
            count_result.get("count") if isinstance(count_result, dict) else None
        )
    except Exception as exc:  # noqa: BLE001 - identity uncertainty is incompatible
        return False, {
            "code": "identity_count_unavailable",
            "error_type": type(exc).__name__,
        }
    if (
        isinstance(identity_count, bool)
        or not isinstance(identity_count, int)
        or identity_count != expected
    ):
        return False, {
            "code": "identity_payload_count_mismatch",
            "identity_matched_points": identity_count,
            "expected_points": expected,
        }
    return True, {"points": points, "identity_matched_points": identity_count}


def _write_active(manifest: dict) -> None:
    """Atomically persist ACTIVE with owner-only permissions and a durable rename."""
    directory = publish_dir()
    target = directory / "ACTIVE"
    temporary = directory / f".ACTIVE.{os.getpid()}.{time.time_ns()}.tmp"
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        os.chmod(target, 0o600)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def restore_allows_serving(result: dict | None) -> bool:
    """Only an exactly verified immutable generation can answer worker operations."""
    if not isinstance(result, dict) or result.get("status") not in {
        "restored",
        "up_to_date",
    }:
        return False
    generation_id = result.get("generation_id")
    manifest_sha = result.get("generation_manifest_sha256")
    collection = result.get("collection")
    snapshot = result.get("snapshot")
    points = result.get("points")
    expected = result.get("expected_points")
    identity_count = result.get("identity_matched_points")
    return bool(
        isinstance(generation_id, str)
        and _GENERATION_ID_RE.fullmatch(generation_id)
        and collection == f"georgian_legal__gen_{generation_id}"
        and isinstance(snapshot, str)
        and snapshot.endswith(".snapshot")
        and Path(snapshot).name == snapshot
        and isinstance(manifest_sha, str)
        and _SHA256_RE.fullmatch(manifest_sha)
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (points, expected, identity_count)
        )
        and points == expected == identity_count
    )


def verified_runtime_manifest(result: dict | None) -> dict:
    """Return the exact, still-active publish manifest for a verified restore.

    Restore success alone is not a sufficient serving binding: a warm process may have
    imported search code for an older collection/model configuration, or publication may
    have advanced after the restore result was produced.  This re-reads both atomic files,
    binds them to the result, and rechecks the live collection before the worker imports or
    dispatches production retrieval code.
    """
    if not restore_allows_serving(result):
        raise RuntimeError("restore result is not an exact verified generation")
    assert isinstance(result, dict)

    manifest = _read_json(publish_dir() / "manifest.json")
    active = _read_json(publish_dir() / "ACTIVE")
    for label, value in (("publish manifest", manifest), ("ACTIVE", active)):
        error = validate_publish_manifest(value) if value is not None else "missing"
        if error is not None:
            raise RuntimeError(f"{label} is not a valid immutable generation: {error}")
    assert manifest is not None and active is not None
    if _restore_identity(manifest) != _restore_identity(active):
        raise RuntimeError("publish manifest and ACTIVE generation identities differ")

    expected_result = {
        "snapshot": manifest["snapshot"],
        "collection": manifest["collection"],
        "generation_id": manifest["generation_id"],
        "generation_manifest_sha256": manifest["generation_manifest_sha256"],
        "expected_points": manifest["points_count"],
        "points": manifest["points_count"],
        "identity_matched_points": manifest["points_count"],
    }
    mismatched = [
        key for key, expected in expected_result.items() if result.get(key) != expected
    ]
    if mismatched:
        raise RuntimeError(
            "restore result no longer matches the active generation: "
            + ", ".join(sorted(mismatched))
        )

    compatible, live = _collection_compatibility(manifest["collection"], manifest)
    if not compatible:
        raise RuntimeError(
            "active collection failed exact runtime compatibility: "
            f"{live.get('code', 'unknown')}"
        )

    # Return a detached value: callers retain this immutable boot identity and compare it
    # with later publications instead of holding a mutable test/operator-owned dictionary.
    return json.loads(json.dumps(manifest, sort_keys=True))


def runtime_manifest_identity(manifest: dict) -> tuple[str, ...]:
    """Public canonical identity for cold-boot versus warm-publication comparisons."""
    error = validate_publish_manifest(manifest)
    if error is not None:
        raise ValueError(f"invalid runtime manifest: {error}")
    return _restore_identity(manifest)


def runtime_readiness(result: dict | None, bound_manifest: dict | None) -> dict:
    """Read-only per-operation proof that boot, publication, and Qdrant still agree."""
    try:
        if not isinstance(bound_manifest, dict):
            raise RuntimeError("no cold-boot generation is bound to this process")
        current = verified_runtime_manifest(result)
        if runtime_manifest_identity(current) != runtime_manifest_identity(
            bound_manifest
        ):
            raise RuntimeError(
                "active generation changed after search runtime import; cold restart required"
            )
        return {
            "ok": True,
            "code": "verified_immutable_generation",
            "collection": current["collection"],
            "collection_name": current["collection"],
            "generation_id": current["generation_id"],
            "generation_manifest_sha256": current["generation_manifest_sha256"],
            "points": current["points_count"],
            "points_count": current["points_count"],
            "expected_points": current["points_count"],
            "identity_matched_points": current["points_count"],
            "issues": [],
        }
    except Exception as exc:  # noqa: BLE001 - readiness is a fail-closed data result
        return {
            "ok": False,
            "code": "worker_runtime_binding_invalid",
            "error": f"{type(exc).__name__}: {exc}",
            "issues": [
                {
                    "gate": "integrity",
                    "code": "worker_runtime_binding_invalid",
                }
            ],
        }


def maybe_restore(force: bool = False) -> dict:
    """Apply the volume's published snapshot if it is new (or ``force``). Returns a status dict.

    Blocking and potentially slow (sha256 of ~24GB + snapshot recover ≈ 10–20 min on the
    first boot after a publish) — the endpoint's execution timeout must cover it (1800s).
    """
    manifest = _read_json(publish_dir() / "manifest.json")
    active = _read_json(publish_dir() / "ACTIVE")
    if manifest is None:
        return {
            "status": "no_manifest",
            "detail": f"nothing published at {publish_dir()}",
        }

    manifest_error = validate_publish_manifest(manifest)
    if manifest_error:
        return {
            "status": "error",
            "code": "invalid_manifest",
            "detail": manifest_error,
        }

    name = manifest["snapshot"]
    collection = manifest["collection"]
    expected = manifest["points_count"]
    collection_checked = False
    existing_points: int | None = None
    if not force and not needs_restore(manifest, active):
        # ACTIVE describes what SHOULD be present — verify reality before trusting it
        # (a wiped storage dir or swapped volume can carry a stale ACTIVE marker).
        compatible, live = _collection_compatibility(collection, manifest)
        collection_checked = True
        if compatible:
            return {
                "status": "up_to_date",
                "snapshot": name,
                "collection": collection,
                "generation_id": manifest["generation_id"],
                "generation_manifest_sha256": manifest["generation_manifest_sha256"],
                "expected_points": expected,
                **live,
            }
        if live.get("code") != "collection_unavailable":
            return {
                "status": "error",
                **live,
                "snapshot": name,
                "detail": (
                    f"ACTIVE names {name!r}, but live collection {collection!r} is not "
                    "the exact green generation described by the immutable manifest"
                ),
            }
        logger.warning(
            "ACTIVE claims %s is applied but collection %s is missing — re-restoring",
            name,
            collection,
        )

    snap = publish_dir() / name
    try:
        snap_mode = snap.lstat().st_mode
    except OSError:
        return {
            "status": "error",
            "detail": f"manifest names {name} but {snap} is missing",
        }
    if stat.S_ISLNK(snap_mode) or not stat.S_ISREG(snap_mode):
        return {
            "status": "error",
            "detail": f"snapshot {snap} is not a regular file",
        }

    # Qdrant's snapshot recovery targets the named collection in place.  Deleting an
    # existing serving collection first would make a failed warm refresh destructive;
    # recovering over it is not an atomic replacement either.  Until the deployment
    # uses a blue/green collection plus an alias swap, only restore into an empty cold
    # worker.  ``force`` deliberately does not weaken this data-safety invariant.
    if not collection_checked:
        existing_points = _collection_points(collection)
    if existing_points is not None:
        # A failed/partial prior recovery may already have created the generation-named
        # collection.  Never recover over it, but surface the exact compatibility reason
        # (notably yellow/optimizer-busy or payload identity drift) when its count matches;
        # this remains fail-closed and makes the cold-restore failure actionable.
        if existing_points == expected:
            compatible, live = _collection_compatibility(collection, manifest)
            if not compatible:
                return {
                    "status": "error",
                    **live,
                    "snapshot": name,
                    "detail": (
                        f"existing collection {collection!r} is not the exact green "
                        "generation described by the manifest; refusing in-place recovery"
                    ),
                }
        return {
            "status": "error",
            "code": "unsafe_warm_restore",
            "snapshot": name,
            "points": existing_points,
            "detail": (
                f"refusing to replace live collection {collection!r} in place; restart "
                "the endpoint on empty ephemeral storage, or deploy blue/green restore "
                "with a verified alias swap"
            ),
        }

    t0 = time.monotonic()
    digest = _sha256(snap)
    if digest != manifest.get("sha256"):
        return {
            "status": "error",
            "detail": f"sha256 mismatch for {name}: volume={digest} manifest={manifest.get('sha256')} "
            "(incomplete upload?) — refusing to restore",
        }
    logger.info(
        "restoring %s into collection %s (sha ok, %.0fs)",
        name,
        collection,
        time.monotonic() - t0,
    )

    # The snapshot already sits inside qdrant's snapshots dir (== publish_dir(), see
    # _spawn), so recover reads it in place — no staging copy, no extra volume space.
    _http(
        "PUT",
        f"/collections/{collection}/snapshots/recover?wait=true",
        {"location": f"file://{snap}", "priority": "snapshot"},
        timeout=1500,
    )

    compatible, restored = _collection_compatibility(collection, manifest)
    if not compatible:
        return {
            "status": "error",
            **restored,
            "detail": (
                "restored collection failed exact generation/schema/payload identity; "
                "refusing to mark this snapshot ACTIVE"
            ),
        }

    applied = dict(
        manifest, applied_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    _write_active(applied)
    took = time.monotonic() - t0
    logger.info("restore complete: %s points in %.0fs", expected, took)
    return {
        "status": "restored",
        "snapshot": name,
        "collection": collection,
        "generation_id": manifest["generation_id"],
        "generation_manifest_sha256": manifest["generation_manifest_sha256"],
        "expected_points": expected,
        **restored,
        "seconds": round(took),
    }
