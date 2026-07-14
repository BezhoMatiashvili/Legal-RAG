#!/usr/bin/env python3
"""Guarded transport for a prebuilt immutable generation snapshot.

This script no longer creates snapshots from ``COLLECTION_NAME``. Build and verify an
explicit generation with ``create_generation.py`` and ``promote_generation.py``; snapshot
creation then belongs to an approved generation-specific backend, never this legacy path.

Remote operations require ``--apply`` plus ``PUBLISH_REMOTE_APPROVED=1``. Legacy cleanup
is permanently inventory-only: snapshots and multipart evidence are never deleted by this
transport. Upload is additionally blocked until a proven conditional remote-manifest
activator is wired in code; an unconditional overwrite of ``publish/manifest.json`` is
never allowed.

Protocol (mirrored by ingest/serverless/qdrant_boot.py): the ~24GB ``<name>.snapshot``
object is uploaded first and ``publish/manifest.json`` LAST — the manifest is the atomic
"this publish is complete" signal, so a half-finished upload can never trigger a restore.

ingest/.env keys used (never printed): RUNPOD_ENDPOINT_ID, RUNPOD_API_KEY,
RUNPOD_VOLUME_ID (= the S3 bucket name),
RUNPOD_S3_ENDPOINT (default https://s3api-eu-cz-1.runpod.io), RUNPOD_S3_ACCESS_KEY,
RUNPOD_S3_SECRET_KEY, RUNPOD_S3_REGION (default eu-cz-1), PUBLISH_VERIFY_TIMEOUT
(deliberately NOT the day-to-day RUNPOD_API_TIMEOUT — a restore boot needs ~30 min),
PUBLISH_COLD_RESTORE_CONFIRMED=1 (automation equivalent of --cold-restore-confirmed),
and PUBLISH_REMOTE_APPROVED=1.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import stat
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Protocol

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1])
)  # ingest/ (the package root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serverless"))

from ingest.config import load_config  # noqa: E402
from qdrant_boot import validate_publish_manifest  # noqa: E402

PUBLISH_DIR = Path(__file__).resolve().parents[1] / ".state" / "publish"
UPLOAD_STATE = PUBLISH_DIR / "upload_state.json"
MANIFEST = PUBLISH_DIR / "manifest.json"
REMOTE_PREFIX = "publish/"
PUBLISH_REMOTE_APPROVED_ENV = "PUBLISH_REMOTE_APPROVED"
REMOTE_ACTIVATION_CAPABILITY = "conditional-manifest-activation-v1"
# RunPod S3 gateway: docs claim parts up to 500MB, but the gateway's edge proxy 413s
# large bodies (observed: 256MB rejected). 95MB stays under the usual 100MB proxy cap;
# ~284 parts for 26GB is fine for ListParts and gives decent resume granularity.
PART_SIZE = 95 * 1024 * 1024


class ConditionalManifestActivator(Protocol):
    """Proven remote compare-and-swap capability; no production adapter exists yet."""

    capability: str

    def activate(
        self,
        *,
        client: object,
        bucket: str,
        key: str,
        body: bytes,
        metadata: Mapping[str, str],
    ) -> None:
        """Conditionally create/replace ``key`` or fail if remote state changed."""


class PublisherLockedError(RuntimeError):
    """Another local publisher holds the operation lock."""


def _create_private_directories(directory: Path) -> None:
    missing: list[Path] = []
    cursor = directory
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if cursor.exists() and (cursor.is_symlink() or not cursor.is_dir()):
        raise RuntimeError(f"publisher parent is not a real directory: {cursor}")
    for path in reversed(missing):
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f"publisher parent is not a real directory: {path}")


@contextlib.contextmanager
def publisher_lock() -> Iterator[None]:
    """Serialize every local publish/verify/cleanup operation without waiting."""
    _create_private_directories(PUBLISH_DIR)
    lock_path = PUBLISH_DIR / ".publisher.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PublisherLockedError(f"another publisher holds {lock_path}") from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _require_remote_approval(*, apply: bool, environ: Mapping[str, str] | None) -> None:
    if not apply or _environment(environ).get(PUBLISH_REMOTE_APPROVED_ENV) != "1":
        raise SystemExit(
            "Remote publication/verification requires both --apply and "
            f"{PUBLISH_REMOTE_APPROVED_ENV}=1."
        )


def _require_conditional_activator(
    activator: ConditionalManifestActivator | None,
) -> ConditionalManifestActivator:
    if (
        activator is None
        or getattr(activator, "capability", None) != REMOTE_ACTIVATION_CAPABILITY
        or not callable(getattr(activator, "activate", None))
    ):
        raise SystemExit(
            "Remote manifest activation is blocked: no proven conditional compare-and-swap "
            "adapter is installed. Do not overwrite publish/manifest.json; first implement "
            "and validate the remote store's conditional create/replace semantics."
        )
    return activator


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    t0 = time.monotonic()
    with path.open("rb") as fh:
        while True:
            block = fh.read(16 * 1024 * 1024)
            if not block:
                break
            h.update(block)
            done += len(block)
            if done % (2 * 1024**3) < 16 * 1024 * 1024:
                print(
                    f"  sha256 {done / 1e9:.1f}/{total / 1e9:.1f} GB "
                    f"({done / max(time.monotonic() - t0, 0.1) / 1e6:.0f} MB/s)"
                )
    return h.hexdigest()


def create() -> None:
    """Permanently disable snapshotting an implicit live/legacy collection."""
    raise SystemExit(
        "Legacy --create is disabled before configuration or Qdrant access. Use "
        "scripts/create_generation.py to build an explicit immutable generation and "
        "scripts/promote_generation.py to create its verified promotion plan. Snapshot "
        "creation requires a separately approved generation-specific backend."
    )


def _s3():
    import boto3
    from botocore.config import Config as BotoConfig

    endpoint = os.getenv("RUNPOD_S3_ENDPOINT", "https://s3api-eu-cz-1.runpod.io")
    region = os.getenv("RUNPOD_S3_REGION", "eu-cz-1")
    access, secret = (
        os.getenv("RUNPOD_S3_ACCESS_KEY"),
        os.getenv("RUNPOD_S3_SECRET_KEY"),
    )
    bucket = os.getenv("RUNPOD_VOLUME_ID")
    if not (access and secret and bucket):
        sys.exit(
            "Set RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_KEY and RUNPOD_VOLUME_ID in "
            "ingest/.env (console: Settings → S3 API Keys; the bucket is the volume id)."
        )
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 5, "mode": "standard"},
            # completing a 26GB multipart takes the gateway >60s server-side
            read_timeout=600,
            connect_timeout=30,
        ),
    )
    return client, bucket


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _load_manifest() -> dict:
    try:
        mode = MANIFEST.lstat().st_mode
        if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != 0o600:
            raise ValueError("manifest must be a non-symlink owner-only regular file")
        manifest = json.loads(
            MANIFEST.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise SystemExit(
            "No valid private local manifest. Stage an owner-only schema-v2 publish "
            f"manifest from a verified immutable promotion plan: {exc}"
        ) from exc
    error = validate_publish_manifest(manifest)
    if error is not None:
        raise SystemExit(
            "Local publication manifest is not an immutable schema-v2 generation: "
            f"{error}. Rebuild it from create_generation.py and promote_generation.py."
        )
    size = manifest.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SystemExit(
            "Local schema-v2 publication manifest requires size_bytes > 0."
        )
    return manifest


def _write_private_json(path: Path, value: object) -> None:
    _create_private_directories(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write to {temporary}")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _snapshot_metadata(manifest: dict) -> dict[str, str]:
    """Metadata bound to the multipart object and returned by S3 ``HEAD``.

    Multipart ETags are not content hashes.  Binding the already-computed local SHA-256
    to the object prevents a same-sized stale/partial object from being accepted on a
    resume path.  S3 request signing and per-part integrity protect the bytes in transit.
    """
    return {"sha256": str(manifest["sha256"]).lower()}


def _remote_snapshot_matches(head: dict, manifest: dict) -> bool:
    metadata = {
        str(k).lower(): str(v).lower() for k, v in (head.get("Metadata") or {}).items()
    }
    return (
        head.get("ContentLength") == manifest.get("size_bytes")
        and metadata.get("sha256") == str(manifest.get("sha256") or "").lower()
    )


def _head_remote_snapshot(client, bucket: str, key: str, manifest: dict) -> dict:
    """Return a cryptographically identified remote snapshot or fail closed."""
    head = client.head_object(Bucket=bucket, Key=key)
    if not _remote_snapshot_matches(head, manifest):
        metadata = head.get("Metadata") or {}
        raise RuntimeError(
            f"remote snapshot {key!r} failed integrity identity: "
            f"size={head.get('ContentLength')} sha256={metadata.get('sha256')!r}, "
            f"expected size={manifest.get('size_bytes')} sha256={manifest.get('sha256')!r}"
        )
    return head


def _cold_restore_confirmed(cli_confirmed: bool = False) -> bool:
    return cli_confirmed or os.getenv("PUBLISH_COLD_RESTORE_CONFIRMED") == "1"


def _require_cold_restore_confirmation(cli_confirmed: bool = False) -> None:
    """Fail closed until the operator has made an in-place restore safe.

    RunPod's FlashBoot and worker-drain controls are not exposed by this script.  The
    explicit assertion keeps manifest activation and restore verification from silently
    targeting a warm worker that still owns a live collection.
    """
    if not _cold_restore_confirmed(cli_confirmed):
        raise SystemExit(
            "Cold-restore safety gate not confirmed. Disable FlashBoot, drain all active "
            "workers, and ensure the next worker starts with empty ephemeral storage; then "
            "rerun with --cold-restore-confirmed (or "
            "PUBLISH_COLD_RESTORE_CONFIRMED=1 in controlled automation)."
        )


def upload(
    *,
    apply: bool = False,
    cold_restore_confirmed: bool = False,
    environ: Mapping[str, str] | None = None,
    manifest_activator: ConditionalManifestActivator | None = None,
) -> None:
    """Run an approved upload while holding the local publisher lock."""
    _require_remote_approval(apply=apply, environ=environ)
    _require_cold_restore_confirmation(cold_restore_confirmed)
    with publisher_lock():
        _upload_locked(manifest_activator=manifest_activator)


def _upload_locked(*, manifest_activator: ConditionalManifestActivator | None) -> None:
    """Resumable snapshot upload followed by conditional manifest activation."""
    manifest = _load_manifest()
    activator = _require_conditional_activator(manifest_activator)
    snap = PUBLISH_DIR / manifest["snapshot"]
    try:
        snap_mode = snap.lstat().st_mode
    except OSError as exc:
        raise SystemExit(
            f"{snap} missing; stage the approved immutable generation snapshot first."
        ) from exc
    if not stat.S_ISREG(snap_mode) or stat.S_IMODE(snap_mode) != 0o600:
        raise SystemExit(
            f"{snap} must be a non-symlink owner-only regular snapshot artifact."
        )
    size = snap.stat().st_size
    if size != manifest["size_bytes"]:
        sys.exit(
            f"{snap} is {size} bytes but the manifest says {manifest['size_bytes']} — "
            "stale file? Rebuild the approved immutable generation snapshot."
        )
    print("verifying the local snapshot SHA-256 before upload...")
    digest = _sha256(snap)
    if digest.lower() != manifest["sha256"].lower():
        sys.exit(
            f"{snap} sha256 is {digest}, but the manifest says {manifest['sha256']} — "
            "refusing to attach trusted metadata to altered bytes."
        )

    client, bucket = _s3()
    key = f"{REMOTE_PREFIX}{manifest['snapshot']}"
    n_parts = (size + PART_SIZE - 1) // PART_SIZE

    # A previous run may have COMPLETED the object and died before the manifest PUT
    # (observed: complete times out client-side but lands server-side, after which the
    # gateway briefly serves a STALE part listing). Probe the object first — it is the
    # only trustworthy signal.
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception:  # noqa: BLE001 - no such object → proceed with the upload
        head = None
    if head is not None and _remote_snapshot_matches(head, manifest):
        print("snapshot object already complete with matching SHA-256; skipping upload")
        _publish_manifest_object(client, bucket, manifest, activator=activator)
        _mark_upload_complete(manifest, key)
        return
    if head is not None:
        raise RuntimeError(
            "refusing to overwrite an existing remote snapshot with a different "
            f"size/SHA-256 identity: {key}. Immutable snapshot keys are never reused."
        )

    # Resume or start the multipart upload; the server's ListParts is the source of truth
    # for what already landed (local state only remembers the upload_id).
    state = {}
    try:
        state = json.loads(UPLOAD_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    state_matches = (
        state.get("key") == key
        and state.get("sha256") == manifest.get("sha256")
        and state.get("size_bytes") == size
    )
    if state.get("upload_id") and not state_matches:
        # Never delete a prior multipart upload inside a publish operation. It remains
        # explicit evidence and becomes a candidate for separately approved cleanup.
        print(
            "retaining superseded multipart upload; use --cleanup dry-run and the "
            "separate artifact-prune approval before aborting it"
        )
    upload_id = state.get("upload_id") if state_matches else None
    done_parts: dict[int, str] = {}
    if upload_id:
        try:
            resp = client.list_parts(
                Bucket=bucket, Key=key, UploadId=upload_id, MaxParts=10000
            )
            done_parts = {p["PartNumber"]: p["ETag"] for p in resp.get("Parts", [])}
            print(
                f"resuming upload {upload_id[:16]}…: {len(done_parts)}/{n_parts} parts already uploaded"
            )
        except Exception as e:  # noqa: BLE001 - completed or expired upload id
            # A previous run may have COMPLETED the object and died before the manifest
            # PUT — probe before re-uploading 24GB for nothing.
            try:
                head = client.head_object(Bucket=bucket, Key=key)
            except Exception:  # noqa: BLE001 - no such object → genuinely fresh upload
                head = None
            if head is not None and _remote_snapshot_matches(head, manifest):
                print(
                    "snapshot object already complete with matching SHA-256; "
                    "skipping upload"
                )
                _publish_manifest_object(client, bucket, manifest, activator=activator)
                _mark_upload_complete(manifest, key)
                return
            if head is not None:
                raise RuntimeError(
                    "refusing to overwrite an existing remote snapshot that appeared "
                    f"during resume: {key}. Immutable snapshot keys are never reused."
                )
            print(f"cannot resume ({type(e).__name__}: {e}); starting a fresh upload")
            print(
                "retaining the failed multipart upload for separately approved cleanup"
            )
            upload_id = None
    if not upload_id:
        upload_id = client.create_multipart_upload(
            Bucket=bucket,
            Key=key,
            Metadata=_snapshot_metadata(manifest),
        )["UploadId"]
        _write_private_json(
            UPLOAD_STATE,
            {
                "status": "in_progress",
                "key": key,
                "upload_id": upload_id,
                "sha256": manifest["sha256"],
                "size_bytes": size,
            },
        )

    t0 = time.monotonic()
    sent = 0
    with snap.open("rb") as fh:
        for part_no in range(1, n_parts + 1):
            if part_no in done_parts:
                continue
            fh.seek((part_no - 1) * PART_SIZE)
            body = fh.read(PART_SIZE)
            # The gateway throws transient 5xx (seen: 503 'failed to fetch user keys'
            # surfaced as AccessDenied) — retry per part instead of dying mid-26GB.
            for attempt in range(5):
                try:
                    resp = client.upload_part(
                        Bucket=bucket,
                        Key=key,
                        UploadId=upload_id,
                        PartNumber=part_no,
                        Body=body,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 4:
                        raise
                    wait = 5 * (attempt + 1)
                    print(
                        f"  part {part_no} failed ({type(e).__name__}: {e}); "
                        f"retry {attempt + 1}/4 in {wait}s"
                    )
                    time.sleep(wait)
            done_parts[part_no] = resp["ETag"]
            sent += len(body)
            rate = sent / max(time.monotonic() - t0, 0.1)
            remaining = size - min(part_no, n_parts) * PART_SIZE
            eta_min = max(remaining, 0) / max(rate, 1) / 60
            print(
                f"  part {part_no}/{n_parts} · {rate * 8 / 1e6:.1f} Mbps observed · "
                f"~{eta_min:.0f} min left (interrupt + rerun --upload to resume)"
            )

    client.complete_multipart_upload(
        Bucket=bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={
            "Parts": [
                {"PartNumber": n, "ETag": e} for n, e in sorted(done_parts.items())
            ]
        },
    )
    print(
        f"snapshot uploaded ({size / 1e9:.1f} GB in {(time.monotonic() - t0) / 60:.1f} min)"
    )

    # CompleteMultipartUpload success alone is insufficient on the RunPod gateway: a
    # client timeout can race an eventually visible object.  Never publish the manifest
    # until HEAD binds the exact expected size and SHA-256 metadata to this object.
    _head_remote_snapshot(client, bucket, key, manifest)
    _publish_manifest_object(client, bucket, manifest, activator=activator)
    _mark_upload_complete(manifest, key)


def _mark_upload_complete(manifest: Mapping[str, object], key: str) -> None:
    _write_private_json(
        UPLOAD_STATE,
        {
            "status": "completed",
            "key": key,
            "sha256": manifest["sha256"],
            "size_bytes": manifest["size_bytes"],
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )


def _abort_quietly(client, bucket: str, key: str, upload_id: str) -> None:
    """Best-effort abort of a superseded multipart upload (orphaned parts bill storage)."""
    try:
        client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        print(f"aborted superseded multipart upload for {key}")
    except Exception as e:  # noqa: BLE001 - already gone is fine; --cleanup sweeps the rest
        print(
            f"note: could not abort old multipart upload for {key}: {type(e).__name__}"
        )


def _publish_manifest_object(
    client,
    bucket: str,
    manifest: dict,
    *,
    activator: ConditionalManifestActivator | None,
) -> None:
    """Conditionally activate the manifest LAST after revalidating snapshot identity."""
    error = validate_publish_manifest(manifest)
    if error is not None:
        raise RuntimeError(f"refusing invalid schema-v2 manifest activation: {error}")
    conditional = _require_conditional_activator(activator)
    snapshot_key = f"{REMOTE_PREFIX}{manifest['snapshot']}"
    _head_remote_snapshot(client, bucket, snapshot_key, manifest)
    conditional.activate(
        client=client,
        bucket=bucket,
        key=f"{REMOTE_PREFIX}manifest.json",
        body=json.dumps(manifest, indent=2).encode(),
        metadata=_snapshot_metadata(manifest),
    )
    print(
        "manifest published — the worker restores on its next job or boot. next: --verify"
    )


def verify(
    *,
    apply: bool = False,
    cold_restore_confirmed: bool = False,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Run an approved worker restore/readiness verification under the local lock."""
    _require_remote_approval(apply=apply, environ=environ)
    _require_cold_restore_confirmation(cold_restore_confirmed)
    with publisher_lock():
        _verify_locked()


def _verify_locked() -> None:
    """Ask the worker to apply the publish, then check restore status and point parity.

    A worker with a live collection refuses non-atomic replacement. Operators must first
    arrange a true cold worker with empty ephemeral storage (or implement blue/green restore).
    """
    from ingest.remote_search import RunPodQueueClient

    manifest = _load_manifest()
    cfg = load_config()
    if not (cfg.runpod_endpoint_id and cfg.runpod_api_key):
        sys.exit("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY in ingest/.env first.")
    # The first boot after a publish does sha256 + restore of ~24GB: allow up to 30 min.
    # Deliberately NOT cfg.runpod_api_timeout — that is the day-to-day MCP budget (240s).
    budget = int(os.getenv("PUBLISH_VERIFY_TIMEOUT") or 1800)
    client = RunPodQueueClient(
        cfg.runpod_endpoint_id, cfg.runpod_api_key, timeout=budget
    )

    # 'refresh' (not 'health') retries a cold restore. A warm worker with live data returns
    # unsafe_warm_restore rather than deleting the last-good collection.
    print(
        f"asking the worker to apply the publish (budget {budget}s — a restore takes 10–20 min)..."
    )
    status = json.loads(client.call("refresh", timeout=budget).get("result") or "{}")
    print(f"worker restore: {json.dumps(status, indent=2)}")
    if (
        status.get("status") not in ("up_to_date", "restored")
        or status.get("snapshot") != manifest["snapshot"]
        or status.get("generation_id") != manifest["generation_id"]
        or status.get("generation_manifest_sha256")
        != manifest["generation_manifest_sha256"]
        or status.get("points") != manifest["points_count"]
        or status.get("identity_matched_points") != manifest["points_count"]
    ):
        sys.exit(
            f"\nFAIL — worker restore status is {status}, expected "
            f"the exact schema-v2 generation {manifest['generation_id']!r} from "
            f"{manifest['snapshot']!r}. "
            "Check the endpoint logs in the RunPod console."
        )

    out = client.call("health", timeout=budget)
    health = json.loads(out.get("result") or "{}")
    print(f"worker sysinfo: {json.dumps(out.get('sysinfo'), indent=2)}")
    print(f"worker health:  {json.dumps(health, indent=2)}")

    points = health.get("points")
    expected = manifest["points_count"]
    if health.get("ok") and points == expected:
        print(f"\nPASS — worker serves exactly {points} points, matching the manifest.")
    else:
        sys.exit(
            f"\nFAIL — worker points={points}, manifest={expected}, "
            f"ok={health.get('ok')}. Check the endpoint logs in the RunPod console."
        )


def cleanup(
    *,
    apply: bool = False,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Inventory legacy snapshot artifacts without ever deleting evidence."""
    _ = environ
    if apply:
        raise SystemExit(
            "Legacy snapshot cleanup deletion is permanently disabled; snapshots and "
            "multipart evidence must be retained."
        )
    with publisher_lock():
        _cleanup_locked()


def _cleanup_locked() -> None:
    manifest = _load_manifest()
    keep_key = f"{REMOTE_PREFIX}{manifest['snapshot']}"
    snap = PUBLISH_DIR / manifest["snapshot"]
    try:
        snap_mode = snap.lstat().st_mode
    except FileNotFoundError:
        snap_mode = None
    if snap_mode is not None:
        if not stat.S_ISREG(snap_mode):
            raise SystemExit(f"refusing non-regular local snapshot artifact: {snap}")
        print(f"retained local {snap} ({snap.stat().st_size / 1e9:.1f} GB)")

    client, bucket = _s3()
    resp = client.list_objects_v2(Bucket=bucket, Prefix=REMOTE_PREFIX)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".snapshot") and key != keep_key:
            print(f"retained superseded volume object {key}")
    # Sweep orphaned multipart uploads — their parts are invisible to list_objects but
    # bill volume storage all the same.
    try:
        for up in (
            client.list_multipart_uploads(Bucket=bucket, Prefix=REMOTE_PREFIX).get(
                "Uploads"
            )
            or []
        ):
            print(
                f"retained orphaned multipart upload {up['UploadId']} for {up['Key']}"
            )
    except Exception as e:  # noqa: BLE001 - the gateway may not support the listing
        print(f"note: could not sweep multipart uploads: {type(e).__name__}: {e}")
    print(
        "legacy cleanup inventory only; no local/remote snapshots or multipart evidence "
        "were deleted, and --apply is disabled."
    )


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    manifest_activator: ConditionalManifestActivator | None = None,
) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--create",
        action="store_true",
        help="disabled legacy sentinel; always explains the immutable workflow",
    )
    action.add_argument(
        "--upload", action="store_true", help="resumable upload to the volume"
    )
    action.add_argument(
        "--verify", action="store_true", help="wake worker, check point parity"
    )
    action.add_argument(
        "--cleanup",
        action="store_true",
        help="inventory retained local/remote artifacts; deletion is disabled",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="request the selected guarded remote or deletion operation",
    )
    ap.add_argument(
        "--cold-restore-confirmed",
        action="store_true",
        help=(
            "assert FlashBoot is disabled, workers are drained, and the next worker "
            "will start with empty ephemeral storage (required for --upload/--verify)"
        ),
    )
    args = ap.parse_args(argv)
    if args.create:
        create()
    elif args.upload:
        upload(
            apply=args.apply,
            cold_restore_confirmed=args.cold_restore_confirmed,
            environ=environ,
            manifest_activator=manifest_activator,
        )
    elif args.verify:
        verify(
            apply=args.apply,
            cold_restore_confirmed=args.cold_restore_confirmed,
            environ=environ,
        )
    else:
        cleanup(apply=args.apply, environ=environ)


if __name__ == "__main__":
    main()
