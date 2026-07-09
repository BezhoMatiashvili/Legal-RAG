#!/usr/bin/env python3
"""Publish the local Qdrant collection to the RunPod network volume (serverless worker data).

The local Qdrant is the write master; the serverless worker serves a read-only copy that
it restores from a snapshot published here. Four subcommands, run in order:

    uv run --group publish python scripts/publish_snapshot.py --create
    uv run --group publish python scripts/publish_snapshot.py --upload      # resumable: rerun after any interruption
    uv run --group publish python scripts/publish_snapshot.py --verify     # triggers the worker's restore boot
    uv run --group publish python scripts/publish_snapshot.py --cleanup

Protocol (mirrored by ingest/serverless/qdrant_boot.py): the ~24GB ``<name>.snapshot``
object is uploaded first and ``publish/manifest.json`` LAST — the manifest is the atomic
"this publish is complete" signal, so a half-finished upload can never trigger a restore.

ingest/.env keys used (never printed): QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME,
RUNPOD_ENDPOINT_ID, RUNPOD_API_KEY, RUNPOD_VOLUME_ID (= the S3 bucket name),
RUNPOD_S3_ENDPOINT (default https://s3api-eu-cz-1.runpod.io), RUNPOD_S3_ACCESS_KEY,
RUNPOD_S3_SECRET_KEY, RUNPOD_S3_REGION (default eu-cz-1), PUBLISH_VERIFY_TIMEOUT
(deliberately NOT the day-to-day RUNPOD_API_TIMEOUT — a restore boot needs ~30 min).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ingest/ (the package root)

from ingest.config import load_config  # noqa: E402

PUBLISH_DIR = Path(__file__).resolve().parents[1] / ".state" / "publish"
UPLOAD_STATE = PUBLISH_DIR / "upload_state.json"
MANIFEST = PUBLISH_DIR / "manifest.json"
QDRANT_CONTAINER = "legal-qdrant"
REMOTE_PREFIX = "publish/"
# RunPod S3 gateway: docs claim parts up to 500MB, but the gateway's edge proxy 413s
# large bodies (observed: 256MB rejected). 95MB stays under the usual 100MB proxy cap;
# ~284 parts for 26GB is fine for ListParts and gives decent resume granularity.
PART_SIZE = 95 * 1024 * 1024


def _qdrant(method: str, path: str, timeout: float = 60) -> dict:
    cfg = load_config()
    req = urllib.request.Request(
        f"{cfg.qdrant_url}{path}", method=method,
        headers={"api-key": cfg.qdrant_api_key} if cfg.qdrant_api_key else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "{}")


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
                print(f"  sha256 {done / 1e9:.1f}/{total / 1e9:.1f} GB "
                      f"({done / max(time.monotonic() - t0, 0.1) / 1e6:.0f} MB/s)")
    return h.hexdigest()


def create() -> None:
    """Snapshot the live collection, pull the file out of the container, write the manifest."""
    cfg = load_config()
    PUBLISH_DIR.mkdir(parents=True, exist_ok=True)

    info = _qdrant("GET", f"/collections/{cfg.collection_name}")["result"]
    points = info["points_count"]
    version = _qdrant("GET", "/").get("version")

    # Reap snapshots a previous interrupted --create left on the container's writable
    # layer (each is ~24GB of invisible disk).
    for old in (_qdrant("GET", f"/collections/{cfg.collection_name}/snapshots")
                .get("result") or []):
        print(f"deleting stale container-side snapshot {old['name']}")
        _qdrant("DELETE", f"/collections/{cfg.collection_name}/snapshots/{old['name']}",
                timeout=300)

    print(f"snapshotting {cfg.collection_name}: {points} points (qdrant {version}) — "
          "this can take several minutes and ~24GB inside the container...")
    t0 = time.monotonic()
    resp = _qdrant("POST", f"/collections/{cfg.collection_name}/snapshots?wait=true",
                   timeout=3600)
    name = resp["result"]["name"]
    print(f"snapshot {name} created in {time.monotonic() - t0:.0f}s; copying out of the container...")

    target = PUBLISH_DIR / name
    try:
        subprocess.run(
            ["docker", "cp",
             f"{QDRANT_CONTAINER}:/qdrant/snapshots/{cfg.collection_name}/{name}", str(target)],
            check=True)
    finally:
        # Free the container-side copy even when the cp fails/is interrupted (the reap
        # above is the backstop if THIS delete is what gets interrupted).
        try:
            _qdrant("DELETE", f"/collections/{cfg.collection_name}/snapshots/{name}",
                    timeout=300)
        except Exception as e:  # noqa: BLE001
            print(f"warning: could not delete container-side snapshot {name}: {e}")

    size = target.stat().st_size
    print(f"hashing {size / 1e9:.1f} GB...")
    digest = _sha256(target)

    manifest = {
        "snapshot": name,
        "sha256": digest,
        "size_bytes": size,
        "points_count": points,
        "collection": cfg.collection_name,
        "qdrant_version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest written: {MANIFEST}\nnext: --upload")


def _s3():
    import boto3
    from botocore.config import Config as BotoConfig

    endpoint = os.getenv("RUNPOD_S3_ENDPOINT", "https://s3api-eu-cz-1.runpod.io")
    region = os.getenv("RUNPOD_S3_REGION", "eu-cz-1")
    access, secret = os.getenv("RUNPOD_S3_ACCESS_KEY"), os.getenv("RUNPOD_S3_SECRET_KEY")
    bucket = os.getenv("RUNPOD_VOLUME_ID")
    if not (access and secret and bucket):
        sys.exit("Set RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_KEY and RUNPOD_VOLUME_ID in "
                 "ingest/.env (console: Settings → S3 API Keys; the bucket is the volume id).")
    client = boto3.client(
        "s3", endpoint_url=endpoint, region_name=region,
        aws_access_key_id=access, aws_secret_access_key=secret,
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"},
                          retries={"max_attempts": 5, "mode": "standard"}))
    return client, bucket


def _load_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sys.exit("No local manifest — run --create first.")


def upload() -> None:
    """Resumable multipart upload of the snapshot, then the manifest LAST (atomic signal)."""
    manifest = _load_manifest()
    snap = PUBLISH_DIR / manifest["snapshot"]
    if not snap.exists():
        sys.exit(f"{snap} missing — rerun --create.")
    size = snap.stat().st_size
    if size != manifest["size_bytes"]:
        sys.exit(f"{snap} is {size} bytes but the manifest says {manifest['size_bytes']} — "
                 "stale file? rerun --create.")

    client, bucket = _s3()
    key = f"{REMOTE_PREFIX}{manifest['snapshot']}"
    n_parts = (size + PART_SIZE - 1) // PART_SIZE

    # Resume or start the multipart upload; the server's ListParts is the source of truth
    # for what already landed (local state only remembers the upload_id).
    state = {}
    try:
        state = json.loads(UPLOAD_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    if state.get("upload_id") and state.get("key") != key:
        # Superseded by a new --create (new timestamped snapshot name): abort the old
        # upload or its invisible parts keep billing volume storage forever.
        _abort_quietly(client, bucket, state["key"], state["upload_id"])
    upload_id = state.get("upload_id") if state.get("key") == key else None
    done_parts: dict[int, str] = {}
    if upload_id:
        try:
            resp = client.list_parts(Bucket=bucket, Key=key, UploadId=upload_id,
                                     MaxParts=10000)
            done_parts = {p["PartNumber"]: p["ETag"] for p in resp.get("Parts", [])}
            print(f"resuming upload {upload_id[:16]}…: {len(done_parts)}/{n_parts} parts already uploaded")
        except Exception as e:  # noqa: BLE001 - completed or expired upload id
            # A previous run may have COMPLETED the object and died before the manifest
            # PUT — probe before re-uploading 24GB for nothing.
            try:
                head = client.head_object(Bucket=bucket, Key=key)
                if head.get("ContentLength") == size:
                    print("snapshot object already complete on the volume; skipping upload")
                    _publish_manifest_object(client, bucket, manifest)
                    UPLOAD_STATE.unlink(missing_ok=True)
                    return
            except Exception:  # noqa: BLE001 - no such object → genuinely fresh upload
                pass
            print(f"cannot resume ({type(e).__name__}: {e}); starting a fresh upload")
            _abort_quietly(client, bucket, key, upload_id)
            upload_id = None
    if not upload_id:
        upload_id = client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
        UPLOAD_STATE.write_text(json.dumps({"key": key, "upload_id": upload_id}),
                                encoding="utf-8")

    t0 = time.monotonic()
    sent = 0
    with snap.open("rb") as fh:
        for part_no in range(1, n_parts + 1):
            if part_no in done_parts:
                continue
            fh.seek((part_no - 1) * PART_SIZE)
            body = fh.read(PART_SIZE)
            resp = client.upload_part(Bucket=bucket, Key=key, UploadId=upload_id,
                                      PartNumber=part_no, Body=body)
            done_parts[part_no] = resp["ETag"]
            sent += len(body)
            rate = sent / max(time.monotonic() - t0, 0.1)
            remaining = size - min(part_no, n_parts) * PART_SIZE
            eta_min = max(remaining, 0) / max(rate, 1) / 60
            print(f"  part {part_no}/{n_parts} · {rate * 8 / 1e6:.1f} Mbps observed · "
                  f"~{eta_min:.0f} min left (interrupt + rerun --upload to resume)")

    client.complete_multipart_upload(
        Bucket=bucket, Key=key, UploadId=upload_id,
        MultipartUpload={"Parts": [{"PartNumber": n, "ETag": e}
                                   for n, e in sorted(done_parts.items())]})
    print(f"snapshot uploaded ({size / 1e9:.1f} GB in {(time.monotonic() - t0) / 60:.1f} min)")

    _publish_manifest_object(client, bucket, manifest)
    UPLOAD_STATE.unlink(missing_ok=True)


def _abort_quietly(client, bucket: str, key: str, upload_id: str) -> None:
    """Best-effort abort of a superseded multipart upload (orphaned parts bill storage)."""
    try:
        client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        print(f"aborted superseded multipart upload for {key}")
    except Exception as e:  # noqa: BLE001 - already gone is fine; --cleanup sweeps the rest
        print(f"note: could not abort old multipart upload for {key}: {type(e).__name__}")


def _publish_manifest_object(client, bucket: str, manifest: dict) -> None:
    """Upload the manifest LAST: it is the atomic 'publish complete' signal for the worker."""
    client.put_object(Bucket=bucket, Key=f"{REMOTE_PREFIX}manifest.json",
                      Body=json.dumps(manifest, indent=2).encode(),
                      ContentType="application/json")
    print("manifest published — the worker restores on its next job or boot. next: --verify")


def verify() -> None:
    """Make the worker apply the publish (refresh op — works on warm workers too), then
    check restore status and point parity. Exit code is the verdict."""
    from ingest.remote_search import RunPodQueueClient

    manifest = _load_manifest()
    cfg = load_config()
    if not (cfg.runpod_endpoint_id and cfg.runpod_api_key):
        sys.exit("Set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY in ingest/.env first.")
    # The first boot after a publish does sha256 + restore of ~24GB: allow up to 30 min.
    # Deliberately NOT cfg.runpod_api_timeout — that is the day-to-day MCP budget (240s).
    budget = int(os.getenv("PUBLISH_VERIFY_TIMEOUT") or 1800)
    client = RunPodQueueClient(cfg.runpod_endpoint_id, cfg.runpod_api_key, timeout=budget)

    # 'refresh' (not 'health') so a warm/FlashBoot-revived worker — which never re-runs
    # its boot-time restore check — applies the new publish right now.
    print(f"asking the worker to apply the publish (budget {budget}s — a restore takes 10–20 min)...")
    status = json.loads(client.call("refresh", timeout=budget).get("result") or "{}")
    print(f"worker restore: {json.dumps(status, indent=2)}")
    if status.get("status") not in ("up_to_date", "restored") \
            or status.get("snapshot") != manifest["snapshot"]:
        sys.exit(f"\nFAIL — worker restore status is {status}, expected "
                 f"up_to_date/restored of {manifest['snapshot']!r}. "
                 "Check the endpoint logs in the RunPod console.")

    out = client.call("health", timeout=budget)
    health = json.loads(out.get("result") or "{}")
    print(f"worker sysinfo: {json.dumps(out.get('sysinfo'), indent=2)}")
    print(f"worker health:  {json.dumps(health, indent=2)}")

    points = health.get("points")
    if health.get("ok") and points == manifest["points_count"]:
        print(f"\nPASS — worker serves {points} points, matching the manifest.")
    else:
        sys.exit(f"\nFAIL — worker points={points}, manifest={manifest['points_count']}, "
                 f"ok={health.get('ok')}. Check the endpoint logs in the RunPod console.")


def cleanup() -> None:
    """Reclaim local disk and delete superseded snapshots on the volume (keeps the current one)."""
    manifest = _load_manifest()
    keep_key = f"{REMOTE_PREFIX}{manifest['snapshot']}"
    snap = PUBLISH_DIR / manifest["snapshot"]
    if snap.exists():
        print(f"deleting local {snap} ({snap.stat().st_size / 1e9:.1f} GB)")
        snap.unlink()

    client, bucket = _s3()
    resp = client.list_objects_v2(Bucket=bucket, Prefix=REMOTE_PREFIX)
    for obj in resp.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".snapshot") and key != keep_key:
            print(f"deleting superseded volume object {key}")
            client.delete_object(Bucket=bucket, Key=key)
    # Sweep orphaned multipart uploads — their parts are invisible to list_objects but
    # bill volume storage all the same.
    try:
        for up in (client.list_multipart_uploads(Bucket=bucket, Prefix=REMOTE_PREFIX)
                   .get("Uploads") or []):
            _abort_quietly(client, bucket, up["Key"], up["UploadId"])
    except Exception as e:  # noqa: BLE001 - the gateway may not support the listing
        print(f"note: could not sweep multipart uploads: {type(e).__name__}: {e}")
    print("cleanup done (the current snapshot stays on the volume for disaster re-restore).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--create", action="store_true", help="snapshot local Qdrant + manifest")
    ap.add_argument("--upload", action="store_true", help="resumable upload to the volume")
    ap.add_argument("--verify", action="store_true", help="wake worker, check point parity")
    ap.add_argument("--cleanup", action="store_true", help="reclaim local/volume disk")
    args = ap.parse_args()
    if not any((args.create, args.upload, args.verify, args.cleanup)):
        ap.error("pick one of --create / --upload / --verify / --cleanup")
    if args.create:
        create()
    if args.upload:
        upload()
    if args.verify:
        verify()
    if args.cleanup:
        cleanup()


if __name__ == "__main__":
    main()
