"""Off-box disaster-recovery backup to RunPod S3 storage.

Uploads local archive files to the ``backup/`` prefix of the same RunPod
network-volume S3 bucket the publish protocol uses — deliberately OUTSIDE
``publish/`` so the snapshot-publish manifest protocol is never touched.

Each object is bound to its local SHA-256 via object metadata (multipart ETags
are not content hashes); re-running skips objects whose remote size+sha already
match, so the script is resumable and repeatable (cron-able).

Env (ingest/.env): RUNPOD_VOLUME_ID (bucket), RUNPOD_S3_ENDPOINT,
RUNPOD_S3_REGION, RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_KEY.

Usage:
    uv run python scripts/backup_offbox.py file1 [file2 ...] [--prefix backup/2026-07-18/]
    uv run python scripts/backup_offbox.py --verify-only file1 ...
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Same gateway constraint as publish_snapshot.py: large parts 413 at the edge
# proxy; 95MB stays under the ~100MB cap.
PART_SIZE = 95 * 1024 * 1024


def _s3():
    import boto3
    from botocore.config import Config as BotoConfig

    endpoint = os.getenv("RUNPOD_S3_ENDPOINT", "https://s3api-eu-cz-1.runpod.io")
    region = os.getenv("RUNPOD_S3_REGION", "eu-cz-1")
    access = os.getenv("RUNPOD_S3_ACCESS_KEY")
    secret = os.getenv("RUNPOD_S3_SECRET_KEY")
    bucket = os.getenv("RUNPOD_VOLUME_ID")
    if not (access and secret and bucket):
        raise SystemExit(
            "Set RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_KEY and RUNPOD_VOLUME_ID "
            "in ingest/.env (console: Settings -> S3 API Keys)."
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
        ),
    )
    return client, bucket


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_matches(client, bucket: str, key: str, size: int, sha: str) -> bool:
    """Cheap resume check: size matches AND the sidecar sha object equals local.

    The RunPod S3 gateway drops object Metadata on multipart uploads, so the
    SHA-256 is stored as a tiny sidecar object ``<key>.sha256`` instead.
    """
    import botocore.exceptions

    try:
        head = client.head_object(Bucket=bucket, Key=key)
        sidecar = client.get_object(Bucket=bucket, Key=key + ".sha256")
        remote_sha = sidecar["Body"].read(200).decode("ascii", "replace").split()[0]
    except (botocore.exceptions.ClientError, IndexError):
        return False
    return head.get("ContentLength") == size and remote_sha.lower() == sha


def _deep_verify(client, bucket: str, key: str, size: int, sha: str) -> bool:
    """Stream the remote object back and hash it — proof the stored bytes match."""
    obj = client.get_object(Bucket=bucket, Key=key)
    digest = hashlib.sha256()
    read = 0
    for chunk in iter(lambda: obj["Body"].read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        read += len(chunk)
    return read == size and digest.hexdigest() == sha


def _resume_state(client, bucket: str, key: str) -> tuple[str, dict[int, str]]:
    """Reuse an in-progress multipart upload for ``key`` if one exists.

    Returns (upload_id, {part_number: etag}) — completed parts are skipped on
    resume, so a 26 GB upload interrupted by a gateway timeout keeps its progress.
    """
    # No Prefix: the RunPod gateway lists multipart keys with a LEADING SLASH
    # ("/backup/..."), so a prefix-scoped listing silently matches nothing.
    listing = client.list_multipart_uploads(Bucket=bucket)
    for entry in listing.get("Uploads", []):
        if entry["Key"].lstrip("/") != key:
            continue
        upload_id = entry["UploadId"]
        done: dict[int, str] = {}
        marker = 0
        while True:
            parts = client.list_parts(
                Bucket=bucket, Key=key, UploadId=upload_id, PartNumberMarker=marker
            )
            for part in parts.get("Parts", []):
                done[part["PartNumber"]] = part["ETag"]
            if not parts.get("IsTruncated"):
                break
            marker = parts["NextPartNumberMarker"]
        print(f"RESUME: reusing multipart upload for {key} ({len(done)} parts done)")
        return upload_id, done
    return client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"], {}


def _upload_part_with_retry(
    client, bucket: str, key: str, upload_id: str, number: int, chunk: bytes
) -> str:
    """The RunPod gateway intermittently 524s a slow part — retry, don't abort."""
    delay = 5.0
    for attempt in range(6):
        try:
            response = client.upload_part(
                Bucket=bucket, Key=key, UploadId=upload_id,
                PartNumber=number, Body=chunk,
            )
            return response["ETag"]
        except Exception as exc:  # noqa: BLE001 - gateway 5xx/524 + connection resets
            if attempt == 5:
                raise
            print(f"  part {number} attempt {attempt + 1} failed ({exc}); "
                  f"retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise AssertionError("unreachable")


def upload_file(client, bucket: str, key: str, path: Path) -> None:
    size = path.stat().st_size
    sha = sha256_file(path)
    if _remote_matches(client, bucket, key, size, sha):
        print(f"SKIP (remote current): {key} size={size} sha256={sha[:16]}…")
        return
    print(f"UPLOAD: {path} -> s3://{bucket}/{key} ({size / 1e9:.2f} GB)")
    upload_id, existing = _resume_state(client, bucket, key)
    parts = []
    try:
        with path.open("rb") as handle:
            number = 1
            done = 0
            while True:
                chunk = handle.read(PART_SIZE)
                if not chunk:
                    break
                if number in existing:
                    parts.append({"ETag": existing[number], "PartNumber": number})
                else:
                    etag = _upload_part_with_retry(
                        client, bucket, key, upload_id, number, chunk
                    )
                    parts.append({"ETag": etag, "PartNumber": number})
                done += len(chunk)
                print(
                    f"  part {number} ok ({done / 1e9:.2f}/{size / 1e9:.2f} GB)",
                    flush=True,
                )
                number += 1
        client.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except BaseException:
        # Deliberately do NOT abort: the multipart survives for a resumed re-run.
        print(f"INTERRUPTED: {key} — re-run to resume (upload id {upload_id})",
              flush=True)
        raise
    if not _deep_verify(client, bucket, key, size, sha):
        raise SystemExit(f"post-upload deep verification FAILED for {key}")
    client.put_object(
        Bucket=bucket, Key=key + ".sha256", Body=f"{sha}  {path.name}\n".encode()
    )
    print(f"VERIFIED (re-downloaded + hashed): {key} sha256={sha}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="local files to back up")
    parser.add_argument("--prefix", default="backup/", help="remote key prefix")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="only check remote size+sha against local, no upload",
    )
    args = parser.parse_args()

    client, bucket = _s3()
    prefix = args.prefix if args.prefix.endswith("/") else args.prefix + "/"
    failed = False
    for name in args.files:
        path = Path(name)
        if not path.is_file():
            print(f"MISSING local file: {path}", file=sys.stderr)
            failed = True
            continue
        key = prefix + path.name
        if args.verify_only:
            ok = _deep_verify(
                client, bucket, key, path.stat().st_size, sha256_file(path)
            )
            print(("OK (deep) " if ok else "MISMATCH/ABSENT ") + key)
            failed = failed or not ok
        else:
            upload_file(client, bucket, key, path)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
