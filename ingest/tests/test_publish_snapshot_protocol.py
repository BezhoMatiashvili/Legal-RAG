"""Offline safety tests for the manifest-last Qdrant publication protocol."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "serverless"))

import publish_snapshot  # noqa: E402
import qdrant_boot  # noqa: E402

REMOTE_APPROVAL = {"PUBLISH_REMOTE_APPROVED": "1"}
PRUNE_APPROVAL = {"ARTIFACT_PRUNE_APPROVED": "1"}


def _manifest(blob: bytes = b"snapshot", *, points: int = 7) -> dict:
    generation_id = "gen_20260713_publish_test"
    return {
        "schema_version": 2,
        "snapshot": "main.snapshot",
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
        "points_count": points,
        "collection": f"georgian_legal__gen_{generation_id}",
        "qdrant_version": "1.15.0",
        "generation_id": generation_id,
        "generation_manifest_sha256": "1" * 64,
        "point_identity": {
            "schema_version": 1,
            "generation_id": generation_id,
            "embedding_model": "BAAI/bge-m3",
            "embedding_revision": "a" * 40,
            "tokenizer_model": "BAAI/bge-m3",
            "tokenizer_revision": "b" * 40,
            "reranker_model": "BAAI/bge-reranker-v2-m3",
            "reranker_revision": "c" * 40,
            "vector_space_id": "2" * 64,
            "chunking_fingerprint": "3" * 64,
            "document_header": True,
            "retrieval_fingerprint": "4" * 64,
        },
        "vector_space": {
            "dense_name": "dense",
            "dense_dimension": 1024,
            "distance": "cosine",
            "sparse_name": "sparse",
        },
    }


def _write_private_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    path.chmod(0o600)


def _write_private_snapshot(path: Path, blob: bytes) -> None:
    path.write_bytes(blob)
    path.chmod(0o600)


def test_create_is_disabled_before_configuration_or_qdrant_access(
    monkeypatch, tmp_path
):
    publish_dir = tmp_path / "publish"
    monkeypatch.setattr(publish_snapshot, "PUBLISH_DIR", publish_dir)
    monkeypatch.setattr(
        publish_snapshot,
        "load_config",
        lambda: pytest.fail("disabled create must not load configuration"),
    )

    with pytest.raises(SystemExit, match="Legacy --create is disabled"):
        publish_snapshot.main(["--create"])

    assert not publish_dir.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 1},
        {"collection": "georgian_legal"},
        {"generation_id": "v1"},
    ],
)
def test_local_manifest_reuses_serverless_schema_v2_validation(
    monkeypatch, tmp_path, change
):
    manifest = {**_manifest(), **change}
    manifest_path = tmp_path / "publish" / "manifest.json"
    _write_private_manifest(manifest_path, manifest)
    monkeypatch.setattr(publish_snapshot, "MANIFEST", manifest_path)

    with pytest.raises(SystemExit, match="immutable schema-v2 generation"):
        publish_snapshot._load_manifest()


def test_size_only_remote_object_is_never_accepted():
    manifest = _manifest()
    assert not publish_snapshot._remote_snapshot_matches(
        {"ContentLength": manifest["size_bytes"], "Metadata": {}}, manifest
    )
    assert not publish_snapshot._remote_snapshot_matches(
        {
            "ContentLength": manifest["size_bytes"],
            "Metadata": {"sha256": "0" * 64},
        },
        manifest,
    )
    assert publish_snapshot._remote_snapshot_matches(
        {
            "ContentLength": manifest["size_bytes"],
            "Metadata": {"sha256": manifest["sha256"]},
        },
        manifest,
    )


class _FakeS3:
    def __init__(self, blob: bytes, manifest: dict):
        self.blob = blob
        self.manifest = manifest
        self.calls = []
        self.metadata = None
        self.parts = {}
        self.complete = False

    def head_object(self, **kwargs):
        self.calls.append(("head", kwargs["Key"]))
        if not self.complete:
            raise FileNotFoundError(kwargs["Key"])
        return {"ContentLength": len(self.blob), "Metadata": self.metadata}

    def create_multipart_upload(self, **kwargs):
        self.calls.append(("create", kwargs["Key"]))
        self.metadata = kwargs["Metadata"]
        return {"UploadId": "upload-1"}

    def upload_part(self, **kwargs):
        self.calls.append(("part", kwargs["PartNumber"]))
        self.parts[kwargs["PartNumber"]] = kwargs["Body"]
        return {"ETag": f"etag-{kwargs['PartNumber']}"}

    def complete_multipart_upload(self, **kwargs):
        self.calls.append(("complete", kwargs["Key"]))
        assert b"".join(self.parts[n] for n in sorted(self.parts)) == self.blob
        self.complete = True

    def put_object(self, **kwargs):
        self.calls.append(("put", kwargs["Key"]))
        assert kwargs["Key"] == "publish/manifest.json"
        assert json.loads(kwargs["Body"])["sha256"] == self.manifest["sha256"]


class _ConditionalActivator:
    capability = publish_snapshot.REMOTE_ACTIVATION_CAPABILITY

    def activate(self, *, client, bucket, key, body, metadata):
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            Metadata=dict(metadata),
        )


def test_upload_binds_sha_metadata_and_publishes_manifest_last(monkeypatch, tmp_path):
    blob = b"abcdefghij"
    manifest = _manifest(blob)
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(mode=0o700)
    _write_private_snapshot(publish_dir / manifest["snapshot"], blob)
    manifest_path = publish_dir / "manifest.json"
    _write_private_manifest(manifest_path, manifest)
    state_path = publish_dir / "upload_state.json"
    fake = _FakeS3(blob, manifest)

    monkeypatch.setattr(publish_snapshot, "PUBLISH_DIR", publish_dir)
    monkeypatch.setattr(publish_snapshot, "MANIFEST", manifest_path)
    monkeypatch.setattr(publish_snapshot, "UPLOAD_STATE", state_path)
    monkeypatch.setattr(publish_snapshot, "PART_SIZE", 4)
    monkeypatch.setattr(publish_snapshot, "_s3", lambda: (fake, "bucket"))

    publish_snapshot.upload(
        apply=True,
        cold_restore_confirmed=True,
        environ=REMOTE_APPROVAL,
        manifest_activator=_ConditionalActivator(),
    )

    assert fake.metadata == {"sha256": manifest["sha256"]}
    assert fake.calls[-1] == ("put", "publish/manifest.json")
    assert ("complete", "publish/main.snapshot") in fake.calls
    assert fake.calls.index(("complete", "publish/main.snapshot")) < len(fake.calls) - 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "completed"
    assert state["sha256"] == manifest["sha256"]
    assert state_path.stat().st_mode & 0o777 == 0o600


def test_upload_retries_transient_part_failure_before_conditional_activation(
    monkeypatch, tmp_path
):
    blob = b"retry-me"
    manifest = _manifest(blob)

    class RetryingS3(_FakeS3):
        def __init__(self, blob, manifest):
            super().__init__(blob, manifest)
            self.attempts = 0

        def upload_part(self, **kwargs):
            self.attempts += 1
            if self.attempts < 3:
                raise RuntimeError("transient gateway error")
            return super().upload_part(**kwargs)

    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(mode=0o700)
    _write_private_snapshot(publish_dir / manifest["snapshot"], blob)
    manifest_path = publish_dir / "manifest.json"
    _write_private_manifest(manifest_path, manifest)
    fake = RetryingS3(blob, manifest)
    waits: list[int] = []
    monkeypatch.setattr(publish_snapshot, "PUBLISH_DIR", publish_dir)
    monkeypatch.setattr(publish_snapshot, "MANIFEST", manifest_path)
    monkeypatch.setattr(
        publish_snapshot, "UPLOAD_STATE", publish_dir / "upload_state.json"
    )
    monkeypatch.setattr(publish_snapshot, "PART_SIZE", len(blob))
    monkeypatch.setattr(publish_snapshot, "_s3", lambda: (fake, "bucket"))
    monkeypatch.setattr(publish_snapshot.time, "sleep", waits.append)

    publish_snapshot.upload(
        apply=True,
        cold_restore_confirmed=True,
        environ=REMOTE_APPROVAL,
        manifest_activator=_ConditionalActivator(),
    )

    assert fake.attempts == 3
    assert waits == [5, 10]
    assert fake.calls[-1] == ("put", "publish/manifest.json")


def test_upload_rehashes_local_snapshot_before_trusting_manifest_metadata(
    monkeypatch, tmp_path
):
    manifest = _manifest(b"right bytes")
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(mode=0o700)
    _write_private_snapshot(
        publish_dir / manifest["snapshot"], b"wrong bytes"
    )  # same length
    manifest_path = publish_dir / "manifest.json"
    _write_private_manifest(manifest_path, manifest)
    monkeypatch.setattr(publish_snapshot, "PUBLISH_DIR", publish_dir)
    monkeypatch.setattr(publish_snapshot, "MANIFEST", manifest_path)
    monkeypatch.setattr(
        publish_snapshot,
        "_s3",
        lambda: pytest.fail("tampered local bytes must fail before S3 access"),
    )

    with pytest.raises(SystemExit, match="refusing to attach trusted metadata"):
        publish_snapshot.upload(
            apply=True,
            cold_restore_confirmed=True,
            environ=REMOTE_APPROVAL,
            manifest_activator=_ConditionalActivator(),
        )


def test_upload_never_overwrites_existing_mismatched_snapshot_object(
    monkeypatch, tmp_path
):
    blob = b"immutable snapshot"
    manifest = _manifest(blob)
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir(mode=0o700)
    _write_private_snapshot(publish_dir / manifest["snapshot"], blob)
    manifest_path = publish_dir / "manifest.json"
    _write_private_manifest(manifest_path, manifest)
    fake = _FakeS3(blob, manifest)
    fake.complete = True
    fake.metadata = {"sha256": "0" * 64}
    monkeypatch.setattr(publish_snapshot, "PUBLISH_DIR", publish_dir)
    monkeypatch.setattr(publish_snapshot, "MANIFEST", manifest_path)
    monkeypatch.setattr(
        publish_snapshot, "UPLOAD_STATE", publish_dir / "upload_state.json"
    )
    monkeypatch.setattr(publish_snapshot, "_s3", lambda: (fake, "bucket"))

    with pytest.raises(RuntimeError, match="Immutable snapshot keys are never reused"):
        publish_snapshot.upload(
            apply=True,
            cold_restore_confirmed=True,
            environ=REMOTE_APPROVAL,
            manifest_activator=_ConditionalActivator(),
        )

    assert fake.calls == [("head", "publish/main.snapshot")]


def test_manifest_put_refuses_size_only_snapshot():
    manifest = _manifest()

    class SizeOnly:
        def __init__(self):
            self.put_called = False

        def head_object(self, **kwargs):
            return {"ContentLength": manifest["size_bytes"], "Metadata": {}}

        def put_object(self, **kwargs):
            self.put_called = True

    client = SizeOnly()
    with pytest.raises(RuntimeError, match="failed integrity identity"):
        publish_snapshot._publish_manifest_object(
            client,
            "bucket",
            manifest,
            activator=_ConditionalActivator(),
        )
    assert client.put_called is False


@pytest.mark.parametrize(
    "operation", [publish_snapshot.upload, publish_snapshot.verify]
)
def test_upload_and_verify_require_apply_plus_remote_approval(monkeypatch, operation):
    monkeypatch.setattr(
        publish_snapshot,
        "_load_manifest",
        lambda: pytest.fail("approval must fail before manifest or remote access"),
    )
    with pytest.raises(SystemExit, match="--apply.*PUBLISH_REMOTE_APPROVED=1"):
        operation(apply=False, cold_restore_confirmed=True, environ=REMOTE_APPROVAL)
    with pytest.raises(SystemExit, match="--apply.*PUBLISH_REMOTE_APPROVED=1"):
        operation(apply=True, cold_restore_confirmed=True, environ={})


def test_upload_and_verify_retain_explicit_cold_restore_gate(monkeypatch):
    monkeypatch.delenv("PUBLISH_COLD_RESTORE_CONFIRMED", raising=False)
    with pytest.raises(SystemExit, match="Disable FlashBoot"):
        publish_snapshot.upload(apply=True, environ=REMOTE_APPROVAL)
    with pytest.raises(SystemExit, match="Disable FlashBoot"):
        publish_snapshot.verify(apply=True, environ=REMOTE_APPROVAL)


def test_upload_blocks_before_s3_without_proven_conditional_activation(
    monkeypatch, tmp_path
):
    manifest = _manifest()
    publish_dir = tmp_path / "publish"
    manifest_path = publish_dir / "manifest.json"
    _write_private_manifest(manifest_path, manifest)
    _write_private_snapshot(publish_dir / manifest["snapshot"], b"snapshot")
    monkeypatch.setattr(publish_snapshot, "PUBLISH_DIR", publish_dir)
    monkeypatch.setattr(publish_snapshot, "MANIFEST", manifest_path)
    monkeypatch.setattr(
        publish_snapshot,
        "_s3",
        lambda: pytest.fail("capability gate must fail before S3 access"),
    )

    with pytest.raises(SystemExit, match="conditional compare-and-swap"):
        publish_snapshot.upload(
            apply=True,
            cold_restore_confirmed=True,
            environ=REMOTE_APPROVAL,
        )


def _restore_fixture(monkeypatch, tmp_path, *, restored_points):
    manifest = _manifest(points=7)
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir()
    (publish_dir / manifest["snapshot"]).write_bytes(b"snapshot")
    (publish_dir / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)
    recovered = False

    def fake_http(method, path, body=None, timeout=10):
        nonlocal recovered
        if method == "GET" and path.startswith("/collections/"):
            if not recovered:
                raise RuntimeError("cold worker: collection absent")
            return {"result": {"points_count": restored_points}}
        if method == "PUT" and "snapshots/recover" in path:
            recovered = True
            return {"result": True}
        raise AssertionError((method, path, body, timeout))

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)
    return publish_dir


@pytest.mark.parametrize("restored_points", [None, 6, 8])
def test_restore_requires_exact_point_count_and_never_marks_mismatch_active(
    monkeypatch, tmp_path, restored_points
):
    publish_dir = _restore_fixture(
        monkeypatch, tmp_path, restored_points=restored_points
    )

    out = qdrant_boot.maybe_restore()

    assert out["status"] == "error"
    assert out["code"] == "point_count_mismatch"
    assert out["expected_points"] == 7
    assert not (publish_dir / "ACTIVE").exists()


def test_active_marker_requires_exact_live_point_count(monkeypatch, tmp_path):
    manifest = _manifest(points=7)
    publish_dir = tmp_path / "publish"
    publish_dir.mkdir()
    (publish_dir / manifest["snapshot"]).write_bytes(b"snapshot")
    (publish_dir / "manifest.json").write_text(json.dumps(manifest))
    (publish_dir / "ACTIVE").write_text(json.dumps(manifest))
    monkeypatch.setattr(qdrant_boot, "VOLUME_ROOT", tmp_path)

    def fake_http(method, path, body=None, timeout=10):
        assert method == "GET"
        return {"result": {"points_count": 8}}

    monkeypatch.setattr(qdrant_boot, "_http", fake_http)

    out = qdrant_boot.maybe_restore()

    assert out["status"] == "error"
    assert out["code"] == "point_count_mismatch"
    assert out["points"] == 8


def test_point_count_participates_in_active_identity():
    manifest = _manifest(points=7)
    assert qdrant_boot.needs_restore(manifest, dict(manifest)) is False
    assert qdrant_boot.needs_restore(manifest, dict(manifest, points_count=8)) is True


def test_verify_rejects_extra_remote_points(monkeypatch, tmp_path):
    manifest = _manifest(points=7)

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def call(self, operation, timeout=None):
            if operation == "refresh":
                result = {
                    "status": "restored",
                    "snapshot": manifest["snapshot"],
                    "generation_id": manifest["generation_id"],
                    "generation_manifest_sha256": manifest[
                        "generation_manifest_sha256"
                    ],
                    "points": manifest["points_count"],
                    "identity_matched_points": manifest["points_count"],
                }
                return {"result": json.dumps(result)}
            if operation == "health":
                return {"result": json.dumps({"ok": True, "points": 8}), "sysinfo": {}}
            raise AssertionError(operation)

    import ingest.remote_search

    monkeypatch.setattr(publish_snapshot, "_load_manifest", lambda: manifest)
    monkeypatch.setattr(publish_snapshot, "PUBLISH_DIR", tmp_path / "publish")
    monkeypatch.setattr(
        publish_snapshot,
        "load_config",
        lambda: SimpleNamespace(runpod_endpoint_id="endpoint", runpod_api_key="key"),
    )
    monkeypatch.setattr(ingest.remote_search, "RunPodQueueClient", Client)

    with pytest.raises(SystemExit, match="worker points=8, manifest=7"):
        publish_snapshot.verify(
            apply=True,
            cold_restore_confirmed=True,
            environ=REMOTE_APPROVAL,
        )


class _CleanupS3:
    def __init__(self):
        self.mutations: list[tuple] = []

    def list_objects_v2(self, **kwargs):
        return {
            "Contents": [
                {"Key": "publish/main.snapshot"},
                {"Key": "publish/previous.snapshot"},
                {"Key": "publish/manifest.json"},
            ]
        }

    def delete_object(self, **kwargs):
        self.mutations.append(("delete", kwargs["Key"]))

    def list_multipart_uploads(self, **kwargs):
        return {"Uploads": [{"Key": "publish/orphan.snapshot", "UploadId": "orphan-1"}]}

    def abort_multipart_upload(self, **kwargs):
        self.mutations.append(("abort", kwargs["Key"], kwargs["UploadId"]))


def test_cleanup_is_permanently_inventory_only(monkeypatch, tmp_path):
    manifest = _manifest()
    publish_dir = tmp_path / "publish"
    manifest_path = publish_dir / "manifest.json"
    _write_private_manifest(manifest_path, manifest)
    snapshot = publish_dir / manifest["snapshot"]
    snapshot.write_bytes(b"snapshot")
    client = _CleanupS3()
    monkeypatch.setattr(publish_snapshot, "PUBLISH_DIR", publish_dir)
    monkeypatch.setattr(publish_snapshot, "MANIFEST", manifest_path)
    monkeypatch.setattr(publish_snapshot, "_s3", lambda: (client, "bucket"))

    publish_snapshot.cleanup()

    assert snapshot.exists()
    assert client.mutations == []

    with pytest.raises(SystemExit, match="permanently disabled"):
        publish_snapshot.cleanup(apply=True, environ={})
    assert snapshot.exists()
    assert client.mutations == []

    with pytest.raises(SystemExit, match="permanently disabled"):
        publish_snapshot.cleanup(apply=True, environ=PRUNE_APPROVAL)
    assert snapshot.exists()
    assert client.mutations == []


def test_local_publisher_lock_rejects_concurrent_operation(monkeypatch, tmp_path):
    publish_dir = tmp_path / "publish"
    monkeypatch.setattr(publish_snapshot, "PUBLISH_DIR", publish_dir)

    with publish_snapshot.publisher_lock():
        with pytest.raises(publish_snapshot.PublisherLockedError):
            with publish_snapshot.publisher_lock():
                pytest.fail("concurrent publisher acquired the local lock")

    lock = publish_dir / ".publisher.lock"
    assert lock.stat().st_mode & 0o777 == 0o600
    assert publish_dir.stat().st_mode & 0o777 == 0o700
