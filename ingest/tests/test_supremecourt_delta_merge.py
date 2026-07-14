import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from qdrant_client import models

import scripts.merge_delta_collection as merge
from ingest.qdrant_store import point_id


SOURCE = "supremecourt"
RUN_ID = "run-1"
SRC = f"georgian_legal_delta_{SOURCE}_{RUN_ID}"
DST = "georgian_legal"


def _point(
    document_id: str,
    chunk_index: int,
    chunk_count: int,
    *,
    source: str = SOURCE,
    content_hash: str = "current",
    point_uuid: str | None = None,
    marker: str | None = None,
):
    return SimpleNamespace(
        id=point_uuid or point_id(source, document_id, chunk_index),
        payload={
            "source": source,
            "document_id": document_id,
            "chunk_index": chunk_index,
            "document_chunk_count": chunk_count,
            "content_hash": content_hash,
            "marker": marker,
        },
        vector={"dense": [0.1, 0.2]},
    )


def _matches_condition(condition, payload: dict) -> bool:
    if isinstance(condition, models.Filter):
        return _matches_filter(condition, payload)
    value = payload.get(condition.key)
    if condition.match is not None:
        if isinstance(condition.match, models.MatchValue):
            return value == condition.match.value
        if isinstance(condition.match, models.MatchAny):
            return value in condition.match.any
    if condition.range is not None:
        lower = condition.range.gte
        upper = condition.range.lt
        if lower is not None and (value is None or value < lower):
            return False
        if upper is not None and (value is None or value >= upper):
            return False
        return True
    return True


def _matches_filter(filter_value, payload: dict) -> bool:
    must = filter_value.must or []
    should = filter_value.should or []
    return all(_matches_condition(item, payload) for item in must) and (
        not should or any(_matches_condition(item, payload) for item in should)
    )


class _Client:
    def __init__(self, collections: dict[str, list]):
        self.collections = {
            name: {str(point.id): copy.deepcopy(point) for point in points}
            for name, points in collections.items()
        }
        self.events: list[str] = []
        self.snapshot_blob = b"durable rollback snapshot"
        self.snapshot_state = None
        self.fail_source_delete = False

    def scroll(
        self,
        collection_name,
        limit,
        with_vectors,
        with_payload,
        offset=None,
        scroll_filter=None,
    ):
        points = sorted(
            self.collections[collection_name].values(), key=lambda point: str(point.id)
        )
        if scroll_filter is not None:
            points = [
                point
                for point in points
                if _matches_filter(scroll_filter, point.payload or {})
            ]
        start = offset or 0
        batch = points[start : start + limit]
        next_offset = start + len(batch)
        return batch, (next_offset if next_offset < len(points) else None)

    def upsert(self, collection_name, points, wait=False):
        assert wait is True
        self.events.append("upsert")
        for point in points:
            self.collections[collection_name][str(point.id)] = copy.deepcopy(point)

    def delete(self, collection_name, points_selector, wait=False):
        assert wait is True
        self.events.append("delete-points")
        selected = points_selector.filter
        self.collections[collection_name] = {
            key: point
            for key, point in self.collections[collection_name].items()
            if not _matches_filter(selected, point.payload or {})
        }

    def count(self, collection_name, exact=False):
        assert exact is True
        return SimpleNamespace(count=len(self.collections[collection_name]))

    def create_snapshot(self, collection_name, wait=False):
        assert wait is True
        self.events.append("snapshot")
        self.snapshot_state = copy.deepcopy(self.collections[collection_name])
        return SimpleNamespace(
            name="main-pre-merge.snapshot",
            size=len(self.snapshot_blob),
            checksum=hashlib.sha256(self.snapshot_blob).hexdigest(),
        )

    def delete_collection(self, collection_name):
        self.events.append("delete-collection")
        if self.fail_source_delete:
            raise RuntimeError("source cleanup failed")
        del self.collections[collection_name]


def _manifest(tmp_path, document_ids: list[str], chunks: int, **updates):
    qualified = sorted(f"{SOURCE}\t{document_id}" for document_id in document_ids)
    value = {
        "schema_version": 1,
        "source": SOURCE,
        "run_id": RUN_ID,
        "collection": SRC,
        "input_sha256": "a" * 64,
        "document_ids": qualified,
        "document_ids_sha256": hashlib.sha256(
            "\n".join(qualified).encode("utf-8")
        ).hexdigest(),
        "expected_documents": len(document_ids),
        "documents": len(document_ids),
        "expected_chunks": chunks,
        "chunks": chunks,
        "points_count": chunks,
        "skipped": 0,
    }
    value.update(updates)
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _expectations(tmp_path, document_ids: list[str], chunks: int):
    return merge.load_merge_expectations(
        src=SRC,
        run_manifest_path=_manifest(tmp_path, document_ids, chunks),
        expected_source=SOURCE,
    )


def _snapshot_copy(client: _Client, lock_path):
    def copy_snapshot(collection, snapshot_name, destination):
        assert collection == DST
        assert snapshot_name == "main-pre-merge.snapshot"
        assert lock_path.is_file()
        assert client.events[-1] == "snapshot"
        client.events.append("copy-snapshot")
        destination.write_bytes(client.snapshot_blob)

    return copy_snapshot


def _restore(client: _Client, artifact):
    assert artifact.path.is_file()
    assert hashlib.sha256(artifact.path.read_bytes()).hexdigest() == artifact.sha256
    client.events.append("restore")
    client.collections[artifact.collection] = copy.deepcopy(client.snapshot_state)


def _workflow(tmp_path, client, expectations, *, delete_source=True):
    lock_path = tmp_path / "coordination" / "qdrant-write.lock"
    return merge.run_merge_workflow(
        client,
        SRC,
        DST,
        run_id=RUN_ID,
        rollback_root=tmp_path / "rollbacks",
        lock_path=lock_path,
        snapshot_copy=_snapshot_copy(client, lock_path),
        expectations=expectations,
        batch_size=2,
        delete_source=delete_source,
        restore_snapshot=_restore,
    )


def test_run_manifest_requires_canonical_ids_hash_and_exact_counts(tmp_path):
    path = _manifest(tmp_path, ["case-b", "case-a"], 3)
    expected = merge.load_merge_expectations(
        src=SRC, run_manifest_path=path, expected_source=SOURCE
    )

    assert expected.document_ids == {"case-a", "case-b"}
    assert expected.documents == 2
    assert expected.chunks == 3

    value = json.loads(path.read_text(encoding="utf-8"))
    value["document_ids_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="document_ids_sha256"):
        merge.load_merge_expectations(src=SRC, run_manifest_path=path)

    path = _manifest(tmp_path, ["case-a"], 2, points_count=3)
    with pytest.raises(RuntimeError, match="chunk/point counts"):
        merge.load_merge_expectations(src=SRC, run_manifest_path=path)


def test_older_run_manifest_requires_digest_bound_companion_ids(tmp_path):
    path = _manifest(tmp_path, ["case-a"], 1)
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("document_ids")
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RuntimeError, match="requires run-manifest document_ids"):
        merge.load_merge_expectations(src=SRC, run_manifest_path=path)

    ids_path = tmp_path / "expected_ids.json"
    ids_path.write_text(json.dumps([f"{SOURCE}\tcase-a"]), encoding="utf-8")
    expected = merge.load_merge_expectations(
        src=SRC, run_manifest_path=path, expected_ids_path=ids_path
    )
    assert expected.document_ids == {"case-a"}


def test_exact_preflight_rejects_missing_manifest_document_before_snapshot(tmp_path):
    expected = _expectations(tmp_path, ["case-a", "case-b"], 2)
    client = _Client(
        {
            SRC: [_point("case-a", 0, 1)],
            DST: [_point("existing", 0, 1, marker="preserve")],
        }
    )

    with pytest.raises(RuntimeError, match="document IDs differ"):
        _workflow(tmp_path, client, expected)

    assert client.events == []


@pytest.mark.parametrize(
    "bad_point",
    [
        _point("extra", 0, 1),
        _point("wrong-source", 0, 1, source="matsne"),
        _point("case-a", 2, 3, content_hash="old"),
        _point("case-a", 1, 2, point_uuid="not-the-uuid5"),
    ],
)
def test_exact_preflight_rejects_extra_source_document_tail_or_wrong_uuid(
    tmp_path, bad_point
):
    expected = _expectations(tmp_path, ["case-a"], 2)
    client = _Client(
        {
            SRC: [_point("case-a", 0, 2), _point("case-a", 1, 2), bad_point],
            DST: [_point("existing", 0, 1, marker="preserve")],
        }
    )

    with pytest.raises(RuntimeError, match="delta exact preflight"):
        _workflow(tmp_path, client, expected)

    assert "snapshot" not in client.events
    assert "upsert" not in client.events


def test_locked_workflow_snapshots_first_preserves_supreme_and_deletes_exact_temp(
    tmp_path,
):
    expected = _expectations(tmp_path, ["case-a"], 2)
    existing = _point("existing", 0, 1, marker="preserve")
    client = _Client(
        {
            SRC: [_point("case-a", 0, 2), _point("case-a", 1, 2)],
            DST: [existing],
        }
    )
    lock_path = tmp_path / "coordination" / "qdrant-write.lock"

    outcome = _workflow(tmp_path, client, expected)

    assert client.events.index("snapshot") < client.events.index("copy-snapshot")
    assert client.events.index("copy-snapshot") < client.events.index("upsert")
    assert client.events[-1] == "delete-collection"
    assert SRC not in client.collections
    assert str(existing.id) in client.collections[DST]
    assert client.collections[DST][str(existing.id)].payload["marker"] == "preserve"
    assert point_id(SOURCE, "case-a", 0) in client.collections[DST]
    assert point_id(SOURCE, "case-a", 1) in client.collections[DST]
    assert not lock_path.exists()
    assert outcome.documents == 1
    assert outcome.chunks == 2
    assert outcome.before_points == 1
    assert outcome.after_points == 3
    assert outcome.source_deleted is True
    assert outcome.rollback.path.is_file()
    assert outcome.rollback.manifest_path.is_file()
    rollback_manifest = json.loads(
        outcome.rollback.manifest_path.read_text(encoding="utf-8")
    )
    assert rollback_manifest["sha256"] == outcome.rollback.sha256
    assert rollback_manifest["points_count"] == 1


def test_guarded_merge_is_idempotent_and_keeps_separate_rollback_attempts(tmp_path):
    expected = _expectations(tmp_path, ["case-a"], 1)
    client = _Client(
        {
            SRC: [_point("case-a", 0, 1)],
            DST: [_point("existing", 0, 1, marker="preserve")],
        }
    )

    first = _workflow(tmp_path, client, expected, delete_source=False)
    second = _workflow(tmp_path, client, expected, delete_source=False)

    assert len(client.collections[DST]) == 2
    assert first.rollback.path != second.rollback.path
    attempts = list((tmp_path / "rollbacks" / RUN_ID).iterdir())
    assert len(attempts) == 2
    assert "restore" not in client.events


def test_post_merge_failure_restores_main_and_retains_delta(tmp_path, monkeypatch):
    expected = _expectations(tmp_path, ["case-a"], 1)
    existing = _point("existing", 0, 1, marker="preserve")
    client = _Client({SRC: [_point("case-a", 0, 1)], DST: [existing]})
    before = copy.deepcopy(client.collections[DST])

    def fail_coverage(*args, **kwargs):
        raise RuntimeError("forced post-merge failure")

    monkeypatch.setattr(merge, "verify_destination_coverage", fail_coverage)
    with pytest.raises(RuntimeError, match="forced post-merge failure"):
        _workflow(tmp_path, client, expected)

    assert client.collections[DST].keys() == before.keys()
    assert client.collections[DST][str(existing.id)].payload == existing.payload
    assert SRC in client.collections
    assert "restore" in client.events
    assert "delete-collection" not in client.events
    assert not (tmp_path / "coordination" / "qdrant-write.lock").exists()


def test_rollback_copy_hash_must_match_before_first_mutation(tmp_path):
    expected = _expectations(tmp_path, ["case-a"], 1)
    client = _Client({SRC: [_point("case-a", 0, 1)], DST: []})
    lock_path = tmp_path / "qdrant-write.lock"

    def corrupt_copy(collection, snapshot_name, destination):
        assert lock_path.is_file()
        destination.write_bytes(b"x" * len(client.snapshot_blob))

    with pytest.raises(RuntimeError, match="SHA-256 differs"):
        merge.run_merge_workflow(
            client,
            SRC,
            DST,
            run_id=RUN_ID,
            rollback_root=tmp_path / "rollbacks",
            lock_path=lock_path,
            snapshot_copy=corrupt_copy,
            expectations=expected,
            delete_source=True,
            restore_snapshot=_restore,
        )

    assert "upsert" not in client.events
    assert SRC in client.collections
    assert not lock_path.exists()


def test_default_rollback_restore_uploads_verified_copy_and_checks_count(tmp_path):
    blob = b"verified rollback bytes"
    snapshot_path = tmp_path / "rollback.snapshot"
    snapshot_path.write_bytes(blob)
    artifact = merge.RollbackArtifact(
        run_id=RUN_ID,
        collection=DST,
        snapshot_name=snapshot_path.name,
        path=snapshot_path,
        sha256=hashlib.sha256(blob).hexdigest(),
        size_bytes=len(blob),
        points_count=1,
        manifest_path=tmp_path / "rollback_manifest.json",
    )
    client = _Client({DST: []})
    captured = {}

    def recover_from_uploaded_snapshot(**kwargs):
        captured.update(kwargs)
        assert kwargs["snapshot"].read() == blob
        client.collections[DST] = {
            str(point_id(SOURCE, "restored", 0)): _point("restored", 0, 1)
        }

    client.http = SimpleNamespace(
        snapshots_api=SimpleNamespace(
            recover_from_uploaded_snapshot=recover_from_uploaded_snapshot
        )
    )

    merge.restore_rollback_snapshot(client, artifact)

    assert captured["collection_name"] == DST
    assert captured["checksum"] == artifact.sha256
    assert captured["priority"] == models.SnapshotPriority.SNAPSHOT
    assert captured["wait"] is True


def test_source_cleanup_failure_keeps_verified_main_and_does_not_rollback(tmp_path):
    expected = _expectations(tmp_path, ["case-a"], 1)
    client = _Client(
        {
            SRC: [_point("case-a", 0, 1)],
            DST: [_point("existing", 0, 1, marker="preserve")],
        }
    )
    client.fail_source_delete = True

    with pytest.raises(RuntimeError, match="source cleanup failed"):
        _workflow(tmp_path, client, expected)

    assert point_id(SOURCE, "case-a", 0) in client.collections[DST]
    assert SRC in client.collections
    assert "restore" not in client.events
    assert not (tmp_path / "coordination" / "qdrant-write.lock").exists()


def test_foreign_lock_refuses_before_snapshot_and_is_not_removed(tmp_path):
    expected = _expectations(tmp_path, ["case-a"], 1)
    client = _Client({SRC: [_point("case-a", 0, 1)], DST: []})
    lock_path = tmp_path / "coordination" / "qdrant-write.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("owner: another-session\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already held"):
        merge.run_merge_workflow(
            client,
            SRC,
            DST,
            run_id=RUN_ID,
            rollback_root=tmp_path / "rollbacks",
            lock_path=lock_path,
            snapshot_copy=_snapshot_copy(client, lock_path),
            expectations=expected,
            delete_source=True,
            restore_snapshot=_restore,
        )

    assert lock_path.read_text(encoding="utf-8") == "owner: another-session\n"
    assert client.events == []


def test_only_manifest_bound_run_scoped_source_can_be_deleted(tmp_path):
    client = _Client({"georgian_legal_delta": [], DST: []})

    with pytest.raises(RuntimeError, match="exact manifest-bound"):
        merge.run_merge_workflow(
            client,
            "georgian_legal_delta",
            DST,
            run_id="legacy-run",
            rollback_root=tmp_path / "rollbacks",
            lock_path=tmp_path / "qdrant-write.lock",
            snapshot_copy=lambda *args: None,
            delete_source=True,
            restore_snapshot=_restore,
        )

    assert client.events == []


def test_low_level_merge_rejects_same_collection_without_scanning():
    client = _Client({DST: [_point("existing", 0, 1)]})

    with pytest.raises(RuntimeError, match="must differ"):
        merge.merge_collection(client, DST, DST)

    assert client.events == []
