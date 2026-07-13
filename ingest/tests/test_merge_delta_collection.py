from types import SimpleNamespace

import pytest
from qdrant_client import models

from scripts.merge_delta_collection import merge_collection


def _point(doc, index, count=None, content_hash=None):
    payload = {
        "source": "matsne",
        "document_id": doc,
        "chunk_index": index,
        "content_hash": content_hash,
    }
    if count is not None:
        payload["document_chunk_count"] = count
    return SimpleNamespace(
        id=f"{doc}-{index}",
        payload=payload,
        vector={"dense": [0.1, 0.2], "sparse": models.SparseVector(indices=[1], values=[0.5])},
    )


class _Client:
    def __init__(self, points):
        self.points = list(points)
        self.upserts = []
        self.deletes = []
        self.events = []

    def scroll(self, collection_name, limit, with_vectors, with_payload, offset=None):
        start = offset or 0
        batch = self.points[start:start + limit]
        next_offset = start + len(batch)
        return batch, (next_offset if next_offset < len(self.points) else None)

    def upsert(self, collection_name, points, wait=False):
        self.events.append("upsert")
        self.upserts.extend(points)
        assert wait is True

    def delete(self, collection_name, points_selector, wait=False):
        self.events.append("delete")
        self.deletes.append(points_selector)
        assert wait is True


def test_merge_ignores_old_source_tail_and_deletes_destination_tail_last():
    client = _Client([
        _point("A", 0, 2, "new"),
        _point("A", 1, 2, "new"),
        _point("A", 2, 3, "old"),  # stale source tail from the previous longer version
        _point("B", 0, 1, "current"),
    ])

    merged = merge_collection(client, "delta", "main", batch_size=2)

    assert merged == 3
    assert {point.id for point in client.upserts} == {"A-0", "A-1", "B-0"}
    assert client.events[-1] == "delete"
    assert client.deletes
    assert len(client.deletes[0].filter.should) == 2


def test_merge_fails_closed_when_current_document_is_partial():
    client = _Client([_point("A", 0, 2, "new")])

    with pytest.raises(RuntimeError, match="completeness preflight"):
        merge_collection(client, "delta", "main")

    assert client.upserts == []
    assert client.deletes == []


def test_merge_rejects_legacy_chunk_zero_without_completeness_marker():
    client = _Client([_point("A", 0, None, "legacy")])

    with pytest.raises(RuntimeError, match="re-embed it with document_chunk_count"):
        merge_collection(client, "delta", "main")

    assert client.upserts == []
    assert client.deletes == []
