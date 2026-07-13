"""Tests for corpus dedup analysis (exact / amendment / near-dup clustering)."""

import pytest

from ingest import dedup


def test_content_hash_stable_and_content_sensitive():
    assert dedup.content_hash("ტექსტი") == dedup.content_hash("ტექსტი")
    assert dedup.content_hash("ა") != dedup.content_hash("ბ")


def test_exact_clusters_only_size_gt_one_and_skip_falsy_keys():
    clusters = dedup.cluster_by_key(
        [("a", "h1"), ("b", "h1"), ("c", "h2"), ("d", None), ("e", "")], "exact"
    )
    assert len(clusters) == 1
    c = clusters[0]
    assert c.kind == "exact" and c.members == ["a", "b"] and c.size == 2


def test_amendment_clusters_flag_oversized_as_suspect():
    pairs = [(f"d{i}", "R-BIG") for i in range(dedup.LARGE_AMENDMENT_CLUSTER + 1)]
    pairs += [("x", "R-OK"), ("y", "R-OK")]
    clusters = {c.key: c for c in dedup.cluster_by_key(pairs, "amendment")}
    assert clusters["R-BIG"].suspect is True
    assert clusters["R-OK"].suspect is False


def test_cluster_stats_summary():
    clusters = dedup.cluster_by_key([("a", "k"), ("b", "k"), ("c", "k")], "exact")
    stats = dedup.cluster_stats(clusters)
    assert stats == {"clusters": 1, "docs_in_clusters": 3, "largest": 3, "suspect": 0}


def test_near_dup_groups_similar_bodies():
    pytest.importorskip("datasketch")
    base = " ".join(f"word{i}" for i in range(60))
    near = base + " word60 word61"          # ~97% overlap
    far = " ".join(f"other{i}" for i in range(60))
    res = dedup.near_dup_clusters([("a", base), ("b", near), ("c", far)])
    assert not res.skipped
    members = {frozenset(c.members) for c in res.clusters}
    assert frozenset({"a", "b"}) in members
    assert all("c" not in c.members for c in res.clusters)


def test_near_dup_skips_gracefully_without_datasketch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "datasketch":
            raise ImportError("simulated missing datasketch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    res = dedup.near_dup_clusters([("a", "x y z"), ("b", "x y z")])
    assert res.skipped and res.reason
