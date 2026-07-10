"""I5: multi-root SnapshotBodies + the EVAL_SETS registry (v2 loader path).

The frozen-v1 pins live in test_goldset.py and are untouched; this file covers the new
additive machinery: a delta root layered under the v1 root, primary-wins collision
semantics, and the version registry evaluate.py's --golden-set flag selects from.
Real-v2-data pins (500 count, v1 byte-identity, holdout superset) are appended once
golden_set_v2.jsonl lands.
"""

import json

import pytest

from eval import goldset
from eval.goldset import EVAL_SETS, SnapshotBodies


@pytest.fixture(autouse=True)
def _clear_source_cache():
    goldset._SOURCE_CACHE.clear()
    yield
    goldset._SOURCE_CACHE.clear()


def write_docs(root, source, docs):
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f"{source}.jsonl", "w", encoding="utf-8") as f:
        for did, body in docs.items():
            f.write(json.dumps({"document_id": did, "body_markdown": body}, ensure_ascii=False) + "\n")


def test_registry_shape():
    assert set(EVAL_SETS) == {"v1", "v2"}
    v1, v2 = EVAL_SETS["v1"], EVAL_SETS["v2"]
    assert v1.version == "v1" and v2.version == "v2"
    assert v1.gold == goldset.DEFAULT_GOLD and v1.holdout == goldset.DEFAULT_HOLDOUT
    assert v1.roots == (goldset.DEFAULT_SNAPSHOT_DOCS,)
    assert v2.gold.name == "golden_set_v2.jsonl"
    assert v2.holdout.name == "holdout_doc_ids_v2.json"
    # primary root first — v1 bodies must win over any delta shadow
    assert v2.roots[0] == goldset.DEFAULT_SNAPSHOT_DOCS
    assert v2.roots[1] == goldset.V2_DELTA_DOCS


def test_single_root_behavior_unchanged(tmp_path):
    root = tmp_path / "v1"
    write_docs(root, "src", {"a": "body-a"})
    bodies = SnapshotBodies(root=root, needed={("src", "a")})
    assert bodies.body("src", "a") == "body-a"
    with pytest.raises(KeyError, match="doc not in snapshot"):
        bodies.body("src", "missing")


def test_extra_root_supplies_delta_docs(tmp_path):
    v1 = tmp_path / "v1"
    delta = tmp_path / "delta"
    write_docs(v1, "src", {"a": "body-a"})
    write_docs(delta, "src", {"b": "body-b"})
    bodies = SnapshotBodies(root=v1, needed={("src", "a"), ("src", "b")}, extra_roots=[delta])
    assert bodies.body("src", "a") == "body-a"
    assert bodies.body("src", "b") == "body-b"


def test_primary_root_wins_on_collision(tmp_path):
    v1 = tmp_path / "v1"
    delta = tmp_path / "delta"
    write_docs(v1, "src", {"a": "primary-body"})
    write_docs(delta, "src", {"a": "shadow-body"})
    bodies = SnapshotBodies(root=v1, needed={("src", "a")}, extra_roots=[delta])
    assert bodies.body("src", "a") == "primary-body"


def test_doc_missing_from_all_roots_raises_keyerror(tmp_path):
    v1 = tmp_path / "v1"
    delta = tmp_path / "delta"
    write_docs(v1, "src", {"a": "body-a"})
    write_docs(delta, "src", {"b": "body-b"})
    bodies = SnapshotBodies(root=v1, needed={("src", "nope")}, extra_roots=[delta])
    with pytest.raises(KeyError, match="doc not in snapshot: src:nope"):
        bodies.body("src", "nope")


def test_source_file_absent_from_delta_root_is_tolerated(tmp_path):
    v1 = tmp_path / "v1"
    delta = tmp_path / "delta"
    write_docs(v1, "src", {"a": "body-a"})
    delta.mkdir()  # no src.jsonl in the delta root
    bodies = SnapshotBodies(root=v1, needed={("src", "a")}, extra_roots=[delta])
    assert bodies.body("src", "a") == "body-a"


def test_source_file_absent_everywhere_raises_filenotfound(tmp_path):
    v1 = tmp_path / "v1"
    delta = tmp_path / "delta"
    v1.mkdir()
    delta.mkdir()
    bodies = SnapshotBodies(root=v1, needed={("src", "a")}, extra_roots=[delta])
    with pytest.raises(FileNotFoundError, match="no docs file for source 'src'"):
        bodies.body("src", "a")


def test_source_files_lists_existing_primary_first(tmp_path):
    v1 = tmp_path / "v1"
    delta = tmp_path / "delta"
    write_docs(v1, "src", {"a": "x"})
    write_docs(delta, "src", {"b": "y"})
    write_docs(delta, "only_delta", {"c": "z"})
    bodies = SnapshotBodies(root=v1, extra_roots=[delta])
    assert bodies.source_files("src") == [v1 / "src.jsonl", delta / "src.jsonl"]
    assert bodies.source_files("only_delta") == [delta / "only_delta.jsonl"]
    assert bodies.source_files("nowhere") == []


def test_needed_none_merges_whole_sources(tmp_path):
    v1 = tmp_path / "v1"
    delta = tmp_path / "delta"
    write_docs(v1, "src", {"a": "primary-a", "shared": "primary-shared"})
    write_docs(delta, "src", {"b": "delta-b", "shared": "delta-shared"})
    bodies = SnapshotBodies(root=v1, extra_roots=[delta])
    assert bodies.body("src", "a") == "primary-a"
    assert bodies.body("src", "b") == "delta-b"
    assert bodies.body("src", "shared") == "primary-shared"


def test_cache_shared_between_single_and_multi_root_instances(tmp_path):
    v1 = tmp_path / "v1"
    delta = tmp_path / "delta"
    write_docs(v1, "src", {"a": "body-a"})
    write_docs(delta, "src", {"b": "body-b"})
    multi = SnapshotBodies(root=v1, needed={("src", "a"), ("src", "b")}, extra_roots=[delta])
    assert multi.body("src", "b") == "body-b"
    # the v1 cache entry the multi-root load created serves a v1-only instance directly
    single = SnapshotBodies(root=v1, needed={("src", "a")})
    assert single.body("src", "a") == "body-a"
    assert (str(v1), "src") in goldset._SOURCE_CACHE
    assert (str(delta), "src") in goldset._SOURCE_CACHE
