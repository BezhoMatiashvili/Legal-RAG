"""I5: build_snapshot_delta — additive v2-delta snapshot for specific doc ids.

Covers: newest-run-wins dedup, record schema parity with the v1 snapshot path, skip of
ids already in v1, quarantine ⇒ failure, the v1 write-guard, and the index-parity
classification (match / recovered-from-older-run / mismatch / missing) with stubbed
payload hashes. No Qdrant, no real artifacts — everything under tmp_path.
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from build_snapshot_delta import build_delta, already_in_v1  # noqa: E402

from ingest import dedup, hygiene, structure  # noqa: E402
from ingest.snapshot import _snapshot_record  # noqa: E402
from ingest.sources import normalize  # noqa: E402

BODY_OLD = "მუხლი 1. საქართველოს კანონი ვრცელდება ყველა პირზე. " * 8
BODY_NEW = "მუხლი 1. ეს კანონი განსაზღვრავს ქონების იჯარით გაცემის წესს. " * 8


def item(slug, body):
    return {"slug": slug, "title": f"title {slug}", "date": "2020-01-01",
            "body_markdown": body, "source_url": f"http://x/{slug}",
            "content_kind": "ruling_full_text", "content_complete": True,
            "extraction_status": "full_text", "source_binary_url": f"http://x/{slug}.pdf"}


def write_run(artifacts, source, run, items):
    d = artifacts / source / "runs" / run
    d.mkdir(parents=True)
    with open(d / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


@pytest.fixture()
def tree(tmp_path):
    artifacts = tmp_path / "artifacts"
    out_root = tmp_path / "snapshots"
    write_run(artifacts, "tbappeal", "20260101T000000Z_a", [item("doc-a", BODY_OLD)])
    write_run(artifacts, "tbappeal", "20260201T000000Z_b", [item("doc-a", BODY_NEW)])
    (out_root / "v1" / "docs").mkdir(parents=True)
    return artifacts, out_root


def read_delta(out_root, source):
    path = out_root / "v2-delta" / "docs" / f"{source}.jsonl"
    return {r["document_id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def test_newest_run_wins_without_verification(tree):
    artifacts, out_root = tree
    manifest = build_delta({"tbappeal": ["doc-a"]}, artifacts_root=artifacts, out_root=out_root,
                           payload_hashes=None)
    assert manifest["ok"]
    rec = read_delta(out_root, "tbappeal")["doc-a"]
    assert rec["body_markdown"] == hygiene.clean_text(BODY_NEW)
    assert rec["source_run"] == "20260201T000000Z_b"


def test_record_matches_v1_snapshot_path_except_version(tree):
    artifacts, out_root = tree
    build_delta({"tbappeal": ["doc-a"]}, artifacts_root=artifacts, out_root=out_root,
                payload_hashes=None)
    rec = read_delta(out_root, "tbappeal")["doc-a"]
    doc = normalize("tbappeal", item("doc-a", BODY_NEW))
    clean = hygiene.clean_text(BODY_NEW)
    expected = _snapshot_record("tbappeal", doc, clean, dedup.content_hash(clean),
                                structure.detect(clean), "20260201T000000Z_b")
    expected["snapshot_version"] = "v2-delta"
    assert rec == expected


def test_ids_already_in_v1_are_skipped(tree):
    artifacts, out_root = tree
    v1_file = out_root / "v1" / "docs" / "tbappeal.jsonl"
    v1_file.write_text(
        json.dumps({"snapshot_version": "1", "doc_id": "tbappeal:doc-a", "source": "tbappeal",
                    "document_id": "doc-a", "body_markdown": "frozen"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    manifest = build_delta({"tbappeal": ["doc-a"]}, artifacts_root=artifacts, out_root=out_root,
                           payload_hashes=None)
    assert manifest["ok"]
    assert manifest["sources"]["tbappeal"]["skipped_in_v1"] == ["doc-a"]
    assert manifest["sources"]["tbappeal"]["written"] == 0
    assert read_delta(out_root, "tbappeal") == {}
    assert v1_file.read_text(encoding="utf-8").count("\n") == 1  # v1 untouched


def test_already_in_v1_reads_ids_without_full_parse(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    recs = [{"snapshot_version": "1", "doc_id": f"s:{i}", "source": "s", "document_id": str(i),
             "body_markdown": "x" * 50} for i in range(5)]
    with open(docs / "s.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert already_in_v1("s", {"1", "3", "99"}, docs) == {"1", "3"}
    assert already_in_v1("s", set(), docs) == set()
    assert already_in_v1("missing_source", {"1"}, docs) == set()


def test_quarantined_doc_fails_the_build(tree):
    artifacts, out_root = tree
    write_run(artifacts, "tbappeal", "20260301T000000Z_c", [item("doc-empty", "   ")])
    manifest = build_delta({"tbappeal": ["doc-empty"]}, artifacts_root=artifacts,
                           out_root=out_root, payload_hashes=None)
    assert not manifest["ok"]
    assert "doc-empty" in manifest["sources"]["tbappeal"]["quarantined"]
    assert read_delta(out_root, "tbappeal") == {}


def test_refuses_v1_version(tree):
    artifacts, out_root = tree
    with pytest.raises(ValueError, match="v1 is frozen"):
        build_delta({"tbappeal": ["doc-a"]}, artifacts_root=artifacts, out_root=out_root,
                    version="v1", payload_hashes=None)


def test_parity_match_on_newest_raw_hash(tree):
    artifacts, out_root = tree
    hashes = {("tbappeal", "doc-a"): dedup.content_hash(BODY_NEW)}
    manifest = build_delta({"tbappeal": ["doc-a"]}, artifacts_root=artifacts, out_root=out_root,
                           payload_hashes=hashes)
    assert manifest["ok"]
    stats = manifest["sources"]["tbappeal"]
    assert stats["written"] == 1 and stats["recovered_from_older_run"] == {}


def test_parity_recovers_from_older_run(tree):
    artifacts, out_root = tree
    # index still holds the OLD revision — the newest artifact must be rejected
    hashes = {("tbappeal", "doc-a"): dedup.content_hash(BODY_OLD)}
    manifest = build_delta({"tbappeal": ["doc-a"]}, artifacts_root=artifacts, out_root=out_root,
                           payload_hashes=hashes)
    assert manifest["ok"]
    stats = manifest["sources"]["tbappeal"]
    assert stats["recovered_from_older_run"] == {"doc-a": "20260101T000000Z_a"}
    rec = read_delta(out_root, "tbappeal")["doc-a"]
    assert rec["body_markdown"] == hygiene.clean_text(BODY_OLD)


def test_parity_match_on_cleaned_hash(tree):
    artifacts, out_root = tree
    # snapshot-embedded lineage: the index hashed the CLEANED body
    hashes = {("tbappeal", "doc-a"): dedup.content_hash(hygiene.clean_text(BODY_NEW))}
    manifest = build_delta({"tbappeal": ["doc-a"]}, artifacts_root=artifacts, out_root=out_root,
                           payload_hashes=hashes)
    assert manifest["ok"]
    assert manifest["sources"]["tbappeal"]["written"] == 1


def test_parity_unresolved_mismatch_fails(tree):
    artifacts, out_root = tree
    hashes = {("tbappeal", "doc-a"): "0" * 64}
    manifest = build_delta({"tbappeal": ["doc-a"]}, artifacts_root=artifacts, out_root=out_root,
                           payload_hashes=hashes)
    assert not manifest["ok"]
    assert manifest["sources"]["tbappeal"]["mismatch_unresolved"] == ["doc-a"]


def test_missing_from_index_fails(tree):
    artifacts, out_root = tree
    manifest = build_delta({"tbappeal": ["doc-a"]}, artifacts_root=artifacts, out_root=out_root,
                           payload_hashes={})
    assert not manifest["ok"]
    assert manifest["sources"]["tbappeal"]["missing_from_index"] == ["doc-a"]


def test_chunk_alignment_risk_flagged_for_raw_lineage(tree):
    artifacts, out_root = tree
    dirty = BODY_NEW + "\x00tail"  # clean_text strips the NUL → cleaned != raw
    write_run(artifacts, "tbappeal", "20260401T000000Z_d", [item("doc-dirty", dirty)])
    hashes = {("tbappeal", "doc-dirty"): dedup.content_hash(dirty)}  # index hashed the RAW body
    manifest = build_delta({"tbappeal": ["doc-dirty"]}, artifacts_root=artifacts,
                           out_root=out_root, payload_hashes=hashes)
    assert manifest["ok"]
    assert manifest["sources"]["tbappeal"]["chunk_alignment_risk_ids"] == ["doc-dirty"]
