"""End-to-end test of the clean-corpus snapshot builder over a tiny fake artifacts tree."""

import dataclasses
import json

from ingest import snapshot
from ingest.config import load_config


def _write_run(tmp_path, source, run_id, items):
    run_dir = tmp_path / "artifacts" / source / "runs" / run_id
    run_dir.mkdir(parents=True)
    with open(run_dir / "items.jsonl", "w", encoding="utf-8") as fp:
        for it in items:
            fp.write(json.dumps(it, ensure_ascii=False) + "\n")


def _cfg(tmp_path):
    cfg = load_config()
    return dataclasses.replace(cfg, artifacts_root=tmp_path / "artifacts")


def test_snapshot_cleans_quarantines_and_manifests(tmp_path, monkeypatch):
    # Avoid loading the BGE-M3 tokenizer in a unit test.
    monkeypatch.setattr(snapshot, "_maybe_token_counter", lambda cfg: None)

    body = "მუხლი 1. სასამართლომ დაადგინა შემდეგი გარემოებანი საქმეზე."
    items = [
        {"decision_document_id": "E1", "case_no": "c1", "decision_type_name": "t", "body_markdown": body},
        {"decision_document_id": "E2", "case_no": "c2", "decision_type_name": "t", "body_markdown": ""},          # empty → quarantine
        {"decision_document_id": "E3", "case_no": "c3", "decision_type_name": "t", "body_markdown": "დად\x00გენ\x00ილება " + body},  # NUL → clean
        {"decision_document_id": "E4", "case_no": "c4", "decision_type_name": "t", "body_markdown": body},          # exact-dup of E1
    ]
    _write_run(tmp_path, "ecd", "20260101T000000Z_start-1970-01-01_end-2026-01-01", items)

    manifest = snapshot.build_snapshot(
        _cfg(tmp_path), out_root=tmp_path / "snap", sources=["ecd"], near_dup=False,
    )

    assert manifest["totals"]["clean"] == 3
    assert manifest["totals"]["quarantined"] == 1
    assert manifest["config_hash"]
    ecd = manifest["sources"]["ecd"]
    assert ecd["quarantined"] == {"empty_body": 1}
    assert ecd["dedup"]["exact"]["clusters"] == 1  # E1 == E4

    # Quarantine file preserves the dropped doc with a reason (never deleted).
    quar = [json.loads(x) for x in open(tmp_path / "snap" / "v1" / "quarantine.jsonl", encoding="utf-8")]
    assert len(quar) == 1 and quar[0]["document_id"] == "E2" and quar[0]["reason"] == "empty_body"

    # Cleaned bodies carry no NUL; E1/E4 share a content hash.
    docs = [json.loads(x) for x in open(tmp_path / "snap" / "v1" / "docs" / "ecd.jsonl", encoding="utf-8")]
    by_id = {d["document_id"]: d for d in docs}
    assert "\x00" not in by_id["E3"]["body_markdown"]
    assert by_id["E1"]["content_hash"] == by_id["E4"]["content_hash"]
    assert by_id["E1"]["structure"]["has_article"] is True
    assert by_id["E1"]["doc_id"] == "ecd:E1"


def test_snapshot_keeps_newest_run_version(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "_maybe_token_counter", lambda cfg: None)
    old = "ძველი ვერსია საკმარისი სიგრძის ტექსტით დოკუმენტისთვის აქ."
    new = "ახალი ვერსია საკმარისი სიგრძის ტექსტით დოკუმენტისთვის აქ."
    _write_run(tmp_path, "ecd", "20260101T000000Z_r1",
               [{"decision_document_id": "E1", "case_no": "c", "decision_type_name": "t", "body_markdown": old}])
    _write_run(tmp_path, "ecd", "20260202T000000Z_r2",
               [{"decision_document_id": "E1", "case_no": "c", "decision_type_name": "t", "body_markdown": new}])

    snapshot.build_snapshot(_cfg(tmp_path), out_root=tmp_path / "snap", sources=["ecd"], near_dup=False)
    docs = [json.loads(x) for x in open(tmp_path / "snap" / "v1" / "docs" / "ecd.jsonl", encoding="utf-8")]
    assert len(docs) == 1
    assert docs[0]["body_markdown"] == new  # newer run wins
    assert docs[0]["source_run"].startswith("20260202")
