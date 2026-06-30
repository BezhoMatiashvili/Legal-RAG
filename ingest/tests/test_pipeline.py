import dataclasses
import json

import pytest

from ingest import pipeline
from ingest.config import load_config
from ingest.embedding import Embedded, Sparse


class FakeEmbedder:
    def encode_passages(self, texts):
        return [Embedded(dense=[0.1, 0.2, 0.3, 0.4], sparse=Sparse([1, 2], [0.5, 0.6])) for _ in texts]

    def encode_query(self, text):
        return self.encode_passages([text])[0]


class FakeClient:
    """Captures upserts/deletes so we can assert ingestion behaviour offline."""

    def __init__(self):
        self.upserts = []   # (points, wait)
        self.deletes = []

    def upsert(self, collection_name, points, wait=False):
        self.upserts.append((list(points), wait))

    def delete(self, collection_name, points_selector):
        self.deletes.append(points_selector)

    def upserted_ids(self):
        return [p.id for batch, _ in self.upserts for p in batch]


def _count_tokens(text):
    return len(text.split())


def _make_cfg(tmp_path):
    cfg = load_config()
    return dataclasses.replace(
        cfg,
        collection_name="test",
        artifacts_root=tmp_path / "artifacts",
        state_dir=tmp_path / "state",
        chunk_tokens=40,
        chunk_overlap=5,
        chunk_min_tokens=1,
    )


def _write_items(cfg, source, items):
    path = pipeline.items_path(cfg, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")


def _ecd_item(doc_id, body="body text here for chunking"):
    return {
        "decision_document_id": doc_id, "case_no": f"case-{doc_id}",
        "decision_type_name": "განაჩენი", "court_name": "court",
        "decision_date": "2020-04-30", "body_markdown": body,
    }


def _run(cfg, client, source, **kw):
    return pipeline.ingest_source(cfg, client, FakeEmbedder(), _count_tokens, source, batch_size=1, **kw)


def test_basic_ingest_upserts_and_checkpoints(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_items(cfg, "ecd", [_ecd_item(1), _ecd_item(2)])
    client = FakeClient()

    docs, chunks, skipped = _run(cfg, client, "ecd")
    assert docs == 2 and chunks > 0 and skipped == 0
    assert client.upserted_ids()                      # points were written
    assert all(wait for _, wait in client.upserts)    # A1: durable (wait=True)
    # A1: checkpoint advances only to a flushed doc, and to the last one.
    ckpt = json.loads((cfg.state_dir / "ecd.json").read_text())
    assert ckpt["last_document_id"] == "2"


def test_empty_body_doc_is_skipped(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_items(cfg, "ecd", [_ecd_item(1, body=""), _ecd_item(2)])
    client = FakeClient()
    docs, chunks, skipped = _run(cfg, client, "ecd")
    assert docs == 1  # the empty-body doc produced no chunks and was not counted


def test_bad_records_are_skipped_not_fatal(tmp_path):
    cfg = _make_cfg(tmp_path)
    path = pipeline.items_path(cfg, "ecd")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join([
            "{not valid json",                          # malformed line
            json.dumps({"case_no": "x", "body_markdown": "y"}),  # missing id
            json.dumps(_ecd_item(3)),                   # good
        ]),
        encoding="utf-8",
    )
    client = FakeClient()
    docs, chunks, skipped = _run(cfg, client, "ecd")
    assert docs == 1 and skipped == 2


def test_stale_delete_called_with_chunk_count(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_items(cfg, "ecd", [_ecd_item(1)])
    client = FakeClient()
    _run(cfg, client, "ecd", skip_stale_delete=False)
    assert client.deletes  # incremental delete issued for the doc
    # ...and skipped on a freshly-created/empty collection:
    client3 = FakeClient()
    _run(cfg, client3, "ecd", skip_stale_delete=True)
    assert client3.deletes == []


def test_resume_skips_through_last_id(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_items(cfg, "ecd", [_ecd_item(1), _ecd_item(2), _ecd_item(3)])
    # Pretend doc 1 was already ingested.
    (cfg.state_dir).mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "ecd.json").write_text(json.dumps({"last_document_id": "1"}), encoding="utf-8")
    client = FakeClient()
    docs, chunks, skipped = _run(cfg, client, "ecd", resume=True)
    assert docs == 2  # only docs 2 and 3 processed


def test_resume_id_not_found_raises(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_items(cfg, "ecd", [_ecd_item(1), _ecd_item(2)])
    (cfg.state_dir).mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "ecd.json").write_text(json.dumps({"last_document_id": "999"}), encoding="utf-8")
    client = FakeClient()
    with pytest.raises(RuntimeError):
        _run(cfg, client, "ecd", resume=True)


def test_delete_checkpoint(tmp_path):
    cfg = _make_cfg(tmp_path)
    (cfg.state_dir).mkdir(parents=True, exist_ok=True)
    (cfg.state_dir / "ecd.json").write_text("{}", encoding="utf-8")
    pipeline.delete_checkpoint(cfg, "ecd")
    assert not (cfg.state_dir / "ecd.json").exists()
    pipeline.delete_checkpoint(cfg, "ecd")  # idempotent (missing_ok)
