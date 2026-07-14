import dataclasses
import json

import pytest

from ingest import pipeline
from ingest.config import load_config
from ingest.dedup import content_hash
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
        self.events = []

    def upsert(self, collection_name, points, wait=False):
        self.events.append("upsert")
        self.upserts.append((list(points), wait))

    def delete(self, collection_name, points_selector, wait=False):
        self.events.append("delete")
        assert wait is True
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
        dense_dim=4,
    )


def _write_items(cfg, source, items):
    path = pipeline.items_path(cfg, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")


def _ecd_item(doc_id, body="body text here for chunking with enough meaningful legal content"):
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
    assert docs == 1 and skipped == 1


def test_writer_cleans_body_before_chunking_and_hashing(tmp_path):
    cfg = _make_cfg(tmp_path)
    raw = "clean\x00 body with enough meaningful legal content for production indexing"
    _write_items(cfg, "ecd", [_ecd_item(1, body=raw)])
    client = FakeClient()

    docs, _, skipped = _run(cfg, client, "ecd")

    assert docs == 1 and skipped == 0
    payload = client.upserts[0][0][0].payload
    assert "\x00" not in payload["text"]
    assert payload["content_hash"] == content_hash(raw.replace("\x00", ""))
    assert payload["document_chunk_count"] == sum(len(batch) for batch, _ in client.upserts)


def test_generation_writer_stamps_every_point_with_same_identity(tmp_path):
    cfg = dataclasses.replace(
        _make_cfg(tmp_path),
        generation_id="gen_20260713_verified",
        collection_name="test__gen_gen_20260713_verified",
        embedding_revision="a" * 40,
        tokenizer_revision="b" * 40,
        reranker_revision="c" * 40,
        rerank_enabled=False,
    )
    _write_items(cfg, "ecd", [_ecd_item(1)])
    client = FakeClient()

    docs, _, skipped = _run(cfg, client, "ecd")

    assert docs == 1 and skipped == 0
    payloads = [point.payload for batch, _wait in client.upserts for point in batch]
    assert payloads
    assert {payload["generation_id"] for payload in payloads} == {cfg.generation_id}
    assert {payload["schema_version"] for payload in payloads} == {1}
    assert {payload["tokenizer_model"] for payload in payloads} == {
        cfg.tokenizer_model
    }
    assert {payload["reranker_model"] for payload in payloads} == {
        cfg.rerank_model
    }
    assert {payload["reranker_revision"] for payload in payloads} == {
        cfg.reranker_revision
    }
    assert len({payload["retrieval_fingerprint"] for payload in payloads}) == 1
    assert all(len(payload["retrieval_fingerprint"]) == 64 for payload in payloads)


def test_near_empty_body_is_quarantined(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_items(cfg, "ecd", [_ecd_item(1, body="too short")])
    client = FakeClient()

    docs, chunks, skipped = _run(cfg, client, "ecd")

    assert (docs, chunks, skipped) == (0, 0, 1)
    assert client.upserts == []


def test_incomplete_summary_is_quarantined_even_when_text_is_long(tmp_path):
    cfg = _make_cfg(tmp_path)
    item = _ecd_item(1)
    item.update(
        {
            "content_kind": "article_summary",
            "content_complete": False,
            "extraction_status": "scanned_no_text",
            "source_binary_url": "https://court.example/ruling.pdf",
        }
    )
    _write_items(cfg, "ecd", [item])
    client = FakeClient()

    assert _run(cfg, client, "ecd") == (0, 0, 1)
    assert client.upserts == []


def test_legacy_unlabeled_tas_text_is_quarantined(tmp_path):
    cfg = _make_cfg(tmp_path)
    _write_items(
        cfg,
        "tas",
        [
            {
                "document_id": "legacy-list-only",
                "document_no": "AR-LEGACY",
                "body_markdown": (
                    "list metadata with enough words to pass the ordinary text "
                    "hygiene threshold but without decision completeness lineage"
                ),
            }
        ],
    )
    client = FakeClient()

    assert _run(cfg, client, "tas") == (0, 0, 1)
    assert client.upserts == []


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
    assert client.events == ["upsert", "delete"]
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


def test_resolve_all_excludes_non_corpus_supremecourt():
    assert pipeline.resolve_sources("all") == list(pipeline.CORPUS_SOURCES)
    assert "supremecourt" not in pipeline.resolve_sources("all")


def test_resolve_sources_accepts_deduplicated_comma_list():
    assert pipeline.resolve_sources("ecd,napr,ecd") == ["ecd", "napr"]
