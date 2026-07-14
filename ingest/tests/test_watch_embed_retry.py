"""Regression: a *transient* embed failure must NOT advance the watch run-offset past the
failed doc (which silently drops it from the index forever), but a deterministically-poison
doc must be dead-lettered after a bounded number of retries so it can't block the drain.

Offline — no Qdrant, no model (the all-fail path never upserts, and skip_unchanged=False
avoids the client.retrieve change-detection lookup)."""

import dataclasses
import json

import pytest

from ingest import pipeline
from ingest.config import load_config


class _AlwaysFailEmbedder:
    def encode_passages(self, texts):
        raise RuntimeError("simulated transient CUDA/OOM embed failure")

    def encode_query(self, text):  # pragma: no cover - not used on this path
        raise RuntimeError("n/a")


class _NoClient:
    """watch never touches Qdrant when every doc fails to embed (nothing to upsert)."""


def _count_tokens(text):
    return len(text.split())


def _make_cfg(tmp_path):
    cfg = load_config()
    return dataclasses.replace(
        cfg, collection_name="test",
        artifacts_root=tmp_path / "artifacts", state_dir=tmp_path / "state",
        chunk_tokens=40, chunk_overlap=5, chunk_min_tokens=1,
        dense_dim=4,
    )


def _write_run(cfg, source, run_id, items):
    d = cfg.artifacts_root / source / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n"
    (d / "items.jsonl").write_bytes(body.encode("utf-8"))


def _item(doc_id):
    return {
        "decision_document_id": doc_id, "case_no": f"c{doc_id}", "decision_type_name": "x",
        "court_name": "court", "decision_date": "2020-04-30",
        "body_markdown": "body text here with enough meaningful legal content for indexing",
    }


def _drain(cfg, state):
    return pipeline.watch_drain_source(
        cfg, _NoClient(), _AlwaysFailEmbedder(), _count_tokens, "ecd", state,
        batch_size=1, skip_unchanged=False)


def test_transient_embed_failure_holds_offset_then_dead_letters(tmp_path):
    cfg = _make_cfg(tmp_path)
    run_id = "20200101T000000Z_s_e"
    _write_run(cfg, "ecd", run_id, [_item(1)])
    state = {"run_offsets": {}}

    # Passes 1..(N-1): the offset must stay at the doc's start so the next pass re-reads it.
    for i in range(1, pipeline._MAX_EMBED_RETRIES):
        docs, _chunks, skipped = _drain(cfg, state)
        assert docs == 0 and skipped == 1
        assert state["run_offsets"].get(run_id) == 0, \
            "run offset advanced past a transiently-failed doc — silent data loss"
        assert state["embed_retry"]["1"] == i
        assert "dead_letter" not in state

    # Pass N: give up on the poison doc (dead-letter) and let the drain progress past it.
    _drain(cfg, state)
    assert any(d["document_id"] == "1" for d in state["dead_letter"])
    assert state["run_offsets"][run_id] > 0, "offset should advance past a dead-lettered doc"
    assert "1" not in state.get("embed_retry", {})


def test_clean_doc_after_a_failure_clears_its_retry_count(tmp_path):
    """A doc that later embeds cleanly must have its transient-failure count cleared."""
    cfg = _make_cfg(tmp_path)
    run_id = "20200101T000000Z_s_e"
    _write_run(cfg, "ecd", run_id, [_item(1)])
    state = {"run_offsets": {}}
    _drain(cfg, state)
    assert state["embed_retry"]["1"] == 1

    # Now the embedder works: the same doc processes and its retry count is cleared.
    from ingest.embedding import Embedded, Sparse

    class _OkEmbedder:
        def encode_passages(self, texts):
            return [Embedded(dense=[0.1, 0.2, 0.3, 0.4], sparse=Sparse([1], [0.5])) for _ in texts]

    class _CapClient:
        def upsert(self, collection_name, points, wait=False):
            pass

        def delete(self, collection_name, points_selector, wait=False):
            assert wait is True
            pass

    pipeline.watch_drain_source(cfg, _CapClient(), _OkEmbedder(), _count_tokens, "ecd", state,
                                batch_size=1, skip_unchanged=False)
    assert "1" not in state.get("embed_retry", {})
    assert state["run_offsets"][run_id] > 0


def test_once_mode_retries_then_fails_loudly_on_dead_letter(tmp_path):
    cfg = _make_cfg(tmp_path)
    run_id = "20200101T000000Z_s_e"
    _write_run(cfg, "ecd", run_id, [_item(1)])

    with pytest.raises(RuntimeError, match="dead-lettered document"):
        pipeline.watch_loop(
            cfg,
            _NoClient(),
            _AlwaysFailEmbedder(),
            _count_tokens,
            ["ecd"],
            batch_size=1,
            poll_interval=0,
            once=True,
        )

    report_path = next((cfg.state_dir / "reports").glob("ingest-*.json"))
    report = json.loads(report_path.read_text())
    assert report["totals"]["dead_letters"] == 1
    assert report["per_source"]["ecd"]["dead_letter_sample"][0]["document_id"] == "1"
