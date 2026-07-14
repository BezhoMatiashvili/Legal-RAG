"""End-to-end harness over the synthetic FakeBackend + experiment-log round-trip.

Proves the whole pipeline runs offline (no torch): all five modes produce scores, the
metrics/stats/log plumbing works, and a real run over the snapshot golden set logs a
regression entry. The trustworthy full-corpus numbers come in Part 3.
"""

import json

import pytest

from ingest.chunking import default_token_counter
from eval import explog, goldset
from eval.backend import MODES, ChunkRecord, FakeBackend
from eval.evaluate import build_fake_corpus, build_query_relevance, run_mode
from eval.metrics import METRIC_NAMES, aggregate, values
from eval.stats import bootstrap_ci

count = default_token_counter
CHUNK_CFG = dict(max_tokens=512, overlap=80, min_tokens=64)


def _records():
    return [
        ChunkRecord("s", "d1", 0, "მუხლი პირველი ხელშეკრულების შესახებ დებულება"),
        ChunkRecord("s", "d2", 0, "სისხლის სამართლის პასუხისმგებლობა დანაშაული"),
        ChunkRecord("s", "d3", 0, "საგადასახადო კოდექსი გადასახადი ბიუჯეტი"),
    ]


def test_fakebackend_all_modes_rank_the_matching_chunk_first():
    be = FakeBackend(_records())
    for mode in MODES:
        hits, lat = be.search("სისხლის სამართლის დანაშაული", mode, k=3)
        assert hits, f"{mode} returned nothing"
        assert (hits[0].document_id) == "d2", f"{mode} mis-ranked"
        assert set(lat) == {"embed", "search", "rerank"}


def test_explog_roundtrip_and_stable_hash(tmp_path):
    path = tmp_path / "experiments.jsonl"
    h1 = explog.config_hash({"mode": "hybrid", "top_k": 10})
    h2 = explog.config_hash({"top_k": 10, "mode": "hybrid"})  # order-independent
    assert h1 == h2 and len(h1) == 16
    explog.append_run({"config_hash": h1, "mode": "hybrid", "metrics": {"recall10": 0.5}}, path)
    explog.append_run({"config_hash": h1, "mode": "bm25", "metrics": {"recall10": 0.4}}, path)
    rows = explog.read_log(path)
    assert len(rows) == 2
    assert rows[0]["config_hash"] == h1


@pytest.mark.snapshot
def test_full_pipeline_over_snapshot_logs_a_run(tmp_path):
    gold = goldset.load_golden_set()
    bodies = goldset.SnapshotBodies(needed=goldset.gold_docs(gold))
    # guards must pass on the real set
    assert goldset.reground(gold, bodies) == 103
    goldset.enforce_holdout(gold, goldset.load_holdout())
    assert goldset.lint_span_coverage(gold, bodies, count_tokens=count, **CHUNK_CFG) == 103

    rel = build_query_relevance(gold, bodies, CHUNK_CFG, count)
    corpus = build_fake_corpus(gold, bodies, CHUNK_CFG, count, n_distractors=40, seed=0)
    assert len(corpus) > len(gold)  # gold chunks + distractors
    be = FakeBackend(corpus)

    scores, lat = run_mode(be, gold, rel, "bm25", "chunk", k=10)
    assert len(scores) == 103
    agg = aggregate(scores)
    # lexical BM25 finds the Georgian evidence for a healthy fraction of ka queries
    assert agg["recall10"] > 0.3

    record = {
        "timestamp": explog.now_iso(),
        "eval_set_version": goldset.EVAL_SET_VERSION,
        "eval_set_hash": goldset.eval_set_hash(),
        "config_hash": explog.config_hash({"mode": "bm25", "relevance": "chunk", **CHUNK_CFG}),
        "mode": "bm25",
        "metrics": agg,
        "cis": {m: bootstrap_ci(values(scores, m), resamples=500).__dict__ for m in METRIC_NAMES},
    }
    path = tmp_path / "experiments.jsonl"
    explog.append_run(record, path)
    (loaded,) = explog.read_log(path)
    assert loaded["eval_set_version"] == "v1"
    assert set(loaded["metrics"]) == set(METRIC_NAMES)
    # round-trips as valid JSON
    assert json.loads(json.dumps(loaded))["mode"] == "bm25"
