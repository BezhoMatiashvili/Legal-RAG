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
from eval.evaluate import (
    build_fake_corpus,
    build_query_relevance,
    candidate_metrics_complete,
    quality_gate_invalid_reasons,
    run_mode,
    run_mode_repeated,
)
from eval.metrics import METRIC_NAMES, Hit, aggregate, values
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


def test_repeated_run_hash_ignores_timing_jitter_but_detects_ranking_drift():
    from types import SimpleNamespace

    gold = [SimpleNamespace(id="q", query="query", query_type="keyword", query_language="ka")]
    rel = {"q": {"chunk": {("s", "gold", 0): 2}}}

    class Stable:
        calls = 0
        last_outcome = None

        def search(self, *_args):
            self.calls += 1
            return [Hit("s", "gold", 0, 1.0)], {
                "embed": self.calls / 1000, "search": 0.0, "rerank": 0.0,
            }

    _scores, latency = run_mode_repeated(Stable(), gold, rel, "hybrid", "chunk", 10, repeats=2)
    assert latency["deterministic_repeats"] is True
    assert len(set(latency["repeat_result_hashes"])) == 1

    class Drifting(Stable):
        def search(self, *_args):
            self.calls += 1
            doc = "gold" if self.calls == 1 else "other"
            return [Hit("s", doc, 0, 1.0)], {"embed": 0.0, "search": 0.0, "rerank": 0.0}

    with pytest.raises(RuntimeError, match="non-deterministic"):
        run_mode_repeated(Drifting(), gold, rel, "hybrid", "chunk", 10, repeats=2)


def test_exact_accuracy_candidate_pool_drives_candidate_recall_at_50_and_80():
    from types import SimpleNamespace

    from ingest.qdrant_store import point_id

    gold = [SimpleNamespace(id="q", query="query", query_type="keyword", query_language="ka")]
    group = frozenset({("s", "gold", "v-current", 3)})
    rel = {"q": {
        "chunk": {("s", "gold", "v-current", 3): 2},
        "evidence_groups": {"rule": group},
        "cluster_id": "family",
    }}
    gold_id = point_id("s", "gold", 3, version_id="v-current")
    exact_pool = (
        *(f"noise-{index}" for index in range(50)),
        gold_id,
        *(f"tail-noise-{index}" for index in range(29)),
    )

    class Backend:
        supports_candidate_depth = False
        last_outcome = None

        def search(self, *_args):
            # This branch trace would put gold at rank one. Exact pool telemetry must win.
            misleading_branch = SimpleNamespace(
                name="global_original",
                query="query",
                filters={},
                hit_ids=(gold_id,),
            )
            self.last_outcome = SimpleNamespace(
                candidate_ids=exact_pool,
                branches=(misleading_branch,),
                degraded=False,
                degraded_reason=None,
                service_abstention=False,
                abstention_reason=None,
                effective_route=None,
                plan=None,
                result_hash="backend-hash",
            )
            return [Hit("s", "wrong", 0, 1.0, version_id="v-current")], {
                "embed": 0.0, "search": 0.0, "rerank": 0.0,
            }

    (score,), latency = run_mode(Backend(), gold, rel, "production", "chunk", 10)
    assert score.candidate_recall50 == 0.0
    assert score.candidate_recall80 == 1.0
    depth = latency["queries"][0]["candidate_depth"]
    assert depth["kind"] == "accuracy_pre_rerank_pool"
    assert depth["source"] == "outcome.candidate_ids"
    assert depth["observed"] == depth["pool_depth"] == 80
    assert depth["pool_sha256"] == depth["candidate_ids_sha256"]
    assert len(depth["pool_sha256"]) == 64


def test_legacy_accuracy_branch_trace_still_drives_candidate_recall():
    from types import SimpleNamespace

    from ingest.qdrant_store import point_id

    gold = [SimpleNamespace(id="q", query="query", query_type="keyword", query_language="ka")]
    group = frozenset({("s", "gold", "v-current", 3)})
    rel = {"q": {
        "chunk": {("s", "gold", "v-current", 3): 2},
        "evidence_groups": {"rule": group},
        "cluster_id": "family",
    }}

    class Backend:
        supports_candidate_depth = False
        last_outcome = None

        def search(self, *_args):
            first_branch = SimpleNamespace(
                name="global_original",
                query="query",
                filters={},
                hit_ids=tuple(f"noise-{index}" for index in range(50)),
            )
            translated_branch = SimpleNamespace(
                name="global_translated",
                query="query-ka",
                filters={},
                hit_ids=(
                    point_id("s", "gold", 3, version_id="v-current"),
                    *(f"translated-noise-{index}" for index in range(29)),
                ),
            )
            self.last_outcome = SimpleNamespace(
                branches=(first_branch, translated_branch),
                degraded=False,
                degraded_reason=None,
                service_abstention=False, abstention_reason=None,
                effective_route=None, plan=None, result_hash="backend-hash",
            )
            return [Hit("s", "wrong", 0, 1.0, version_id="v-current")], {
                "embed": 0.0, "search": 0.0, "rerank": 0.0,
            }

    (score,), latency = run_mode(Backend(), gold, rel, "production", "chunk", 10)
    assert score.candidate_recall50 == 0.0
    assert score.candidate_recall80 == 1.0
    depth = latency["queries"][0]["candidate_depth"]
    assert depth["kind"] == "accuracy_ordered_union"
    assert depth["source"] == "legacy_branch_reconstruction"
    assert depth["observed"] == depth["pool_depth"] == 80


def test_accuracy_strict_requires_candidate_metrics_but_legacy_production_can_report_unavailable():
    from types import SimpleNamespace

    missing = [
        SimpleNamespace(
            failed=False,
            candidate_recall50=None,
            candidate_recall80=None,
        )
    ]
    latency = {"repeat_count": 2, "deterministic_repeats": True}
    assert candidate_metrics_complete(missing) is False
    assert quality_gate_invalid_reasons("production", missing, latency) == []
    assert quality_gate_invalid_reasons("accuracy_strict", missing, latency) == [
        "candidate_metrics_incomplete"
    ]

    complete = [
        SimpleNamespace(
            failed=False,
            candidate_recall50=0.0,
            candidate_recall80=1.0,
        )
    ]
    assert candidate_metrics_complete(complete) is True
    assert quality_gate_invalid_reasons("accuracy_strict", complete, latency) == []


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
    assert set(METRIC_NAMES) <= set(loaded["metrics"])
    assert loaded["metrics"]["recall10"] == loaded["metrics"]["success10"]
    # round-trips as valid JSON
    assert json.loads(json.dumps(loaded))["mode"] == "bm25"
