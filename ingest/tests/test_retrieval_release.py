import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from eval.evaluate import run_mode
from eval.metrics import Hit
from eval.retrieval_release import (
    RETRIEVAL_ONLY_DISCLAIMER,
    RetrievalReleaseError,
    compare_release_to_baseline,
    create_handoff_report,
    load_repeat,
    persist_repeat,
)


def _identity():
    return {
        "run_id": "eval-run-1",
        "snapshot_id": "v3_512_attested_20260715_01",
        "generation_id": "v3_512_candidate_20260715_01",
        "physical_collection": "georgian_legal__gen_v3_512_candidate_20260715_01",
        "snapshot_sha256": "1" * 64,
        "generation_sha256": "2" * 64,
        "configuration_sha256": "3" * 64,
        "collection_sha256": "4" * 64,
        "collection_configuration_sha256": "0" * 64,
        "vector_checksum_sha256": "5" * 64,
        "vector_probe_sha256": "1" * 64,
        "qrel_adapter_sha256": "6" * 64,
        "baseline_manifest_sha256": "7" * 64,
        "verification_1_sha256": "8" * 64,
        "verification_2_sha256": "9" * 64,
        "dependency_lock_sha256": "a" * 64,
        "runtime_identity_sha256": "b" * 64,
        "code_identity_sha256": "c" * 64,
        "embedding_artifact_sha256": "d" * 64,
        "tokenizer_artifact_sha256": "e" * 64,
        "reranker_artifact_sha256": "f" * 64,
        "points_count": 2,
        "corpus_hash": "d" * 64,
        "vector_space_id": "e" * 64,
        "chunk_config_id": "f" * 64,
        "header_config_id": "0" * 64,
        "retrieval_fingerprint_revision": 2,
        "retrieval_fingerprint": "1" * 64,
        "dirty_patch_hash": "4" * 64,
        "golden_set_sha256": (
            "753e2985315be3e408c3db3303f90625d9c66984b5fc9519cf7e32a8d46252c6"
        ),
        "translation_sha256": (
            "0884870a8fa68527c959de3781c4515458d96784c4f599158427f058ce727fad"
        ),
        "holdout_sha256": (
            "eaee96072f66a0f3f63d6d1cbe61e4566d3b405daedf389211bb351d05b7ad3e"
        ),
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": "1" * 40,
        "tokenizer_model": "BAAI/bge-m3",
        "tokenizer_revision": "2" * 40,
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "reranker_revision": "3" * 40,
        "runtime_image_digest": "sha256:" + "2" * 64,
        "repository_revision": "3" * 40,
    }


def _provenance(track):
    identity = _identity()
    return {
        "collection_alias": "georgian_legal",
        "serving_alias": "georgian_legal",
        "physical_collection": identity["physical_collection"],
        "queried_collection": identity["physical_collection"],
        "access_kind": "direct_physical",
        "generation_id": identity["generation_id"],
        "points_count": identity["points_count"],
        "corpus_hash": identity["corpus_hash"],
        "snapshot_hash": identity["snapshot_sha256"],
        "embedding_model": identity["embedding_model"],
        "embedding_revision": identity["embedding_revision"],
        "tokenizer_model": identity["tokenizer_model"],
        "tokenizer_revision": identity["tokenizer_revision"],
        "reranker_model": identity["reranker_model"],
        "reranker_revision": identity["reranker_revision"],
        "vector_space_id": identity["vector_space_id"],
        "chunk_config_id": identity["chunk_config_id"],
        "header_config_id": identity["header_config_id"],
        "retrieval_fingerprint_revision": identity[
            "retrieval_fingerprint_revision"
        ],
        "retrieval_fingerprint": identity["retrieval_fingerprint"],
        "dependency_identity": identity["dependency_lock_sha256"],
        "image_identity": identity["runtime_image_digest"],
        "git_sha": identity["repository_revision"],
        "dirty_patch_hash": identity["dirty_patch_hash"],
        "execution_mode": track,
        "frozen_set_hashes": {
            "golden_v2": identity["golden_set_sha256"],
            "holdout_v2": identity["holdout_sha256"],
            "authored_query_translations": identity["translation_sha256"],
            "v2_candidate_qrels": identity["qrel_adapter_sha256"],
        },
    }


def _gold_and_relevance():
    gold = [
        SimpleNamespace(
            id=f"q{index}",
            query=f"rule {index}",
            query_type="paraphrase",
            query_language="ka",
            gold_source="matsne",
            source="matsne",
            tags=("high_risk",),
            risk_level="high",
            expected_outcome="answer",
        )
        for index in (1, 2)
    ]
    relevance = {
        query.id: {
            "chunk": {("matsne", query.id, "v1", 0): 2},
            "doc": {("matsne", query.id, "v1"): 2},
            "evidence_groups": {"rule": frozenset({("matsne", query.id, "v1", 0)})},
            "cluster_id": query.id,
        }
        for query in gold
    }
    return gold, relevance


class _StableBackend:
    supports_candidate_depth = True
    last_outcome = None

    def search(self, query, _mode, _k):
        query_id = query.split()[-1]
        hit = Hit("matsne", f"q{query_id}", 0, 1.0, version_id="v1")
        self.last_candidate_hits = [hit]
        self.last_outcome = None
        return [hit], {"embed": 0.001, "search": 0.002, "rerank": 0.003}


def _repeat_artifacts(tmp_path):
    gold, relevance = _gold_and_relevance()
    paths = {"production": [], "accuracy_strict": []}
    bootstrap_scores, bootstrap_latency = run_mode(
        _StableBackend(), gold, relevance, "production", "chunk", 10
    )
    del bootstrap_scores
    first_queries = bootstrap_latency["queries"]
    baseline = _pairable_baseline(first_queries)
    for track in paths:
        for repeat_index in (1, 2):
            scores, latency = run_mode(
                _StableBackend(), gold, relevance, track, "chunk", 10
            )
            path = tmp_path / f"{track}-{repeat_index}.json"
            persist_repeat(
                output=path,
                track=track,
                repeat_index=repeat_index,
                scores=scores,
                latency=latency,
                identity=_identity(),
                evaluation_provenance=_provenance(track),
                baseline={"root": tmp_path, **baseline},
            )
            paths[track].append(path)
    return paths, first_queries


def _pairable_baseline(candidate_queries):
    queries = []
    for candidate_row in copy.deepcopy(candidate_queries):
        row = {
            "query_id": candidate_row["query_id"],
            "raw_candidates": [
                {
                    "point_id": f"baseline-{candidate_row['query_id']}-{index}",
                    "score": 0.0,
                }
                for index in range(80)
            ],
            "final_ranking": [
                {
                    "point_id": f"baseline-final-{candidate_row['query_id']}",
                    "score": 0.0,
                }
            ],
            "branch_provenance": {},
            "entity_matches": {},
            "document_matches": {},
            "route_decision": {},
            "score": candidate_row["score"],
        }
        for metric in (
            "success1",
            "success5",
            "success10",
            "candidate_recall50",
            "candidate_recall80",
            "required_evidence_recall10",
            "document_identity1",
            "passage_accuracy1",
            "context_duplication10",
            "context_noise10",
        ):
            row["score"][metric] = 0.0
        queries.append(row)

    def canonical(value):
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    ranking_hash = canonical(
        [
            {
                "query_id": row["query_id"],
                "raw_candidates": row["raw_candidates"],
                "final_ranking": row["final_ranking"],
            }
            for row in queries
        ]
    )
    decision_hash = canonical(queries)
    return {
        "manifest_sha256": "7" * 64,
        "ranking_hashes": [ranking_hash, ranking_hash],
        "decision_result_hashes": [decision_hash, decision_hash],
        "queries": queries,
    }


def test_repeat_is_create_only_and_reloads_with_reconciled_hashes(tmp_path):
    paths, _queries = _repeat_artifacts(tmp_path)
    repeat = load_repeat(paths["production"][0])
    assert repeat["ranking_hash"] != repeat["decision_result_hash"]
    assert repeat["query_count"] == 2
    assert repeat["queries"][0]["ranked_hits"][0]["point_id"]

    gold, relevance = _gold_and_relevance()
    scores, latency = run_mode(
        _StableBackend(), gold, relevance, "production", "chunk", 10
    )
    with pytest.raises(FileExistsError, match="already exists"):
        persist_repeat(
            output=paths["production"][0],
            track="production",
            repeat_index=1,
            scores=scores,
            latency=latency,
            identity=_identity(),
            evaluation_provenance=_provenance("production"),
            baseline={"manifest_sha256": "7" * 64},
        )


def test_repeat_refuses_incomplete_or_runtime_drifted_provenance(tmp_path):
    gold, relevance = _gold_and_relevance()
    scores, latency = run_mode(
        _StableBackend(), gold, relevance, "production", "chunk", 10
    )
    output = tmp_path / "incomplete-provenance.json"
    with pytest.raises(
        RetrievalReleaseError,
        match="evaluation provenance is incomplete or invalid",
    ):
        persist_repeat(
            output=output,
            track="production",
            repeat_index=1,
            scores=scores,
            latency=latency,
            identity=_identity(),
            evaluation_provenance={
                "physical_collection": _identity()["physical_collection"],
                "generation_id": _identity()["generation_id"],
                "snapshot_hash": _identity()["snapshot_sha256"],
            },
            baseline={"manifest_sha256": "7" * 64},
        )
    assert not output.exists()

    drifted = _provenance("production")
    drifted["dependency_identity"] = "9" * 64
    with pytest.raises(RetrievalReleaseError, match="dependency identity differs"):
        persist_repeat(
            output=output,
            track="production",
            repeat_index=1,
            scores=scores,
            latency=latency,
            identity=_identity(),
            evaluation_provenance=drifted,
            baseline={"manifest_sha256": "7" * 64},
        )
    assert not output.exists()

    drifted = _provenance("production")
    drifted["reranker_revision"] = "5" * 40
    with pytest.raises(
        RetrievalReleaseError,
        match="model/runtime identity differs for reranker_revision",
    ):
        persist_repeat(
            output=output,
            track="production",
            repeat_index=1,
            scores=scores,
            latency=latency,
            identity=_identity(),
            evaluation_provenance=drifted,
            baseline={"manifest_sha256": "7" * 64},
        )
    assert not output.exists()


def test_repeat_reload_recomputes_query_trace_and_failure_counts(tmp_path):
    paths, _queries = _repeat_artifacts(tmp_path)
    path = paths["production"][0]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["queries"][0]["ranked_hits"][0]["score"] = -999.0
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="query ranking hash does not reconcile"):
        load_repeat(path)

    paths, _queries = _repeat_artifacts(tmp_path / "failure-count")
    path = paths["production"][0]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["failure_counts"]["degraded"] = 1
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="failure counts do not reconcile"):
        load_repeat(path)

    paths, _queries = _repeat_artifacts(tmp_path / "result-alias")
    path = paths["production"][0]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["queries"][0]["result_hash"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="result_hash alias differs"):
        load_repeat(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value.__setitem__("run_id", "forged-run"),
            "run_id differs",
        ),
        (
            lambda value: value["evaluation_provenance"].__setitem__(
                "corpus_hash", "0" * 64
            ),
            "collection/configuration identity differs for corpus_hash",
        ),
        (
            lambda value: value["evaluation_provenance"][
                "frozen_set_hashes"
            ].pop("authored_query_translations"),
            "does not exactly bind the frozen v2 inputs",
        ),
    ),
)
def test_repeat_reload_revalidates_top_level_and_provenance(
    tmp_path, mutation, message
):
    paths, _queries = _repeat_artifacts(tmp_path)
    path = paths["production"][0]
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(RetrievalReleaseError, match=message):
        load_repeat(path)


def test_repeat_refuses_stale_query_hash_before_create(tmp_path):
    gold, relevance = _gold_and_relevance()
    scores, latency = run_mode(
        _StableBackend(), gold, relevance, "production", "chunk", 10
    )
    latency["queries"][0]["ranked_hits"][0]["score"] = -999.0
    output = tmp_path / "invalid-repeat.json"

    with pytest.raises(ValueError, match="query ranking hash does not reconcile"):
        persist_repeat(
            output=output,
            track="production",
            repeat_index=1,
            scores=scores,
            latency=latency,
            identity=_identity(),
            evaluation_provenance=_provenance("production"),
            baseline={"manifest_sha256": "7" * 64},
        )
    assert not output.exists()


def test_pairable_repeats_produce_metrics_verdict_and_handoff(tmp_path):
    paths, candidate_queries = _repeat_artifacts(tmp_path)
    comparison_path = tmp_path / "comparison.json"
    compare_release_to_baseline(
        production_paths=paths["production"],
        accuracy_strict_paths=paths["accuracy_strict"],
        baseline=_pairable_baseline(candidate_queries),
        output=comparison_path,
        require_frozen_v2=False,
        resamples=100,
    )
    comparison = json.loads(comparison_path.read_text())
    assert comparison["verdict"] == "improved"
    assert comparison["metrics"]["success_at_1"]["alias_of"] == "success1"
    assert comparison["metrics"]["success_at_1"]["metric"] == "success_at_1"
    assert comparison["metrics"]["success1"]["metric_label"] == "Success@1"
    assert comparison["metrics"]["success1"]["baseline"] == 0.0
    assert comparison["metrics"]["success1"]["candidate"] == 1.0
    assert comparison["metrics"]["success1"]["paired_delta_direction"] == (
        "candidate_minus_baseline"
    )
    assert comparison["metrics"]["success1"]["sample_count"] == 2
    assert len(comparison["metrics"]["success1"]["confidence_interval_95"]) == 2
    assert (
        comparison["metrics"]["evidence_span_coverage10"]["independent_corroboration"]
        is False
    )
    source_slice = comparison["slices"]["source:matsne"]["metrics"]
    assert source_slice["success1"]["n"] == 2
    assert source_slice["success1"]["n_clusters"] == 2
    assert "diff_lo" in source_slice["success1"]
    assert "p_value_adjusted" in source_slice["success1"]
    assert comparison["slices"]["source:matsne"]["holm_family"] == (
        "metrics_within_this_slice"
    )
    assert source_slice["evidence_span_coverage10"]["alias_of"] == (
        "required_evidence_recall10"
    )
    assert comparison["reproducibility"]["candidate"] is True
    assert comparison["not_labeled"]["supremecourt"] == "not_labeled"
    assert comparison["next_experiment"]["experiment"] == "document_first_retrieval"

    handoff_path = tmp_path / "handoff.json"
    create_handoff_report(
        output=handoff_path,
        comparison_path=comparison_path,
        source_counts={"matsne": 2},
        quarantine_counts={"tas": 25},
        commands_and_checks=[{"command": "pytest", "status": "passed"}],
        external_resource_usage={"paid_compute_usd": 0},
        remaining_limitations=[],
    )
    handoff = json.loads(handoff_path.read_text())
    assert handoff["scope_disclaimer"] == RETRIEVAL_ONLY_DISCLAIMER
    assert handoff["promotion_plan_created"] is False
    assert handoff["alias_operation_invoked"] is False


def test_summary_only_baseline_has_inconclusive_ceiling(tmp_path):
    paths, _candidate_queries = _repeat_artifacts(tmp_path)
    bound = load_repeat(paths["production"][0])["baseline_identity"]
    output = tmp_path / "comparison-summary-only.json"
    compare_release_to_baseline(
        production_paths=paths["production"],
        accuracy_strict_paths=paths["accuracy_strict"],
        baseline={
            "manifest_sha256": "7" * 64,
            "ranking_hashes": bound["ranking_hashes"],
            "decision_result_hashes": bound["decision_result_hashes"],
            "query_ids": ["q1", "q2"],
        },
        output=output,
        require_frozen_v2=False,
        resamples=20,
    )
    comparison = json.loads(output.read_text())
    assert comparison["verdict"] == "inconclusive"
    assert "baseline_per_query_traces_unavailable" in comparison["verdict_reasons"]


def test_exception_and_answerable_abstention_are_persisted_zero_failures():
    gold, relevance = _gold_and_relevance()

    class Exploding:
        supports_candidate_depth = False

        def search(self, *_args):
            raise RuntimeError("boom")

    scores, latency = run_mode(Exploding(), gold, relevance, "production", "chunk", 10)
    assert all(score.failed and score.success1 == 0.0 for score in scores)
    assert all(score.context_noise10 == 0.0 for score in scores)
    assert all(row["status"] == "exception" for row in latency["queries"])
    assert len(latency["exceptions"]) == 2


def test_run_mode_status_shapes_reconcile_in_release_artifact(tmp_path):
    gold = [
        SimpleNamespace(
            id=query_id,
            query=query_id,
            query_type="keyword",
            query_language="ka",
            gold_source="matsne",
            expected_outcome=expected_outcome,
        )
        for query_id, expected_outcome in (
            ("ok", "answer"),
            ("degraded", "answer"),
            ("failed-abstention", "answer"),
            ("expected-abstention", "abstain"),
        )
    ]
    relevance = {
        query.id: {
            "chunk": {("matsne", query.id, 0): 2},
            "doc": {("matsne", query.id): 2},
            "cluster_id": query.id,
        }
        for query in gold
    }

    class StatusBackend:
        supports_candidate_depth = False
        last_outcome = None

        def search(self, query, _mode, _k):
            degraded = query == "degraded"
            abstention = query in {"failed-abstention", "expected-abstention"}
            self.last_outcome = SimpleNamespace(
                branches=(),
                candidate_ranking=(),
                timings_ms={"search": 1.0},
                degraded=degraded,
                degraded_reason="remote_reranker_failed" if degraded else None,
                service_abstention=abstention,
                abstention_reason="no_admissible_evidence" if abstention else None,
                effective_route="hybrid",
                plan=None,
                result_hash=f"backend-{query}",
            )
            hits = [] if abstention else [Hit("matsne", query, 0, 1.0)]
            return hits, {"embed": 0.0, "search": 0.001, "rerank": 0.0}

    scores, latency = run_mode(
        StatusBackend(), gold, relevance, "production", "chunk", 10
    )
    assert [row["status"] for row in latency["queries"]] == [
        "ok",
        "degraded",
        "failed_abstention",
        "abstain",
    ]
    assert [score.failed for score in scores] == [False, True, True, False]

    output = tmp_path / "status-repeat.json"
    persist_repeat(
        output=output,
        track="production",
        repeat_index=1,
        scores=scores,
        latency=latency,
        identity=_identity(),
        evaluation_provenance=_provenance("production"),
        baseline={"manifest_sha256": "7" * 64},
    )
    artifact = load_repeat(output)
    assert artifact["failure_counts"] == {
        "failed": 2,
        "degraded": 1,
        "skipped_or_unchunkable": 0,
        "exceptions": 0,
        "answerable_abstentions": 1,
    }
