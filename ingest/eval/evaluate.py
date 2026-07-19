"""Retrieval evaluation harness for the Georgian legal RAG index.

Upgraded, span-anchored harness that gates every retrieval change (mission Phase 1). It:

  * loads the span-anchored golden set, re-grounds every evidence quote against the clean
    snapshot, enforces the document holdout, and lints span→chunk coverage (all fail loud);
  * maps each gold span to the chunks that cover it **under the current chunking config**
    (so the set survives re-chunking/re-embedding);
  * scores five modes — **BM25** (the mandatory baseline) vs dense vs sparse vs hybrid vs
    +rerank — with Success@5/@10, required-evidence recall, candidate recall, document and
    passage accuracy, context quality, nDCG/MRR, and stage latency;
  * attaches document/version-family cluster-bootstrap CIs, runs Holm-corrected paired
    significance tests for A/B comparisons, and
    appends every run to a persistent experiment log (config hash + eval-set version).

Usage (from the ingest/ project root):

    uv run python -m eval.evaluate --backend fake --mode all --log     # offline demo/self-test
    uv run python -m eval.evaluate --mode all --relevance chunk --log  # real index (Part 3)
    uv run python -m eval.evaluate --backend fake --compare hybrid rerank
"""

import argparse
import dataclasses
import hashlib
import json
import random
from collections.abc import Mapping
from pathlib import Path

from ingest.chunking import chunk_document, default_token_counter
from ingest.config import load_config
from ingest.promotion import SERVING_ALIAS, physical_collection_name
from ingest.retrieval import EvaluationProvenance

from . import explog, goldset
from .backend import MODES, PRODUCTION_MODES, ChunkRecord, FakeBackend, ProductionBackend
from .metrics import (
    LEGACY_METRIC_ALIASES,
    METRIC_DIRECTIONS,
    METRIC_NAMES,
    aggregate,
    breakdown,
    chunk_key_parts,
    evidence_group_recall,
    paired_values,
    percentiles,
    query_score,
    reduce_ranking,
    values,
)
from .spanmap import map_spans_to_chunks
from .stats import DEFAULT_RESAMPLES, DEFAULT_SEED, bootstrap_ci, compare, holm_correct

IMMUTABLE_512_CANDIDATE_GENERATION = "v3_512_candidate_20260715_01"


def _token_counter(kind: str, tokenizer_model: str, tokenizer_revision: str | None = None):
    if kind == "word":
        return default_token_counter
    from ingest.embedding import make_token_counter

    return make_token_counter(tokenizer_model, tokenizer_revision)


def build_query_relevance(gold, bodies, chunk_cfg, count_tokens):
    """Build graded qrels plus required span-equivalence groups for every query.

    One annotation is one required evidence unit by default.  Every chunk overlapping that
    span is an interchangeable member of the unit.  Explicitly equal ``evidence_group`` ids
    combine multiple annotated alternatives into one unit.  Unchunkable/uncovered required
    groups remain present and mark the query unscorable; ``run_mode`` records a zero-score
    failure row instead of dropping it from the denominator.
    """
    rel: dict[str, dict[str, dict]] = {}
    for q in gold:
        by_doc: dict[tuple[str, str, str | None], list[tuple[int, object]]] = {}
        for index, relevance in enumerate(q.relevance):
            source = relevance.source or q.gold_source
            by_doc.setdefault(
                (source, relevance.document_id, relevance.version_id), []
            ).append((index, relevance))
        chunk_keys: dict[tuple, int] = {}
        doc_keys: dict[tuple, int] = {}
        evidence_groups: dict[str, set[tuple]] = {}
        issues: list[str] = []
        for (source, document_id, version_id), indexed_rels in by_doc.items():
            rels = [item for _index, item in indexed_rels]
            for _index, item in indexed_rels:
                document_key = (
                    (source, document_id, version_id)
                    if version_id is not None
                    else (source, document_id)
                )
                doc_keys[document_key] = max(doc_keys.get(document_key, 0), item.grade)
                if item.required:
                    group_id = item.evidence_group or f"span:{_index}"
                    evidence_groups.setdefault(group_id, set())
            try:
                body = goldset.relevance_body(bodies, source, rels[0])
                mapped = map_spans_to_chunks(
                    body,
                    [(item.char_start, item.char_end) for item in rels],
                    count_tokens=count_tokens,
                    **chunk_cfg,
                )
            except (ValueError, AssertionError, KeyError, FileNotFoundError) as exc:
                # Known corpus/chunker defects are retained as scored failures.  Unexpected
                # implementation errors still fail loudly instead of being mislabeled data.
                identity = f"{source}:{document_id}"
                if version_id is not None:
                    identity += f"@{version_id}"
                issues.append(f"{identity}: {type(exc).__name__}: {exc}")
                continue
            for local_index, (global_index, item) in enumerate(indexed_rels):
                keys = {
                    (
                        (source, document_id, version_id, chunk_index)
                        if version_id is not None
                        else (source, document_id, chunk_index)
                    )
                    for chunk_index in mapped.get(local_index, set())
                }
                for key in keys:
                    chunk_keys[key] = max(chunk_keys.get(key, 0), item.grade)
                if item.required:
                    group_id = item.evidence_group or f"span:{global_index}"
                    evidence_groups[group_id].update(keys)

        if not evidence_groups:
            issues.append("query has no required evidence groups")
        empty_groups = sorted(group_id for group_id, keys in evidence_groups.items() if not keys)
        if empty_groups:
            issues.append(f"required evidence groups map to no chunks: {empty_groups}")
        failure_reason = "; ".join(issues) if issues else None
        rel[q.id] = {
            "chunk": chunk_keys,
            "doc": doc_keys,
            "evidence_groups": {
                group_id: frozenset(keys) for group_id, keys in evidence_groups.items()
            },
            "cluster_id": q.cluster_id or f"{q.gold_source}:{q.gold_document_id}",
            "failure_reason": failure_reason,
            "issues": issues,
        }
    return rel


def build_fake_corpus(gold, bodies, chunk_cfg, count_tokens, *, n_distractors=200, seed=0, scan_cap=2000):
    """Chunk the gold docs + a bounded sample of distractor docs into a synthetic corpus.

    Distractors are drawn from the first ``scan_cap`` docs of each gold source (the docs
    jsonl are multi-GB, so a full scan is avoided); this only feeds the synthetic ranking
    pool, so a bounded, non-exhaustive sample is fine.
    """
    records: list[ChunkRecord] = []
    gold_set = goldset.gold_docs(gold)

    def add(source, document_id, body, *, version_id=None):
        try:
            chunks = chunk_document(body, count_tokens=count_tokens, **chunk_cfg)
        except (ValueError, AssertionError, KeyError, FileNotFoundError):
            return False
        for c in chunks:
            records.append(
                ChunkRecord(
                    source,
                    document_id,
                    c.chunk_index,
                    c.text,
                    version_id=version_id,
                )
            )
        return True

    gold_identities = {
        (rel.source or query.gold_source, rel.document_id, rel.version_id): rel
        for query in gold
        for rel in query.relevance
    }
    for (source, document_id, version_id), relevance in sorted(
        gold_identities.items(), key=lambda item: tuple(str(part) for part in item[0])
    ):
        add(
            source,
            document_id,
            goldset.relevance_body(bodies, source, relevance),
            version_id=version_id,
        )

    # Canonical v3 body adapters intentionally expose only the explicitly supplied frozen
    # documents, not a filesystem snapshot to sample.  Their gold corpus is still fully
    # usable for hermetic evaluator tests; distractor sampling remains a v1/v2 facility.
    source_files = getattr(bodies, "source_files", None)
    if not callable(source_files):
        return records

    rng = random.Random(seed)
    sources = sorted({s for s, _ in gold_set})
    per_source = max(1, n_distractors // len(sources))
    for source in sources:
        candidates: list[tuple[str, str]] = []
        seen_ids: set[str] = set()
        # Primary root first keeps the v1 candidate order (hence rng.sample) unchanged.
        for path in source_files(source):
            if len(candidates) >= scan_cap:
                break
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    if (source, d["document_id"]) not in gold_set and d["document_id"] not in seen_ids:
                        seen_ids.add(d["document_id"])
                        candidates.append((d["document_id"], d["body_markdown"]))
                    if len(candidates) >= scan_cap:
                        break
        for document_id, body in rng.sample(candidates, min(per_source, len(candidates))):
            add(source, document_id, body)
    return records


def eval_set_knob(version: str) -> dict[str, str]:
    """config_hash material for the eval-set choice.

    v1 is the hash-neutral anchor: it contributes nothing so every pre-switch row (all
    computed when v1 was the default) keeps its hash byte-stable — we can never fold a
    knob into v1 retroactively. Every other version (v2 is the default gate since
    2026-07-11) folds in as a knob, so its runs are distinct, forever-comparable rows.
    """
    return {} if version == "v1" else {"eval_set": version}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json_hash(value) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _ranked_hit_records(hits) -> list[dict]:
    from ingest.qdrant_store import point_id

    return [
        {
            "point_id": (
                hit.point_id
                or point_id(
                    hit.source,
                    hit.document_id,
                    hit.chunk_index,
                    version_id=hit.version_id,
                )
            ),
            "source": hit.source,
            "document_id": hit.document_id,
            "chunk_index": hit.chunk_index,
            **({"version_id": hit.version_id} if hit.version_id is not None else {}),
            # Decimal string preserves the exact Python float while remaining stable JSON.
            "score": float(hit.score).hex(),
        }
        for hit in hits
    ]


def _trace_ranked_point(value) -> dict:
    """Serialize current and compatibility retrieval ranking rows."""

    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if isinstance(value, dict):
        return dict(value)
    return {
        "point_id": str(getattr(value, "point_id", "")),
        "score": str(getattr(value, "score", "missing")),
        "source": getattr(value, "source", None),
        "document_id": getattr(value, "document_id", None),
        "version_id": getattr(value, "version_id", None),
        "chunk_index": getattr(value, "chunk_index", None),
    }


def _plan_trace(plan) -> dict | None:
    if plan is None:
        return None
    entities = getattr(plan, "entities", None)
    entity_trace = None
    if entities is not None:
        entity_trace = {
            name: getattr(entities, name, None)
            for name in (
                "document_number",
                "registration_code",
                "case_number",
                "article_id",
                "clause_id",
            )
            if hasattr(entities, name)
        }
        citation = getattr(entities, "citation", None)
        if citation is not None:
            entity_trace["citation"] = {
                "filters": dict(getattr(citation, "filters", {}) or {}),
                "raw": getattr(citation, "raw", None),
            }
    return {
        "question": getattr(plan, "question", None),
        "language": getattr(getattr(plan, "language", None), "value", None),
        "intent": getattr(getattr(plan, "intent", None), "value", None),
        "as_of": getattr(plan, "as_of", None),
        "answerable_language": getattr(plan, "answerable_language", None),
        "entities": entity_trace,
    }


def _ordered_accuracy_candidate_ids(outcome) -> tuple[str, ...] | None:
    """Return the executor's exact pool, reconstructing only legacy fixture outcomes.

    Current strict outcomes always expose ``candidate_ids``.  Attribute absence is the
    explicit compatibility signal for older SimpleNamespace/test fixtures.  That fallback
    approximates the historical union from branch traces and deliberately excludes
    unrestricted diagnostics and post-rerank cross-reference expansion.
    """
    missing = object()
    captured = getattr(outcome, "candidate_ids", missing)
    if captured is not missing:
        return tuple(captured)

    ranked = getattr(outcome, "candidate_ranking", missing)
    if ranked is not missing and (
        ranked or bool(getattr(outcome, "result_hash", ""))
    ):
        return tuple(
            str(
                point.get("point_id")
                if isinstance(point, Mapping)
                else getattr(point, "point_id", "")
            )
            for point in ranked
        )

    branches = getattr(outcome, "branches", None)
    if not branches:
        return None

    exact = [branch for branch in branches if branch.name == "exact_entity"]
    contributing = [
        branch
        for branch in branches
        if branch.name != "exact_entity"
        and not branch.name.startswith("global_unrestricted_")
        and not branch.name.startswith("cross_reference:")
    ]
    ordered: list[str] = []
    seen: set[str] = set()
    for branch in (*exact, *contributing):
        for point in branch.hit_ids:
            point_id = str(point)
            if point_id in seen:
                continue
            seen.add(point_id)
            ordered.append(point_id)
    return tuple(ordered)


def _accuracy_candidate_recall(outcome, evidence_groups, k: int) -> float | None:
    """Required-evidence recall in the global candidate pool's first ``k`` rows."""
    ordered_ids = _ordered_accuracy_candidate_ids(outcome)
    if ordered_ids is None:
        return None
    from ingest.qdrant_store import point_id

    candidate_ids = set(ordered_ids[:k])
    if not evidence_groups:
        return 0.0
    covered = 0
    for alternatives in evidence_groups.values():
        gold_ids = set()
        for key in alternatives:
            source, document_id, chunk_index, version_id = chunk_key_parts(key)
            gold_ids.add(
                point_id(
                    source,
                    document_id,
                    chunk_index,
                    version_id=version_id,
                )
            )
        covered += bool(candidate_ids & gold_ids)
    return covered / len(evidence_groups)


def _candidate_measurements(
    backend, outcome, hits, evidence_groups, relevant_documents
):
    """Return recall@50/80 plus auditable observed-depth metadata."""
    branch_recalls = {
        depth: _accuracy_candidate_recall(outcome, evidence_groups, depth)
        for depth in (50, 80)
    } if outcome is not None else {50: None, 80: None}
    if any(value is not None for value in branch_recalls.values()):
        candidate_ids = _ordered_accuracy_candidate_ids(outcome) or ()
        candidate_ranking = [
            _trace_ranked_point(point)
            for point in getattr(outcome, "candidate_ranking", ())
        ]
        exact_accuracy_pool = hasattr(outcome, "candidate_ids")
        exact_production_pool = bool(
            getattr(outcome, "result_hash", "")
            and hasattr(outcome, "candidate_ranking")
        )
        pool_hash = _stable_json_hash(list(candidate_ids))
        document_presence: dict[str, bool | None] = {}
        for depth in (50, 80):
            if candidate_ranking:
                document_presence[str(depth)] = any(
                    (
                        row.get("source"),
                        row.get("document_id"),
                        row.get("version_id"),
                    )
                    in relevant_documents
                    for row in candidate_ranking[:depth]
                )
            else:
                document_presence[str(depth)] = None
        return branch_recalls, {
            "kind": (
                "accuracy_pre_rerank_pool"
                if exact_accuracy_pool
                else (
                    "production_pre_rerank_pool"
                    if exact_production_pool
                    else "accuracy_ordered_union"
                )
            ),
            "source": (
                "outcome.candidate_ids"
                if exact_accuracy_pool
                else (
                    "outcome.candidate_ranking"
                    if exact_production_pool
                    else "legacy_branch_reconstruction"
                )
            ),
            "observed": len(candidate_ids),
            "pool_depth": len(candidate_ids),
            "observed_at_depth": {
                str(depth): min(depth, len(candidate_ids)) for depth in (50, 80)
            },
            "pool_sha256": pool_hash,
            # Compatibility alias for experiment readers written before exact pool capture.
            "candidate_ids_sha256": pool_hash,
            "ordered_candidates": candidate_ranking,
            "gold_document_present_at_depth": document_presence,
            "gold_passage_present_at_depth": {
                str(depth): bool(branch_recalls[depth]) for depth in (50, 80)
            },
        }

    candidate_hits = getattr(backend, "last_candidate_hits", None)
    if candidate_hits is None and getattr(backend, "supports_candidate_depth", False):
        candidate_hits = hits
    if candidate_hits is None:
        return {50: None, 80: None}, {
            "kind": "unavailable",
            "observed": 0,
            "ordered_candidates": [],
            "gold_document_present_at_depth": {"50": None, "80": None},
            "gold_passage_present_at_depth": {"50": None, "80": None},
        }
    ranked = reduce_ranking(candidate_hits, "chunk")
    candidate_records = _ranked_hit_records(candidate_hits)
    relevant_without_null_version = {
        (source, document_id, version_id)
        for source, document_id, version_id in relevant_documents
    }
    return (
        {depth: evidence_group_recall(ranked, evidence_groups, depth) for depth in (50, 80)},
        {
            "kind": "ranked_candidates",
            "observed": len(ranked),
            "ordered_candidates": candidate_records,
            "gold_document_present_at_depth": {
                str(depth): any(
                    (row["source"], row["document_id"], row.get("version_id"))
                    in relevant_without_null_version
                    for row in candidate_records[:depth]
                )
                for depth in (50, 80)
            },
            "gold_passage_present_at_depth": {
                str(depth): bool(evidence_group_recall(ranked, evidence_groups, depth))
                for depth in (50, 80)
            },
        },
    )


def run_mode(backend, gold, rel, mode, level, k):
    """Evaluate one track without removing failed queries from any denominator."""
    scores = []
    lat = {
        "embed": [], "search": [], "rerank": [], "total": [],
        "degraded": [], "skipped": [], "exceptions": [], "queries": [],
    }
    for q in gold:
        query_rel = rel[q.id]
        evidence_groups = query_rel.get("evidence_groups")
        if evidence_groups is None:
            # Compatibility for unit fixtures and third-party callers built against eval-r1:
            # the flattened qrel set represented one evidence span in frozen v1/v2.
            evidence_groups = {"legacy": frozenset(query_rel.get("chunk", {}))}
        cluster_id = query_rel.get("cluster_id") or getattr(q, "cluster_id", "") or q.id
        failure_reason = query_rel.get("failure_reason")
        source = getattr(q, "gold_source", None) or getattr(q, "source", "")
        tags = tuple(getattr(q, "tags", ()) or ())
        risk_level = str(getattr(q, "risk_level", "") or "")
        expected_outcome = str(getattr(q, "expected_outcome", "answer") or "answer")
        answerable = expected_outcome not in {"abstain", "clarify", "unanswerable"}

        def append_failure(reason, *, status, category):
            score = query_score(
                q.id,
                q.query_type,
                q.query_language,
                [],
                query_rel[level],
                level,
                evidence_groups=evidence_groups,
                candidate_recall={50: 0.0, 80: 0.0},
                cluster_id=cluster_id,
                failed=True,
                failure_reason=reason,
                source=source,
                tags=tags,
                risk_level=risk_level,
                expected_outcome=expected_outcome,
            )
            scores.append(score)
            event = {"query_id": q.id, "reason": reason}
            lat[category].append(event)
            material = {
                "query_id": q.id,
                "status": status,
                "answerable": answerable,
                "expected_outcome": expected_outcome,
                "failure_reason": reason,
                "ranked_hits": [],
                "branches": [],
                "route_decisions": None,
                "entity_matches": None,
                "document_matches": {
                    "rank_1_correct": False,
                    "candidate_depth_50": False,
                    "candidate_depth_80": False,
                },
                "candidate_depth": {
                    "kind": "unavailable",
                    "observed": 0,
                    "ordered_candidates": [],
                    "gold_document_present_at_depth": {"50": False, "80": False},
                    "gold_passage_present_at_depth": {"50": False, "80": False},
                },
                "score": dataclasses.asdict(score),
            }
            ranking_material = {"ranked_hits": [], "candidate_depth": material["candidate_depth"]}
            lat["queries"].append(
                {
                    **material,
                    "timings_ms": {"total": 0.0},
                    "ranking_hash": _stable_json_hash(ranking_material),
                    "decision_result_hash": _stable_json_hash(material),
                    # Historical alias; release artifacts use the two hashes above.
                    "result_hash": _stable_json_hash(material),
                }
            )

        if failure_reason:
            append_failure(failure_reason, status="unscorable", category="skipped")
            continue

        supports_candidate_depth = bool(
            getattr(backend, "supports_candidate_depth", False)
        )
        candidate_depth_via_native_trace = bool(
            getattr(backend, "candidate_depth_via_native_trace", False)
        )
        fetch_k = (
            max(k, 80)
            if supports_candidate_depth and not candidate_depth_via_native_trace
            else k
        )
        try:
            hits, stage = backend.search(q.query, mode, fetch_k)
        except Exception as exc:
            append_failure(
                f"retrieval_exception:{type(exc).__name__}:{exc}",
                status="exception",
                category="exceptions",
            )
            continue
        outcome = getattr(backend, "last_outcome", None)
        degraded = bool(outcome is not None and outcome.degraded)
        relevant_documents = {
            (key[0], key[1], key[2] if len(key) == 3 else None)
            for key in query_rel.get("doc", {})
        }
        candidate_recall, candidate_depth = _candidate_measurements(
            backend, outcome, hits, evidence_groups, relevant_documents
        )
        service_abstention = bool(
            outcome is not None and getattr(outcome, "service_abstention", False)
        )
        answerable_abstention = service_abstention and answerable
        execution_failure_reason = None
        if degraded:
            lat["degraded"].append({
                "query_id": q.id,
                "reason": outcome.degraded_reason,
                "timings_ms": dict(outcome.timings_ms),
            })
            # Strict evaluation treats every degraded execution as an accuracy failure even
            # if a fallback returned plausible-looking hits.
            scored_hits = []
            candidate_recall = {50: 0.0, 80: 0.0}
            execution_failure_reason = str(outcome.degraded_reason or "degraded_retrieval")
        elif answerable_abstention:
            scored_hits = []
            candidate_recall = {50: 0.0, 80: 0.0}
            execution_failure_reason = (
                "answerable_abstention:"
                + str(getattr(outcome, "abstention_reason", None) or "unspecified")
            )
        else:
            scored_hits = hits
        score = query_score(
            q.id,
            q.query_type,
            q.query_language,
            scored_hits,
            query_rel[level],
            level,
            evidence_groups=evidence_groups,
            candidate_recall=candidate_recall,
            cluster_id=cluster_id,
            failed=execution_failure_reason is not None,
            failure_reason=execution_failure_reason,
            source=source,
            tags=tags,
            risk_level=risk_level,
            expected_outcome=expected_outcome,
        )
        scores.append(score)
        for s in ("embed", "search", "rerank"):
            lat[s].append(stage.get(s, 0.0))
        lat["total"].append(sum(stage.values()))
        route = getattr(outcome, "effective_route", None)
        plan = getattr(outcome, "plan", None)
        if route is None and plan is not None:
            route = {
                "intent": getattr(getattr(plan, "intent", None), "value", None),
                "language": getattr(getattr(plan, "language", None), "value", None),
            }
        branch_traces = []
        for branch in getattr(outcome, "branches", ()):
            ranked_branch = [
                _trace_ranked_point(point) for point in getattr(branch, "hits", ())
            ]
            if not ranked_branch:
                ranked_branch = [
                    {"point_id": str(point_id), "score": "missing"}
                    for point_id in branch.hit_ids
                ]
            branch_traces.append(
                {
                    "name": branch.name,
                    "query": branch.query,
                    "filters": dict(branch.filters),
                    "route": getattr(branch, "route", "unknown"),
                    "hit_ids": list(branch.hit_ids),
                    "hits": ranked_branch,
                    "elapsed_ms": getattr(branch, "elapsed_ms", None),
                    "is_original_query": branch.query == q.query,
                    "is_translated_query": branch.query != q.query,
                }
            )
        route_decisions = {
            "effective_route": route,
            "executor": dict(getattr(outcome, "route_decisions", {}) or {}),
            "plan": _plan_trace(plan),
            "query_variants": [
                {
                    "branch": branch["name"],
                    "query": branch["query"],
                    "route": branch["route"],
                    "is_original": branch["is_original_query"],
                    "is_translated": branch["is_translated_query"],
                }
                for branch in branch_traces
            ],
        }
        status = (
            "degraded" if degraded
            else "failed_abstention" if answerable_abstention
            else "abstain" if service_abstention
            else "ok"
        )
        material = {
            "query_id": q.id,
            "status": status,
            "answerable": answerable,
            "expected_outcome": expected_outcome,
            "failure_reason": execution_failure_reason,
            "route_decisions": route_decisions,
            "entity_matches": (
                (route_decisions.get("plan") or {}).get("entities")
                if route_decisions.get("plan")
                else None
            ),
            "document_matches": {
                "rank_1_correct": bool(score.document_identity1),
                "candidate_depth_50": (
                    candidate_depth.get("gold_document_present_at_depth", {}).get("50")
                ),
                "candidate_depth_80": (
                    candidate_depth.get("gold_document_present_at_depth", {}).get("80")
                ),
            },
            "abstention_reason": getattr(outcome, "abstention_reason", None),
            "degraded_reason": getattr(outcome, "degraded_reason", None),
            "ranked_hits": _ranked_hit_records(hits),
            "branches": branch_traces,
            "candidate_depth": candidate_depth,
            "backend_result_hash": getattr(outcome, "result_hash", None),
            "retrieval_fingerprint": getattr(outcome, "retrieval_fingerprint", None),
            "generation_id": getattr(outcome, "generation_id", None),
            "translator_version": getattr(outcome, "translator_version", None),
            "identity_ambiguous": getattr(outcome, "identity_ambiguous", False),
            "score": dataclasses.asdict(score),
        }
        ranking_material = {
            "ranked_hits": material["ranked_hits"],
            "candidate_ranking": candidate_depth.get("ordered_candidates", []),
            "branch_rankings": [
                {"name": branch["name"], "route": branch["route"], "hits": branch["hits"]}
                for branch in branch_traces
            ],
        }
        decision_material = {
            **material,
            "branches": [
                {key: value for key, value in branch.items() if key != "elapsed_ms"}
                for branch in branch_traces
            ],
        }
        timings_ms = {name: float(value) * 1000 for name, value in stage.items()}
        timings_ms.setdefault("total", sum(float(value) for value in stage.values()) * 1000)
        lat["queries"].append({
            **material,
            "outcome_timings_ms": dict(getattr(outcome, "timings_ms", {})),
            "timings_ms": timings_ms,
            "ranking_hash": _stable_json_hash(ranking_material),
            "decision_result_hash": _stable_json_hash(decision_material),
            "result_hash": _stable_json_hash(decision_material),
        })
    lat["ranking_hash"] = _stable_json_hash(
        [{"query_id": row["query_id"], "ranking_hash": row["ranking_hash"]}
         for row in lat["queries"]]
    )
    lat["decision_result_hash"] = _stable_json_hash(
        [
            {"query_id": row["query_id"], "decision_result_hash": row["decision_result_hash"]}
            for row in lat["queries"]
        ]
    )
    lat["result_hash"] = lat["decision_result_hash"]
    return scores, lat


def run_mode_repeated(backend, gold, rel, mode, level, k, *, repeats: int = 1):
    """Run an identical track repeatedly and fail on ranking or decision drift."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    first_scores, first_lat = run_mode(backend, gold, rel, mode, level, k)
    ranking_hashes = [first_lat["ranking_hash"]]
    decision_hashes = [first_lat["decision_result_hash"]]
    per_query_ranking = [
        {row["query_id"]: row["ranking_hash"] for row in first_lat["queries"]}
    ]
    per_query_decisions = [
        {row["query_id"]: row["decision_result_hash"] for row in first_lat["queries"]}
    ]
    for repeat_index in range(1, repeats):
        _scores, repeated_lat = run_mode(backend, gold, rel, mode, level, k)
        ranking_hashes.append(repeated_lat["ranking_hash"])
        decision_hashes.append(repeated_lat["decision_result_hash"])
        current_ranking = {
            row["query_id"]: row["ranking_hash"] for row in repeated_lat["queries"]
        }
        current_decisions = {
            row["query_id"]: row["decision_result_hash"] for row in repeated_lat["queries"]
        }
        per_query_ranking.append(current_ranking)
        per_query_decisions.append(current_decisions)
        if (
            current_ranking != per_query_ranking[0]
            or current_decisions != per_query_decisions[0]
        ):
            changed = sorted(
                query_id
                for query_id in set(per_query_decisions[0]) | set(current_decisions)
                if (
                    per_query_ranking[0].get(query_id) != current_ranking.get(query_id)
                    or per_query_decisions[0].get(query_id) != current_decisions.get(query_id)
                )
            )
            raise RuntimeError(
                f"non-deterministic {mode} evaluation on repeat {repeat_index + 1}: "
                f"changed query result hashes {changed[:20]}"
            )
    first_lat["repeat_count"] = repeats
    first_lat["repeat_ranking_hashes"] = ranking_hashes
    first_lat["repeat_decision_result_hashes"] = decision_hashes
    first_lat["repeat_result_hashes"] = decision_hashes
    first_lat["deterministic_repeats"] = (
        len(set(ranking_hashes)) == 1 and len(set(decision_hashes)) == 1
    )
    return first_scores, first_lat


def candidate_metrics_complete(scores) -> bool:
    """Whether candidate recall@50/@80 was observed for every evaluated query."""

    return bool(scores) and all(
        score.candidate_recall50 is not None and score.candidate_recall80 is not None
        for score in scores
    )


def quality_gate_invalid_reasons(mode, scores, latency) -> list[str]:
    """Return deterministic run-level gate failures for one evaluator track."""

    base_mode = mode.split("+")[0]
    failed_count = sum(1 for score in scores if score.failed)
    return [
        reason
        for condition, reason in (
            (failed_count > 0, "failed_or_degraded_queries"),
            (
                not bool(latency.get("deterministic_repeats", True)),
                "non_deterministic_results",
            ),
            (
                base_mode in PRODUCTION_MODES and latency.get("repeat_count", 1) < 2,
                "production_track_not_repeated",
            ),
            (
                base_mode == "accuracy_strict" and not candidate_metrics_complete(scores),
                "candidate_metrics_incomplete",
            ),
        )
        if condition
    ]


def selected_modes(backend_kind: str, requested: str | None, *, has_translations: bool) -> list[str]:
    """Choose tracks without ever mixing authored translations into direct production."""

    mode = requested or ("production" if backend_kind == "qdrant" else "all")
    if mode == "all":
        modes = ["production", *MODES] if backend_kind == "qdrant" else list(MODES)
        if backend_kind == "qdrant" and has_translations:
            modes[1:1] = ["accuracy_strict", "client_translated"]
        return modes
    if mode in PRODUCTION_MODES and backend_kind != "qdrant":
        raise ValueError("production-parity modes require --backend qdrant")
    if mode in {"client_translated", "accuracy_strict"} and not has_translations:
        raise ValueError(f"{mode} requires --translate-queries PATH for the English eval slice")
    return [mode]


def _fmt_ci(ci) -> str:
    if ci is None:
        return "N/A"
    return f"{ci.mean:.3f} [{ci.lo:.3f},{ci.hi:.3f}]"


def _metric_values_and_clusters(scores, metric):
    selected = [score for score in scores if getattr(score, metric) is not None]
    return [float(getattr(score, metric)) for score in selected], [score.cluster_id for score in selected]


def _metric_ci(scores, metric):
    metric_values, clusters = _metric_values_and_clusters(scores, metric)
    return bootstrap_ci(metric_values, clusters=clusters) if metric_values else None


def _metric_ci_record(scores, metric):
    metric_values, clusters = _metric_values_and_clusters(scores, metric)
    if not metric_values:
        return None
    ci = bootstrap_ci(metric_values, clusters=clusters)
    return {
        **ci.__dict__,
        "n": len(metric_values),
        "n_clusters": len(set(clusters)),
        "method": "percentile cluster bootstrap",
        "resampling_unit": "document/version family",
        "alpha": 0.05,
        "resamples": DEFAULT_RESAMPLES,
        "seed": DEFAULT_SEED,
    }


def print_table(results, level, k):
    print(f"\n== Accuracy metrics (relevance={level}, selected_k={k}) — cluster mean [95% CI] ==")
    for mode, (scores, lat) in results.items():
        print(f"\n  {mode}:")
        for m in METRIC_NAMES:
            metric_values = values(scores, m)
            print(f"    {m:<30} {_fmt_ci(_metric_ci(scores, m)):<24} n={len(metric_values)}")
        tot = percentiles(lat["total"])
        print(f"    {'lat p50/p95 (ms)':<30} {tot['p50'] * 1000:.1f}/{tot['p95'] * 1000:.1f}")


def print_breakdowns(results):
    for mode, (scores, _lat) in results.items():
        print(f"\n-- {mode}: per-query-type --")
        for grp, agg in breakdown(scores, "query_type").items():
            print(f"   {grp:<18} n={agg['n']:<3} " + " ".join(f"{m}={agg[m]:.3f}" for m in METRIC_NAMES))
        print(f"-- {mode}: per-language --")
        for grp, agg in breakdown(scores, "language").items():
            print(f"   {grp:<18} n={agg['n']:<3} " + " ".join(f"{m}={agg[m]:.3f}" for m in METRIC_NAMES))


def print_stage_latency(results):
    print("\n== Per-stage latency p50/p95 (ms) ==")
    print(f"{'mode':<8}{'embed':>16}{'search':>16}{'rerank':>16}")
    for mode, (_scores, lat) in results.items():
        row = f"{mode:<8}"
        for s in ("embed", "search", "rerank"):
            p = percentiles(lat[s])
            row += f"{p['p50'] * 1000:>7.1f}/{p['p95'] * 1000:<8.1f}"
        print(row)


def print_degraded(results) -> None:
    print("\n== Failed executions (retained as zero-score failures) ==")
    any_degraded = False
    for mode, (_scores, lat) in results.items():
        events = lat.get("degraded", [])
        skipped = lat.get("skipped", [])
        exceptions = lat.get("exceptions", [])
        if events or skipped or exceptions:
            any_degraded = True
        if events:
            print(f"   {mode}: {len(events)} degraded quer{'y' if len(events) == 1 else 'ies'}")
            for event in events[:5]:
                print(f"      {event['query_id']}: {event['reason']}")
        if skipped:
            print(f"   {mode}: {len(skipped)} unscorable/unchunkable quer"
                  f"{'y' if len(skipped) == 1 else 'ies'}")
            for event in skipped[:5]:
                print(f"      {event['query_id']}: {event['reason']}")
        if exceptions:
            print(f"   {mode}: {len(exceptions)} retrieval exception(s)")
            for event in exceptions[:5]:
                print(f"      {event['query_id']}: {event['reason']}")
    if not any_degraded:
        print("   none")


def _value(value, field):
    return value.get(field) if isinstance(value, dict) else getattr(value, field, None)


def _resolve_physical_collection(client, target: str, generation_id: str) -> str:
    """Validate an exact physical target or the one supported serving alias.

    Direct candidate evaluation deliberately avoids alias lookup.  Alias evaluation is
    permitted only through ``SERVING_ALIAS`` and only when the complete alias inventory
    contains one mapping to the expected physical generation.
    """
    expected = physical_collection_name(generation_id)
    if target == expected:
        return expected
    if target != SERVING_ALIAS:
        raise RuntimeError(
            f"evaluation target must be {expected!r} or serving alias "
            f"{SERVING_ALIAS!r}; got {target!r}"
        )
    response = client.get_aliases()
    aliases = _value(response, "aliases")
    if not isinstance(aliases, list):
        raise RuntimeError("Qdrant alias inventory is unavailable")
    targets = [
        _value(item, "collection_name")
        for item in aliases
        if _value(item, "alias_name") == SERVING_ALIAS
    ]
    if targets != [expected]:
        raise RuntimeError(
            f"serving alias {SERVING_ALIAS!r} must resolve exactly to "
            f"{expected!r}; got {targets!r}"
        )
    return expected


def _header_identity(document_header: bool) -> str:
    material = json.dumps(
        {"document_header": document_header}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(material.encode()).hexdigest()


def build_evaluation_provenance(
    index_info: dict,
    frozen_set_hashes: dict[str, str],
    execution_mode: str,
) -> EvaluationProvenance:
    """Build and validate a quality-claim identity from verified index metadata."""
    empty_patch_hash = hashlib.sha256(b"").hexdigest()
    provenance = EvaluationProvenance(
        collection_alias=index_info["collection_alias"],
        serving_alias=index_info["serving_alias"],
        physical_collection=index_info["physical_collection"],
        queried_collection=index_info["queried_collection"],
        access_kind=index_info["access_kind"],
        generation_id=index_info["generation_id"],
        points_count=index_info["n_points"],
        corpus_hash=index_info["corpus_hash"],
        snapshot_hash=index_info["snapshot_hash"],
        embedding_model=index_info["embedding_model"],
        embedding_revision=index_info["embedding_revision"],
        tokenizer_model=index_info["tokenizer_model"],
        tokenizer_revision=index_info["tokenizer_revision"],
        reranker_model=index_info["reranker_model"],
        reranker_revision=index_info["reranker_revision"],
        vector_space_id=index_info["vector_space_id"],
        chunk_config_id=index_info["chunk_config_id"],
        header_config_id=index_info["header_config_id"],
        retrieval_fingerprint_revision=index_info["retrieval_fingerprint_revision"],
        retrieval_fingerprint=index_info["retrieval_fingerprint"],
        frozen_set_hashes=frozen_set_hashes,
        dependency_identity=index_info["dependency_identity"],
        image_identity=index_info["image_identity"],
        git_sha=index_info["git_sha"],
        dirty_patch_hash=index_info.get("dirty_patch_hash") or empty_patch_hash,
        execution_mode=execution_mode,
    )
    provenance.validate_complete()
    return provenance


def qdrant_deps(
    cfg, *, verification_paths: tuple[Path, Path] | None = None
):
    """Build the shared, expensive Qdrant/BGE deps once (so a paired A/B reuses them)."""
    from ingest.artifacts import load_verified_generation_coverage
    from ingest.collection_compatibility import (
        require_collection_compatibility,
        require_config_manifest_compatibility,
    )
    from ingest.generation import load_generation
    from ingest.qdrant_store import make_client

    if cfg.generation_dir is None:
        raise RuntimeError("production-parity Qdrant evaluation requires GENERATION_DIR")
    manifest = load_generation(cfg.generation_dir).manifest
    release_verification_pair = None
    if manifest.generation_id == IMMUTABLE_512_CANDIDATE_GENERATION:
        if verification_paths is None:
            raise RuntimeError(
                "immutable 512 evaluation requires the two explicit run-specific "
                "physical verification sidecars"
            )
        from .release_verification import validate_physical_verification_pair

        release_verification_pair = validate_physical_verification_pair(
            cfg.generation_dir, verification_paths
        )
    else:
        verified = load_verified_generation_coverage(cfg.generation_dir.parent)
        if not any(
            item.generation_id == manifest.generation_id
            and item.manifest_path == (cfg.generation_dir / "manifest.json").resolve()
            for item in verified
        ):
            raise RuntimeError(
                "production-parity evaluation requires a matching all-green generation "
                "verification sidecar"
            )
    require_config_manifest_compatibility(cfg, manifest)
    client = make_client(cfg)
    physical_collection = _resolve_physical_collection(
        client, cfg.collection_name, manifest.generation_id
    )
    access_kind = (
        "direct_physical"
        if cfg.collection_name == physical_collection
        else "serving_alias"
    )
    compatibility = require_collection_compatibility(
        client, physical_collection, manifest
    )
    index_info = {
        "kind": "qdrant",
        "collection_alias": SERVING_ALIAS,  # legacy provenance field
        "serving_alias": SERVING_ALIAS,
        "collection": cfg.collection_name,  # legacy report-builder alias
        "collection_name": cfg.collection_name,
        "physical_collection": physical_collection,
        "queried_collection": cfg.collection_name,
        "access_kind": access_kind,
        "generation_id": manifest.generation_id,
        "n_points": compatibility.points_count,
        "points": compatibility.points_count,  # legacy report-builder alias
        "points_count": compatibility.points_count,
        "corpus_hash": manifest.source.state_sha256,
        "snapshot_hash": manifest.corpus.snapshot_sha256,
        "retrieval_fingerprint_revision": manifest.retrieval_fingerprint_revision,
        "retrieval_fingerprint": manifest.retrieval_fingerprint,
        "embedding_model": manifest.model.embedding_model,
        "embed_model": manifest.model.embedding_model,  # legacy report-builder alias
        "embedding_revision": manifest.model.embedding_revision,
        "tokenizer_model": manifest.model.tokenizer_model,
        "tokenizer_revision": manifest.model.tokenizer_revision,
        "reranker_model": manifest.model.reranker_model,
        "reranker_revision": manifest.model.reranker_revision,
        "vector_space_id": manifest.vector_space.id,
        "chunk_config_id": manifest.chunking.fingerprint,
        "header_config_id": _header_identity(manifest.chunking.document_header),
        "dependency_identity": manifest.dependency.lock_sha256,
        "image_identity": manifest.dependency.image_digest,
        "git_sha": manifest.code.git_sha,
        "dirty_patch_hash": manifest.code.dirty_patch_sha256,
    }
    if release_verification_pair is not None:
        index_info["verification_1_sha256"] = release_verification_pair.sha256[0]
        index_info["verification_2_sha256"] = release_verification_pair.sha256[1]
    # Reject incomplete image/model/runtime identity before heavyweight imports.
    build_evaluation_provenance(
        index_info, {"preflight": "0" * 64}, "preflight"
    )

    from ingest.embedding import BGEM3Embedder

    embedder = BGEM3Embedder(cfg)
    reranker = None
    if cfg.rerank_enabled:
        import os
        remote = os.environ.get("RERANK_REMOTE_URL")
        if remote:  # offload cross-encoder scoring to a GPU pod (retrieval stays local)
            from ingest.rerank import RemoteBGEReranker
            reranker = RemoteBGEReranker(remote)
        else:
            from ingest.rerank import make_reranker
            reranker = make_reranker(cfg)
    return client, embedder, reranker, index_info


def make_backend(kind, cfg, gold, bodies, chunk_cfg, count_tokens, *, knobs=None, deps=None):
    knobs = knobs or {}
    if kind == "fake":
        records = build_fake_corpus(gold, bodies, chunk_cfg, count_tokens)
        return FakeBackend(records), {"kind": "fake", "n_chunks": len(records)}
    from .backend import QdrantBackend

    client, embedder, reranker, info = deps or qdrant_deps(cfg)
    return QdrantBackend(cfg, client, embedder, reranker=reranker, **knobs), info


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate legal RAG retrieval quality.")
    ap.add_argument("--backend", choices=("fake", "qdrant"), default="qdrant")
    ap.add_argument(
        "--mode",
        choices=(*MODES, *PRODUCTION_MODES, "all"),
        default=None,
        help="default: production for qdrant; all ablations for fake",
    )
    ap.add_argument("--relevance", choices=("chunk", "doc"), default="chunk")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--tokenizer", choices=("word", "bge"), default=None,
                    help="chunking token counter; default: word for fake, bge for qdrant")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="paired A/B on two modes")
    # Retrieval-tuning knobs (all measured through the harness; none re-embed). Each changes
    # the config_hash so its run is a distinct, forever-comparable row in experiments.jsonl.
    ap.add_argument("--rerank-candidates", type=int, default=None, help="rerank pool depth")
    ap.add_argument("--fusion", choices=("rrf", "dbsf"), default="rrf", help="hybrid fusion")
    ap.add_argument("--prefetch-limit", type=int, default=None, help="override recall pool depth")
    ap.add_argument("--hnsw-ef", type=int, default=None, help="HNSW search ef")
    ap.add_argument("--rescore", choices=("on", "off"), default=None, help="int8 rescore")
    ap.add_argument("--sparse-weight", type=float, default=None, help="weighted dense/sparse fusion")
    ap.add_argument("--max-per-doc", type=int, default=None, help="diversity: cap chunks per doc")
    ap.add_argument("--mmr-lambda", type=float, default=None, help="diversity: MMR trade-off 0..1")
    ap.add_argument("--translate-queries", metavar="PATH", default=None,
                    help="I2: authored EN→KA query-translation JSON (embeds the KA text)")
    ap.add_argument("--citation-route", choices=("ids", "full"), default=None,
                    help="I1: pin exact citation/alias matches above semantic hits")
    ap.add_argument("--golden-set", choices=sorted(goldset.EVAL_SETS), default="v2",
                    help="I5: eval-set version (default v2 — the gating yardstick since 2026-07-11, "
                         "frozen at 337 pairs, hash 753e2985315be3e4). Pass --golden-set v1 for the "
                         "frozen historical anchor; v1 config_hashes stay comparable to pre-switch rows.")
    ap.add_argument(
        "--candidate-qrels",
        type=Path,
        default=None,
        help=(
            "exact create-only v2-to-candidate qrel artifact; the immutable release "
            "workflow prohibits fuzzy re-anchoring"
        ),
    )
    ap.add_argument(
        "--physical-verification",
        type=Path,
        action="append",
        default=[],
        help=(
            "run-specific physical verification sidecar; the immutable 512 candidate "
            "requires exactly two in canonical 01,02 order"
        ),
    )
    ap.add_argument("--ab", action="store_true",
                    help="paired A/B: --mode with knobs OFF (A) vs the given knobs ON (B)")
    ap.add_argument(
        "--repeat", type=int, default=None,
        help="identical repeats per track (default: 2 for production tracks, otherwise 1)",
    )
    ap.add_argument("--log", action="store_true", help="append runs to the experiment log")
    ap.add_argument("--log-path", default=str(explog.DEFAULT_LOG), help="experiment-log path")
    args = ap.parse_args()
    if args.top_k < 10:
        raise SystemExit("--top-k must be >= 10 because the canonical metrics include @10")
    if args.repeat is not None and args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    cfg = load_config()
    if (
        getattr(cfg, "generation_id", None) == IMMUTABLE_512_CANDIDATE_GENERATION
        and args.candidate_qrels is None
    ):
        raise SystemExit(
            "the immutable 512 candidate requires --candidate-qrels; unversioned v2 "
            "labels may not be fuzzy or implicitly rebound"
        )
    chunk_cfg = {
        "max_tokens": cfg.chunk_tokens,
        "overlap": cfg.chunk_overlap,
        "min_tokens": cfg.chunk_min_tokens,
    }
    tok_kind = args.tokenizer or ("word" if args.backend == "fake" else "bge")
    count_tokens = _token_counter(tok_kind, cfg.tokenizer_model, cfg.tokenizer_revision)

    eval_spec = goldset.EVAL_SETS[args.golden_set]
    eval_hash = goldset.eval_set_hash(eval_spec.gold)
    eval_file_sha = _file_sha256(eval_spec.gold)
    holdout_hash = _file_sha256(eval_spec.holdout)
    gold = goldset.load_golden_set(eval_spec.gold)
    holdout = goldset.load_holdout(eval_spec.holdout)
    bodies = goldset.SnapshotBodies(
        root=eval_spec.roots[0], needed=goldset.gold_docs(gold), extra_roots=eval_spec.roots[1:]
    )

    n_spans = goldset.reground(gold, bodies)
    goldset.enforce_holdout(gold, holdout)
    candidate_qrel_artifact = None
    candidate_qrel_hash = None
    candidate_qrel_failures: dict[str, str] = {}
    if args.candidate_qrels is not None:
        if args.golden_set != "v2":
            raise SystemExit("--candidate-qrels is valid only with --golden-set v2")
        from .v2_candidate_qrels import (
            artifact_sha256,
            bind_gold_queries_to_candidate,
            load_v2_candidate_qrel_artifact,
        )

        candidate_qrel_artifact = load_v2_candidate_qrel_artifact(args.candidate_qrels)
        gold, candidate_qrel_failures = bind_gold_queries_to_candidate(
            gold, candidate_qrel_artifact
        )
        candidate_qrel_hash = artifact_sha256(args.candidate_qrels)
    rel = build_query_relevance(gold, bodies, chunk_cfg, count_tokens)
    for query_id, reason in candidate_qrel_failures.items():
        query_rel = rel[query_id]
        query_rel["issues"].append(f"candidate_qrel:{reason}")
        query_rel["failure_reason"] = (
            reason
            if reason == "frozen_incomplete_source_label"
            else f"candidate_qrel:{reason}"
        )
    n_linted = sum(len(query_rel["evidence_groups"]) for query_rel in rel.values())
    unscorable = {
        query_id: query_rel["failure_reason"]
        for query_id, query_rel in rel.items()
        if query_rel["failure_reason"]
    }
    print(
        f"Loaded {len(gold)} gold queries · {len(holdout)} holdout docs · "
        f"re-grounding {n_spans}/0 drift · evidence-groups {n_linted} · "
        f"unscorable {len(unscorable)} (counted as failures) "
        f"(eval_set={eval_spec.version}, tokenizer={tok_kind})"
    )

    rescore = {"on": True, "off": False}.get(args.rescore)
    translations, translations_hash = (None, None)
    translations_file_sha = None
    if args.translate_queries:
        from .translations import load_query_translations

        translations_path = Path(args.translate_queries)
        translations, translations_hash = load_query_translations(translations_path, gold)
        translations_file_sha = _file_sha256(translations_path)
        uncovered = {q.query for q in gold if q.query_language == "en"} - set(translations)
        if uncovered:
            print(
                f"WARNING: translations cover {len(translations)} EN queries; {len(uncovered)} EN "
                f"gold queries have none (wrong file for --golden-set {args.golden_set}?)"
            )
    knobs = {
        "rerank_candidates": args.rerank_candidates, "fusion": args.fusion,
        "prefetch_limit": args.prefetch_limit, "hnsw_ef": args.hnsw_ef, "rescore": rescore,
        "sparse_weight": args.sparse_weight, "max_per_doc": args.max_per_doc,
        "mmr_lambda": args.mmr_lambda, "translations": None,
        "citation_route": args.citation_route,
    }
    # The raw translation dict never enters config_hash/logs — its file content hash does.
    active_knobs = {k: v for k, v in knobs.items()
                    if v not in (None, "rrf") and k != "translations"}
    # I5: non-default eval set is eval config too — v1 folds nothing, so history is stable.
    active_knobs.update(eval_set_knob(args.golden_set))
    # Env-selected reranker backend (I7) is eval config too — fold non-default into the hash.
    if args.backend == "qdrant" and cfg.rerank_enabled and cfg.rerank_backend != "torch":
        active_knobs["rerank_backend"] = cfg.rerank_backend
    # I8: reranker input construction/truncation change retrieval behavior — fold non-default
    # into the hash so two runs that differ only by these env vars don't collide (G5).
    if args.backend == "qdrant" and cfg.rerank_enabled and cfg.rerank_context_enriched:
        active_knobs["rerank_context_enriched"] = True
    if args.backend == "qdrant" and cfg.rerank_enabled and cfg.rerank_max_length != 512:
        active_knobs["rerank_max_length"] = cfg.rerank_max_length

    verification_paths = (
        tuple(args.physical_verification) if args.physical_verification else None
    )
    if verification_paths is not None and len(verification_paths) != 2:
        raise SystemExit("--physical-verification must be supplied exactly twice")
    deps = (
        qdrant_deps(cfg, verification_paths=verification_paths)
        if args.backend == "qdrant"
        else None
    )
    backend, index_info = make_backend(
        args.backend, cfg, gold, bodies, chunk_cfg, count_tokens, knobs=knobs, deps=deps)
    production_backend = None
    if deps is not None:
        client, embedder, reranker, _info = deps
        production_backend = ProductionBackend(
            cfg, client, embedder, reranker=reranker, translations=translations
        )
    print(f"Backend: {index_info}  knobs={active_knobs or 'defaults'}")
    if candidate_qrel_artifact is not None and args.backend == "qdrant":
        if (
            candidate_qrel_artifact["candidate_snapshot"]["snapshot_sha256"]
            != index_info["snapshot_hash"]
        ):
            raise SystemExit(
                "candidate qrel snapshot does not match the evaluated physical collection"
            )

    if args.ab:
        m = args.mode if args.mode not in (None, "all") else "rerank"
        if m in PRODUCTION_MODES:
            raise SystemExit(
                "--ab tuning knobs are ablations; compare production tracks with --compare"
            )
        base, _ = make_backend(
            args.backend, cfg, gold, bodies, chunk_cfg, count_tokens, knobs=None, deps=deps)
        repeats = args.repeat or 1
        results = {
            m: run_mode_repeated(
                base, gold, rel, m, args.relevance, args.top_k, repeats=repeats
            ),
            f"{m}+knobs": run_mode_repeated(
                backend, gold, rel, m, args.relevance, args.top_k, repeats=repeats
            ),
        }
        args.compare = [m, f"{m}+knobs"]  # drive the paired-compare + logging paths below
    else:
        try:
            modes = selected_modes(
                args.backend, args.mode, has_translations=translations is not None
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if args.compare:
            modes = list(dict.fromkeys(args.compare))
        unknown = set(modes) - set(MODES) - set(PRODUCTION_MODES)
        if unknown:
            raise SystemExit(f"unknown comparison mode(s): {sorted(unknown)}")
        results = {}
        for mode in modes:
            selected_backend = production_backend if mode in PRODUCTION_MODES else backend
            if selected_backend is None:
                raise SystemExit(f"{mode} requires --backend qdrant")
            repeats = args.repeat or (2 if mode in PRODUCTION_MODES else 1)
            results[mode] = run_mode_repeated(
                selected_backend, gold, rel, mode, args.relevance, args.top_k,
                repeats=repeats,
            )

    provenances: dict[str, EvaluationProvenance] = {}
    if args.backend == "qdrant":
        for label in results:
            frozen_hashes = {
                f"golden_{eval_spec.version}": eval_file_sha,
                f"holdout_{eval_spec.version}": holdout_hash,
            }
            if label.split("+")[0] in {"client_translated", "accuracy_strict"} and translations_file_sha:
                frozen_hashes["authored_query_translations"] = translations_file_sha
            if candidate_qrel_hash:
                frozen_hashes["v2_candidate_qrels"] = candidate_qrel_hash
            provenances[label] = build_evaluation_provenance(
                index_info, frozen_hashes, label
            )
        print(
            "Evaluation provenance: "
            + json.dumps(
                {label: value.to_dict() for label, value in provenances.items()},
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    print_table(results, args.relevance, args.top_k)
    print_stage_latency(results)
    print_degraded(results)
    print_breakdowns(results)

    comparison_bundle = None
    if args.compare:
        a, b = args.compare
        compared_degraded = bool(results[a][1].get("degraded") or results[b][1].get("degraded"))
        if compared_degraded:
            print("\n== Paired A/B invalid: a compared track degraded ==")
            comparison_bundle = {
                "baseline": a, "candidate": b, "valid": False,
                "reason": "compared_track_degraded", "metrics": {},
            }
        else:
            print(f"\n== Paired A/B: {a} (A) vs {b} (B), relevance={args.relevance} ==")
            raw_comparisons = {}
            for m in METRIC_NAMES:
                aa, bb, clusters = paired_values(results[a][0], results[b][0], m)
                if not aa:
                    continue
                raw_comparisons[m] = compare(
                    m,
                    aa,
                    bb,
                    clusters=clusters,
                    higher_is_better=METRIC_DIRECTIONS.get(m, "higher") == "higher",
                )
            corrected = holm_correct(raw_comparisons)
            for m, c in corrected.items():
                print(f"   {m:<30} Δ={c.diff:+.3f} [{c.diff_lo:+.3f},{c.diff_hi:+.3f}] "
                      f"p={c.p_value_adjusted:.4f} → {c.verdict} "
                      f"(Holm; raw p={c.p_value:.4f})")
            paired_ids = [score.id for score in results[a][0]]
            comparison_bundle = {
                "baseline": a,
                "candidate": b,
                "valid": True,
                "paired_query_ids_hash": _stable_json_hash(paired_ids),
                "n_queries": len(paired_ids),
                "method": "document/version-cluster bootstrap + cluster sign-flip",
                "multiplicity": "Holm family-wise correction",
                "alpha": 0.05,
                "resamples": DEFAULT_RESAMPLES,
                "seed": DEFAULT_SEED,
                "metrics": {name: comparison.__dict__ for name, comparison in corrected.items()},
            }

    if args.log:
        for label, (scores, lat) in results.items():
            base_mode = label.split("+")[0]
            # In --ab, the base label logs with knobs OFF; every other run logs the active
            # knobs. Plain runs (no knobs) hash exactly as before → historical rows comparable.
            # The eval set is not a knob you can switch off — the --ab base keeps it.
            if base_mode in PRODUCTION_MODES:
                eff = dict(eval_set_knob(args.golden_set))
                if cfg.rerank_enabled and cfg.rerank_backend != "torch":
                    eff["rerank_backend"] = cfg.rerank_backend
                if cfg.rerank_enabled and cfg.rerank_context_enriched:
                    eff["rerank_context_enriched"] = True
                if cfg.rerank_enabled and cfg.rerank_max_length != 512:
                    eff["rerank_max_length"] = cfg.rerank_max_length
                if base_mode in {"client_translated", "accuracy_strict"} and translations_hash:
                    eff["translate_queries"] = translations_hash
            else:
                eff = (
                    dict(eval_set_knob(args.golden_set))
                    if (args.ab and label == base_mode)
                    else active_knobs
                )
            eff_rc = eff.get("rerank_candidates") or cfg.rerank_candidates
            ch = explog.config_hash({
                "mode": base_mode, "relevance": args.relevance, "top_k": args.top_k,
                "tokenizer": tok_kind, **chunk_cfg,
                "rerank_candidates": eff_rc,
                **{k: v for k, v in eff.items() if k != "rerank_candidates"},
            })
            failed_count = sum(1 for score in scores if score.failed)
            degraded_count = len(lat.get("degraded", []))
            skipped_count = len(lat.get("skipped", []))
            candidate_complete = candidate_metrics_complete(scores)
            invalid_reasons = quality_gate_invalid_reasons(label, scores, lat)
            valid_for_quality_gate = not invalid_reasons
            ci_records = {m: _metric_ci_record(scores, m) for m in METRIC_NAMES}
            ci_records.update({
                alias: ci_records[canonical]
                for alias, canonical in LEGACY_METRIC_ALIASES.items()
            })
            record = {
                "schema_version": explog.SCHEMA_VERSION,
                "scoring_revision": explog.LOGIC_REV,
                "timestamp": explog.now_iso(),
                "eval_set_version": eval_spec.version,
                "eval_set_hash": eval_hash,
                "config_hash": ch,
                "mode": label,
                "relevance_level": args.relevance,
                "top_k": args.top_k,
                # Descriptive-only mirror of the tuning knobs folded into config_hash, so the
                # experiment log is self-describing for ablation tables (does NOT affect the hash).
                "knobs": {"rerank_candidates": eff_rc,
                          **{k: v for k, v in eff.items() if k != "rerank_candidates"}},
                "backend": {
                    **index_info,
                    "execution_mode": label,
                    "evaluated_queries": len(scores),
                    "expected_queries": len(gold),
                    "successful_queries": len(scores) - failed_count,
                    "failed_queries": failed_count,
                    "degraded_queries": degraded_count,
                    "unscorable_queries": skipped_count,
                },
                "run_status": "valid" if valid_for_quality_gate else "invalid",
                "valid_for_quality_gate": valid_for_quality_gate,
                "invalid_reasons": invalid_reasons,
                "evaluation_provenance": (
                    provenances[label].to_dict() if label in provenances else None
                ),
                "metrics": aggregate(scores),
                "metric_definitions": {
                    "success1": "binary any relevant passage at rank 1",
                    "success5": "binary any relevant passage in top 5 (legacy Recall@5)",
                    "success10": "binary any relevant passage in top 10 (legacy Recall@10)",
                    "required_evidence_recall10": "required evidence groups satisfied in top 10",
                    "candidate_recall50": "required evidence groups present in the exact pre-rerank pool at 50",
                    "candidate_recall80": "required evidence groups present in the exact pre-rerank pool at 80",
                    "document_identity1": "rank-1 document is a gold evidence document",
                    "passage_accuracy1": "rank-1 chunk overlaps required evidence",
                    "context_duplication10": "top-10 share adding no new evidence or repeating a chunk",
                    "context_noise10": "top-10 share not overlapping required evidence",
                },
                "cis": ci_records,
                "latency_ms": {
                    s: {kk: v * 1000 for kk, v in percentiles(lat[s]).items()}
                    for s in ("embed", "search", "rerank", "total")
                },
                "per_query_type": breakdown(scores, "query_type"),
                "per_language": breakdown(scores, "language"),
                "per_source": breakdown(scores, "source"),
                "per_risk_level": breakdown(scores, "risk_level"),
                "per_query": [score.__dict__ for score in scores],
                "query_traces": lat.get("queries", []),
                "result_hash": lat.get("result_hash"),
                "ranking_hash": lat.get("ranking_hash"),
                "decision_result_hash": lat.get("decision_result_hash"),
                "repeat_count": lat.get("repeat_count", 1),
                "repeat_result_hashes": lat.get("repeat_result_hashes", [lat.get("result_hash")]),
                "repeat_ranking_hashes": lat.get(
                    "repeat_ranking_hashes", [lat.get("ranking_hash")]
                ),
                "repeat_decision_result_hashes": lat.get(
                    "repeat_decision_result_hashes", [lat.get("decision_result_hash")]
                ),
                "deterministic_repeats": lat.get("deterministic_repeats", True),
                "candidate_metrics_complete": candidate_complete,
                "comparison": comparison_bundle,
            }
            explog.append_run(record, Path(args.log_path))
        print(f"\nAppended {len(results)} run(s) to {args.log_path}")

    failed_production = sum(
        sum(bool(score.failed) for score in scores)
        for label, (scores, _lat) in results.items()
        if label.split("+")[0] in PRODUCTION_MODES
    )
    incomplete_strict_candidates = sum(
        1
        for label, (scores, _lat) in results.items()
        if label.split("+")[0] == "accuracy_strict"
        and not candidate_metrics_complete(scores)
    )
    if failed_production or incomplete_strict_candidates:
        raise SystemExit(
            "production quality gate invalid: "
            f"{failed_production} failed/degraded execution(s), "
            f"{incomplete_strict_candidates} strict track(s) missing candidate metrics"
        )


if __name__ == "__main__":
    main()
