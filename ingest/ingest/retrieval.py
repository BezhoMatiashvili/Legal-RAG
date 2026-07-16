"""Shared production-retrieval contracts and compatibility executor.

The MCP tool, serverless worker, and production-parity evaluation must construct the
same request and execute the same policy.  This module deliberately starts as a thin
adapter over :func:`ingest.search.hybrid_search`; keeping that function as the single
ranking implementation makes the initial refactor behaviour-neutral.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import re
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Any, Mapping

from .config import Config, RETRIEVAL_FINGERPRINT_REVISION, retrieval_fingerprint
from .citations import citation_lookup
from .query_planner import (
    QueryIntent,
    QueryPlan,
    TranslationIntegrityError,
    Translator,
    build_query_variants,
    plan_query,
)
from .legal_references import extract_article_references
from .promotion import SERVING_ALIAS, physical_collection_name
from .search import detect_language, hybrid_search, rerank_points


@dataclass(frozen=True)
class TemporalContext:
    """Explicit temporal bounds carried with a retrieval request."""

    date_from: str | None = None
    date_to: str | None = None
    as_of: str | None = None

    def as_filters(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.date_from:
            out["date_from"] = self.date_from
        if self.date_to:
            out["date_to"] = self.date_to
        if self.as_of:
            # ``as_of`` is legal effectiveness, never a publication-date upper bound.
            out["as_of"] = self.as_of
        return out


@dataclass(frozen=True)
class RetrievalRequest:
    """Provider-neutral input to the production retrieval path."""

    query: str
    requested_limit: int = 10
    route: bool = True
    filters: Mapping[str, Any] = field(default_factory=dict)
    language: str | None = None
    # Declared language of the question.  ``language`` above remains the legacy corpus
    # payload filter; keeping the concepts separate prevents an English question from
    # accidentally filtering out the Georgian corpus.
    query_language: str | None = None
    temporal_context: TemporalContext = field(default_factory=TemporalContext)
    track: str = "direct"

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= self.requested_limit <= 50:
            raise ValueError("requested_limit must be between 1 and 50")
        object.__setattr__(self, "filters", MappingProxyType(dict(self.filters)))

    def search_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.filters)
        if self.language is not None:
            existing = kwargs.get("language")
            if existing is not None and existing != self.language:
                raise ValueError("language conflicts with filters['language']")
            kwargs["language"] = self.language
        for key, value in self.temporal_context.as_filters().items():
            existing = kwargs.get(key)
            if existing is not None and existing != value:
                raise ValueError(f"temporal context conflicts with filters[{key!r}]")
            kwargs[key] = value
        return kwargs


@dataclass(frozen=True)
class RetrievalPolicy:
    """All serving choices that can change candidate selection or final hits."""

    rerank_candidates: int
    rerank_min_score: float | None
    dense_enabled: bool = True
    sparse_enabled: bool = True
    dense_rescore: bool = True
    rerank_enabled: bool = True
    citation_format: str = "production"
    abstain_on_empty: bool = True
    allow_remote_reranker_degraded: bool = True
    max_per_doc: int | None = None
    mmr_lambda: float | None = None

    @classmethod
    def from_config(cls, cfg: Config) -> RetrievalPolicy:
        return cls(
            rerank_candidates=cfg.rerank_candidates,
            rerank_min_score=cfg.rerank_min_score,
            rerank_enabled=cfg.rerank_enabled,
        )

    def candidate_depth(self, requested_limit: int, *, reranker_present: bool) -> int:
        """Mirror the production recall-pool floor in ``hybrid_search``."""

        if reranker_present:
            return max(self.rerank_candidates, requested_limit * 5, 50)
        return max(requested_limit * 5, 50)


@dataclass(frozen=True)
class RetrievalOutcome:
    """Structured result shared by serving and quality evaluation."""

    hits: tuple[Any, ...]
    timings_ms: Mapping[str, float]
    degraded: bool
    degraded_reason: str | None
    service_abstention: bool
    abstention_reason: str | None
    retrieval_fingerprint: str
    effective_route: str
    generation_id: str | None
    track: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "hits", tuple(self.hits))
        object.__setattr__(self, "timings_ms", MappingProxyType(dict(self.timings_ms)))


@dataclass(frozen=True)
class RetrievalBranch:
    """Reproducible record of one candidate branch before global reranking."""

    name: str
    query: str
    filters: Mapping[str, Any]
    hit_ids: tuple[str, ...]
    elapsed_ms: float
    route: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "filters", MappingProxyType(dict(self.filters)))
        object.__setattr__(self, "hit_ids", tuple(self.hit_ids))


@dataclass(frozen=True)
class AccuracyRetrievalOutcome:
    """Strict original+translation, entity-first retrieval result.

    ``result_hash`` binds the plan, branch queries/filters, ordered candidate IDs, and final
    IDs.  It is stable across repeated runs with identical dependencies and therefore can
    be persisted by the answer trace as determinism evidence.
    """

    hits: tuple[Any, ...]
    plan: QueryPlan
    branches: tuple[RetrievalBranch, ...]
    timings_ms: Mapping[str, float]
    service_abstention: bool
    abstention_reason: str | None
    degraded: bool
    degraded_reason: str | None
    identity_ambiguous: bool
    retrieval_fingerprint: str
    generation_id: str | None
    translator_version: str | None
    result_hash: str
    resolved_current_date: str | None = None
    candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "hits", tuple(self.hits))
        object.__setattr__(self, "branches", tuple(self.branches))
        object.__setattr__(self, "candidate_ids", tuple(self.candidate_ids))
        object.__setattr__(self, "timings_ms", MappingProxyType(dict(self.timings_ms)))


def _stable_point_id(point: Any) -> str:
    pid = getattr(point, "id", None)
    if pid is not None:
        return str(pid)
    payload = getattr(point, "payload", None) or {}
    return ":".join(
        str(payload.get(name) or "")
        for name in ("source", "document_id", "version_id", "chunk_index")
    )


def _stable_score(point: Any) -> str:
    raw = getattr(point, "score", None)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return "missing"
    value = float(raw)
    return format(value, ".17g") if math.isfinite(value) else str(value)


def _evidence_equivalence_key(point: Any) -> tuple[Any, ...]:
    payload = getattr(point, "payload", None) or {}
    # A digest of text alone is not legal-evidence identity: the same words may appear
    # in different authorities or in both a current and repealed version.  Schema-v2's
    # passage_id binds the source/document/version/offsets/hash.  The fallback mirrors
    # that scope for synthetic tests and rejects only the same canonical location seen
    # through multiple retrieval branches.
    passage_id = payload.get("passage_id")
    if passage_id:
        return ("passage_id", passage_id)
    return (
        "canonical_location",
        payload.get("source"),
        payload.get("document_id"),
        payload.get("version_id"),
        payload.get("char_start"),
        payload.get("char_end"),
        payload.get("passage_hash"),
        payload.get("text"),
    )


def _union_candidates(*groups: Any) -> list[Any]:
    """Stable union, deduplicating chunks that carry equivalent canonical evidence."""

    out: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        for point in group:
            key = _evidence_equivalence_key(point)
            if key in seen:
                continue
            seen.add(key)
            out.append(point)
    return out


def _accuracy_hash(
    plan: QueryPlan,
    branches: list[RetrievalBranch],
    candidate_ids: tuple[str, ...],
    hits: list[Any] | tuple[Any, ...],
    fingerprint: str,
    resolved_current_date: str,
    *,
    generation_id: str | None,
    abstention_reason: str | None,
    degraded_reason: str | None,
    identity_ambiguous: bool,
    translator_version: str | None,
) -> str:
    material = {
        "question": plan.question,
        "language": plan.language.value,
        "intent": plan.intent.value,
        "as_of": plan.as_of,
        "resolved_current_date": resolved_current_date,
        "fingerprint": fingerprint,
        "generation_id": generation_id,
        "service_abstention": abstention_reason is not None,
        "abstention_reason": abstention_reason,
        "degraded": degraded_reason is not None,
        "degraded_reason": degraded_reason,
        "identity_ambiguous": identity_ambiguous,
        "translator_version": translator_version,
        "branches": [
            {
                "name": branch.name,
                "query": branch.query,
                "filters": dict(branch.filters),
                "hit_ids": list(branch.hit_ids),
                "route": branch.route,
            }
            for branch in branches
        ],
        "candidate_ids": list(candidate_ids),
        "final_ranking": [
            {"point_id": _stable_point_id(point), "score": _stable_score(point)}
            for point in hits
        ],
    }
    blob = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _accuracy_outcome(
    cfg: Config,
    plan: QueryPlan,
    *,
    hits: list[Any] | tuple[Any, ...] = (),
    branches: list[RetrievalBranch] | tuple[RetrievalBranch, ...] = (),
    candidate_ids: tuple[str, ...] = (),
    timings_ms: Mapping[str, float] | None = None,
    abstention_reason: str | None = None,
    degraded_reason: str | None = None,
    identity_ambiguous: bool = False,
    translator_version: str | None = None,
    resolved_current_date: str,
) -> AccuracyRetrievalOutcome:
    branch_list = list(branches)
    hit_list = list(hits)
    fp = retrieval_fingerprint(cfg)
    return AccuracyRetrievalOutcome(
        hits=tuple(hit_list),
        plan=plan,
        branches=tuple(branch_list),
        timings_ms=timings_ms or {},
        service_abstention=abstention_reason is not None,
        abstention_reason=abstention_reason,
        degraded=degraded_reason is not None,
        degraded_reason=degraded_reason,
        identity_ambiguous=identity_ambiguous,
        retrieval_fingerprint=fp,
        generation_id=getattr(cfg, "generation_id", None),
        translator_version=translator_version,
        result_hash=_accuracy_hash(
            plan,
            branch_list,
            candidate_ids,
            hit_list,
            fp,
            resolved_current_date,
            generation_id=getattr(cfg, "generation_id", None),
            abstention_reason=abstention_reason,
            degraded_reason=degraded_reason,
            identity_ambiguous=identity_ambiguous,
            translator_version=translator_version,
        ),
        resolved_current_date=resolved_current_date,
        candidate_ids=candidate_ids,
    )


def _payload_effective_on(payload: Mapping[str, Any], as_of: str) -> bool:
    try:
        when = date.fromisoformat(as_of)
        start = date.fromisoformat(str(payload.get("effective_from") or "")[:10])
        raw_end = payload.get("effective_to")
        end = date.fromisoformat(str(raw_end)[:10]) if raw_end else None
    except ValueError:
        return False
    return bool(payload.get("version_id")) and start <= when and (end is None or when < end)


def _admissible_answer_candidate(point: Any) -> bool:
    """Allow only complete canonical extraction into the answer reranker."""

    payload = getattr(point, "payload", None) or {}
    return (
        payload.get("content_complete") is True
        and payload.get("extraction_status") == "full_text"
    )


def _requires_present_law_proof(plan: QueryPlan) -> bool:
    """Treat an undated rule lookup as requiring operative-version proof.

    Case research and explicitly historical research concern immutable past material.
    Everything else without ``as_of`` can influence a statement of presently operative
    law, even when the planner classified it as an exact article/document or general
    research query rather than spotting an explicit "current" keyword.
    """

    return plan.as_of is None and plan.intent not in {
        QueryIntent.CASE_LOOKUP,
        QueryIntent.CASES_APPLYING_LAW,
        QueryIntent.HISTORICAL,
    }


def _resolved_current_date(value: str | date | None) -> str:
    if value is None:
        return date.today().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError) as exc:
        raise ValueError("current_date must be an ISO date (YYYY-MM-DD)") from exc


def _is_normative_payload(payload: Mapping[str, Any]) -> bool:
    return payload.get("source") == "matsne" or payload.get("document_type") in {
        "legislation",
        "normative_act",
    }


def _presently_operative(payload: Mapping[str, Any], current_date: str) -> bool:
    """Prove that a normative/legislative candidate is operative on the bound date."""

    if not _is_normative_payload(payload):
        return True
    if payload.get("status") != "in_force" or not payload.get("version_id"):
        return False
    try:
        when = date.fromisoformat(current_date)
        start = date.fromisoformat(str(payload.get("effective_from") or "")[:10])
        raw_end = payload.get("effective_to")
        end = date.fromisoformat(str(raw_end)[:10]) if raw_end else None
        raw_repeal = payload.get("repeal_date")
        repeal = date.fromisoformat(str(raw_repeal)[:10]) if raw_repeal else None
    except ValueError:
        return False
    return start <= when and (end is None or when < end) and (repeal is None or when < repeal)


def _select_temporal_candidates(
    points: list[Any],
    plan: QueryPlan,
    resolved_current_date: str,
) -> tuple[list[Any], str | None]:
    """Select one proven normative version per document before passage reranking.

    The raw branches remain in the trace for diagnostics, but historical, future, or
    overlapping legislative versions cannot compete with the operative version. Court and
    case material is retained without applying legislative status semantics.
    """

    present_law = _requires_present_law_proof(plan)
    if plan.as_of is None and not present_law:
        return points, None

    non_normative: list[Any] = []
    eligible_by_document: dict[tuple[str, str], list[Any]] = {}
    saw_normative = False
    for point in points:
        payload = getattr(point, "payload", None) or {}
        if not _is_normative_payload(payload):
            non_normative.append(point)
            continue
        saw_normative = True
        lineage_proven = (
            payload.get("version_lineage_complete") is True
            and payload.get("version_ambiguous") is not True
        )
        if plan.as_of is not None:
            effective = _payload_effective_on(payload, plan.as_of)
        else:
            effective = _presently_operative(payload, resolved_current_date)
        if not lineage_proven or not effective:
            continue
        key = (str(payload.get("source") or ""), str(payload.get("document_id") or ""))
        eligible_by_document.setdefault(key, []).append(point)

    selected_normative: list[Any] = []
    ambiguous_version = False
    for document_points in eligible_by_document.values():
        versions = {
            str((getattr(point, "payload", None) or {}).get("version_id") or "")
            for point in document_points
        }
        if "" in versions or len(versions) != 1:
            ambiguous_version = True
            continue
        selected_normative.extend(document_points)

    allowed_ids = {id(point) for point in (*non_normative, *selected_normative)}
    selected = [point for point in points if id(point) in allowed_ids]
    if ambiguous_version or (saw_normative and not selected_normative):
        reason = (
            "temporal_lineage_missing_or_ambiguous"
            if plan.as_of is not None
            else "operative_status_unverified"
        )
        return selected, reason
    return selected, None


@dataclass(frozen=True)
class EvaluationProvenance:
    """Complete immutable identity required for a production-parity quality claim."""

    collection_alias: str
    serving_alias: str
    physical_collection: str
    queried_collection: str
    access_kind: str
    generation_id: str
    points_count: int
    corpus_hash: str
    snapshot_hash: str
    embedding_model: str
    embedding_revision: str | None
    tokenizer_model: str
    tokenizer_revision: str | None
    reranker_model: str
    reranker_revision: str | None
    vector_space_id: str
    chunk_config_id: str
    header_config_id: str
    retrieval_fingerprint_revision: int
    retrieval_fingerprint: str
    frozen_set_hashes: Mapping[str, str]
    dependency_identity: str
    image_identity: str
    git_sha: str
    dirty_patch_hash: str
    execution_mode: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frozen_set_hashes", MappingProxyType(dict(self.frozen_set_hashes))
        )

    def validate_complete(self, *, require_revisions: bool = True) -> None:
        required = {
            "collection_alias": self.collection_alias,
            "serving_alias": self.serving_alias,
            "physical_collection": self.physical_collection,
            "queried_collection": self.queried_collection,
            "access_kind": self.access_kind,
            "generation_id": self.generation_id,
            "corpus_hash": self.corpus_hash,
            "snapshot_hash": self.snapshot_hash,
            "embedding_model": self.embedding_model,
            "tokenizer_model": self.tokenizer_model,
            "reranker_model": self.reranker_model,
            "vector_space_id": self.vector_space_id,
            "chunk_config_id": self.chunk_config_id,
            "header_config_id": self.header_config_id,
            "retrieval_fingerprint": self.retrieval_fingerprint,
            "dependency_identity": self.dependency_identity,
            "image_identity": self.image_identity,
            "git_sha": self.git_sha,
            "dirty_patch_hash": self.dirty_patch_hash,
            "execution_mode": self.execution_mode,
        }
        missing = sorted(name for name, value in required.items() if not value)
        if (
            isinstance(self.points_count, bool)
            or not isinstance(self.points_count, int)
            or self.points_count < 0
        ):
            missing.append("points_count")
        if self.retrieval_fingerprint_revision != RETRIEVAL_FINGERPRINT_REVISION:
            missing.append("retrieval_fingerprint_revision")
        if not self.frozen_set_hashes or any(not k or not v for k, v in self.frozen_set_hashes.items()):
            missing.append("frozen_set_hashes")
        if require_revisions:
            for name in ("embedding_revision", "tokenizer_revision", "reranker_revision"):
                if not getattr(self, name):
                    missing.append(name)
        sha_fields = {
            "corpus_hash": self.corpus_hash,
            "snapshot_hash": self.snapshot_hash,
            "vector_space_id": self.vector_space_id,
            "chunk_config_id": self.chunk_config_id,
            "header_config_id": self.header_config_id,
            "retrieval_fingerprint": self.retrieval_fingerprint,
            "dependency_identity": self.dependency_identity,
            "dirty_patch_hash": self.dirty_patch_hash,
        }
        sha256 = re.compile(r"^[0-9a-f]{64}$")
        for name, value in sha_fields.items():
            if not isinstance(value, str) or not sha256.fullmatch(value):
                missing.append(name)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_identity or ""):
            missing.append("image_identity")
        if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", self.git_sha or ""):
            missing.append("git_sha")
        revision = re.compile(r"^[0-9a-f]{7,64}$")
        for name in ("embedding_revision", "tokenizer_revision", "reranker_revision"):
            value = getattr(self, name)
            if value is not None and not revision.fullmatch(value):
                missing.append(name)
        if any(
            not isinstance(value, str) or not sha256.fullmatch(value)
            for value in self.frozen_set_hashes.values()
        ):
            missing.append("frozen_set_hashes")
        expected_physical = physical_collection_name(self.generation_id)
        if self.physical_collection != expected_physical:
            missing.append("physical_collection")
        if self.serving_alias != SERVING_ALIAS or self.collection_alias != SERVING_ALIAS:
            missing.append("serving_alias")
        if self.access_kind == "direct_physical":
            if self.queried_collection != expected_physical:
                missing.append("queried_collection")
        elif self.access_kind == "serving_alias":
            if self.queried_collection != SERVING_ALIAS:
                missing.append("queried_collection")
        else:
            missing.append("access_kind")
        if missing:
            raise ValueError(f"incomplete evaluation provenance: {', '.join(sorted(set(missing)))}")

    def to_dict(self) -> dict[str, Any]:
        self.validate_complete()
        return {
            "collection_alias": self.collection_alias,
            "serving_alias": self.serving_alias,
            "physical_collection": self.physical_collection,
            "queried_collection": self.queried_collection,
            "access_kind": self.access_kind,
            "generation_id": self.generation_id,
            "points_count": self.points_count,
            "corpus_hash": self.corpus_hash,
            "snapshot_hash": self.snapshot_hash,
            "embedding_model": self.embedding_model,
            "embedding_revision": self.embedding_revision,
            "tokenizer_model": self.tokenizer_model,
            "tokenizer_revision": self.tokenizer_revision,
            "reranker_model": self.reranker_model,
            "reranker_revision": self.reranker_revision,
            "vector_space_id": self.vector_space_id,
            "chunk_config_id": self.chunk_config_id,
            "header_config_id": self.header_config_id,
            "retrieval_fingerprint_revision": self.retrieval_fingerprint_revision,
            "retrieval_fingerprint": self.retrieval_fingerprint,
            "frozen_set_hashes": dict(self.frozen_set_hashes),
            "dependency_identity": self.dependency_identity,
            "image_identity": self.image_identity,
            "git_sha": self.git_sha,
            "dirty_patch_hash": self.dirty_patch_hash,
            "execution_mode": self.execution_mode,
        }


def _is_remote_reranker(reranker: Any) -> bool:
    return getattr(reranker, "device", None) == "remote"


def execute_retrieval(
    cfg: Config,
    client: Any,
    embedder: Any,
    reranker: Any,
    request: RetrievalRequest,
    policy: RetrievalPolicy | None = None,
) -> RetrievalOutcome:
    """Execute today's production path and return a structured, auditable outcome."""

    effective = policy or RetrievalPolicy.from_config(cfg)
    if not effective.dense_enabled:
        raise ValueError("the production executor requires the dense branch")
    if not effective.sparse_enabled and request.route:
        raise ValueError("sparse-disabled policies are explicit ablations, not production")

    active_reranker = reranker if effective.rerank_enabled else None
    kwargs = request.search_kwargs()
    degraded = False
    degraded_reason = None
    stage_timings = {"embed": 0.0, "search": 0.0, "rerank": 0.0}
    started = time.perf_counter()
    try:
        hits = hybrid_search(
            cfg,
            client,
            embedder,
            request.query,
            top_k=request.requested_limit,
            reranker=active_reranker,
            rerank_candidates=effective.rerank_candidates,
            rerank_min_score=effective.rerank_min_score,
            route=request.route,
            max_per_doc=effective.max_per_doc,
            mmr_lambda=effective.mmr_lambda,
            timings_ms=stage_timings,
            **kwargs,
        )
    except Exception as exc:
        if not (
            effective.allow_remote_reranker_degraded
            and active_reranker is not None
            and _is_remote_reranker(active_reranker)
        ):
            raise
        degraded = True
        degraded_reason = f"{type(exc).__name__}: {exc}"
        hits = hybrid_search(
            cfg,
            client,
            embedder,
            request.query,
            top_k=request.requested_limit,
            reranker=None,
            rerank_candidates=effective.rerank_candidates,
            rerank_min_score=effective.rerank_min_score,
            route=request.route,
            max_per_doc=effective.max_per_doc,
            mmr_lambda=effective.mmr_lambda,
            timings_ms=stage_timings,
            **kwargs,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    hits = tuple(hits)
    abstained = effective.abstain_on_empty and not hits
    if request.route:
        route = "dense_only" if detect_language(request.query) == "en" else "hybrid"
    else:
        route = "hybrid_forced"
    return RetrievalOutcome(
        hits=hits,
        timings_ms={**stage_timings, "total": elapsed_ms},
        degraded=degraded,
        degraded_reason=degraded_reason,
        service_abstention=abstained,
        abstention_reason="no_results_after_policy" if abstained else None,
        retrieval_fingerprint=retrieval_fingerprint(cfg),
        effective_route=route,
        generation_id=getattr(cfg, "generation_id", None),
        track=request.track,
    )


def execute_accuracy_retrieval(
    cfg: Config,
    client: Any,
    embedder: Any,
    reranker: Any,
    request: RetrievalRequest,
    *,
    translator: Translator | None = None,
    candidate_depth: int = 80,
    current_date: str | date | None = None,
) -> AccuracyRetrievalOutcome:
    """Execute the strict accuracy profile's candidate pipeline.

    The function always traces an unrestricted semantic branch, adds the original English
    branch and a privately translated Georgian branch when needed, and adds exact/document
    scoped branches when an identifier resolves. Unrestricted hits are diagnostic-only
    when explicit filters exist; only policy-approved branches enter the global rerank.
    Translator/reranker failure, unsupported language, ambiguous exact
    identity, missing temporal lineage, unproved operative status, and incomplete evidence
    all fail closed. Generation freshness is checked by ``LegalAnswerService`` against a
    signed/verified post-build audit; immutable point payloads are not an audit channel.

    This is intentionally separate from :func:`execute_retrieval`: the latter remains the
    backwards-compatible search tool, while this contract is safe for answer generation.
    """

    started = time.perf_counter()
    resolved_current_date = _resolved_current_date(current_date)
    if candidate_depth < 80:
        raise ValueError("the strict accuracy profile requires candidate_depth >= 80")
    as_of = request.temporal_context.as_of
    plan = plan_query(
        request.query,
        language=request.query_language,
        as_of=as_of,
    )
    if not plan.answerable_language:
        return _accuracy_outcome(
            cfg,
            plan,
            abstention_reason=plan.clarification_reason or "unsupported_language",
            timings_ms={"total": (time.perf_counter() - started) * 1000},
            resolved_current_date=resolved_current_date,
        )
    try:
        variants = build_query_variants(plan, translator)
    except Exception as exc:
        if not isinstance(exc, TranslationIntegrityError):
            exc = TranslationIntegrityError(f"translator failed: {type(exc).__name__}: {exc}")
        return _accuracy_outcome(
            cfg,
            plan,
            abstention_reason="translator_degraded",
            degraded_reason=str(exc),
            timings_ms={"total": (time.perf_counter() - started) * 1000},
            resolved_current_date=resolved_current_date,
        )
    translator_version = next(
        (variant.translator_version for variant in variants if variant.translator_version),
        None,
    )
    if reranker is None:
        return _accuracy_outcome(
            cfg,
            plan,
            abstention_reason="reranker_unavailable",
            degraded_reason="strict retrieval requires a reranker",
            translator_version=translator_version,
            timings_ms={"total": (time.perf_counter() - started) * 1000},
            resolved_current_date=resolved_current_date,
        )

    filters = {
        key: value
        for key, value in request.search_kwargs().items()
        if value is not None
    }
    branches: list[RetrievalBranch] = []
    candidate_groups: list[list[Any]] = []
    candidate_ids: tuple[str, ...] = ()
    temporal_candidate_issue: str | None = None

    def run_branch(
        name: str,
        query: str,
        branch_filters: Mapping[str, Any],
        *,
        contributes_candidates: bool = True,
    ) -> list[Any]:
        branch_started = time.perf_counter()
        points = hybrid_search(
            cfg,
            client,
            embedder,
            query,
            top_k=candidate_depth,
            reranker=None,
            route=request.route,
            **dict(branch_filters),
        )
        elapsed = (time.perf_counter() - branch_started) * 1000
        points = list(points)
        branches.append(
            RetrievalBranch(
                name=name,
                query=query,
                filters=branch_filters,
                hit_ids=tuple(_stable_point_id(point) for point in points),
                elapsed_ms=elapsed,
                route=(
                    "dense_only"
                    if request.route and detect_language(query) == "en"
                    else ("hybrid" if request.route else "hybrid_forced")
                ),
            )
        )
        if contributes_candidates:
            candidate_groups.append(points)
        return points

    try:
        for variant in variants:
            run_branch(f"global_{variant.track}", variant.text, filters)
            if filters:
                # User/entity/temporal filters can be wrong or over-constrained. Keep a
                # truly unrestricted semantic branch for trace/agreement diagnostics, but
                # never let it silently override the explicit answering scope.
                run_branch(
                    f"global_unrestricted_{variant.track}",
                    variant.text,
                    {},
                    contributes_candidates=False,
                )

        # Georgian is the canonical corpus language and therefore the global rerank query.
        canonical_query = variants[-1].text
        exact_points: list[Any] = []
        if plan.entities.citation is not None:
            exact_started = time.perf_counter()
            dense = embedder.encode_query(canonical_query).dense
            exact_points = list(
                citation_lookup(
                    client,
                    cfg.collection_name,
                    dense,
                    plan.entities.citation,
                    effective_on=(
                        plan.as_of
                        or (
                            resolved_current_date
                            if _requires_present_law_proof(plan)
                            else None
                        )
                    ),
                    require_in_force=(
                        plan.as_of is None and _requires_present_law_proof(plan)
                    ),
                )
            )
            branches.append(
                RetrievalBranch(
                    name="exact_entity",
                    query=canonical_query,
                    filters=plan.entities.citation.filters,
                    hit_ids=tuple(_stable_point_id(point) for point in exact_points),
                    elapsed_ms=(time.perf_counter() - exact_started) * 1000,
                    route="exact_entity_dense",
                )
            )
            candidate_groups.insert(0, exact_points)

            identities: list[tuple[str, str]] = []
            for point in exact_points:
                payload = point.payload or {}
                identity = (payload.get("source"), payload.get("document_id"))
                if all(identity):
                    if identity not in identities:
                        identities.append(identity)
            for source, document_id in identities[:3]:
                scoped = {
                    key: value
                    for key, value in filters.items()
                    if key not in {
                        "source", "document_id", "document_number", "registration_code",
                        "article_id", "clause_id", "contains",
                    }
                }
                scoped.update({"source": source, "document_id": document_id})
                if plan.entities.article_id:
                    scoped["article_id"] = plan.entities.article_id
                    branch_name = f"document_article:{source}:{document_id}"
                    run_branch(branch_name, canonical_query, scoped)
                else:
                    branch_name = f"document:{source}:{document_id}"
                    run_branch(branch_name, canonical_query, scoped)

        candidates = _union_candidates(*candidate_groups)
        if exact_points and not any(
            bool((point.payload or {}).get("identity_ambiguous"))
            for point in exact_points
        ):
            resolved_identities = {
                (
                    (point.payload or {}).get("source"),
                    (point.payload or {}).get("document_id"),
                    (point.payload or {}).get("version_id"),
                )
                for point in exact_points
            }
            resolved_identities.discard((None, None, None))
            if len(resolved_identities) == 1:
                # Once exact identity and effective version are proven, unrestricted
                # semantic results remain trace diagnostics only. They cannot override the
                # cited authority merely because an unrelated passage reranks higher.
                resolved_identity = next(iter(resolved_identities))
                candidates = [
                    point for point in candidates
                    if (
                        (point.payload or {}).get("source"),
                        (point.payload or {}).get("document_id"),
                        (point.payload or {}).get("version_id"),
                    ) == resolved_identity
                ]
        # Incomplete/quarantined records may be discoverable for diagnostics, but never
        # enter an answer-generating evidence set.
        candidates = [point for point in candidates if _admissible_answer_candidate(point)]
        candidates, temporal_candidate_issue = _select_temporal_candidates(
            candidates, plan, resolved_current_date
        )
        # This is the exact global pool presented to the first reranker.  Capture it only
        # after all identity, authority, admissibility, and temporal policy has run.  Later
        # diagnostic/cross-reference branches deliberately cannot rewrite this telemetry.
        candidate_ids = tuple(_stable_point_id(point) for point in candidates)
        rerank_started = time.perf_counter()
        ranked = rerank_points(
            reranker,
            canonical_query,
            candidates,
            top_k=request.requested_limit,
            # A reranker score is not answer confidence.  Selective-risk calibration and
            # deterministic validators, not a raw sigmoid threshold, decide abstention.
            min_score=None,
            enrich_context=getattr(cfg, "rerank_context_enriched", False),
        ) if candidates else []
        rerank_ms = (time.perf_counter() - rerank_started) * 1000

        # Deterministically expand explicit intra-document article references from the
        # winning passages, then perform the planned second passage rerank. Generated
        # summaries/context are never inserted as evidence.
        expanded_groups: list[list[Any]] = []
        expanded_seen: set[tuple[str, str, str]] = set()
        for point in ranked[:3]:
            payload = point.payload or {}
            source = payload.get("source")
            document_id = payload.get("document_id")
            if not source or not document_id:
                continue
            references = extract_article_references(
                payload.get("text") or "",
                own_article_id=(
                    str(payload.get("article_id"))
                    if payload.get("article_id") not in (None, "")
                    else None
                ),
            )
            for article_id in references:
                identity = (str(source), str(document_id), article_id)
                if identity in expanded_seen:
                    continue
                expanded_seen.add(identity)
                scoped = {
                    key: value
                    for key, value in filters.items()
                    if key not in {
                        "source", "document_id", "document_number", "registration_code",
                        "article_id", "clause_id", "contains",
                    }
                }
                scoped.update(
                    {
                        "source": source,
                        "document_id": document_id,
                        "article_id": article_id,
                    }
                )
                structural = run_branch(
                    f"cross_reference:{source}:{document_id}:{article_id}",
                    canonical_query,
                    scoped,
                )
                if structural:
                    expanded_groups.append(structural)
        cross_reference_rerank_ms = 0.0
        if expanded_groups:
            second_started = time.perf_counter()
            second_candidates = [
                point
                for point in _union_candidates(ranked, *expanded_groups)
                if _admissible_answer_candidate(point)
            ]
            second_candidates, second_temporal_issue = _select_temporal_candidates(
                second_candidates, plan, resolved_current_date
            )
            temporal_candidate_issue = temporal_candidate_issue or second_temporal_issue
            ranked = rerank_points(
                reranker,
                canonical_query,
                second_candidates,
                top_k=request.requested_limit,
                min_score=None,
                enrich_context=getattr(cfg, "rerank_context_enriched", False),
            )
            cross_reference_rerank_ms = (time.perf_counter() - second_started) * 1000
            rerank_ms += cross_reference_rerank_ms
    except Exception as exc:
        return _accuracy_outcome(
            cfg,
            plan,
            branches=branches,
            candidate_ids=candidate_ids,
            abstention_reason="retrieval_degraded",
            degraded_reason=f"{type(exc).__name__}: {exc}",
            translator_version=translator_version,
            timings_ms={"total": (time.perf_counter() - started) * 1000},
            resolved_current_date=resolved_current_date,
        )

    identity_ambiguous = any(
        bool((point.payload or {}).get("identity_ambiguous")) for point in exact_points
    )
    abstention_reason: str | None = None
    if identity_ambiguous:
        abstention_reason = "ambiguous_entity"
    elif plan.entities.citation is not None and not exact_points:
        abstention_reason = "unresolved_entity"
    elif temporal_candidate_issue is not None:
        abstention_reason = temporal_candidate_issue
    elif not ranked:
        abstention_reason = "insufficient_evidence"
    elif plan.as_of is not None:
        # Historical assertions require a real version lineage, not publication dates.
        uncertain = [
            point
            for point in ranked
            if not _payload_effective_on(point.payload or {}, plan.as_of)
            or (point.payload or {}).get("version_lineage_complete") is not True
            or (point.payload or {}).get("version_ambiguous") is True
        ]
        if uncertain:
            abstention_reason = "temporal_lineage_missing_or_ambiguous"
    elif _requires_present_law_proof(plan):
        if any(
            not _presently_operative(point.payload or {}, resolved_current_date)
            for point in ranked
        ):
            abstention_reason = "operative_status_unverified"

    total_ms = (time.perf_counter() - started) * 1000
    return _accuracy_outcome(
        cfg,
        plan,
        hits=ranked,
        branches=branches,
        candidate_ids=candidate_ids,
        timings_ms={
            "candidate_branches": sum(branch.elapsed_ms for branch in branches),
            "rerank": rerank_ms,
            "cross_reference_rerank": cross_reference_rerank_ms,
            "total": total_ms,
        },
        abstention_reason=abstention_reason,
        identity_ambiguous=identity_ambiguous,
        translator_version=translator_version,
        resolved_current_date=resolved_current_date,
    )
