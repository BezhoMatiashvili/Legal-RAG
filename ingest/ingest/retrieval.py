"""Shared production-retrieval contracts and compatibility executor.

The MCP tool, serverless worker, and production-parity evaluation must construct the
same request and execute the same policy.  This module deliberately starts as a thin
adapter over :func:`ingest.search.hybrid_search`; keeping that function as the single
ranking implementation makes the initial refactor behaviour-neutral.
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .config import Config, retrieval_fingerprint
from .search import detect_language, hybrid_search


@dataclass(frozen=True)
class TemporalContext:
    """Explicit temporal bounds carried with a retrieval request."""

    date_from: str | None = None
    date_to: str | None = None
    as_of: str | None = None

    def as_filters(self) -> dict[str, str]:
        if self.as_of and self.date_to and self.as_of != self.date_to:
            raise ValueError("temporal context cannot set conflicting as_of and date_to")
        out: dict[str, str] = {}
        if self.date_from:
            out["date_from"] = self.date_from
        if self.as_of or self.date_to:
            out["date_to"] = self.as_of or self.date_to or ""
        return out


@dataclass(frozen=True)
class RetrievalRequest:
    """Provider-neutral input to the production retrieval path."""

    query: str
    requested_limit: int = 10
    route: bool = True
    filters: Mapping[str, Any] = field(default_factory=dict)
    language: str | None = None
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
class EvaluationProvenance:
    """Complete immutable identity required for a production-parity quality claim."""

    collection_alias: str
    physical_collection: str
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
            "physical_collection": self.physical_collection,
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
        if self.physical_collection != f"georgian_legal__gen_{self.generation_id}":
            missing.append("physical_collection")
        if missing:
            raise ValueError(f"incomplete evaluation provenance: {', '.join(sorted(set(missing)))}")

    def to_dict(self) -> dict[str, Any]:
        self.validate_complete()
        return {
            "collection_alias": self.collection_alias,
            "physical_collection": self.physical_collection,
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
