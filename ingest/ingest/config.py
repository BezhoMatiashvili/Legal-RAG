"""Env-driven configuration. Reads a .env file if present (python-dotenv)."""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ingest/ingest/config.py -> parents[2] == repo root (sibling of scraper/ and artifacts/).
REPO_ROOT = Path(__file__).resolve().parents[2]

_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")
_GENERATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{7,127}$")

# Retrieval identity revision 2 deliberately separates behavior from storage routing.
# Collection, alias, and generation names are recorded in provenance and cache identity,
# but they do not change the model/chunk/search behavior fingerprint itself.
RETRIEVAL_FINGERPRINT_REVISION = 2


class ConfigurationError(ValueError):
    """Configuration is ambiguous or unsafe for the selected runtime mode."""


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _str_opt(name: str) -> str | None:
    raw = (os.getenv(name) or "").strip()
    return raw or None


def _float_opt(name: str, default: float | None) -> float | None:
    """Optional float: unset → default; empty string or 'none'/'off' → None (no gate)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in ("", "none", "off"):
        return None
    return float(raw)


def _device_opt(name: str) -> str | None:
    """A device string (``cpu``/``cuda``/``cuda:0``/``mps``) or None.

    Defensive against python-dotenv leaking an inline ``.env`` comment as the value
    (e.g. ``EMBED_DEVICE=   # blank = auto``): a real device has no whitespace or ``#``,
    so anything else is treated as unset.
    """
    raw = (os.getenv(name) or "").strip()
    if not raw or raw.startswith("#") or any(c.isspace() for c in raw):
        return None
    return raw


def _route_opt(name: str) -> str | None:
    """Citation-route mode: ``ids`` | ``full``; anything else (unset/empty/off) → None."""
    raw = (os.getenv(name) or "").strip().lower()
    return raw if raw in ("ids", "full") else None


@dataclass(frozen=True)
class Config:
    qdrant_url: str
    qdrant_api_key: str | None
    collection_name: str
    # 'local' = embed/search/rerank in-process (the default); 'remote' = the MCP tools RPC
    # to the RunPod serverless worker and load no models / need no local Qdrant.
    search_backend: str
    runpod_endpoint_id: str | None
    runpod_api_key: str | None
    runpod_api_timeout: int
    embed_model: str
    embedding_revision: str | None
    tokenizer_model: str
    tokenizer_revision: str | None
    dense_dim: int
    embed_device: str | None
    embed_use_fp16: bool
    embed_batch_size: int
    rerank_enabled: bool
    rerank_model: str
    reranker_revision: str | None
    rerank_candidates: int
    rerank_min_score: float | None
    rerank_device: str | None
    rerank_use_fp16: bool
    rerank_remote_url: str | None
    # I7: 'torch' (default) or 'onnx' (int8 export, scripts/export_onnx_reranker.py).
    rerank_backend: str
    onnx_rerank_path: Path
    # I8: enrich cross-encoder input with title/type/status/number/heading (default off =
    # raw chunk body only, current behavior).
    rerank_context_enriched: bool
    # I8: query+passage truncation for the cross-encoder (model supports up to 8192; 512
    # was sized to chunk length, not model capacity). Mirrors RERANK_MAX_LENGTH already
    # read by scripts/runpod_rerank_server.py.
    rerank_max_length: int
    chunk_tokens: int
    chunk_overlap: int
    chunk_min_tokens: int
    # I6: v2 embed headers (№/date/status/consolidation in the embedded prefix).
    # Changing this INVALIDATES existing vectors — only flip together with a re-embed.
    embed_header_v2: bool
    # I1: citation exact-match routing — None = off, "ids" | "full".
    citation_route: str | None
    artifacts_root: Path
    state_dir: Path
    query_log_enabled: bool
    query_log_path: Path
    # Result cache: identical repeat queries return the cached formatted result instead of
    # re-running retrieval (the RunPod RPC in remote mode / CPU rerank locally). In-memory,
    # per-process. Quality-neutral (same bytes, just faster) → deliberately NOT folded into
    # retrieval_fingerprint.
    result_cache_enabled: bool
    result_cache_ttl: int
    result_cache_size: int
    # A production process must be tied to an immutable corpus and model identity.
    # These fields remain optional in development so existing local fingerprints and
    # offline workflows retain their current behavior.
    generation_id: str | None
    generation_dir: Path | None
    production_mode: bool
    # Internal worker mode: handler.py has already verified and bound the atomic publish
    # manifest plus live Qdrant generation before importing the MCP module. load_config()
    # never enables this; only the in-process handler installer can construct it.
    verified_worker_binding: bool


def _validate_revision(value: str | None, *, field: str) -> None:
    if value is None:
        raise ConfigurationError(f"{field} is required when PRODUCTION_MODE=true")
    if not _IMMUTABLE_REVISION_RE.fullmatch(value):
        raise ConfigurationError(
            f"{field} must be an immutable lowercase hexadecimal revision "
            "when PRODUCTION_MODE=true"
        )


def _validate_model_name(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty model identity")


def validate_production_config(cfg: Config) -> None:
    """Reject ambiguous corpus/model identities before any heavyweight model import."""
    if not cfg.production_mode:
        return
    if cfg.generation_id is None:
        raise ConfigurationError("GENERATION_ID is required when PRODUCTION_MODE=true")
    if cfg.generation_dir is None and not cfg.verified_worker_binding:
        raise ConfigurationError("GENERATION_DIR is required when PRODUCTION_MODE=true")
    if cfg.verified_worker_binding and cfg.search_backend != "local":
        raise ConfigurationError(
            "verified worker binding requires SEARCH_BACKEND=local"
        )
    if (
        not _GENERATION_ID_RE.fullmatch(cfg.generation_id)
        or cfg.generation_id in {"legacy", "snapshot_v1"}
        or cfg.generation_id.startswith("v1")
    ):
        raise ConfigurationError(
            "GENERATION_ID must be a non-legacy 8-128 character lowercase generation ID"
        )
    _validate_model_name(cfg.embed_model, field="EMBED_MODEL")
    _validate_model_name(cfg.tokenizer_model, field="TOKENIZER_MODEL")
    _validate_model_name(cfg.rerank_model, field="RERANK_MODEL")
    _validate_revision(cfg.embedding_revision, field="EMBED_REVISION")
    _validate_revision(cfg.tokenizer_revision, field="TOKENIZER_REVISION")
    _validate_revision(cfg.reranker_revision, field="RERANK_REVISION")


def load_config() -> Config:
    artifacts = os.getenv("ARTIFACTS_ROOT")
    artifacts_root = Path(artifacts) if artifacts else REPO_ROOT / "artifacts"
    state_dir = REPO_ROOT / "ingest" / ".state"
    ql_path = os.getenv("QUERY_LOG_PATH")
    embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
    cfg = Config(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        collection_name=os.getenv("COLLECTION_NAME", "georgian_legal"),
        search_backend=(os.getenv("SEARCH_BACKEND") or "local").strip().lower() or "local",
        runpod_endpoint_id=(os.getenv("RUNPOD_ENDPOINT_ID") or "").strip() or None,
        runpod_api_key=os.getenv("RUNPOD_API_KEY") or None,
        runpod_api_timeout=_int("RUNPOD_API_TIMEOUT", 240),
        embed_model=embed_model,
        embedding_revision=_str_opt("EMBED_REVISION"),
        tokenizer_model=os.getenv("TOKENIZER_MODEL", embed_model),
        tokenizer_revision=_str_opt("TOKENIZER_REVISION"),
        dense_dim=_int("DENSE_DIM", 1024),
        embed_device=_device_opt("EMBED_DEVICE"),
        embed_use_fp16=_bool("EMBED_USE_FP16", False),
        embed_batch_size=_int("EMBED_BATCH_SIZE", 8),
        rerank_enabled=_bool("RERANK_ENABLED", True),
        rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        reranker_revision=_str_opt("RERANK_REVISION"),
        rerank_candidates=_int("RERANK_CANDIDATES", 80),
        rerank_min_score=_float_opt("RERANK_MIN_SCORE", 0.3),
        rerank_device=_device_opt("RERANK_DEVICE") or _device_opt("EMBED_DEVICE"),
        rerank_use_fp16=_bool("RERANK_USE_FP16", False),
        rerank_remote_url=(os.getenv("RERANK_REMOTE_URL") or "").strip() or None,
        rerank_backend=((os.getenv("RERANK_BACKEND") or "").strip().lower()
                        if (os.getenv("RERANK_BACKEND") or "").strip().lower() in ("torch", "onnx")
                        else "torch"),
        onnx_rerank_path=Path(os.getenv("ONNX_RERANK_PATH")
                              or state_dir / "onnx" / "bge-reranker-v2-m3-int8.onnx"),
        rerank_context_enriched=_bool("RERANK_CONTEXT_ENRICHED", False),
        rerank_max_length=_int("RERANK_MAX_LENGTH", 512),
        chunk_tokens=_int("CHUNK_TOKENS", 512),
        chunk_overlap=_int("CHUNK_OVERLAP", 80),
        chunk_min_tokens=_int("CHUNK_MIN_TOKENS", 64),
        embed_header_v2=_bool("EMBED_HEADER_V2", False),
        citation_route=_route_opt("CITATION_ROUTE"),
        artifacts_root=artifacts_root.resolve(),
        state_dir=state_dir,
        query_log_enabled=_bool("QUERY_LOG_ENABLED", True),
        query_log_path=Path(ql_path) if ql_path else state_dir / "queries.jsonl",
        # Opt-in only: the corpus is mutable and a cache has no cross-process index revision
        # signal in local mode. Accuracy/freshness wins over repeat-query latency by default.
        result_cache_enabled=_bool("RESULT_CACHE_ENABLED", False),
        result_cache_ttl=_int("RESULT_CACHE_TTL", 1800),
        result_cache_size=_int("RESULT_CACHE_SIZE", 512),
        generation_id=_str_opt("GENERATION_ID"),
        generation_dir=(
            Path(value).expanduser().resolve()
            if (value := _str_opt("GENERATION_DIR")) is not None
            else None
        ),
        production_mode=_bool("PRODUCTION_MODE", False),
        verified_worker_binding=False,
    )
    validate_production_config(cfg)
    return cfg


def _retrieval_fingerprint_material(cfg: Config) -> dict[str, object]:
    material = {
        "retrieval_fingerprint_revision": RETRIEVAL_FINGERPRINT_REVISION,
        "embed_model": cfg.embed_model,
        "dense_dim": cfg.dense_dim,
        "rerank_enabled": cfg.rerank_enabled,
        "rerank_model": cfg.rerank_model if cfg.rerank_enabled else None,
        "rerank_candidates": cfg.rerank_candidates,
        "rerank_min_score": cfg.rerank_min_score,
        "chunk_tokens": cfg.chunk_tokens,
        "chunk_overlap": cfg.chunk_overlap,
        "chunk_min_tokens": cfg.chunk_min_tokens,
    }
    if cfg.embedding_revision:
        material["embedding_revision"] = cfg.embedding_revision
    if cfg.tokenizer_model != cfg.embed_model:
        material["tokenizer_model"] = cfg.tokenizer_model
    if cfg.tokenizer_revision:
        material["tokenizer_revision"] = cfg.tokenizer_revision
    if cfg.rerank_enabled and cfg.reranker_revision:
        material["reranker_revision"] = cfg.reranker_revision
    # Conditional so the fingerprint is byte-stable while the knob is at its default (G5).
    if cfg.rerank_enabled and cfg.rerank_backend != "torch":
        material["rerank_backend"] = cfg.rerank_backend
    if cfg.rerank_enabled and cfg.rerank_context_enriched:  # I8, default-off (G5)
        material["rerank_context_enriched"] = True
    if cfg.rerank_enabled and cfg.rerank_max_length != 512:  # I8, default-stable (G5)
        material["rerank_max_length"] = cfg.rerank_max_length
    if cfg.citation_route:  # conditional: fingerprint byte-stable while the knob is off (G5)
        material["citation_route"] = cfg.citation_route
    if cfg.embed_header_v2:  # changes corpus vectors; default-off hash stays byte-stable
        material["embed_header_v2"] = True
    return material


def retrieval_fingerprint_sha256(cfg: Config) -> str:
    """Full cryptographic retrieval identity used by immutable generation artifacts."""
    material = _retrieval_fingerprint_material(cfg)
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def retrieval_fingerprint(cfg: Config) -> str:
    """Stable 16-hex display digest of the knobs that determine search results.

    Fingerprint revision 2 excludes storage routing and intentionally differs from legacy
    digests. Immutable generation manifests and point payloads use the full digest from
    :func:`retrieval_fingerprint_sha256` and record the revision separately.
    """
    return retrieval_fingerprint_sha256(cfg)[:16]
