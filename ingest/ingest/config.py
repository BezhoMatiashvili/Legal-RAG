"""Env-driven configuration. Reads a .env file if present (python-dotenv)."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ingest/ingest/config.py -> parents[2] == repo root (sibling of scraper/ and artifacts/).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


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
    dense_dim: int
    embed_device: str | None
    embed_use_fp16: bool
    embed_batch_size: int
    rerank_enabled: bool
    rerank_model: str
    rerank_candidates: int
    rerank_min_score: float | None
    rerank_device: str | None
    rerank_use_fp16: bool
    rerank_remote_url: str | None
    # I7: 'torch' (default) or 'onnx' (int8 export, scripts/export_onnx_reranker.py).
    rerank_backend: str
    onnx_rerank_path: Path
    chunk_tokens: int
    chunk_overlap: int
    chunk_min_tokens: int
    # I6: v2 embed headers (№/date/status/consolidation in the embedded prefix).
    # Changing this INVALIDATES existing vectors — only flip together with a re-embed.
    embed_header_v2: bool
    artifacts_root: Path
    state_dir: Path
    query_log_enabled: bool
    query_log_path: Path


def load_config() -> Config:
    artifacts = os.getenv("ARTIFACTS_ROOT")
    artifacts_root = Path(artifacts) if artifacts else REPO_ROOT / "artifacts"
    state_dir = REPO_ROOT / "ingest" / ".state"
    ql_path = os.getenv("QUERY_LOG_PATH")
    return Config(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        collection_name=os.getenv("COLLECTION_NAME", "georgian_legal"),
        search_backend=(os.getenv("SEARCH_BACKEND") or "local").strip().lower() or "local",
        runpod_endpoint_id=(os.getenv("RUNPOD_ENDPOINT_ID") or "").strip() or None,
        runpod_api_key=os.getenv("RUNPOD_API_KEY") or None,
        runpod_api_timeout=_int("RUNPOD_API_TIMEOUT", 240),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
        dense_dim=_int("DENSE_DIM", 1024),
        embed_device=_device_opt("EMBED_DEVICE"),
        embed_use_fp16=_bool("EMBED_USE_FP16", False),
        embed_batch_size=_int("EMBED_BATCH_SIZE", 8),
        rerank_enabled=_bool("RERANK_ENABLED", True),
        rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
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
        chunk_tokens=_int("CHUNK_TOKENS", 512),
        chunk_overlap=_int("CHUNK_OVERLAP", 80),
        chunk_min_tokens=_int("CHUNK_MIN_TOKENS", 64),
        embed_header_v2=_bool("EMBED_HEADER_V2", False),
        artifacts_root=artifacts_root.resolve(),
        state_dir=state_dir,
        query_log_enabled=_bool("QUERY_LOG_ENABLED", True),
        query_log_path=Path(ql_path) if ql_path else state_dir / "queries.jsonl",
    )


def retrieval_fingerprint(cfg: Config) -> str:
    """Stable 16-hex digest of the knobs that determine what a search returns.

    Stamped into MCP responses and the query log so any answer is traceable to the exact
    index + retrieval config that produced it. Lives here (not in ``eval.explog``) so the
    ``ingest`` package never imports ``eval`` — the layering only goes eval → ingest.
    """
    material = {
        "collection_name": cfg.collection_name,
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
    # Conditional so the fingerprint is byte-stable while the knob is at its default (G5).
    if cfg.rerank_enabled and cfg.rerank_backend != "torch":
        material["rerank_backend"] = cfg.rerank_backend
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
