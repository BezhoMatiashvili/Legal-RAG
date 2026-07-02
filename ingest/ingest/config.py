"""Env-driven configuration. Reads a .env file if present (python-dotenv)."""

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


@dataclass(frozen=True)
class Config:
    qdrant_url: str
    qdrant_api_key: str | None
    collection_name: str
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
    chunk_tokens: int
    chunk_overlap: int
    chunk_min_tokens: int
    artifacts_root: Path
    state_dir: Path


def load_config() -> Config:
    artifacts = os.getenv("ARTIFACTS_ROOT")
    artifacts_root = Path(artifacts) if artifacts else REPO_ROOT / "artifacts"
    return Config(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
        collection_name=os.getenv("COLLECTION_NAME", "georgian_legal"),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
        dense_dim=_int("DENSE_DIM", 1024),
        embed_device=os.getenv("EMBED_DEVICE") or None,
        embed_use_fp16=_bool("EMBED_USE_FP16", False),
        embed_batch_size=_int("EMBED_BATCH_SIZE", 8),
        rerank_enabled=_bool("RERANK_ENABLED", True),
        rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        rerank_candidates=_int("RERANK_CANDIDATES", 80),
        rerank_min_score=_float_opt("RERANK_MIN_SCORE", 0.3),
        rerank_device=os.getenv("RERANK_DEVICE") or os.getenv("EMBED_DEVICE") or None,
        rerank_use_fp16=_bool("RERANK_USE_FP16", False),
        chunk_tokens=_int("CHUNK_TOKENS", 512),
        chunk_overlap=_int("CHUNK_OVERLAP", 80),
        chunk_min_tokens=_int("CHUNK_MIN_TOKENS", 64),
        artifacts_root=artifacts_root.resolve(),
        state_dir=REPO_ROOT / "ingest" / ".state",
    )
