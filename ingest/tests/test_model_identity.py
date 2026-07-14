"""Fail-closed production identity and fingerprint regression coverage."""

from __future__ import annotations

import dataclasses

import pytest

from ingest.config import (
    ConfigurationError,
    load_config,
    retrieval_fingerprint,
    retrieval_fingerprint_sha256,
)


_REV_A = "a" * 40
_REV_B = "b" * 40
_REV_C = "c" * 40


def _clear_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "EMBED_REVISION",
        "TOKENIZER_MODEL",
        "TOKENIZER_REVISION",
        "RERANK_REVISION",
        "GENERATION_ID",
        "GENERATION_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PRODUCTION_MODE", "false")


def test_unset_identity_preserves_development_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_identity(monkeypatch)
    cfg = load_config()
    assert cfg.embedding_revision is None
    assert cfg.tokenizer_model == cfg.embed_model
    assert cfg.tokenizer_revision is None
    assert cfg.reranker_revision is None

    explicit_legacy_identity = dataclasses.replace(
        cfg,
        embedding_revision=None,
        tokenizer_model=cfg.embed_model,
        tokenizer_revision=None,
        reranker_revision=None,
        generation_id=None,
        generation_dir=None,
        production_mode=False,
    )
    assert retrieval_fingerprint(cfg) == retrieval_fingerprint(explicit_legacy_identity)
    assert len(retrieval_fingerprint_sha256(cfg)) == 64
    assert retrieval_fingerprint_sha256(cfg).startswith(retrieval_fingerprint(cfg))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("embedding_revision", _REV_A),
        ("tokenizer_model", "immutable/local-tokenizer"),
        ("tokenizer_revision", _REV_B),
        ("reranker_revision", _REV_C),
    ),
)
def test_explicit_model_identity_changes_fingerprint(field: str, value: str) -> None:
    cfg = dataclasses.replace(load_config(), rerank_enabled=True)
    assert retrieval_fingerprint(dataclasses.replace(cfg, **{field: value})) != (
        retrieval_fingerprint(cfg)
    )


def test_disabled_reranker_revision_does_not_change_fingerprint() -> None:
    cfg = dataclasses.replace(load_config(), rerank_enabled=False)
    changed = dataclasses.replace(cfg, reranker_revision=_REV_C)
    assert retrieval_fingerprint(changed) == retrieval_fingerprint(cfg)


@pytest.mark.parametrize(
    ("missing", "expected"),
    (
        ("GENERATION_ID", "GENERATION_ID"),
        ("GENERATION_DIR", "GENERATION_DIR"),
        ("EMBED_REVISION", "EMBED_REVISION"),
        ("TOKENIZER_REVISION", "TOKENIZER_REVISION"),
        ("RERANK_REVISION", "RERANK_REVISION"),
    ),
)
def test_production_rejects_missing_identity(
    monkeypatch: pytest.MonkeyPatch, missing: str, expected: str
) -> None:
    values = {
        "GENERATION_ID": "gen_20260713_verified",
        "GENERATION_DIR": "/verified/generation",
        "EMBED_REVISION": _REV_A,
        "TOKENIZER_REVISION": _REV_B,
        "RERANK_REVISION": _REV_C,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.delenv(missing, raising=False)

    with pytest.raises(ConfigurationError, match=expected):
        load_config()


def test_production_accepts_complete_immutable_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENERATION_ID", "gen_20260713_verified")
    monkeypatch.setenv("GENERATION_DIR", "/verified/generation")
    monkeypatch.setenv("EMBED_REVISION", _REV_A)
    monkeypatch.setenv("TOKENIZER_REVISION", _REV_B)
    monkeypatch.setenv("RERANK_REVISION", _REV_C)
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("PRODUCTION_MODE", "true")

    cfg = load_config()
    assert cfg.production_mode is True
    assert cfg.generation_id == "gen_20260713_verified"


def test_production_requires_disabled_reranker_revision_for_generation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENERATION_ID", "gen_20260713_verified")
    monkeypatch.setenv("GENERATION_DIR", "/verified/generation")
    monkeypatch.setenv("EMBED_REVISION", _REV_A)
    monkeypatch.setenv("TOKENIZER_REVISION", _REV_B)
    monkeypatch.delenv("RERANK_REVISION", raising=False)
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("PRODUCTION_MODE", "true")

    with pytest.raises(ConfigurationError, match="RERANK_REVISION"):
        load_config()


@pytest.mark.parametrize("field", ("EMBED_MODEL", "TOKENIZER_MODEL", "RERANK_MODEL"))
def test_production_rejects_empty_model_identity(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    monkeypatch.setenv("GENERATION_ID", "gen_20260713_verified")
    monkeypatch.setenv("GENERATION_DIR", "/verified/generation")
    monkeypatch.setenv("EMBED_REVISION", _REV_A)
    monkeypatch.setenv("TOKENIZER_REVISION", _REV_B)
    monkeypatch.setenv("RERANK_REVISION", _REV_C)
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv(field, "")

    with pytest.raises(ConfigurationError, match=field):
        load_config()


@pytest.mark.parametrize("revision", ("main", "LATEST", "sha256:not-a-model-commit"))
def test_production_rejects_mutable_revision(
    monkeypatch: pytest.MonkeyPatch, revision: str
) -> None:
    monkeypatch.setenv("GENERATION_ID", "gen_20260713_verified")
    monkeypatch.setenv("GENERATION_DIR", "/verified/generation")
    monkeypatch.setenv("EMBED_REVISION", revision)
    monkeypatch.setenv("TOKENIZER_REVISION", _REV_B)
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("PRODUCTION_MODE", "true")

    with pytest.raises(ConfigurationError, match="immutable lowercase hexadecimal"):
        load_config()
