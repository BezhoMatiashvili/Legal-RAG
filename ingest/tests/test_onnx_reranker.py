"""Unit tests for the int8 ONNX reranker knob (improvement I7) — no model load."""

import dataclasses
from pathlib import Path

import pytest

from ingest.config import load_config, retrieval_fingerprint
from ingest.rerank import ONNXBGEReranker, make_reranker


def _cfg(**over):
    return dataclasses.replace(load_config(), **over)


def test_config_defaults_to_torch_backend(monkeypatch):
    monkeypatch.delenv("RERANK_BACKEND", raising=False)
    assert load_config().rerank_backend == "torch"


def test_config_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("RERANK_BACKEND", "tensorrt")
    assert load_config().rerank_backend == "torch"
    monkeypatch.setenv("RERANK_BACKEND", "ONNX")
    assert load_config().rerank_backend == "onnx"


def test_fingerprint_unchanged_for_torch_changes_for_onnx():
    torch_cfg = _cfg(rerank_backend="torch", rerank_enabled=True)
    onnx_cfg = _cfg(rerank_backend="onnx", rerank_enabled=True)
    # torch is the fingerprint-neutral default; onnx must fork the fingerprint.
    assert retrieval_fingerprint(torch_cfg) == retrieval_fingerprint(
        _cfg(rerank_backend="torch", rerank_enabled=True))
    assert retrieval_fingerprint(onnx_cfg) != retrieval_fingerprint(torch_cfg)


def test_onnx_reranker_fails_loud_without_export():
    cfg = _cfg(rerank_backend="onnx", onnx_rerank_path=Path("/nonexistent/model.onnx"))
    with pytest.raises(FileNotFoundError, match="export_onnx_reranker"):
        ONNXBGEReranker(cfg)


def test_onnx_missing_export_is_checked_before_optional_imports(monkeypatch):
    import builtins

    cfg = _cfg(rerank_backend="onnx", onnx_rerank_path=Path("/nonexistent/model.onnx"))
    real_import = builtins.__import__
    imported_optional = []

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"numpy", "onnxruntime", "transformers"}:
            imported_optional.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(FileNotFoundError, match="export_onnx_reranker"):
        ONNXBGEReranker(cfg)
    assert imported_optional == []


def test_make_reranker_dispatches_on_backend():
    cfg = _cfg(rerank_backend="onnx", onnx_rerank_path=Path("/nonexistent/model.onnx"))
    with pytest.raises(FileNotFoundError):
        make_reranker(cfg)  # onnx branch reached (fails on the missing file, not torch load)
