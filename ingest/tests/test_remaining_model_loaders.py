"""Remaining evaluation and operational model loaders honor configured revisions."""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from eval import dump_judge_batch, eval_answer_quality, evaluate


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
REVISION = "a" * 40


def _load_script(name: str):
    module_name = f"_test_{name}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "module", (evaluate, eval_answer_quality, dump_judge_batch)
)
def test_evaluation_token_counters_forward_model_and_revision(
    monkeypatch: pytest.MonkeyPatch, module
) -> None:
    calls = []
    counter = object()

    def make_token_counter(model_name, revision=None):
        calls.append((model_name, revision))
        return counter

    monkeypatch.setitem(
        sys.modules,
        "ingest.embedding",
        types.SimpleNamespace(make_token_counter=make_token_counter),
    )

    assert module._token_counter("bge", "tokenizer/repo", REVISION) is counter
    assert module._token_counter("bge", "tokenizer/repo", None) is counter
    assert calls == [("tokenizer/repo", REVISION), ("tokenizer/repo", None)]


def _fake_transformers(calls):
    class TokenizerLoader:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("tokenizer", model_name, kwargs))
            return object()

    class FakeModel:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    class ModelLoader:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("model", model_name, kwargs))
            return FakeModel()

    return types.SimpleNamespace(
        AutoTokenizer=TokenizerLoader,
        AutoModelForSequenceClassification=ModelLoader,
    )


def test_onnx_export_forwards_revision_only_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("export_onnx_reranker")
    calls = []
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls))

    module._load_reranker("reranker/repo", None)
    module._load_reranker("reranker/repo", REVISION)

    assert calls == [
        ("tokenizer", "reranker/repo", {}),
        ("model", "reranker/repo", {}),
        ("tokenizer", "reranker/repo", {"revision": REVISION}),
        ("model", "reranker/repo", {"revision": REVISION}),
    ]


def test_runpod_server_reads_revision_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RERANK_REVISION", REVISION)
    monkeypatch.setenv("PRODUCTION_MODE", "true")

    module = _load_script("runpod_rerank_server")

    assert module.RERANK_REVISION == REVISION
    assert module.PRODUCTION_MODE is True


@pytest.mark.parametrize("revision", (None, "main", "A" * 40))
def test_runpod_production_rejects_missing_or_mutable_revision_before_imports(
    monkeypatch: pytest.MonkeyPatch, revision: str | None
) -> None:
    module = _load_script("runpod_rerank_server")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "transformers"}:
            raise AssertionError(f"heavy import before revision validation: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(SystemExit, match="RERANK_REVISION"):
        module._load_runtime("reranker/repo", revision, True)


def test_runpod_loader_preserves_unpinned_and_forwards_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("runpod_rerank_server")
    calls = []
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(calls))

    module._load_runtime("reranker/repo", None, False)
    module._load_runtime("reranker/repo", REVISION, True)

    assert calls == [
        ("tokenizer", "reranker/repo", {}),
        ("model", "reranker/repo", {}),
        ("to", "cpu"),
        ("eval",),
        ("tokenizer", "reranker/repo", {"revision": REVISION}),
        ("model", "reranker/repo", {"revision": REVISION}),
        ("to", "cpu"),
        ("eval",),
    ]


def test_finetune_cross_encoder_forwards_revision_only_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("finetune_reranker")
    calls = []

    class CrossEncoder:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

    package = types.ModuleType("sentence_transformers")
    cross_encoder = types.ModuleType("sentence_transformers.cross_encoder")
    cross_encoder.CrossEncoder = CrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", package)
    monkeypatch.setitem(sys.modules, "sentence_transformers.cross_encoder", cross_encoder)

    module._load_cross_encoder("reranker/repo", None)
    module._load_cross_encoder("reranker/repo", REVISION)

    assert calls == [
        ("reranker/repo", {"num_labels": 1, "max_length": 512}),
        (
            "reranker/repo",
            {"num_labels": 1, "max_length": 512, "revision": REVISION},
        ),
    ]


def test_finetune_hard_negatives_cover_same_document_neighbors_and_versions() -> None:
    from scripts.finetune_reranker import (
        HARD_NEGATIVE_KINDS,
        _select_hard_negatives,
    )

    gold = {
        "source": "matsne",
        "document_id": "law-1",
        "version_id": "v2",
        "version_family": "law-1-lineage",
        "chunk_index": 7,
        "article_id": "10",
    }

    def point(text: str, **payload):
        return types.SimpleNamespace(payload={"text": text, **payload})

    common = {"source": "matsne", "document_id": "law-1"}
    points = [
        point("positive", **common, version_id="v2", chunk_index=7, article_id="10"),
        point("wrong passage", **common, version_id="v2", chunk_index=8, article_id="10"),
        point("neighbor", **common, version_id="v2", chunk_index=9, article_id="11"),
        point("old version", **common, version_id="v1", chunk_index=7, article_id="10"),
        point(
            "blind leak",
            source="matsne",
            document_id="blind-law",
            version_id="v1",
            chunk_index=1,
            version_family="blind-lineage",
        ),
        point(
            "similar authority",
            source="matsne",
            document_id="law-2",
            version_id="v1",
            chunk_index=1,
        ),
    ]
    selected = _select_hard_negatives(
        points,
        gold=gold,
        positive_family="law-1-lineage",
        excluded_families={"blind-lineage"},
        limit=4,
    )
    assert {kind for _text, kind in selected} == HARD_NEGATIVE_KINDS
    assert "positive" not in {text for text, _kind in selected}
    assert "blind leak" not in {text for text, _kind in selected}


def test_finetune_exclusion_uses_explicit_version_family(tmp_path) -> None:
    from scripts.finetune_reranker import _excluded_families, _record_family

    test = tmp_path / "test.jsonl"
    test.write_text(
        json.dumps({
            "gold": {
                "source": "matsne",
                "document_id": "law-1-v2",
                "version_family": "law-1-lineage",
            }
        }) + "\n",
        encoding="utf-8",
    )
    holdout = tmp_path / "holdout.json"
    holdout.write_text(
        json.dumps([{"source": "ecd", "document_id": "case-1"}]),
        encoding="utf-8",
    )
    assert _excluded_families(test, holdout) == {
        "law-1-lineage",
        "ecd:case-1",
    }
    assert _record_family({
        "gold": {"source": "matsne", "document_id": "law-1-v1"},
        "version_family": "law-1-lineage",
    }) == "law-1-lineage"
