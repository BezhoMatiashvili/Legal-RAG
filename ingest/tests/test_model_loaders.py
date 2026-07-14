"""Model loaders honor immutable revisions without importing heavyweight stacks."""

from __future__ import annotations

import sys
import types
from dataclasses import replace

from ingest.config import load_config
from ingest.embedding import BGEM3Embedder, make_token_counter
from ingest.rerank import BGEReranker, ONNXBGEReranker


def test_embedding_revision_resolves_one_immutable_snapshot(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def snapshot_download(*, repo_id, revision):
        calls.append(("snapshot", (repo_id, revision)))
        return "/cache/snapshots/exact-commit"

    class Model:
        def __init__(self, source, **kwargs):
            calls.append(("model", (source, kwargs)))

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setitem(
        sys.modules, "FlagEmbedding", types.SimpleNamespace(BGEM3FlagModel=Model)
    )
    cfg = replace(load_config(), embedding_revision="a" * 40)

    BGEM3Embedder(cfg)

    assert calls == [
        ("snapshot", (cfg.embed_model, "a" * 40)),
        ("model", ("/cache/snapshots/exact-commit", {"use_fp16": False})),
    ]


def test_unpinned_embedding_preserves_direct_model_name(monkeypatch) -> None:
    calls = []

    class Model:
        def __init__(self, source, **kwargs):
            calls.append((source, kwargs))

    monkeypatch.setitem(
        sys.modules, "FlagEmbedding", types.SimpleNamespace(BGEM3FlagModel=Model)
    )
    cfg = replace(load_config(), embedding_revision=None)

    BGEM3Embedder(cfg)

    assert calls == [(cfg.embed_model, {"use_fp16": False})]


def test_token_counter_forwards_revision_only_when_configured(monkeypatch) -> None:
    calls = []

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append((model, kwargs))
            return cls()

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return text.split()

    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Tokenizer)
    )

    assert make_token_counter("repo/model")("one two") == 2
    assert make_token_counter("repo/model", "b" * 40)("one") == 1
    assert calls == [
        ("repo/model", {}),
        ("repo/model", {"revision": "b" * 40}),
    ]


def _fake_torch():
    class Model:
        def to(self, device):
            return self

        def eval(self):
            return None

    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
        set_num_threads=lambda _n: None,
        set_num_interop_threads=lambda _n: None,
        Model=Model,
    )


def test_torch_reranker_forwards_same_revision_to_tokenizer_and_model(
    monkeypatch,
) -> None:
    calls = []
    torch = _fake_torch()

    class Loader:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append((cls.__name__, model, kwargs))
            return torch.Model() if cls.__name__ == "ModelLoader" else object()

    class TokenizerLoader(Loader):
        pass

    class ModelLoader(Loader):
        pass

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=TokenizerLoader,
            AutoModelForSequenceClassification=ModelLoader,
        ),
    )
    cfg = replace(load_config(), reranker_revision="c" * 40)

    BGEReranker(cfg)

    expected = {"revision": "c" * 40}
    assert calls == [
        ("TokenizerLoader", cfg.rerank_model, expected),
        ("ModelLoader", cfg.rerank_model, expected),
    ]


def test_onnx_reranker_forwards_tokenizer_revision(monkeypatch, tmp_path) -> None:
    calls = []
    model_path = tmp_path / "reranker.onnx"
    model_path.write_bytes(b"not loaded by fake runtime")

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append((model, kwargs))
            return cls()

    class SessionOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(
            SessionOptions=SessionOptions,
            InferenceSession=lambda *_args, **_kwargs: object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Tokenizer)
    )
    cfg = replace(
        load_config(), onnx_rerank_path=model_path, reranker_revision="d" * 40
    )

    ONNXBGEReranker(cfg)

    assert calls == [(cfg.rerank_model, {"revision": "d" * 40})]
