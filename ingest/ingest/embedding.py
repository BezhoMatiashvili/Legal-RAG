"""BGE-M3 embeddings: dense (1024-d) + multilingual learned-sparse, from one model.

FlagEmbedding/transformers are imported lazily so the offline unit tests (chunking,
sources, ids) don't require the heavy ML stack to be installed.
"""

from collections.abc import Callable
from dataclasses import dataclass

from .config import Config


@dataclass(frozen=True)
class Sparse:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class Embedded:
    dense: list[float]
    sparse: Sparse


def _embedding_model_source(model_name: str, revision: str | None) -> str:
    """Resolve a pinned Hub model to its immutable local snapshot directory.

    ``BGEM3FlagModel`` does not forward a Hugging Face ``revision`` to all of its
    internal loaders. Resolving first is therefore the only way to ensure its model
    and tokenizer come from the same configured commit. The legacy unpinned path is
    deliberately untouched.
    """
    if revision is None:
        return model_name
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_name, revision=revision)


class BGEM3Embedder:
    def __init__(self, cfg: Config):
        from FlagEmbedding import BGEM3FlagModel

        model_source = _embedding_model_source(cfg.embed_model, cfg.embedding_revision)
        kwargs = {"use_fp16": cfg.embed_use_fp16}
        if cfg.embed_device:
            kwargs["devices"] = cfg.embed_device
        try:
            self.model = BGEM3FlagModel(model_source, **kwargs)
        except TypeError:
            # Older FlagEmbedding used `device` instead of `devices`.
            kwargs.pop("devices", None)
            if cfg.embed_device:
                kwargs["device"] = cfg.embed_device
            self.model = BGEM3FlagModel(model_source, **kwargs)
        self.batch_size = cfg.embed_batch_size

    def _encode(self, texts: list[str]) -> list[Embedded]:
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = out["dense_vecs"]
        lexical = out["lexical_weights"]
        results: list[Embedded] = []
        for i in range(len(texts)):
            vec = dense[i]
            dense_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
            weights = lexical[i] or {}
            indices = [int(k) for k in weights]
            values = [float(v) for v in weights.values()]
            results.append(Embedded(dense=dense_list, sparse=Sparse(indices, values)))
        return results

    def encode_passages(self, texts) -> list[Embedded]:
        return self._encode(list(texts))

    def encode_query(self, text: str) -> Embedded:
        return self._encode([text])[0]


def make_token_counter(
    model_name: str, revision: str | None = None
) -> Callable[[str], int]:
    """A token counter using the model's own tokenizer (accurate chunk sizing)."""
    from transformers import AutoTokenizer

    kwargs = {"revision": revision} if revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)

    def count(text: str) -> int:
        if not text:
            return 0
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count
