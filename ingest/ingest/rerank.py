"""Cross-encoder reranking with BGE-reranker-v2-m3 (multilingual, pairs with BGE-M3).

A second-stage reranker re-scores the hybrid (dense+sparse) candidate pool by reading
the query and each chunk *together*, which a bi-encoder cannot do — it is the single
biggest precision lever for the legal corpus and yields a calibrated score we can gate
on.

We drive the model directly through ``transformers`` (already a dependency — the same
stack BGE-M3 runs on) rather than FlagEmbedding's ``FlagReranker``: the latter loads a
slow tokenizer whose ``prepare_for_model`` API has been removed in current transformers,
which crashes at score time. Going through ``AutoTokenizer`` (fast) +
``AutoModelForSequenceClassification`` is both robust and gives us explicit control over
device, fp16, batching and the sigmoid normalisation. The heavy imports are lazy so the
offline unit tests don't need the ML stack.
"""

from .config import Config

_MAX_LENGTH = 512   # query+chunk truncation (chunks are already ~512 tokens)
_BATCH_SIZE = 16


def _auto_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BGEReranker:
    """Cross-encoder exposing ``score(query, texts) -> list[float]`` (0..1, higher = better)."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = cfg.rerank_device or _auto_device(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.rerank_model)
        model = AutoModelForSequenceClassification.from_pretrained(cfg.rerank_model)
        model = model.to(self.device)
        if cfg.rerank_use_fp16 and self.device != "cpu":
            model = model.half()
        model.eval()
        self.model = model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score for each ``text`` against ``query`` (sigmoid of the logit)."""
        if not texts:
            return []
        torch = self._torch
        out: list[float] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            pairs = [[query, t] for t in batch]
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, max_length=_MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                probs = torch.sigmoid(logits)
            out.extend(probs.cpu().tolist())
        return out
