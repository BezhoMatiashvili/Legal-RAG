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

import logging
import os

from .config import Config

logger = logging.getLogger(__name__)

_BATCH_SIZE = 16


class RerankerParityError(RuntimeError):
    """The remote reranker's reported model/max_length differs from local config.

    Raised before any scoring request so a drifted pod can never silently return
    valid-looking floats on a different calibration scale (memory-bank
    reranker-score-parity).
    """


def _revision_kwargs(revision: str | None) -> dict[str, str]:
    """Keep the historical loader call byte-for-byte equivalent while unpinned."""
    return {"revision": revision} if revision is not None else {}


def _auto_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _configure_cpu_threads(torch) -> None:
    """Pin the cross-encoder's CPU parallelism to physical cores with a single interop
    thread. Default torch spawns one intraop thread per *logical* core (18 on this 14-core
    box) and, stacked across processes, oversubscribes and stalls the rerank. Threads are
    read from ``OMP_NUM_THREADS`` (set in the MCP launch env) so torch and OpenMP agree;
    falls back to the logical count for CLI use. ``set_num_interop_threads`` may only be
    called before the first parallel op, so it is best-effort — the intraop cap still
    applies. Thread count does not affect scores (fp reduction-order drift ~1e-6)."""
    n = int(os.getenv("OMP_NUM_THREADS") or os.cpu_count() or 1)
    try:
        torch.set_num_threads(n)
    except Exception:  # noqa: BLE001 - never fail rerank init over a tuning hint
        pass
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already past the first parallel op; the intraop cap above still holds


class BGEReranker:
    """Cross-encoder exposing ``score(query, texts) -> list[float]`` (0..1, higher = better)."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        if (cfg.rerank_device or _auto_device(torch)) == "cpu":
            _configure_cpu_threads(torch)  # before the first forward
        self.device = cfg.rerank_device or _auto_device(torch)
        self.max_length = cfg.rerank_max_length
        revision = _revision_kwargs(cfg.reranker_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.rerank_model, **revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.rerank_model, **revision
        )
        model = model.to(self.device)
        if cfg.rerank_use_fp16 and self.device != "cpu":
            model = model.half()
        model.eval()
        self.model = model

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score for each ``text`` against ``query`` (sigmoid of the logit).

        Candidates are length-bucketed — sorted by length, batched, then scattered back to
        input order — so each padded batch holds similarly-sized inputs instead of a 64-token
        chunk riding in a 512-wide batch beside a 512-token one. Padding tokens are masked,
        so the scores are identical to the unsorted order; this only removes wasted FLOPs."""
        if not texts:
            return []
        torch = self._torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        scores: list[float] = [0.0] * len(texts)
        for start in range(0, len(order), _BATCH_SIZE):
            idx = order[start : start + _BATCH_SIZE]
            pairs = [[query, texts[i]] for i in idx]
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                probs = torch.sigmoid(logits)
            for i, p in zip(idx, probs.cpu().tolist()):
                scores[i] = p
        return scores


class ONNXBGEReranker:
    """int8 ONNX cross-encoder (improvement I7) — same interface/scores as BGEReranker.

    Runs the dynamically-quantized export of the same model (see
    ``scripts/export_onnx_reranker.py``) through onnxruntime on CPU: ~2x the fp32 torch
    throughput at <0.01 nDCG cost (gated by the I7 parity eval). Tokenization is the
    identical ``AutoTokenizer``; scoring keeps the same length-bucketed batching and
    sigmoid normalisation, so it is a drop-in behind ``RERANK_BACKEND=onnx``.
    """

    def __init__(self, cfg: Config):
        path = cfg.onnx_rerank_path
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX reranker not found at {path} — run "
                "`uv run --group onnx python scripts/export_onnx_reranker.py` first")

        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self._np = np
        self.max_length = cfg.rerank_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.rerank_model, **_revision_kwargs(cfg.reranker_revision)
        )
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.getenv("OMP_NUM_THREADS") or os.cpu_count() or 1)
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.device = "onnx-cpu"  # parity with BGEReranker (callers may log .device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance for each ``text`` vs ``query`` — length-bucketed, sigmoid(logit)."""
        if not texts:
            return []
        np = self._np
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        scores: list[float] = [0.0] * len(texts)
        for start in range(0, len(order), _BATCH_SIZE):
            idx = order[start : start + _BATCH_SIZE]
            pairs = [[query, texts[i]] for i in idx]
            enc = self.tokenizer(pairs, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="np")
            logits = self.session.run(
                ["logits"],
                {"input_ids": enc["input_ids"].astype(np.int64),
                 "attention_mask": enc["attention_mask"].astype(np.int64)},
            )[0].reshape(-1)
            probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
            for i, p in zip(idx, probs.tolist()):
                scores[i] = p
        return scores


def make_reranker(cfg: Config):
    """The configured local reranker: ``RERANK_BACKEND=onnx`` → int8 ONNX, else torch."""
    if cfg.rerank_backend == "onnx":
        return ONNXBGEReranker(cfg)
    return BGEReranker(cfg)


class RemoteBGEReranker:
    """Drop-in for :class:`BGEReranker` that delegates scoring to a remote HTTP reranker.

    Used to offload the CPU-bound cross-encoder to a GPU pod during the Phase-C sweep: retrieval
    and metrics stay local, only the (query, candidate-text) pairs cross an SSH tunnel to the
    pod's ``runpod_rerank_server.py`` (identical model / tokenizer / max_length / sigmoid → the
    same scores as CPU). Enabled by ``RERANK_REMOTE_URL`` (e.g. ``http://localhost:8900``); loads
    no ML deps locally. Interface is identical: ``score(query, texts) -> list[float]``.
    """

    def __init__(
        self,
        url: str,
        timeout: int = 300,
        expected_model: str | None = None,
        expected_max_length: int | None = None,
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.device = "remote"  # parity with BGEReranker (callers may log .device)
        # Local half of the score-parity contract. Both sides read RERANK_MODEL /
        # RERANK_MAX_LENGTH from their *own* process env (memory-bank
        # reranker-score-parity: a real silent-drift vector), so defaults here must
        # mirror Config's defaults for callers that don't pass explicit expectations.
        self.expected_model = expected_model or os.environ.get(
            "RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self.expected_max_length = expected_max_length or int(
            os.environ.get("RERANK_MAX_LENGTH", "512")
        )
        self._parity_checked = False

    def _check_parity(self) -> None:
        """One-shot server-config check before the first scoring request.

        A *reported mismatch* raises (scores would be silently on a different scale —
        the exception surfaces through the caller's existing remote-failure handling).
        An unreachable server or an older server that doesn't report ``max_length``
        only warns: reachability problems already fail loudly in :meth:`score`, and
        parity can't be disproven without the report.
        """
        import json
        import urllib.request

        self._parity_checked = True
        try:
            with urllib.request.urlopen(f"{self.url}/", timeout=30) as resp:
                info = json.loads(resp.read().decode())
        except Exception as exc:  # noqa: BLE001 - reachability is score()'s job
            logger.warning("remote reranker parity check skipped (unreachable): %s", exc)
            return
        remote_model = info.get("model")
        remote_max_length = info.get("max_length")
        if remote_model is not None and remote_model != self.expected_model:
            raise RerankerParityError(
                f"remote reranker model mismatch: local expects {self.expected_model!r},"
                f" server runs {remote_model!r} — scores are not comparable"
            )
        if remote_max_length is None:
            logger.warning(
                "remote reranker does not report max_length (older server); "
                "RERANK_MAX_LENGTH parity is UNVERIFIED — local expects %d",
                self.expected_max_length,
            )
        elif int(remote_max_length) != self.expected_max_length:
            raise RerankerParityError(
                f"remote reranker max_length mismatch: local expects "
                f"{self.expected_max_length}, server runs {remote_max_length} — "
                "truncation differs, scores are not comparable"
            )

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if not self._parity_checked:
            self._check_parity()
        import json
        import urllib.request

        data = json.dumps({"query": query, "texts": texts}).encode()
        req = urllib.request.Request(
            f"{self.url}/score", data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())["scores"]
