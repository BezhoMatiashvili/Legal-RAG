#!/usr/bin/env python
"""GPU rerank service (runs ON the RunPod pod). Mirrors ingest.rerank.BGEReranker.score exactly
— same model (BAAI/bge-reranker-v2-m3), AutoTokenizer(fast), max_length=512 truncation, sigmoid
of the logit — but on CUDA, so scores match the CPU reranker while running orders of magnitude
faster. Stdlib HTTP only (no framework):
    POST /score  {"query": str, "texts": [str, ...]}  -> {"scores": [float, ...]}
    GET  /health -> {"ok": true, "device": "cuda", "model_loaded": bool}
Env: RERANK_MODEL (default BAAI/bge-reranker-v2-m3), PORT (default 8900),
     RERANK_REVISION (immutable model commit), PRODUCTION_MODE (default false),
     RERANK_MAX_LENGTH (default 512), RERANK_BATCH (default 64 — GPU can go wide).
"""

import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_REVISION = (os.environ.get("RERANK_REVISION") or "").strip() or None
PRODUCTION_MODE = (os.environ.get("PRODUCTION_MODE") or "").strip().lower() in {
    "1",
    "true",
    "yes",
}
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))
BATCH = int(os.environ.get("RERANK_BATCH", "64"))
PORT = int(os.environ.get("PORT", "8900"))
DEVICE = "uninitialized"
_IMMUTABLE_REVISION_RE = re.compile(r"[0-9a-f]{40,64}")
_torch = None
_tok = None
_model = None


def _load_runtime(model_name: str, revision: str | None, production_mode: bool):
    if production_mode and revision is None:
        raise SystemExit("RERANK_REVISION is required when PRODUCTION_MODE=true")
    if production_mode and not _IMMUTABLE_REVISION_RE.fullmatch(revision or ""):
        raise SystemExit(
            "RERANK_REVISION must be a 40-64 character lowercase hexadecimal commit "
            "when PRODUCTION_MODE=true"
        )

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    revision_kwargs = {"revision": revision} if revision is not None else {}
    print(f"[rerank_server] loading {model_name} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **revision_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **revision_kwargs
    ).to(device).eval()
    return torch, tokenizer, model, device


def score(query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    if _torch is None or _tok is None or _model is None:
        raise RuntimeError("reranker runtime is not initialized")
    out: list[float] = []
    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        pairs = [[query, t] for t in batch]
        inputs = _tok(pairs, padding=True, truncation=True, max_length=MAX_LENGTH,
                      return_tensors="pt").to(DEVICE)
        with _torch.no_grad():
            logits = _model(**inputs, return_dict=True).logits.view(-1).float()
            probs = _torch.sigmoid(logits)
        out.extend(probs.cpu().tolist())
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._send(200, {"ok": True, "device": DEVICE, "model": MODEL})

    def do_POST(self):  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n))
            scores = score(payload["query"], payload["texts"])
            self._send(200, {"scores": scores})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": repr(e)})

    def log_message(self, *args):  # silence per-request logging
        pass


def main() -> None:
    global _torch, _tok, _model, DEVICE
    _torch, _tok, _model, DEVICE = _load_runtime(
        MODEL, RERANK_REVISION, PRODUCTION_MODE
    )
    print(f"[rerank_server] model ready on {DEVICE}", flush=True)
    print(f"[rerank_server] listening on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
