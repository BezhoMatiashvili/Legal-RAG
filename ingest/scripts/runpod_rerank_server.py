#!/usr/bin/env python
"""GPU rerank service (runs ON the RunPod pod). Mirrors ingest.rerank.BGEReranker.score exactly
— same model (BAAI/bge-reranker-v2-m3), AutoTokenizer(fast), max_length=512 truncation, sigmoid
of the logit — but on CUDA, so scores match the CPU reranker while running orders of magnitude
faster. Stdlib HTTP only (no framework):
    POST /score  {"query": str, "texts": [str, ...]}  -> {"scores": [float, ...]}
    GET  /health -> {"ok": true, "device": "cuda", "model_loaded": bool}
Env: RERANK_MODEL (default BAAI/bge-reranker-v2-m3), PORT (default 8900),
     RERANK_MAX_LENGTH (default 512), RERANK_BATCH (default 64 — GPU can go wide).
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))
BATCH = int(os.environ.get("RERANK_BATCH", "64"))
PORT = int(os.environ.get("PORT", "8900"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[rerank_server] loading {MODEL} on {DEVICE} ...", flush=True)
_tok = AutoTokenizer.from_pretrained(MODEL)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(DEVICE).eval()
print(f"[rerank_server] model ready on {DEVICE}", flush=True)


def score(query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    out: list[float] = []
    for start in range(0, len(texts), BATCH):
        batch = texts[start : start + BATCH]
        pairs = [[query, t] for t in batch]
        inputs = _tok(pairs, padding=True, truncation=True, max_length=MAX_LENGTH,
                      return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = _model(**inputs, return_dict=True).logits.view(-1).float()
            probs = torch.sigmoid(logits)
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


if __name__ == "__main__":
    print(f"[rerank_server] listening on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
