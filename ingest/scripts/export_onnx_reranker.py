#!/usr/bin/env python
"""Export BAAI/bge-reranker-v2-m3 to ONNX and dynamically quantize to int8 (improvement I7).

CPU rerank is the serving bottleneck (~25-67 s/query at rc=50-80); int8 dynamic
quantization documented at ~2x speedup for <1 nDCG point. Exported with the torch
exporter (optimum needs transformers<5, which would downgrade the embed stack — not
acceptable), quantized with onnxruntime. The fp32 ONNX intermediate (~2.3 GB, external
data) is deleted after quantization; the int8 model (~600 MB) is what serves.

    uv run --group onnx python scripts/export_onnx_reranker.py
    # → ingest/.state/onnx/bge-reranker-v2-m3-int8.onnx  (+ tokenizer parity check)
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.config import load_config  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / ".state" / "onnx"
FP32 = OUT_DIR / "fp32" / "bge-reranker-v2-m3-fp32.onnx"
INT8 = OUT_DIR / "bge-reranker-v2-m3-int8.onnx"
_MAX_LENGTH = 512


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cfg = load_config()
    print(f"loading {cfg.rerank_model} (fp32, cpu)...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.rerank_model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.rerank_model)
    model.eval()

    sample = tokenizer([["query", "text"]], padding=True, truncation=True,
                       max_length=_MAX_LENGTH, return_tensors="pt")
    FP32.parent.mkdir(parents=True, exist_ok=True)

    class Logits(torch.nn.Module):  # export just the logits tensor, not the ModelOutput
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          return_dict=True).logits

    print("exporting fp32 ONNX (external data, ~2.3 GB)...")
    torch.onnx.export(
        Logits(model),
        (sample["input_ids"], sample["attention_mask"]),
        str(FP32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # classic exporter: stable for XLM-R, no torch.compile surprises
    )

    print("quantizing to int8 (dynamic)...")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(FP32), str(INT8), weight_type=QuantType.QInt8,
                     use_external_data_format=False)

    print("parity check (torch fp32 vs onnx int8 on 3 pairs)...")
    import numpy as np
    import onnxruntime as ort

    pairs = [
        ["შვებულების ხანგრძლივობა", "დასაქმებულს უფლება აქვს ისარგებლოს ანაზღაურებადი შვებულებით წელიწადში 24 სამუშაო დღით."],
        ["გირაოს ოდენობა", "ეს ტექსტი სრულიად სხვა თემაზეა და შეკითხვას არ ეხება."],
        ["annual leave", "ყოველწლიური ანაზღაურებადი შვებულება განისაზღვრება შრომის კოდექსით."],
    ]
    enc = tokenizer(pairs, padding=True, truncation=True, max_length=_MAX_LENGTH,
                    return_tensors="pt")
    with torch.no_grad():
        ref = torch.sigmoid(model(**enc, return_dict=True).logits.view(-1)).tolist()
    sess = ort.InferenceSession(str(INT8), providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"input_ids": enc["input_ids"].numpy(),
                                "attention_mask": enc["attention_mask"].numpy()})[0]
    got = (1 / (1 + np.exp(-out.reshape(-1)))).tolist()
    for r, g in zip(ref, got):
        print(f"  torch={r:.4f}  int8={g:.4f}  Δ={abs(r - g):.4f}")

    shutil.rmtree(FP32.parent, ignore_errors=True)  # drop the 2.3 GB intermediate
    print(f"done: {INT8} ({INT8.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
