## Phase C — retrieval comparison (full 2.45M-chunk index, 103-pair golden set)

_Chunk-level relevance. Metric cells are mean [95% bootstrap CI]. Latency is total per-query p50/p95 ms on CPU. 11 distinct qdrant runs; 5 fake-backend (Part-2 synthetic) rows retained for reference._

### 1. Modes (BM25 floor → dense/sparse/hybrid → +rerank → routed)

_rerank = rerank@80; its latency is CPU-derived (see §3 note), others are measured CPU._

| config | R@5 [95% CI] | R@10 [95% CI] | nDCG@10 [95% CI] | MRR@10 [95% CI] | lat p50/p95 (ms) |
|---|---|---|---|---|---|
| `bm25` | 0.243 [0.165, 0.330] | 0.243 [0.165, 0.330] | 0.192 [0.124, 0.267] | 0.176 [0.111, 0.248] | 70 / 243 |
| `dense` | 0.155 [0.087, 0.233] | 0.214 [0.136, 0.291] | 0.106 [0.064, 0.153] | 0.080 [0.045, 0.121] | 323 / 432 |
| `sparse` | 0.233 [0.155, 0.320] | 0.282 [0.194, 0.369] | 0.182 [0.121, 0.248] | 0.160 [0.102, 0.223] | 307 / 405 |
| `hybrid` | 0.233 [0.155, 0.320] | 0.330 [0.243, 0.427] | 0.182 [0.126, 0.244] | 0.146 [0.094, 0.203] | 474 / 642 |
| `rerank` | 0.359 [0.272, 0.456] | 0.388 [0.301, 0.485] | 0.289 [0.212, 0.367] | 0.270 [0.195, 0.348] | 40800 / 46920 (CPU) |
| `routed` | 0.252 [0.175, 0.340] | 0.330 [0.243, 0.427] | 0.187 [0.129, 0.249] | 0.151 [0.098, 0.208] | 440 / 564 |

### 2. Cross-lingual slice — the routing question (22 EN pairs)

_per-query-type = cross_lingual:_

| config | n | R@5 | R@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|
| `bm25` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| `dense` | 22 | 0.136 | 0.136 | 0.062 | 0.047 |
| `sparse` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| `hybrid` | 22 | 0.045 | 0.136 | 0.047 | 0.027 |
| `rerank` | 22 | 0.227 | 0.273 | 0.184 | 0.156 |
| `routed` | 22 | 0.136 | 0.136 | 0.062 | 0.047 |

_per-language = en:_

| config | n | R@5 | R@10 | nDCG@10 | MRR@10 |
|---|---|---|---|---|---|
| `bm25` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| `dense` | 22 | 0.136 | 0.136 | 0.062 | 0.047 |
| `sparse` | 22 | 0.000 | 0.000 | 0.000 | 0.000 |
| `hybrid` | 22 | 0.045 | 0.136 | 0.047 | 0.027 |
| `rerank` | 22 | 0.227 | 0.273 | 0.184 | 0.156 |
| `routed` | 22 | 0.136 | 0.136 | 0.062 | 0.047 |

### 3. Rerank-depth ablation (quality: GPU pod, fp32; latency: CPU, derived)

_Latency is derived: rerank@80 measured 40.8 s/query end-to-end on the pre-length-bucketing CPU reranker (4203 s / 103 q); rerank scales ~linearly with depth, so shallower depths are ~proportional. Treat as an upper bound._

| config | R@5 [95% CI] | R@10 [95% CI] | nDCG@10 [95% CI] | MRR@10 [95% CI] | lat p50/p95 (ms) |
|---|---|---|---|---|---|
| rerank@10 | 0.311 [0.223, 0.398] | 0.330 [0.243, 0.427] | 0.249 [0.174, 0.326] | 0.232 [0.160, 0.308] | 5538 / 6368 (CPU) |
| rerank@30 | 0.320 [0.233, 0.417] | 0.340 [0.252, 0.437] | 0.262 [0.186, 0.341] | 0.245 [0.170, 0.323] | 15612 / 17954 (CPU) |
| rerank@50 | 0.340 [0.252, 0.437] | 0.359 [0.272, 0.456] | 0.270 [0.193, 0.349] | 0.255 [0.179, 0.332] | 25688 / 29541 (CPU) |
| rerank@80 | 0.359 [0.272, 0.456] | 0.388 [0.301, 0.485] | 0.289 [0.212, 0.367] | 0.270 [0.195, 0.348] | 40800 / 46920 (CPU) |

### 4. Diversity (max-per-doc / MMR) vs no-diversity rerank@80

| config | R@5 [95% CI] | R@10 [95% CI] | nDCG@10 [95% CI] | MRR@10 [95% CI] | lat p50/p95 (ms) |
|---|---|---|---|---|---|
| `rerank` (base) | 0.359 [0.272, 0.456] | 0.388 [0.301, 0.485] | 0.289 [0.212, 0.367] | 0.270 [0.195, 0.348] | 40800 / 46920 (CPU) |
| max_per_doc=2 | 0.359 [0.272, 0.456] | 0.388 [0.301, 0.485] | 0.289 [0.212, 0.368] | 0.271 [0.195, 0.348] | 40800 / 46920 (CPU) |
| mmr_lambda=0.5 | 0.087 [0.039, 0.146] | 0.136 [0.078, 0.204] | 0.074 [0.037, 0.117] | 0.062 [0.028, 0.103] | 40800 / 46920 (CPU) |

### 5. Fusion — DBSF vs RRF (hybrid)

| config | R@5 [95% CI] | R@10 [95% CI] | nDCG@10 [95% CI] | MRR@10 [95% CI] | lat p50/p95 (ms) |
|---|---|---|---|---|---|
| `hybrid` (base) | 0.233 [0.155, 0.320] | 0.330 [0.243, 0.427] | 0.182 [0.126, 0.244] | 0.146 [0.094, 0.203] | 474 / 642 |

### 6. Prefetch recall-pool depth (hybrid)

_(pending — no runs logged yet)_

### 7. HNSW ef × int8 rescore (dense) — quantization recall/latency knee

_(pending — no runs logged yet)_

### 8. Cross-lingual sparse-weight sweep (hybrid)

_(pending — no runs logged yet)_

_cross_lingual slice:_

_(pending)_

### 9. Paired significance (from --compare / --ab)

**routed (B) vs hybrid (A)** — relevance=chunk

| metric | Δ (B−A) | 95% CI | p | verdict |
|---|---|---|---|---|
| R@5 | +0.019 | [+0.000, +0.049] | 0.5009 | TIE |
| R@10 | +0.000 | [+0.000, +0.000] | 1.0000 | TIE |
| nDCG@10 | +0.004 | [+0.000, +0.009] | 0.1073 | TIE |
| MRR@10 | +0.005 | [-0.000, +0.012] | 0.1073 | TIE |

